#!/usr/bin/env python3
"""Diagnose and size the OSL full-body robot's position-servo parameters.

This script only *reads* the models.  It reports, per actuated joint, the
normalized position mapping actually encoded in ``gainprm``/``biasprm`` and how
it differs from the midpoint convention the generator emits, the
equality-constrained effective inertia, the closed-loop natural frequency and
damping ratio the current gains imply, and ``Kd * dt / I`` -- the explicit-Euler
damping stability number, which diverges at 2.

It then prints a candidate parameter table built from three rules.  Every one is
motion-independent: no trajectory clip enters any parameter.

1. ``forcerange`` -- the peak moment the sibling muscle model can produce about
   that coordinate.  See :func:`muscle_moment_capacity`.

2. ``Kp = min(forcerange / saturation_error, I_eff * max_bandwidth**2)`` -- the
   servo reaches its torque limit at ``saturation_error`` radians of tracking
   error, capped so no joint exceeds ``max_bandwidth`` rad/s.

3. ``Kd = 2 * zeta * sqrt(Kp * I_eff)`` -- a target damping ratio rather than a
   flat per-region constant.

``I_eff`` in rules 2 and 3 is the equality-constrained inertia of the *mechanism*,
measured on a servo-neutralised, contact-free copy (:func:`servo_free_copy`).  Both
neutralisations matter.  Probing the actuated robot makes the finite difference
nonlinear and reports arm inertia 48-62% low; probing with self-contacts live sizes
the arms against qpos0's 21.47 mm humerus/thorax interpenetration, which triples
measured shoulder inertia and leaves ``shoulder_rot`` at ``Kd * dt / I`` = 1.735 --
87% of the Euler bound -- as soon as the arms separate.

Rule 3 is what keeps explicit Euler viable.  Since ``Kd / I = 2 * zeta * omega``,
the stability condition ``Kd * dt / I < 2`` becomes ``zeta * omega * dt < 1``,
satisfied automatically at any sane bandwidth.  A flat per-region ``Kd`` carries
no such guarantee: the region-uniform values this table replaced reached
``Kd * dt / I`` of 3.0-3.6 at 1 ms and 3.3-6.8 at the 2 ms the training env
actually uses, which was the sole cause of the distal-joint blowups previously
mistaken for torque saturation.

Note that ``mjDSBL_EULERDAMP`` -- left enabled by the OSL environments -- makes
``dof_damping`` implicit but does *not* cover an actuator's ``biasprm[2]``.  That
asymmetry is why the socket joints tolerate ``damping=10000`` while a servo
``Kd`` of 30 diverges, and why the SMPL humanoid keeps actuator ``Kd`` at 0.8-4.0
and puts its real damping in ``dof_damping``.

Usage::

    .venv/bin/python scripts/analyze_osl_fullbody_robot_actuators.py

``--trajectory`` is optional and purely diagnostic: it adds the gravity/dynamic
load seen over a clip and cross-checks the stability number against the smallest
effective inertia that clip reaches.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import warnings
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = REPO_ROOT / "models" / "osl_fullbody" / "body" / "osl_fullbody_robot.xml"
MUSCLE_XML = REPO_ROOT / "models" / "osl_fullbody" / "body" / "osl_fullbody.xml"


def servo_actuators(model: mujoco.MjModel) -> list[tuple[int, str, int, int]]:
    """Return ``(actuator_id, joint_name, joint_id, dof_address)`` per servo."""
    out = []
    for actuator in range(model.nu):
        if model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE:
            continue
        joint = model.actuator_trnid[actuator, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        out.append((actuator, name, joint, model.jnt_dofadr[joint]))
    return out


def _probe_inertia(model: mujoco.MjModel, qpos: np.ndarray, dofs: list[int],
                   *, strict: bool = True, rtol: float = 1e-3) -> np.ndarray:
    """Effective inertia by unit-force probe, guarded for linearity.

    Takes ``model`` exactly as given -- it does **not** neutralise anything.  Use
    :func:`constrained_inertia` unless you specifically need the unguarded model,
    which in practice means testing the guard itself.

    ``mj_fullM`` reports the inertia with every other coordinate locked, which
    overstates a joint that drives equality followers.  Applying a unit
    generalized force and differencing the acceleration against the unforced pose
    measures what the servo actually accelerates.

    That difference is only inertia if the response is **linear in the applied
    force**, so each dof is probed at 1 N*m and 2 N*m and checked for
    superposition.  A nonlinear response is not a small error: measured on the
    servo'd robot at qpos0, arm inertia comes out 48-62% low and manufactures a
    27% left/right asymmetry at a bilaterally symmetric pose.
    """
    base = mujoco.MjData(model)
    base.qpos[:] = qpos
    mujoco.mj_forward(model, base)
    reference = base.qacc.copy()

    inertia = np.empty(len(dofs))
    offenders: list[tuple[int, float]] = []
    for index, dof in enumerate(dofs):
        deltas = []
        for force in (1.0, 2.0):
            data = mujoco.MjData(model)
            data.qpos[:] = qpos
            data.qfrc_applied[dof] = force
            mujoco.mj_forward(model, data)
            deltas.append(data.qacc[dof] - reference[dof])

        # A non-finite or non-positive response is not a large inertia, it is an
        # invalid probe.  Clamping it to a tiny epsilon would silently report an
        # enormous inertia instead.
        if not np.isfinite(deltas).all() or deltas[0] <= 0.0:
            offenders.append((index, float("inf")))
            inertia[index] = np.nan
            continue

        error = abs(deltas[1] - 2.0 * deltas[0]) / max(abs(2.0 * deltas[0]), 1e-30)
        if error > rtol:
            offenders.append((index, error))
        inertia[index] = 1.0 / deltas[0]

    if offenders:
        detail = ", ".join(f"dof {dofs[i]} (err={e:.2e})" for i, e in offenders)
        if strict:
            raise RuntimeError(
                f"inertia probe is nonlinear at {len(offenders)}/{len(dofs)} dofs, so "
                f"1/(qacc_forced - qacc_ref) is not inertia: {detail}. "
                "Probe a servo-neutralised, contact-free copy (see constrained_inertia)."
            )
        warnings.warn(
            f"inertia probe nonlinear at {len(offenders)}/{len(dofs)} dofs; "
            "values are approximate",
            RuntimeWarning,
            stacklevel=2,
        )
    return inertia


def servo_free_copy(model: mujoco.MjModel) -> mujoco.MjModel:
    """A copy with the position servos disarmed and contacts off, for measurement.

    Two things have to go before an inertia probe means anything:

    * **The servos.**  ``biasprm[0] = Kp * midpoint``, so at ``ctrl = 0`` each servo
      commands its joint-range *midpoint* rather than holding position.  At qpos0
      that saturates 7 of 29 actuators and drives ``|qacc|`` to 7070 rad/s^2.  The
      force itself cancels between the two probes -- it depends only on ``(q, qdot)``,
      which both share -- but it is what pushes the contacts nonlinear.
    * **Contacts.**  At qpos0 both humeri sit 21.47 mm inside ``thorax_coll1`` at
      ~1346 N per side, via explicit ``<pair>`` elements.  Under the servo loads a
      pyramidal facet multiplier clamps to zero across the probe
      (2.1999 -> 0.5935 -> 0), which is a piecewise-linear transition even though
      ``nefc`` never changes -- a constant row count is not a constant active set.
      That self-contact also triples measured shoulder inertia (3.3745 vs 1.1227),
      so sizing against it is sizing against an artifact of the reference pose.

    Only ``mjDSBL_CONTACT`` actually disables them: zeroing ``geom_contype`` /
    ``geom_conaffinity`` leaves explicit ``<pair>`` contacts in force and ``nefc``
    unchanged at 53.

    Joint limits are deliberately left alone -- they move the result by 3e-13 at
    qpos0 -- and gravity cancels in the difference.
    """
    neutral = copy.copy(model)  # independent in MuJoCo 3.4.0
    for actuator, _, _, _ in servo_actuators(neutral):
        neutral.actuator_gainprm[actuator, 0] = 0.0
        neutral.actuator_biasprm[actuator, :3] = 0.0
    neutral.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return neutral


def constrained_inertia(model: mujoco.MjModel, qpos: np.ndarray, dofs: list[int],
                        *, strict: bool = True) -> np.ndarray:
    """Equality-constrained effective inertia of the *mechanism*.

    Defensive wrapper: measures on :func:`servo_free_copy` so a caller cannot
    accidentally probe the actuated robot, which is the bug this replaced.  Pass
    ``strict=False`` for diagnostic poses where a nonlinear probe should warn
    rather than abort -- trajectory frames with joints riding their limits still
    trip the guard even contact-free.
    """
    return _probe_inertia(servo_free_copy(model), qpos, dofs, strict=strict)


def equality_children(model: mujoco.MjModel) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Map each driver joint to the ``(follower, polycoef)`` pairs it drives."""
    children: dict[int, list[tuple[int, np.ndarray]]] = {}
    for equality in np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT):
        driver = int(model.eq_obj2id[equality])
        if driver < 0:
            continue
        follower = int(model.eq_obj1id[equality])
        children.setdefault(driver, []).append((follower, model.eq_data[equality, :5].copy()))
    return children


def project_joint_equalities(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Set scalar joint-equality followers to their exact polynomial values."""
    projected = np.array(qpos, dtype=float, copy=True)
    equalities = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    for _ in range(3):
        for equality in equalities:
            follower = int(model.eq_obj1id[equality])
            driver = int(model.eq_obj2id[equality])
            value = projected[model.jnt_qposadr[driver]] if driver >= 0 else 0.0
            projected[int(model.jnt_qposadr[follower])] = np.polynomial.polynomial.polyval(
                value, model.eq_data[equality, :5]
            )
    return projected


def transmission_ratios(model: mujoco.MjModel, joint: int, qpos: np.ndarray,
                        children: dict[int, list[tuple[int, np.ndarray]]]) -> dict[int, float]:
    """``dq_k / dq_joint`` for ``joint`` and every joint it drives.

    A generalized force about a driver coordinate is the sum of the moment about
    the driver itself plus each follower's moment scaled by ``dq_follower /
    dq_driver``.  Ignoring that term understates a driver's capacity; deriving it
    from accelerations instead would fold in inertial coupling to unrelated
    coordinates, which is not joint torque at all.  The polynomial derivative is
    the only correct route.
    """
    ratios = {joint: 1.0}
    frontier = [(joint, 1.0)]
    for _ in range(4):
        following = []
        for driver, scale in frontier:
            for follower, coefficients in children.get(driver, []):
                derivative = np.polynomial.polynomial.polyder(coefficients)
                slope = scale * float(np.polynomial.polynomial.polyval(
                    qpos[model.jnt_qposadr[driver]], derivative))
                ratios[follower] = ratios.get(follower, 0.0) + slope
                following.append((follower, slope))
        frontier = following
        if not frontier:
            break
    return ratios


def to_muscle_qpos(muscle: mujoco.MjModel, robot: mujoco.MjModel,
                   robot_qpos: np.ndarray) -> np.ndarray:
    """Map a robot qpos onto the muscle model by joint name."""
    qpos = mujoco.MjData(muscle).qpos.copy()
    widths = {0: 7, 1: 4, 2: 1, 3: 1}
    for joint in range(robot.njnt):
        name = mujoco.mj_id2name(robot, mujoco.mjtObj.mjOBJ_JOINT, joint)
        source = mujoco.mj_name2id(muscle, mujoco.mjtObj.mjOBJ_JOINT, name)
        if source < 0:
            continue
        width = widths[robot.jnt_type[joint]]
        src = muscle.jnt_qposadr[source]
        dst = robot.jnt_qposadr[joint]
        qpos[src:src + width] = robot_qpos[dst:dst + width]
    return qpos


def background_poses(robot: mujoco.MjModel, servos: list[tuple[int, str, int, int]]
                     ) -> dict[str, np.ndarray]:
    """The two reference poses capacity is swept at, in robot coordinates.

    ``qpos0`` alone is not enough.  There the arm hangs at the side
    (``shoulder_elv = 0``), where the shoulder elevation *plane* is kinematically
    degenerate: ``elv_angle`` measures exactly 0.0 N*m, a real gimbal
    singularity rather than an artifact.  Sweeping ``elv_angle`` over its own
    full range does not help, because the singularity is a function of
    ``shoulder_elv``, not of ``elv_angle``.  The midrange pose puts
    ``shoulder_elv`` at 90 degrees and recovers ~153 N*m.

    Both poses are bilaterally symmetric -- the robot's joint ranges are
    identical for left/right pairs -- so all seven bilateral arm pairs come out
    exactly equal with no manual symmetrization.  The legs stay asymmetric,
    correctly: the amputation removes muscles crossing the right hip.
    """
    midrange = robot.qpos0.copy()
    for _, _, joint, _ in servos:
        midrange[robot.jnt_qposadr[joint]] = robot.jnt_range[joint].mean()
    return {"qpos0": robot.qpos0.copy(), "midrange": midrange}


def muscle_moment_capacity(robot: mujoco.MjModel, servos: list[tuple[int, str, int, int]],
                           steps: int = 41) -> dict[str, tuple[float, float, float, str]]:
    """Peak moment the muscle sibling model can produce about each coordinate.

    MuJoCo muscle force is ``gain(length, velocity) * act`` -- linear in
    activation with no coupling between muscles -- so the maximum moment about a
    coordinate is a linear program whose optimum is "every agonist at ``act=1``,
    every antagonist at 0".  Summing the positive and negative moment
    contributions *separately* evaluates exactly that optimum.  It is not the net
    moment of a co-contracted model, which would largely cancel.

    Contributions are accumulated in the *independent* coordinate system: a
    driver's moment includes each equality follower's moment times the
    transmission ratio (:func:`transmission_ratios`).  Deriving that instead from
    ``I_eff * qacc`` is wrong -- it charges inertial coupling to unrelated
    coordinates as joint torque, and overestimates the trunk by 13x.  19 of the
    27 servo'd joints drive no followers, so raw and corrected agree for them;
    the torso gains 1.65-1.9x and ``elv_angle`` inverts, its single ratio being
    -1.0 so that its capacity is a *difference* of two moment columns.

    Capacity varies with pose through moment arms and the force-length factor, so
    each joint is swept over its own range at each of the two
    :func:`background_poses`, and the peak is kept.

    Returns ``{joint: (peak, max_positive, max_negative, where)}``.  These are
    isometric ceilings: ``qvel = 0``, so the force-velocity factor sits at its
    optimum.
    """
    muscle = mujoco.MjSpec.from_file(str(MUSCLE_XML.resolve())).compile()
    data = mujoco.MjData(muscle)
    is_muscle = np.array(
        [muscle.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_MUSCLE for a in range(muscle.nu)]
    )
    children = equality_children(muscle)
    moment = np.zeros((muscle.nu, muscle.nv))
    backgrounds = {
        label: to_muscle_qpos(muscle, robot, qpos)
        for label, qpos in background_poses(robot, servos).items()
    }

    def moments_about(qpos: np.ndarray, name: str) -> tuple[float, float]:
        joint = mujoco.mj_name2id(muscle, mujoco.mjtObj.mjOBJ_JOINT, name)
        # An unreconciled follower leaves a large equality residual, which moves
        # every moment arm in the chain.
        data.qpos[:] = project_joint_equalities(muscle, qpos)
        data.act[:] = 1.0
        data.ctrl[:] = 1.0
        mujoco.mj_forward(muscle, data)
        mujoco.mju_sparse2dense(
            moment, np.asarray(data.actuator_moment).ravel(),
            data.moment_rownnz, data.moment_rowadr, data.moment_colind,
        )
        generalized = np.zeros(muscle.nu)
        for coordinate, ratio in transmission_ratios(muscle, joint, data.qpos, children).items():
            generalized += ratio * moment[:, muscle.jnt_dofadr[coordinate]]
        contribution = generalized * np.asarray(data.actuator_force) * is_muscle
        return contribution[contribution > 0].sum(), contribution[contribution < 0].sum()

    capacity = {}
    for _, name, joint, _ in servos:
        lower, upper = robot.jnt_range[joint]
        address = muscle.jnt_qposadr[
            mujoco.mj_name2id(muscle, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        best = (0.0, 0.0, 0.0, "")
        for label, background in backgrounds.items():
            for angle in np.linspace(lower, upper, steps):
                qpos = background.copy()
                qpos[address] = angle
                positive, negative = moments_about(qpos, name)
                peak = max(positive, -negative)
                if peak > best[0]:
                    best = (peak, positive, negative,
                            f"{label}@{math.degrees(angle):+.0f}d")
        capacity[name] = best
    return capacity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", type=Path, default=None,
                        help="optional environment-format .npz; purely diagnostic -- adds the "
                             "gravity/dynamic load over the clip and cross-checks the stability "
                             "number at the smallest effective inertia the clip reaches")
    parser.add_argument("--stride", type=int, default=25,
                        help="sample every Nth trajectory frame for the diagnostics (default: 25)")
    parser.add_argument("--saturation-error", type=float, default=0.35,
                        help="tracking error in rad at which a servo should reach forcerange")
    parser.add_argument("--max-bandwidth", type=float, default=150.0,
                        help="cap on the proposed closed-loop natural frequency, rad/s")
    parser.add_argument("--damping-ratio", type=float, default=1.0,
                        help="target closed-loop damping ratio for the proposed Kd")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="write the proposed table to this path")
    args = parser.parse_args()

    robot = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    servos = servo_actuators(robot)
    dofs = [dof for _, _, _, dof in servos]
    dt = robot.opt.timestep

    # Every sizing quantity comes from the reconciled reference pose and the joint
    # ranges, so nothing here depends on a motion clip.
    reference = project_joint_equalities(robot, robot.qpos0)
    inertia = constrained_inertia(robot, reference, dofs)
    capacity = muscle_moment_capacity(robot, servos)

    # Diagnostics only.
    load = np.zeros(len(servos))
    inertia_min = inertia
    approximate = 0
    if args.trajectory is not None:
        frames = np.load(args.trajectory, allow_pickle=True)["qpos"].astype(np.float64)
        frames = frames[::args.stride]
        loads = []
        for frame in frames:
            data = mujoco.MjData(robot)
            data.qpos[:] = frame
            mujoco.mj_forward(robot, data)
            loads.append([abs(data.qfrc_bias[dof]) for dof in dofs])
        load = np.percentile(np.array(loads), 95, axis=0)
        # Diagnostic poses are not guaranteed linear even contact-free: joints
        # riding their limits trip the guard on roughly a quarter of frames.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            per_frame = [constrained_inertia(robot, frame, dofs, strict=False)
                         for frame in frames]
            approximate = len(caught)
        inertia_min = np.nanmin(np.array(per_frame), axis=0)

    print(f"model {ROBOT_XML.relative_to(REPO_ROOT)}  dt={dt}  "
          f"integrator={mujoco.mjtIntegrator(robot.opt.integrator).name}")
    print("sizing poses: equality-reconciled qpos0 + all-midrange, each swept per joint")
    print(f"diagnostics:  {args.trajectory.name if args.trajectory else '(none)'}")
    if args.trajectory is not None and approximate:
        print(f"              WARNING: {approximate}/{len(frames)} frames had a nonlinear "
              f"inertia probe; the Kd*dt/I column is approximate")
    print()

    header = (f"{'joint':22s} {'u=-1':>7s} {'u=+1':>7s} {'lo':>7s} {'hi':>7s} {'in-rng':>7s} "
              f"{'I_eq':>7s} {'Kp':>7s} {'Kd':>6s} {'wn':>6s} {'zeta':>6s} {'Kd*dt/I':>8s} "
              f"{'flim':>6s} {'muscle':>7s}")
    print("CURRENT XML")
    print(header)
    print("-" * len(header))
    for index, (actuator, name, joint, dof) in enumerate(servos):
        lo, hi = robot.jnt_range[joint]
        kp = -robot.actuator_biasprm[actuator, 1]
        kd = -robot.actuator_biasprm[actuator, 2]
        gain = robot.actuator_gainprm[actuator, 0]
        bias = robot.actuator_biasprm[actuator, 0]
        low_target, high_target = (bias - gain) / kp, (bias + gain) / kp
        span = max(high_target - low_target, 1e-9)
        in_range = max(0.0, min(high_target, hi) - max(low_target, lo)) / span
        omega = math.sqrt(kp / inertia[index])
        zeta = kd / (2 * math.sqrt(kp * inertia[index]))
        stability = kd * dt / inertia_min[index]
        flag = ("  <== UNSTABLE" if stability >= 2.0
                else "  <== marginal" if stability >= 1.0 else "")
        print(f"{name:22s} {math.degrees(low_target):7.1f} {math.degrees(high_target):7.1f} "
              f"{math.degrees(lo):7.1f} {math.degrees(hi):7.1f} {in_range * 100:6.0f}% "
              f"{inertia[index]:7.4f} {kp:7.1f} {kd:6.1f} {omega:6.1f} {zeta:6.2f} "
              f"{stability:8.2f} {robot.actuator_forcerange[actuator, 1]:6.0f} "
              f"{capacity[name][0]:7.0f}{flag}")

    print(f"\nPROPOSED (forcerange from muscle capacity, Kp = flim/{args.saturation_error} "
          f"capped at wn={args.max_bandwidth}, Kd from zeta={args.damping_ratio})")
    proposed = (f"{'joint':22s} {'flim':>6s} {'+Nm':>7s} {'-Nm':>7s} {'peak at':>16s} "
                f"{'Kp':>7s} {'Kd':>6s} {'wn':>6s} {'Kd*dt/I':>8s} {'load':>6s} {'err':>6s}")
    print(proposed)
    print("-" * len(proposed))
    table = {}
    for index, (actuator, name, joint, dof) in enumerate(servos):
        peak, positive, negative, where = capacity[name]
        flim = max(5.0, round(peak / 5.0) * 5.0)
        kp = min(flim / args.saturation_error, inertia[index] * args.max_bandwidth ** 2)
        kd = 2 * args.damping_ratio * math.sqrt(kp * inertia[index])
        print(f"{name:22s} {flim:6.0f} {positive:7.1f} {negative:7.1f} {where:>16s} "
              f"{kp:7.1f} {kd:6.2f} {math.sqrt(kp / inertia[index]):6.1f} "
              f"{kd * dt / inertia_min[index]:8.2f} {load[index]:6.1f} "
              f"{math.degrees(load[index] / kp):5.2f}d")
        table[name] = {"forcerange": flim, "kp": round(float(kp), 1), "kd": round(float(kd), 2),
                       "inertia": float(inertia[index]), "peak_at": where}

    if args.json_out:
        args.json_out.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
