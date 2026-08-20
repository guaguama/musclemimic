#!/usr/bin/env python3
"""Interactive pose-holding and actuator-tuning viewer for OSLFullBodyRobot.

Examples
--------
Hold a constraint-consistent version of qpos0 with the root fixed::

    .venv/bin/python scripts/tune_osl_fullbody_robot.py

Hold frame 100 of a pre-retargeted environment-format trajectory::

    .venv/bin/python scripts/tune_osl_fullbody_robot.py \
        --trajectory /path/to/trajectory.npz --frame 100 --vertical-offset 0.05

Try stronger torso stiffness without changing the tracked XML::

    .venv/bin/python scripts/tune_osl_fullbody_robot.py \
        --trajectory /path/to/trajectory.npz --frame 100 \
        --kp torso=400 --kd torso=30 --limit torso=360

Overrides accept ``all``, ``torso``, ``arms``, ``right_hip``, ``left_leg``,
an actuated joint name, or a position-actuator name.  They modify only the
in-memory model.  Close the viewer or press Ctrl-C to exit.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "models/osl_fullbody/body/osl_fullbody_robot.xml"

POSITION_REGIONS = {
    "all": range(27),
    "torso": range(0, 3),
    "arms": range(3, 17),
    "right_arm": range(3, 10),
    "left_arm": range(10, 17),
    "right_hip": range(17, 20),
    "left_leg": range(20, 27),
}


def _name(model: mujoco.MjModel, objtype: mujoco.mjtObj, objid: int) -> str:
    return mujoco.mj_id2name(model, objtype, int(objid)) or f"id_{objid}"


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def project_joint_equalities(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Set scalar joint-equality followers to their exact polynomial values."""
    projected = np.array(qpos, dtype=float, copy=True)
    # ``eq_objtype`` is left at zero for MuJoCo joint equalities; ``eq_type``
    # is the authoritative discriminator here.
    joint_equalities = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    # All equalities in this model point from a follower (obj1) to either an
    # independent scalar joint (obj2) or the constant zero input (obj2 == -1).
    # Iterate in case a future model introduces a short follower chain.
    for _ in range(3):
        for equality_id in joint_equalities:
            follower_id = int(model.eq_obj1id[equality_id])
            independent_id = int(model.eq_obj2id[equality_id])
            follower_qadr = int(model.jnt_qposadr[follower_id])
            independent = (
                projected[model.jnt_qposadr[independent_id]] if independent_id >= 0 else 0.0
            )
            coefficients = model.eq_data[equality_id, :5]
            projected[follower_qadr] = np.polynomial.polynomial.polyval(
                independent, coefficients
            )
    return projected


def load_target_qpos(
    model: mujoco.MjModel, trajectory: Path | None, frame: int
) -> tuple[np.ndarray, str]:
    """Load a full-model target qpos, then reconcile equality followers."""
    if trajectory is None:
        qpos = model.qpos0.copy()
        label = "qpos0"
    else:
        with np.load(trajectory, allow_pickle=True) as archive:
            if "qpos" not in archive:
                raise ValueError(f"{trajectory} has no 'qpos' array")
            qposes = np.asarray(archive["qpos"])
            if qposes.ndim != 2 or qposes.shape[1] != model.nq:
                raise ValueError(
                    f"{trajectory} qpos shape is {qposes.shape}; expected (frames, {model.nq})"
                )
            resolved_frame = frame if frame >= 0 else len(qposes) + frame
            if not 0 <= resolved_frame < len(qposes):
                raise IndexError(f"frame {frame} is outside a {len(qposes)}-frame trajectory")
            qpos = qposes[resolved_frame].astype(float, copy=True)
        label = f"{trajectory.name} frame {resolved_frame}"
    return project_joint_equalities(model, qpos), label


def offset_root_height(
    model: mujoco.MjModel, qpos: np.ndarray, vertical_offset: float
) -> np.ndarray:
    """Translate the target root vertically, leaving the floor unchanged."""
    if not np.isfinite(vertical_offset):
        raise ValueError(f"vertical offset must be finite, got {vertical_offset}")
    shifted = np.array(qpos, dtype=float, copy=True)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_id < 0 or model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("vertical offset requires a free joint named 'root'")
    root_qadr = int(model.jnt_qposadr[root_id])
    shifted[root_qadr + 2] += vertical_offset
    return shifted


def compile_test_model(
    xml_path: Path, full_model: mujoco.MjModel, full_target: np.ndarray, fix_root: bool
) -> tuple[mujoco.MjModel, np.ndarray]:
    """Compile the test model and map the full-model target into its coordinates."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if fix_root:
        root_id = mujoco.mj_name2id(full_model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_id < 0 or full_model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("--fix-root requires a free joint named 'root'")
        root_qadr = int(full_model.jnt_qposadr[root_id])
        root_spec = spec.joint("root")
        root_spec.parent.pos = full_target[root_qadr : root_qadr + 3]
        root_spec.parent.quat = full_target[root_qadr + 3 : root_qadr + 7]
        spec.delete(root_spec)

    model = spec.compile()
    target = model.qpos0.copy()
    for joint_id in range(model.njnt):
        joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        source_id = mujoco.mj_name2id(full_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if source_id < 0:
            raise ValueError(f"compiled joint {joint_name!r} is absent from the source model")
        width = _joint_qpos_width(int(model.jnt_type[joint_id]))
        source_qadr = int(full_model.jnt_qposadr[source_id])
        target_qadr = int(model.jnt_qposadr[joint_id])
        target[target_qadr : target_qadr + width] = full_target[
            source_qadr : source_qadr + width
        ]
    return model, target


def equality_residual(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    equality_rows = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    if not np.any(equality_rows):
        return 0.0
    return float(np.max(np.abs(data.efc_pos[: data.nefc][equality_rows])))


def _parse_assignments(assignments: Iterable[str], flag: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for assignment in assignments:
        try:
            key, text_value = assignment.split("=", 1)
            value = float(text_value)
        except ValueError as error:
            raise ValueError(f"{flag} expects NAME=VALUE, got {assignment!r}") from error
        if not key or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid {flag} assignment {assignment!r}")
        result[key] = value
    return result


def _position_actuator_ids(model: mujoco.MjModel, selector: str) -> list[int]:
    if selector in POSITION_REGIONS:
        return list(POSITION_REGIONS[selector])
    for actuator_id in range(min(27, model.nu)):
        actuator_name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if selector in (actuator_name, joint_name):
            return [actuator_id]
    choices = ", ".join(POSITION_REGIONS)
    raise ValueError(f"unknown actuator selector {selector!r}; use {choices}, a joint, or an actuator")


def apply_parameter_overrides(
    model: mujoco.MjModel,
    kp_assignments: Iterable[str],
    kd_assignments: Iterable[str],
    limit_assignments: Iterable[str],
) -> None:
    """Apply absolute position-servo parameters to the compiled model."""
    for selector, kp in _parse_assignments(kp_assignments, "--kp").items():
        for actuator_id in _position_actuator_ids(model, selector):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            lower, upper = model.jnt_range[joint_id]
            midpoint = (lower + upper) / 2.0
            half_range = (upper - lower) / 2.0
            model.actuator_gainprm[actuator_id, 0] = kp * half_range
            model.actuator_biasprm[actuator_id, 0] = kp * midpoint
            model.actuator_biasprm[actuator_id, 1] = -kp
    for selector, kd in _parse_assignments(kd_assignments, "--kd").items():
        for actuator_id in _position_actuator_ids(model, selector):
            model.actuator_biasprm[actuator_id, 2] = -kd
    for selector, limit in _parse_assignments(limit_assignments, "--limit").items():
        for actuator_id in _position_actuator_ids(model, selector):
            model.actuator_forcelimited[actuator_id] = True
            model.actuator_forcerange[actuator_id] = (-limit, limit)


def position_controls_for_target(model: mujoco.MjModel, target: np.ndarray) -> np.ndarray:
    """Return normalized controls that make each body servo target target qpos."""
    controls = np.zeros(27)
    for actuator_id in range(27):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        gain = model.actuator_gainprm[actuator_id, 0]
        bias_constant = model.actuator_biasprm[actuator_id, 0]
        bias_position = model.actuator_biasprm[actuator_id, 1]
        if gain == 0:
            raise ValueError(f"actuator {actuator_id} has zero gain")
        raw = -(bias_constant + bias_position * target[qadr]) / gain
        controls[actuator_id] = np.clip(raw, *model.actuator_ctrlrange[actuator_id])
        # Retargeted float32 caches can exceed a serialized joint endpoint by
        # roughly 1e-4 in normalized-control units.  Clamp that harmless
        # roundoff, but reject a materially unreachable target.
        if abs(raw - controls[actuator_id]) > 1e-3:
            joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(
                f"target for {joint_name} requires control {raw:.5g}, outside "
                f"{model.actuator_ctrlrange[actuator_id]}"
            )
    return controls


def osl_position_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    kp: float,
    kd: float,
    knee_limit: float,
    ankle_limit: float,
) -> None:
    for actuator_id, torque_limit in ((27, knee_limit), (28, ankle_limit)):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        torque = kp * (target[qadr] - data.qpos[qadr]) - kd * data.qvel[dadr]
        torque = float(np.clip(torque, -torque_limit, torque_limit))
        gear = model.actuator_gear[actuator_id, 0]
        data.ctrl[actuator_id] = np.clip(
            torque / gear,
            model.actuator_ctrlrange[actuator_id, 0],
            model.actuator_ctrlrange[actuator_id, 1],
        )


def build_target_ghost_scene(
    model: mujoco.MjModel, target_data: mujoco.MjData
) -> mujoco.MjvScene:
    """Build MuJoCo's native render scene for the desired configuration.

    In particular, ``MjvGeom.dataid`` is a render-time asset ID and is not
    always equal to ``model.geom_dataid`` for meshes.  Letting
    :func:`mjv_updateScene` prepare the geoms matches the path used by the
    project's enhanced trajectory-goal visualization.
    """
    scene = mujoco.MjvScene(model, maxgeom=max(1000, model.ngeom * 2))
    option = mujoco.MjvOption()
    perturbation = mujoco.MjvPerturb()
    camera = mujoco.MjvCamera()
    mujoco.mjv_updateScene(
        model,
        target_data,
        option,
        perturbation,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return scene


def _copy_mjv_geom(source: mujoco.MjvGeom, destination: mujoco.MjvGeom) -> None:
    for field in ("pos", "mat", "size", "rgba"):
        getattr(destination, field)[:] = getattr(source, field)
    for field in (
        "type",
        "dataid",
        "objtype",
        "objid",
        "category",
        "matid",
        "segid",
        "emission",
        "specular",
        "shininess",
        "reflectance",
        "texcoord",
        "modelrbound",
        "transparent",
        "camdist",
    ):
        setattr(destination, field, getattr(source, field))
    destination.label = source.label


def add_target_ghost(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    target_scene: mujoco.MjvScene,
    rgba: np.ndarray,
) -> None:
    """Copy group-0 target geoms using GoalTrajMimicv2's selection/color."""
    scene = viewer.user_scn
    scene.ngeom = 0
    for source_id in range(target_scene.ngeom):
        source = target_scene.geoms[source_id]
        if source.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        if source.objid < 0 or model.geom_group[source.objid] != 0:
            continue
        if scene.ngeom >= scene.maxgeom:
            break
        ghost = scene.geoms[scene.ngeom]
        _copy_mjv_geom(source, ghost)
        ghost.rgba[:] = rgba
        ghost.transparent = 1
        scene.ngeom += 1


def print_parameter_table(model: mujoco.MjModel, controls: np.ndarray) -> None:
    print("\nPosition servos (in-memory values)")
    print(f"{'joint':24} {'Kp':>8} {'Kd':>8} {'limit':>9} {'target u':>10}")
    for actuator_id in range(27):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        kp = -model.actuator_biasprm[actuator_id, 1]
        kd = -model.actuator_biasprm[actuator_id, 2]
        limit = model.actuator_forcerange[actuator_id, 1]
        print(f"{joint_name:24} {kp:8.2f} {kd:8.2f} {limit:9.2f} {controls[actuator_id]:10.4f}")


def report_tracking(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> None:
    rows = []
    for actuator_id in range(27):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        error_deg = np.degrees(target[qadr] - data.qpos[qadr])
        force = float(data.actuator_force[actuator_id])
        limit = float(model.actuator_forcerange[actuator_id, 1])
        rows.append((abs(error_deg), joint_id, error_deg, force, abs(force) / limit))
    largest = sorted(rows, reverse=True)[:6]
    text = ", ".join(
        f"{_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)} "
        f"err={error:+.1f}deg tau={force:+.1f}Nm ({fraction:.0%})"
        for _, joint_id, error, force, fraction in largest
    )
    print(f"t={data.time:7.2f}s | {text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--trajectory", type=Path, help="environment-format NPZ containing qpos")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--vertical-offset",
        type=float,
        default=0.0,
        metavar="METERS",
        help="raise both the simulated root and target ghost relative to the floor",
    )
    root_group = parser.add_mutually_exclusive_group()
    root_group.add_argument("--fix-root", action="store_true", default=True)
    root_group.add_argument("--free-root", action="store_false", dest="fix_root")
    parser.add_argument("--kp", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--kd", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--limit", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--osl-kp", type=float, default=300.0)
    parser.add_argument("--osl-kd", type=float, default=30.0)
    parser.add_argument("--osl-knee-limit", type=float, default=49.4 * 2.88)
    parser.add_argument("--osl-ankle-limit", type=float, default=58.4 * 2.88)
    parser.add_argument("--report-interval", type=float, default=1.0)
    parser.add_argument("--no-ghost", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    xml_path = args.xml.resolve()
    trajectory = args.trajectory.resolve() if args.trajectory else None
    full_model = mujoco.MjModel.from_xml_path(str(xml_path))
    full_target, pose_label = load_target_qpos(full_model, trajectory, args.frame)
    full_target = offset_root_height(full_model, full_target, args.vertical_offset)
    model, target = compile_test_model(xml_path, full_model, full_target, args.fix_root)
    apply_parameter_overrides(model, args.kp, args.kd, args.limit)

    residual = equality_residual(model, target)
    if residual > 1e-6:
        raise RuntimeError(f"target equality residual is unexpectedly large: {residual:.6g}")

    controls = position_controls_for_target(model, target)
    data = mujoco.MjData(model)
    data.qpos[:] = target
    data.qvel[:] = 0
    data.ctrl[:27] = controls
    osl_position_control(
        model,
        data,
        target,
        args.osl_kp,
        args.osl_kd,
        args.osl_knee_limit,
        args.osl_ankle_limit,
    )
    mujoco.mj_forward(model, data)

    target_data = mujoco.MjData(model)
    target_data.qpos[:] = target
    mujoco.mj_forward(model, target_data)
    target_scene = build_target_ghost_scene(model, target_data)

    print(f"Pose: {pose_label}")
    print(f"Vertical offset: {args.vertical_offset:+.4f} m")
    print(f"Root: {'fixed at the target pose' if args.fix_root else 'free'}")
    print(f"Maximum target equality residual: {residual:.3e}")
    print_parameter_table(model, controls)
    print("\nPurple ghost: desired pose. Solid model: simulated pose.")
    print("Close the viewer or press Ctrl-C to exit. Overrides do not edit the XML.\n")

    sync_period = 1.0 / 60.0
    next_sync = time.monotonic()
    next_report = args.report_interval if args.report_interval > 0 else np.inf
    # Same default target color as GoalTrajMimicv2, with enough opacity to be
    # legible beside the solid simulated skeleton.
    ghost_rgba = np.array([0.471, 0.38, 0.812, 0.5], dtype=np.float32)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.monotonic()
                data.ctrl[:27] = controls
                osl_position_control(
                    model,
                    data,
                    target,
                    args.osl_kp,
                    args.osl_kd,
                    args.osl_knee_limit,
                    args.osl_ankle_limit,
                )
                mujoco.mj_step(model, data)

                if data.time >= next_report:
                    report_tracking(model, data, target)
                    next_report += args.report_interval

                now = time.monotonic()
                if now >= next_sync:
                    with viewer.lock():
                        if not args.no_ghost:
                            add_target_ghost(viewer, model, target_scene, ghost_rgba)
                    viewer.sync()
                    next_sync = now + sync_period

                remaining = model.opt.timestep - (time.monotonic() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
