#!/usr/bin/env python3
"""Generate the flattened, position-controlled OSL full-body robot MJCF.

The source model is intentionally left untouched.  This script applies the same
fingerless topology used by :class:`MyoFullBody`, removes muscle actuation and
its tendons, and installs normalized absolute position servos followed by the
two original OSL torque motors.

Position-control mapping
------------------------
Each servo maps ``ctrl`` in ``[-1, 1]`` onto the joint's full range::

    q_target = midpoint + half_range * ctrl
    force    = Kp * (q_target - q) - Kd * qdot

encoded as ``gainprm[0] = Kp * half_range`` and
``biasprm[:3] = (Kp * midpoint, -Kp, -Kd)``, so ``-1`` and ``+1`` land exactly on
the lower and upper joint limits.  This is the same convention the SMPL humanoid
reference uses across all 69 of its actuators.

How POSITION_SERVOS was derived
-------------------------------
Three rules, all motion-independent -- no trajectory clip enters any parameter.
``scripts/analyze_osl_fullbody_robot_actuators.py`` reproduces the table and
exposes the three knobs (``--saturation-error``, ``--max-bandwidth``,
``--damping-ratio``); rerun it rather than editing numbers by hand.

1. ``forcerange`` is the peak moment the sibling muscle model
   (``osl_fullbody.xml``) can produce about that coordinate.  Muscle force is
   ``gain(length, velocity) * act``, linear in activation with no coupling
   between muscles, so the maximum is a linear program whose optimum is "every
   agonist at ``act=1``, every antagonist at 0" -- evaluated by summing positive
   and negative moment contributions separately, not as a co-contracted net.
   Moments are accumulated in the independent coordinate system, adding each
   equality follower's moment times ``dq_follower/dq_driver``; 19 of these 27
   joints drive no followers, but the torso gains 1.65-1.9x and ``elv_angle``
   inverts.  Capacity is the peak over each joint's own range at two reference
   poses, qpos0 and all-midrange -- qpos0 alone leaves ``elv_angle`` at exactly
   0 N*m, since the elevation plane is degenerate with the arm at the side.

2. ``Kp = min(forcerange / 0.35 rad, I_eff * 150**2)`` -- the servo saturates at
   20 degrees of tracking error, capped at a 150 rad/s natural frequency.

3. ``Kd = 2 * zeta * sqrt(Kp * I_eff)`` with ``zeta = 1``, where ``I_eff`` is the
   equality-constrained effective inertia of the *mechanism* at the reconciled
   qpos0 -- measured with the position servos disarmed and contacts disabled.

   Both neutralisations are load-bearing.  Probing the actuated robot makes the
   finite difference nonlinear (at ``ctrl = 0`` every servo commands its range
   midpoint, saturating 7 of 29 actuators) and reports arm inertia 48-62% low.
   Probing with contacts live sizes the arms against qpos0's 21.47 mm
   humerus/thorax interpenetration, which triples measured shoulder inertia; on
   that basis ``shoulder_rot`` sat at ``Kd * dt / I`` = 1.735 -- 87% of the Euler
   bound -- as soon as the arms separated.  qpos0 is only an XML default and is
   never a pose the model holds, but it corrupts any measurement taken there.

Rule 3 is what keeps explicit Euler viable.  ``Kd / I = 2 * zeta * omega``, so the
stability condition ``Kd * dt / I < 2`` reduces to ``zeta * omega * dt < 1``,
satisfied at any sane bandwidth.  The flat per-region values this table replaced
had no such guarantee and reached 3.0-3.6 at 1 ms and 3.3-6.8 at the 2 ms the
training env uses -- the sole cause of distal-joint blowups previously mistaken
for torque saturation.  ``mjDSBL_EULERDAMP`` (left enabled by the OSL
environments) makes ``dof_damping`` implicit but does *not* cover an actuator's
``biasprm[2]``, which is why the socket joints tolerate ``damping=10000`` while a
servo ``Kd`` of 30 diverges.

OSL_MOTORS is deliberately untouched: the prosthesis keeps its original gear
ratios and +/-2.88 control range.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_XML = REPO_ROOT / "models" / "osl_fullbody" / "body" / "osl_fullbody.xml"
OUTPUT_XML = REPO_ROOT / "models" / "osl_fullbody" / "body" / "osl_fullbody_robot.xml"
ASSET_ROOT = SOURCE_XML.parents[1]

# Keep this list in lockstep with MyoFullBody._apply_spec_changes.  Deleting a
# joint leaves the hand bodies in place, which is how the existing fingerless
# OSL environment preserves body/mass topology while reducing nq/nv.
FINGER_JOINTS = (
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "md2_flexion_r",
    "md3_flexion_r",
    "md4_flexion_r",
    "md5_flexion_r",
    "pm2_flexion_r",
    "pm3_flexion_r",
    "pm4_flexion_r",
    "pm5_flexion_r",
    "cmc_flexion_l",
    "cmc_abduction_l",
    "mp_flexion_l",
    "ip_flexion_l",
    "mcp2_flexion_l",
    "mcp2_abduction_l",
    "mcp3_flexion_l",
    "mcp3_abduction_l",
    "mcp4_flexion_l",
    "mcp4_abduction_l",
    "mcp5_flexion_l",
    "mcp5_abduction_l",
    "md2_flexion_l",
    "md3_flexion_l",
    "md4_flexion_l",
    "md5_flexion_l",
    "pm2_flexion_l",
    "pm3_flexion_l",
    "pm4_flexion_l",
    "pm5_flexion_l",
)

# (joint name, Kp, Kd, absolute actuator force limit).  Values are verbatim output of
# scripts/analyze_osl_fullbody_robot_actuators.py -- see this module's docstring for the
# three rules that produce them.  Regenerate rather than hand-editing: gainprm encodes
# Kp * half_range, so editing a Kp in the XML alone silently rescales that joint's
# normalized action semantics.
POSITION_SERVOS = (
    # Torso.
    ("flex_extension",       871.4,  68.54,  305.0),
    ("lat_bending",         1228.6,  87.52,  430.0),
    ("axial_rotation",       771.4,  30.21,  270.0),
    # Right arm.
    ("elv_angle_r",          442.9,  38.73,  155.0),           # 0 N*m at qpos0 alone (gimbal singularity)
    ("shoulder_elv_r",       314.3,  37.57,  110.0),
    ("shoulder_rot_r",       121.6,   1.62,   90.0),
    ("elbow_flex_r",         242.9,   6.20,   85.0),
    ("pro_sup_r",             42.9,   0.64,   15.0),
    ("deviation_r",           68.6,   0.91,   35.0),
    ("flexion_r",             81.0,   1.08,   40.0),
    # Left arm -- identical to the right: both sizing poses are bilaterally
    # symmetric, so every pair falls out equal with no manual symmetrization.
    ("elv_angle_l",          442.9,  38.73,  155.0),           # 0 N*m at qpos0 alone (gimbal singularity)
    ("shoulder_elv_l",       314.3,  37.57,  110.0),
    ("shoulder_rot_l",       121.6,   1.62,   90.0),
    ("elbow_flex_l",         242.9,   6.20,   85.0),
    ("pro_sup_l",             42.9,   0.64,   15.0),
    ("deviation_l",           68.6,   0.91,   35.0),
    ("flexion_l",             81.0,   1.08,   40.0),
    # Residual right hip.  Genuinely weaker than the left -- the amputation removes
    # muscles crossing this hip -- so do NOT symmetrize these with the left leg.
    ("hip_flexion_r",        814.3,  22.78,  285.0),
    ("hip_adduction_r",      671.4,  23.15,  235.0),
    ("hip_rotation_r",       457.1,  10.61,  160.0),
    # Intact left leg.
    ("hip_flexion_l",       1600.0,  34.16,  560.0),
    ("hip_adduction_l",     1200.0,  39.37,  420.0),
    ("hip_rotation_l",       528.6,   9.23,  185.0),
    ("knee_angle_l",        1557.1,  26.99,  545.0),
    ("ankle_angle_l",        343.8,   4.58,  425.0),           # Kp bandwidth-capped at wn=150
    ("subtalar_angle_l",     304.6,   4.06,  135.0),
    ("mtp_angle_l",           14.3,   0.76,    5.0),           # MyoLeg's toe flexors are weak; below human MVC
)

OSL_MOTORS = (
    ("osl_knee_torque_actuator", "osl_knee_angle_r", 49.4),
    ("osl_ankle_torque_actuator", "osl_ankle_angle_r", 58.4),
)


def _delete_finger_joints(spec: mujoco.MjSpec) -> None:
    joints = {joint.name: joint for joint in spec.joints}
    missing = set(FINGER_JOINTS) - joints.keys()
    if missing:
        raise RuntimeError(f"source model is missing finger joints: {sorted(missing)}")
    for name in FINGER_JOINTS:
        spec.delete(joints[name])


def _remove_actuation_and_muscle_tendons(spec: mujoco.MjSpec) -> None:
    muscle_tendons = {
        actuator.target
        for actuator in spec.actuators
        if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    }
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    tendons = {tendon.name: tendon for tendon in spec.tendons}
    missing = muscle_tendons - tendons.keys()
    if missing:
        raise RuntimeError(f"muscle actuators reference missing tendons: {sorted(missing)}")
    for name in muscle_tendons:
        spec.delete(tendons[name])


def _add_position_servo(
    spec: mujoco.MjSpec,
    joint_name: str,
    kp: float,
    kd: float,
    force_limit: float,
) -> None:
    joint = spec.joint(joint_name)
    if joint is None or joint.type != mujoco.mjtJoint.mjJNT_HINGE:
        raise RuntimeError(f"position servo joint must be a scalar hinge: {joint_name}")
    if not joint.limited:
        raise RuntimeError(f"position servo joint must have limits: {joint_name}")

    lower, upper = map(float, joint.range)
    midpoint = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0
    if half_range <= 0:
        raise RuntimeError(f"invalid range for position servo joint: {joint_name}")

    actuator = spec.add_actuator(
        name=f"{joint_name}_position_actuator",
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )
    # Affine general actuator:
    #   force = gainprm[0] * ctrl + biasprm[0:3] @ [1, q, qdot]
    #         = Kp * (midpoint + half_range * ctrl - q) - Kd * qdot
    # Thus ctrl=-1/+1 maps exactly to the joint's lower/upper limit.
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm[0] = kp * half_range
    actuator.biasprm[:3] = (kp * midpoint, -kp, -kd)
    actuator.ctrllimited = True
    actuator.ctrlrange = (-1.0, 1.0)
    actuator.forcelimited = True
    actuator.forcerange = (-force_limit, force_limit)


def _add_osl_motor(spec: mujoco.MjSpec, name: str, joint_name: str, gear: float) -> None:
    actuator = spec.add_actuator(
        name=name,
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )
    actuator.set_to_motor()
    actuator.gear[0] = gear
    actuator.ctrllimited = True
    actuator.ctrlrange = (-2.88, 2.88)


def build_spec() -> mujoco.MjSpec:
    """Build the robot spec before standalone XML serialization."""
    spec = mujoco.MjSpec.from_file(str(SOURCE_XML.resolve()))
    spec.modelname = "OSLFullBodyRobot"
    _delete_finger_joints(spec)
    _remove_actuation_and_muscle_tendons(spec)
    for servo in POSITION_SERVOS:
        _add_position_servo(spec, *servo)
    for motor in OSL_MOTORS:
        _add_osl_motor(spec, *motor)
    return spec


def _make_asset_paths_relative(xml: str) -> str:
    """Rewrite resolved asset filenames for a file stored in ``body/``."""
    root = ET.fromstring(xml)
    asset_root = ASSET_ROOT.resolve()
    for element in root.iter():
        filename = element.get("file")
        if not filename or not os.path.isabs(filename):
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(asset_root)
        except ValueError as exc:
            raise RuntimeError(f"generated asset escapes {asset_root}: {path}") from exc
        element.set("file", relative.as_posix())

    # ElementTree would discard comments and MuJoCo's stable formatting.  Apply
    # only the validated absolute-to-relative filename replacements to the
    # original serialization.
    for element in root.iter():
        filename = element.get("file")
        if filename:
            original = (asset_root / filename).as_posix()
            xml = xml.replace(f'file="{original}"', f'file="{filename}"')
    return xml


def generate_xml() -> str:
    spec = build_spec()
    xml = _make_asset_paths_relative(spec.to_xml())
    marker = f'<mujoco model="{spec.modelname}">'
    generated_notice = (
        "\n  <!-- Generated by scripts/generate_osl_fullbody_robot.py; "
        "do not edit manually. -->"
    )
    xml = xml.replace(marker, marker + generated_notice, 1)

    parsed = ET.fromstring(xml)
    absolute_assets = [
        element.get("file")
        for element in parsed.iter()
        if element.get("file") and os.path.isabs(element.get("file", ""))
    ]
    if absolute_assets:
        raise RuntimeError(f"absolute generated asset paths: {absolute_assets}")
    return xml


def _compile_generated(xml: str) -> mujoco.MjModel:
    # from_xml_string needs an asset dictionary for relative external assets;
    # compilation from the final path validates the same resolution users get.
    OUTPUT_XML.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_XML.with_suffix(".xml.tmp")
    temporary.write_text(xml, encoding="utf-8")
    try:
        return mujoco.MjModel.from_xml_path(str(temporary))
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the tracked XML differs from deterministic regeneration",
    )
    args = parser.parse_args()

    xml = generate_xml()
    model = _compile_generated(xml)
    expected = (84, 83, 78, 104, 44, 29, 0)
    actual = (model.nq, model.nv, model.njnt, model.nbody, model.neq, model.nu, model.na)
    if actual != expected:
        raise RuntimeError(f"unexpected model dimensions: {actual}, expected {expected}")

    if args.check:
        if not OUTPUT_XML.exists() or OUTPUT_XML.read_text(encoding="utf-8") != xml:
            print(f"{OUTPUT_XML.relative_to(REPO_ROOT)} is out of date", file=sys.stderr)
            return 1
        print(f"{OUTPUT_XML.relative_to(REPO_ROOT)} is up to date")
        return 0

    OUTPUT_XML.write_text(xml, encoding="utf-8")
    print(f"wrote {OUTPUT_XML.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
