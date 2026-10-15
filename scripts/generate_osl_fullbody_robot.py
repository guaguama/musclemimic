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

import dataclasses
from dataclasses import dataclass


REPO_ROOT = Path(__file__).resolve().parents[1]
_BODY = REPO_ROOT / "models" / "osl_fullbody" / "body"
SOURCE_XML = _BODY / "osl_fullbody.xml"
OUTPUT_XML = _BODY / "osl_fullbody_robot.xml"
ASSET_ROOT = SOURCE_XML.parents[1]

# (joint name, Kp, Kd, absolute actuator force limit).  Values are verbatim output of
#   scripts/analyze_osl_fullbody_robot_actuators.py --model osl_fullbody
# (recorded in scripts/osl_fullbody_robot_actuator_candidate.json) -- see this
# module's docstring for the three rules that produce them.  Regenerate rather than
# hand-editing: gainprm encodes Kp * half_range, so editing a Kp in the XML alone
# silently rescales that joint's normalized action semantics.
POSITION_SERVOS = (
    # Torso.
    ("flex_extension",          871.4,  68.89,  305.0),
    ("lat_bending",            1228.6,  90.77,  430.0),
    ("axial_rotation",          771.4,  30.30,  270.0),
    # Right arm.
    ("elv_angle_r",             442.9,  38.70,  155.0),         # 0 N*m at qpos0 alone (gimbal singularity)
    ("shoulder_elv_r",          314.3,  37.50,  110.0),
    ("shoulder_rot_r",          109.8,   1.46,   90.0),         # Kp bandwidth-capped at wn=150
    ("elbow_flexion_r",         242.9,   5.35,   85.0),
    ("pro_sup_r",                42.9,   0.61,   15.0),
    ("deviation_r",              58.0,   0.77,   35.0),         # Kp bandwidth-capped at wn=150
    ("flexion_r",                70.6,   0.94,   40.0),         # Kp bandwidth-capped at wn=150
    # Left arm -- identical to the right: both sizing poses are bilaterally
    # symmetric, so every pair falls out equal with no manual symmetrization.
    ("elv_angle_l",             442.9,  38.70,  155.0),         # 0 N*m at qpos0 alone (gimbal singularity)
    ("shoulder_elv_l",          314.3,  37.50,  110.0),
    ("shoulder_rot_l",          109.8,   1.46,   90.0),         # Kp bandwidth-capped at wn=150
    ("elbow_flexion_l",         242.9,   5.35,   85.0),
    ("pro_sup_l",                42.9,   0.61,   15.0),
    ("deviation_l",              58.0,   0.77,   35.0),         # Kp bandwidth-capped at wn=150
    ("flexion_l",                70.6,   0.94,   40.0),         # Kp bandwidth-capped at wn=150
    # Residual right hip.  The official export keeps enough hip musculature that
    # flexion matches the intact side -- these are NOT symmetrized, they simply
    # came out near-equal.
    ("hip_flexion_r",          1600.0,  32.09,  560.0),
    ("hip_adduction_r",        1157.1,  39.54,  405.0),
    ("hip_rotation_r",          485.7,   8.63,  170.0),
    # Intact left leg.
    ("hip_flexion_l",          1600.0,  34.41,  560.0),
    ("hip_adduction_l",        1200.0,  41.23,  420.0),
    ("hip_rotation_l",          528.6,   9.29,  185.0),
    ("knee_angle_l",           1257.1,  24.65,  440.0),         # 545 in v2; myo_sim 0.2.3 fixed the left-knee coupling
    ("ankle_angle_l",           339.3,   4.52,  425.0),         # Kp bandwidth-capped at wn=150
    ("subtalar_angle_l",        303.3,   4.04,  130.0),         # Kp bandwidth-capped at wn=150
    ("mtp_angle_l",              14.3,   0.76,    5.0),         # MyoLeg's toe flexors are weak; below human MVC
)

# Joint names carry the ``OSL_KA_L1_`` namespace of the official export; the actuator
# names do not.  The export scales force through ``gainprm``; ``gear`` here
# reproduces it exactly, since force is ``gear * gainprm[0] * ctrl`` either way, so
# peak torque stays 142.27 / 168.19 N*m at the unchanged +-2.88 ctrlrange.
OSL_MOTORS = (
    ("osl_knee_torque_actuator", "OSL_KA_L1_osl_knee_angle_r", 49.4),
    ("osl_ankle_torque_actuator", "OSL_KA_L1_osl_ankle_angle_r", 58.4),
)


# The official export dropped the ``_r`` suffix on four right-hand joints.  The
# corrected list already exists for the environment; importing it keeps the two in
# lockstep by construction.  Imported lazily so this script does not pull in JAX
# until a profile is actually resolved.
def _finger_joints() -> tuple[str, ...]:
    from musclemimic.environments.humanoids.osl_fullbody import _FINGER_JOINTS

    return tuple(_FINGER_JOINTS)


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """Everything that differs between the models this script can emit."""

    source_xml: Path
    output_xml: Path
    model_name: str
    finger_joints: tuple[str, ...]
    position_servos: tuple[tuple[str, float, float, float], ...]
    osl_motors: tuple[tuple[str, str, float], ...]
    dimensions: tuple[int, ...]  # (nq, nv, njnt, nbody, neq, nu, na)

    @property
    def asset_root(self) -> Path:
        return self.source_xml.parents[1]


DEFAULT_MODEL = "osl_fullbody"

MODELS: dict[str, ModelProfile] = {
    # Both socket variants share ONE servo table: the analysis is run once, on the
    # rigid model.  That is safe rather than exact -- the compliant socket lowers
    # effective inertia at the right hip (up to 50% at hip_adduction_r), so the
    # shared Kd lands above zeta = 1 there.  Overdamped, not unstable; the robot
    # tests hold the compliant variant to the explicit-Euler bound instead.
    "osl_fullbody": ModelProfile(
        source_xml=SOURCE_XML,
        output_xml=OUTPUT_XML,
        model_name="OSLFullBodyRobot",
        finger_joints=(),  # filled lazily -- see _resolve
        position_servos=POSITION_SERVOS,
        osl_motors=OSL_MOTORS,
        dimensions=(80, 79, 74, 106, 44, 29, 0),
    ),
    "osl_fullbody_compliant_socket": ModelProfile(
        source_xml=_BODY / "osl_fullbody_compliant_socket.xml",
        output_xml=_BODY / "osl_fullbody_robot_compliant_socket.xml",
        model_name="OSLFullBodyRobot",
        finger_joints=(),
        position_servos=POSITION_SERVOS,
        osl_motors=OSL_MOTORS,
        dimensions=(84, 83, 78, 106, 44, 29, 0),
    ),
}


def _resolve(name: str) -> ModelProfile:
    """The profile for ``name``, with the finger list filled in."""
    profile = MODELS[name]
    if not profile.finger_joints:
        profile = dataclasses.replace(profile, finger_joints=_finger_joints())
    return profile


def _strip_keyframes(spec: mujoco.MjSpec) -> None:
    """Drop every keyframe, which deleting a joint otherwise invalidates.

    A spec carrying explicit ``<key qpos=...>`` fails to compile once a joint goes:
    ``keyframe 'stand': invalid qpos size, expected 119, got 120``.  The official
    export ships five (``stand``, ``walk_*``, ``squat``, ``lunge``).

    ``spec.nkey = 0`` is the non-obvious half: ``<size nkey="5"/>`` allocates the
    slots independently of the ``<keyframe>`` entries, so without it the model
    keeps five all-zero keys -- and an all-zero qpos carries a zero root quaternion.

    Guarded on there actually being keyframes: a source may declare ``<size nkey>``
    with no ``<keyframe>`` at all (the hand-assembled model on ``old_osl`` does), and
    clearing ``nkey`` there would change its output for no reason.
    """
    keys = list(spec.keys)
    if not keys:
        return
    for key in keys:
        spec.delete(key)
    spec.nkey = 0


def _delete_finger_joints(spec: mujoco.MjSpec, finger_joints: tuple[str, ...]) -> None:
    joints = {joint.name: joint for joint in spec.joints}
    missing = set(finger_joints) - joints.keys()
    if missing:
        raise RuntimeError(f"source model is missing finger joints: {sorted(missing)}")
    for name in finger_joints:
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


def _rebase_assets(spec: mujoco.MjSpec, profile: "ModelProfile") -> None:
    """Point both asset dirs at the model root and rewrite mesh paths to match.

    The generated file lives in ``body/`` while ``_make_asset_paths_relative``
    writes filenames relative to the MODEL ROOT, so ``meshdir``/``texturedir`` have
    to be ``..``.  The vendored source has ``meshdir="../meshes"`` and no
    ``texturedir``, with bare basenames like ``sacrum.stl``.  Repointing ``meshdir``
    alone would resolve those to ``body/../sacrum.stl``; leaving it alone would
    resolve the skybox (which arrives absolute, via the scene include) to
    ``body/scene/...``.  So the subdirectory moves out of ``meshdir`` and into each
    filename.  A source already at ``meshdir=".."`` passes through unchanged.
    """
    asset_root = profile.asset_root.resolve()
    mesh_root = (profile.source_xml.parent / (spec.meshdir or ".")).resolve()
    prefix = mesh_root.relative_to(asset_root)
    if prefix != Path("."):
        for mesh in spec.meshes:
            if mesh.file and not os.path.isabs(mesh.file):
                mesh.file = (prefix / mesh.file).as_posix()
    spec.meshdir = ".."
    spec.texturedir = ".."


def build_spec(profile: ModelProfile = None) -> mujoco.MjSpec:
    """Build the robot spec before standalone XML serialization."""
    profile = profile if profile is not None else _resolve(DEFAULT_MODEL)
    spec = mujoco.MjSpec.from_file(str(profile.source_xml.resolve()))
    spec.modelname = profile.model_name
    _rebase_assets(spec, profile)
    _strip_keyframes(spec)
    _delete_finger_joints(spec, profile.finger_joints)
    _remove_actuation_and_muscle_tendons(spec)
    for servo in profile.position_servos:
        _add_position_servo(spec, *servo)
    for motor in profile.osl_motors:
        _add_osl_motor(spec, *motor)
    return spec


def _make_asset_paths_relative(xml: str, asset_root: Path = None) -> str:
    """Rewrite resolved asset filenames for a file stored in ``body/``."""
    root = ET.fromstring(xml)
    asset_root = (asset_root if asset_root is not None else ASSET_ROOT).resolve()
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


def generate_xml(profile: ModelProfile = None) -> str:
    profile = profile if profile is not None else _resolve(DEFAULT_MODEL)
    spec = build_spec(profile)
    xml = _make_asset_paths_relative(spec.to_xml(), profile.asset_root)
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


def _compile_generated(xml: str, profile: ModelProfile = None) -> mujoco.MjModel:
    # from_xml_string needs an asset dictionary for relative external assets;
    # compilation from the final path validates the same resolution users get.
    profile = profile if profile is not None else _resolve(DEFAULT_MODEL)
    output = profile.output_xml
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".xml.tmp")
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
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL,
        help="which robot model to generate (default: %(default)s)",
    )
    args = parser.parse_args()

    profile = _resolve(args.model)
    output = profile.output_xml
    xml = generate_xml(profile)
    model = _compile_generated(xml, profile)
    actual = (model.nq, model.nv, model.njnt, model.nbody, model.neq, model.nu, model.na)
    if actual != profile.dimensions:
        raise RuntimeError(
            f"unexpected model dimensions: {actual}, expected {profile.dimensions}"
        )

    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != xml:
            print(f"{output.relative_to(REPO_ROOT)} is out of date", file=sys.stderr)
            return 1
        print(f"{output.relative_to(REPO_ROOT)} is up to date")
        return 0

    output.write_text(xml, encoding="utf-8")
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
