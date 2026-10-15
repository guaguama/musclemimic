"""Tests for the position-controlled robot twin of the official MyoAssist OSL model.

Three properties of this model are easy to mistake for oversights; each is
commented where it appears:

* both socket variants are generated from ONE servo table, so zeta == 1 is a
  rigid-variant invariant and the compliant variant is held to the stability
  bound instead;
* the passive socket coordinates exist only in the compliant variant, and are
  named ``socket_rotation_1/2/3``;
* the analyzer's fingerless-muscle sizing basis is pinned here, because that
  equivalence is what lets a model be sized before its robot XML exists.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from loco_mujoco.core.mujoco_base import Mujoco
from musclemimic.environments.humanoids import (
    MjxOSLFullBodyRobot,
    OSLFullBodyRobot,
)
from scripts import analyze_osl_fullbody_robot_actuators as analyzer
from scripts import generate_osl_fullbody_robot as generator


RIGID = "osl_fullbody"
COMPLIANT = "osl_fullbody_compliant_socket"
_TIMESTEP = 0.002  # the training env's dt, not the XML's

EXPECTED_JOINTS = [servo[0] for servo in generator.MODELS[RIGID].position_servos]
EXPECTED_ACTIONS = [
    *[f"{name}_position_actuator" for name in EXPECTED_JOINTS],
    *[motor[0] for motor in generator.MODELS[RIGID].osl_motors],
]
_SOCKET_JOINTS = (
    "socket_piston",
    "socket_rotation_1",
    "socket_rotation_2",
    "socket_rotation_3",
)


def _names(model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, objtype, index) for index in range(count)]


@pytest.fixture(scope="module")
def rigid_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(generator.MODELS[RIGID].output_xml.resolve()))


@pytest.fixture(scope="module")
def compliant_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(
        str(generator.MODELS[COMPLIANT].output_xml.resolve())
    )


@pytest.mark.parametrize("model_name", [RIGID, COMPLIANT])
def test_generation_is_deterministic(model_name):
    result = subprocess.run(
        [sys.executable, str(Path(generator.__file__).resolve()), "--model", model_name,
         "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "fixture_name, dimensions",
    [("rigid_model", (80, 79, 74, 106, 44, 29, 0)),
     ("compliant_model", (84, 83, 78, 106, 44, 29, 0))],
)
def test_model_dimensions_mass_and_actuation(request, fixture_name, dimensions):
    model = request.getfixturevalue(fixture_name)
    actual = (model.nq, model.nv, model.njnt, model.nbody, model.neq, model.nu, model.na)
    assert actual == dimensions
    # Mass must survive muscle removal untouched -- it is the same skeleton.
    assert model.body_mass.sum() == pytest.approx(82.28906, rel=1e-6)
    assert not np.any(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    # The five keyframes are dropped: deleting a finger joint from a spec that has
    # them fails to compile, and <size nkey> would otherwise leave all-zero slots
    # whose zero root quaternion is not a valid pose.
    assert model.nkey == 0


@pytest.mark.parametrize("fixture_name", ["rigid_model", "compliant_model"])
def test_actuator_order_and_servo_parameters(request, fixture_name):
    model = request.getfixturevalue(fixture_name)
    assert _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu) == EXPECTED_ACTIONS

    servos = generator.MODELS[RIGID].position_servos
    for actuator_id, (joint_name, kp, kd, force_limit) in enumerate(servos):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        lower, upper = model.jnt_range[joint_id]
        midpoint = (lower + upper) / 2.0
        half_range = (upper - lower) / 2.0
        assert model.actuator_trnid[actuator_id, 0] == joint_id
        assert model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
        np.testing.assert_allclose(model.actuator_ctrlrange[actuator_id], (-1.0, 1.0))
        np.testing.assert_allclose(
            model.actuator_forcerange[actuator_id], (-force_limit, force_limit)
        )
        # rtol=1e-5 is the six-significant-figure MJCF serialization floor, not
        # arithmetic error: Kp * half_range round-trips through text.
        np.testing.assert_allclose(
            model.actuator_gainprm[actuator_id, 0], kp * half_range, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            model.actuator_biasprm[actuator_id, :3],
            (kp * midpoint, -kp, -kd),
            rtol=1e-5,
            atol=1e-6,
        )

    targeted = set(model.actuator_trnid[:, 0].tolist())
    followers = set(model.eq_obj1id[model.eq_objtype == mujoco.mjtObj.mjOBJ_JOINT])
    assert followers.isdisjoint(targeted)
    assert not np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_BALL)


def test_socket_coordinates_are_passive_and_only_in_the_compliant_variant(
    rigid_model, compliant_model
):
    """The rigid export welds the socket away entirely; the compliant one keeps it free."""
    rigid_joints = set(_names(rigid_model, mujoco.mjtObj.mjOBJ_JOINT, rigid_model.njnt))
    assert not rigid_joints & set(_SOCKET_JOINTS)

    targeted = set(compliant_model.actuator_trnid[:, 0].tolist())
    for name in _SOCKET_JOINTS:
        joint_id = mujoco.mj_name2id(compliant_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, name
        assert joint_id not in targeted, f"{name} must stay passive"


def test_servo_gains_satisfy_the_sizing_rules_on_the_rigid_variant(rigid_model):
    """Pin the sizing *rules*, not the table-to-XML copy.

    ``test_actuator_order_...`` asserts XML <-> table, so it stays green even if
    both were regenerated from a contaminated probe.  This recomputes the rules
    against a freshly measured servo-free, contact-free ``I_eff``.

    Rigid variant only: see the compliant test below for why zeta is not 1 there.
    """
    servos = analyzer.servo_actuators(rigid_model)
    reference = analyzer.project_joint_equalities(rigid_model, rigid_model.qpos0)
    inertia = analyzer.constrained_inertia(
        rigid_model, reference, [dof for _, _, _, dof in servos]
    )
    assert np.all(np.isfinite(inertia))

    table = {n: (kp, kd, f) for n, kp, kd, f in generator.MODELS[RIGID].position_servos}
    stability = {}
    for index, (_, joint_name, _, _) in enumerate(servos):
        kp, kd, force_limit = table[joint_name]
        moment = inertia[index]

        # Rule 2, with atol for the 1-decimal Kp stored in the table.
        expected_kp = min(force_limit / 0.35, moment * 150.0**2)
        assert kp == pytest.approx(expected_kp, abs=0.051), joint_name

        # Rule 3: critically damped at the sizing pose.
        zeta = kd / (2.0 * math.sqrt(kp * moment))
        assert zeta == pytest.approx(1.0, abs=0.01), f"{joint_name} zeta={zeta:.3f}"

        stability[joint_name] = kd * _TIMESTEP / moment

    worst = max(stability, key=stability.get)
    assert stability[worst] < 1.0, f"{worst} Kd*dt/I={stability[worst]:.3f}"

    # Bilateral arm pairs fall out equal because both sizing poses are symmetric.
    for stem in ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion",
                 "pro_sup", "deviation", "flexion"):
        assert table[f"{stem}_r"] == table[f"{stem}_l"], stem


def test_shared_table_stays_stable_on_the_compliant_variant(compliant_model):
    """The compliant variant runs the rigid variant's gains -- deliberately.

    One analysis pass feeds both models, which is safe rather than exact.  The
    compliant socket sits in the right-leg chain and lowers effective inertia
    there by up to 50% (``hip_adduction_r`` 0.3377 -> 0.1648), so those three
    coordinates land near zeta 1.1-1.43 instead of 1: overdamped, not unstable.

    So the invariant here is the explicit-Euler bound, NOT zeta == 1.  Asserting
    zeta would fail at ``hip_adduction_r`` by design.
    """
    servos = analyzer.servo_actuators(compliant_model)
    reference = analyzer.project_joint_equalities(compliant_model, compliant_model.qpos0)
    inertia = analyzer.constrained_inertia(
        compliant_model, reference, [dof for _, _, _, dof in servos]
    )
    assert np.all(np.isfinite(inertia))

    worst_ratio, worst_zeta = 0.0, 0.0
    for index, (actuator, joint_name, _, _) in enumerate(servos):
        kp = -compliant_model.actuator_biasprm[actuator, 1]
        kd = -compliant_model.actuator_biasprm[actuator, 2]
        worst_ratio = max(worst_ratio, kd * _TIMESTEP / inertia[index])
        worst_zeta = max(worst_zeta, kd / (2.0 * math.sqrt(kp * inertia[index])))

    # Measured 0.600; 1.0 leaves headroom while staying far from the bound of 2.0.
    assert worst_ratio < 1.0, f"worst Kd*dt/I = {worst_ratio:.3f}"
    # Measured 1.432 at hip_adduction_r.  Guards against the socket ever coupling
    # more strongly than it does today, which would push this toward instability.
    assert worst_zeta < 1.6, f"worst zeta = {worst_zeta:.3f}"


def test_osl_motor_parameters_are_unchanged(rigid_model):
    """Peak torque must stay 142.27 / 168.19 N*m at the +-2.88 ctrlrange.

    The official source expresses the scale as ``gainprm`` where the hand-built
    model used ``gear``; force is ``gear * gainprm[0] * ctrl`` either way, so the
    generator's ``gear`` form reproduces it exactly.
    """
    motors = generator.MODELS[RIGID].osl_motors
    for offset, (name, joint_name, gear) in enumerate(motors, start=len(EXPECTED_JOINTS)):
        assert mujoco.mj_id2name(rigid_model, mujoco.mjtObj.mjOBJ_ACTUATOR, offset) == name
        joint_id = mujoco.mj_name2id(rigid_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert rigid_model.actuator_trnid[offset, 0] == joint_id
        effective = (
            rigid_model.actuator_gear[offset, 0]
            * rigid_model.actuator_gainprm[offset, 0]
        )
        assert effective == pytest.approx(gear)
        np.testing.assert_allclose(rigid_model.actuator_ctrlrange[offset], (-2.88, 2.88))
    assert rigid_model.actuator_ctrlrange[len(EXPECTED_JOINTS), 1] == pytest.approx(2.88)


def test_fingerless_muscle_model_is_a_valid_sizing_basis():
    """Pin the equivalence that removes the analyzer's chicken-and-egg.

    A model being sized for the first time has no robot XML to measure ``I_eff``
    on, so ``sizing_basis`` falls back to the muscle model with fingers deleted.
    Measured, that reproduces the robot model's servo-free basis to 4.16e-10 --
    but only *with* the finger deletion: probing the muscle model as-is is up to
    35% low at the wrist, because free hand DoFs absorb acceleration that a welded
    hand does not.  If this drifts, every future table is silently wrong.
    """
    profile = generator._resolve(RIGID)
    robot = mujoco.MjModel.from_xml_path(str(profile.output_xml.resolve()))
    servos = analyzer.servo_actuators(robot)
    names = [name for _, name, _, _ in servos]
    from_robot = analyzer.constrained_inertia(
        robot, analyzer.project_joint_equalities(robot, robot.qpos0),
        [dof for _, _, _, dof in servos], strict=False,
    )

    basis, has_servos = analyzer.sizing_basis(
        generator.ModelProfile(
            source_xml=profile.source_xml,
            output_xml=profile.output_xml.with_name("__absent__.xml"),
            model_name=profile.model_name,
            finger_joints=profile.finger_joints,
            position_servos=profile.position_servos,
            osl_motors=profile.osl_motors,
            dimensions=profile.dimensions,
        )
    )
    assert has_servos is False
    dofs = [
        basis.jnt_dofadr[mujoco.mj_name2id(basis, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in names
    ]
    from_muscle = analyzer.constrained_inertia(
        basis, analyzer.project_joint_equalities(basis, basis.qpos0), dofs, strict=False
    )
    worst = float(np.max(np.abs(from_muscle - from_robot) / np.abs(from_robot)))
    assert worst < 1e-6, f"sizing bases diverged: {worst:.3e}"


def test_environment_registration_and_cpu_rollout():
    assert Mujoco.registered_envs["OSLFullBodyRobot"] is OSLFullBodyRobot
    assert Mujoco.registered_envs["MjxOSLFullBodyRobot"] is MjxOSLFullBodyRobot
    # validation_video_recorder strips a leading "Mjx" to find the CPU class.
    assert "MjxOSLFullBodyRobot".replace("Mjx", "", 1) == "OSLFullBodyRobot"
    # The site mapper and video recorder dispatch on the "OSLFullBody" substring.
    assert "OSLFullBody" in OSLFullBodyRobot.__name__


@pytest.mark.parametrize("compliant, dimensions", [(False, (80, 79)), (True, (84, 83))])
def test_both_socket_variants_construct_and_step(compliant, dimensions):
    env = OSLFullBodyRobot(compliant_socket=compliant, horizon=1100)
    model = env._model
    assert (model.nq, model.nv) == dimensions
    assert env.info.action_space.shape == (29,)
    assert len(env.sites_for_mimic) == 17
    env.reset()
    for _ in range(1000):
        env.step(np.zeros(env.info.action_space.shape))
    assert np.isfinite(env._data.qpos).all()


def test_robot_hydra_config_is_an_additive_variant():
    config_dir = str(Path(__file__).resolve().parents[2] / "fullbody")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base = compose(config_name="conf_osl_fullbody")
        robot = compose(config_name="conf_osl_fullbody_robot")

    assert robot.experiment.env_params.env_name == "MjxOSLFullBodyRobot"
    assert robot.experiment.ppo_config.init_std == pytest.approx(0.2)
    assert robot.experiment.ppo_config.init_std_motors == pytest.approx(0.2)
    for flag in (
        "enable_muscle_length_observations",
        "enable_muscle_velocity_observations",
        "enable_muscle_force_observations",
        "enable_muscle_excitation_observations",
        "enable_muscle_activation_observations",
    ):
        assert robot.experiment.env_params[flag] is False

    # The socket toggle and the trajectory come through inherited, unmodified.
    assert robot.experiment.env_params.compliant_socket is False
    assert robot.experiment.env_params.strict_trajectory_joints is True
    assert (
        robot.experiment.task_factory.params.amass_dataset_conf.traj_path
        == base.experiment.task_factory.params.amass_dataset_conf.traj_path
    )
    assert len(robot.experiment.env_params.goal_params.sites_for_mimic) == 17
    assert robot.experiment.env_params.reward_params == base.experiment.env_params.reward_params
