"""Tests for the flattened position-controlled OSL full-body robot."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from loco_mujoco.core.mujoco_base import Mujoco
from musclemimic.environments.humanoids import MjxOSLFullBodyRobot, OSLFullBodyRobot
from scripts import analyze_osl_fullbody_robot_actuators as analyzer
from scripts import generate_osl_fullbody_robot as generator


ROBOT_XML = generator.OUTPUT_XML
EXPECTED_JOINTS = [servo[0] for servo in generator.POSITION_SERVOS]
EXPECTED_ACTIONS = [
    *[f"{name}_position_actuator" for name in EXPECTED_JOINTS],
    *[motor[0] for motor in generator.OSL_MOTORS],
]


def _model(path: Path = ROBOT_XML) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path.resolve()))


def _names(model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, objtype, index) for index in range(count)]


@pytest.fixture(scope="module")
def robot_model() -> mujoco.MjModel:
    return _model()


@pytest.fixture(scope="module")
def fingerless_source_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(generator.SOURCE_XML.resolve()))
    generator._delete_finger_joints(spec)
    return spec.compile()


def test_generation_is_deterministic_standalone_and_relative(robot_model):
    subprocess.run(
        [sys.executable, str(Path(generator.__file__).resolve()), "--check"],
        cwd=generator.REPO_ROOT,
        check=True,
    )
    generated = ROBOT_XML.read_text(encoding="utf-8")

    root = ET.fromstring(generated)
    assert not any(
        os.path.isabs(element.get("file", ""))
        for element in root.iter()
        if element.get("file")
    )
    assert not any(
        "mimic" in (element.get("name") or "")
        for element in root.iter("site")
    )
    assert robot_model.nu == 29  # proves direct compilation resolved all assets
    assert np.all(robot_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)


def test_model_dimensions_mass_and_anatomical_parity(robot_model, fingerless_source_model):
    robot = robot_model
    source = fingerless_source_model
    assert (robot.nq, robot.nv, robot.njnt, robot.nbody, robot.neq, robot.nu, robot.na) == (
        84,
        83,
        78,
        104,
        44,
        29,
        0,
    )
    assert robot.ntendon == 8
    assert robot.body_mass.sum() == pytest.approx(82.4731, abs=1e-4)

    assert _names(robot, mujoco.mjtObj.mjOBJ_BODY, robot.nbody) == _names(
        source, mujoco.mjtObj.mjOBJ_BODY, source.nbody
    )
    assert _names(robot, mujoco.mjtObj.mjOBJ_JOINT, robot.njnt) == _names(
        source, mujoco.mjtObj.mjOBJ_JOINT, source.njnt
    )
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_axis",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "geom_bodyid",
        "geom_type",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "eq_type",
        "eq_obj1id",
        "eq_obj2id",
        "eq_data",
    ):
        np.testing.assert_allclose(
            getattr(robot, field), getattr(source, field), rtol=2e-6, atol=1e-6, err_msg=field
        )


def test_actuator_order_standard_servo_parameters_and_passive_coordinates(robot_model):
    model = robot_model
    assert _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu) == EXPECTED_ACTIONS

    for actuator_id, (joint_name, kp, kd, force_limit) in enumerate(generator.POSITION_SERVOS):
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
        # MuJoCo serializes to six significant figures, so a derived product such as
        # subtalar's Kp * half_range (106.325504 -> "106.326") carries up to ~5e-6
        # relative error.  That text format, not the arithmetic, sets the floor here.
        np.testing.assert_allclose(
            model.actuator_gainprm[actuator_id, 0], kp * half_range, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            model.actuator_biasprm[actuator_id, :3],
            (kp * midpoint, -kp, -kd),
            rtol=1e-5,
            atol=1e-6,
        )

    targeted_joint_ids = set(model.actuator_trnid[:, 0].tolist())
    for passive_name in (
        "socket_piston",
        "socket_rotation_x",
        "socket_rotation_y",
        "socket_rotation_z",
    ):
        passive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, passive_name)
        assert passive_id not in targeted_joint_ids

    equality_follower_ids = set(model.eq_obj1id[model.eq_objtype == mujoco.mjtObj.mjOBJ_JOINT])
    assert equality_follower_ids.isdisjoint(targeted_joint_ids)
    assert not np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_BALL)


def test_servo_gains_satisfy_the_sizing_rules_at_the_reference_pose(robot_model):
    """Pin the sizing *rules*, not the table-to-XML copy.

    ``test_actuator_order_...`` asserts XML <-> POSITION_SERVOS, so it stays green
    even if both were regenerated from a bad measurement.  This recomputes the
    rules from scratch against a freshly measured, servo-free and contact-free
    ``I_eff`` and fails loudly on a regeneration that used a contaminated probe --
    which showed up as zeta 0.61-0.72 on the shoulder rather than 1.0.

    Scope: this is a *reference-pose* invariant.  ``I_eff`` is pose-dependent, so
    zeta is only 1.0 at the pose the table was sized at; the stability margin
    below is likewise not a global worst case over the joint ranges.
    """
    servos = analyzer.servo_actuators(robot_model)
    reference = analyzer.project_joint_equalities(robot_model, robot_model.qpos0)
    # strict=True: a nonlinear probe here means the invariant cannot be checked.
    inertia = analyzer.constrained_inertia(
        robot_model, reference, [dof for _, _, _, dof in servos]
    )
    assert np.all(np.isfinite(inertia))

    table = {name: (kp, kd, limit) for name, kp, kd, limit in generator.POSITION_SERVOS}
    stability = {}
    for index, (_, joint_name, _, _) in enumerate(servos):
        kp, kd, force_limit = table[joint_name]
        moment_of_inertia = inertia[index]

        # Rule 2, with atol for the 1-decimal Kp stored in POSITION_SERVOS.
        expected_kp = min(force_limit / 0.35, moment_of_inertia * 150.0**2)
        assert kp == pytest.approx(expected_kp, abs=0.051), joint_name

        # Rule 3: critically damped.
        zeta = kd / (2.0 * math.sqrt(kp * moment_of_inertia))
        assert zeta == pytest.approx(1.0, abs=0.01), f"{joint_name} zeta={zeta:.3f}"

        # The training env runs at 2 ms, not the XML's 1 ms.
        stability[joint_name] = kd * 0.002 / moment_of_inertia

    worst = max(stability, key=stability.get)
    assert stability[worst] < 1.0, f"{worst} Kd*dt/I={stability[worst]:.3f}"

    # Bilateral arm pairs must match in the stored table.  Contact-free the raw
    # inertias differ in the 4th significant figure -- genuinely, since the arm
    # reacts against a body whose right leg is a prosthesis -- so compare the
    # rounded values that actually ship, not the measurement.
    for stem in ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flex",
                 "pro_sup", "deviation", "flexion"):
        assert table[f"{stem}_r"] == table[f"{stem}_l"], stem


def test_osl_motor_parameters_are_unchanged(robot_model):
    for offset, (name, joint_name, gear) in enumerate(generator.OSL_MOTORS, start=27):
        assert mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_ACTUATOR, offset) == name
        joint_id = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert robot_model.actuator_trnid[offset, 0] == joint_id
        assert robot_model.actuator_gear[offset, 0] == pytest.approx(gear)
        np.testing.assert_allclose(robot_model.actuator_ctrlrange[offset], (-2.88, 2.88))


def test_environment_registration_touch_sensors_mimic_sites_and_cpu_rollout():
    assert Mujoco.registered_envs["OSLFullBodyRobot"] is OSLFullBodyRobot
    assert Mujoco.registered_envs["MjxOSLFullBodyRobot"] is MjxOSLFullBodyRobot

    env = OSLFullBodyRobot(
        horizon=1100,
        n_substeps=1,
        enable_muscle_length_observations=False,
        enable_muscle_velocity_observations=False,
        enable_muscle_force_observations=False,
        enable_muscle_excitation_observations=False,
        enable_muscle_activation_observations=False,
    )
    assert env.info.action_space.shape == (29,)
    for sensor_name in ("r_osl_foot", "l_foot", "l_toes"):
        assert mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0
    for site_name in env.sites_for_mimic:
        assert mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, site_name) >= 0

    observation = env.reset()
    assert np.isfinite(observation).all()
    action = np.zeros(29)
    for _ in range(1000):
        observation, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    assert np.isfinite(env._data.qpos).all()
    assert np.isfinite(env._data.qvel).all()


@pytest.mark.integration
def test_mjx_jax_zero_action_reset_step_is_finite():
    env = MjxOSLFullBodyRobot(
        horizon=2,
        n_substeps=1,
        mjx_backend="jax",
        enable_muscle_length_observations=False,
        enable_muscle_velocity_observations=False,
        enable_muscle_force_observations=False,
        enable_muscle_excitation_observations=False,
        enable_muscle_activation_observations=False,
    )
    state = env.mjx_reset(jax.random.PRNGKey(0))
    state = env.mjx_step(state, jnp.zeros(29))
    assert np.isfinite(np.asarray(state.observation)).all()
    assert np.isfinite(np.asarray(state.data.qpos)).all()


def test_robot_hydra_config_is_an_additive_osl_variant():
    config_dir = Path(__file__).resolve().parents[2] / "fullbody"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(config_name="conf_osl_fullbody")
        robot = compose(config_name="conf_osl_fullbody_robot")

    assert robot.experiment.env_params.env_name == "MjxOSLFullBodyRobot"
    for key in (
        "enable_muscle_length_observations",
        "enable_muscle_velocity_observations",
        "enable_muscle_force_observations",
        "enable_muscle_excitation_observations",
        "enable_muscle_activation_observations",
    ):
        assert robot.experiment.env_params[key] is False
    assert robot.experiment.env_params.enable_touch_sensor_observations is True
    assert robot.experiment.env_params.control_type == "DefaultControl"
    assert robot.experiment.ppo_config.init_std == pytest.approx(0.2)
    assert robot.experiment.ppo_config.init_std_motors == pytest.approx(0.2)
    assert len(robot.experiment.env_params.goal_params.sites_for_mimic) == 17
    assert robot.experiment.task_factory == base.experiment.task_factory
    assert robot.experiment.env_params.reward_params == base.experiment.env_params.reward_params
    assert robot.experiment.validation == base.experiment.validation
