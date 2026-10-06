"""Tests for the NewOSLFullBody environment (the official MyoAssist OSL model).

Part 2 covers registration, the socket/Euler-damping toggle, the corrected finger
lists, and the trajectory name-coverage gate.  Part 3 adds the regenerated-clip
compatibility tests below the Part 2 block.  Nothing here trains -- envs are constructed
directly, as in test_osl_fullbody_robot.py.
"""

from __future__ import annotations

import os
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
from musclemimic.core.utils.site_mapping import create_site_mapper
from musclemimic.environments.humanoids import MjxNewOSLFullBody, NewOSLFullBody
from musclemimic.environments.humanoids import new_osl_fullbody as env_module


# The clip predates the official export and lives outside this repo.
_LEGACY_TRAJ = Path(
    "/home/dmma/src/00_AMBER/00_projbackflip/clipper/outputs/musclemimic/osl_fullbody/"
    "WalkingStraightForwards03_poses_crop0-521_ankle-7.npz"
)
# The four joints the official export renamed, and the only ones the gate should name.
_RENAMED_JOINTS = {
    "elbow_flexion_r",
    "elbow_flexion_l",
    "OSL_KA_L1_osl_knee_angle_r",
    "OSL_KA_L1_osl_ankle_angle_r",
}
# The clip clipper regenerated against models/new_osl_fullbody (Part 3).  Also outside
# this repo, so every test that uses it skips when it is absent.
_NEW_TRAJ = Path(
    "/home/dmma/src/00_AMBER/00_projbackflip/clipper/outputs/musclemimic/new_osl_fullbody/"
    "WalkingStraightForwards03_poses_crop0-521_z-0.025_ankle-7.npz"
)
_MUSCLE_FREE_KWARGS = dict(
    enable_muscle_length_observations=False,
    enable_muscle_velocity_observations=False,
    enable_muscle_force_observations=False,
    enable_muscle_excitation_observations=False,
    enable_muscle_activation_observations=False,
)


def _names(model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, objtype, index) for index in range(count)]


@pytest.fixture(scope="module")
def rigid_env() -> NewOSLFullBody:
    return NewOSLFullBody(horizon=1100, n_substeps=1, **_MUSCLE_FREE_KWARGS)


@pytest.fixture(scope="module")
def compliant_env() -> NewOSLFullBody:
    return NewOSLFullBody(
        compliant_socket=True, horizon=1100, n_substeps=1, **_MUSCLE_FREE_KWARGS
    )


def test_environments_are_registered():
    assert Mujoco.registered_envs["NewOSLFullBody"] is NewOSLFullBody
    assert Mujoco.registered_envs["MjxNewOSLFullBody"] is MjxNewOSLFullBody
    # validation_video_recorder.py strips a leading "Mjx" to find the CPU class.
    assert "MjxNewOSLFullBody".replace("Mjx", "", 1) == "NewOSLFullBody"


@pytest.mark.parametrize(
    "fixture_name, dimensions",
    [("rigid_env", (80, 79, 341)), ("compliant_env", (84, 83, 341))],
)
def test_model_dimensions_sites_and_sensors(request, fixture_name, dimensions):
    env = request.getfixturevalue(fixture_name)
    model = env._model
    assert (model.nq, model.nv, model.nu) == dimensions
    assert env.info.action_space.shape == (dimensions[2],)

    assert len(env.sites_for_mimic) == 17
    for site_name in env.sites_for_mimic:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) >= 0

    # The official export ships the parent's four touch sensors, so unlike OSLFullBody
    # this class needs no _get_observation_specification override.
    for sensor_name in ("r_foot", "r_toes", "l_foot", "l_toes"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0

    actuators = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    motors = [
        name
        for name, dyntype in zip(actuators, model.actuator_dyntype)
        if dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
    ]
    assert motors == ["osl_knee_torque_actuator", "osl_ankle_torque_actuator"]


def test_mimic_map_points_at_the_official_prosthetic_bodies(rigid_env):
    mimic = rigid_env.body2sites_for_mimic
    assert len(mimic) == 17
    assert mimic["OSL_KA_L1_osl_knee_assembly"] == "right_knee_mimic"
    assert mimic["OSL_KA_L1_osl_ankle_assembly"] == "right_ankle_mimic"
    assert mimic["OSL_KA_L1_osl_foot_assembly"] == "right_toes_mimic"
    for body_name in mimic:
        assert mujoco.mj_name2id(rigid_env._model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0


def test_disable_fingers_removes_every_renamed_finger(rigid_env):
    """Regression guard for the parent's stale `_r` / `_left` names.

    MyoFullBody's exact-match lists hit only 36/40 joints and 31/62 muscles on this
    model; the corrected lists in the env module hit all of them.
    """
    model = rigid_env._model
    joints = set(_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt))
    actuators = set(_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu))
    assert len(env_module._FINGER_JOINTS) == 40
    assert len(env_module._FINGER_MUSCLES) == 62
    assert not joints & set(env_module._FINGER_JOINTS)
    assert not actuators & set(env_module._FINGER_MUSCLES)

    raw = mujoco.MjModel.from_xml_path(str(env_module._RIGID_XML.resolve()))
    assert model.nu == raw.nu - len(env_module._FINGER_MUSCLES)
    assert model.njnt == raw.njnt - len(env_module._FINGER_JOINTS)

    # Keyframes must be stripped: deleting a joint from a spec that has them fails to
    # compile ("keyframe 'stand': invalid qpos size").
    assert raw.nkey == 5
    assert model.nkey == 0


def test_socket_toggle_drives_euler_damping():
    """The two flags are one coupled toggle -- see the module docstring."""
    eulerdamp = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)

    rigid = NewOSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    assert int(rigid._model.opt.disableflags) & eulerdamp

    compliant = NewOSLFullBody(compliant_socket=True, horizon=10, **_MUSCLE_FREE_KWARGS)
    assert not int(compliant._model.opt.disableflags) & eulerdamp
    socket_joints = {"socket_piston", "socket_rotation_1", "socket_rotation_2", "socket_rotation_3"}
    assert socket_joints <= set(
        _names(compliant._model, mujoco.mjtObj.mjOBJ_JOINT, compliant._model.njnt)
    )

    # Explicit override in the safe direction is allowed.
    override = NewOSLFullBody(disable_eulerdamp=False, horizon=10, **_MUSCLE_FREE_KWARGS)
    assert not int(override._model.opt.disableflags) & eulerdamp

    with pytest.raises(ValueError, match="numerically unsafe"):
        NewOSLFullBody(compliant_socket=True, disable_eulerdamp=True, **_MUSCLE_FREE_KWARGS)


@pytest.mark.parametrize("fixture_name", ["rigid_env", "compliant_env"])
def test_cpu_rollout_is_finite(request, fixture_name):
    env = request.getfixturevalue(fixture_name)
    observation = env.reset()
    assert np.isfinite(observation).all()
    action = np.zeros(env.info.action_space.shape)
    for _ in range(1000):
        observation, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
    assert np.isfinite(env._data.qpos).all()
    assert np.isfinite(env._data.qvel).all()


def test_site_mapper_dispatch_survives_the_class_name():
    """`create_site_mapper` matches substrings and tests "MyoFullBody" first."""
    model = mujoco.MjModel.from_xml_path(str(env_module._RIGID_XML.resolve()))
    for class_name in ("NewOSLFullBody", "MjxNewOSLFullBody"):
        mapper = create_site_mapper(model, class_name, ["pelvis_mimic"], ["pelvis_mimic"])
        assert mapper.requires_mapping, class_name


@pytest.mark.skipif(not _LEGACY_TRAJ.is_file(), reason="the clip lives outside this repo")
@pytest.mark.parametrize("fixture_name", ["rigid_env", "compliant_env"])
def test_legacy_trajectory_is_rejected_by_name(request, fixture_name):
    """filter_and_extend zero-pads missing model joints silently; the gate must not."""
    env = request.getfixturevalue(fixture_name)
    with pytest.raises(RuntimeError) as excinfo:
        env.load_trajectory(traj_path=str(_LEGACY_TRAJ), warn=False)
    message = str(excinfo.value)
    assert "does not cover 4 joint(s)" in message
    for joint_name in _RENAMED_JOINTS:
        assert joint_name in message
    assert "strict_trajectory_joints=False" in message


def test_coverage_gate_can_be_opted_out(rigid_env):
    lenient = NewOSLFullBody(strict_trajectory_joints=False, horizon=10, **_MUSCLE_FREE_KWARGS)
    assert lenient._strict_trajectory_joints is False
    # A trajectory naming every model joint passes the strict gate.
    covering = _names(rigid_env._model, mujoco.mjtObj.mjOBJ_JOINT, rigid_env._model.njnt)
    rigid_env._assert_trajectory_covers_model_joints(covering)


@pytest.mark.skipif(not _LEGACY_TRAJ.is_file(), reason="the clip lives outside this repo")
def test_padded_trajectory_stays_on_one_backend():
    """Regression guard for the padding path's mixed-backend trajectory.

    filter_and_extend pads a missing model joint via TrajectoryData.add_joint(), whose
    backend defaults to jnp, so qpos becomes a jax array while split_points stays numpy.
    TrajectoryHandler reads is_numpy off qpos alone, so engine.instantiate_env's to_jax()
    would no-op and mjx_reset would die with TracerArrayConversionError.
    """
    import jax.numpy as jnp

    env = NewOSLFullBody(strict_trajectory_joints=False, horizon=10, **_MUSCLE_FREE_KWARGS)
    env.load_trajectory(traj_path=str(_LEGACY_TRAJ), warn=False)
    assert env.th.is_numpy, "load_trajectory must renormalize onto one backend"

    env.th.to_jax()
    data = env.th.traj.data
    for field in ("qpos", "qvel", "split_points"):
        assert isinstance(getattr(data, field), jnp.ndarray), field


def test_hydra_configs_target_the_new_model():
    config_dir = Path(__file__).resolve().parents[2] / "fullbody"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(config_name="conf_new_osl_fullbody")
        clf = compose(config_name="conf_new_osl_fullbody_clf")
        legacy = compose(config_name="conf_osl_fullbody")

    for config in (base, clf):
        params = config.experiment.env_params
        assert params.env_name == "MjxNewOSLFullBody"
        assert params.compliant_socket is False
        assert params.strict_trajectory_joints is True
        assert params.disable_fingers is True
        assert set(params.reward_params.joint_torque_weights) == {
            "OSL_KA_L1_osl_knee_angle_r",
            "OSL_KA_L1_osl_ankle_angle_r",
        }
        assert len(params.goal_params.sites_for_mimic) == 17
        assert len(config.experiment.validation.rel_site_names) == 17

    # The 17 mimic site names are unchanged from the legacy config -- only bodies moved.
    assert (
        base.experiment.env_params.goal_params.sites_for_mimic
        == legacy.experiment.env_params.goal_params.sites_for_mimic
    )
    # The legacy config is untouched.
    assert legacy.experiment.env_params.env_name == "MjxOSLFullBody"

    assert clf.experiment.env_params.reward_type == "CLFReward"
    for weight in (
        "qpos_w_sum",
        "qvel_w_sum",
        "root_pos_w_sum",
        "root_vel_w_sum",
        "rpos_w_sum",
        "rquat_w_sum",
        "rvel_w_sum",
    ):
        assert clf.experiment.env_params.reward_params[weight] == pytest.approx(0.0)
    assert clf.experiment.reward_curriculum.enabled is False


@pytest.mark.integration
@pytest.mark.parametrize("compliant", [False, True])
def test_mjx_reset_and_step(compliant):
    """Exercises the _modify_spec_for_mjx touch-sensor workaround on the jax backend."""
    import jax

    env = MjxNewOSLFullBody(
        compliant_socket=compliant, horizon=2, mjx_backend="jax", **_MUSCLE_FREE_KWARGS
    )
    key = jax.random.PRNGKey(0)
    state = env.mjx_reset(key)
    assert np.isfinite(np.asarray(state.observation)).all()
    action = jax.numpy.zeros(env.info.action_space.shape)
    state = env.mjx_step(state, action)
    assert np.isfinite(np.asarray(state.observation)).all()


# ---------------------------------------------------------------------------------------
# Part 3 -- the regenerated clip
# ---------------------------------------------------------------------------------------

_needs_new_traj = pytest.mark.skipif(
    not _NEW_TRAJ.is_file(), reason="the regenerated clip lives outside this repo"
)


def _load_new_traj(env: NewOSLFullBody) -> None:
    """Load the regenerated clip the way ImitationFactory does (sparse + mimic sites)."""
    env.load_trajectory(
        traj_path=str(_NEW_TRAJ),
        warn=False,
        cache_type="sparse",
        site_names=env.sites_for_mimic,
    )


@_needs_new_traj
def test_new_trajectory_passes_the_gate_on_the_rigid_model():
    env = NewOSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    _load_new_traj(env)  # must not raise
    assert np.asarray(env.th.traj.data.qpos).shape[1] == env._model.nq
    model_joints = _names(env._model, mujoco.mjtObj.mjOBJ_JOINT, env._model.njnt)
    # Reconciled order is model order; the clip already ships in that order.
    assert list(env.th.traj.info.joint_names) == model_joints


@_needs_new_traj
def test_new_trajectory_joint_types_match_the_model():
    """filter_and_extend reconciles joints by NAME and takes jnt_type from the npz.

    Nothing in loco_mujoco cross-checks the two, so a type disagreement on a shared
    name survives load and only surfaces later as an opaque shape error when qpos is
    written into MjData.  Pin it here instead.
    """
    env = NewOSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    payload = np.load(_NEW_TRAJ, allow_pickle=True)
    model_joints = _names(env._model, mujoco.mjtObj.mjOBJ_JOINT, env._model.njnt)
    clip_types = dict(zip((str(n) for n in payload["joint_names"]), payload["jnt_type"]))
    for index, name in enumerate(model_joints):
        assert clip_types[name] == env._model.jnt_type[index], name


@_needs_new_traj
def test_new_trajectory_was_generated_against_this_model():
    """Forward kinematics here must reproduce the clip's own recorded site_xpos.

    Under the sparse cache every site position is recomputed on the compiled model, so
    the reference is defined jointly by the clip's qpos and this model's geometry.  If
    clipper's vendored copy of models/new_osl_fullbody ever drifts, nothing raises --
    the reference just goes quietly wrong.  This is that guard.  See the model README.
    """
    env = NewOSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    payload = np.load(_NEW_TRAJ, allow_pickle=True)
    model, data = env._model, mujoco.MjData(env._model)
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(name))
        for name in payload["site_names"]
    ]
    assert all(site_id >= 0 for site_id in site_ids)
    worst = 0.0
    for frame in range(0, len(payload["qpos"]), 17):
        data.qpos[:] = payload["qpos"][frame]
        mujoco.mj_forward(model, data)
        worst = max(worst, np.abs(data.site_xpos[site_ids] - payload["site_xpos"][frame]).max())
    # float32 round-trip is ~1.2e-07 m; 1e-6 leaves headroom without admitting real drift.
    assert worst < 1e-6, f"clip/model mismatch: max |d| = {worst:.3e} m"


@_needs_new_traj
def test_new_trajectory_sites_match_sites_for_mimic_in_order():
    env = NewOSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    payload = np.load(_NEW_TRAJ, allow_pickle=True)
    assert [str(name) for name in payload["site_names"]] == list(env.sites_for_mimic)


@_needs_new_traj
def test_compliant_socket_accepts_the_rigid_clip_and_pads_sockets_to_neutral():
    """A neutral socket IS the rigid configuration, so padding is the correct reference.

    Clipper generates from the rigid model and never names these four DoFs, so this is
    the default path for every compliant-socket run -- not an edge case.
    """
    env = NewOSLFullBody(compliant_socket=True, horizon=10, **_MUSCLE_FREE_KWARGS)
    _load_new_traj(env)  # must not raise despite the four absent joints
    model = env._model
    qpos = np.asarray(env.th.traj.data.qpos)
    assert qpos.shape[1] == model.nq
    for name in env_module._SOCKET_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, name
        assert (qpos[:, model.jnt_qposadr[joint_id]] == 0.0).all(), name
    # The padding path mixes jnp/np backends; _normalize_trajectory_backend must undo it.
    assert env.th.is_numpy


@_needs_new_traj
def test_socket_exemption_does_not_excuse_other_joints():
    """The exemption must name exactly the socket DoFs, not soften the gate generally."""
    env = NewOSLFullBody(compliant_socket=True, horizon=10, **_MUSCLE_FREE_KWARGS)
    covering = _names(env._model, mujoco.mjtObj.mjOBJ_JOINT, env._model.njnt)
    dropped = "elbow_flexion_l"
    assert dropped in covering
    with pytest.raises(RuntimeError) as excinfo:
        env._assert_trajectory_covers_model_joints(
            [name for name in covering if name != dropped]
        )
    assert dropped in str(excinfo.value)


def test_socket_joints_are_not_exempt_on_the_rigid_model(rigid_env):
    """On the rigid model these joints do not exist, so there is nothing to exempt."""
    model_joints = _names(rigid_env._model, mujoco.mjtObj.mjOBJ_JOINT, rigid_env._model.njnt)
    assert not set(env_module._SOCKET_JOINTS) & set(model_joints)


@_needs_new_traj
def test_both_configs_point_at_the_regenerated_clip():
    """conf_new_osl_fullbody_clf inherits the training block, so one path drives both."""
    config_dir = str(Path(__file__).resolve().parents[2] / "fullbody")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for name in ("conf_new_osl_fullbody", "conf_new_osl_fullbody_clf"):
            config = compose(config_name=name)
            training = config.experiment.task_factory.params.amass_dataset_conf.traj_path
            validation = config.experiment.validation.amass_dataset_conf.traj_path
            assert [str(p) for p in training] == [str(_NEW_TRAJ)], name
            assert [str(p) for p in validation] == [str(_NEW_TRAJ)], name
