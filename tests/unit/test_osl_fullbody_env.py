"""Tests for the OSLFullBody environment (the MyoAssist full-body + OSL model, v3).

The first block covers registration, the socket/Euler-damping toggle, the corrected
finger lists, and the trajectory name-coverage gate.  The second checks the clipper
reference clip against this model.  Nothing here trains -- envs are constructed
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
from musclemimic.environments.humanoids import MjxOSLFullBody, OSLFullBody
from musclemimic.environments.humanoids import osl_fullbody as env_module


# The clip clipper generated against models/osl_fullbody.  It lives outside this repo,
# so every test that uses it skips when it is absent.
_TRAJ = Path(
    "/home/dmma/src/00_AMBER/00_projbackflip/clipper/outputs/musclemimic/osl_fullbody/"
    "WalkingStraightForwards03_poses_crop0-521_z-0.0016_ankle-7_bioankle-5.npz"
)
# The official export's names for four joints the hand-assembled model (old_osl branch)
# called elbow_flex_{r,l} and osl_{knee,ankle}_angle_r.  A clip made for that model is
# the realistic way to reach the gate and the padding path.
_RENAMED_JOINTS = {
    "elbow_flexion_r": "elbow_flex_r",
    "elbow_flexion_l": "elbow_flex_l",
    "OSL_KA_L1_osl_knee_angle_r": "osl_knee_angle_r",
    "OSL_KA_L1_osl_ankle_angle_r": "osl_ankle_angle_r",
}
_needs_traj = pytest.mark.skipif(
    not _TRAJ.is_file(), reason="the clipper clip lives outside this repo"
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


@pytest.fixture
def legacy_named_traj(tmp_path) -> Path:
    """The reference clip with four joints renamed to the hand-assembled model's names."""
    with np.load(_TRAJ, allow_pickle=True) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["joint_names"] = np.array(
        [_RENAMED_JOINTS.get(str(name), str(name)) for name in payload["joint_names"]]
    )
    path = tmp_path / "legacy_named.npz"
    np.savez(path, **payload)
    return path


@pytest.fixture(scope="module")
def rigid_env() -> OSLFullBody:
    return OSLFullBody(horizon=1100, n_substeps=1, **_MUSCLE_FREE_KWARGS)


@pytest.fixture(scope="module")
def compliant_env() -> OSLFullBody:
    return OSLFullBody(
        compliant_socket=True, horizon=1100, n_substeps=1, **_MUSCLE_FREE_KWARGS
    )


def test_environments_are_registered():
    assert Mujoco.registered_envs["OSLFullBody"] is OSLFullBody
    assert Mujoco.registered_envs["MjxOSLFullBody"] is MjxOSLFullBody
    # validation_video_recorder.py strips a leading "Mjx" to find the CPU class.
    assert "MjxOSLFullBody".replace("Mjx", "", 1) == "OSLFullBody"


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

    rigid = OSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    assert int(rigid._model.opt.disableflags) & eulerdamp

    compliant = OSLFullBody(compliant_socket=True, horizon=10, **_MUSCLE_FREE_KWARGS)
    assert not int(compliant._model.opt.disableflags) & eulerdamp
    socket_joints = {"socket_piston", "socket_rotation_1", "socket_rotation_2", "socket_rotation_3"}
    assert socket_joints <= set(
        _names(compliant._model, mujoco.mjtObj.mjOBJ_JOINT, compliant._model.njnt)
    )

    # Explicit override in the safe direction is allowed.
    override = OSLFullBody(disable_eulerdamp=False, horizon=10, **_MUSCLE_FREE_KWARGS)
    assert not int(override._model.opt.disableflags) & eulerdamp

    with pytest.raises(ValueError, match="numerically unsafe"):
        OSLFullBody(compliant_socket=True, disable_eulerdamp=True, **_MUSCLE_FREE_KWARGS)


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
    for class_name in ("OSLFullBody", "MjxOSLFullBody"):
        mapper = create_site_mapper(model, class_name, ["pelvis_mimic"], ["pelvis_mimic"])
        assert mapper.requires_mapping, class_name


@_needs_traj
@pytest.mark.parametrize("fixture_name", ["rigid_env", "compliant_env"])
def test_legacy_trajectory_is_rejected_by_name(request, fixture_name, legacy_named_traj):
    """filter_and_extend zero-pads missing model joints silently; the gate must not."""
    env = request.getfixturevalue(fixture_name)
    with pytest.raises(RuntimeError) as excinfo:
        env.load_trajectory(traj_path=str(legacy_named_traj), warn=False)
    message = str(excinfo.value)
    assert "does not cover 4 joint(s)" in message
    for joint_name in _RENAMED_JOINTS:
        assert joint_name in message
    assert "strict_trajectory_joints=False" in message


def test_coverage_gate_can_be_opted_out(rigid_env):
    lenient = OSLFullBody(strict_trajectory_joints=False, horizon=10, **_MUSCLE_FREE_KWARGS)
    assert lenient._strict_trajectory_joints is False
    # A trajectory naming every model joint passes the strict gate.
    covering = _names(rigid_env._model, mujoco.mjtObj.mjOBJ_JOINT, rigid_env._model.njnt)
    rigid_env._assert_trajectory_covers_model_joints(covering)


def test_load_trajectory_override_keeps_the_base_signature():
    """ImitationFactory.make calls load_trajectory with every base keyword (motion_names
    included), so an override that drops one fails at env construction."""
    import inspect

    from musclemimic.environments.base import LocoEnv

    base = inspect.signature(LocoEnv.load_trajectory).parameters
    override = inspect.signature(OSLFullBody.load_trajectory).parameters
    assert list(override) == list(base)


@_needs_traj
def test_padded_trajectory_stays_on_one_backend(legacy_named_traj):
    """Regression guard for the padding path's mixed-backend trajectory.

    filter_and_extend pads a missing model joint via TrajectoryData.add_joint(), whose
    backend defaults to jnp, so qpos becomes a jax array while split_points stays numpy.
    TrajectoryHandler reads is_numpy off qpos alone, so engine.instantiate_env's to_jax()
    would no-op and mjx_reset would die with TracerArrayConversionError.
    """
    import jax.numpy as jnp

    env = OSLFullBody(strict_trajectory_joints=False, horizon=10, **_MUSCLE_FREE_KWARGS)
    env.load_trajectory(traj_path=str(legacy_named_traj), warn=False)
    assert env.th.is_numpy, "load_trajectory must renormalize onto one backend"

    env.th.to_jax()
    data = env.th.traj.data
    for field in ("qpos", "qvel", "split_points"):
        assert isinstance(getattr(data, field), jnp.ndarray), field


def test_hydra_configs_target_the_model(rigid_env):
    config_dir = Path(__file__).resolve().parents[2] / "fullbody"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(config_name="conf_osl_fullbody")
        clf = compose(config_name="conf_osl_fullbody_clf")

    for config in (base, clf):
        params = config.experiment.env_params
        assert params.env_name == "MjxOSLFullBody"
        assert params.compliant_socket is False
        assert params.strict_trajectory_joints is True
        assert params.disable_fingers is True
        assert set(params.reward_params.joint_torque_weights) == {
            "OSL_KA_L1_osl_knee_angle_r",
            "OSL_KA_L1_osl_ankle_angle_r",
        }
        assert len(params.goal_params.sites_for_mimic) == 17
        assert len(config.experiment.validation.rel_site_names) == 17

    assert list(base.experiment.env_params.goal_params.sites_for_mimic) == list(
        rigid_env.sites_for_mimic
    )

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

    env = MjxOSLFullBody(
        compliant_socket=compliant, horizon=2, mjx_backend="jax", **_MUSCLE_FREE_KWARGS
    )
    key = jax.random.PRNGKey(0)
    state = env.mjx_reset(key)
    assert np.isfinite(np.asarray(state.observation)).all()
    action = jax.numpy.zeros(env.info.action_space.shape)
    state = env.mjx_step(state, action)
    assert np.isfinite(np.asarray(state.observation)).all()


# ---------------------------------------------------------------------------------------
# The clipper reference clip
# ---------------------------------------------------------------------------------------


def _load_traj(env: OSLFullBody) -> None:
    """Load the reference clip the way ImitationFactory does (sparse + mimic sites)."""
    env.load_trajectory(
        traj_path=str(_TRAJ),
        warn=False,
        cache_type="sparse",
        site_names=env.sites_for_mimic,
    )


@_needs_traj
def test_trajectory_passes_the_gate_on_the_rigid_model():
    env = OSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    _load_traj(env)  # must not raise
    assert np.asarray(env.th.traj.data.qpos).shape[1] == env._model.nq
    model_joints = _names(env._model, mujoco.mjtObj.mjOBJ_JOINT, env._model.njnt)
    # Reconciled order is model order; the clip already ships in that order.
    assert list(env.th.traj.info.joint_names) == model_joints


@_needs_traj
def test_trajectory_joint_types_match_the_model():
    """filter_and_extend reconciles joints by NAME and takes jnt_type from the npz.

    Nothing in loco_mujoco cross-checks the two, so a type disagreement on a shared
    name survives load and only surfaces later as an opaque shape error when qpos is
    written into MjData.  Pin it here instead.
    """
    env = OSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    payload = np.load(_TRAJ, allow_pickle=True)
    model_joints = _names(env._model, mujoco.mjtObj.mjOBJ_JOINT, env._model.njnt)
    clip_types = dict(zip((str(n) for n in payload["joint_names"]), payload["jnt_type"]))
    for index, name in enumerate(model_joints):
        assert clip_types[name] == env._model.jnt_type[index], name


@_needs_traj
def test_trajectory_was_generated_against_this_model():
    """Forward kinematics here must reproduce the clip's own recorded site_xpos.

    Under the sparse cache every site position is recomputed on the compiled model, so
    the reference is defined jointly by the clip's qpos and this model's geometry.  If
    clipper's vendored copy of models/osl_fullbody ever drifts, nothing raises --
    the reference just goes quietly wrong.  This is that guard.  See the model README.
    """
    env = OSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    payload = np.load(_TRAJ, allow_pickle=True)
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


@_needs_traj
def test_trajectory_body_frames_match_this_model():
    """The clip records the body_pos it was generated with; compare it to ours directly.

    A cheaper and more specific detector than the FK test above for the one edit most
    likely to drift between the two vendored copies: the hand-lengthened OSL tibial
    pylon (``OSL_KA_L1_osl_ankle_assembly`` sits 22.5 mm lower than in the stock export).
    """
    model = mujoco.MjModel.from_xml_path(str(env_module._RIGID_XML.resolve()))
    payload = np.load(_TRAJ, allow_pickle=True)
    clip_names = [str(name) for name in payload["body_names"]]
    assert clip_names == _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    # float32 storage; 1e-6 m is well below the 22.5 mm edit this guards.
    np.testing.assert_allclose(payload["body_pos"], model.body_pos, atol=1e-6)
    ankle = clip_names.index("OSL_KA_L1_osl_ankle_assembly")
    assert payload["body_pos"][ankle][1] == pytest.approx(-0.1786, abs=1e-6)


@_needs_traj
def test_trajectory_sites_match_sites_for_mimic_in_order():
    env = OSLFullBody(horizon=10, **_MUSCLE_FREE_KWARGS)
    payload = np.load(_TRAJ, allow_pickle=True)
    assert [str(name) for name in payload["site_names"]] == list(env.sites_for_mimic)


@_needs_traj
def test_compliant_socket_accepts_the_rigid_clip_and_pads_sockets_to_neutral():
    """A neutral socket IS the rigid configuration, so padding is the correct reference.

    Clipper generates from the rigid model and never names these four DoFs, so this is
    the default path for every compliant-socket run -- not an edge case.
    """
    env = OSLFullBody(compliant_socket=True, horizon=10, **_MUSCLE_FREE_KWARGS)
    _load_traj(env)  # must not raise despite the four absent joints
    model = env._model
    qpos = np.asarray(env.th.traj.data.qpos)
    assert qpos.shape[1] == model.nq
    for name in env_module._SOCKET_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, name
        assert (qpos[:, model.jnt_qposadr[joint_id]] == 0.0).all(), name
    # The padding path mixes jnp/np backends; _normalize_trajectory_backend must undo it.
    assert env.th.is_numpy


@_needs_traj
def test_socket_exemption_does_not_excuse_other_joints():
    """The exemption must name exactly the socket DoFs, not soften the gate generally."""
    env = OSLFullBody(compliant_socket=True, horizon=10, **_MUSCLE_FREE_KWARGS)
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


@_needs_traj
def test_configs_point_at_the_reference_clip():
    """The CLF and robot configs inherit the training block, so one path drives all three."""
    config_dir = str(Path(__file__).resolve().parents[2] / "fullbody")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for name in ("conf_osl_fullbody", "conf_osl_fullbody_clf", "conf_osl_fullbody_robot"):
            config = compose(config_name=name)
            training = config.experiment.task_factory.params.amass_dataset_conf.traj_path
            validation = config.experiment.validation.amass_dataset_conf.traj_path
            assert [str(p) for p in training] == [str(_TRAJ)], name
            assert [str(p) for p in validation] == [str(_TRAJ)], name
