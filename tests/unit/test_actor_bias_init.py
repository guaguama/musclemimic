"""Neutral servo initialization, actor isolation, and standing regression."""

from types import SimpleNamespace

import flax.serialization
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from omegaconf import OmegaConf

from loco_mujoco.core.control_functions.default import DefaultControl
from musclemimic.algorithms.common.moe_networks import SoftMoEActorCritic
from musclemimic.algorithms.common.networks import ActorCritic, actor_output_bias_initializer
from musclemimic.algorithms.ppo.config import get_ppo_config
from musclemimic.algorithms.ppo.ppo import PPOJax
from musclemimic.environments.humanoids import (
    MyoBimanualArm,
    MyoFullBody,
    MyoLeg80_OSL_KA,
    NewOSLFullBody,
    NewOSLFullBodyRobot,
    OSLFullBody,
    OSLFullBodyRobot,
)


def resolve(env, mode="qpos0", dim=None):
    return PPOJax._resolve_actor_bias_init(
        env, {"actor_bias_init": mode}, len(env._action_indices) if dim is None else dim
    )


@pytest.fixture(scope="module")
def robots():
    return [OSLFullBodyRobot(), NewOSLFullBodyRobot(), NewOSLFullBodyRobot(compliant_socket=True)]


def test_robot_vectors(robots):
    vectors = []
    expected = {
        "flex_extension": 0.283391,
        "lat_bending": 0,
        "axial_rotation": 0,
        "elv_angle": -0.155590,
        "shoulder_elv": -0.95,
        "shoulder_rot": -0.142701,
        "elbow_flex": -0.95,
        "elbow_flexion": -0.95,
        "pro_sup": 0,
        "deviation": -3 / 7,
        "flexion": 0,
        "hip_flexion": -0.6,
        "hip_adduction": 0.25,
        "hip_rotation": 0,
        "knee_angle": -0.95,
        "ankle_angle": 1 / 7,
        "subtalar_angle": 0,
        "mtp_angle": 0,
    }
    for env in robots:
        bias = np.asarray(resolve(env))
        assert bias.shape == (29,)
        assert np.count_nonzero(bias) == 17
        np.testing.assert_array_equal(bias[-2:], 0)
        for i, a in enumerate(env._action_indices[:-2]):
            joint = env._model.joint(env._model.actuator_trnid[a, 0]).name
            stem = joint.removesuffix("_r").removesuffix("_l")
            assert bias[i] == pytest.approx(expected[stem], abs=3e-6), joint
        vectors.append(bias)
    np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(vectors[1], vectors[2])


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (MyoFullBody, {}),
        (MyoBimanualArm, {}),
        (MyoLeg80_OSL_KA, {}),
        (OSLFullBody, {}),
        (NewOSLFullBody, {}),
        (NewOSLFullBody, {"compliant_socket": True}),
    ],
)
def test_muscle_environments_unchanged(cls, kwargs):
    env = cls(**kwargs)
    muscle = env._model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    assert muscle.any()
    # Construct the runtime environment so spec ctrlrange rewrites are included.
    np.testing.assert_array_equal(env._model.actuator_ctrlrange[muscle], np.tile([-1, 1], (muscle.sum(), 1)))
    assert resolve(env) is None


def synthetic():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body><joint name="j" type="slide" ref="0.2"/>
    <geom size="0.1" mass="1"/></body></worldbody><actuator>
    <general joint="j" gear="2" gainprm="4" biastype="affine"
    biasprm="1 -10 -2" ctrlrange="-2 4"/></actuator></mujoco>""")
    env = SimpleNamespace(
        _model=model, _action_indices=np.array([0]), _n_intermediate_steps=1, simulation_dt=model.opt.timestep
    )
    env._control_func = DefaultControl(env)
    return env


def test_nonunit_gear_and_controller_normalization():
    env = synthetic()
    # ctrl0=(10*2*.2-1)/4=.75; normalized=(.75-1)/3
    assert resolve(env) == pytest.approx([-1 / 12])
    env._model.qpos0[0] = 100
    assert resolve(env) == [0.95]
    env._model.qpos0[0] = -100
    assert resolve(env) == [-0.95]


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("actuator_biastype", 0, mujoco.mjtBias.mjBIAS_NONE),
        ("actuator_gaintype", 0, mujoco.mjtGain.mjGAIN_AFFINE),
        ("actuator_dyntype", 0, mujoco.mjtDyn.mjDYN_FILTER),
        ("actuator_trntype", 0, mujoco.mjtTrn.mjTRN_TENDON),
        ("actuator_trnid", (0, 0), -1),
        ("jnt_type", 0, mujoco.mjtJoint.mjJNT_BALL),
        ("actuator_gainprm", (0, 0), 0),
        ("actuator_gainprm", (0, 0), np.inf),
        ("actuator_gear", (0, 0), 0),
        ("actuator_gear", (0, 0), np.nan),
        ("actuator_biasprm", (0, 0), np.nan),
        ("actuator_biasprm", (0, 1), 0),
        ("actuator_biasprm", (0, 1), 1),
        ("actuator_biasprm", (0, 2), np.inf),
        ("qpos0", 0, np.nan),
    ],
)
def test_isolated_eligibility_guards(field, index, value):
    env = synthetic()
    assert resolve(env) is not None
    getattr(env._model, field)[index] = value
    assert resolve(env) is None


@pytest.mark.parametrize(
    "field,value",
    [("norm_act_mean", np.nan), ("norm_act_delta", 0), ("norm_act_delta", -1), ("norm_act_delta", np.inf)],
)
def test_bad_normalization(field, value):
    env = synthetic()
    getattr(env._control_func, field)[0] = value
    assert resolve(env) is None


def test_wrappers_order_subset_and_missing_metadata(robots):
    original = robots[0]
    ids = np.asarray(original._action_indices)[[28, 5, 0, 3, 7]]
    env = SimpleNamespace(_model=original._model, _action_indices=ids, simulation_dt=0.002, _n_intermediate_steps=1)
    env._control_func = DefaultControl(env)
    expected = np.asarray(resolve(original))[[28, 5, 0, 3, 7]]
    np.testing.assert_array_equal(resolve(env), expected)
    for wrapped in (SimpleNamespace(unwrapped=env), SimpleNamespace(unwrapped=lambda: env)):
        np.testing.assert_array_equal(
            PPOJax._resolve_actor_bias_init(wrapped, {"actor_bias_init": "qpos0"}, 5), expected
        )
        assert (
            PPOJax._resolve_init_std(wrapped, OmegaConf.create({"init_std": 1, "init_std_motors": 0.2}), 5) == [0.2] * 5
        )
    assert resolve(env, dim=6) is None
    for name in ("_model", "_action_indices", "_control_func"):
        missing = SimpleNamespace(**vars(env))
        delattr(missing, name)
        assert PPOJax._resolve_actor_bias_init(missing, {}, 5) is None
    env._control_func = object()
    assert resolve(env) is None
    env._control_func = DefaultControl(env, apply_mode="incremental")
    assert resolve(env) is None


@pytest.mark.parametrize("mode", [None, False, "none"])
def test_disabled(mode):
    assert resolve(synthetic(), mode) is None


@pytest.mark.parametrize("mode", [True, "typo", 0, 1])
def test_invalid_mode(mode):
    with pytest.raises(ValueError, match="actor_bias_init"):
        resolve(synthetic(), mode)


def test_zero_vector_and_default():
    env = synthetic()
    # actor_bias_init is opt-in: the code default is "none", so an absent key
    # disables it.  Only configs that ask for it (the two robot configs) get a bias.
    assert PPOJax._resolve_actor_bias_init(env, {}, 1) is None
    assert resolve(env) is not None
    env._model.actuator_biasprm[0, 0] = 0
    env._model.qpos0[0] = 0.2  # ctrl0=1 equals controller midpoint
    assert resolve(env) is None


@pytest.mark.parametrize("kind", ["ordinary", "residual", "moe", "critic_output_moe"])
def test_only_actor_output_bias_changes_and_restore(kind):
    cls = SoftMoEActorCritic if "moe" in kind else ActorCritic
    kwargs = {"action_dim": 3, "hidden_layer_dims": (8, 8), "init_std": (0.2, 0.3, 0.4)}
    if kind == "residual":
        kwargs["use_residual"] = True
    if "moe" in kind:
        kwargs.update(num_experts=2, moe_at_layers=(0,))
    if kind == "critic_output_moe":
        kwargs.update(apply_moe_to="critic", apply_moe_to_output=True, expert_hidden_ratio=1)
    base = cls(**kwargs)
    bias = (0.25, -0.6, 0.95)
    shifted = cls(**kwargs, actor_bias_init=bias)
    x = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)
    params0 = base.init(jax.random.key(1), x)
    params1 = shifted.init(jax.random.key(1), x)
    flat0, flat1 = flatten_dict(params0), flatten_dict(params1)
    changed = [key for key in flat0 if not np.array_equal(flat0[key], flat1[key])]
    assert len(changed) == 1
    assert changed[0][-1] == "bias" and "actor" in "/".join(changed[0])
    for key in flat0.keys() - set(changed):
        np.testing.assert_array_equal(flat0[key], flat1[key])
    (pi0, v0), _ = base.apply(params0, x, mutable=["run_stats"])
    (pi1, v1), _ = shifted.apply(params1, x, mutable=["run_stats"])
    np.testing.assert_allclose(pi1.mean() - pi0.mean(), np.broadcast_to(bias, (4, 3)), atol=1e-7)
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_array_equal(pi0.stddev(), pi1.stddev())
    restored = flax.serialization.from_bytes(params1, flax.serialization.to_bytes(params0))
    (pi_restored, _), _ = shifted.apply(restored, x, mutable=["run_stats"])
    np.testing.assert_array_equal(pi_restored.mean(), pi0.mean())


@pytest.mark.parametrize("cls", [ActorCritic, SoftMoEActorCritic])
@pytest.mark.parametrize("bias", [(0.1,), ((0.1, 0.2),), (0.1, np.nan), (0.1, np.inf)])
def test_bad_vector(cls, bias):
    with pytest.raises(ValueError, match="actor_bias_init"):
        cls(2, hidden_layer_dims=(4,), actor_bias_init=bias).init(jax.random.key(0), jnp.ones((2, 3)))


def test_initializer_dtype_and_moe_output():
    fn = actor_output_bias_initializer((0.1, 0.2), 2)
    assert fn(None, (2,), jnp.float16).dtype == jnp.float16
    for target in ("actor", "both"):
        kwargs = {
            "action_dim": 2,
            "hidden_layer_dims": (4,),
            "num_experts": 2,
            "apply_moe_to_output": True,
            "apply_moe_to": target,
            "expert_hidden_ratio": 1,
        }
        with pytest.raises(ValueError, match="MoE actor output"):
            SoftMoEActorCritic(**kwargs, actor_bias_init=(0.1, 0.2)).init(jax.random.key(0), jnp.ones((2, 3)))
        SoftMoEActorCritic(**kwargs, actor_bias_init=(0, 0)).init(jax.random.key(0), jnp.ones((2, 3)))


@pytest.mark.parametrize("mode", [None, False, "none", "qpos0"])
def test_config_precedence(mode):
    exp = OmegaConf.create({"ppo_config": {"actor_bias_init": mode}})
    assert get_ppo_config(exp).actor_bias_init == mode
    exp.actor_bias_init = "top-level"
    assert get_ppo_config(exp).actor_bias_init == "top-level"


@pytest.mark.parametrize("compliant", [False, True])
def test_neutral_stance_physics(compliant):
    env = NewOSLFullBodyRobot(compliant_socket=compliant)
    model = env._model
    model.opt.timestep = 0.002
    servo = env._action_indices[:-2]
    qadr = model.jnt_qposadr[model.actuator_trnid[servo, 0]]
    for biased in (False, True):
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0
        action = np.asarray(resolve(env)) if biased else np.zeros(29)
        ctrl, _ = env._control_func.generate_action(env, action, model, data, None, np)
        data.ctrl[env._action_indices] = ctrl
        mujoco.mj_step(model, data, nstep=250)
        height = data.qpos[2]
        drift = np.max(np.abs(data.qpos[qadr] - model.qpos0[qadr]))
        if biased:
            assert height > 0.90
            assert drift < 0.12
        else:
            assert height < 0.65
            assert drift > 1


def test_ppo_resume_keeps_restored_bias():
    import optax

    from musclemimic.algorithms.ppo.config import PPOAgentState
    from musclemimic.algorithms.ppo.runner import _init_train_state

    env = SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(5,))))
    tx = optax.adam(1e-3)
    old = ActorCritic(2, hidden_layer_dims=(4,), actor_bias_init=(0.3, -0.4))
    new = old.clone(actor_bias_init=(-0.8, 0.9))
    ts = _init_train_state(jax.random.key(0), env, old, tx, None)
    restored = _init_train_state(
        jax.random.key(99), env, new, tx, PPOAgentState(ts), OmegaConf.create({"actor_bias_init": "none"})
    )
    for before, after in zip(jax.tree.leaves(ts.params), jax.tree.leaves(restored.params), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("use_moe", [False, True])
def test_ppo_passes_bias_to_network_and_logs_fresh_setting(robots, use_moe, caplog, capsys):
    env = robots[0]
    config = OmegaConf.create(
        {
            "experiment": {
                "actor_hidden_layers": [8],
                "critic_hidden_layers": [8],
                "init_std": 0.2,
                "activation": "tanh",
                "learnable_std": True,
                "use_moe": use_moe,
                # opt-in, exactly as the two robot configs set it
                "actor_bias_init": "qpos0",
            }
        }
    )
    network = PPOJax._create_network(env, config)
    np.testing.assert_array_equal(network.actor_bias_init, resolve(env))
    message = caplog.text + capsys.readouterr().out
    assert "fresh actor initialization: 17/29" in message
    assert "restored parameters take precedence" in message
