"""Tests for CLFReward and the CLF math module.

Structured in four groups:
  * pure math (no env) -- validates the closed forms against scipy
  * error signals -- validates that ``edot`` is the true derivative of ``e``
  * state / reset guard -- the AutoResetWrapper leak defence
  * integration -- MimicReward parity, backends, config parsing
"""

import numpy as np
import mujoco
import pytest
from scipy.linalg import eigh, solve_continuous_are
from scipy.spatial.transform import Rotation as R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils.math import calculate_relative_site_quantities
from musclemimic.core.reward import clf_math as cm
from musclemimic.core.reward.clf import CLFReward, CLFRewardState
from musclemimic.core.reward.trajectory_based import MimicReward, MimicRewardState

from tests.unit.test_mimic_reward import (
    MINIMAL_MJCF,
    FakeCarry,
    FakeTrajInfo,
    FakeTrajectoryHandler,
    make_carry,
    make_env,
    make_sim_data,
    make_traj_data,
)

QR_GRID = [(1.0, 1.0, 0.1), (100.0, 10.0, 0.1), (1.0, 1.0, 1.0),
           (2500.0, 50.0, 0.02), (25.0, 250.0, 0.1), (0.5, 0.01, 0.5)]

_A = np.array([[0.0, 1.0], [0.0, 0.0]])
_B = np.array([[0.0], [1.0]])


# =====================================================================
# Helpers
# =====================================================================
class CarryWithStep:
    """FakeCarry plus ``cur_step_in_episode``, which the real MJX carry has."""

    def __init__(self, traj_state, reward_state, cur_step_in_episode=5,
                 qvel_w_sum=0.1, root_vel_w_sum=0.1):
        self.traj_state = traj_state
        self.reward_state = reward_state
        self.cur_step_in_episode = cur_step_in_episode
        self.qvel_w_sum = qvel_w_sum
        self.root_vel_w_sum = root_vel_w_sum

    def replace(self, **kwargs):
        new = CarryWithStep(self.traj_state, self.reward_state, self.cur_step_in_episode,
                            self.qvel_w_sum, self.root_vel_w_sum)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new


def _perturb(d, seed, backend=np):
    """Give fake data non-trivial site kinematics so every branch is exercised."""
    rng = np.random.RandomState(seed)
    n = d.site_xpos.shape[0]
    d.site_xpos = backend.asarray(rng.uniform(-0.3, 0.3, size=d.site_xpos.shape))
    d.site_xmat = backend.asarray(
        R.from_rotvec(rng.uniform(-0.4, 0.4, size=(n, 3))).as_matrix().reshape(-1, 9)
    )
    d.xpos = backend.asarray(rng.uniform(-0.2, 0.2, size=d.xpos.shape))
    d.cvel = backend.asarray(rng.uniform(-0.5, 0.5, size=d.cvel.shape))
    d.subtree_com = backend.asarray(rng.uniform(-0.1, 0.1, size=d.subtree_com.shape))
    return d


QPOS_SIM = np.array([0.11, -0.22, 0.93, 0.9848, 0.0, 0.1736, 0.0, 0.31])
QPOS_TRAJ = np.array([0.05, -0.15, 0.90, 1.0, 0.0, 0.0, 0.0, 0.20])
QVEL_SIM = np.array([0.21, -0.11, 0.05, 0.03, -0.07, 0.02, 0.44])
QVEL_TRAJ = np.array([0.10, -0.05, 0.01, 0.01, -0.02, 0.01, 0.30])
ACTION = np.array([0.7, -1.4, 0.3])

BASE_PARAMS = dict(
    qpos_w_sum=0.13, qvel_w_sum=0.17, root_pos_w_sum=0.11, root_vel_w_sum=0.19,
    rpos_w_sum=0.61, rquat_w_sum=0.07, rvel_w_sum=0.09,
    action_out_of_bounds_coeff=0.013, joint_acc_coeff=0.0007, joint_torque_coeff=0.0003,
    action_rate_coeff=0.0011, activation_energy_coeff=0.0023,
)


def _fixture(backend=np):
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    traj = _perturb(make_traj_data(QPOS_TRAJ, QVEL_TRAJ, backend=backend), 1, backend)
    init = make_traj_data(np.array([0.02, -0.03, 0.9, 1.0, 0, 0, 0, 0.1]), backend=backend)
    sim = _perturb(make_sim_data(QPOS_SIM, QVEL_SIM, backend=backend), 2, backend)
    sim.qfrc_actuator = backend.asarray(np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.05, 0.6]))
    sim.act = backend.asarray(np.array([0.3, 0.6, 0.1]))
    th = FakeTrajectoryHandler(
        traj, init_data=init,
        traj_info=FakeTrajInfo(["pelvis_mimic", "upper_body_mimic", "child_mimic"]),
    )
    return model, make_env(model, th), sim


def _carry_for(reward, model, sim, cur_step=5, backend=np, valid=True):
    base = make_carry()
    rs = reward.init_state(_ENV_CACHE[0], None, model, sim, backend)
    rs = rs.replace(
        last_qvel=backend.asarray(np.array([0.05, -0.02, 0.0, 0.0, -0.01, 0.0, 0.10])),
        last_action=backend.asarray(np.array([0.1, -0.2, 0.05])),
    )
    if valid and hasattr(rs, "clf_valid"):
        rs = rs.replace(clf_valid=backend.asarray(1.0, dtype=backend.float32))
    return CarryWithStep(base.traj_state, rs, cur_step, 0.17, 0.19)


_ENV_CACHE = [None]


@pytest.fixture
def fx():
    model, env, sim = _fixture()
    _ENV_CACHE[0] = env
    return model, env, sim


# =====================================================================
# 1. Pure math
# =====================================================================
@pytest.mark.parametrize("q_pos,q_vel,r", QR_GRID)
def test_care_matches_scipy(q_pos, q_vel, r):
    p11, p12, p22 = cm.di_care_block(q_pos, q_vel, r)
    expected = solve_continuous_are(_A, _B, np.diag([q_pos, q_vel]), np.array([[r]]))
    assert np.allclose(cm.block_matrix(p11, p12, p22), expected, atol=1e-10)


@pytest.mark.parametrize("q_pos,q_vel,r", QR_GRID)
def test_lambda_max_equals_spectral_norm(q_pos, q_vel, r):
    p11, p12, p22 = cm.di_care_block(q_pos, q_vel, r)
    p = cm.block_matrix(p11, p12, p22)
    lam = cm.block_lambda_max(p11, p12, p22)
    assert lam == pytest.approx(np.linalg.eigvalsh(p)[-1], rel=1e-12)
    assert lam == pytest.approx(np.linalg.norm(p, 2), rel=1e-12)


@pytest.mark.parametrize("q_pos,q_vel,r", QR_GRID)
def test_block_is_positive_definite(q_pos, q_vel, r):
    p11, p12, p22 = cm.di_care_block(q_pos, q_vel, r)
    assert p11 > 0 and p22 > 0 and (p11 * p22 - p12 * p12) > 0


@pytest.mark.parametrize("q_pos,q_vel,r", QR_GRID)
def test_closed_loop_q_analytic_form(q_pos, q_vel, r):
    p11, p12, p22 = cm.di_care_block(q_pos, q_vel, r)
    k = np.array([[np.sqrt(q_pos / r), p22 / r]])
    expected = np.diag([q_pos, q_vel]) + k.T @ np.array([[r]]) @ k
    assert np.allclose(cm.closed_loop_q(q_pos, q_vel, r, p11, p12), expected, atol=1e-12)


@pytest.mark.parametrize("q_pos,q_vel,r", QR_GRID)
def test_alpha_is_exact_generalized_eigenvalue(q_pos, q_vel, r):
    """The exact rate is min gen-eig of (Q_cl, P), and it is NOT the conservative bound."""
    p11, p12, p22 = cm.di_care_block(q_pos, q_vel, r)
    q_cl = cm.closed_loop_q(q_pos, q_vel, r, p11, p12)
    p = cm.block_matrix(p11, p12, p22)
    expected = eigh(q_cl, p, eigvals_only=True)[0]
    exact = cm.exact_decay_rate(q_pos, q_vel, r)
    assert exact == pytest.approx(expected, rel=1e-10)
    # lambda_min(Q_cl)/lambda_max(P) is conservative: it must never exceed the true rate.
    assert cm.conservative_decay_rate(q_pos, q_vel, r) <= exact + 1e-12


def test_conservative_bound_materially_understates_rate():
    """Guards the reason we do not use the conservative bound to pick alpha."""
    exact = cm.exact_decay_rate(1.0, 1.0, 0.1)
    cons = cm.conservative_decay_rate(1.0, 1.0, 0.1)
    assert exact == pytest.approx(1.28402, rel=1e-4)
    assert cons == pytest.approx(0.72457, rel=1e-4)
    assert exact / cons > 1.7


@pytest.mark.parametrize("q_pos,q_vel,r", QR_GRID)
def test_V_matches_dense_P(q_pos, q_vel, r):
    """Elementwise V must equal eta^T P eta / n with the full block-diagonal P."""
    n = 7
    rng = np.random.RandomState(3)
    e, ed = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    p11, p12, p22 = cm.di_care_block(q_pos, q_vel, r)
    p_full = np.kron(np.eye(n), cm.block_matrix(p11, p12, p22))
    eta = np.empty(2 * n)
    eta[0::2], eta[1::2] = e, ed          # interleaved to match the kron layout
    expected = float(eta @ p_full @ eta) / n
    v, _ = cm.clf_value(e, ed, p11, p12, p22, n, np)
    assert float(v) == pytest.approx(expected, rel=1e-12)


def test_v_max_is_worst_case_at_unit_eta():
    """V_max must be V at (e=1, ed=1) per channel -- not lambda_max(P)."""
    p11, p12, p22 = cm.di_care_block(1.0, 1.0, 0.1)
    n = 5
    v, _ = cm.clf_value(np.ones(n), np.ones(n), p11, p12, p22, n, np)
    assert float(v) == pytest.approx(cm.v_max_unit(p11, p12, p22), rel=1e-12)
    assert cm.v_max_unit(p11, p12, p22) > cm.block_lambda_max(p11, p12, p22)


def test_V_zero_and_cross_zero_without_error():
    p11, p12, p22 = cm.di_care_block(1.0, 1.0, 0.1)
    v, cross = cm.clf_value(np.zeros(4), np.zeros(4), p11, p12, p22, 4, np)
    assert float(v) == 0.0 and float(cross) == 0.0


def test_cross_term_sign_and_V_ordering():
    """The whole point of the CLF: contracting error scores strictly better."""
    p11, p12, p22 = cm.di_care_block(1.0, 1.0, 0.1)
    e = np.linspace(0.2, 1.0, 9)
    v_shrink, c_shrink = cm.clf_value(e, -0.8 * e, p11, p12, p22, 9, np)
    v_grow, c_grow = cm.clf_value(e, +0.8 * e, p11, p12, p22, 9, np)
    assert float(c_shrink) < 0.0 < float(c_grow)
    assert float(v_shrink) < float(v_grow)


def test_vdot_exact_term_survives_zero_edd():
    """With edd=0 the analytic half of Vdot remains, so Vdot is NOT zero."""
    p11, p12, p22 = cm.di_care_block(1.0, 1.0, 0.1)
    e, ed = np.array([0.5, -0.3]), np.array([0.2, 0.1])
    vdot = cm.clf_vdot(e, ed, np.zeros(2), p11, p12, p22, 2, np)
    expected = 2.0 / 2 * float(np.sum(ed * (p11 * e + p12 * ed)))
    assert float(vdot) == pytest.approx(expected, rel=1e-12)
    assert float(vdot) != 0.0


@pytest.mark.parametrize("bad", [dict(q_pos=0.0), dict(q_pos=-1.0), dict(r=0.0), dict(q_vel=-1.0)])
def test_care_rejects_invalid_weights(bad):
    kw = dict(q_pos=1.0, q_vel=1.0, r=0.1)
    kw.update(bad)
    with pytest.raises(ValueError):
        cm.di_care_block(**kw)


# =====================================================================
# 2. Error signals -- edot must be the true derivative of e
# =====================================================================
def test_edot_is_true_derivative_of_site_rpos():
    """
    Regression test for the site_rvel trap.

    Finite-differences ``site_rpos`` on real MuJoCo data and compares against the helper's
    analytic relative velocity. Also asserts that ``site_rvel`` -- the quantity
    ``calculate_relative_site_quantities`` returns -- FAILS the same test, which is why it must
    not be used to build eta.
    """
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    data = mujoco.MjData(model)
    data.qpos[:] = QPOS_SIM
    data.qvel[:] = QVEL_SIM
    mujoco.mj_forward(model, data)

    th = FakeTrajectoryHandler(
        make_traj_data(QPOS_TRAJ), traj_info=FakeTrajInfo(
            ["pelvis_mimic", "upper_body_mimic", "child_mimic"]))
    env = make_env(model, th)
    reward = CLFReward(env, **BASE_PARAMS)

    def site_rpos_of(d):
        rpos, _, rvel = calculate_relative_site_quantities(
            d, reward._rel_site_ids, reward._rel_body_ids, model.body_rootid, np)
        return rpos, rvel

    rpos0, rvel0 = site_rpos_of(data)
    v_rel, _ = reward._relative_site_velocities(data, np, None)

    # central difference of site_rpos under the true dynamics
    h = 1e-6
    fwd, bwd = mujoco.MjData(model), mujoco.MjData(model)
    for dst, sgn in ((fwd, +1.0), (bwd, -1.0)):
        dst.qpos[:] = QPOS_SIM
        dst.qvel[:] = QVEL_SIM
        mujoco.mj_integratePos(model, dst.qpos, dst.qvel, sgn * h)
        mujoco.mj_forward(model, dst)
    fd = (site_rpos_of(fwd)[0] - site_rpos_of(bwd)[0]) / (2.0 * h)

    assert np.allclose(v_rel, fd, atol=1e-4), f"helper {v_rel} vs finite diff {fd}"
    # and the trap: site_rvel's linear half is not this derivative
    assert not np.allclose(rvel0[:, 3:], fd, atol=1e-3)


def test_relative_angular_velocity_is_in_main_site_frame():
    """omega_rel must be R_main^T (omega_i - omega_main), matching rotvec(R_main^T R_i)."""
    from loco_mujoco.core.utils.math import calc_site_velocities

    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    data = mujoco.MjData(model)
    data.qpos[:] = QPOS_SIM
    data.qvel[:] = QVEL_SIM
    mujoco.mj_forward(model, data)

    th = FakeTrajectoryHandler(make_traj_data(QPOS_TRAJ),
                              traj_info=FakeTrajInfo(["pelvis_mimic", "upper_body_mimic", "child_mimic"]))
    reward = CLFReward(make_env(model, th), **BASE_PARAMS)
    _, omega_rel = reward._relative_site_velocities(data, np, None)

    xvel = calc_site_velocities(reward._rel_site_ids, data, reward._rel_body_ids,
                               reward._clf_site_root_body_ids, np, flg_local=False)
    ang = xvel[:, :3]
    r_main = data.site_xmat[reward._rel_site_ids[0]].reshape(3, 3)
    expected = (ang[1:] - ang[0]) @ r_main       # == R_main^T @ (w_i - w_main) per row
    assert np.allclose(omega_rel, expected, atol=1e-12)
    # sanity: the raw world-frame difference is a genuinely different quantity here
    assert not np.allclose(omega_rel, ang[1:] - ang[0], atol=1e-6)


# =====================================================================
# 3. State and the reset guard
# =====================================================================
def test_state_subclasses_mimic_state(fx):
    model, env, sim = fx
    reward = CLFReward(env, **BASE_PARAMS)
    st = reward.init_state(env, None, model, sim, np)
    assert isinstance(st, CLFRewardState) and isinstance(st, MimicRewardState)
    # the autoreset wrapper reads this field by name
    assert hasattr(st, "imitation_error_total")


def test_ed_prev_is_a_vector_with_stable_shape(fx):
    model, env, sim = fx
    reward = CLFReward(env, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    st = reward.init_state(env, None, model, sim, np)
    assert st.ed_prev.shape == (reward._n_channels,)
    carry = _carry_for(reward, model, sim)
    _, new_carry, _ = reward(None, ACTION, None, False, {}, env, model, sim, carry, np)
    assert new_carry.reward_state.ed_prev.shape == (reward._n_channels,)


def test_init_state_does_not_read_kinematics(fx):
    """Reset runs before mjx.forward, so site_xpos may be stale/NaN there."""
    model, env, sim = fx
    reward = CLFReward(env, **BASE_PARAMS)
    sim.site_xpos = np.full_like(sim.site_xpos, np.nan)
    sim.site_xmat = np.full_like(sim.site_xmat, np.nan)
    st = reward.init_state(env, None, model, sim, np)
    assert np.all(np.isfinite(st.ed_prev)) and np.isfinite(st.clf_valid)


@pytest.mark.parametrize("poison", [+1e6, -1e6])
def test_first_step_penalty_is_exactly_zero(fx, poison):
    """
    reward_state is NOT reset by AutoResetWrapper, so ed_prev leaks across episodes.
    cur_step_in_episode <= 1 must mask the penalty regardless of how stale ed_prev is.
    """
    model, env, sim = fx
    reward = CLFReward(env, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    carry = _carry_for(reward, model, sim, cur_step=1)
    carry = carry.replace(reward_state=carry.reward_state.replace(
        ed_prev=np.full(reward._n_channels, poison, dtype=np.float32)))
    _, _, info = reward(None, ACTION, None, False, {}, env, model, sim, carry, np)
    assert float(info["penalty_clf"]) == 0.0
    assert float(info["clf_violation"]) == 0.0
    # Vdot is NOT masked -- its analytic half needs no history
    assert np.isfinite(float(info["clf_vdot"]))


def test_second_step_penalty_fires_on_violation(fx):
    model, env, sim = fx
    reward = CLFReward(env, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    carry = _carry_for(reward, model, sim, cur_step=2)
    carry = carry.replace(reward_state=carry.reward_state.replace(
        ed_prev=np.full(reward._n_channels, -1e6, dtype=np.float32)))   # -> Vdot very positive
    _, _, info = reward(None, ACTION, None, False, {}, env, model, sim, carry, np)
    assert float(info["clf_vdot"]) > 0.0
    assert float(info["clf_violation"]) == pytest.approx(1.0)
    assert float(info["penalty_clf"]) == pytest.approx(-0.25)


def test_ed_prev_written_unconditionally_on_masked_step(fx):
    model, env, sim = fx
    reward = CLFReward(env, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    carry = _carry_for(reward, model, sim, cur_step=1)
    carry = carry.replace(reward_state=carry.reward_state.replace(
        ed_prev=np.full(reward._n_channels, 7.0, dtype=np.float32)))
    _, new_carry, _ = reward(None, ACTION, None, False, {}, env, model, sim, carry, np)
    assert not np.allclose(new_carry.reward_state.ed_prev, 7.0)
    assert float(new_carry.reward_state.clf_valid) == 1.0


def test_clf_valid_is_a_secondary_guard(fx):
    """With clf_valid=0 the penalty is masked even at a late step."""
    model, env, sim = fx
    reward = CLFReward(env, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    carry = _carry_for(reward, model, sim, cur_step=9, valid=False)
    carry = carry.replace(reward_state=carry.reward_state.replace(
        ed_prev=np.full(reward._n_channels, -1e6, dtype=np.float32)))
    _, _, info = reward(None, ACTION, None, False, {}, env, model, sim, carry, np)
    assert float(info["penalty_clf"]) == 0.0


def test_nonfinite_V_gives_zero_level_term(fx):
    """A diverged state must get the WORST level term, not exp(-5) and certainly not 1."""
    model, env, sim = fx
    reward = CLFReward(env, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    carry = _carry_for(reward, model, sim)
    sim.site_xpos = np.full_like(sim.site_xpos, np.nan)
    _, _, info = reward(None, ACTION, None, False, {}, env, model, sim, carry, np)
    assert float(info["reward_clf"]) == 0.0


# =====================================================================
# 4. Integration
# =====================================================================
def test_registered_under_its_class_name():
    import musclemimic.core  # noqa: F401  (registration side effect)
    assert Reward.registered["CLFReward"] is CLFReward
    assert CLFReward.get_name() == "CLFReward"


def test_mimic_parity_when_clf_weights_zero(fx):
    """The hook must be exactly additive: zero CLF weights => bit-identical MimicReward."""
    model, env, sim = fx
    assert MimicReward._HAS_EXTRA_TERMS is False
    assert CLFReward._HAS_EXTRA_TERMS is True

    mimic = MimicReward(env, **BASE_PARAMS)
    clf = CLFReward(env, clf_w=0.0, clf_pen_w=0.0, **BASE_PARAMS)
    t_m, _, i_m = mimic(None, ACTION, None, False, {}, env, model, sim,
                        _carry_for(mimic, model, sim), np)
    t_c, _, i_c = clf(None, ACTION, None, False, {}, env, model, sim,
                      _carry_for(clf, model, sim), np)
    assert float(t_m) == float(t_c)                     # 0 ULP
    for key in set(i_m) & set(i_c):
        assert float(i_m[key]) == float(i_c[key]), key


def test_reward_preclip_is_exported_by_mimic_reward(fx):
    """Needed for clf/frac_clipped; the max(reward, 0) clip is otherwise invisible."""
    model, env, sim = fx
    mimic = MimicReward(env, **BASE_PARAMS)
    total, _, info = mimic(None, ACTION, None, False, {}, env, model, sim,
                           _carry_for(mimic, model, sim), np)
    assert "reward_preclip" in info
    assert float(info["reward_preclip"]) >= float(total) - 1e-12


def test_clf_weights_are_independent(fx):
    model, env, sim = fx
    kw = dict(clf_tau=0.10, **BASE_PARAMS)
    r0 = CLFReward(env, clf_w=0.0, clf_pen_w=0.0, **kw)
    r_lvl = CLFReward(env, clf_w=0.5, clf_pen_w=0.0, **kw)
    r_pen = CLFReward(env, clf_w=0.0, clf_pen_w=0.5, **kw)
    t0, _, _ = r0(None, ACTION, None, False, {}, env, model, sim, _carry_for(r0, model, sim), np)
    t1, _, i1 = r_lvl(None, ACTION, None, False, {}, env, model, sim, _carry_for(r_lvl, model, sim), np)
    t2, _, i2 = r_pen(None, ACTION, None, False, {}, env, model, sim, _carry_for(r_pen, model, sim), np)
    assert float(i1["penalty_clf"]) == 0.0
    assert float(i2["reward_clf"]) == 0.0
    assert float(t1) == pytest.approx(float(t0) + float(i1["reward_clf"]), abs=1e-9)


def test_numpy_jax_parity(fx):
    import jax.numpy as jnp

    model, env_np, sim_np = fx
    reward_np = CLFReward(env_np, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    out_np = reward_np(None, ACTION, None, False, {}, env_np, model, sim_np,
                       _carry_for(reward_np, model, sim_np), np)

    model_j, env_j, sim_j = _fixture(backend=jnp)
    _ENV_CACHE[0] = env_j
    reward_j = CLFReward(env_j, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    carry_j = _carry_for(reward_j, model_j, sim_j, backend=jnp)
    carry_j = carry_j.replace(reward_state=carry_j.reward_state.replace(
        ed_prev=jnp.zeros((reward_j._n_channels,), dtype=jnp.float32)))
    out_j = reward_j(None, jnp.asarray(ACTION), None, False, {}, env_j, model_j, sim_j, carry_j, jnp)

    for key in ("clf_V", "clf_vdot", "clf_cross", "clf_violation", "reward_clf"):
        assert float(out_np[2][key]) == pytest.approx(float(out_j[2][key]), rel=2e-5, abs=2e-5), key
    assert float(out_np[0]) == pytest.approx(float(out_j[0]), rel=2e-5, abs=2e-5)


def test_groups_accept_omegaconf_dictconfig(fx):
    from omegaconf import OmegaConf

    model, env, sim = fx
    groups = {"rpos": {"enabled": True, "scale": 0.07},
              "rangles": {"enabled": False, "scale": 0.2},
              "root_pos": {"enabled": False, "scale": 0.1},
              "root_ori": {"enabled": False, "scale": 0.2}}
    r_plain = CLFReward(env, clf_groups=groups, **BASE_PARAMS)
    r_conf = CLFReward(env, clf_groups=OmegaConf.create(groups), **BASE_PARAMS)
    assert r_plain._clf_enabled == r_conf._clf_enabled == ("rpos",)
    assert r_plain._n_channels == r_conf._n_channels
    assert np.allclose(r_plain._clf_scales, r_conf._clf_scales)
    assert np.allclose(r_conf._clf_scales, 0.07)


def test_default_group_selection_and_channel_count(fx):
    model, env, sim = fx
    reward = CLFReward(env, **BASE_PARAMS)
    # MINIMAL_MJCF: 3 mimic sites -> 2 relative sites -> 6 channels each for rpos/rangles,
    # plus 3 for root_pos and 3 for root_ori.
    assert reward._clf_enabled == ("rpos", "rangles", "root_pos", "root_ori")
    assert reward._n_channels == 6 + 6 + 3 + 3


def test_unknown_group_raises(fx):
    model, env, sim = fx
    with pytest.raises(ValueError, match="Unknown clf_groups"):
        CLFReward(env, clf_groups={"not_a_group": {"enabled": True}}, **BASE_PARAMS)


@pytest.mark.parametrize("bad", [dict(clf_tau=0.0), dict(clf_c_viol=0.0)])
def test_invalid_scalars_raise(fx, bad):
    model, env, sim = fx
    with pytest.raises(ValueError):
        CLFReward(env, **bad, **BASE_PARAMS)


def test_alpha_auto_derived_from_exact_rate(fx):
    model, env, sim = fx
    reward = CLFReward(env, clf_q_pos=1.0, clf_q_vel=1.0, clf_r=0.1,
                       clf_alpha_safety=0.5, **BASE_PARAMS)
    assert reward._clf_alpha == pytest.approx(0.5 * cm.exact_decay_rate(1.0, 1.0, 0.1), rel=1e-12)
    explicit = CLFReward(env, clf_alpha=0.25, **BASE_PARAMS)
    assert explicit._clf_alpha == 0.25


def test_group_contributions_sum_to_V(fx):
    """Per-group V contributions must partition the total exactly."""
    model, env, sim = fx
    reward = CLFReward(env, clf_w=1.0, clf_pen_w=0.25, **BASE_PARAMS)
    carry = _carry_for(reward, model, sim)
    _, _, info = reward(None, ACTION, None, False, {}, env, model, sim, carry, np)
    parts = [float(info[f"clf_V_{g}"])
             for g in ("rpos", "rangles", "root_pos", "root_ori", "qpos_joint")]
    assert sum(parts) == pytest.approx(float(info["clf_V"]), rel=1e-10)
    # disabled group contributes exactly zero
    assert float(info["clf_V_qpos_joint"]) == 0.0
    # keys exist regardless of which groups are enabled (static log schema)
    only_rpos = CLFReward(env, clf_groups={"rangles": {"enabled": False},
                                          "root_pos": {"enabled": False},
                                          "root_ori": {"enabled": False}}, **BASE_PARAMS)
    _, _, info2 = only_rpos(None, ACTION, None, False, {}, env, model, sim,
                            _carry_for(only_rpos, model, sim), np)
    for g in ("rpos", "rangles", "root_pos", "root_ori", "qpos_joint"):
        assert f"clf_V_{g}" in info2
