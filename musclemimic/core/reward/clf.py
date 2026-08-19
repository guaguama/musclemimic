"""
Control-Lyapunov-Function (CLF) imitation reward.

``MimicReward`` scores tracking with independent exponential kernels on position error and on
velocity error: it rewards error being small and velocity error being small, separately.
Nothing in it rewards error *contracting*. A policy that parks at a wrong-but-static
configuration scores well ("some position error, low velocity error").

``CLFReward`` adds a Lyapunov function over the tracking error,

    V = (1/n) sum_j [p11 e_j^2 + 2 p12 e_j edot_j + p22 edot_j^2],

whose cross term ``2 p12 e_j edot_j`` is negative exactly when the error is shrinking, plus a
penalty on violating the decrease condition ``Vdot + alpha V <= 0``. ``P`` comes from the CARE
of a double-integrator surrogate per channel (see :mod:`.clf_math`).

Approach ported from https://github.com/... robot_rl (IsaacLab/torch), with three deliberate
departures documented inline: an exact rather than conservative ``alpha``, a per-env reset
guard, and world-frame site velocities rather than ``site_rvel``.

**Approximations.** The position channels (``rpos``, ``root_pos``) are exactly paired: their
``edot`` is the true time derivative of their ``e``. The orientation channels (``rangles``,
``root_ori``) are paired only to first order: the exact map from a relative angular velocity to
the derivative of a rotation vector is ``thetadot = J^-1(theta) omega_rel``, and this class uses
``thetadot ~= omega_rel``, which is exact only as ``theta -> 0``. The frame of ``omega_rel`` *is*
handled exactly. So the Lyapunov property is exact for the position channels and approximate for
the orientation channels.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.utils.math import calc_site_velocities, quat_scalarfirst2scalarlast
from musclemimic.core.reward.clf_math import (
    clf_value,
    clf_vdot,
    di_care_block,
    exact_decay_rate,
    v_max_unit,
)
from musclemimic.core.reward.trajectory_based import MimicReward, MimicRewardState


# Fixed channel order. Groups are concatenated in this order, so the layout is stable across
# runs and the precomputed scale vector lines up with the per-step error vectors.
_GROUP_ORDER = ("rpos", "rangles", "root_pos", "root_ori", "qpos_joint")

_GROUP_DEFAULTS = {
    "rpos": {"enabled": True, "scale": 0.05},        # metres
    "rangles": {"enabled": True, "scale": 0.20},     # radians
    "root_pos": {"enabled": True, "scale": 0.10},    # metres
    "root_ori": {"enabled": True, "scale": 0.20},    # radians
    "qpos_joint": {"enabled": False, "scale": 0.10}, # radians (hinge) / metres (slide)
}


@struct.dataclass
class CLFRewardState(MimicRewardState):
    """
    State of CLFReward.

    Subclasses :class:`MimicRewardState` so ``imitation_error_total`` survives -- the autoreset
    wrapper reads that field by name (``musclemimic/core/wrappers/mjx.py``), and the existing
    tests assert its presence.

    Attributes:
        ed_prev: Previous normalized velocity error, shape ``(n_channels,)``. Needed for the
            finite-differenced second derivative. Must be a vector, not a scalar: a 0-d leaf
            that grew into a vector after the first call would be a ``lax.scan`` carry
            structure mismatch.
        clf_valid: 0.0 until the first call has written ``ed_prev``, then 1.0. Secondary guard
            only -- see ``_extra_reward_terms`` for why ``cur_step_in_episode`` is primary.
    """

    ed_prev: Union[np.ndarray, jnp.ndarray] = 0.0
    clf_valid: Union[np.ndarray, jnp.ndarray] = 0.0


class CLFReward(MimicReward):
    """
    DeepMimic reward extended with a control-Lyapunov-function term.

    Adds two independently weighted terms to whatever ``MimicReward`` already computes:

    * ``clf_w * exp(-V / V_max)`` -- a level term, added to the tracking sum.
    * ``-clf_pen_w * violation`` -- a decrease-condition penalty, added outside the existing
      penalty floor and before the non-negativity clip.

    Both default to weights that make the class a drop-in superset; set the exponential
    tracking weights (``rpos_w_sum`` etc.) to zero in the config to make the CLF the sole
    tracking signal, which is how the reference implementation operates.

    .. warning::
        Never enable ``clf_pen_w`` without ``clf_w``. The decrease condition is *relative*
        (``V_k <= beta V_{k-1}``), so the permitted absolute growth scales with ``V``: on its
        own the penalty mildly rewards sitting at large ``V``. The level term removes that.

    Args:
        env (Any): Environment instance.
        clf_w (float): Weight of ``exp(-V / V_max)``. Default 0.0 (off).
        clf_pen_w (float): Weight of the decrease-condition violation. Default 0.0 (off).
        clf_tau (float): Shared time scale, seconds. Must be shared across channels: with an
            independent per-channel velocity scale ``dV/dt`` is not the derivative of ``V`` in
            the surrogate's coordinates and ``alpha`` loses its meaning. Default 0.10.
        clf_q_pos (float): CARE position weight. Default 1.0.
        clf_q_vel (float): CARE velocity weight. Default 1.0.
        clf_r (float): CARE control weight. Default 0.1.
        clf_alpha (float, optional): Target decay rate in scaled time. ``None`` (default)
            auto-derives ``clf_alpha_safety * exact_decay_rate(q_pos, q_vel, r)``.
        clf_alpha_safety (float): Fraction of the achievable rate to demand. Default 0.5.
        clf_c_viol (float): Violation normalizer, as a fraction of ``V_max``. Default 0.10,
            i.e. the penalty saturates at roughly 10%-per-step growth of ``V``.
        clf_eps_dead (float): Deadband as a fraction of ``V_max``, suppressing noise-driven
            penalties once converged. Default 0.01.
        clf_groups (dict): Per-group ``{enabled, scale}``. See ``_GROUP_DEFAULTS``.
        **kwargs: Forwarded to :class:`MimicReward`.
    """

    _HAS_EXTRA_TERMS = True

    def __init__(self, env: Any, **kwargs):
        # --- CLF weights and shaping constants ------------------------------------------
        self._clf_w = float(kwargs.pop("clf_w", 0.0))
        self._clf_pen_w = float(kwargs.pop("clf_pen_w", 0.0))
        self._clf_tau = float(kwargs.pop("clf_tau", 0.10))
        q_pos = float(kwargs.pop("clf_q_pos", 1.0))
        q_vel = float(kwargs.pop("clf_q_vel", 1.0))
        r = float(kwargs.pop("clf_r", 0.1))
        alpha_cfg = kwargs.pop("clf_alpha", None)
        alpha_safety = float(kwargs.pop("clf_alpha_safety", 0.5))
        c_viol = float(kwargs.pop("clf_c_viol", 0.10))
        eps_dead_frac = float(kwargs.pop("clf_eps_dead", 0.01))
        groups_cfg = kwargs.pop("clf_groups", None)

        if self._clf_tau <= 0.0:
            raise ValueError(f"clf_tau must be > 0, got {self._clf_tau}")
        if c_viol <= 0.0:
            raise ValueError(f"clf_c_viol must be > 0, got {c_viol}")

        super().__init__(env, **kwargs)

        # --- CARE block (three scalars; no matrix, no runtime scipy) ---------------------
        self._p11, self._p12, self._p22 = di_care_block(q_pos, q_vel, r)
        self._v_max = v_max_unit(self._p11, self._p12, self._p22) + 1e-8
        self._eps_dead = eps_dead_frac * self._v_max
        self._viol_norm = c_viol * self._v_max

        # alpha from the *exact* achievable decay rate. lambda_min(Q_cl)/lambda_max(P) is a
        # conservative sufficient bound that understates this by 1.03x-48x over the usable Q/R
        # range, so it is not a safe basis for the target rate.
        if alpha_cfg is None:
            self._clf_alpha = alpha_safety * exact_decay_rate(q_pos, q_vel, r)
        else:
            self._clf_alpha = float(alpha_cfg)

        # --- control timestep ------------------------------------------------------------
        self._clf_dt = float(getattr(env, "dt", 0.01))
        if self._clf_dt <= 0.0:
            raise ValueError(f"env.dt must be > 0, got {self._clf_dt}")

        # --- channel groups --------------------------------------------------------------
        self._clf_group_cfg = self._parse_groups(groups_cfg)
        self._clf_enabled, self._clf_scales, self._n_channels = self._build_channel_layout(env)

        # site-velocity bookkeeping (mirrors calculate_relative_site_quantities)
        model = env._model
        self._clf_site_root_body_ids = np.asarray(model.body_rootid)[self._rel_body_ids]

    # ------------------------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _parse_groups(groups_cfg: Any) -> Dict[str, Dict[str, Any]]:
        """
        Normalize the group config to plain nested dicts.

        ``engine.py`` passes ``env_params`` straight through without ``OmegaConf.to_container``,
        so this arrives as a nested ``DictConfig`` during training but as a plain dict in tests.
        Normalize at both levels, as ``joint_torque_weights`` already does in
        :class:`MimicReward`.
        """
        raw = dict(groups_cfg or {})
        unknown = set(raw) - set(_GROUP_ORDER)
        if unknown:
            raise ValueError(
                f"Unknown clf_groups entries {sorted(unknown)}; valid groups are {list(_GROUP_ORDER)}"
            )
        out = {}
        for name in _GROUP_ORDER:
            cfg = dict(_GROUP_DEFAULTS[name])
            cfg.update(dict(raw.get(name, {}) or {}))
            scale = float(cfg["scale"])
            if scale <= 0.0:
                raise ValueError(f"clf_groups.{name}.scale must be > 0, got {scale}")
            out[name] = {"enabled": bool(cfg["enabled"]), "scale": scale}
        return out

    def _group_size(self, name: str, env: Any) -> int:
        """Channel count for a group, or 0 if it cannot be built for this env."""
        n_rel_sites = max(len(self._rel_site_ids) - 1, 0)
        if name in ("rpos", "rangles"):
            return 3 * n_rel_sites if len(self._rel_site_ids) > 1 else 0
        if name in ("root_pos", "root_ori"):
            return 3 if self._free_joint_qpos_ind is not None else 0
        if name == "qpos_joint":
            n_pos = int(np.count_nonzero(self._joint_qpos_mask))
            n_vel = int(np.count_nonzero(self._joint_qvel_mask))
            if n_pos != n_vel:
                raise ValueError(
                    f"clf_groups.qpos_joint needs a 1:1 qpos/qvel pairing but found "
                    f"{n_pos} position and {n_vel} velocity entries. This model most likely has "
                    f"ball joints (4 qpos / 3 qvel), which this group does not support; disable "
                    f"it or extend CLFReward to handle them."
                )
            return n_pos
        raise ValueError(f"unhandled group {name}")

    def _build_channel_layout(self, env: Any) -> Tuple[Tuple[str, ...], np.ndarray, int]:
        enabled, scales = [], []
        self._clf_slices = {}
        offset = 0
        for name in _GROUP_ORDER:
            if not self._clf_group_cfg[name]["enabled"]:
                continue
            size = self._group_size(name, env)
            if size == 0:
                continue
            enabled.append(name)
            scales.append(np.full(size, self._clf_group_cfg[name]["scale"], dtype=np.float32))
            self._clf_slices[name] = (offset, offset + size)
            offset += size
        if not enabled:
            raise ValueError(
                "CLFReward has no usable channel groups. Enable at least one of "
                f"{list(_GROUP_ORDER)} that is supported by this environment."
            )
        scale_vec = np.concatenate(scales)
        return tuple(enabled), scale_vec, int(scale_vec.size)

    # ------------------------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------------------------
    def init_state(self, env: Any, key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
        """
        Initialize the reward state.

        Must not read ``data.site_xpos``: on the MJX path ``Reward.reset`` runs before
        ``mjx.forward``, and on the CPU path ``data`` still holds the default pose, so any
        kinematics read here would be stale or NaN.
        """
        return CLFRewardState(
            last_qvel=data.qvel,
            last_action=backend.zeros(env.info.action_space.shape[0]),
            ed_prev=backend.zeros((self._n_channels,), dtype=backend.float32),
            clf_valid=backend.zeros((), dtype=backend.float32),
        )

    # ------------------------------------------------------------------------------------
    # per-group error signals
    # ------------------------------------------------------------------------------------
    def _traj_site_indices(self):
        """Trajectory-space site indices, or None when the model ids are already correct."""
        if self._site_mapper.requires_mapping:
            return self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
        return None

    def _relative_site_velocities(self, data, backend, traj_site_indices):
        """
        World-frame relative site velocities, split into the two halves the CLF needs.

        Returns ``(v_rel, omega_rel)`` with shapes ``(K, 3)`` where ``K = n_sites - 1``:

        * ``v_rel[i] = v_i - v_main`` in the world frame -- exactly ``d/dt`` of
          ``site_rpos = p_i - p_main``.
        * ``omega_rel[i] = R_main^T (omega_i - omega_main)`` -- the relative angular velocity
          expressed in the main site frame, matching the frame of
          ``site_rangles = rotvec(R_main^T R_i)``.

        .. note::
            This deliberately does not reuse ``site_rvel`` from
            ``calculate_relative_site_quantities``. That quantity is
            ``R_main @ (v_main - v_i)`` -- negated *and* rotated, with sim and reference each
            rotated by their own ``R_main`` -- and its angular half uses the opposite sign
            convention. Pairing it with ``site_rpos`` would invert the sign of the CLF cross
            term, i.e. reward the tracking error growing.
        """
        idx = traj_site_indices if traj_site_indices is not None else self._rel_site_ids
        xvel = calc_site_velocities(
            self._rel_site_ids, data, self._rel_body_ids, self._clf_site_root_body_ids,
            backend, flg_local=False, trajectory_site_indices=traj_site_indices,
        )                                            # (K+1, 6) world frame, [ang, lin]
        ang, lin = xvel[:, :3], xvel[:, 3:]
        v_rel = lin[1:] - lin[0]
        ang_rel_world = ang[1:] - ang[0]
        r_main = data.site_xmat[idx[0]].reshape(3, 3)
        omega_rel = backend.einsum("jk,ik->ij", r_main.T, ang_rel_world)
        return v_rel, omega_rel

    def _root_orientation(self, data, traj_data, backend):
        """
        Root orientation error and its paired derivative.

        ``e = rotvec(R_traj^T R_sim)``, expressed in the reference root frame.

        MuJoCo stores a free joint's ``qvel[3:6]`` as the angular velocity in the **body-local**
        frame (verified empirically, not assumed). The reference's own angular velocity is
        already in the reference root frame, so the relative angular velocity in that frame is
        ``R_rel @ omega_sim_local - omega_traj_local`` with ``R_rel = R_traj^T R_sim`` -- the
        same relative rotation whose rotation vector is ``e``.
        """
        rot = np_R if backend == np else jnp_R
        fq, fv = self._free_joint_qpos_ind, self._free_joint_qvel_ind
        m_sim = rot.from_quat(quat_scalarfirst2scalarlast(data.qpos[fq[3:7]])).as_matrix()
        m_traj = rot.from_quat(quat_scalarfirst2scalarlast(traj_data.qpos[fq[3:7]])).as_matrix()
        m_rel = m_traj.T @ m_sim
        e = rot.from_matrix(m_rel).as_rotvec()
        ed = m_rel @ data.qvel[fv[3:]] - traj_data.qvel[fv[3:]]
        return e, ed

    def _channel_errors(self, ctx, data, backend):
        """Concatenate the enabled groups' (e, edot) into two ``(n_channels,)`` vectors."""
        es, eds = [], []
        traj_data = ctx["traj_data"]
        need_sites = ("rpos" in self._clf_enabled) or ("rangles" in self._clf_enabled)
        if need_sites:
            traj_idx = self._traj_site_indices()
            v_rel_sim, om_rel_sim = self._relative_site_velocities(data, backend, None)
            v_rel_traj, om_rel_traj = self._relative_site_velocities(traj_data, backend, traj_idx)

        for name in self._clf_enabled:
            if name == "rpos":
                es.append((ctx["site_rpos"] - ctx["site_rpos_traj"]).ravel())
                eds.append((v_rel_sim - v_rel_traj).ravel())
            elif name == "rangles":
                es.append((ctx["site_rangles"] - ctx["site_rangles_traj"]).ravel())
                eds.append((om_rel_sim - om_rel_traj).ravel())
            elif name == "root_pos":
                fq, fv = self._free_joint_qpos_ind, self._free_joint_qvel_ind
                traj_xyz = traj_data.qpos[fq[:3]]
                if ctx["xy_offset"] is not None:
                    offset = backend.concatenate(
                        [ctx["xy_offset"], backend.zeros(1, dtype=ctx["xy_offset"].dtype)]
                    )
                    traj_xyz = traj_xyz - offset
                es.append(data.qpos[fq[:3]] - traj_xyz)
                # free-joint qvel[0:3] is world-frame linear velocity, and the xy offset is
                # constant within an episode, so this pairing is exact.
                eds.append(data.qvel[fv[:3]] - traj_data.qvel[fv[:3]])
            elif name == "root_ori":
                e, ed = self._root_orientation(data, traj_data, backend)
                es.append(e)
                eds.append(ed)
            elif name == "qpos_joint":
                es.append(ctx["qpos"][self._joint_qpos_mask] - ctx["qpos_traj"][self._joint_qpos_mask])
                eds.append(ctx["qvel"][self._joint_qvel_mask] - ctx["qvel_traj"][self._joint_qvel_mask])

        return backend.concatenate(es), backend.concatenate(eds)

    def _group_contributions(self, e, ed, backend) -> Dict[str, Any]:
        """
        Per-group contribution to ``V``, summing exactly to the total.

        Without this it is impossible to tell which group dominates ``V``, and therefore which
        ``scale`` is the effective tuning knob. Emitted for every group in ``_GROUP_ORDER``
        (zero for disabled ones) so the logged key set is static.
        """
        inv_n = 1.0 / float(self._n_channels)
        out = {}
        for name in _GROUP_ORDER:
            key = f"clf_V_{name}"
            span = self._clf_slices.get(name)
            if span is None:
                out[key] = backend.zeros((), dtype=e.dtype)
                continue
            lo, hi = span
            eg, edg = e[lo:hi], ed[lo:hi]
            out[key] = inv_n * (
                self._p11 * backend.sum(eg * eg)
                + 2.0 * self._p12 * backend.sum(eg * edg)
                + self._p22 * backend.sum(edg * edg)
            )
        return out

    # ------------------------------------------------------------------------------------
    # the hook
    # ------------------------------------------------------------------------------------
    def _extra_reward_terms(self, ctx: Dict[str, Any], env: Any,
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType):
        reward_state = carry.reward_state
        scales = backend.asarray(self._clf_scales)

        # normalize: e~ = e/s, ed~ = ed*tau/s, so d(e~)/dt' = ed~ exactly for t' = t/tau
        e_raw, ed_raw = self._channel_errors(ctx, data, backend)
        e = e_raw / scales
        ed = ed_raw * (self._clf_tau / scales)

        v, cross = clf_value(e, ed, self._p11, self._p12, self._p22, self._n_channels, backend)

        # First step of an episode. cur_step_in_episode is the load-bearing signal: it IS reset
        # by the autoreset wrapper (which does not reset reward_state), and it is incremented
        # only after the reward call, so it reads 1 exactly on the first step of every episode
        # on every backend. clf_valid is defence in depth for paths without that field.
        cur_step = getattr(carry, "cur_step_in_episode", 1)
        is_first = cur_step <= 1
        zero = backend.zeros((), dtype=e.dtype)
        one = backend.ones((), dtype=e.dtype)

        edd = backend.where(is_first, backend.zeros_like(ed), (ed - reward_state.ed_prev) * (self._clf_tau / self._clf_dt))
        vdot = clf_vdot(e, ed, edd, self._p11, self._p12, self._p22, self._n_channels, backend)

        violation = backend.clip(
            (vdot + self._clf_alpha * v - self._eps_dead) / self._viol_norm, 0.0, 1.0
        )
        mask = backend.where(is_first, zero, one) * reward_state.clf_valid
        violation = mask * violation

        # A non-finite V must map to the WORST level term, not the best: nan_to_num(V, 0) would
        # award the maximum tracking reward to a diverged state.
        v_norm = v / self._v_max
        level = backend.where(
            backend.isfinite(v_norm), backend.exp(-backend.minimum(v_norm, 5.0)), zero
        )

        extra_reward = self._clf_w * level
        extra_penalty = -(self._clf_pen_w * violation)

        # ed_prev/clf_valid are written unconditionally, including on masked steps: a
        # conditional write would reintroduce the cross-episode leak the mask exists to stop.
        extra_state = {
            "ed_prev": ed,
            "clf_valid": one,
            # imitation_error_total is not consumed anywhere in production today, but keep it
            # meaningful: its usual definition is a weighted sum of the tracking distances,
            # which collapses to zero once those weights are zeroed for a CLF-only run.
            "imitation_error_total": v,
        }
        extra_info = {
            "clf_V": v,
            "clf_V_norm": v_norm,
            "clf_vdot": vdot,
            "clf_cross": cross,
            "clf_violation": violation,
            "reward_clf": self._clf_w * level,
            "penalty_clf": extra_penalty,
        }
        extra_info.update(self._group_contributions(e, ed, backend))
        return extra_reward, extra_penalty, extra_info, extra_state


CLFReward.register()
