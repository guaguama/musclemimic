"""
Control-Lyapunov-Function math for the double-integrator surrogate.

The CLF used by :class:`~musclemimic.core.reward.clf.CLFReward` models each tracked output
channel as an independent double integrator::

    d/dt [e, edot] = [[0, 1], [0, 0]] [e, edot] + [0, 1] u

With a diagonal ``Q = diag(q_pos, q_vel)`` and scalar ``R = [r]`` shared across channels, the
CARE solution ``P`` is block diagonal with identical 2x2 blocks, so the whole Lyapunov
function collapses to three scalars. Nothing here needs a matrix at runtime.

All functions in this module are pure and environment-free. ``scipy`` is used only by the
construction-time helpers (:func:`exact_decay_rate`); the value/derivative functions that run
inside the jitted step are backend-agnostic and scipy-free.
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, Tuple

import numpy as np


def di_care_block(q_pos: float, q_vel: float, r: float) -> Tuple[float, float, float]:
    """
    Closed-form CARE solution for one double-integrator channel.

    Solves ``A^T P + P A - P B R^-1 B^T P + Q = 0`` analytically for
    ``A = [[0, 1], [0, 0]]``, ``B = [[0], [1]]``, ``Q = diag(q_pos, q_vel)``, ``R = [r]``.
    Agrees with ``scipy.linalg.solve_continuous_are`` to ~1e-13 over a wide parameter range.

    Args:
        q_pos (float): Position-error weight. Must be > 0.
        q_vel (float): Velocity-error weight. Must be >= 0.
        r (float): Control weight. Must be > 0.

    Returns:
        Tuple[float, float, float]: ``(p11, p12, p22)``, the upper triangle of the symmetric
        positive-definite 2x2 block.
    """
    if q_pos <= 0.0:
        raise ValueError(f"q_pos must be > 0, got {q_pos}")
    if q_vel < 0.0:
        raise ValueError(f"q_vel must be >= 0, got {q_vel}")
    if r <= 0.0:
        raise ValueError(f"r must be > 0, got {r}")

    p12 = math.sqrt(r * q_pos)
    p22 = math.sqrt(r * q_vel + 2.0 * (r ** 1.5) * math.sqrt(q_pos))
    p11 = p22 * math.sqrt(q_pos / r)
    return p11, p12, p22


def block_matrix(p11: float, p12: float, p22: float) -> np.ndarray:
    """Return the 2x2 block as a dense array (tests and diagnostics only)."""
    return np.array([[p11, p12], [p12, p22]], dtype=np.float64)


def block_lambda_max(p11: float, p12: float, p22: float) -> float:
    """
    Largest eigenvalue of the 2x2 block.

    For a symmetric positive-definite ``P`` this equals ``||P||_2``; the two are the same
    number, not two different bounds.
    """
    mid = 0.5 * (p11 + p22)
    half_diff = 0.5 * (p11 - p22)
    return mid + math.sqrt(half_diff * half_diff + p12 * p12)


def closed_loop_q(q_pos: float, q_vel: float, r: float,
                  p11: float, p12: float) -> np.ndarray:
    """
    ``Q_cl = Q + K^T R K`` for the LQR-optimal ``K = R^-1 B^T P = [sqrt(q_pos/r), p22/r]``.

    Under that controller ``Vdot = -eta^T Q_cl eta``, which is what bounds the achievable
    decay rate. The closed form below is exact to machine precision.
    """
    return np.array(
        [[2.0 * q_pos, p11],
         [p11, 2.0 * q_vel + 2.0 * p12]],
        dtype=np.float64,
    )


def exact_decay_rate(q_pos: float, q_vel: float, r: float) -> float:
    """
    Largest ``alpha`` for which ``Vdot + alpha * V <= 0`` holds for every ``eta`` under the
    surrogate's optimal control.

    Because ``Vdot = -eta^T Q_cl eta`` and ``V = eta^T P eta``, the tightest such ``alpha`` is
    ``min_eta (eta^T Q_cl eta) / (eta^T P eta)``, i.e. the smallest generalized eigenvalue of
    the pencil ``(Q_cl, P)``.

    .. note::
        ``lambda_min(Q_cl) / lambda_max(P)`` (see :func:`conservative_decay_rate`) is a
        *conservative sufficient* bound on the same quantity, not the achievable rate. It
        understates this value by between ~1.03x and ~48x over the usable Q/R range, so it is
        not a safe basis for choosing ``alpha``.

    Uses scipy; construction-time only.
    """
    from scipy.linalg import eigh

    p11, p12, p22 = di_care_block(q_pos, q_vel, r)
    q_cl = closed_loop_q(q_pos, q_vel, r, p11, p12)
    p = block_matrix(p11, p12, p22)
    return float(eigh(q_cl, p, eigvals_only=True)[0])


def conservative_decay_rate(q_pos: float, q_vel: float, r: float) -> float:
    """``lambda_min(Q_cl) / lambda_max(P)`` -- the conservative bound, kept for comparison."""
    p11, p12, p22 = di_care_block(q_pos, q_vel, r)
    q_cl = closed_loop_q(q_pos, q_vel, r, p11, p12)
    return float(np.linalg.eigvalsh(q_cl)[0] / block_lambda_max(p11, p12, p22))


def v_max_unit(p11: float, p12: float, p22: float) -> float:
    """
    Per-channel ``V`` when the normalized position *and* velocity errors are both one unit.

    This is the exact worst case at the reference operating point, and is the correct scale for
    ``exp(-V / V_max)``. It is *not* ``lambda_max(P)``: that bound holds only when the unit is
    the 2-norm of the whole ``[e, edot]`` pair, and is ~1.7x too small at the defaults.
    """
    return p11 + 2.0 * p12 + p22


def clf_value(e: Any, ed: Any, p11: float, p12: float, p22: float,
              n_channels: int, backend: ModuleType) -> Tuple[Any, Any]:
    """
    Mean-per-channel Lyapunov value and its cross term.

    ``V = (1/n) sum_j [p11 e_j^2 + 2 p12 e_j edot_j + p22 edot_j^2]``

    The mean (rather than the sum) keeps ``V`` comparable to a per-channel ``V_max`` that does
    not shift when the channel set changes. The decrease condition is homogeneous of degree one
    in ``P``, so this rescaling does not affect the CLF property.

    Args:
        e: Normalized position errors, shape ``(n_channels,)``.
        ed: Normalized velocity errors, shape ``(n_channels,)``.
        p11, p12, p22 (float): CARE block entries.
        n_channels (int): Channel count (static).
        backend (ModuleType): numpy or jax.numpy.

    Returns:
        Tuple: ``(V, cross)``. ``cross`` is the ``2 p12 <e, edot> / n`` contribution alone --
        the only part with no analogue in the exponential mimic terms. It is negative exactly
        when the tracking error is contracting.
    """
    inv_n = 1.0 / float(n_channels)
    cross = (2.0 * p12 * inv_n) * backend.sum(e * ed)
    v = (p11 * inv_n) * backend.sum(e * e) + cross + (p22 * inv_n) * backend.sum(ed * ed)
    # P is positive definite, so V >= 0 mathematically; float32 roundoff can give ~-1e-7.
    return backend.maximum(v, 0.0), cross


def clf_vdot(e: Any, ed: Any, edd: Any, p11: float, p12: float, p22: float,
             n_channels: int, backend: ModuleType) -> Any:
    """
    Time derivative of :func:`clf_value` in the scaled time ``t' = t / tau``.

    ``Vdot = (1/n) sum_j [2 edot_j (p11 e_j + p12 edot_j) + 2 eddot_j (p12 e_j + p22 edot_j)]``

    The first sum needs no history -- it is exact given ``ed``. Only ``edd`` is a finite
    difference, so passing ``edd = 0`` (the first step of an episode) degrades the estimate
    rather than zeroing it. ``Vdot`` is therefore generally nonzero on a first step; mask the
    *penalty*, not this value.
    """
    inv_n = 1.0 / float(n_channels)
    exact_part = backend.sum(ed * (p11 * e + p12 * ed))
    fd_part = backend.sum(edd * (p12 * e + p22 * ed))
    return (2.0 * inv_n) * (exact_part + fd_part)
