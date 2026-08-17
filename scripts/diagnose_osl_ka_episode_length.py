#!/usr/bin/env python3
"""Diagnose short OSL KA episode lengths without modifying training code.

The script builds OSL KA train/validation environments from a Hydra config and
runs small rollouts outside PPO.  It reports the first done step, the likely done
cause, and MJX autoreset accounting so short validation videos can be separated
from real two-step terminations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

# Import registration side effects used by TaskFactory.make().
import loco_mujoco.core.control_functions  # noqa: F401
import loco_mujoco.core.domain_randomizer  # noqa: F401
import loco_mujoco.core.initial_state_handler  # noqa: F401
import loco_mujoco.core.observations  # noqa: F401
import loco_mujoco.core.reward  # noqa: F401
import loco_mujoco.core.terminal_state_handler  # noqa: F401
import loco_mujoco.core.terrain  # noqa: F401
import musclemimic.core  # noqa: F401
import musclemimic.environments  # noqa: F401
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms.common.env_state_utils import unwrap_to_mjx
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.core.wrappers import LogEnvState

VAL_DATASET_CONF_KEYS = ("amass_dataset_conf", "lafan1_dataset_conf", "custom_dataset_conf")
ROLE_CHOICES = ("train", "validation")
BACKEND_CHOICES = ("cpu", "mjx")
START_MODE_CHOICES = ("fixed", "random")
ACTION_MODE_CHOICES = ("zero", "deterministic", "low_noise", "config_std")

AUDIT_DATA_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc",
    "qacc_warmstart",
    "qfrc_actuator",
    "qfrc_applied",
    "qfrc_bias",
    "qfrc_constraint",
    "qfrc_fluid",
    "qfrc_gravcomp",
    "qfrc_passive",
    "qfrc_smooth",
    "xpos",
    "xquat",
    "xmat",
    "site_xpos",
    "site_xmat",
    "cvel",
    "subtree_com",
    "ten_length",
    "ten_velocity",
    "actuator_length",
    "actuator_velocity",
    "sensordata",
)

RESET_PARITY_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "xpos",
    "xquat",
    "site_xpos",
    "ten_length",
    "actuator_length",
)


@dataclasses.dataclass
class RolloutSummary:
    role: str
    backend: str
    start_mode: str
    action_mode: str
    num_envs: int
    steps_requested: int
    episodes_requested: int
    first_done_step: int | None
    first_done_env: int | None
    first_done_cause: str | None
    done_count: int
    cause_counts: dict[str, int]
    episode_lengths: list[int]
    reset_accounting_ok: bool | None = None
    reset_accounting_errors: dict[str, int] | None = None
    final_autoreset_done_count: list[int] | None = None
    audit_findings: list[dict[str, Any]] | None = None
    reset_parity: dict[str, Any] | None = None
    skipped: str | None = None
    error: str | None = None


def _parse_csv(raw: str | None, allowed: tuple[str, ...], label: str) -> list[str]:
    if raw is None or raw == "all":
        return list(allowed)
    values = [v.strip() for v in raw.split(",") if v.strip()]
    bad = sorted(set(values) - set(allowed))
    if bad:
        raise ValueError(f"Unsupported {label}: {bad}. Allowed: {', '.join(allowed)}")
    return values


def _load_hydra_config(config_path: str) -> DictConfig:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        cfg = compose(config_name=path.stem)
    OmegaConf.resolve(cfg)
    return cfg


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True) or {}
    return dict(cfg)


def _apply_max_motions(task_params: dict[str, Any], max_motions: int | None) -> None:
    if max_motions is None:
        return
    for key in VAL_DATASET_CONF_KEYS:
        dataset_conf = task_params.get(key)
        if isinstance(dataset_conf, dict):
            dataset_conf["max_motions"] = int(max_motions)


def _role_env_params(config: DictConfig, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    env_params = _to_plain_dict(config.experiment.env_params)
    task_params = _to_plain_dict(config.experiment.task_factory.params)

    if role == "validation":
        val_cfg = config.experiment.get("validation", {})
        env_params["terminal_state_type"] = val_cfg.get("terminal_state_type", "NoTerminalStateHandler")
        env_params["terminal_state_params"] = _to_plain_dict(val_cfg.get("terminal_state_params", {}))
        env_params["num_envs"] = int(val_cfg.get("num_envs", env_params.get("num_envs", 1)))

        if val_cfg.get("start_from_beginning", False):
            env_params.setdefault("th_params", {})
            env_params["th_params"]["start_from_random_step"] = False

        val_th_params = val_cfg.get("th_params", None)
        if val_th_params is not None:
            merged = env_params.get("th_params", {}) or {}
            env_params["th_params"] = {**merged, **_to_plain_dict(val_th_params)}

        for key in VAL_DATASET_CONF_KEYS:
            val_dataset = val_cfg.get(key, None)
            if val_dataset is not None:
                task_params[key] = _to_plain_dict(val_dataset)

    return env_params, task_params


def _apply_backend_params(
    env_params: dict[str, Any],
    backend: str,
    mjx_backend: str | None,
    num_envs: int,
) -> dict[str, Any]:
    env_params = dict(env_params)
    env_name = str(env_params.get("env_name", ""))

    if backend == "cpu":
        if env_name.startswith("Mjx"):
            env_params["env_name"] = env_name[3:]
        env_params.pop("mjx_backend", None)
        env_params["num_envs"] = 1
        return env_params

    if not env_name.startswith("Mjx"):
        env_params["env_name"] = f"Mjx{env_name}"
    if mjx_backend is not None:
        env_params["mjx_backend"] = mjx_backend
    env_params["num_envs"] = int(num_envs)
    return env_params


def _parse_model_option_value(raw: str) -> Any:
    text = raw.strip()
    if hasattr(mujoco.mjtDisableBit, text):
        return int(getattr(mujoco.mjtDisableBit, text))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _default_mjx_model_options() -> dict[str, Any]:
    return {
        "iterations": 4,
        "ls_iterations": 8,
        "disableflags": int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
    }


def _apply_diagnostic_overrides(
    env_params: dict[str, Any],
    backend: str,
    args: argparse.Namespace | None,
) -> dict[str, Any]:
    if args is None:
        return env_params

    env_params = dict(env_params)
    if getattr(args, "timestep", None) is not None:
        env_params["timestep"] = float(args.timestep)
    if getattr(args, "n_substeps", None) is not None:
        env_params["n_substeps"] = int(args.n_substeps)

    option_updates: dict[str, Any] = {}
    if getattr(args, "solver_iterations", None) is not None:
        option_updates["iterations"] = int(args.solver_iterations)
    if getattr(args, "ls_iterations", None) is not None:
        option_updates["ls_iterations"] = int(args.ls_iterations)

    for raw in getattr(args, "model_option", []) or []:
        if "=" not in raw:
            raise ValueError(f"--model-option must use key=value syntax, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--model-option has an empty key: {raw!r}")
        option_updates[key] = _parse_model_option_value(value)

    if option_updates:
        model_options = dict(env_params.get("model_option_conf", {}) or {})
        if backend == "mjx" and not model_options:
            # Preserve the MJX MyoFullBody/OSL defaults when overriding only one option.
            model_options = _default_mjx_model_options()
        model_options.update(option_updates)
        env_params["model_option_conf"] = model_options

    return env_params


def _apply_start_mode(
    env_params: dict[str, Any],
    start_mode: str,
    traj_index: int,
    traj_start_step: int,
) -> dict[str, Any]:
    env_params = dict(env_params)
    th_params = dict(env_params.get("th_params", {}) or {})

    if start_mode == "fixed":
        th_params["random_start"] = False
        th_params["fixed_start_conf"] = [int(traj_index), int(traj_start_step)]
        th_params["start_from_random_step"] = False
    elif start_mode == "random":
        th_params.pop("fixed_start_conf", None)
        th_params["random_start"] = True
        th_params["start_from_random_step"] = True
    else:
        raise ValueError(f"Unsupported start mode: {start_mode}")

    env_params["th_params"] = th_params
    return env_params


def _make_env(
    config: DictConfig,
    role: str,
    backend: str,
    mjx_backend: str | None,
    num_envs: int,
    start_mode: str,
    traj_index: int,
    traj_start_step: int,
    max_motions: int | None,
    diagnostic_args: argparse.Namespace | None = None,
):
    env_params, task_params = _role_env_params(config, role)
    env_params = _apply_backend_params(env_params, backend, mjx_backend, num_envs)
    env_params = _apply_start_mode(env_params, start_mode, traj_index, traj_start_step)
    env_params = _apply_diagnostic_overrides(env_params, backend, diagnostic_args)
    _apply_max_motions(task_params, max_motions)

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**env_params, **task_params)

    if backend == "mjx" and getattr(env, "th", None) is not None and env.th.is_numpy:
        env.th.to_jax()

    return env


def _action_std_from_config(env: Any, config: DictConfig) -> np.ndarray:
    ppo_cfg = config.experiment.get("ppo_config", {})
    init_std = float(ppo_cfg.get("init_std", config.experiment.get("init_std", 1.0)))
    std = np.full(env.info.action_space.shape, init_std, dtype=np.float32)

    init_std_motors = ppo_cfg.get("init_std_motors", config.experiment.get("init_std_motors", None))
    if init_std_motors is None:
        return std

    muscle_dyntype = int(mujoco.mjtDyn.mjDYN_MUSCLE)
    for dim, actuator_id in enumerate(getattr(env, "_action_indices", [])):
        if int(env._model.actuator_dyntype[int(actuator_id)]) != muscle_dyntype:
            std[dim] = float(init_std_motors)
    return std


def _make_action(
    env: Any,
    mode: str,
    step_no: int,
    rng: np.random.Generator,
    config_std: np.ndarray,
    low_noise_std: float,
    batch: int | None = None,
) -> np.ndarray:
    shape = env.info.action_space.shape
    low = np.asarray(env.info.action_space.low, dtype=np.float32)
    high = np.asarray(env.info.action_space.high, dtype=np.float32)
    midpoint = ((low + high) / 2.0).astype(np.float32)

    if mode == "zero":
        action = np.zeros(shape, dtype=np.float32)
    elif mode == "deterministic":
        # Deterministic policy mode for a freshly initialized zero-mean Gaussian is the action-space midpoint.
        action = midpoint
    elif mode == "low_noise":
        action = rng.normal(loc=midpoint, scale=float(low_noise_std), size=shape).astype(np.float32)
    elif mode == "config_std":
        action = rng.normal(loc=midpoint, scale=config_std, size=shape).astype(np.float32)
    else:
        raise ValueError(f"Unsupported action mode: {mode}")

    # Keep the deterministic pattern stable but non-identical across steps when action limits are asymmetric.
    if mode == "deterministic" and action.size:
        action = action + np.zeros_like(action) * math.sin(float(step_no))

    if batch is not None:
        action = np.broadcast_to(action, (batch, *shape)).copy()
    return action


def _as_bool(value: Any) -> bool:
    arr = np.asarray(jax.device_get(value))
    if arr.shape == ():
        return bool(arr.item())
    return bool(np.any(arr))


def _as_int_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.int64)


def _broadcast_bool_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(jax.device_get(value), dtype=bool)
    if arr.shape == shape:
        return arr
    if arr.shape == ():
        return np.full(shape, bool(arr.item()), dtype=bool)
    return np.broadcast_to(arr, shape).astype(bool, copy=False)


def _as_numpy_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(jax.device_get(value))
    except Exception:
        return None
    if arr.dtype == np.dtype("O"):
        return None
    try:
        np.isfinite(arr)
    except TypeError:
        return None
    return arr


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return getattr(obj, name)
    except Exception:
        return default


def _first_bad_index(mask: np.ndarray) -> list[int]:
    bad = np.argwhere(mask)
    if bad.size == 0:
        return []
    return [int(v) for v in bad[0].tolist()]


def _audit_array(stage: str, field: str, value: Any, num_envs: int) -> dict[str, Any] | None:
    arr = _as_numpy_array(value)
    if arr is None or arr.size == 0:
        return None

    finite_mask = np.isfinite(arr)
    bad_mask = ~finite_mask
    nonfinite_count = int(np.sum(bad_mask))
    if nonfinite_count == 0:
        return None

    first_index = _first_bad_index(bad_mask)
    first_env = first_index[0] if first_index and arr.shape and arr.shape[0] == num_envs else None
    finite_values = arr[finite_mask]
    finite_min = None
    finite_max = None
    if finite_values.size and not np.iscomplexobj(finite_values):
        finite_min = float(np.min(finite_values))
        finite_max = float(np.max(finite_values))

    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
        nan_count = int(np.sum(np.isnan(arr)))
        if not np.iscomplexobj(arr):
            posinf_count = int(np.sum(np.isposinf(arr)))
            neginf_count = int(np.sum(np.isneginf(arr)))

    return {
        "stage": stage,
        "field": field,
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "nonfinite_count": nonfinite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "first_index": first_index,
        "first_env": first_env,
        "finite_min": finite_min,
        "finite_max": finite_max,
    }


def _tree_path_to_string(path: tuple[Any, ...]) -> str:
    parts = []
    for item in path:
        name = getattr(item, "name", None)
        key = getattr(item, "key", None)
        idx = getattr(item, "idx", None)
        if name is not None:
            parts.append(str(name))
        elif key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(item))
    return ".".join(parts)


def _audit_tree_leaves(
    stage: str,
    prefix: str,
    obj: Any,
    num_envs: int,
    max_findings: int,
) -> list[dict[str, Any]]:
    findings = []
    if max_findings <= 0:
        return findings
    try:
        leaves, _ = jax.tree_util.tree_flatten_with_path(obj)
    except Exception:
        return findings

    for path, leaf in leaves:
        finding = _audit_array(stage, f"{prefix}.{_tree_path_to_string(path)}", leaf, num_envs)
        if finding is not None:
            findings.append(finding)
            if len(findings) >= max_findings:
                break
    return findings


def _safe_raw_mjx_observation(env: Any, mjx_state: Any) -> Any | None:
    def _obs_one(data, carry):
        result = env._mjx_create_observation(env.sys, data, carry)
        if isinstance(result, tuple):
            return result[0]
        return result

    try:
        qpos = getattr(mjx_state.data, "qpos", None)
        if qpos is not None and hasattr(qpos, "shape") and len(qpos.shape) >= 2:
            return jax.vmap(_obs_one)(mjx_state.data, mjx_state.additional_carry)
        return _obs_one(mjx_state.data, mjx_state.additional_carry)
    except Exception:
        return None


def _audit_mjx_state(
    env: Any,
    mjx_state: Any,
    stage: str,
    num_envs: int,
    max_findings: int,
    include_impl: bool = True,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if max_findings <= 0:
        return findings

    state_fields = {
        "state.observation": getattr(mjx_state, "observation", None),
        "state.reward": getattr(mjx_state, "reward", None),
        "state.absorbing": getattr(mjx_state, "absorbing", None),
        "state.done": getattr(mjx_state, "done", None),
    }
    for name, value in state_fields.items():
        finding = _audit_array(stage, name, value, num_envs)
        if finding is not None:
            findings.append(finding)
            if len(findings) >= max_findings:
                return findings

    data = getattr(mjx_state, "data", None)
    if data is None:
        return findings

    for field in AUDIT_DATA_FIELDS:
        value = _safe_getattr(data, field)
        if value is None:
            continue
        finding = _audit_array(stage, f"data.{field}", value, num_envs)
        if finding is not None:
            findings.append(finding)
            if len(findings) >= max_findings:
                return findings

    if len(findings) < max_findings:
        finding = _audit_array(stage, "raw_observation", _safe_raw_mjx_observation(env, mjx_state), num_envs)
        if finding is not None:
            findings.append(finding)
            if len(findings) >= max_findings:
                return findings

    if include_impl and len(findings) < max_findings:
        impl = getattr(data, "_impl", None)
        if impl is not None:
            findings.extend(
                _audit_tree_leaves(
                    stage,
                    "data._impl",
                    impl,
                    num_envs,
                    max_findings=max_findings - len(findings),
                )
            )

    return findings


def _print_audit_findings(findings: list[dict[str, Any]] | None, max_lines: int = 12) -> None:
    if not findings:
        print("     audit: no non-finite fields observed in audited stages", flush=True)
        return
    first = findings[0]
    print(
        "     audit: first_nonfinite "
        f"stage={first['stage']} field={first['field']} env={first['first_env']} "
        f"index={first['first_index']} count={first['nonfinite_count']} "
        f"nan={first['nan_count']} +inf={first['posinf_count']} -inf={first['neginf_count']} "
        f"finite_range=[{first['finite_min']}, {first['finite_max']}]",
        flush=True,
    )
    for finding in findings[1:max_lines]:
        print(
            "            "
            f"stage={finding['stage']} field={finding['field']} env={finding['first_env']} "
            f"index={finding['first_index']} count={finding['nonfinite_count']} "
            f"nan={finding['nan_count']} +inf={finding['posinf_count']} -inf={finding['neginf_count']}",
            flush=True,
        )
    if len(findings) > max_lines:
        print(f"            ... {len(findings) - max_lines} more non-finite field findings", flush=True)


def _env0_array(value: Any, num_envs: int) -> np.ndarray | None:
    arr = _as_numpy_array(value)
    if arr is None:
        return None
    if arr.shape and arr.shape[0] == num_envs:
        return arr[0]
    return arr


def _compare_cpu_mjx_reset(cpu_env: Any, mjx_state: Any, seed: int, num_envs: int) -> dict[str, Any]:
    try:
        cpu_env.reset(jax.random.PRNGKey(seed))
    except Exception as exc:
        return {"error": f"cpu_reset_failed: {type(exc).__name__}: {exc}"}

    field_summaries: dict[str, Any] = {}
    shape_mismatches: dict[str, Any] = {}
    worst_field = None
    worst_index: list[int] = []
    worst_abs = -1.0

    for field in RESET_PARITY_FIELDS:
        cpu_value = _safe_getattr(cpu_env._data, field)
        mjx_value = _safe_getattr(mjx_state.data, field)
        if cpu_value is None or mjx_value is None:
            continue
        cpu_arr = _as_numpy_array(cpu_value)
        mjx_arr = _env0_array(mjx_value, num_envs)
        if cpu_arr is None or mjx_arr is None:
            continue
        if cpu_arr.shape != mjx_arr.shape:
            shape_mismatches[field] = {
                "cpu_shape": [int(dim) for dim in cpu_arr.shape],
                "mjx_shape": [int(dim) for dim in mjx_arr.shape],
            }
            continue
        if cpu_arr.size == 0:
            max_abs = 0.0
            max_index = []
        else:
            diff = np.abs(cpu_arr - mjx_arr)
            max_abs = float(np.nanmax(diff))
            max_index = [int(v) for v in np.unravel_index(int(np.nanargmax(diff)), diff.shape)]
        field_summaries[field] = {"max_abs": max_abs, "max_index": max_index}
        if max_abs > worst_abs:
            worst_abs = max_abs
            worst_field = field
            worst_index = max_index

    return {
        "worst_field": worst_field,
        "max_abs": None if worst_abs < 0.0 else worst_abs,
        "max_index": worst_index,
        "fields": field_summaries,
        "shape_mismatches": shape_mismatches,
    }


def _print_reset_parity(reset_parity: dict[str, Any] | None) -> None:
    if not reset_parity:
        return
    if "error" in reset_parity:
        print(f"     reset_parity: {reset_parity['error']}", flush=True)
        return
    print(
        f"     reset_parity: worst_field={reset_parity.get('worst_field')} "
        f"max_abs={reset_parity.get('max_abs')} index={reset_parity.get('max_index')} "
        f"shape_mismatches={reset_parity.get('shape_mismatches', {})}",
        flush=True,
    )


def _safe_goal_done_cpu(env: Any, carry: Any) -> bool:
    try:
        return bool(env._goal.is_done(env, env._model, env._data, carry, np))
    except Exception:
        return False


def _safe_goal_done_mjx(env: Any, mjx_state: Any) -> np.ndarray:
    done_shape = np.asarray(jax.device_get(mjx_state.done)).shape
    try:
        goal_done = env._goal.mjx_is_done(env, env._model, mjx_state.data, mjx_state.additional_carry, jnp)
        return _broadcast_bool_array(goal_done, done_shape)
    except Exception:
        return np.zeros(done_shape, dtype=bool)


def _safe_traj_end_cpu(env: Any, carry: Any) -> bool:
    try:
        return bool(env.th.reached_trajectory_end(carry.traj_state, np)) if getattr(env, "th", None) else False
    except Exception:
        return False


def _safe_traj_end_mjx(env: Any, mjx_state: Any) -> np.ndarray:
    done_shape = np.asarray(jax.device_get(mjx_state.done)).shape
    try:
        if getattr(env, "th", None) is None:
            return np.zeros(done_shape, dtype=bool)
        traj_end = env.th.reached_trajectory_end(mjx_state.additional_carry.traj_state, jnp)
        return _broadcast_bool_array(traj_end, done_shape)
    except Exception:
        return np.zeros(done_shape, dtype=bool)


def _cpu_done_cause(
    env: Any,
    obs: np.ndarray,
    absorbing: bool,
    done: bool,
    episode_step: int,
) -> str | None:
    if np.any(np.isnan(obs)):
        return "nan_observation"
    if not done:
        return None

    carry = env._additional_carry
    causes = []
    if bool(absorbing):
        causes.append("absorbing_terminal")
    if _safe_traj_end_cpu(env, carry):
        causes.append("trajectory_end")
    if episode_step >= int(env.info.horizon):
        causes.append("horizon")
    if _safe_goal_done_cpu(env, carry):
        causes.append("goal_done")
    return "+".join(causes) if causes else "done_unknown"


def _format_traj_state(carry: Any, env_idx: int | None = None) -> str:
    traj_state = getattr(carry, "traj_state", None)
    if traj_state is None or not hasattr(traj_state, "traj_no"):
        return "traj=n/a"

    def _item(value):
        arr = np.asarray(jax.device_get(value))
        if env_idx is not None and arr.shape:
            return int(arr[env_idx])
        return int(arr.item()) if arr.shape == () else arr.tolist()

    try:
        traj_no = _item(traj_state.traj_no)
        substep = _item(traj_state.subtraj_step_no)
        init = _item(traj_state.subtraj_step_no_init)
        return f"traj={traj_no}, subtraj_step={substep}, init_step={init}"
    except Exception:
        return "traj=unavailable"


def _run_cpu_rollout(
    env: Any,
    config: DictConfig,
    role: str,
    start_mode: str,
    action_mode: str,
    steps: int,
    episodes: int,
    seed: int,
    low_noise_std: float,
    verbose: bool,
) -> RolloutSummary:
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    config_std = _action_std_from_config(env, config)
    cause_counts: Counter[str] = Counter()
    episode_lengths: list[int] = []
    first_done_step = None
    first_done_cause = None
    done_count = 0

    for episode in range(episodes):
        key, reset_key = jax.random.split(key)
        obs = np.asarray(env.reset(reset_key))
        if verbose:
            print(f"    reset episode={episode + 1}: {_format_traj_state(env._additional_carry)}", flush=True)

        for episode_step in range(1, steps + 1):
            action = _make_action(env, action_mode, episode_step, rng, config_std, low_noise_std)
            obs, _reward, absorbing, done, info = env.step(action)
            cause = _cpu_done_cause(env, np.asarray(obs), bool(absorbing), bool(done), episode_step)

            if cause is not None:
                done_count += 1
                cause_counts[cause] += 1
                episode_lengths.append(episode_step)
                if first_done_step is None:
                    first_done_step = episode_step
                    first_done_cause = cause
                if verbose:
                    info_part = ", ".join(f"{k}={v}" for k, v in info.items() if k in ("traj_no", "subtraj_step_no"))
                    print(
                        f"    done episode={episode + 1} step={episode_step}: cause={cause}; "
                        f"{_format_traj_state(env._additional_carry)}; {info_part}",
                        flush=True,
                    )
                break
        else:
            episode_lengths.append(steps)

    return RolloutSummary(
        role=role,
        backend="cpu",
        start_mode=start_mode,
        action_mode=action_mode,
        num_envs=1,
        steps_requested=steps,
        episodes_requested=episodes,
        first_done_step=first_done_step,
        first_done_env=0 if first_done_step is not None else None,
        first_done_cause=first_done_cause,
        done_count=done_count,
        cause_counts=dict(cause_counts),
        episode_lengths=episode_lengths,
    )


def _mjx_done_causes(env: Any, mjx_state: Any, log_state: LogEnvState) -> list[str]:
    done = np.asarray(jax.device_get(mjx_state.done), dtype=bool)
    absorbing = np.asarray(jax.device_get(mjx_state.absorbing), dtype=bool)
    traj_end = _safe_traj_end_mjx(env, mjx_state)
    goal_done = _safe_goal_done_mjx(env, mjx_state)
    returned_lengths = np.asarray(jax.device_get(log_state.metrics.returned_episode_lengths), dtype=np.int64)

    nan_state = (
        _nonfinite_env_mask(mjx_state.data.qpos, done.shape)
        | _nonfinite_env_mask(mjx_state.data.qvel, done.shape)
        | _nonfinite_env_mask(mjx_state.observation, done.shape)
        | _nonfinite_env_mask(_safe_raw_mjx_observation(env, mjx_state), done.shape)
    )
    horizon = returned_lengths >= int(env.info.horizon)

    causes = []
    for env_idx, is_done in enumerate(done):
        if not is_done and not nan_state[env_idx]:
            causes.append("")
            continue

        parts = []
        if nan_state[env_idx]:
            parts.append("nan_observation")
        if absorbing[env_idx]:
            parts.append("absorbing_terminal")
        if traj_end[env_idx]:
            parts.append("trajectory_end")
        if horizon[env_idx]:
            parts.append("horizon")
        if goal_done[env_idx]:
            parts.append("goal_done")
        if is_done and not parts:
            parts.append("done_unknown")
        causes.append("+".join(parts))
    return causes


def _nonfinite_env_mask(value: Any, done_shape: tuple[int, ...]) -> np.ndarray:
    arr = _as_numpy_array(value)
    if arr is None:
        return np.zeros(done_shape, dtype=bool)
    bad = ~np.isfinite(arr)
    if bad.shape == done_shape:
        return bad.astype(bool)
    if bad.shape == ():
        return np.full(done_shape, bool(bad.item()), dtype=bool)
    if done_shape and bad.shape and bad.shape[0] == done_shape[0]:
        return np.any(bad.reshape((done_shape[0], -1)), axis=1)
    return np.full(done_shape, bool(np.any(bad)), dtype=bool)


def _mjx_base_done_causes(env: Any, mjx_state: Any) -> list[str]:
    done = np.asarray(jax.device_get(mjx_state.done), dtype=bool)
    if done.shape == ():
        done = done.reshape((1,))
    absorbing = _broadcast_bool_array(mjx_state.absorbing, done.shape)
    traj_end = _safe_traj_end_mjx(env, mjx_state)
    goal_done = _safe_goal_done_mjx(env, mjx_state)
    nan_state = (
        _nonfinite_env_mask(mjx_state.data.qpos, done.shape)
        | _nonfinite_env_mask(mjx_state.data.qvel, done.shape)
        | _nonfinite_env_mask(mjx_state.observation, done.shape)
        | _nonfinite_env_mask(_safe_raw_mjx_observation(env, mjx_state), done.shape)
    )
    cur_step = getattr(mjx_state.additional_carry, "cur_step_in_episode", None)
    if cur_step is None:
        horizon = np.zeros(done.shape, dtype=bool)
    else:
        horizon = _as_int_array(cur_step) >= int(env.info.horizon)
        if horizon.shape != done.shape:
            horizon = _broadcast_bool_array(horizon, done.shape)

    causes = []
    for env_idx, is_done in enumerate(done):
        if not is_done and not nan_state[env_idx]:
            causes.append("")
            continue
        parts = []
        if nan_state[env_idx]:
            parts.append("nan_observation")
        if absorbing[env_idx]:
            parts.append("absorbing_terminal")
        if traj_end[env_idx]:
            parts.append("trajectory_end")
        if horizon[env_idx]:
            parts.append("horizon")
        if goal_done[env_idx]:
            parts.append("goal_done")
        if is_done and not parts:
            parts.append("done_unknown")
        causes.append("+".join(parts))
    return causes


def _run_raw_mjx_probe(
    base_env: Any,
    config: DictConfig,
    role: str,
    start_mode: str,
    action_mode: str,
    steps: int,
    seed: int,
    num_envs: int,
    low_noise_std: float,
    use_jit: bool,
    verbose: bool,
    nan_audit: bool = False,
    audit_max_fields: int = 16,
    compare_cpu_env: Any | None = None,
) -> RolloutSummary:
    rng = np.random.default_rng(seed)
    config_std = _action_std_from_config(base_env, config)

    reset_fn = jax.vmap(base_env.mjx_reset, in_axes=(0,))
    step_fn = jax.vmap(base_env.mjx_step, in_axes=(0, 0))
    if use_jit:
        reset_fn = jax.jit(reset_fn)
        step_fn = jax.jit(step_fn)

    reset_keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    mjx_state = reset_fn(reset_keys)

    audit_findings: list[dict[str, Any]] = []
    reset_parity = None
    if compare_cpu_env is not None:
        reset_parity = _compare_cpu_mjx_reset(compare_cpu_env, mjx_state, seed, num_envs)
    if nan_audit:
        audit_findings.extend(
            _audit_mjx_state(
                base_env,
                mjx_state,
                "post_reset",
                num_envs,
                max_findings=max(0, audit_max_fields - len(audit_findings)),
            )
        )

    if verbose:
        carry = mjx_state.additional_carry
        for env_idx in range(min(num_envs, 8)):
            print(f"    raw reset env={env_idx}: {_format_traj_state(carry, env_idx)}", flush=True)

    cause_counts: Counter[str] = Counter()
    episode_lengths: list[int] = []
    first_done_step = None
    first_done_env = None
    first_done_cause = None
    done_count = 0

    for step_no in range(1, steps + 1):
        action = _make_action(
            base_env,
            action_mode,
            step_no,
            rng,
            config_std,
            low_noise_std,
            batch=num_envs,
        )
        mjx_state = step_fn(mjx_state, jnp.asarray(action))

        if nan_audit and len(audit_findings) < audit_max_fields:
            audit_findings.extend(
                _audit_mjx_state(
                    base_env,
                    mjx_state,
                    f"base_step_{step_no}",
                    num_envs,
                    max_findings=max(0, audit_max_fields - len(audit_findings)),
                )
            )

        done_np = np.asarray(jax.device_get(mjx_state.done), dtype=bool)
        causes = _mjx_base_done_causes(base_env, mjx_state)
        for env_idx, is_done in enumerate(done_np):
            cause = causes[env_idx]
            if not is_done and not cause:
                continue
            done_count += 1
            cause = cause or "done_unknown"
            cause_counts[cause] += 1
            episode_lengths.append(step_no)
            if first_done_step is None:
                first_done_step = step_no
                first_done_env = env_idx
                first_done_cause = cause
            if verbose:
                print(
                    f"    raw done env={env_idx} step={step_no} cause={cause}; "
                    f"{_format_traj_state(mjx_state.additional_carry, env_idx)}",
                    flush=True,
                )

    if not episode_lengths:
        episode_lengths = [steps] * num_envs

    return RolloutSummary(
        role=role,
        backend=f"mjx/{getattr(base_env, 'mjx_backend', 'unknown')}/raw",
        start_mode=start_mode,
        action_mode=action_mode,
        num_envs=num_envs,
        steps_requested=steps,
        episodes_requested=num_envs,
        first_done_step=first_done_step,
        first_done_env=first_done_env,
        first_done_cause=first_done_cause,
        done_count=done_count,
        cause_counts=dict(cause_counts),
        episode_lengths=episode_lengths,
        audit_findings=audit_findings if nan_audit else None,
        reset_parity=reset_parity,
    )


def _run_mjx_rollout(
    base_env: Any,
    config: DictConfig,
    role: str,
    start_mode: str,
    action_mode: str,
    steps: int,
    seed: int,
    num_envs: int,
    low_noise_std: float,
    use_jit: bool,
    verbose: bool,
    nan_audit: bool = False,
    audit_max_fields: int = 16,
    compare_cpu_env: Any | None = None,
) -> RolloutSummary:
    rng = np.random.default_rng(seed)
    config_std = _action_std_from_config(base_env, config)
    wrapped = wrap_env(base_env, config.experiment)

    reset_fn = wrapped.reset
    step_fn = wrapped.step_with_transition
    if use_jit:
        reset_fn = jax.jit(reset_fn)
        step_fn = jax.jit(step_fn)

    reset_keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    obs, env_state = reset_fn(reset_keys)
    del obs

    cause_counts: Counter[str] = Counter()
    episode_lengths: list[int] = []
    first_done_step = None
    first_done_env = None
    first_done_cause = None
    done_count = 0
    accounting_errors: Counter[str] = Counter()

    prev_log_state = env_state.find(LogEnvState)
    prev_done_count = _as_int_array(prev_log_state.env_state.info.get("AutoResetWrapper_done_count", np.zeros(num_envs)))

    if verbose:
        mjx_state, _ = unwrap_to_mjx(prev_log_state.env_state)
        carry = mjx_state.additional_carry
        for env_idx in range(min(num_envs, 8)):
            print(f"    reset env={env_idx}: {_format_traj_state(carry, env_idx)}", flush=True)

    final_autoreset_done_count = prev_done_count
    audit_findings: list[dict[str, Any]] = []
    reset_parity = None
    if compare_cpu_env is not None or nan_audit:
        reset_mjx_state, _ = unwrap_to_mjx(prev_log_state.env_state)
        if compare_cpu_env is not None:
            reset_parity = _compare_cpu_mjx_reset(compare_cpu_env, reset_mjx_state, seed, num_envs)
        if nan_audit:
            audit_findings.extend(
                _audit_mjx_state(
                    base_env,
                    reset_mjx_state,
                    "post_reset",
                    num_envs,
                    max_findings=max(0, audit_max_fields - len(audit_findings)),
                )
            )

    for step_no in range(1, steps + 1):
        action = _make_action(
            base_env,
            action_mode,
            step_no,
            rng,
            config_std,
            low_noise_std,
            batch=num_envs,
        )
        obs, _reward, _absorbing, done, info, env_state, transition_state, _transition_obs = step_fn(
            env_state, jnp.asarray(action)
        )
        del obs

        done_np = np.asarray(jax.device_get(done), dtype=bool)
        transition_log_state = transition_state.find(LogEnvState)
        transition_mjx_state, _ = unwrap_to_mjx(transition_log_state.env_state)
        post_log_state = env_state.find(LogEnvState)
        post_mjx_state, _ = unwrap_to_mjx(post_log_state.env_state)

        causes = _mjx_done_causes(base_env, transition_mjx_state, transition_log_state)
        if nan_audit and len(audit_findings) < audit_max_fields:
            audit_findings.extend(
                _audit_mjx_state(
                    base_env,
                    transition_mjx_state,
                    f"wrapped_transition_{step_no}",
                    num_envs,
                    max_findings=max(0, audit_max_fields - len(audit_findings)),
                )
            )

        done_count_arr = _as_int_array(info.get("AutoResetWrapper_done_count", prev_done_count))
        done_count_delta = done_count_arr - prev_done_count
        final_autoreset_done_count = done_count_arr
        if np.any(done_count_delta != done_np.astype(np.int64)):
            accounting_errors["done_count_delta_mismatch"] += int(
                np.sum(done_count_delta != done_np.astype(np.int64))
            )

        post_done = np.asarray(jax.device_get(post_mjx_state.done), dtype=bool)
        transition_done = np.asarray(jax.device_get(transition_mjx_state.done), dtype=bool)
        if np.any(post_done & done_np):
            accounting_errors["post_reset_done_not_cleared"] += int(np.sum(post_done & done_np))
        if np.any(transition_done != done_np):
            accounting_errors["transition_done_mismatch"] += int(np.sum(transition_done != done_np))

        returned_lengths = np.asarray(
            jax.device_get(transition_log_state.metrics.returned_episode_lengths), dtype=np.int64
        )
        for env_idx, is_done in enumerate(done_np):
            if not is_done:
                continue
            done_count += 1
            cause = causes[env_idx] or "done_unknown"
            cause_counts[cause] += 1
            episode_lengths.append(int(returned_lengths[env_idx]))
            if first_done_step is None:
                first_done_step = step_no
                first_done_env = env_idx
                first_done_cause = cause
            if verbose:
                print(
                    f"    done env={env_idx} scan_step={step_no} episode_len={returned_lengths[env_idx]} "
                    f"cause={cause}; {_format_traj_state(transition_mjx_state.additional_carry, env_idx)}",
                    flush=True,
                )

        prev_done_count = done_count_arr

    reset_accounting_ok = not accounting_errors
    return RolloutSummary(
        role=role,
        backend=f"mjx/{getattr(base_env, 'mjx_backend', 'unknown')}",
        start_mode=start_mode,
        action_mode=action_mode,
        num_envs=num_envs,
        steps_requested=steps,
        episodes_requested=num_envs,
        first_done_step=first_done_step,
        first_done_env=first_done_env,
        first_done_cause=first_done_cause,
        done_count=done_count,
        cause_counts=dict(cause_counts),
        episode_lengths=episode_lengths,
        reset_accounting_ok=reset_accounting_ok,
        reset_accounting_errors=dict(accounting_errors),
        final_autoreset_done_count=final_autoreset_done_count.astype(int).tolist(),
        audit_findings=audit_findings if nan_audit else None,
        reset_parity=reset_parity,
    )


def _print_summary(summary: RolloutSummary, json_lines: bool) -> None:
    row = dataclasses.asdict(summary)
    if json_lines:
        print(json.dumps(row, sort_keys=True), flush=True)
        return

    label = f"{summary.role} {summary.backend} start={summary.start_mode} action={summary.action_mode}"
    if summary.skipped:
        print(f"[skip] {label}: {summary.skipped}", flush=True)
        return
    if summary.error:
        print(f"[error] {label}: {summary.error}", flush=True)
        return

    lengths = summary.episode_lengths
    mean_len = float(np.mean(lengths)) if lengths else float("nan")
    first = (
        f"first_done_step={summary.first_done_step} env={summary.first_done_env} "
        f"cause={summary.first_done_cause}"
        if summary.first_done_step is not None
        else "first_done_step=None"
    )
    print(
        f"[ok] {label}: {first}; done_count={summary.done_count}; "
        f"mean_len={mean_len:.2f}; causes={summary.cause_counts}",
        flush=True,
    )
    if summary.reset_accounting_ok is not None:
        print(
            f"     reset_accounting_ok={summary.reset_accounting_ok}; "
            f"errors={summary.reset_accounting_errors}; "
            f"final_done_count={summary.final_autoreset_done_count}",
            flush=True,
        )
    _print_reset_parity(summary.reset_parity)
    if summary.audit_findings is not None:
        _print_audit_findings(summary.audit_findings)


def _warp_unavailable_reason() -> str | None:
    try:
        import warp
    except Exception as exc:
        return f"warp import failed: {exc}"
    try:
        if not warp.is_cuda_available():
            return "warp CUDA is not available"
    except Exception as exc:
        return f"warp CUDA check failed: {exc}"
    return None


def _make_error_summary(
    role: str,
    backend: str,
    start_mode: str,
    action_mode: str,
    steps: int,
    episodes: int,
    error: str | None = None,
    skipped: str | None = None,
) -> RolloutSummary:
    return RolloutSummary(
        role=role,
        backend=backend,
        start_mode=start_mode,
        action_mode=action_mode,
        num_envs=0,
        steps_requested=steps,
        episodes_requested=episodes,
        first_done_step=None,
        first_done_env=None,
        first_done_cause=None,
        done_count=0,
        cause_counts={},
        episode_lengths=[],
        skipped=skipped,
        error=error,
    )


def _run_one(args: argparse.Namespace, config: DictConfig, role: str, backend: str, start_mode: str, action_mode: str):
    if backend == "mjx" and args.mjx_backend == "warp":
        reason = _warp_unavailable_reason()
        if reason is not None and not args.force_warp:
            return _make_error_summary(
                role,
                "mjx/warp",
                start_mode,
                action_mode,
                args.steps,
                args.num_envs,
                skipped=f"{reason}; pass --force-warp to try anyway",
            )

    try:
        # ImitationFactory samples max_motions via Python's random module. Reset
        # it per diagnostic cell so CPU/MJX and action-mode comparisons use the
        # same capped motion instead of advancing to a different trajectory.
        random.seed(args.seed)
        env = _make_env(
            config=config,
            role=role,
            backend=backend,
            mjx_backend=args.mjx_backend,
            num_envs=args.num_envs,
            start_mode=start_mode,
            traj_index=args.traj_index,
            traj_start_step=args.traj_start_step,
            max_motions=args.max_motions,
            diagnostic_args=args,
        )
        if backend == "cpu":
            return _run_cpu_rollout(
                env=env,
                config=config,
                role=role,
                start_mode=start_mode,
                action_mode=action_mode,
                steps=args.steps,
                episodes=args.episodes,
                seed=args.seed,
                low_noise_std=args.low_noise_std,
                verbose=args.verbose,
            )
        compare_cpu_env = None
        if args.compare_cpu_reset:
            random.seed(args.seed)
            compare_cpu_env = _make_env(
                config=config,
                role=role,
                backend="cpu",
                mjx_backend=None,
                num_envs=1,
                start_mode=start_mode,
                traj_index=args.traj_index,
                traj_start_step=args.traj_start_step,
                max_motions=args.max_motions,
                diagnostic_args=args,
            )
        if args.raw_mjx_probe:
            return _run_raw_mjx_probe(
                base_env=env,
                config=config,
                role=role,
                start_mode=start_mode,
                action_mode=action_mode,
                steps=args.steps,
                seed=args.seed,
                num_envs=args.num_envs,
                low_noise_std=args.low_noise_std,
                use_jit=not args.no_jit,
                verbose=args.verbose,
                nan_audit=args.nan_audit,
                audit_max_fields=args.audit_max_fields,
                compare_cpu_env=compare_cpu_env,
            )
        return _run_mjx_rollout(
            base_env=env,
            config=config,
            role=role,
            start_mode=start_mode,
            action_mode=action_mode,
            steps=args.steps,
            seed=args.seed,
            num_envs=args.num_envs,
            low_noise_std=args.low_noise_std,
            use_jit=not args.no_jit,
            verbose=args.verbose,
            nan_audit=args.nan_audit,
            audit_max_fields=args.audit_max_fields,
            compare_cpu_env=compare_cpu_env,
        )
    except Exception as exc:
        if args.raise_errors:
            raise
        if args.verbose:
            traceback.print_exc()
        return _make_error_summary(
            role,
            backend if backend == "cpu" else f"mjx/{args.mjx_backend}",
            start_mode,
            action_mode,
            args.steps,
            args.episodes if backend == "cpu" else args.num_envs,
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone OSL KA episode-length diagnostics outside PPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="fullbody/conf_myoleg80_osl_ka.yaml", help="Hydra config YAML path.")
    parser.add_argument("--roles", default="train,validation", help="Comma list: train,validation,all.")
    parser.add_argument("--backends", default="cpu,mjx", help="Comma list: cpu,mjx,all.")
    parser.add_argument("--start-modes", default="fixed,random", help="Comma list: fixed,random,all.")
    parser.add_argument(
        "--action-modes",
        default="zero,deterministic,low_noise,config_std",
        help="Comma list: zero,deterministic,low_noise,config_std,all.",
    )
    parser.add_argument("--steps", type=int, default=64, help="Max rollout steps per diagnostic run.")
    parser.add_argument("--episodes", type=int, default=4, help="CPU episodes per diagnostic run.")
    parser.add_argument("--num-envs", type=int, default=4, help="MJX vectorized env count for diagnostics.")
    parser.add_argument("--seed", type=int, default=0, help="Base PRNG seed.")
    parser.add_argument("--traj-index", type=int, default=0, help="Fixed-start trajectory index.")
    parser.add_argument("--traj-start-step", type=int, default=0, help="Fixed-start trajectory frame.")
    parser.add_argument("--max-motions", type=int, default=None, help="Optional dataset load cap for quick diagnosis.")
    parser.add_argument("--mjx-backend", default=None, help="Override MJX backend, e.g. warp or jax.")
    parser.add_argument("--force-warp", action="store_true", help="Try Warp even if the CUDA availability check fails.")
    parser.add_argument("--low-noise-std", type=float, default=0.02, help="Std for low_noise actions.")
    parser.add_argument("--no-jit", action="store_true", help="Disable JIT around MJX diagnostic reset/step.")
    parser.add_argument(
        "--raw-mjx-probe",
        action="store_true",
        help="For MJX backends, bypass wrappers/autoreset and step base_env.mjx_step directly.",
    )
    parser.add_argument(
        "--nan-audit",
        action="store_true",
        help="Report the first non-finite MJX state/data/raw-observation fields by rollout stage.",
    )
    parser.add_argument(
        "--audit-max-fields",
        type=int,
        default=16,
        help="Maximum non-finite field findings to retain and print per diagnostic run.",
    )
    parser.add_argument(
        "--compare-cpu-reset",
        action="store_true",
        help="For MJX runs, compare env0 fixed reset fields against a CPU MuJoCo reset.",
    )
    parser.add_argument("--timestep", type=float, default=None, help="Diagnostic-only env timestep override.")
    parser.add_argument("--n-substeps", type=int, default=None, help="Diagnostic-only env n_substeps override.")
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=None,
        help="Diagnostic-only model option override for option.iterations.",
    )
    parser.add_argument(
        "--ls-iterations",
        type=int,
        default=None,
        help="Diagnostic-only model option override for option.ls_iterations.",
    )
    parser.add_argument(
        "--model-option",
        action="append",
        default=[],
        help="Diagnostic-only MuJoCo option override as key=value. Values are JSON-parsed when possible.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON lines instead of human-readable summaries.")
    parser.add_argument("--verbose", action="store_true", help="Print reset and done boundaries.")
    parser.add_argument("--raise-errors", action="store_true", help="Raise the first environment error.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = _load_hydra_config(args.config)
    if args.mjx_backend is None:
        args.mjx_backend = str(config.experiment.env_params.get("mjx_backend", "jax"))

    roles = _parse_csv(args.roles, ROLE_CHOICES, "roles")
    backends = _parse_csv(args.backends, BACKEND_CHOICES, "backends")
    start_modes = _parse_csv(args.start_modes, START_MODE_CHOICES, "start modes")
    action_modes = _parse_csv(args.action_modes, ACTION_MODE_CHOICES, "action modes")

    # Keep config local and mutable for wrap_env/diagnostic overrides without touching the YAML.
    with open_dict(config.experiment):
        config.experiment.num_envs = int(args.num_envs)
        config.experiment.normalize_env = bool(config.experiment.get("normalize_env", False))
        config.experiment.gamma = float(config.experiment.get("gamma", 0.99))
        config.experiment.len_obs_history = int(config.experiment.get("len_obs_history", 1))

    if not args.json:
        print(f"config={args.config}", flush=True)
        print(
            f"roles={roles} backends={backends} start_modes={start_modes} action_modes={action_modes} "
            f"steps={args.steps} episodes={args.episodes} num_envs={args.num_envs} "
            f"mjx_backend={args.mjx_backend} raw_mjx_probe={args.raw_mjx_probe} "
            f"nan_audit={args.nan_audit} compare_cpu_reset={args.compare_cpu_reset}",
            flush=True,
        )

    had_error = False
    for role in roles:
        for backend in backends:
            for start_mode in start_modes:
                for action_mode in action_modes:
                    summary = _run_one(args, config, role, backend, start_mode, action_mode)
                    had_error = had_error or bool(summary.error)
                    _print_summary(summary, args.json)

    return 1 if had_error else 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.exit(main())
