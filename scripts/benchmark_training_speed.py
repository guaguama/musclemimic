#!/usr/bin/env python3
"""Benchmark PPO training throughput for the two regular OSL configurations.

The public command is an orchestrator.  Every measurement is made by a fresh
worker process so compilations, allocator state, and failures cannot leak from
one repeat into another.  Workers compose the production Hydra configs and use
the production environment, PPO configuration, and training-function builders.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FULLBODY_CONFIG_DIR = REPO_ROOT / "fullbody"
DEFAULT_OUTPUT = REPO_ROOT / "scripts" / "training_speed_results.json"
DEFAULT_TIMESTEPS = 3_276_800
NUM_STEPS = 80
DEFAULT_ENV_COUNTS = (512, 1024, 2048, 4096)
TMP_PREFIX = "musclemimic-benchmark-"
MARKER_NAME = ".musclemimic-benchmark-owner"


@dataclass(frozen=True)
class BenchmarkCase:
    model: str
    config_name: str
    env_count: int
    trajectory: str


CASE_DEFINITIONS = {
    "fullbody": {
        "config_name": "conf_osl_fullbody",
        "trajectory": (
            "/home/dmma/src/00_AMBER/00_projbackflip/clipper/outputs/"
            "musclemimic/osl_fullbody/WalkingStraightForwards03_poses_crop0-521_ankle-7.npz"
        ),
        "env_name": "MjxOSLFullBody",
    },
    "myoleg80": {
        "config_name": "conf_myoleg80_osl_ka",
        "trajectory": (
            "/home/dmma/src/00_AMBER/00_projbackflip/clipper/outputs/"
            "musclemimic/osl_ka/WalkingStraightForward03_poses_crop0-562_ankle-3.npz"
        ),
        "env_name": "MjxMyoLeg80_OSL_KA",
    },
}


def parse_env_counts(value: str) -> list[int]:
    """Parse a comma-separated, unique list of positive environment counts."""
    try:
        counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("environment counts must be integers") from exc
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("environment counts must be positive")
    return list(dict.fromkeys(counts))


def build_matrix(env_counts: list[int], models: tuple[str, ...] = ("fullbody", "myoleg80")) -> list[BenchmarkCase]:
    """Build required cases, reusing full-body 2048 for model comparison."""
    cases = []
    if "fullbody" in models:
        cases.extend(
            BenchmarkCase("fullbody", CASE_DEFINITIONS["fullbody"]["config_name"], count,
                          CASE_DEFINITIONS["fullbody"]["trajectory"])
            for count in env_counts
        )
    if "myoleg80" in models:
        cases.append(
            BenchmarkCase("myoleg80", CASE_DEFINITIONS["myoleg80"]["config_name"], 2048,
                          CASE_DEFINITIONS["myoleg80"]["trajectory"])
        )
    return cases


def validate_timestep_matrix(cases: list[BenchmarkCase], timesteps: int) -> None:
    """Require every case to execute the exact requested number of timesteps."""
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    bad = [case for case in cases if timesteps % (NUM_STEPS * case.env_count)]
    if bad:
        values = ", ".join(f"{case.model}:{case.env_count}" for case in bad)
        raise ValueError(
            f"timesteps={timesteps} is not divisible by num_steps={NUM_STEPS} times "
            f"num_envs for: {values}"
        )


def _config_get(config: Any, dotted: str) -> Any:
    current = config
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def assert_regular_config(config: Any, case: BenchmarkCase) -> None:
    """Reject accidental CLF/robot/config substitutions before benchmarking."""
    expected = CASE_DEFINITIONS[case.model]
    if case.config_name != expected["config_name"]:
        raise ValueError(f"unexpected config for {case.model}: {case.config_name}")
    if "clf" in case.config_name.lower() or "robot" in case.config_name.lower():
        raise ValueError(f"only regular non-robot configs are allowed: {case.config_name}")
    reward_type = str(_config_get(config, "experiment.env_params.reward_type"))
    env_name = str(_config_get(config, "experiment.env_params.env_name"))
    if reward_type != "MimicReward":
        raise ValueError(f"{case.config_name} uses {reward_type}, expected MimicReward")
    if env_name != expected["env_name"]:
        raise ValueError(f"{case.config_name} uses {env_name}, expected {expected['env_name']}")


def compose_benchmark_config(case: BenchmarkCase, timesteps: int) -> Any:
    """Compose and override a production config entirely in memory."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(FULLBODY_CONFIG_DIR)):
            config = compose(config_name=case.config_name)
    finally:
        GlobalHydra.instance().clear()

    assert_regular_config(config, case)
    with open_dict(config):
        exp = config.experiment
        exp.env_params.num_envs = case.env_count
        exp.num_envs = case.env_count
        exp.total_timesteps = timesteps
        exp.ppo_config.num_steps = NUM_STEPS
        exp.n_seeds = 1
        exp.seeds = [0]
        exp.vmap_across_seeds = False
        exp.save_checkpoints = False
        exp.checkpoints_on_validation = False
        exp.save_checkpoints_on_validation = False
        exp.checkpoint_interval = 0
        exp.auto_resume = False
        exp.resume_from = None
        exp.run_id = None
        exp.validation.active = False
        # PPO config derivation still computes validation_interval even when inactive.
        exp.validation.num = 1
        config.wandb.mode = "disabled"
        dataset = exp.task_factory.params.amass_dataset_conf
        dataset.dataset_group = None
        dataset.traj_path = [case.trajectory]
        if "rel_dataset_path" in dataset:
            dataset.rel_dataset_path = None

    assert_regular_config(config, case)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if resolved["experiment"]["ppo_config"]["num_steps"] != NUM_STEPS:
        raise AssertionError("benchmark must keep num_steps=80")
    return config


def classify_failure(returncode: int, text: str) -> str:
    lowered = text.lower()
    oom_markers = (
        "out of memory", "resource_exhausted", "cuda_error_out_of_memory",
        "failed to allocate", "std::bad_alloc", "oom",
    )
    if any(marker in lowered for marker in oom_markers):
        return "oom"
    if returncode < 0:
        return "signal"
    return "error"


def aggregate_results(
    runs: list[dict[str, Any]], gpu_memory_total_bytes: int | None,
) -> dict[str, Any]:
    """Aggregate repeats and apply the throughput/memory recommendation rule."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["model"], int(run["env_count"])), []).append(run)

    medians: list[dict[str, Any]] = []
    for (model, env_count), items in sorted(grouped.items()):
        successes = [item for item in items if item.get("status") == "success"]
        entry: dict[str, Any] = {
            "model": model,
            "env_count": env_count,
            "attempts": len(items),
            "successful_repeats": len(successes),
            "failures": [item.get("failure_type", "error") for item in items if item.get("status") != "success"],
            "feasible": len(successes) == len(items) and bool(items),
        }
        if successes:
            entry["median_setup_seconds"] = statistics.median(float(item["setup_seconds"]) for item in successes)
            entry["median_jit_compile_seconds"] = statistics.median(
                float(item["jit_compile_seconds"]) for item in successes
            )
            entry["median_execution_seconds"] = statistics.median(
                float(item["execution_seconds"]) for item in successes
            )
            entry["median_timesteps_per_second"] = statistics.median(
                float(item["timesteps_per_second"]) for item in successes
            )
            peaks = [item.get("peak_allocated_vram_bytes") for item in successes]
            peaks = [int(value) for value in peaks if value is not None]
            entry["peak_allocated_vram_bytes"] = max(peaks) if peaks else None
            device_peaks = [item.get("nvidia_smi_peak_used_vram_bytes") for item in successes]
            device_peaks = [int(value) for value in device_peaks if value is not None]
            entry["peak_device_used_vram_bytes"] = max(device_peaks) if device_peaks else None
        medians.append(entry)

    fullbody = [entry for entry in medians if entry["model"] == "fullbody"]
    memory_limit = gpu_memory_total_bytes * 0.90 if gpu_memory_total_bytes else None
    eligible = [
        entry for entry in fullbody
        if entry["feasible"]
        and "median_timesteps_per_second" in entry
        and (memory_limit is None or _recommendation_peak_bytes(entry) < memory_limit)
    ]
    recommendation = None
    if eligible:
        best = max(float(entry["median_timesteps_per_second"]) for entry in eligible)
        near_best = [entry for entry in eligible if float(entry["median_timesteps_per_second"]) >= best * 0.97]
        recommendation = min(near_best, key=lambda entry: int(entry["env_count"]))["env_count"]

    by_key = {(entry["model"], entry["env_count"]): entry for entry in medians}
    full_2048 = by_key.get(("fullbody", 2048), {})
    leg_2048 = by_key.get(("myoleg80", 2048), {})
    slowdown = None
    if full_2048.get("feasible") and leg_2048.get("feasible"):
        slowdown = (
            float(full_2048["median_execution_seconds"])
            / float(leg_2048["median_execution_seconds"])
        )
    return {
        "median_throughput_by_case": medians,
        "fullbody_slowdown_ratio_vs_myoleg80_at_2048": slowdown,
        "recommended_fullbody_env_count": recommendation,
        "recommendation_rule": (
            "smallest feasible count within 3% of best throughput and below 90% actual peak device VRAM"
        ),
    }


def _recommendation_peak_bytes(entry: dict[str, Any]) -> float:
    """Prefer whole-device usage because JAX stats omit Warp allocations."""
    peak = entry.get("peak_device_used_vram_bytes")
    if peak is None:
        peak = entry.get("peak_allocated_vram_bytes")
    return float(peak) if peak is not None else float("inf")


def _safe_cleanup(root: Path, token: str) -> None:
    """Remove only a marked, direct child of /tmp created by this script."""
    resolved = root.resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    marker = resolved / MARKER_NAME
    if resolved.parent != tmp or not resolved.name.startswith(TMP_PREFIX):
        raise ValueError(f"refusing to clean unsafe benchmark path: {resolved}")
    if not marker.is_file() or marker.read_text(encoding="utf-8") != token:
        raise ValueError(f"refusing to clean unowned benchmark path: {resolved}")
    shutil.rmtree(resolved)


def _run_command(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False, **kwargs)


def _nvidia_query(gpu_index: int, fields: str) -> subprocess.CompletedProcess[str]:
    return _run_command([
        "nvidia-smi", "-i", str(gpu_index), f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ])


def collect_preflight(
    cases: list[BenchmarkCase], timesteps: int, gpu_index: int, require_gpu: bool
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    config_names = {case.config_name for case in cases}
    expected_names = {item["config_name"] for item in CASE_DEFINITIONS.values()}
    if not config_names <= expected_names:
        errors.append(f"unexpected configs: {sorted(config_names - expected_names)}")
    for name in config_names:
        if not (FULLBODY_CONFIG_DIR / f"{name}.yaml").is_file():
            errors.append(f"missing config: {name}.yaml")
    for trajectory in sorted({case.trajectory for case in cases}):
        if not Path(trajectory).is_file():
            errors.append(f"missing trajectory: {trajectory}")
    reward_source = REPO_ROOT / "musclemimic/core/reward/trajectory_based.py"
    if not reward_source.is_file() or "class MimicReward(" not in reward_source.read_text(encoding="utf-8"):
        errors.append("MimicReward implementation not found")
    checked_configs: set[str] = set()
    for case in cases:
        if case.config_name in checked_configs:
            continue
        checked_configs.add(case.config_name)
        try:
            compose_benchmark_config(case, timesteps)
        except Exception as exc:
            errors.append(f"invalid regular config {case.config_name}: {exc}")

    gpu = _nvidia_query(gpu_index, "index,name,memory.total,driver_version")
    hardware: dict[str, Any] = {"gpu_index": gpu_index}
    if gpu.returncode:
        message = gpu.stderr.strip() or gpu.stdout.strip() or "nvidia-smi failed"
        (errors if require_gpu else warnings).append(message)
    else:
        parts = [part.strip() for part in gpu.stdout.strip().split(",")]
        if len(parts) >= 4:
            hardware.update({
                "gpu_name": parts[1],
                "gpu_memory_total_mib": int(parts[2]),
                "gpu_memory_total_bytes": int(parts[2]) * 1024 * 1024,
                "nvidia_driver": parts[3],
            })
        apps = _run_command([
            "nvidia-smi", "-i", str(gpu_index),
            "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits",
        ])
        competitors = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
        if competitors:
            errors.append("competing GPU compute processes: " + "; ".join(competitors))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "hardware": hardware}


def _package_versions() -> dict[str, str | None]:
    names = ("jax", "jaxlib", "mujoco", "mujoco-mjx", "warp-lang", "flax", "optax", "numpy")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_metadata() -> dict[str, Any]:
    revision = _run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    status = _run_command(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


class GpuMonitor:
    def __init__(self, gpu_index: int, interval: float = 0.2):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {"gpu_utilization_mean_percent": None, "gpu_utilization_max_percent": None,
                    "nvidia_smi_peak_used_vram_bytes": None}
        utils = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        return {
            "gpu_utilization_mean_percent": statistics.fmean(utils),
            "gpu_utilization_max_percent": max(utils),
            "nvidia_smi_peak_used_vram_bytes": max(memory) * 1024 * 1024,
        }

    def _sample(self) -> None:
        while not self._stop.is_set():
            result = _nvidia_query(self.gpu_index, "utilization.gpu,memory.used")
            if result.returncode == 0:
                try:
                    util, memory = (part.strip() for part in result.stdout.strip().split(","))
                    self.samples.append((float(util), int(memory)))
                except (TypeError, ValueError):
                    pass
            self._stop.wait(self.interval)


def _device_peak_bytes(device: Any) -> int | None:
    stats = device.memory_stats() or {}
    for key in ("peak_bytes_in_use", "peak_bytes_in_use", "bytes_in_use"):
        value = stats.get(key)
        if value is not None:
            return int(value)
    return None


def _model_dimensions(env: Any, agent_conf: Any) -> dict[str, Any]:
    base = env.unwrapped() if callable(getattr(env, "unwrapped", None)) else getattr(env, "unwrapped", env)
    model = getattr(base, "_model", None)
    exp = agent_conf.config.experiment
    return {
        "observation_dim": int(env.mdp_info.observation_space.shape[0]),
        "action_dim": int(env.info.action_space.shape[0]),
        "nq": int(model.nq) if model is not None else None,
        "nv": int(model.nv) if model is not None else None,
        "nu": int(model.nu) if model is not None else None,
        "actor_hidden_layers": list(exp.actor_hidden_layers),
        "critic_hidden_layers": list(exp.critic_hidden_layers),
    }


def disable_benchmark_checkpoint_hooks() -> None:
    """Suppress production's unconditional final save in benchmark workers.

    Periodic checkpointing honors ``save_checkpoints=False``, but the production
    training function currently calls its final-save helper unconditionally.
    Replacing only that worker-local module symbol keeps benchmark tracing and
    execution away from every real checkpoint directory without modifying
    production training code.
    """
    from musclemimic.algorithms.ppo import runner as ppo_runner

    def _no_final_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        return None

    ppo_runner._save_final_checkpoint = _no_final_checkpoint


def run_worker(spec_path: Path, result_path: Path) -> int:
    """Run one isolated compile + execution measurement."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    case = BenchmarkCase(**spec["case"])
    gpu_index = int(spec["gpu_index"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    jax_preallocate = bool(spec.get("jax_preallocate", False))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if jax_preallocate else "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
    result: dict[str, Any] = {
        "model": case.model, "config_name": case.config_name, "env_count": case.env_count,
        "repeat": int(spec["repeat"]), "requested_timesteps": int(spec["timesteps"]),
        "seed": 0, "jax_preallocate": jax_preallocate,
    }
    monitor: GpuMonitor | None = None
    try:
        setup_started = time.perf_counter()
        import jax

        from musclemimic.runner.engine import build_agent_conf, build_train_fn, instantiate_env, pick_algorithm

        if jax.default_backend() != "gpu":
            raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, expected 'gpu'")
        disable_benchmark_checkpoint_hooks()
        config = compose_benchmark_config(case, int(spec["timesteps"]))
        env = instantiate_env(config)
        algorithm = pick_algorithm(config)
        agent_conf = build_agent_conf(algorithm, env, config)
        train_fn = build_train_fn(algorithm, env, agent_conf, None, None, logging_interval=1, val_env=None)
        rng = jax.random.PRNGKey(0)
        result["model_dimensions"] = _model_dimensions(env, agent_conf)
        result["setup_seconds"] = time.perf_counter() - setup_started

        compile_started = time.perf_counter()
        executable = jax.jit(train_fn).lower(rng).compile()
        result["jit_compile_seconds"] = time.perf_counter() - compile_started

        monitor = GpuMonitor(gpu_index)
        monitor.start()
        execution_started = time.perf_counter()
        output = executable(rng)
        jax.block_until_ready(output)
        result["execution_seconds"] = time.perf_counter() - execution_started
        result.update(monitor.stop())
        monitor = None
        result["timesteps_per_second"] = int(spec["timesteps"]) / result["execution_seconds"]
        result["peak_allocated_vram_bytes"] = _device_peak_bytes(jax.devices()[0])
        result["status"] = "success"
    except BaseException as exc:
        if monitor is not None:
            result.update(monitor.stop())
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        result.update({
            "status": "failed", "failure_type": classify_failure(1, detail),
            "error": str(exc), "traceback_tail": detail[-8000:],
        })
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "success" else 1


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_orchestrator(args: argparse.Namespace) -> int:
    cases = build_matrix(args.env_counts, tuple(args.models))
    validate_timestep_matrix(cases, args.timesteps)
    preflight = collect_preflight(cases, args.timesteps, args.gpu_index, require_gpu=not args.dry_run)
    plan = {
        "cases": [asdict(case) for case in cases], "repeats": args.repeats,
        "timesteps_per_run": args.timesteps, "num_steps": NUM_STEPS,
        "worker_processes": len(cases) * args.repeats,
        "jax_preallocate": args.jax_preallocate,
    }
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "plan": plan, "preflight": preflight}, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 2
    if not preflight["ok"]:
        print(json.dumps(preflight, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    token = uuid.uuid4().hex
    temp_root = Path(tempfile.mkdtemp(prefix=TMP_PREFIX))
    (temp_root / MARKER_NAME).write_text(token, encoding="utf-8")
    runs: list[dict[str, Any]] = []
    started = time.time()
    try:
        run_number = 0
        for case in cases:
            for repeat in range(args.repeats):
                run_number += 1
                stem = f"{run_number:02d}-{case.model}-{case.env_count}-repeat{repeat + 1}"
                spec_path = temp_root / f"{stem}.spec.json"
                result_path = temp_root / f"{stem}.result.json"
                log_path = temp_root / f"{stem}.log"
                spec = {"case": asdict(case), "timesteps": args.timesteps,
                        "repeat": repeat + 1, "gpu_index": args.gpu_index,
                        "jax_preallocate": args.jax_preallocate}
                _write_json(spec_path, spec)
                print(f"[{run_number}/{len(cases) * args.repeats}] {case.model} "
                      f"num_envs={case.env_count} repeat={repeat + 1}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path),
                         "--worker-result", str(result_path)],
                        cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
                    )
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = {
                        "model": case.model, "config_name": case.config_name,
                        "env_count": case.env_count, "repeat": repeat + 1,
                        "requested_timesteps": args.timesteps, "seed": 0, "status": "failed",
                        "failure_type": classify_failure(completed.returncode, log_text),
                        "error": f"worker exited {completed.returncode} without a result",
                        "log_tail": log_text[-8000:],
                    }
                runs.append(result)

        aggregate = aggregate_results(runs, preflight["hardware"].get("gpu_memory_total_bytes"))
        report = {
            "schema_version": 1, "status": "complete", "created_unix_time": time.time(),
            "elapsed_wall_seconds": time.time() - started, "plan": plan,
            "hardware": preflight["hardware"], "dependencies": _package_versions(),
            "git": _git_metadata(), "preflight": preflight, "runs": runs, **aggregate,
        }
        _write_json(args.output.resolve(), report)
        print(json.dumps({
            "output": str(args.output.resolve()),
            "recommended_fullbody_env_count": report["recommended_fullbody_env_count"],
            "fullbody_slowdown_ratio_vs_myoleg80_at_2048": (
                report["fullbody_slowdown_ratio_vs_myoleg80_at_2048"]
            ),
        }, indent=2, sort_keys=True))
        return 0 if all(run["status"] == "success" for run in runs) else 1
    finally:
        _safe_cleanup(temp_root, token)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-counts", type=parse_env_counts, default=list(DEFAULT_ENV_COUNTS),
                        help="comma-separated full-body counts (default: 512,1024,2048,4096)")
    parser.add_argument("--models", choices=("fullbody", "myoleg80"), action="append",
                        help="model to benchmark; repeat for both (default: both)")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jax-preallocate", action="store_true",
                        help="enable JAX's default GPU preallocation in each worker")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.models is None:
        args.models = ["fullbody", "myoleg80"]
    if args.worker_spec or args.worker_result:
        if not args.worker_spec or not args.worker_result:
            raise SystemExit("both worker arguments are required")
        return run_worker(args.worker_spec, args.worker_result)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
