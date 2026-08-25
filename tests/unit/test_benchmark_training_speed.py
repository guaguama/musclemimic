"""Unit tests for the isolated training-throughput benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import benchmark_training_speed as benchmark


def _run(model: str, env_count: int, throughput: float, peak: int, status: str = "success"):
    run = {
        "model": model, "env_count": env_count, "status": status,
        "setup_seconds": 2.0, "jit_compile_seconds": 3.0,
        "execution_seconds": 1000.0 / throughput,
        "timesteps_per_second": throughput, "peak_allocated_vram_bytes": peak,
        "nvidia_smi_peak_used_vram_bytes": peak,
    }
    if status != "success":
        run["failure_type"] = status
    return run


def test_matrix_contains_only_regular_configs_and_reuses_fullbody_2048():
    matrix = benchmark.build_matrix([512, 1024, 2048, 4096])
    assert [(case.model, case.env_count) for case in matrix] == [
        ("fullbody", 512), ("fullbody", 1024), ("fullbody", 2048),
        ("fullbody", 4096), ("myoleg80", 2048),
    ]
    assert all("clf" not in case.config_name and "robot" not in case.config_name for case in matrix)
    benchmark.validate_timestep_matrix(matrix, 3_276_800)
    with pytest.raises(ValueError, match="not divisible"):
        benchmark.validate_timestep_matrix(benchmark.build_matrix([3072]), 3_276_800)


def test_matrix_can_select_only_fullbody_for_allocator_probe():
    matrix = benchmark.build_matrix([2048], ("fullbody",))
    assert [(case.model, case.env_count) for case in matrix] == [("fullbody", 2048)]


def test_aggregation_medians_slowdown_and_smallest_near_best_recommendation():
    gib = 1024**3
    runs = []
    for throughput in (95.0, 97.0):
        runs.append(_run("fullbody", 1024, throughput, 10 * gib))
    for throughput in (99.0, 101.0):
        runs.append(_run("fullbody", 2048, throughput, 12 * gib))
    for throughput in (101.0, 103.0):
        runs.append(_run("fullbody", 4096, throughput, 15 * gib))
    for throughput in (198.0, 202.0):
        runs.append(_run("myoleg80", 2048, throughput, 8 * gib))

    result = benchmark.aggregate_results(runs, 24 * gib)
    assert result["recommended_fullbody_env_count"] == 2048
    assert result["fullbody_slowdown_ratio_vs_myoleg80_at_2048"] == pytest.approx(2.0)


def test_recommendation_excludes_over_90_percent_memory_and_failed_repeat():
    gib = 1024**3
    runs = [
        _run("fullbody", 1024, 90.0, 10 * gib), _run("fullbody", 1024, 90.0, 10 * gib),
        _run("fullbody", 2048, 100.0, 23 * gib), _run("fullbody", 2048, 100.0, 23 * gib),
        _run("fullbody", 4096, 120.0, 1, "oom"), _run("fullbody", 4096, 120.0, 1, "oom"),
    ]
    result = benchmark.aggregate_results(runs, 24 * gib)
    assert result["recommended_fullbody_env_count"] == 1024
    by_count = {item["env_count"]: item for item in result["median_throughput_by_case"]}
    assert not by_count[4096]["feasible"]
    assert by_count[4096]["failures"] == ["oom", "oom"]


def test_recommendation_uses_device_peak_when_warp_memory_is_outside_jax():
    gib = 1024**3
    runs = [
        _run("fullbody", 2048, 100.0, 6 * gib),
        _run("fullbody", 2048, 100.0, 6 * gib),
        _run("fullbody", 4096, 130.0, 10 * gib),
        _run("fullbody", 4096, 130.0, 10 * gib),
    ]
    for run in runs[2:]:
        run["nvidia_smi_peak_used_vram_bytes"] = 23 * gib
    result = benchmark.aggregate_results(runs, 24 * gib)
    assert result["recommended_fullbody_env_count"] == 2048


def test_oom_classification():
    assert benchmark.classify_failure(1, "RESOURCE_EXHAUSTED: failed to allocate") == "oom"
    assert benchmark.classify_failure(-9, "killed") == "signal"
    assert benchmark.classify_failure(1, "bad configuration") == "error"


def test_regular_config_assertions():
    case = benchmark.build_matrix([512])[0]
    config = {
        "experiment": {"env_params": {"reward_type": "MimicReward", "env_name": "MjxOSLFullBody"}}
    }
    benchmark.assert_regular_config(config, case)
    config["experiment"]["env_params"]["reward_type"] = "CLFReward"
    with pytest.raises(ValueError, match="expected MimicReward"):
        benchmark.assert_regular_config(config, case)


@pytest.mark.parametrize("case", benchmark.build_matrix([512])[:1] + benchmark.build_matrix([]))
def test_real_regular_configs_compose_with_benchmark_overrides(case):
    config = benchmark.compose_benchmark_config(case, benchmark.DEFAULT_TIMESTEPS)
    benchmark.assert_regular_config(config, case)
    assert config.experiment.env_params.num_envs == case.env_count
    assert config.experiment.ppo_config.num_steps == 80
    assert config.experiment.validation.active is False
    assert config.experiment.save_checkpoints is False
    assert config.experiment.auto_resume is False
    assert config.wandb.mode == "disabled"
    assert config.experiment.task_factory.params.amass_dataset_conf.dataset_group is None
    assert list(config.experiment.task_factory.params.amass_dataset_conf.traj_path) == [case.trajectory]


def test_guarded_cleanup_accepts_owned_root_and_rejects_unowned(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark.tempfile, "gettempdir", lambda: str(tmp_path))
    owned = tmp_path / f"{benchmark.TMP_PREFIX}owned"
    owned.mkdir()
    (owned / benchmark.MARKER_NAME).write_text("token", encoding="utf-8")
    (owned / "worker.log").write_text("x", encoding="utf-8")
    benchmark._safe_cleanup(owned, "token")
    assert not owned.exists()

    unowned = tmp_path / f"{benchmark.TMP_PREFIX}unowned"
    unowned.mkdir()
    with pytest.raises(ValueError, match="unowned"):
        benchmark._safe_cleanup(unowned, "token")
    assert unowned.exists()


def test_benchmark_disables_unconditional_final_checkpoint(monkeypatch):
    from musclemimic.algorithms.ppo import runner

    sentinel = object()
    monkeypatch.setattr(runner, "_save_final_checkpoint", sentinel)
    benchmark.disable_benchmark_checkpoint_hooks()
    assert runner._save_final_checkpoint is not sentinel
    assert runner._save_final_checkpoint("ignored") is None
