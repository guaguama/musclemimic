"""Reproduce fresh PPO smokes and paired pre-update survival comparisons.

Run each model in a separate process to release compiled GPU executables:
  python scripts/verify_actor_bias_init.py --model rigid
  python scripts/verify_actor_bias_init.py --model compliant

Uses the configured architecture, exploration, trajectories and termination.
No W&B, videos, or checkpoints are produced. JSON contains finite-value checks
and first-episode survival times (including censoring at the configured horizon).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.algorithms.ppo.ppo import PPOJax
from musclemimic.algorithms.ppo.runner import _init_train_state
from musclemimic.runner.engine import (
    _maybe_share_validation_trajectory,
    build_metrics_handler,
    instantiate_env,
    instantiate_validation_env,
)


def configuration(model):
    name = "conf_osl_fullbody_robot"
    overrides = [
        "experiment.env_params.num_envs=8",
        "experiment.total_timesteps=20480",
        "experiment.ppo_config.num_minibatches=8",
        "experiment.validation.num_envs=8",
        "experiment.validation.num=1",
        "experiment.save_checkpoints=false",
        "experiment.checkpoints_on_validation=false",
        "experiment.validation.video_frequency=0",
        "experiment.auto_resume=false",
        "experiment.resume_from=null",
        "wandb.mode=disabled",
    ]
    if model == "compliant":
        overrides.append("experiment.env_params.compliant_socket=true")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "fullbody"), version_base=None):
        return compose(config_name=name, overrides=overrides)


def finite_tree(tree):
    return all(
        np.isfinite(np.asarray(x)).all()
        for x in jax.tree.leaves(tree)
        if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.number)
    )


def survival(env, agent_conf, seeds):
    """First done for paired states/noise; subsequent autoresets are ignored."""
    cfg = agent_conf.config
    wrapped = wrap_env(env, cfg.experiment)
    network = agent_conf.network
    unbiased = network.clone(actor_bias_init=None)
    reset = jax.jit(wrapped.reset)
    horizon = int(env.info.horizon)
    chunk = 80

    def rollout(net):
        def step(carry, unused):
            ts, state, obs, rng, first, ended, elapsed = carry
            rng, noise = jax.random.split(rng)
            (pi, _), updates = net.apply({"params": ts.params, "run_stats": ts.run_stats}, obs, mutable=["run_stats"])
            obs, _, _, done, _, state = wrapped.step(state, pi.sample(seed=noise))
            elapsed += 1
            first = jnp.where(done & ~ended, elapsed, first)
            return (ts.replace(run_stats=updates["run_stats"]), state, obs, rng, first, ended | done, elapsed), None

        return jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=chunk)[0])

    runs = [rollout(unbiased), rollout(network)]
    records = []
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        reset_keys = jax.random.split(jax.random.fold_in(key, 100), 8)
        obs, state = reset(reset_keys)
        pair = {}
        for mode, net, run in zip(("none", "qpos0"), (unbiased, network), runs, strict=True):
            ts = _init_train_state(key, wrapped, net, agent_conf.tx, None, cfg.experiment)
            carry = (
                ts,
                state,
                obs,
                jax.random.fold_in(key, 200),
                jnp.full((8,), horizon),
                jnp.zeros((8,), dtype=bool),
                jnp.array(0),
            )
            for _ in range((horizon + chunk - 1) // chunk):
                carry = run(carry)
                if np.asarray(carry[5]).all():
                    break
            steps = np.minimum(np.asarray(carry[4]), horizon)
            pair[mode] = {
                "steps": steps.tolist(),
                "seconds": (steps * env.dt).tolist(),
                "censored": (~np.asarray(carry[5])).tolist(),
            }
        records.append({"seed": seed, **pair})
        print("SURVIVAL", json.dumps(records[-1]), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rigid", "compliant"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/actor_bias_verification"))
    parser.add_argument("--survival-only", action="store_true")
    parser.add_argument("--training-only", action="store_true", help="Reuse this model's saved survival measurements")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = configuration(args.model)
    print(OmegaConf.to_yaml(config), flush=True)
    env = instantiate_env(config)
    agent_conf = PPOJax.init_agent_conf(env, config)
    assert np.count_nonzero(agent_conf.network.actor_bias_init) == 17
    path = args.output_dir / f"{args.model}.json"
    report = (
        json.loads(path.read_text())
        if args.training_only
        else {"model": args.model, "biased_dimensions": 17, "survival": survival(env, agent_conf, (0, 1, 2))}
    )
    path.write_text(json.dumps(report, indent=2))
    if not args.survival_only:
        val_env = instantiate_validation_env(config, share_trajectory=True)
        _maybe_share_validation_trajectory(env, val_env, config)
        mh = build_metrics_handler(config, val_env)
        assert mh is not None
        losses = []
        validation_updates = []

        def log_update(metrics):
            if metrics.get("has_validation_update", False):
                validation_updates.append(int(metrics["max_timestep"]))
            if "ppo/total_loss" in metrics:
                losses.append(
                    {key: float(metrics[key]) for key in ("ppo/total_loss", "ppo/value_loss", "ppo/policy_loss")}
                )

        # Returned training metrics contain episode summaries. Losses are exposed
        # by the per-update callback, so verify those separately after completion.
        train = jax.jit(
            PPOJax.build_train_fn(
                env, agent_conf, mh=mh, val_env=val_env, online_logging_callback=log_update, logging_interval=1
            )
        )
        # The runner currently calls its final saver unconditionally, even with
        # save_checkpoints=false. Suppress that output hook for this diagnostic.
        with patch("musclemimic.algorithms.ppo.runner._save_final_checkpoint"):
            result = jax.block_until_ready(train(jax.random.PRNGKey(0)))
            jax.effects_barrier()
        assert len(losses) == config.experiment.num_updates, "Missing per-update loss measurements"
        assert all(np.isfinite(list(row.values())).all() for row in losses), "Nonfinite losses"
        assert validation_updates == [20480], "Expected one validation rollout at the final update"
        assert np.asarray(result["validation_metrics"].euclidean_distance.qpos).size > 0
        assert finite_tree(result["validation_metrics"]), "Nonfinite validation metrics"
        assert finite_tree(result["agent_state"].train_state.params), "Nonfinite parameters"
        assert finite_tree(result["training_metrics"]), "Nonfinite training metrics"
        metrics = result["training_metrics"]
        report["training"] = {
            "timesteps": int(config.experiment.num_updates * config.experiment.num_steps * 8),
            "optimizer_steps": int(result["agent_state"].train_state.step),
            "finite_parameters": True,
            "finite_losses": True,
            "validation_timesteps": validation_updates,
            "finite_validation_metrics": True,
            "losses": losses,
            "finite_training_metrics": True,
            "metric_shapes": [list(x.shape) for x in jax.tree.leaves(metrics)],
        }
        path.write_text(json.dumps(report, indent=2))
    print("VERIFIED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
