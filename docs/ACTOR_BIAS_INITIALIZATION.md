# Neutral actor output initialization

`actor_bias_init` is **opt-in**: the code default is `none`, and the two robot
configs (`fullbody/conf_osl_fullbody_robot.yaml`,
`fullbody/conf_new_osl_fullbody_robot.yaml`) set `qpos0` explicitly. The
eligibility test is purely structural, so a code-level default would silently
activate on any future config whose model happened to match; enabling it per
config keeps that choice deliberate. Where enabled, eligible direct
`DefaultControl` position servos receive the normalized command that balances
their affine position force at the model's `qpos0` with zero velocity. The actor
output bias is capped at ±0.95. Unsupported dimensions, including the OSL torque
motors and muscle actuators, keep zero bias. An entirely zero vector retains the
previous initializer.

For actuator gain `g`, scalar transmission gear `gear`, position stiffness
`kp = -biasprm[1]`, and affine offset `biasprm[0]`, the target is
`ctrl0 = (kp * gear * qpos0 - biasprm[0]) / g`. Normalization uses the controller's
actual mean and half-width. Only fixed-gain, affine-bias hinge/slide joint
actuators without activation dynamics are eligible. Incremental controllers are
excluded. Invalid or degenerate actuator coefficients are skipped.

Both robot configs already carry the key, so override it without a `+`:

```bash
experiment.ppo_config.actor_bias_init=none
```

Use `+` only on a config that does not define the key (e.g. a muscle config,
where the result is a no-op anyway); `++` works either way.

`null` and `false` also disable this setting; unrecognized values raise an error.
An explicitly supplied top-level `experiment.actor_bias_init` takes precedence
above the nested PPO setting. Checkpoint parameters override initialization;
restoration, exploration noise, critic parameters and output kernels are
unchanged. A nonzero bias with an MoE actor output is unsupported; a critic-only
MoE output remains supported. The cap leaves limited action headroom and does
not prevent clipping of Gaussian samples.

## Verification

Verified on 2026-09-15. The original-code baseline passed 593 tests, with 25
skipped and 19 deselected. After implementation, `make test` passed 651 tests,
with the same skip/deselection counts. The focused PPO/config/initialization
suite passed 96 tests. Both suites emitted one existing JAXopt deprecation
warning. Checkpoint tests needed execution outside the filesystem/network
sandbox. The baseline used a temporary copy of the original algorithm files
while retaining the pre-existing workspace model changes.

All three robot vectors have 29 dimensions and 17 nonzero entries. The two new
socket variants are identical; legacy and new values agree within `1e-6`
(approximately `8e-7` maximum difference). Legacy elbows are named
`elbow_flex_{l,r}`, while the new model uses `elbow_flexion_{l,r}`. All six current
muscle variants retain their prior initialization, including runtime control
range rewrites.

The independent MuJoCo standing regression starts at `qpos0`, with zero
velocity, direct control and 2 ms physics steps. After 0.5 seconds:

| Model | Zero-action height | Biased height | Zero-action max servo drift | Biased max servo drift |
| --- | ---: | ---: | ---: | ---: |
| New rigid | 0.546 m | 0.932 m | 1.531 rad | 0.075 rad |
| New compliant | 0.544 m | 0.932 m | 1.530 rad | 0.075 rad |

Tests use tolerances: biased height above 0.90 m and drift below 0.12 rad;
zero-action height below 0.65 m and drift above 1 rad.

## Reproduction

`scripts/verify_actor_bias_init.py` runs paired survival measurements before
training, followed by a fresh training smoke. Run each model in a separate
process on a GPU with the configured trajectory files available:

```bash
.venv/bin/python scripts/verify_actor_bias_init.py --model legacy
.venv/bin/python scripts/verify_actor_bias_init.py --model rigid
.venv/bin/python scripts/verify_actor_bias_init.py --model compliant
```

The script uses 8 training environments, 20,480 timesteps, 8 minibatches, 8
validation environments and one validation interval. It retains the configured
architecture, exploration standard deviation (0.2), trajectories, termination
settings and 30,000-step horizon. It checks final parameter finiteness and all
32 updates' losses through the logging callback. W&B and video are disabled.
The runner currently calls its final checkpoint saver even when checkpoint
saving is disabled, so this diagnostic suppresses only that final output hook.
It does not change production checkpoint behavior.

Survival uses three network seeds (0, 1, 2), each with eight initial states.
Enabled/disabled pairs share network initialization, initial states and action
noise seeds. Measurements are times to the first episode termination under the
training termination settings; subsequent autoresets are ignored. These are
separate from step-zero logging and the neutral-stance regression. JSON results
are written to `/tmp/actor_bias_verification/` by default.

Observed pre-update survival (24 paired trials per model, no horizon censoring):

| Model | Mean without bias | Mean with bias | Improved pairs |
| --- | ---: | ---: | ---: |
| Legacy | 0.559 s | 1.172 s | 23 / 24 |
| New rigid | 0.534 s | 1.033 s | 24 / 24 |
| New compliant | 0.489 s | 1.038 s | 24 / 24 |

These short trials establish an initialization and early-survival improvement;
they do not establish learned walking or long-term training performance.
