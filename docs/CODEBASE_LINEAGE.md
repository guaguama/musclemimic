# Codebase lineage: LocoMuJoCo → Kinesis → MuscleMimic

*A functional and quantitative comparison of the three papers/repos in this project tree.*

Analysis date: 2026-08-26. Repos analysed:

| Repo | Path | HEAD | Python LOC |
|---|---|---|---|
| loco-mujoco (upstream, live) | `~/src/00_AMBER/00_projbackflip/loco-mujoco` | `3921fed` (v1.1.0, 2026-03-10) | 27,593 (pkg) |
| Kinesis | `~/src/00_AMBER/01_projpegleg/Kinesis` | `cddab3c` ("Kinesis 2.0") | 17,853 (src+poselib+scripts) |
| **musclemimic (this repo)** | `~/src/00_AMBER/01_projpegleg/musclemimic` | `6561833` (branch `dev`) | 80,000 total |

---

## 1. TL;DR

**The short answer to "how much is unique to musclemimic":**

- The repo vendors a **frozen snapshot of LocoMuJoCo v1.0.1** (April 2025) in `loco_mujoco/`, and adds a new top-level package `musclemimic/` that *replaces* LocoMuJoCo's environment, algorithm, and wrapper layers.
- Accounting over the 80,000 lines of Python, with the 17,046-line motion-clip manifest broken out because it is data, not code:

  | Category | Lines | Share of code |
  |---|---:|---:|
  | Motion-clip data manifest (`loco_mujoco/smpl/const.py`) | 17,046 | — |
  | **Total actual code** | **62,954** | 100% |
  | ├─ LocoMuJoCo code, byte-identical or trivially different | 4,769 | 7.6% |
  | ├─ LocoMuJoCo files extended in place (vendored tree) | 12,114 | 19.2% |
  | ├─ `musclemimic/` files with a direct LocoMuJoCo ancestor | ~7,600 | 12.1% |
  | └─ Net-new code (no upstream ancestor) | ~38,500 | 61.1% |

  Of the net-new total: 16,070 tests, 4,790 C3D web viewer, 4,053 scripts, 2,121 runner/engine, ~2,000 MSK environments, ~1,900 MSK reward/termination, 1,080 entrypoints, 614 viser viewer, plus the checkpoint/curriculum/adaptive-sampling/Soft-MoE machinery.

  Caveat on the third row: "has an ancestor" is file-level provenance, not line-level. Those ~7,600 lines range from `base_robot_humanoid.py` (byte-identical bar one import) to `ppo.py` (9% shared lines with upstream's `ppo_jax.py`).
- **The underlying training behaviour is the same as LocoMuJoCo's.** Both are single-JIT-function PPO in JAX over MJX/MjWarp parallel environments, with a `GoalTrajMimic` goal, a `MimicReward` DeepMimic-style reward, `TrajInitialStateHandler` reference-state initialisation, and site-deviation early termination. LocoMuJoCo's own `examples/training_examples/jax_rl_mimic/conf.yaml` is structurally the same experiment as `fullbody/conf_fullbody.yaml` (side-by-side in §8).
- **What MSK models actually required** was not a new RL algorithm. It was: (a) muscle-state observations, (b) a working reset for the MuJoCo-Warp backend under high contact counts, (c) muscle-appropriate reward/termination terms, (d) a retargeting pipeline that respects biomechanical joint constraints, (e) strictly-on-policy PPO (E=1) because muscle activation dynamics amplify off-policy drift, and (f) a scalable trajectory-sampling scheme to reach hundreds of motions. Items (a)–(d) and (f) are the bulk of the new code; item (e) is a one-line config change with a paper-worth of justification behind it.
- **Kinesis shares no code with either.** It is a PyTorch fork of PHC (Perpetual Humanoid Control) driving MyoSuite on CPU. Its contribution to musclemimic is *conceptual* (the reward shape, muscle-energy regularisation, hard-negative mining) plus one concrete artifact: the curated **KIT-Locomotion clip list**, which lives verbatim in `loco_mujoco/smpl/const.py` as `KIT_KINESIS_TRAINING_MOTIONS` (972 clips) / `KIT_KINESIS_TESTING_MOTIONS` (108 clips).

---

## 2. What each paper set out to do

### 2.1 LocoMuJoCo (Al-Hafez, Zhao, Peters, Tateo — NeurIPS 2023 Robot Learning Workshop)

A **4-page workshop paper**, and the scope matches: it is a *benchmark*, explicitly not a training framework.

Original intent, as stated:

- 12 environments / 27 tasks: Talos, Atlas, Unitree H1, Unitree A1, a torque-actuated skeleton, and a **musculoskeletal human model** — six base embodiments, with both human models scaled to 4 ages (infant ~2y, child ~5y, teenager ~12y, adult).
- Three dataset flavours per task, to span IL paradigms: real noisy mocap (embodiment mismatch), ground-truth expert with actions, and ground-truth *sub-optimal* expert (for preference-based IL).
- POMDP variants via state masking (carry weight, model height) for teacher-student setups.
- Interfaces: **Gymnasium** and **Mushroom-RL**.
- Baselines: GAIL, VAIL, GAIfO, IQ-Learn, LS-IQ, SQIL — all *imitation* algorithms, all Mushroom-RL, all CPU.
- Handcrafted per-task reward functions intended **as evaluation metrics**, with the paper explicitly noting they are "simple and not useful for RL to train realistic gaits."

The paper's closing future-work item is *better metrics for cloned-behaviour quality* — a benchmark concern, not a training one.

At the time of the paper the package was **5,248 lines** across ~15 files, with no JAX, no MJX, no `task_factories`, no trajectory module (just `utils/trajectory.py`), and no SMPL retargeting.

### 2.2 Kinesis (Simos, Chiappa, Mathis — arXiv 2503.14637, v3 Mar 2026)

**Goal:** show that model-free RL motion imitation scales to real musculoskeletal models — up to 290 musculotendon actuators — and that the resulting muscle activations correlate with human EMG.

Key design points:

- Three MyoSuite MyoLeg variants: Legs (80 muscles), Legs+Abs (86), Legs+Back (290).
- Trained on **KIT-Locomotion**: 1,054 clips / 1.8 h of mocap curated from AMASS+KIT into five skills (walk, gradual turn, turn-in-place, walk backwards, run). 946 train / 108 test.
- Reward: exponential kernels on keypoint position and velocity (pelvis, knees, ankles, toes, head), plus **L1+L2 muscle activation energy** to break the over-actuation redundancy, plus an optional upright term.
- Two actuation schemes compared: indirect (PD on desired muscle length, then invert the force-length-velocity relation) vs. **direct** (`m = (a+1)/2`). Direct won on both quality and speed.
- **PPO + Lattice** (correlated exploration for high-dimensional action spaces).
- **Hard negative mining + Mixture-of-Experts**: train expert 1 on everything, remove the clips it solves, train expert 2 on the remainder, repeat; then freeze all experts and train a gating MLP over them. Three experts sufficed.
- Downstream: text-to-control (zero-shot, via MDM), target reaching, penalty kicks.
- Infrastructure: **PyTorch, CPU MuJoCo, 128 parallel environment processes**, A100 for the gradient step. ~10 days for the 290-muscle model (per MuscleMimic's Sec. 2.2). 17,853 lines across `src`, `poselib` and `scripts`.

Code lineage: forked from **PHC / SMPLSim (Luo et al.)** — visible in the repo layout (`src/agents/agent_im.py`, `src/learning/policy_lattice.py`, `src/utils/smpl_skeleton/smpl_local_robot.py`), the `smpl_sim`/`smplx` git pins in `requirements.txt`, and `mujoco==3.1.2` + `warp-lang==0.10.1`. Nothing about it touches LocoMuJoCo.

### 2.3 MuscleMimic (Li, Wang, Ziliotto, Simos, Kovecses, Durandau, Mathis — arXiv 2603.25544, Mar 2026)

**Goal:** remove the two barriers to full-body MSK motor learning — the compute cost of muscle-level simulation, and the absence of validated open full-body models.

Contributions:

- Two MSK embodiments:
  - **MyoBimanualArm** — fixed thorax, 76 joints (36 with fingers off), 126 muscles (64), 54 DoF (14).
  - **MyoFullBody** — free root, 123 joints (83), **416 muscles (354)**, 72 DoF (32), 17 mimic sites, full self-collision.
- A retargeting pipeline with two paths: **Mocap-Body** (MuJoCo mocap-body IK, inherited from LocoMuJoCo's SMPL fitting) and **GMR-Fit** (new: Mink-based IK on top of General Motion Retargeting, with joint equality constraints). GMR-Fit cuts joint-limit violations from ~12% of frames to ~0.3% and tendon-jump rate from 30% to 3% of motions, at ~3× the retargeting cost.
- GPU-parallel training via MuJoCo Warp: 1.3×10⁴ steps/s at 8,192 environments on one H100; 1e9 environment steps in ~20 h.
- The **E=1 finding**: single-epoch (strictly on-policy) PPO updates beat E=3 and E=10 asymptotically. KL divergence stays below 1e-1 for E=1 vs. spikes above 1e10 for E=10. Attributed to muscle activation dynamics (Eq. 1) amplifying small policy changes.
- Biomechanical validation against experimental walking/running: joint kinematics mean r = 0.90, plus GRF, joint moments and EMG comparison.

The Methods section states the relationship outright:

> "We implemented MuscleMimic as a JAX-based framework extending LocoMuJoCo [76] with three major additions: customized retargeting pipeline with additional support to GMR and more motion captured datasets, native MuJoCo Warp support for GPU-accelerated simulation with flexible collision support, and extensive redesign for MSK systems."

---

## 3. Lineage graph

```
                    MyoSuite (MyoArm, MyoLegs, MyoBack)
                            │  models only
                            ├───────────────────────────┐
                            ▼                           ▼
  PHC / SMPLSim ──────► Kinesis (PyTorch, CPU)    MuscleMimic models
  (Luo et al.)          amathislab/Kinesis        (MyoFullBody, MyoBimanualArm)
                            │                           │
                            │ KIT-Locomotion clip list   │
                            │ + reward/energy/MoE ideas  │
                            ▼                           ▼
  LocoMuJoCo v1.0.1 ──────────────────► MuscleMimic (JAX, GPU)
  robfiras/loco-mujoco                  amathislab/musclemimic
  (Apr 2025, vendored)                          │
       │                                        │ your fork
       │ upstream continued independently       ▼
       ▼                                  guaguama/musclemimic:dev
  LocoMuJoCo v1.1.0 (Mar 2026)            (+ OSL prosthesis envs, CLF reward)
  — added its own MjWarp backend
```

**Fork point evidence.** musclemimic's git history is squashed (39 commits, no shared objects with loco-mujoco's 723). But `loco_mujoco/__init__.py` in this repo declares `__version__ = '1.0.1'`, which upstream tagged at commit `8a94284` on 2025-04-18. Best-match search of the vendored `core/mujoco_base.py` against every historical upstream revision lands on `bb6edc2` (2025-04-17, "removed n_envs attribute from core"), 52 differing lines. So: **forked at LocoMuJoCo v1.0.0/1.0.1, mid-April 2025.**

Upstream then went quiet (3 commits in May 2025, nothing until March 2026) and returned with 25 commits producing v1.1.0. This matters for one specific claim below.

---

## 4. What LocoMuJoCo is *today* vs. what the paper described

The paper is badly out of date as a description of the code. The repository was rewritten between **December 2024 and April 2025** — 307 commits in Dec/Jan/Feb alone, starting from a commit literally titled "First mayor refactoring done."

| | Paper (Nov 2023) | Today (v1.1.0, Mar 2026) |
|---|---|---|
| Package size | 5,248 LOC | 27,593 LOC |
| Simulation | CPU MuJoCo only | CPU MuJoCo **+ MJX + MjWarp** |
| RL interface | Gymnasium + Mushroom-RL | Gymnasium wrapper retained; **native JAX** is the primary path |
| Algorithms | GAIL, VAIL, GAIfO, IQ-Learn, LS-IQ, SQIL (Mushroom-RL, CPU) | **PPO, GAIL, AMP, DeepMimic** — single-file JAX/Flax |
| Training loop | external library | **environment + training fused into one JIT-compiled function** |
| Environments | 12 (6 base + age scaling) | **12 humanoids + 4 quadrupeds**: Atlas, Talos, H1, H1v2, G1, Apollo, BoosterT1, ToddlerBot, FourierGR1T2, SkeletonTorque, SkeletonMuscle, MyoSkeleton, A1, Go2, Spot, AnymalC |
| Datasets | in-house Qualisys mocap + MPC | **>22,000 mocap clips**: AMASS, LAFAN1, native, retargeted per-humanoid, HuggingFace-hosted with FK caching |
| Retargeting | none (datasets shipped pre-mapped) | **SMPL shape+motion fitting, and robot-to-robot retargeting** |
| Metrics | handcrafted per-task reward | **DTW + discrete Fréchet distance in JAX**, `MetricsHandler` |
| Modularity | monolithic env classes | **pluggable components**: observation types, reward, control function, domain randomizer, terrain, initial-state handler, terminal-state handler — all swappable at construction |

**Is it a full-fledged tool for training new policies, or still mainly a benchmark?**

Functionally it is now a full training framework. It has:

- a complete JAX RL stack (PPO/GAIL/AMP/DeepMimic) with Hydra configs (`examples/training_examples/`),
- fused env+train JIT compilation,
- MJX and MjWarp parallel backends,
- domain and terrain randomization for sim-to-real,
- reference-state initialisation, mimic goals, mimic rewards, trajectory-based termination.

Its own README advertises a DeepMimic agent trained to omnidirectional human-like walking **in 36 minutes on an RTX 3080 Ti**. That is a training tool, not a leaderboard.

But its *self-conception* and its centre of gravity are still benchmark-shaped:

- The README's first line is still "LocoMuJoCo is an **imitation learning benchmark**", with RL support framed as a secondary affordance ("also supports custom reward function classes, making it suitable for pure reinforcement learning as well").
- The dataset story dominates (22,000 clips, retargeted to every embodiment) — that is a benchmark investment, not a training one.
- The algorithms are deliberately "clean single-file JAX algorithms **for quick benchmarking**" — reference implementations, not a production trainer. No curriculum, no adaptive sampling, no checkpoint-resume, no preemption handling, no validation-video pipeline, no MoE.
- The pretrained-policy column in `loco_mujoco/environments/README.md` is 🔶 (pending) for **every single environment**. A training framework would have shipped checkpoints.

So: *capable* of training new policies, and pleasant to do it in, but still a benchmark in its priorities. Everything MuscleMimic added on the training side is precisely what you need to go from "reproduce a reference result" to "run a 2-billion-step job on a preemptible cluster and get a generalist policy out."

**One important consequence of the fork timing.** LocoMuJoCo v1.1.0's headline feature is the MjWarp backend, added 2026-03-09 — *eleven months after musclemimic forked* and after MuscleMimic's own Warp work. The two solutions are independent and the difference is instructive:

`loco_mujoco/core/mujoco_mjx.py` (upstream, v1.1.0):

```python
if self._use_mjwarp:
    warnings.warn(
        "mjwarp is experimental and may not work with all environments. "
        "Resetting is error prone. Training might work, but will not be optimal.",
        UserWarning)
    time.sleep(3)
    ...
# and in _mjx_reset_in_step:
if self._use_mjwarp:
    # todo: this is not a good way of resetting the data, all other entities
    # should be reset as well! there is an issue with the warp backend that
    # does not allow to reset the data properly
    data = state.data.replace(qpos=jnp.zeros_like(state.data.qpos),
                              qvel=jnp.zeros_like(state.data.qvel))
```

Upstream's Warp reset zeroes `qpos`/`qvel` and admits it is wrong. MuscleMimic instead **solved the reset problem**, which was mandatory: you cannot do reference-state initialisation from 972 motion clips if reset is broken. See §6.2.

---

## 5. Kinesis's actual contribution to this repo

Zero lines of code. Verified by:

- Language/stack mismatch: Kinesis is PyTorch + `mujoco==3.1.2` + CPU multiprocessing; musclemimic is JAX/Flax + `mujoco-mjx==3.4.0` + `warp-lang==1.10.0`.
- Architecture mismatch: Kinesis's `src/agents/agent_im.py` / `src/env/myolegs_im.py` structure is PHC's; musclemimic's is LocoMuJoCo's.
- Grep for `kinesis` across all `*.py`/`*.yaml` in this repo returns **only dataset-group names** (`KIT_KINESIS_TRAINING_MOTIONS`, `KIT_KINESIS_TESTING_MOTIONS`, `..._MINT`, `..._MINT_STRAIGHT_FORWARDS`, `KIT_KINESIS_TRANSITION_TRAINING_MOTIONS`) plus config references to them.

What did carry over:

| Kinesis idea | How it appears in musclemimic |
|---|---|
| KIT-Locomotion curation (972/108 clips) | verbatim clip lists in `loco_mujoco/smpl/const.py`; the default `dataset_group` in `fullbody/conf_fullbody.yaml` |
| Exponential-kernel keypoint reward | already present in LocoMuJoCo's `MimicReward`; musclemimic keeps that formulation |
| Muscle activation energy penalty | `activation_energy_coeff` in `musclemimic/core/reward/trajectory_based.py` (new; LocoMuJoCo has no such term) |
| Direct actuation `m=(a+1)/2` | `MyoFullBody._apply_spec_changes` rewrites every muscle actuator's `ctrlrange` to `[-1, 1]`; `DefaultControl` then rescales linearly |
| Indirect/PD-style actuation | `DefaultControl(apply_mode="incremental")` — integrates clipped deltas onto previous control |
| Hard negative mining | replaced by **soft adaptive sampling** (`musclemimic/algorithms/common/adaptive_sampling.py`) — one policy, per-trajectory sampling weights from EMA'd early-termination rates, instead of Kinesis's discrete expert-splitting |
| Mixture of Experts | replaced by **Soft-MoE layers** inside one end-to-end network (`moe_networks.py`), not a gate over frozen whole-policy experts |
| Scaling MSK complexity (80→290 muscles) | pushed to 416 muscles, and from CPU to 8,192 GPU envs |

The last three rows are genuine methodological divergence, not reimplementation. Kinesis's MoE is a *post-hoc distillation* of independently trained experts; musclemimic's is a *differentiable routing layer* trained jointly. Kinesis's negative mining is a discrete curriculum over dataset subsets; musclemimic's is a continuous prioritised-replay-like weighting that keeps a single policy.

---

## 6. MuscleMimic: quantitative accounting

### 6.1 Repository layout

```
musclemimic/                          <- this repo
├── loco_mujoco/     34,032 LOC       <- vendored LocoMuJoCo v1.0.1, stripped and extended
├── musclemimic/     24,758 LOC       <- new package; replaces LM's env/algo/wrapper layers
├── tests/           16,070 LOC       <- entirely new (no overlap with LM's 5,829-line suite)
├── scripts/          4,053 LOC       <- retargeting/benchmark/model-gen tooling
├── fullbody/           618 LOC       <- Hydra entrypoints + 10 configs
├── bimanual/           462 LOC       <- Hydra entrypoints + 4 configs
└── models/                           <- OSL prosthesis models (your dev branch)
                    ─────────
                      80,000 LOC total Python
```

Note the top-level structure: the fork **kept the `loco_mujoco` package name** but deleted its `environments/`, `algorithms/`, `core/wrappers/`, `core/mujoco_mjx.py`, and `core/reward/trajectory_based.py`, then re-created those under `musclemimic/`. The vendored `loco_mujoco/__init__.py` says so explicitly:

> "This package is a heavily modified fork of the original loco-mujoco library. Many core components have been rewritten or extended. Using original loco-mujoco package will lead to incompatibilities with this codebase."

### 6.2 The vendored `loco_mujoco/`: how much actually diverged

Bucketed by whitespace-and-blank-line-insensitive diff against upstream v1.1.0 (which slightly overstates divergence, since upstream also moved forward):

| Bucket | Files | Lines | Contents |
|---|---:|---:|---|
| **Byte-identical** | 25 | 2,227 | `domain_randomizer/{base,no_randomization}`, `initial_state_handler/base`, `reward/{base,utils}`, `terminal_state_handler/{base,no_terminal,traj}`, `terrain/dynamic`, `utils/{backend,decorators,env}`, `smpl/parser` (696 lines!), `smpl/utils/smoothing`, `utils/running_stats`, `utils/video`, `stateful_object`, LAFAN1 loader consts |
| **Trivial (≤20 semantic lines)** | 18 | 2,542 | `domain_randomizer/default` (839 lines, 3 changed), `reward/default` (513 lines, 2 changed), `control_functions/pd`, `initial_state_handler/{default,traj_init_state}`, `terminal_state_handler/height`, `terrain/{static,__init__}`, `visuals/scene`, `task_factories/{base,rl_factory}` |
| **Moderate (21–150)** | 14 | 3,835 | `core/mujoco_base` (60), `control_functions/default` (+incremental mode), `core/utils/math` (+65), `terrain/rough` (+91), `visuals/video_recorder` (+65), `datasets/data_generation/utils` (+123), `observations/visualizer` |
| **Heavy (>150)** | 10 | 25,325 | see below |
| **New files** | 2 | 103 | `terminal_state_handler/bimanual`, `datasets/__init__` |

The ten heavy files, with the actual nature of the change:

| File | LOC here | Upstream | Semantic Δ | What changed |
|---|---:|---:|---:|---|
| `smpl/const.py` | 17,046 | 189 | 17,042 | **Data, not code.** Motion-clip manifests: `KIT_KINESIS_TRAINING_MOTIONS` (972), `KIT_KINESIS_TESTING_MOTIONS` (108), `AMASS_BIMANUAL_TRAIN/TEST_MOTIONS`, `AMASS_TRANSITION_MOTIONS`, `AMASS_RANDOM_TRAINING_MOTIONS`, plus `_MINT` and `_MINT_STRAIGHT_FORWARDS` subsets |
| `smpl/retargeting.py` | 2,295 | 964 | 1,873 | **GMR-Fit pipeline added.** Upstream provided `fit_smpl_shape`, `fit_smpl_motion`, `to_t_pose`, `motion_transfer_robot_to_robot`, `extend_motion`, `load_retargeted_amass_trajectory`. New: `ensure_gmr_fitted_shape`, `fit_gmr_motion`, `get_gmr_fitted_shape_{dir,path}`, `max_penetration_with_floor`, `_compute_qvel_from_qpos`, `_record_site_kinematics`, `retarget_trajectory_for_bimanual`, `retarget_smpl_to_bimanual_via_intermediate`, GMR cache plumbing |
| `core/observations/base.py` | 1,544 | 1,158 | 412 | **Identical class list plus 6 new observation types** at the end: `ActuatorLength`, `ActuatorVelocity`, `ActuatorForce`, `ActuatorExcitation`, `ActuatorActivation`, `TouchSensor`. Everything before line ~1190 matches upstream within ±5 lines |
| `trajectory/dataclasses.py` | 1,570 | 1,220 | 592 | Same class/method inventory (diff of `^class`/`^    def` lists is 3 entries). Adds `TrajectoryCacheType` enum (sparse/dense caching), `scalar_first` quaternion handling; drops `interpolate_xmat`. Rest is type-hint modernisation (`Union[...]`→`|`) plus docstrings |
| `core/visuals/viewer.py` | 984 | 892 | 152 | Mimic-site visualisation, muscle/tendon rendering hooks |
| `core/observations/goals.py` | 676 | 1,113 | 649 | **A move, not a deletion.** `GoalTrajMimic` and `GoalTrajMimicv2` were lifted out to `musclemimic/core/goals/trajectory.py` and extended there |
| `trajectory/handler.py` | 459 | 341 | 274 | Adds `reached_trajectory_end`, `last_step_idx`, `get_traj_data_at`, `materialize_trajectory`, and pluggable start-index strategies (`_random_start_idx`, `_fixed_start_idx`, `_selected_idx`, `_default_idx`) |
| `task_factories/imitation_factory.py` | 407 | 287 | 384 | `dataset_group` support, GMR cache resolution, trajectory-cache-type wiring |
| `task_factories/dataset_confs.py` | 200 | 78 | 168 | Named dataset groups mapping to the `const.py` manifests |
| `utils/dataset.py` | 144 | 327 | 311 | **Shrank.** LocoMuJoCo's HuggingFace dataset-download machinery for its own robot datasets was removed |

**Reading of this table:** LocoMuJoCo's *architecture* is intact and load-bearing. The pluggable-component system — `Observation`/`ObservationType` registry, `Reward`, `ControlFunction`, `TerminalStateHandler`, `InitialStateHandler`, `DomainRandomizer`, `Terrain`, `StatefulObject`, the `info_property` descriptor, `TrajectoryHandler` — is used exactly as designed. MuscleMimic's MSK work is expressed as *new plug-ins registered into upstream's abstractions*, which is the strongest possible evidence that the architecture was a good fit.

Two clean illustrations:

`musclemimic/environments/humanoids/base_robot_humanoid.py` vs upstream — **byte-identical except one import line**:

```diff
-from loco_mujoco.environments import LocoEnv
+from musclemimic.environments import LocoEnv
```

`musclemimic/environments/base.py` (992 lines) vs upstream `loco_mujoco/environments/base.py` (906) — the same class `LocoEnv(Mjx)` with the same 30 methods in the same order (`load_trajectory`, `_is_done`, `_mjx_is_done`, `_simulation_post_step`, `create_dataset`, `play_trajectory`, `set_sim_state_from_traj_data`, `mjx_reset`, `_reset_carry`, `_modify_spec_for_mjx`, `generate`, all the `@info_property` accessors…). Substantive additions: `_update_info_dictionary`, and six new fields on `LocoCarry`:

```python
@struct.dataclass
class LocoCarry(MjxAdditionalCarry):
    traj_state: TrajState | None = None
    selected_traj_idx: jax.Array          # deterministic eval reset
    sampling_weights: jax.Array | None    # (num_envs, n_traj) adaptive sampling
    termination_threshold: jax.Array      # curriculum-controlled
    ema_done_counts: jax.Array | None     # adaptive sampling state
    ema_early_counts: jax.Array | None
    qvel_w_sum: jax.Array                 # reward curriculum
    root_vel_w_sum: jax.Array
```

versus upstream's:

```python
@struct.dataclass
class LocoCarry(MjxAdditionalCarry):
    traj_state: TrajState
```

That single diff is a fair summary of the whole project: same skeleton, threaded through with adaptive machinery.

### 6.3 The `musclemimic/` package: derived vs. new

Line-multiset similarity against the upstream file each module replaces (crude, but directionally right):

| musclemimic module | LOC | Upstream counterpart | LOC | Shared lines | Verdict |
|---|---:|---|---:|---:|---|
| `environments/humanoids/base_robot_humanoid.py` | 42 | same path | 42 | 31 (74%) | copy |
| `environments/base.py` | 992 | `environments/base.py` | 906 | 564 (57%) | extended copy |
| `algorithms/common/base_algorithm.py` | 178 | same | 138 | 101 (57%) | extended copy |
| `core/mujoco_mjx.py` | 755 | `core/mujoco_mjx.py` | 610 | 333 (44%) | heavily extended |
| `core/reward/trajectory_based.py` | 775 | same | 402 | 285 (37%) | extended |
| `core/goals/trajectory.py` | 585 | `core/observations/goals.py` §GoalTrajMimic | ~400 | 185 (32%) | extended |
| `algorithms/common/networks.py` | 308 | same | 125 | 87 (28%) | extended |
| `core/wrappers/mjx.py` | 882 | same | 261 | 162 (18%) | mostly new |
| `utils/metrics.py` | 840 | `utils/metrics.py` | 358 | 166 (20%) | mostly new |
| `algorithms/ppo/ppo.py` | 378 | `algorithms/ppo_jax.py` | 531 | 33 (9%) | rewrite |
| `rl_core/rollout_buffer.py` | 213 | `core/wrappers/rollout.py` | 146 | 8 (4%) | new |

And by subpackage:

| Subpackage | LOC | Character |
|---|---:|---|
| `algorithms/` | 5,503 | PPO skeleton derived from upstream `ppo_jax.py` / PureJaxRL; everything around it new (runner 1,572; checkpoint mgmt 413+158+146; Soft-MoE 322+56; curriculum 269; optimizer/Muon 203; adaptive sampling 115) |
| `core/` | 5,856 | goals (1,250), reward (1,418 incl. your CLF 643), terminal handlers (1,021), mjx backend (755), wrappers (882), site mapping (347), initial state (146) |
| `web_viewer/` | 4,790 | **entirely new** — C3D marker viewer, SMPL-X/H fitting, JAX Stage-I solver, browser retargeting |
| `environments/` | 2,803 | MyoFullBody 585, MyoBimanualArm 501+301, OSL variants 368 (yours), LocoEnv 992 (derived) |
| `utils/` | 2,425 | metrics 840, GMR cache 218, retarget comparison 753, demo cache, model utils |
| `runner/` | 2,121 | **entirely new** — Hydra engine, eval utils 983, checkpointing, validation video recorder, logging |
| `viewer/` | 614 | **entirely new** — Viser-based tendon/muscle viewer |
| `retargeting/` | 352 | new visualisation CLI |
| `rl_core/` | 233 | new rollout buffer |

---

## 7. What was actually required to train MSK models

This is the core of the question, and the answer is **not a new RL algorithm**. The two items that cost the most code are §7.2 (the Warp reset) and §7.8 (retargeting).

### 7.1 Muscle-state observations *(new — ~330 lines)*

LocoMuJoCo has **no** observation type that exposes actuator internals. For a torque-controlled robot you don't need one. For a Hill-type muscle model the policy is controlling a system with its own first-order activation state, and Eq. 1 in the paper says that state is not recoverable from joint kinematics.

Added to `loco_mujoco/core/observations/base.py`, registered through upstream's `ObservationType` registry:

- `ActuatorLength` — muscle-tendon unit length *l*
- `ActuatorVelocity` — contraction velocity *l̇* (positive = lengthening)
- `ActuatorForce` — *F* (negative by convention; muscles pull)
- `ActuatorExcitation` — neural drive *u* ∈ [−1,1] from `data.ctrl`
- `ActuatorActivation` — activation state *a* from `data.act`
- `TouchSensor` — contact normal-force magnitude (feet/toes)

Toggled per-observation in `MyoFullBody.__init__` (`enable_muscle_length_observations`, …) and enabled in `fullbody/conf_fullbody.yaml`. With 416 muscles, turning on all five muscle channels adds ~2,000 observation dimensions.

`fullbody/conf_fullbody.yaml` sets all five to `true` plus touch sensors.

### 7.2 A working MuJoCo-Warp backend *(heavily extended — ~900 lines across two files)*

This is where the "order-of-magnitude speedup" claim actually gets paid for, and it is not a one-liner.

The problem: MJX with `impl='warp'` does not implement `mjx.put_data`. LocoMuJoCo's design resets an environment by re-uploading a CPU `MjData` template (`data = self._first_data`), so on Warp its reset is simply broken (upstream zeroes `qpos`, and its own comment concedes this). Reference-state initialisation from a 972-clip dataset makes reset the single most important operation in the loop, so "reset is error prone" is not survivable.

What `musclemimic/core/mujoco_mjx.py` does instead:

- `_get_mjx_backend()` — explicit `warp`/`jax` selection with CUDA and `warp-lang` preflight checks.
- Sizes the contact arena explicitly (`nconmax`, `njmax`, `naconmax`) and scales it with `num_envs` — necessary because MyoFullBody has full self-collision across every internal geometry, so contact counts are large and variable.
- Constructs fresh `Data` via `mjx.make_data(..., impl='warp', nconmax=…, njmax=…, naconmax=…)` per reset **outside** `vmap` and hydrates it field-by-field (`qpos`, `qvel`, `ctrl`, `act`, `xpos`, `xquat`, `site_xpos`, `cvel`, `xipos`, `subtree_com`, and the reshaped `xmat`/`ximat`/`site_xmat`), rather than `put_data`.
- Drops upstream's shared `self._first_data` entirely, with the comment "This avoids sharing Data objects across vmapped instances (causes BatchTracer leaks)."
- Raises an explicit `NotImplementedError` from `_mjx_reset_in_step` under Warp and routes resets through a new wrapper instead.

That wrapper is `AutoResetWrapper` in `musclemimic/core/wrappers/mjx.py` — ~330 of the file's 882 lines, with no upstream counterpart. It implements out-of-band reset: `_compute_reset_candidates`, `_make_where_done`, `_apply_carry_reset`, `_apply_autoreset`, `_step_with_autoreset`, plus observation-buffer surgery (`_find_observation_buffer`, `_replace_observation_buffer`) so history-stacked observations reset coherently. The wrapper stack otherwise mirrors upstream's exactly (`LocoMjxWrapper`, `BaseWrapper`, `LogWrapper`, `NStepWrapper`, `VecEnv`, `NormalizeVecReward`), with two more additions: a `reset_to(rng_key, traj_idx)` method threaded through every wrapper (deterministic per-clip reset, required for reproducible per-motion evaluation), and `step_with_transition`.

`tests/unit/test_mjx_reset.py` (1,986 lines) and `tests/unit/test_warp_backend.py` (837 lines) exist because of this.

### 7.3 MSK-appropriate reward terms *(extended — +373 lines)*

`MimicReward` is **upstream's** class, and the whole exponential-kernel structure with `*_w_exp` / `*_w_sum` coefficient naming is upstream's. musclemimic's additions:

| Addition | Why MSK needed it |
|---|---|
| `root_pos_w_exp` / `root_pos_w_sum`, `root_vel_w_exp` / `root_vel_w_sum` (local `[vx, vy, yaw_rate]`) | free-root MSK locomotion needs explicit root tracking; upstream tracked only qpos/qvel/sites |
| `activation_energy_coeff` | the over-actuation term from Kinesis — 416 muscles admit infinitely many activation patterns for the same motion; without it antagonists co-contract |
| `joint_torque_weights: dict[joint_name, weight]` | per-joint torque penalties (materialised into an `nv`-length vector at construction) |
| `use_mean_exp_reward` | `mean(exp(-βd))` vs upstream's `exp(-β·mean(d))`; changes gradient shape when averaging over thousands of parallel envs |
| fixed-base support | upstream unconditionally does `np.concatenate(quat_in_qpos)` and `mj_jntname2qvelid(root_free_joint_xml_name)`; MyoBimanualArm has no free joint, so both throw. Now guarded with empty-mask fallbacks |
| `create_site_mapper` | maps environment mimic-site order ↔ trajectory site order; the two disagree once you retarget across MSK morphologies |
| root-XY offset correction (`_root_qpos_ids_xy`) | trajectory qpos is world-frame; episodes reset to origin. Without subtracting the init offset, qpos error is dominated by a constant translation |
| `_extra_reward_terms` hook + `_HAS_EXTRA_TERMS` class flag | subclass extension point, gated at trace time so plain `MimicReward` pays no JIT cost. This is the hook your CLF reward hangs off |

### 7.4 Relative-site early termination *(new — 1,021 lines)*

LocoMuJoCo terminates on height (`HeightBasedTerminalStateHandler`) or root pose (`RootPoseTrajTerminalStateHandler`). Neither works for MSK imitation over hundreds of clips: absolute world-frame error accumulates as root drift even when the gait is perfectly formed, so absolute termination kills biomechanically valid episodes.

New handlers in `musclemimic/core/terminal_state_handler/`:

- `MeanRelativeSiteDeviationTerminalStateHandler` — mean Euclidean distance across all mimic sites, **expressed relative to the root frame**; the paper's δ_site. Tolerates global drift, enforces local posture. (Used for validation in `conf_fullbody.yaml`, threshold 0.5 m.)
- `MeanRelativeSiteDeviationWithRootTerminalStateHandler` — the above **plus** the paper's δ_root world-frame pelvis check and a root-orientation check. This is what training actually uses (`mean_site_deviation_threshold: 1.0`, `root_deviation_threshold: 1.0`, `root_orientation_threshold: 1.0`).
- `MeanSiteDeviationTerminalStateHandler` — absolute (non-relative) variant.
- `EnhancedFullBodyTerminalStateHandler` — subclasses `HeightBasedTerminalStateHandler`, adding per-site deviation checks on top of the height check.
- `BimanualTerminalStateHandler` (in the vendored tree) — fixed-base variant.

All four are registered into upstream's `TerminalStateHandler` registry at the bottom of `enhanced_fullbody.py`, so they are selectable by name from YAML exactly like upstream's own handlers.

Also: the threshold is **not a constant**. It lives on `LocoCarry.termination_threshold` and is driven by the curriculum in §7.6.

### 7.5 n-step goal lookahead *(implemented — the upstream TODO)*

Upstream `GoalTrajMimic.__init__` literally reads:

```python
self.n_step_lookahead = 1   # todo: implement n_step_lookahead
```

`musclemimic/core/goals/trajectory.py` implements it, and this is exactly the observation structure in the paper's Fig. 11:

- `n_step_lookahead` (5 in the config) with `n_step_stride` (20 steps = 0.2 s at 100 Hz control).
- `enable_motion_phase` — normalised progress ∈ [0,1] through the clip.
- `use_concise_lookahead` — per future step, emit only root-position delta (3) + root-velocity delta (6) + site relative positions, instead of the full qpos+qvel+rpos+rangles+rvel block. With 17 sites and 72 DoF, the full form would blow up the observation by 5×; the concise form is what makes 5-step lookahead affordable.
- `enable_mimic_site_rpos_observations` — toggle current-sim site rpos.
- Site-index mapping via `create_site_mapper` + `attach_trajectory_sites`.

Paired with `NStepWrapper(split_goal=True)` for state-history stacking that keeps goal and proprioception in separate blocks.

### 7.6 Adaptive sampling and curricula *(new — ~400 lines)*

Three independent adaptive loops, all threaded through `LocoCarry` so they survive `vmap`/`scan`:

**Adaptive trajectory sampling** (`algorithms/common/adaptive_sampling.py`). Per-trajectory done and early-termination counts are accumulated with `jax.ops.segment_sum` over the rollout, EMA'd, converted to a rate, raised to a prioritisation exponent α, normalised, then mixed with a uniform floor:

```python
rate_hat  = new_ema_early / (new_ema_done + eps_div)
priorities = jnp.power(rate_hat + eps_pow, alpha)
weights_1d = (1 - floor_mix) * normalized + floor_mix * uniform
```

This is the scalable replacement for Kinesis's hard negative mining: hard clips get sampled more, but one policy covers everything, and `floor_mix` guarantees no clip is starved. Non-finite guards throughout because it runs inside the JIT'd training step.

**Adaptive termination threshold** (`curriculum.py`). A one-way ratchet: if EMA'd early-termination rate stays below `low_band` for `consecutive_k` updates, tighten the threshold (`thr *= 0.95`); if above `high_band`, loosen — but never above the initial value. Starts loose so a fresh policy gets episodes long enough to learn from, then tightens as tracking improves.

**Reward-weight curriculum** (`RewardCurriculumState`). Grows `qvel_w_sum` and `root_vel_w_sum` multiplicatively (η = 0.02 per trigger) from 0.1 toward 0.4 once termination rate is stably low. Position tracking first, velocity tracking once posture is stable.

All three are `enabled: false` in `fullbody/conf_fullbody.yaml` — they are tools for the hard cases, not always-on.

### 7.7 Strictly-on-policy PPO and the training loop *(rewrite — 1,572 lines)*

The algorithmic core is recognisably upstream's (which is recognisably PureJaxRL's): `_update_step` → `_env_step` scan → GAE → `_update_epoch` → `_update_minibatch` → `_loss_fn`, plus a `_evaluation_step` and an `io_callback` for logging. `musclemimic/algorithms/ppo/` splits this across `runner.py` / `ppo.py` / `loss.py` / `config.py` / `moe.py` / `checkpoint.py` / `inference.py` and adds:

- **`update_epochs: 1`** — the paper's central training finding. Not code, but the reason all the rest exists: at E=1 you need ~8,000 parallel envs to get a usable batch, which forces the Warp work in §7.2.
- `num_minibatches: 128` with `num_steps: 20`, `num_envs: 8192` — the batch-size sweep in the paper's Fig. 4, where larger batches give higher asymptotic reward, lower KL, and smoother convergence of the policy standard deviation.
- **Muon optimizer** (`optimizer.py`): `optax.contrib.muon` for 2D weights, Adam for 1D (biases, LayerNorm). Plus linear and warmup-cosine schedules, `min_lr_ratio`, adaptive LR.
- **Gated residual actor-critic** (`ResidualFCNet` in `networks.py`): 16×1024 SiLU blocks with LayerNorm, a learnable residual gate initialised to σ(−2) ≈ 0.12, and near-zero orthogonal init (gain 0.01) on each block's second layer, so the net starts as approximately the identity. Upstream's `ActorCritic` is a plain 2-layer MLP. Upstream's `RunningMeanStd` (Welford) is reused verbatim.
- **Soft-MoE layers** (`moe_networks.py`, `ppo/moe.py`): `SoftMoELayer`/`SoftMoEActorCritic` with load-balance loss, gate entropy, expert-utilisation variance, top-2 usage metrics.
- **Checkpoint/resume infrastructure** (`checkpoint_manager.py`, `checkpoint_utils.py`, `checkpoint_hooks.py`, `auto_resume`): config-hash run IDs, `max_checkpoints_to_keep`, async checkpointing, LR-schedule reset control, preemption robustness. A 2-billion-step job needs this; a 36-minute benchmark run does not.
- **Validation-all-trajectories** (`_run_validation_all`, `_rollout_eval_all_batch`, `_reduce_eval_all_batch`): evaluate every held-out clip in one batched jitted rollout, with per-clip length masking. Plus `validation_video_recorder.py` for periodic W&B video.
- `MetricsHandler` extended 44 → 786 lines with per-site, per-quantity metrics (`JointPosition`, `JointVelocity`, `RelSitePosition`, `RelSiteVelocity`, `RelSiteOrientation` × `EuclideanDistance`).

### 7.8 Retargeting that respects biomechanics *(extended — +1,331 lines)*

The paper's Mocap-Body path **is** LocoMuJoCo's: SMPL-H shape fitting in T-pose to get (β, s, Δp, ΔR), then MuJoCo mocap-body IK. The paper cites loco-mujoco for exactly this step.

The problem with it on MSK models: MuJoCo mocap bodies don't enforce joint limits, so 12.3% of frames violated joint limits on the KINESIS set, and 30% of motions contained tendon-length jumps. A motion the model physically cannot reach makes the imitation target unachievable — the paper measures this directly (Mocap-Body retargeting gives joint-angle errors >5σ worse and much lower returns).

GMR-Fit (`ensure_gmr_fitted_shape`, `fit_gmr_motion`) reuses the SMPL fit but drives Mink-based IK through General Motion Retargeting with explicit **equality constraints between dependent joints** (shoulder, knee — MSK models have coupled joints, which is why MyoFullBody has 123 joints but only 72 DoF).

| Method | Joint-limit viol. | Tendon-jump rate | RMSE | Time / frame |
|---|---:|---:|---:|---:|
| Mocap-Body (inherited from LocoMuJoCo) | 12.26% | 30.14% | 0.039 m | 0.076 s |
| GMR-Fit (new) | 0.27% | 3.20% | 0.025 m | 0.251 s |

Measured over the 972-motion KINESIS set (MuscleMimic paper, Table 4). GMR-Fit costs ~3× the retargeting time and buys a 45× reduction in joint-limit violations.

Plus: `max_penetration_with_floor` and floating correction (post-processing in the paper's Fig. 10), `_compute_qvel_from_qpos` (finite-difference velocities after resampling to 100 Hz), `_record_site_kinematics`, dataset-group caching (`gmr_cache.py`, `musclemimic-download-gmr-caches`), and the bimanual intermediate-retargeting path for the fixed-base model.

### 7.9 Model construction *(new — in the models package)*

`MyoFullBody._apply_spec_changes` is 180 lines of `MjSpec` surgery:

- Delete 40 named finger joints, 62 named finger muscles, and their tendons when `disable_fingers=True` (416 → 354 actuators, 123 → 83 joints, 72 → 32 DoF). Uses exact-name matching, with a comment noting substring matching would catch hip joints.
- Add 17 mimic sites at anatomical landmarks by body name (`body2sites_for_mimic`).
- Rewrite every actuator with `dyntype == mjDYN_MUSCLE` to `ctrlrange = [-1, 1]`, `ctrllimited = True` — the direct-actuation convention, so the policy's `tanh`-free Gaussian output maps linearly to excitation.

The models themselves ship as a separate package (`musclemimic_models>=1.0.2`), mirroring upstream's own 2026-03 move to `loco-mujoco-models`.

### 7.10 Timestep and control rate

`MyoFullBody` defaults to `timestep=0.002` (500 Hz physics) with `n_substeps=5` → **100 Hz control**. Kinesis ran 150 Hz physics / 30 Hz control. The higher control rate is needed because muscle activation time constants are τ_act = 10 ms / τ_deact = 40 ms; at 30 Hz you cannot resolve activation onset.

---

## 8. What did *not* change: the training behaviour is LocoMuJoCo's

Worth stating plainly, because it is the honest answer to "or is the underlying training behavior the same as locomujoco?"

**Yes — substantially the same.** Compare LocoMuJoCo's own DeepMimic example config with MuscleMimic's:

| | `loco-mujoco/examples/training_examples/jax_rl_mimic/conf.yaml` | `musclemimic/fullbody/conf_fullbody.yaml` |
|---|---|---|
| factory | `ImitationFactory` | `ImitationFactory` |
| backend | `use_mjwarp: True`, `nconmax: 7000` | `mjx_backend: warp` |
| envs | `num_envs: 2048` | `num_envs: 8192` |
| goal | `GoalTrajMimic` | `GoalTrajMimic` (+ `n_step_lookahead: 5`) |
| reward | `MimicReward` with `qpos_w_sum`, `qvel_w_sum`, `rpos_w_sum`, `rquat_w_sum`, `rvel_w_sum` | `MimicReward` with the same five **plus** `root_pos_w_sum`, `root_vel_w_sum`, `activation_energy_coeff` |
| control | `DefaultControl` | `DefaultControl` |
| init state | trajectory-based (default) | `TrajInitialStateHandler` |
| algorithm | PPO (JAX, fused JIT) | `PPOJax` (JAX, fused JIT) |
| epochs | `update_epochs: 4` | `update_epochs: 1` |
| net | `[512, 256]` | `[1024] × 16`, gated residual |
| optimizer | Adam | Muon (2D) + Adam (1D) |

These are the same experiment. The reward formulation, the goal formulation, the reference-state initialisation, the wrapper stack, the fused env+train JIT structure, the observation registry, the `MjSpec`-based model modification, the `TrajectoryHandler` — all upstream, all unchanged in kind.

What is *not* the same:

1. **Scale of parallelism**, and the reset correctness work that makes 8,192 muscle-actuated envs with full self-collision viable on Warp.
2. **E=1 instead of E=4**, which the paper argues is specific to muscle dynamics.
3. **What the policy observes** (muscle state) and **what it is penalised for** (activation energy).
4. **What ends an episode** (relative-site deviation, not absolute pose).
5. **Which clip gets sampled next** (adaptive), and **how strict the bar is** (curriculum).
6. **Production hardening**: preemption-safe resume, per-clip deterministic eval, validation over the full test set, video logging.

A useful mental model: **LocoMuJoCo supplied the simulator abstraction and the imitation-learning recipe; MuscleMimic supplied the embodiment, the data pipeline, and the engineering to run that recipe two orders of magnitude harder.**

For completeness, LocoMuJoCo does already ship a muscle-actuated environment: `SkeletonMuscle` (`use_muscles=True`), documented as "92 muscles on the lower limb. The upper body is torque actuated" — the Hamner/Delp model from the 2023 paper. And `MyoSkeleton` (151 joints) is **position-controlled per joint**, not muscle-driven. So the jump is 92 lower-limb muscles → **416 full-body muscles**, plus everything muscle-specific listed above, none of which upstream has.

---

## 9. What MuscleMimic dropped

The fork narrowed as much as it widened. Removed from the vendored `loco_mujoco/`:

- **All 16 robot/quadruped environments** — Atlas, Talos, H1, H1v2, G1, Apollo, BoosterT1, ToddlerBot, FourierGR1T2, SkeletonTorque, SkeletonMuscle, MyoSkeleton, A1, Go2, Spot, AnymalC (9,130 LOC). `musclemimic/environments/humanoids/__init__.py` registers exactly five models (MyoBimanualArm, MyoFullBody, and your three OSL variants).
- **GAIL and AMP** (`gail_jax.py`, `amp_jax.py`). Only PPO survives — adversarial IL is out of scope for kinematic imitation of MSK models.
- **The Gymnasium wrapper** (`core/wrappers/gymnasium.py`). Pure-JAX only, plus CPU MuJoCo for eval/viewing (`fullbody/eval.py --use_mujoco`).
- **LocoMuJoCo's HuggingFace dataset downloaders** for its own robot datasets (`utils/dataset.py` shrank 327 → 144); replaced with the GMR-cache and demo-cache mechanisms.
- **LocoMuJoCo's entire test suite** (5,829 lines: `test_observation.py`, `test_reward.py`, `test_goals.py`, `test_mjx.py`, `test_trajectory.py`, …). Zero filename overlap with musclemimic's 16,070-line suite, which targets the new machinery instead: `test_mjx_reset.py` (1,986), `test_warp_backend.py` (837), `test_n_step_lookahead.py` (761), `test_curriculum.py` (621), `test_mimic_reward.py` (602), `test_muscle_observations.py` (488), `test_adaptive_sampling.py` (453), `test_split_goal_integration.py` (370), `test_resume_checkpoint_scenarios.py` (725).

That last point is worth flagging as a risk: because upstream's regression tests were dropped, silent behavioural drift in the *inherited* components (domain randomizer, terrain, control functions, math utils) would not be caught here.

---

## 10. Your dev-branch work on top

For orientation, `git diff upstream/main..HEAD` (i.e. `5d4c8af..6561833`) — your changes on top of amathislab/musclemimic:

**159 files, +19,270 lines** total; of that, **+6,110 / −76 lines of Python** across 33 files. The bulk of the non-Python delta is the `models/myoLeg80_OSL_KA/` mesh and XML assets.

New environments (`musclemimic/environments/humanoids/`):
- `myoleg80_osl_ka.py` (140) — MyoLeg80 with an OSL knee-ankle prosthesis
- `osl_fullbody.py` (146) — MyoFullBody + OSL prosthesis
- `osl_fullbody_robot.py` (76) — procedurally generated torque-actuated equivalent

New reward:
- `core/reward/clf.py` (466) + `clf_math.py` (177) — Control-Lyapunov-Function reward, hung off the `_extra_reward_terms` hook added upstream

Tooling (`scripts/`):
- `diagnose_osl_ka_episode_length.py` (1,508), `benchmark_training_speed.py` (625), `tune_osl_fullbody_robot.py` (484), `analyze_osl_fullbody_robot_actuators.py` (373), `generate_osl_fullbody_robot.py` (348), `project_gmr_cache_to_osl_ka.py` (323)

Tests: `test_clf_reward.py` (541), `test_osl_fullbody_robot.py` (253), `test_benchmark_training_speed.py` (141), `test_osl_fullbody_robot_tuning.py` (115)

Configs: five `fullbody/conf_*` variants, `loco_mujoco/smpl/robot_confs/MyoLeg80_OSL_KA.yaml`

Modifications to inherited code: `imitation_factory.py` (+70), `trajectory_based.py` (+115), `ppo.py` (+49), `metrics.py` (+36), `runner.py` (+34), `retargeting/visualize.py` (+34), `site_mapping.py` (+31), `networks.py`/`moe_networks.py` (+21 each), `environments/base.py` (+14).

So your work sits at the same architectural layer MuscleMimic itself does relative to LocoMuJoCo: new embodiment registered through the existing abstractions, new reward through the provided hook, plus tooling. You have not had to touch the simulation backend, the wrapper stack, or the PPO core — which is a reasonable signal that the layering holds.

---

## 11. Practical implications

1. **When debugging, know which layer you are in.** If a bug is in domain randomization, terrain sampling, `TrajectoryHandler` indexing, SMPL parsing, PD control, or the observation registry — that is LocoMuJoCo code, and upstream's repo, docs (readthedocs), tests, and issue tracker apply. If it is in reset, muscle observations, GMR retargeting, termination, adaptive sampling, checkpoint resume, or the PPO runner — that is MuscleMimic-only and you are on your own.

2. **Upstream LocoMuJoCo is a live source of fixes but not a drop-in merge.** v1.1.0 (Mar 2026) brought a `mjspec` interface migration, `mujoco>=3.5.0`, a viewer fix for the new MuJoCo version, and its own MjWarp backend. This repo pins `mujoco==3.4.0`. Cherry-picking specific fixes from `robfiras/loco-mujoco` into `loco_mujoco/` is feasible for the untouched 4,769 lines; anything touching `observations/`, `goals.py`, `trajectory/`, or `smpl/` will conflict heavily.

3. **Kinesis is a reference implementation and a baseline, not a dependency.** Useful for: the EMG validation benchmark, the reward-weight values in its Table II, the direct-vs-PD actuation comparison, and the KIT-Locomotion curation rationale. Not useful as code.

4. **The `_extra_reward_terms` hook is the sanctioned extension point** for reward work (you are already using it for CLF). It is gated by the class-level `_HAS_EXTRA_TERMS` flag so plain `MimicReward` pays no trace cost — keep that pattern.

5. **`update_epochs: 1` is load-bearing, not a leftover.** If you find yourself raising it to get faster early progress, expect the KL blow-up in the paper's Fig. 3C. The compensating lever is `num_envs`, not epochs.

6. **The dropped upstream test suite is a real gap.** If you modify anything in the vendored `loco_mujoco/core/` (control functions, math, terrain, observations), there is no regression net. Porting `loco-mujoco/tests/test_{observation,reward,control_functions,domain_randomizer}.py` would be cheap insurance.

---

## Appendix: reproducing these numbers

```bash
MM=~/src/00_AMBER/01_projpegleg/musclemimic
UP=~/src/00_AMBER/00_projbackflip/loco-mujoco

# fork point
grep __version__ $MM/loco_mujoco/__init__.py            # 1.0.1
git -C $UP log --oneline --all | grep -i "version bump 1.0"

# per-file divergence of the vendored package
for f in $(cd $MM/loco_mujoco && find . -name '*.py' | sort); do
  a=$MM/loco_mujoco/$f; b=$UP/loco_mujoco/$f
  [ -f "$b" ] || { echo "NEW  $f"; continue; }
  printf "%-55s %5s %5s  sem=%s\n" "$f" "$(wc -l <$a)" "$(wc -l <$b)" \
    "$(diff -w -B "$b" "$a" | grep -c '^[<>]')"
done

# what moved rather than vanished
grep -n '^class ' $UP/loco_mujoco/core/observations/goals.py    # GoalTrajMimic at 715
grep -n '^class ' $MM/musclemimic/core/goals/trajectory.py      # GoalTrajMimic at 26

# the implemented TODO
grep -n 'n_step_lookahead' $UP/loco_mujoco/core/observations/goals.py | head -2

# upstream's broken warp reset
sed -n '/_mjx_reset_in_step/,/^    def mjx_step/p' $UP/loco_mujoco/core/mujoco_mjx.py

# your own delta
git -C $MM diff --stat upstream/main..HEAD -- '*.py'
```
