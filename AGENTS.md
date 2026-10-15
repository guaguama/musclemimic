# MuscleMimic Development Guidelines

This file applies to the entire repository. Add a nested `AGENTS.md` only when a
subsystem needs stricter or more specific guidance.

## Project priorities

MuscleMimic is a JAX-based research codebase for muscle-actuated motion
imitation. Preserve these properties when changing it:

1. Correct physical and numerical behavior.
2. Reproducible training, evaluation, and checkpoint resume.
3. JAX/MJX/Warp performance without obscuring correctness.
4. Clear module ownership and stable public interfaces.
5. Small, reviewable changes with focused tests.

Do not trade correctness or reproducibility for a cleaner-looking abstraction.
When a refactor changes results, configuration semantics, compilation behavior,
or checkpoint compatibility, treat it as a behavior change and document it.

## Repository architecture

Use this target dependency direction:

```text
fullbody/ and bimanual/ entry points, scripts, analyses, examples
                              |
                              v
              runner, evaluation, and visualization
                              |
                              v
        algorithms, rl_core, environments, and musclemimic/core
                              |
                              v
       loco_mujoco simulation, trajectory, and data primitives
```

Responsibilities:

- `loco_mujoco/`: generic simulation, control, observation, trajectory, dataset,
  terrain, and retargeting primitives. It should not depend on application-level
  `musclemimic` training or runner code.
- `musclemimic/core/`: MuscleMimic-specific MJX state, goals, rewards, terminal
  handlers, and wrappers.
- `musclemimic/environments/`: environment and robot composition. Do not put
  training loops, logging backends, or CLI behavior here.
- `musclemimic/algorithms/` and `musclemimic/rl_core/`: learning algorithms,
  networks, optimizers, and rollout data structures. Keep environment-specific
  policy out of reusable algorithm code where practical.
- `musclemimic/runner/` and `musclemimic/evaluation/`: host-side orchestration,
  lifecycle, logging, checkpoint coordination, and evaluation.
- `musclemimic/viewer/` and `musclemimic/web_viewer/`: presentation and
  interactive tooling. Core simulation and algorithm modules must not depend on
  viewers.
- `fullbody/` and `bimanual/`: thin Hydra entry points and experiment
  configuration, not alternate implementations of shared behavior.
- `tests/`: behavior and regression coverage. Mirror source boundaries where it
  makes tests easier to find.

Some imports currently run against this direction. Treat them as existing
technical debt, not precedent. Do not introduce a new reverse dependency to
complete an unrelated task. If a boundary must be crossed, expose a small public
interface or move the genuinely shared concept to a neutral lower layer.

## Module and API design

- Depend on public interfaces. Do not import a leading-underscore symbol from
  another module.
- Keep `__init__.py` exports intentional and small. Avoid new wildcard imports
  and compatibility re-export chains.
- Before changing a public function, class, config key, registry name, or
  serialized field, search its uses in source, configs, tests, scripts, and
  examples.
- Preserve public signatures and configuration/checkpoint compatibility by
  default. If a break is necessary, provide migration guidance and, when
  practical, a deprecation path.
- Prefer cohesive modules over generic dumping grounds such as `utils.py`.
  Separate orchestration, domain computation, I/O, and presentation when they
  evolve independently.
- Extract one boundary at a time. First characterize existing behavior with a
  test; then move code without mixing in algorithmic changes.
- Avoid new dependencies unless the standard library or an existing dependency
  cannot reasonably solve the problem.
- Comments should explain intent, constraints, units, or non-obvious edge cases,
  not restate the code.
- Add type annotations to new public interfaces. Put types in signatures rather
  than repeating them in docstrings.
- Document units and array shape conventions for physical quantities at public
  boundaries, for example `[m]`, `[rad]`, `[N]`, and
  `[num_envs, num_joints]`.

## Documentation writing

Repo documentation (`docs/`, READMEs, docstrings) follows a strict prose style: no em or en
dashes, no tables or emoji, active voice. The full rule is in
`.memory/documentation-writing-style.md`; read it before writing documentation.

## Python project quality and code clarity

Style baselines (PEP 8, PEP 257, Google style, pytest conventions) and the reproducibility
checklists this project aligns with are in `.memory/python-quality-baselines.md`.

- Avoid non-obvious magic numbers in logic. Use named `UPPER_CASE` constants for domain
  thresholds, schedule points, buffer sizes, and algorithmic defaults; if a literal must stay
  inline, comment its provenance.
- Prefer explicit keys or fields over positional tuples for logs, rollout metrics, and trainer
  outputs. Metric and logging schema changes need tests and migration notes.
- Document units and scale at public boundaries, e.g. `[m]`, `[rad]`, `[N]`, `[step]`,
  `[num_envs, num_joints]`.
- Prefer simple code over abstractions; a helper must reduce complexity and test burden.
- Keep tests deterministic and isolated; use `tmp_path`/`tmp_path_factory` for filesystem tests.

## JAX, MJX, and numerical code

- Functions transformed by `jax.jit`, `jax.vmap`, or `jax.grad` must be pure.
  Pass state explicitly and return updated state. Keep file I/O, logging,
  mutation, and other host side effects outside transformed functions.
- Thread PRNG keys explicitly and split them before independent uses. Never
  silently reuse a key unless identical samples are intentional and documented.
- Do not convert tracers to NumPy arrays or use traced values in Python boolean
  control flow. Use JAX arrays and `jax.lax` control-flow primitives, or make
  truly static values explicit.
- Keep array shapes and pytree structure stable across compiled steps. Be
  deliberate about static arguments; high-cardinality static values can cause
  repeated compilation.
- Place JIT boundaries around cohesive outer computations. Do not scatter new
  `jax.jit` decorators through low-level helpers without measuring compilation
  and runtime effects.
- Keep dtype, precision, device placement, and host-device transfers explicit
  when they affect results or performance.
- Preserve backend behavior across supported MuJoCo, MJX, and Warp paths. A fix
  for one backend must not silently route another backend through incompatible
  code.
- Use `jax.debug.print` for runtime values inside transformed code. Remove
  temporary debug output before finishing the change.
- Warm up compiled functions and call `.block_until_ready()` when timing JAX
  execution. Report compilation time separately from steady-state runtime.
- Numerical changes need tests with justified tolerances. Do not loosen a
  tolerance merely to hide a regression or backend mismatch.

## Configuration, data, and checkpoints

- Treat Hydra configuration keys and defaults as user-facing interfaces. Keep
  entry-point configs declarative and put shared behavior in Python modules.
- Do not embed machine-specific dataset, asset, cache, or output paths in source
  or committed configs.
- Preserve deterministic seed flow through environment reset, trajectory
  sampling, policy action, and evaluation.
- Treat checkpoint layout and resume behavior as compatibility-sensitive. Test
  fresh start, save/load, and resume when changing state or manifests.
- Do not commit generated datasets, model assets, checkpoints, recordings,
  caches, or experiment output.

## Development workflow

This repository uses Python 3.11, `uv`, Ruff, and pytest. Follow
`CONTRIBUTING.md` for setup and the authoritative workflow.

```bash
make install-dev
make precommit-install

# Fast feedback: run the narrowest relevant test first.
make test PYTEST_ARGS='tests/unit/test_relevant_area.py -q'

# Repository checks.
make lint
make test
make ci
```

- Developer tools are expected in `.venv/bin`; use the Make targets rather than
  inventing a parallel environment workflow.
- `make lint` intentionally covers a curated migration subset. Do not broaden
  that scope or reformat unrelated files in a feature or bug-fix change.
- Mark tests requiring external resources, special assets, or hardware as
  integration tests. Keep unit tests local and deterministic.
- For a regression fix, verify that the new test fails without the fix and
  passes with it.
- Test both eager and JIT-transformed behavior when tracing is relevant.
  Disabling JIT alone is not proof that compiled execution works.
- Prefer targeted tests during iteration; run `make ci` before handing off a
  change when the local environment supports it.
- If a required check cannot run because of missing hardware, credentials,
  gated data, or optional dependencies, state exactly what was not run and why.

## Change hygiene

- Inspect the working tree before editing and preserve unrelated user changes.
- Keep refactors separate from behavior changes when practical.
- Avoid drive-by formatting, renaming, or cleanup outside the requested scope.
- Update tests and user-facing documentation when commands, configs, public
  behavior, or expected outputs change.
- In the handoff, summarize the behavior changed, list validation performed,
  and call out remaining risks or checks not run.

## Agent memory

`.memory/` is shared working memory for every agent on this repo (Claude Code, Codex, or any
other). It exists so the agent picking up a thread does not re-derive what the previous one
already worked out.

### Start of session

Read `.memory/MEMORY.md`, then open the memories your task touches. This step is mandatory, and
sync does not perform it. Sync reports whether two folders agree; it says nothing about what this
session has read. "All in sync" on a cold session means nothing has been read.

Claude Code keeps a second, native copy of these memories and imports changes through a
`SessionStart` hook. Other agents have no native copy; `.memory/` is their only store, so they
need no sync at startup.

### Commands

The Make/`.venv` workflow above does not apply to `.memory/sync.py`. It is a dependency-free
stdlib script that must keep working when the project environment is broken, half-synced, or
absent, which is when memory matters most. Call it with `python3`.

- Claude Code: nothing at startup (the `SessionStart` hook pulls) and nothing after writing a
  memory (the `Stop` hook pushes).
- Any other agent: read `MEMORY.md` at startup; after writing or editing a memory, run
  `python3 .memory/sync.py --repo-only`.

`--repo-only` rebuilds `.memory/MEMORY.md` and writes nothing outside the working tree, so it
needs no permissions beyond the repo. Claude imports those changes at its next session start.

Sync reports changed memories as one `name: description` line each, never as full text: it runs
before a session has a topic, so it cannot judge relevance. Open the files your task touches.
`--print-bodies` dumps full text. `--resolve` cannot be combined with `--pull` or `--push`,
because it moves whichever side it is told to keep.

### What is worth writing

Write a memory when a session produced something the next agent would otherwise rediscover the
hard way:

- a change and the reasoning behind it, especially a decision that could plausibly have gone
  the other way
- a problem hit and how it was resolved, including wrong turns worth not repeating
- behavior of this codebase that is not visible from the code you would naturally read
- a correction to something an earlier memory got wrong

Do not write what the repo already records: file layout, what a function does, anything
recoverable from `git log`.

Write for the actual reader: a different agent, starting cold, mid-thread, with no transcript.
State why, since the diff already shows what. Say plainly what is done, what is in progress, and
what is untested.

### When to write

Write at checkpoints, not after every turn: a task or sub-task is finished, a decision is
settled, a dead end is confirmed, or the user is handing off or ending the session. Also write
whenever the user asks. Between checkpoints, keep notes in the conversation and fold them into
one memory edit at the checkpoint. A routine turn (a question answered, a small fix, an
exploratory read) needs no memory. Do not edit an existing memory, or bump its `updated`, just
because the work touched its topic. Edit it only when its content is wrong or a checkpoint adds
something substantive.

### Format

One file per topic, `kebab-case.md`, directly in `.memory/`:

```markdown
---
name: dof-ordering-gotcha
description: "One line. This is what lands in MEMORY.md, so make it the hook."
metadata:
  type: project
  agent: claude
  updated: 2026-09-16
---

Body. Bold the load-bearing gotcha. Separate what was verified from what was
assumed. End with **How to apply:**. Link related memories as
[[other-memory]]; a link to a memory that does not exist yet is fine, it
marks something worth writing.
```

`type` is one of:

- `project`: the work itself (what was built, why, where it stands, what is open). The default.
- `reference`: durable facts that hold regardless of the current goal (codebase mechanics,
  dataset locations, external sources).
- `feedback`: a rule the user gave about how to work. Include why, and how to apply it.
- `user`: something about the user that changes how work should be done here.

Optional `title:` overrides the index title derived from `name`. Memory files are working notes
and are exempt from the documentation prose style; tables and dashes are fine there.

### Rules

- One topic per file, amended in place. When a checkpoint adds to an existing memory, edit that
  file and bump `updated` rather than adding a near-duplicate. Correct wrong claims in place;
  do not leave them standing next to the fix.
- Set `agent:` to yourself when you write or amend a file, so a claim can be traced back.
- Keep files under about 150 lines. Past that, split by sub-topic and link the pieces.
- Never hand-edit `.memory/MEMORY.md`. It is regenerated from every file's `name`, `title`,
  `description`, and `type`. To change an index line, change the memory's `description`.
- Regenerate the index after writing, per the commands above.

### Conflicts

Sync compares the shared copy, the native copy, and the hashes recorded at the last sync. A file
changed on one side is copied. A file changed on both sides is a conflict: neither copy is
overwritten, a unified diff is printed, and sync exits non-zero.

Resolving means making both copies byte-identical. Editing only one side leaves the conflict
standing, because on the next run neither copy matches the recorded hash. Write the merged text
to both, or run `python3 .memory/sync.py --resolve shared|native` to keep one side wholesale.

Conflicts are rare here; the usual pattern is one agent handing off to another, not two writing
at once.

### Git

`.memory/` is deliberately untracked (listed in `.git/info/exclude`, so there is zero committed
diff). `origin` is the `guaguama/musclemimic` fork and `upstream` is `amathislab/musclemimic`;
working memory does not belong in either history. Never `git add -f .memory`.
