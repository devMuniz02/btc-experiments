"""
Training Migration Blueprint
===========================

Purpose
-------
This file is a reviewable migration blueprint for consolidating all training
entrypoints under a single future script:

    scripts/train.py

It is intentionally not runnable code. It documents the target architecture,
CLI shape, artifact contract, migration sequence, and the list of current
training launchers that should be removed after parity is confirmed.

This blueprint reuses the orchestration pattern of `src/utils/experiment_runners.py`
because that script already provides the right long-term structure:

- one top-level parser
- subcommands
- shared argument groups
- normalized config resolution
- centralized artifact writing
- optional reporting and dashboard hooks

The long-term goal is:

- leave exactly one user-facing training script in `scripts/`: `train.py`
- remove all previous experiment-specific training launchers from `scripts/`
- keep parquet-first experiment storage
- make the final CLI easy to run in GitHub Actions and easy to extend for
  MLflow and future Azure execution


Canonical Architecture
----------------------
Target user-facing surface in `scripts/`:

- `scripts/train.py`

Target supporting responsibilities:

- dataset loading/splitting/building
- model factory/build logic
- evaluation and checkpoint helpers
- artifact naming and persistence
- dashboard/report generation
- logging and integration hooks

Important policy:

- only `scripts/train.py` survives as a training entrypoint
- no future experiment should require a new `run_experiment_*` file
- new behavior must be added as flags, presets, or supported modes inside
  `scripts/train.py`
- helper modules may still exist, but they are not user-facing training scripts

Recommended implementation ownership:

- orchestration pattern inspired by `src/utils/experiment_runners.py`
- reusable training logic continues to live in reusable modules
- artifact contracts stay centralized
- dashboard/report scripts remain consumers of standardized outputs


Reference Pattern from `experiment_runners.py`
----------------------------------------------
The future `scripts/train.py` should copy the shape, not the exact current
implementation, of `src/utils/experiment_runners.py`.

Required structural traits:

1. One `parse_args()` entrypoint.
2. Subcommands instead of many standalone scripts.
3. Shared argument normalization helpers.
4. Shared output-dir and artifact path resolution.
5. Central dispatch by subcommand.
6. Centralized metadata and report writing.
7. Machine-readable JSON result output on stdout.
8. Explicit error handling via exit codes.

The future file should feel like:

- one control plane script
- many modes
- one artifact contract

It should not feel like:

- one script per experiment
- one script per training variation
- one script per migration or report case


Canonical `scripts/train.py` Command Surface
--------------------------------------------
Minimum required subcommands:

- `train`
- `reeval`
- `report`
- `migrate`

Optional subcommands if still worth keeping inside the canonical surface:

- `purge-window`
- `ensemble-select`

Guiding rule:

- keep `train`, `reeval`, `report`, and `migrate` at minimum because those are
  the most useful for CI, MLflow, and future cloud execution
- `purge-window` and `ensemble-select` are optional if the team wants them as
  first-class canonical commands; otherwise they can become post-training
  utilities under the same artifact contract


Single CLI Contract
-------------------
The future `scripts/train.py` should group arguments by responsibility.

Common arguments:

- `--experiment`
- `--output-dir`
- `--device`
- `--verbose`
- `--overwrite`
- `--max-workers`
- `--rebuild-artifacts`
- `--skip-dashboard`
- `--emit-json`

Training selection arguments:

- `--families`
- `--train-lengths`
- `--window-lengths`
- `--model-variations`
- `--time-variants`
- `--envs`
- `--selection-mode`

Dataset/split arguments:

- dataset source name or variant
- split regime
- fixed vs rolling behavior
- train/test row counts
- optional market-hours or derived-variant selectors

Window-mode arguments:

- base window mode
- continue mode
- retrain mode
- post-base mode
- window size / window count / window naming controls

Algorithm/hyperparameter arguments:

- epochs
- batch size
- learning rate
- PPO-specific knobs
- actor-critic knobs
- bandit strategy
- post-base fine-tuning knobs
- early stopping controls

Migration/report arguments:

- rewrite paths
- rebuild metadata
- rebuild dashboard artifacts
- missing-only reevaluation
- direct/windowed target scope

Integration arguments:

- MLflow enable/disable
- tracking URI override
- experiment/run tags
- Azure-neutral metadata flags
- CI-safe toggles


Canonical Config Resolution Pattern
-----------------------------------
The future script should follow a strict normalized control flow.

1. Parse args.
2. Normalize env values.
3. Normalize family/model/window/train-length values.
4. Resolve output root.
5. Resolve artifact path strategy.
6. Load any existing metadata/summary rows.
7. Dispatch by subcommand.
8. Emit structured result payload.

This should be implemented once in the canonical script, not repeated across
many ad hoc launchers.


Standardized Training Modes
---------------------------
The `train` subcommand must absorb all existing experiment-style training cases.

Required coverage:

- direct/base training
- window training
- continue-from-checkpoint training
- retrain-per-window training
- PPO variants
- actor-critic variants
- bandit variants
- post-base fine-tuning variants
- direct and experiment-driven training flows

If a current script has special behavior, that behavior must become one of:

- a CLI flag
- a preset mode
- a normalized model variation
- a normalized training mode

It must not survive as another standalone `scripts/run_experiment_*` file.


Standardized Orchestration Flow
-------------------------------
The `train` subcommand should execute in this order:

1. Parse and validate config.
2. Resolve canonical training mode.
3. Resolve dataset contract and split contract.
4. Resolve artifact names and output directories.
5. Determine skip, resume, continue, retrain, or overwrite behavior.
6. Build dataset bundle(s).
7. Build or restore model/policy.
8. Execute direct or windowed training loop.
9. Evaluate train/test/holdout outputs as required.
10. Save checkpoints, metadata, predictions, and summary rows.
11. Rebuild report/dashboard artifacts if requested.
12. Print a machine-readable JSON result.


Window / Continue / Retrain Strategy
------------------------------------
The blueprint should lock the following behavior:

Non-window flow:

- train once against the resolved split
- evaluate
- save canonical artifacts

Windowed base flow:

- iterate windows in canonical order
- save checkpoint metadata per window
- save unified row outputs under one normalized artifact scheme

Continue flow:

- discover last completed checkpoint/window
- load previous checkpoint
- continue only from the next unresolved stage

Retrain flow:

- train each target window according to canonical retrain rules
- preserve deterministic naming and summary contracts

Post-base flow:

- treat as a normalized mode of `train.py`
- never as a dedicated launcher script

Compatibility rule:

- retain current naming/path compatibility where practical so old consumers can
  migrate cleanly into the new canonical surface


Parquet-First Artifact Contract
-------------------------------
The standardized training system should preserve parquet-first storage for
heavy experiment data.

Target persisted outputs:

- `summary.json`
  metadata only

- `summary_rows.parquet`
  full row set

- prediction parquet files
  per-target model output streams

- model checkpoint files

- metadata JSON per trained artifact

- manifest JSON
  path pointers and lightweight metadata only

Rules:

- heavy row payloads belong in parquet
- prediction streams belong in parquet
- JSON is reserved for metadata, manifests, status payloads, and browser-facing
  compact dashboard artifacts
- path generation must be centralized so local runs, CI jobs, and MLflow-linked
  runs all produce the same structure


Report and Dashboard Contract
-----------------------------
Dashboard and report scripts stay in the repo, but they are consumers of
standardized outputs rather than alternate training entrypoints.

Expected consumers:

- report rebuilders
- dashboard artifact builders
- dashboard servers
- table generators

Required behavior:

- read metadata + parquet row storage
- never depend on giant embedded `models` payloads in summary JSON
- stay compatible with canonical manifest and summary pointers


CI / Automation Readiness
-------------------------
The future canonical script must be suitable for direct use in GitHub Actions.

Required traits:

- non-interactive
- deterministic output paths
- explicit `--output-dir`
- machine-readable JSON on stdout
- stable exit codes
- optional artifact rebuild toggles
- no dependency on local wrapper batch files

Recommended result payload shape:

- command
- status
- output_dir
- summary_path
- manifest_path
- models_path
- artifact_paths
- counts
- warnings

Exit code expectations:

- `0` success
- non-zero validation failure
- non-zero migration/integrity failure
- no ambiguous “soft fail” behavior for CI


MLflow Integration Expectations
-------------------------------
The repo already contains MLflow usage patterns in `src/utils/*`.
The future `scripts/train.py` should be designed so MLflow hooks can attach
cleanly without requiring wrapper scripts.

Expected support:

- run parameter logging
- run metric logging
- run tag logging
- artifact path logging
- explicit metadata fields for:
  - family
  - training mode
  - env
  - split regime
  - train length
  - window length
  - model variation
  - threshold set

MLflow should be treated as:

- an optional integration layer
- not a reason to split the CLI into many scripts


Azure and Platform-Neutral Execution
------------------------------------
Azure support is a future target. The blueprint should stay neutral and avoid
GitHub-only assumptions.

Required platform-neutral properties:

- filesystem-based artifacts
- explicit CLI args
- deterministic output locations
- optional tracking hooks
- no hidden state in local shells
- no orchestration logic tied only to GitHub Actions

The same canonical `scripts/train.py` should be callable from:

- local terminal runs
- GitHub Actions
- Azure jobs
- MLflow-triggered workflows


Migration Sequence
------------------
The actual migration should happen in this order:

1. Define canonical `scripts/train.py` CLI surface.
2. Map every current training launcher to canonical flags/modes/subcommands.
3. Extract any missing reusable helpers out of one-off scripts.
4. Preserve artifact compatibility where possible.
5. Standardize parquet-first saving for all canonical modes.
6. Standardize report/dashboard rebuild hooks.
7. Add CI-safe machine-readable outputs.
8. Add MLflow hook points.
9. Switch docs, workflows, and operators to `scripts/train.py`.
10. Delete obsolete training launchers only after parity is confirmed.


Keep vs Delete After Migration
------------------------------
Long-term scripts policy:

- keep exactly one training script in `scripts/`: `train.py`
- keep dataset/support scripts
- keep utility/helper scripts
- keep dashboard/report scripts
- keep model-support modules if they are not standalone training entrypoints
- delete all previous experiment-specific training launchers after parity

Delete-after-parity targets:

- `run_btc_direction_learning.py`
- `run_all_btc_direction_learning.py`
- `run_btc_rl_ppo.py`
- `run_rolling_btc_direction_learning.py`
- `run_rolling_btc_direction_learning_sweep.py`
- all `run_experiment_*` training scripts
- all `run_experiments_v2*` training/orchestration launchers, now coordinated through `src/utils/experiment_runners.py`

This includes, once the canonical CLI fully replaces them:

- direct training runners
- rolling/sweep runners
- experiment-specific LSTM/PPO/Transformer runners
- retrain wrappers
- window-train wrappers
- post-base wrappers
- experiment orchestration wrappers

Dashboard/report scripts stay:

- they consume the outputs of `scripts/train.py`
- they are not training entrypoints

Important safety rule:

- “safe to delete” means only after behavior parity is proven
- until then, old scripts are migration references, not immediate deletion


Reviewable Acceptance Criteria
------------------------------
This blueprint is acceptable only if all of the following are true:

- it clearly names `scripts/train.py` as the only future training script in
  `scripts/`
- it explicitly reuses the `experiment_runners.py` orchestration style
- it locks parquet-backed storage as the default heavy artifact format
- it defines CI-friendly machine-readable result output
- it includes MLflow-ready hook expectations
- it stays platform-neutral for future Azure execution
- it explains how current training launchers collapse into one script surface
- it includes an explicit deletion target list for old training launchers

Operational acceptance criteria for the future migration:

- any current experiment runner can be expressed as one invocation of
  `scripts/train.py`
- no new training workflow requires adding another script file
- GitHub Actions can call the canonical script directly
- MLflow logging can attach to canonical subcommands without wrapper scripts
- Azure can reuse the same CLI and artifact contract later without redesign


Non-Goals
---------
This blueprint does not:

- implement `scripts/train.py`
- prescribe Azure-specific runtime details
- define every final parser flag name exactly
- force all helper code into one physical file

The constraint is one training entrypoint script, not one total Python file.


Assumptions
-----------
- This file is for review only.
- The single-script requirement applies to user-facing training entrypoints in
  `scripts/`.
- `src/utils/experiment_runners.py` is the intended structural template.
- Parquet-first storage is the desired long-term artifact contract.
- GitHub Actions and MLflow are current integration references in this repo.
- Azure is a future consumer of the same canonical CLI, not a separate design.
"""
