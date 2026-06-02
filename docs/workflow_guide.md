# Quant-Stream Workflow Guide

Quant-Stream uses a deterministic folder-state engine. Paths are relative to the repository root.

## Local Python Environment

Use a Python 3.11 environment with `numpy`, `pandas`, `pyarrow`, `pyyaml`, `streamlit`, `plotly`, `black`, `flake8`,
and `mypy` installed. The strict active layout intentionally keeps environment files out of the remote-ready tree.

## Data Ingestion

Variation data is written to `data/var_X/btc_1h_clean.parquet`.

For the first canonical variation:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\generate_var_1_dataset.ps1
```

The script uses the Conda environment `btc-quant-stream` and the fixed test anchor
`2025-10-08 15:00:00+00:00`. The ingestion helpers live in `src/utils.py`; the script calls the active
`python -m src.utils ingest` command to create `data/var_1/btc_1h_clean.parquet`.

The canonical fetch policy uses Binance `BTC/USDT` for every row Binance can provide, because Binance is the live
trading venue. Older pre-Binance rows are backfilled from Kraken `BTC/USD` only when needed to complete the fixed
105,000 row matrix.

The ingestion command fetches 100,000 one-hour candles before the test anchor and 5,000 after it, with extra lookback
padding for technical indicators. Indicator calculations happen on unscaled OHLCV values.

## Run Lifecycle

Place a YAML request in `automation/run_requests/`.

The runner validates schema and domain constraints, derives a deterministic SHA-256 `model_id`, checks active duplicate
folders and deleted tombstones, then processes the request.

```powershell
python automation_runner.py --once
```

Successful runs move to `automation/runs_done/[model_id].yaml`, write model assets under
`models/dev/var_X/[model_id]/`, and append prediction columns to `data/var_X/global_results.parquet`.

The active runner supports `lstm`, `transformer`, `mamba`, `nn`, `rf`, `xgboost`, `bc`, `dagger`, `ppo`,
`ppo_continue`, `actor_critic`, `mamba_post_base`, and `ensemble`. Training modes include `static_baseline`,
`sliding_window_current_only`, `sliding_window_continue`, `sliding_window_retrain`, `reinforcement_ppo`, and
`post_base`.

Rejected runs move to `automation/rejected_runs/` with a YAML comment explaining the failure.

## Delete Lifecycle

Move a completed run YAML into `automation/delete_requests/` and run:

```powershell
python automation_runner.py --once
```

The engine deletes the model folder, removes `[model_id]_pred` and `[model_id]_prob` columns from global results, and
writes a tombstone to `automation/deleted_runs/[model_id].yaml`.

## State Sync

Use sync checks before long training sessions:

```powershell
python state_sync.py --check
```

Without `--check`, stale parquet columns and unreferenced scalers are cleaned in place.

## Remote Readiness

Azure and MLflow are disabled in `automation/config.yaml` for now:

```yaml
mlflow_enabled: false
azure_enabled: false
tracking_mode: local_buffer
```

Before linking the repository to a remote, confirm ignored heavy artifacts:

```powershell
git status --ignored
```

Datasets, model weights, scaler binaries, generated automation YAML files, caches, and `legacy_unused/` are ignored by
the root `.gitignore`.

## Windows Local Automation

Run one local automation pass manually from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\run_quant_stream_once.ps1
```

The local scripts use `conda run -n btc-quant-stream ...`, so they work from Windows Task Scheduler without manually
activating the environment first.

Start a watcher that triggers whenever YAML files are created or changed in `automation/run_requests/` or
`automation/delete_requests/`:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\watch_quant_stream.ps1
```

Run local code quality manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\run_local_code_quality.ps1
```

Start the full local CI/CD watcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\watch_local_ci_cd.ps1
```

This watcher behaves like a small local CI/CD lane:

- root Python file changes, `src/**/*.py`, `src/**/*.cu`, docs, and workflow edits trigger local code quality.
- `automation/run_requests/*.yaml` and `automation/delete_requests/*.yaml` trigger the local request runner.
- GitHub Actions remain code-quality-only; training/request execution stays local.

Create a Windows Task Scheduler startup watcher:

```powershell
schtasks /Create /TN "QuantStreamLocalCICD" /SC ONLOGON /TR "powershell -ExecutionPolicy Bypass -File C:\Users\emman\proyects\btc-experiments\automation\watch_local_ci_cd.ps1" /F
```

Install the local hook and request watcher in one step:

```powershell
powershell -ExecutionPolicy Bypass -File .\automation\setup_local_ci_cd.ps1
```

Create an optional manual one-shot task:

```powershell
schtasks /Create /TN "QuantStreamRunOnce" /SC ONDEMAND /TR "powershell -ExecutionPolicy Bypass -File C:\Users\emman\proyects\btc-experiments\automation\run_quant_stream_once.ps1" /F
schtasks /Run /TN "QuantStreamRunOnce"
```

The PowerShell scripts under `automation/*.ps1` are local-only and ignored by Git. If Git already tracks any of them,
untrack them once:

```powershell
git rm --cached automation\*.ps1
```

## Push Local State To Remote Without Pulling

Use this when the remote repository exists but this local strict Quant-Stream layout should replace the remote content.
Do not run `git pull`.

```powershell
git remote remove origin
git remote add origin https://github.com/devMuniz02/btc-experiments.git
git status --ignored
git add .
git commit -m "Restructure repository for Quant-Stream local automation"
git push --force-with-lease origin HEAD:main
```

If the remote default branch is not `main`, replace `main` with the target branch name. `--force-with-lease` is safer
than raw `--force`, but it still replaces the remote branch with this local branch state.
