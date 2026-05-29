# BTC Experiments Fixed-5K MLOps

This repo uses one canonical control-plane entrypoint:

`scripts/train.py`

The operating model is request-driven training with:

- GitHub Actions for commit-time code quality only
- local compute for fixed-5k training execution
- MLflow for train/eval tracking
- Azure Blob storage for heavy artifacts and model checkpoints
- Azure-hosted dashboards backed by the same generated report artifacts

## What This Repo Tracks

- fixed-5k training and reevaluation control-plane code
- request-file orchestration under `train_requests/requests/`
- GitHub Actions quality workflows
- dashboard/report builders and local dashboard serving code
- MLflow and Azure publication hooks

## What This Repo Does Not Track

- datasets
- model binaries
- parquet outputs
- generated dashboard artifacts
- local logs
- machine-local runtime state

## Control Plane

The canonical CLI is:

```bash
python scripts/train.py <subcommand> [options]
```

Supported subcommands:

- `train`
- `reeval`
- `report`
- `migrate`
- `request-run`
- `request-reconcile`

Examples:

```bash
.\set_local_azure_env.ps1
python scripts/train.py request-run --runner local --publish-azure --request-file train_requests/requests/20260528-120000-lstm-35k-base.json --mlflow-enabled
python scripts/train.py request-reconcile --runner local --publish-azure --mlflow-enabled
python scripts/train.py report --experiment 1 --rebuild-artifacts
```

## Request-Driven Local Training With Azure Publication

Each fixed-5k run is declared as one JSON file in `train_requests/requests/`.

- request files are validated and normalized before execution
- `request-run --runner local --publish-azure` trains locally and then uploads artifacts to Azure
- `request-reconcile --runner local --publish-azure` repairs or refreshes Azure publication metadata
- request files are updated in place with Azure publication metadata, MLflow metadata, and artifact links

See:

- [train_requests/README.md](train_requests/README.md)
- [train_requests/schema.example.json](train_requests/schema.example.json)
- [docs/fixed_5k_mlops.md](docs/fixed_5k_mlops.md)

## Artifact Contract

Heavy experiment data is stored in Azure Blob storage:

- uploaded experiment directories
- model checkpoints
- parquet outputs
- generated dashboard bundles

Lightweight metadata stays in JSON:

- `summary.json`
- `manifest.json`
- request files
- Azure-synced result payloads

## MLflow and Azure

MLflow is the tracking layer for:

- train runs
- reevaluation runs
- locally executed request runs
- reconciliation syncs and Azure publication repairs

Azure Blob storage is the system-of-record for heavy artifacts and best-model checkpoints. Request results persist the canonical artifact URIs, MLflow run ID, and Azure dashboard link.

## Automation

GitHub Actions:

- `.github/workflows/code-quality.yml` runs on pushes and pull requests
- `.github/workflows/fixed-5k-train-request.yml` is intentionally disabled
- `.github/workflows/fixed-5k-train-reconcile.yml` is intentionally disabled

Local machine plus Azure:

- local compute runs the fixed-5k training jobs
- MLflow tracks the local runs
- Azure Blob stores heavy outputs
- Azure-hosted dashboards expose published report artifacts
