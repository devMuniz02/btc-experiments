# Fixed-5K MLOps Design

## Purpose

This repo implements a fixed-5k request-driven MLOps flow with:

- one canonical control-plane CLI: `scripts/train.py`
- one request JSON per run under `train_requests/requests/`
- GitHub Actions for code quality only
- local compute for actual training execution
- MLflow for train/eval tracking
- Azure Blob storage for heavy artifacts
- Azure dashboard hosting for experiment monitoring

## Operating Model

### Canonical CLI

User-facing subcommands:

- `train`
- `reeval`
- `report`
- `migrate`
- `request-run`
- `request-reconcile`

### Request Files

Each fixed-5k run is represented by one JSON file in `train_requests/requests/`.

Workflow:

1. Create or update a request file using `train_requests/schema.example.json`.
2. Source local Azure and MLflow secrets with `.\set_local_azure_env.ps1`.
3. Submit the request through `scripts/train.py request-run --runner local --publish-azure`.
4. The control plane validates the JSON and writes canonical formatting back to the request file.
5. Local compute runs the canonical train/report flow.
6. The request file is updated with Azure publication metadata, MLflow run metadata, and Azure Blob/dashboard URIs.
7. `request-reconcile --runner local --publish-azure` repairs missing Azure publication metadata when needed.

### Status Lifecycle

- `pending`
- `completed`
- `failed`
- `reconcile_required`

## Artifact Contract

### Heavy Data

Heavy data lives in Azure Blob storage:

- uploaded experiment directories
- model checkpoints
- parquet outputs
- dashboard artifact bundles

### Lightweight Metadata

Lightweight metadata stays in JSON:

- `summary.json`
- `manifest.json`
- request files
- request result payloads

### Best-Model Storage

Best models should be treated as Azure Blob-backed artifacts. MLflow runs and model metadata should reference those Azure URIs rather than local repo paths.

## Automation Targets

### GitHub Actions

- `code-quality.yml` is the only commit-triggered workflow
- `fixed-5k-train-request.yml` is intentionally disabled
- `fixed-5k-train-reconcile.yml` is intentionally disabled

GitHub-hosted runners should not execute fixed-5k model training.

### Local Compute And Azure Publication

The local machine is the execution backend for fixed-5k request training:

- receives a local request invocation
- runs the canonical train/report flow
- logs to MLflow
- uploads outputs to Azure Blob storage
- produces machine-readable result metadata for reconcile flows

### Azure Dashboard Hosting

Dashboard/report artifacts remain produced by the repo’s reporting code, but the public dashboard should be hosted from Azure and linked from the request result metadata.

## MLflow Expectations

Every train and reevaluation path should log:

- params: family, train length, window length, variation, env, request id, output dir, command
- tags: runner, workflow source, fixed-5k scope, Azure publication state, commit SHA
- metrics: metrics available from the completed result rows
- artifacts: summary JSON, manifest JSON, request snapshot JSON, result JSON

MLflow remains the tracking and model-catalog layer, while Azure Blob storage remains the system-of-record for heavy artifacts.
