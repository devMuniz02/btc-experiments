# Fixed-5K Train Requests

This folder is the control plane for fixed-5k training runs.

Live request files belong under:

- `train_requests/requests/`

Operating model:

- one JSON file represents one requested run
- `scripts/train.py request-run --runner local --publish-azure --request-file <path>` trains locally and publishes artifacts to Azure
- `scripts/train.py request-reconcile --runner local --publish-azure` repairs Azure publication metadata and links
- request files stay in the repo and are updated in place with status, Azure publication metadata, MLflow metadata, and artifact URIs

Lifecycle:

- `pending`: ready to submit
- `completed`: local training completed and final artifact metadata was hydrated
- `failed`: local training failed
- `reconcile_required`: local training succeeded but Azure publication metadata or artifacts were incomplete

Conventions:

- use one file per request
- keep `scope` set to `fixed_5k`
- prefer immutable request ids and filenames
- do not store model binaries or generated parquet outputs in this folder

Examples:

```bash
.\set_local_azure_env.ps1
python scripts/train.py request-run --runner local --publish-azure --request-file train_requests/requests/20260528-120000-lstm-35k-base.json --mlflow-enabled
python scripts/train.py request-reconcile --runner local --publish-azure --mlflow-enabled
```

Schema reference:

- see `train_requests/schema.example.json`
