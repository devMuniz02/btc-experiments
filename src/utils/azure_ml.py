from __future__ import annotations

import json
import os
import re
import shlex
import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUEST_RESULT_BLOB_NAME = "request_result.json"
REQUEST_SNAPSHOT_BLOB_NAME = "request_snapshot.json"
STATIC_WEBSITE_CONTAINER = "$web"
AZURE_TERMINAL_STATUSES = {"completed", "failed", "canceled", "cancelled"}
REQUEST_TERMINAL_FAILURES = {"failed", "canceled", "cancelled"}


def load_azure_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    return {
        "subscription_id": str(source.get("AZURE_ML_SUBSCRIPTION_ID") or "").strip(),
        "resource_group": str(source.get("AZURE_ML_RESOURCE_GROUP") or "").strip(),
        "workspace": str(source.get("AZURE_ML_WORKSPACE") or "").strip(),
        "compute": str(source.get("AZURE_ML_COMPUTE") or "").strip(),
        "experiment_name": str(source.get("AZURE_ML_EXPERIMENT_NAME") or "fixed-5k-train-requests").strip(),
        "environment_image": str(
            source.get("AZURE_ML_ENVIRONMENT_IMAGE") or "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
        ).strip(),
        "storage_account_url": str(source.get("AZURE_STORAGE_ACCOUNT_URL") or "").strip(),
        "storage_container": str(source.get("AZURE_STORAGE_CONTAINER") or "").strip(),
        "storage_prefix": str(source.get("AZURE_STORAGE_PREFIX") or "fixed-5k").strip().strip("/"),
        "storage_connection_string": str(source.get("AZURE_STORAGE_CONNECTION_STRING") or "").strip(),
        "dashboard_base_url": str(source.get("AZURE_DASHBOARD_BASE_URL") or "").strip().rstrip("/"),
        "mlflow_tracking_uri": str(source.get("MLFLOW_TRACKING_URI") or "").strip(),
        "mlflow_tracking_username": str(source.get("MLFLOW_TRACKING_USERNAME") or "").strip(),
        "mlflow_tracking_password": str(source.get("MLFLOW_TRACKING_PASSWORD") or "").strip(),
    }


def missing_required_config_fields(config: Mapping[str, str]) -> list[str]:
    required = [
        "subscription_id",
        "resource_group",
        "workspace",
        "compute",
        "storage_container",
        "dashboard_base_url",
    ]
    missing = [field for field in required if not str(config.get(field) or "").strip()]
    if (
        not str(config.get("storage_connection_string") or "").strip()
        and not str(config.get("storage_account_url") or "").strip()
    ):
        missing.append("storage_account_url")
    return sorted(dict.fromkeys(missing))


def validate_azure_config(config: Mapping[str, str]) -> None:
    missing = missing_required_config_fields(config)
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"Missing Azure configuration: {fields}")


def missing_blob_publish_config_fields(config: Mapping[str, str]) -> list[str]:
    required = [
        "storage_container",
        "dashboard_base_url",
    ]
    missing = [field for field in required if not str(config.get(field) or "").strip()]
    if (
        not str(config.get("storage_connection_string") or "").strip()
        and not str(config.get("storage_account_url") or "").strip()
    ):
        missing.append("storage_account_url")
    return sorted(dict.fromkeys(missing))


def validate_blob_publish_config(config: Mapping[str, str]) -> None:
    missing = missing_blob_publish_config_fields(config)
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"Missing Azure publish configuration: {fields}")


def sanitize_job_name(value: str, *, max_length: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    collapsed = re.sub(r"-{2,}", "-", normalized)
    if not collapsed:
        collapsed = "fixed-5k-request"
    return collapsed[:max_length].rstrip("-")


def build_job_name(request: Mapping[str, Any]) -> str:
    request_id = str(request.get("request_id") or "fixed-5k-request").strip()
    experiment = int(request.get("experiment") or 1)
    return sanitize_job_name(f"fixed-5k-exp-{experiment}-{request_id}")


def build_portal_url(config: Mapping[str, str], job_name: str) -> str:
    return (
        "https://ml.azure.com/runs/"
        f"{job_name}?wsid=/subscriptions/{config['subscription_id']}"
        f"/resourceGroups/{config['resource_group']}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{config['workspace']}"
    )


def build_request_blob_prefix(config: Mapping[str, str], request: Mapping[str, Any]) -> str:
    request_id = str(request.get("request_id") or "fixed-5k-request").strip()
    experiment = int(request.get("experiment") or 1)
    prefix = str(config.get("storage_prefix") or "fixed-5k").strip("/")
    return f"{prefix}/experiments/{experiment}/requests/{request_id}"


def build_blob_uri(config: Mapping[str, str], *parts: str) -> str:
    account_url = str(config.get("storage_account_url") or "").rstrip("/")
    container = str(config.get("storage_container") or "").strip("/")
    suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
    if account_url:
        return f"{account_url}/{container}/{suffix}"
    return f"azure-blob://{container}/{suffix}"


def build_dashboard_uri(config: Mapping[str, str], request: Mapping[str, Any]) -> str:
    base_url = str(config.get("dashboard_base_url") or "").rstrip("/")
    if not base_url:
        return ""
    request_id = str(request.get("request_id") or "").strip()
    experiment = int(request.get("experiment") or 1)
    return f"{base_url}/fixed-5k/experiments/{experiment}/requests/{request_id}/index.html"


def map_azure_job_status(raw_status: str) -> str:
    normalized = str(raw_status or "").strip().lower()
    if normalized in {"queued", "notstarted", "starting", "preparing"}:
        return "submitted"
    if normalized in {"submitted", "scheduled", "provisioning", "running", "finalizing"}:
        return "running"
    if normalized == "completed":
        return "completed"
    if normalized in REQUEST_TERMINAL_FAILURES:
        return "failed"
    return "reconcile_required"


def build_result_blob_uri(config: Mapping[str, str], request: Mapping[str, Any]) -> str:
    return build_blob_uri(config, build_request_blob_prefix(config, request), REQUEST_RESULT_BLOB_NAME)


def build_request_snapshot_blob_uri(config: Mapping[str, str], request: Mapping[str, Any]) -> str:
    return build_blob_uri(config, build_request_blob_prefix(config, request), REQUEST_SNAPSHOT_BLOB_NAME)


def build_request_artifact_root_uri(config: Mapping[str, str], request: Mapping[str, Any]) -> str:
    return build_blob_uri(config, build_request_blob_prefix(config, request), "artifacts")


def build_best_model_artifact_uri(config: Mapping[str, str], request: Mapping[str, Any]) -> str:
    return build_blob_uri(config, build_request_blob_prefix(config, request), "artifacts", "best-model")


def build_train_command_args(train_args: Mapping[str, Any]) -> list[str]:
    argv = [
        "python",
        "scripts/train.py",
        "train",
        "--experiment",
        str(train_args["experiment"]),
        "--output-dir",
        str(train_args["output_dir"]),
        "--runner",
        str(train_args["runner"]),
        "--mlflow-experiment",
        str(train_args["mlflow_experiment"]),
        "--mlflow-tracking-uri",
        str(train_args["mlflow_tracking_uri"]),
        "--selection-mode",
        str(train_args["selection_mode"]),
        "--device",
        str(train_args["device"]),
        "--max-workers",
        str(train_args["max_workers"]),
        "--verbose",
        str(train_args["verbose"]),
        "--experiments-root",
        str(train_args["experiments_root"]),
        "--mamba-epochs",
        str(train_args["mamba_epochs"]),
        "--mamba-batch-size",
        str(train_args["mamba_batch_size"]),
        "--overfit-max-epochs",
        str(train_args["overfit_max_epochs"]),
        "--bandit-strategy",
        str(train_args["bandit_strategy"]),
        "--post-base-total-updates",
        str(train_args["post_base_total_updates"]),
        "--post-base-learning-rate",
        str(train_args["post_base_learning_rate"]),
        "--post-base-clip-epsilon",
        str(train_args["post_base_clip_epsilon"]),
        "--post-base-entropy-coef",
        str(train_args["post_base_entropy_coef"]),
        "--post-base-value-loss-coef",
        str(train_args["post_base_value_loss_coef"]),
        "--post-base-ppo-epochs",
        str(train_args["post_base_ppo_epochs"]),
        "--post-base-minibatch-size",
        str(train_args["post_base_minibatch_size"]),
        "--retrain-epochs",
        str(train_args["retrain_epochs"]),
        "--emit-json",
    ]
    if bool(train_args.get("mlflow_enabled")):
        argv.append("--mlflow-enabled")
    if bool(train_args.get("overwrite")):
        argv.append("--overwrite")
    if bool(train_args.get("mamba_disable_early_stopping")):
        argv.append("--mamba-disable-early-stopping")
    if bool(train_args.get("overfit")):
        argv.append("--overfit")
    if bool(train_args.get("visualize_test_acc")):
        argv.append("--visualize-test-acc")
    for flag, values in (
        ("--families", train_args.get("families", [])),
        ("--train-lengths", train_args.get("train_lengths", [])),
        ("--window-lengths", train_args.get("window_lengths", [])),
        ("--model-variations", train_args.get("model_variations", [])),
        ("--time-variants", train_args.get("time_variants", [])),
        ("--envs", train_args.get("envs", [])),
    ):
        value_list = [str(value) for value in values if str(value)]
        if value_list:
            argv.append(flag)
            argv.extend(value_list)
    return argv


def build_report_command_args(request: Mapping[str, Any]) -> list[str]:
    runner = str(request.get("runner") or "azure")
    mlflow = request.get("mlflow", {}) if isinstance(request.get("mlflow"), dict) else {}
    argv = [
        "python",
        "scripts/train.py",
        "report",
        "--experiment",
        str(request.get("experiment") or 1),
        "--output-dir",
        str(request.get("output_dir") or ""),
        "--runner",
        runner,
        "--mlflow-experiment",
        str(mlflow.get("experiment_name") or "fixed-5k-train-requests"),
        "--mlflow-tracking-uri",
        str(mlflow.get("tracking_uri") or ""),
        "--rebuild-artifacts",
        "--emit-json",
    ]
    if bool(mlflow.get("enabled", True)):
        argv.append("--mlflow-enabled")
    return argv


def build_submission_payload(
    request_path: Path,
    request: Mapping[str, Any],
    train_args: Mapping[str, Any],
    config: Mapping[str, str],
) -> dict[str, Any]:
    job_name = build_job_name(request)
    request_snapshot_uri = build_request_snapshot_blob_uri(config, request)
    result_blob_uri = build_result_blob_uri(config, request)
    artifact_root_uri = build_request_artifact_root_uri(config, request)
    env_vars = {
        "MLFLOW_TRACKING_URI": str(config.get("mlflow_tracking_uri") or ""),
        "MLFLOW_TRACKING_USERNAME": str(config.get("mlflow_tracking_username") or ""),
        "MLFLOW_TRACKING_PASSWORD": str(config.get("mlflow_tracking_password") or ""),
        "AZURE_STORAGE_ACCOUNT_URL": str(config.get("storage_account_url") or ""),
        "AZURE_STORAGE_CONTAINER": str(config.get("storage_container") or ""),
        "AZURE_STORAGE_PREFIX": str(config.get("storage_prefix") or ""),
        "AZURE_STORAGE_CONNECTION_STRING": str(config.get("storage_connection_string") or ""),
        "AZURE_DASHBOARD_BASE_URL": str(config.get("dashboard_base_url") or ""),
        "FIXED_5K_REQUEST_RESULT_BLOB_URI": result_blob_uri,
        "FIXED_5K_REQUEST_SNAPSHOT_BLOB_URI": request_snapshot_uri,
        "FIXED_5K_ARTIFACT_ROOT_URI": artifact_root_uri,
        "FIXED_5K_REQUEST_ID": str(request.get("request_id") or ""),
        "FIXED_5K_AZURE_JOB_NAME": job_name,
    }
    command = " && ".join(
        [
            "python -m pip install --upgrade pip",
            "pip install -r requirements.txt",
            "pip install azure-storage-blob",
            " ".join(
                [
                    "python",
                    "scripts/azure_train_request_job.py",
                    "--request-file",
                    shlex.quote(str(request_path.as_posix())),
                    "--train-command-json",
                    shlex.quote(json.dumps(build_train_command_args(train_args))),
                    "--report-command-json",
                    shlex.quote(json.dumps(build_report_command_args(request))),
                ]
            ),
        ]
    )
    return {
        "job_name": job_name,
        "display_name": str(request.get("request_id") or job_name),
        "experiment_name": str(config.get("experiment_name") or "fixed-5k-train-requests"),
        "compute": str(config.get("compute") or ""),
        "environment_image": str(config.get("environment_image") or ""),
        "code": str(PROJECT_ROOT),
        "command": command,
        "tags": {
            "request_id": str(request.get("request_id") or ""),
            "runner": "azure",
            "scope": str(request.get("scope") or "fixed_5k"),
            "family": str(request.get("family") or ""),
            "train_length": str(request.get("train_length") or ""),
            "commit_sha": str(request.get("commit_sha") or ""),
        },
        "environment_variables": env_vars,
        "portal_url": build_portal_url(config, job_name),
        "result_blob_uri": result_blob_uri,
        "artifact_root_uri": artifact_root_uri,
        "dashboard_uri": build_dashboard_uri(config, request),
        "model_artifact_uri": build_best_model_artifact_uri(config, request),
    }


def _create_ml_client(config: Mapping[str, str]) -> Any:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace"],
    )


def submit_train_request(
    request_path: Path,
    request: Mapping[str, Any],
    train_args: Mapping[str, Any],
    *,
    config: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    resolved_config = load_azure_config() if config is None else dict(config)
    validate_azure_config(resolved_config)
    payload = build_submission_payload(request_path, request, train_args, resolved_config)

    from azure.ai.ml import command
    from azure.ai.ml.entities import Environment

    ml_client = _create_ml_client(resolved_config)
    azure_job = command(
        code=payload["code"],
        command=payload["command"],
        compute=payload["compute"],
        environment=Environment(image=payload["environment_image"]),
        display_name=payload["display_name"],
        experiment_name=payload["experiment_name"],
        environment_variables=payload["environment_variables"],
        tags=payload["tags"],
    )
    azure_job.name = payload["job_name"]
    submitted = ml_client.jobs.create_or_update(azure_job)
    return {
        "workspace": resolved_config["workspace"],
        "compute": resolved_config["compute"],
        "job_name": payload["job_name"],
        "job_id": str(getattr(submitted, "name", "") or payload["job_name"]),
        "portal_url": payload["portal_url"],
        "experiment_name": payload["experiment_name"],
        "result_blob_uri": payload["result_blob_uri"],
        "artifact_root_uri": payload["artifact_root_uri"],
        "dashboard_uri": payload["dashboard_uri"],
        "model_artifact_uri": payload["model_artifact_uri"],
        "request_snapshot_blob_uri": build_request_snapshot_blob_uri(resolved_config, request),
        "storage_container": resolved_config["storage_container"],
        "storage_prefix": resolved_config["storage_prefix"],
    }


def fetch_job_snapshot(request: Mapping[str, Any], *, config: Mapping[str, str] | None = None) -> dict[str, Any]:
    resolved_config = load_azure_config() if config is None else dict(config)
    validate_azure_config(resolved_config)
    azure_meta = request.get("azure", {}) if isinstance(request.get("azure"), dict) else {}
    job_name = str(azure_meta.get("job_name") or azure_meta.get("job_id") or "").strip()
    if not job_name:
        raise ValueError("Missing Azure job_name for request.")
    ml_client = _create_ml_client(resolved_config)
    job = ml_client.jobs.get(job_name)
    raw_status = str(getattr(job, "status", "") or "")
    return {
        "raw_status": raw_status,
        "request_status": map_azure_job_status(raw_status),
        "job_name": job_name,
        "job_id": str(getattr(job, "name", "") or job_name),
        "portal_url": build_portal_url(resolved_config, job_name),
        "artifact_root_uri": build_request_artifact_root_uri(resolved_config, request),
        "dashboard_uri": build_dashboard_uri(resolved_config, request),
        "model_artifact_uri": build_best_model_artifact_uri(resolved_config, request),
        "result_blob_uri": build_result_blob_uri(resolved_config, request),
    }


def _create_blob_service_client(config: Mapping[str, str]) -> Any:
    from azure.storage.blob import BlobServiceClient

    connection_string = str(config.get("storage_connection_string") or "").strip()
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    from azure.identity import DefaultAzureCredential

    return BlobServiceClient(
        account_url=str(config["storage_account_url"]).rstrip("/"),
        credential=DefaultAzureCredential(),
    )


def _blob_client_for_uri(blob_uri: str, config: Mapping[str, str]) -> Any:
    service = _create_blob_service_client(config)
    account_url = str(config.get("storage_account_url") or "").rstrip("/")
    container = str(config.get("storage_container") or "").strip("/")
    if account_url and blob_uri.startswith(f"{account_url}/{container}/"):
        blob_name = blob_uri.removeprefix(f"{account_url}/{container}/")
    elif blob_uri.startswith(f"azure-blob://{container}/"):
        blob_name = blob_uri.removeprefix(f"azure-blob://{container}/")
    else:
        raise ValueError(f"Blob URI is outside configured container: {blob_uri}")
    return service.get_blob_client(container=container, blob=blob_name)


def _blob_client_for_container_blob(container: str, blob_name: str, config: Mapping[str, str]) -> Any:
    service = _create_blob_service_client(config)
    return service.get_blob_client(container=container, blob=blob_name.strip("/"))


def upload_json_blob(blob_uri: str, payload: Mapping[str, Any], *, config: Mapping[str, str] | None = None) -> None:
    resolved_config = load_azure_config() if config is None else dict(config)
    validate_blob_publish_config(resolved_config)
    client = _blob_client_for_uri(blob_uri, resolved_config)
    client.upload_blob(json.dumps(payload, indent=2), overwrite=True)


def upload_text_blob(
    container: str,
    blob_name: str,
    text: str,
    *,
    content_type: str,
    config: Mapping[str, str] | None = None,
) -> None:
    resolved_config = load_azure_config() if config is None else dict(config)
    validate_blob_publish_config(resolved_config)
    from azure.storage.blob import ContentSettings

    client = _blob_client_for_container_blob(container, blob_name, resolved_config)
    client.upload_blob(
        text.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )


def _map_result_payload_paths_to_blob_uris(
    result_payload: Mapping[str, Any],
    *,
    artifact_root: Path,
    artifact_root_uri: str,
) -> dict[str, Any]:
    published_result = json.loads(json.dumps(result_payload))
    for key in ("summary_path", "manifest_path", "models_path"):
        path_value = str(published_result.get(key) or "").strip()
        if path_value:
            published_result[key] = map_local_path_to_blob_uri(path_value, str(artifact_root), artifact_root_uri)
    matched_rows = published_result.get("matched_rows", [])
    if isinstance(matched_rows, list):
        for row in matched_rows:
            if not isinstance(row, dict):
                continue
            for key in ("saved_model_path", "predictions_path"):
                path_value = str(row.get(key) or "").strip()
                if path_value:
                    row[key] = map_local_path_to_blob_uri(path_value, str(artifact_root), artifact_root_uri)
    return published_result


def build_request_dashboard_html(request: Mapping[str, Any], result_payload: Mapping[str, Any]) -> str:
    request_id = html.escape(str(request.get("request_id") or ""))
    family = html.escape(str(request.get("family") or ""))
    train_length = html.escape(str(request.get("train_length") or ""))
    model_variation = html.escape(str(request.get("model_variation") or ""))
    experiment = html.escape(str(request.get("experiment") or ""))
    mlflow_run_id = html.escape(str(result_payload.get("mlflow_run_id") or ""))
    artifact_root = html.escape(str(result_payload.get("artifact_root") or ""))
    dashboard_uri = html.escape(str(result_payload.get("dashboard_uri") or ""))
    model_artifact_uri = html.escape(str(result_payload.get("model_artifact_uri") or ""))
    summary_path = html.escape(str(result_payload.get("summary_path") or ""))
    manifest_path = html.escape(str(result_payload.get("manifest_path") or ""))
    models_path = html.escape(str(result_payload.get("models_path") or ""))

    rows_html: list[str] = []
    matched_rows = result_payload.get("matched_rows", [])
    if isinstance(matched_rows, list):
        for row in matched_rows:
            if not isinstance(row, Mapping):
                continue
            rows_html.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('name') or ''))}</td>"
                f"<td>{html.escape(str(row.get('time_variant') or ''))}</td>"
                f"<td>{html.escape(str(row.get('threshold_pct') or ''))}</td>"
                f"<td>{html.escape(str(row.get('test_accuracy') or ''))}</td>"
                f"<td>{html.escape(str(row.get('daily_net_wins') or ''))}</td>"
                f"<td><a href=\"{html.escape(str(row.get('saved_model_path') or ''))}\">model</a></td>"
                f"<td><a href=\"{html.escape(str(row.get('predictions_path') or ''))}\">predictions</a></td>"
                "</tr>"
            )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fixed-5K Request Dashboard {request_id}</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #0b1220; color: #dbe7f3; font-family: Segoe UI, system-ui, sans-serif; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-bottom: 24px; }}
    .card {{ background: #111b2d; border: 1px solid #22304a; border-radius: 16px; padding: 18px; }}
    .label {{ color: #8fa4c2; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .value {{ margin-top: 8px; font-size: 1rem; word-break: break-word; }}
    a {{ color: #72c7ff; }}
    table {{ width: 100%; border-collapse: collapse; background: #111b2d; border: 1px solid #22304a; border-radius: 16px; overflow: hidden; }}
    th, td {{ padding: 12px; border-bottom: 1px solid #22304a; text-align: left; }}
    th {{ background: #16243b; color: #9eb0c9; text-transform: uppercase; font-size: 0.78rem; }}
  </style>
</head>
<body>
  <h1>Fixed-5K Request Dashboard</h1>
  <p>Static Azure-hosted summary for local run <strong>{request_id}</strong>.</p>
  <div class="grid">
    <div class="card"><div class="label">Experiment</div><div class="value">{experiment}</div></div>
    <div class="card"><div class="label">Family</div><div class="value">{family}</div></div>
    <div class="card"><div class="label">Train Length</div><div class="value">{train_length}</div></div>
    <div class="card"><div class="label">Variation</div><div class="value">{model_variation}</div></div>
    <div class="card"><div class="label">MLflow Run</div><div class="value">{mlflow_run_id}</div></div>
    <div class="card"><div class="label">Primary Model URI</div><div class="value"><a href="{model_artifact_uri}">{model_artifact_uri}</a></div></div>
  </div>
  <div class="grid">
    <div class="card"><div class="label">Artifact Root</div><div class="value"><a href="{artifact_root}">{artifact_root}</a></div></div>
    <div class="card"><div class="label">Summary JSON</div><div class="value"><a href="{summary_path}">{summary_path}</a></div></div>
    <div class="card"><div class="label">Manifest JSON</div><div class="value"><a href="{manifest_path}">{manifest_path}</a></div></div>
    <div class="card"><div class="label">Models Table</div><div class="value"><a href="{models_path}">{models_path}</a></div></div>
  </div>
  <h2>Matched Rows</h2>
  <table>
    <thead>
      <tr><th>Name</th><th>Variant</th><th>Threshold</th><th>Test Accuracy</th><th>Daily Net Wins</th><th>Model</th><th>Predictions</th></tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""


def download_json_blob(blob_uri: str, *, config: Mapping[str, str] | None = None) -> dict[str, Any]:
    resolved_config = load_azure_config() if config is None else dict(config)
    validate_blob_publish_config(resolved_config)
    client = _blob_client_for_uri(blob_uri, resolved_config)
    raw = client.download_blob().readall().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in blob: {blob_uri}")
    return payload


def upload_directory_to_blob(
    local_root: Path,
    blob_root_uri: str,
    *,
    config: Mapping[str, str] | None = None,
) -> list[str]:
    resolved_config = load_azure_config() if config is None else dict(config)
    validate_blob_publish_config(resolved_config)
    uploaded: list[str] = []
    if not local_root.exists():
        return uploaded
    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_root).as_posix()
        blob_uri = f"{blob_root_uri.rstrip('/')}/{relative}"
        client = _blob_client_for_uri(blob_uri, resolved_config)
        with path.open("rb") as handle:
            client.upload_blob(handle, overwrite=True)
        uploaded.append(blob_uri)
    return uploaded


def map_local_path_to_blob_uri(local_path: str, local_root: str, blob_root_uri: str) -> str:
    local_path_obj = Path(local_path).resolve()
    local_root_obj = Path(local_root).resolve()
    try:
        relative = local_path_obj.relative_to(local_root_obj).as_posix()
    except ValueError:
        return local_path
    return f"{blob_root_uri.rstrip('/')}/{relative}"


def publish_local_run(
    request_path: Path,
    request: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    *,
    config: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    resolved_config = load_azure_config() if config is None else dict(config)
    validate_blob_publish_config(resolved_config)

    artifact_root = Path(str(result_payload.get("artifact_root") or "")).resolve()
    if not artifact_root.exists():
        raise ValueError(f"Local artifact root does not exist: {artifact_root}")

    artifact_root_uri = build_request_artifact_root_uri(resolved_config, request)
    request_snapshot_blob_uri = build_request_snapshot_blob_uri(resolved_config, request)
    result_blob_uri = build_result_blob_uri(resolved_config, request)
    uploaded_uris = upload_directory_to_blob(artifact_root, artifact_root_uri, config=resolved_config)

    dashboard_uri = build_dashboard_uri(resolved_config, request)
    model_artifact_uri = build_best_model_artifact_uri(resolved_config, request)
    matched_rows = (
        result_payload.get("matched_rows", []) if isinstance(result_payload.get("matched_rows"), list) else []
    )
    if matched_rows:
        first_row = matched_rows[0] if isinstance(matched_rows[0], Mapping) else {}
        model_path = str(first_row.get("saved_model_path") or "").strip()
        if model_path:
            model_artifact_uri = map_local_path_to_blob_uri(model_path, str(artifact_root), artifact_root_uri)

    published_result = _map_result_payload_paths_to_blob_uris(
        result_payload,
        artifact_root=artifact_root,
        artifact_root_uri=artifact_root_uri,
    )
    published_result["artifact_root"] = artifact_root_uri
    published_result["model_artifact_uri"] = model_artifact_uri
    published_result["dashboard_uri"] = dashboard_uri

    request_snapshot = json.loads(json.dumps(request))
    request_snapshot.setdefault("azure", {})
    request_snapshot["azure"].update(
        {
            "storage_container": str(resolved_config.get("storage_container") or "").strip(),
            "storage_prefix": str(resolved_config.get("storage_prefix") or "").strip(),
            "artifact_root_uri": artifact_root_uri,
            "dashboard_uri": dashboard_uri,
            "model_artifact_uri": model_artifact_uri,
            "request_snapshot_blob_uri": request_snapshot_blob_uri,
        }
    )
    request_snapshot["result"] = {
        **(request_snapshot.get("result", {}) if isinstance(request_snapshot.get("result"), dict) else {}),
        **published_result,
    }
    upload_json_blob(request_snapshot_blob_uri, request_snapshot, config=resolved_config)

    upload_json_blob(result_blob_uri, published_result, config=resolved_config)

    request_id = str(request.get("request_id") or "").strip()
    experiment = int(request.get("experiment") or 1)
    dashboard_blob_name = f"fixed-5k/experiments/{experiment}/requests/{request_id}/index.html"
    upload_text_blob(
        STATIC_WEBSITE_CONTAINER,
        dashboard_blob_name,
        build_request_dashboard_html(request, published_result),
        content_type="text/html; charset=utf-8",
        config=resolved_config,
    )

    return {
        "storage_container": str(resolved_config.get("storage_container") or "").strip(),
        "storage_prefix": str(resolved_config.get("storage_prefix") or "").strip(),
        "artifact_root_uri": artifact_root_uri,
        "dashboard_uri": dashboard_uri,
        "model_artifact_uri": model_artifact_uri,
        "request_snapshot_blob_uri": request_snapshot_blob_uri,
        "result_blob_uri": result_blob_uri,
        "uploaded_blob_uris": uploaded_uris,
    }
