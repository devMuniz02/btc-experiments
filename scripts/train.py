from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._helpers import load_json, write_json_atomic
from src.utils import azure_ml
from src.utils import experiment_runners as unified_v2
from src.utils.experiment_support import (
    EXPERIMENTS_V2_ROOT,
    load_summary_rows,
    summary_rows_path,
)

DEFAULT_REQUEST_SCOPE = "fixed_5k"
DEFAULT_REQUEST_EXPERIMENT = 1
DEFAULT_REQUEST_STATUS = "pending"
DEFAULT_REQUEST_FOLDER = PROJECT_ROOT / "train_requests" / "requests"
DEFAULT_RUNNER = "local"
DEFAULT_MLFLOW_EXPERIMENT = "fixed-5k-train-requests"
SUPPORTED_RUNNERS = ("local", "github", "azure")
SUPPORTED_REQUEST_STATUSES = {
    "pending",
    "submitted",
    "running",
    "completed",
    "failed",
    "reconcile_required",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical fixed-5k training control plane built on the EXPERIMENTSV2 orchestration style."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train fixed-5k targets through the canonical control plane.")
    _add_common_args(train_parser)
    _add_training_selection_args(train_parser)
    _add_training_runtime_args(train_parser)

    reeval_parser = subparsers.add_parser("reeval", help="Reevaluate saved fixed-5k targets.")
    _add_common_args(reeval_parser)
    reeval_parser.add_argument("--families", nargs="+", default=[])
    reeval_parser.add_argument("--train-lengths", nargs="+", default=[])
    reeval_parser.add_argument("--target-scope", choices=["all", "direct", "windowed"], default="all")
    reeval_parser.add_argument("--missing-only", action="store_true")
    reeval_parser.add_argument("--envs", nargs="+", default=["ternary"])
    reeval_parser.add_argument("--max-workers", type=int, default=4)

    report_parser = subparsers.add_parser(
        "report",
        help="Rebuild metadata, parquet-backed reports, and dashboard artifacts.",
    )
    _add_common_args(report_parser)
    report_parser.add_argument("--rebuild-artifacts", action="store_true")

    migrate_parser = subparsers.add_parser(
        "migrate", help="Normalize legacy fixed-5k outputs into the canonical layout."
    )
    _add_common_args(migrate_parser)
    migrate_parser.add_argument("--rewrite-paths", action="store_true")
    migrate_parser.add_argument("--rebuild-artifacts", action="store_true")

    request_run_parser = subparsers.add_parser("request-run", help="Execute one tracked train request file.")
    _add_request_common_args(request_run_parser)
    request_run_parser.add_argument("--request-file", required=True)

    request_reconcile_parser = subparsers.add_parser(
        "request-reconcile",
        help="Scan tracked request files and run or rerun any request whose artifacts are incomplete.",
    )
    _add_request_common_args(request_reconcile_parser)
    request_reconcile_parser.add_argument("--retry-failed", action="store_true")

    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", type=int, default=DEFAULT_REQUEST_EXPERIMENT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--runner", choices=SUPPORTED_RUNNERS, default=DEFAULT_RUNNER)
    parser.add_argument("--emit-json", action="store_true")
    parser.add_argument("--mlflow-enabled", action="store_true")
    parser.add_argument("--mlflow-experiment", default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-tracking-uri", default="")


def _add_training_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection-mode", default="all")
    parser.add_argument("--families", nargs="+", default=[])
    parser.add_argument("--train-lengths", nargs="+", default=[])
    parser.add_argument("--window-lengths", nargs="+", default=[])
    parser.add_argument("--model-variations", nargs="+", default=[])
    parser.add_argument("--time-variants", nargs="+", default=["FULL"])
    parser.add_argument("--envs", nargs="+", default=["ternary"])


def _add_training_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--experiments-root", default=str(PROJECT_ROOT / "EXPERIMENTS"))
    parser.add_argument("--mamba-epochs", type=int, default=0)
    parser.add_argument("--mamba-batch-size", type=int, default=0)
    parser.add_argument("--mamba-disable-early-stopping", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-max-epochs", type=int, default=5000)
    parser.add_argument("--visualize-test-acc", action="store_true")
    parser.add_argument("--bandit-strategy", choices=["ts", "ucb"], default="ts")
    parser.add_argument("--post-base-total-updates", type=int, default=0)
    parser.add_argument("--post-base-learning-rate", type=float, default=0.0)
    parser.add_argument("--post-base-clip-epsilon", type=float, default=0.0)
    parser.add_argument("--post-base-entropy-coef", type=float, default=0.0)
    parser.add_argument("--post-base-value-loss-coef", type=float, default=0.0)
    parser.add_argument("--post-base-ppo-epochs", type=int, default=0)
    parser.add_argument("--post-base-minibatch-size", type=int, default=0)
    parser.add_argument("--retrain-epochs", type=int, default=0)


def _add_request_common_args(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument("--requests-dir", default=str(DEFAULT_REQUEST_FOLDER))
    parser.add_argument("--publish-azure", action="store_true")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_output_dir(experiment: int, output_dir: str) -> Path:
    text = str(output_dir or "").strip()
    if text:
        return Path(text).resolve()
    return (EXPERIMENTS_V2_ROOT / str(int(experiment))).resolve()


def _normalize_request_payload(payload: dict[str, Any], *, request_path: Path | None = None) -> dict[str, Any]:
    normalized = dict(payload)
    request_id = str(normalized.get("request_id") or (request_path.stem if request_path is not None else "")).strip()
    if not request_id:
        raise ValueError("request_id is required.")
    scope = str(normalized.get("scope") or DEFAULT_REQUEST_SCOPE).strip().lower()
    if scope != DEFAULT_REQUEST_SCOPE:
        raise ValueError(f"Unsupported request scope: {scope}. Only '{DEFAULT_REQUEST_SCOPE}' is supported.")
    status = str(normalized.get("status") or DEFAULT_REQUEST_STATUS).strip().lower()
    if status not in SUPPORTED_REQUEST_STATUSES:
        raise ValueError(f"Unsupported request status: {status}")
    family = str(normalized.get("family") or "").strip().upper()
    if not family:
        raise ValueError("family is required.")
    train_length = str(normalized.get("train_length") or "").strip().upper()
    if not train_length:
        raise ValueError("train_length is required.")
    model_variation = str(normalized.get("model_variation") or "BASE").strip().upper()
    env = str(normalized.get("env") or "ternary").strip().lower()
    if env not in {"ternary", "intensity11"}:
        raise ValueError(f"Unsupported env: {env}")
    time_variants = normalized.get("time_variants", ["FULL"])
    if not isinstance(time_variants, list) or not time_variants:
        raise ValueError("time_variants must be a non-empty list.")
    normalized_time_variants = [str(value).strip().upper() for value in time_variants if str(value).strip()]
    if not normalized_time_variants:
        raise ValueError("time_variants must contain at least one value.")
    window_length = str(normalized.get("window_length") or "").strip().upper()
    experiment = int(normalized.get("experiment") or DEFAULT_REQUEST_EXPERIMENT)
    output_dir = str(Path(str(normalized.get("output_dir") or _resolve_output_dir(experiment, "")).strip()).resolve())
    runner = str(normalized.get("runner") or DEFAULT_RUNNER).strip().lower()
    if runner not in SUPPORTED_RUNNERS:
        raise ValueError(f"Unsupported runner: {runner}")
    overwrite = bool(normalized.get("overwrite", False))
    selection_mode = str(normalized.get("selection_mode") or "all").strip()
    mlflow_config = normalized.get("mlflow", {})
    if not isinstance(mlflow_config, dict):
        mlflow_config = {}
    azure_config = normalized.get("azure", {})
    if not isinstance(azure_config, dict):
        azure_config = {}
    result = normalized.get("result", {})
    if not isinstance(result, dict):
        result = {}
    return {
        **normalized,
        "request_id": request_id,
        "scope": scope,
        "status": status,
        "family": family,
        "train_length": train_length,
        "window_length": window_length,
        "model_variation": model_variation,
        "time_variants": normalized_time_variants,
        "env": env,
        "experiment": experiment,
        "output_dir": output_dir,
        "runner": runner,
        "overwrite": overwrite,
        "selection_mode": selection_mode,
        "requested_at": str(normalized.get("requested_at") or _utc_now()),
        "requested_by": str(normalized.get("requested_by") or os.getenv("GITHUB_ACTOR") or DEFAULT_RUNNER),
        "commit_sha": str(normalized.get("commit_sha") or os.getenv("GITHUB_SHA") or "").strip(),
        "mlflow": {
            "enabled": bool(mlflow_config.get("enabled", True)),
            "experiment_name": str(mlflow_config.get("experiment_name") or DEFAULT_MLFLOW_EXPERIMENT),
            "tracking_uri": str(mlflow_config.get("tracking_uri") or "").strip(),
            "tags": mlflow_config.get("tags", {}) if isinstance(mlflow_config.get("tags", {}), dict) else {},
        },
        "azure": {
            "workspace": str(azure_config.get("workspace") or "").strip(),
            "compute": str(azure_config.get("compute") or "").strip(),
            "job_name": str(azure_config.get("job_name") or "").strip(),
            "job_id": str(azure_config.get("job_id") or "").strip(),
            "portal_url": str(azure_config.get("portal_url") or "").strip(),
            "experiment_name": str(azure_config.get("experiment_name") or "").strip(),
            "storage_container": str(azure_config.get("storage_container") or "").strip(),
            "storage_prefix": str(azure_config.get("storage_prefix") or "").strip(),
            "result_blob_uri": str(azure_config.get("result_blob_uri") or "").strip(),
            "artifact_root_uri": str(azure_config.get("artifact_root_uri") or "").strip(),
            "dashboard_uri": str(azure_config.get("dashboard_uri") or "").strip(),
            "model_artifact_uri": str(azure_config.get("model_artifact_uri") or "").strip(),
            "request_snapshot_blob_uri": str(azure_config.get("request_snapshot_blob_uri") or "").strip(),
            "published_at": str(azure_config.get("published_at") or "").strip(),
        },
        "result": result,
    }


def _load_request_file(request_path: Path) -> dict[str, Any]:
    payload = load_json(request_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Request file must contain a JSON object: {request_path}")
    return _normalize_request_payload(payload, request_path=request_path)


def _write_request_file(request_path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(request_path, payload)


def _validate_request_file_for_execution(request_path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        request = _load_request_file(request_path)
    except Exception as exc:
        return None, f"invalid request json: {exc}"
    # Normalize healthy request files back to canonical JSON formatting before execution.
    _write_request_file(request_path, request)
    return request, ""


def _apply_request_runtime_overrides(
    request: dict[str, Any],
    *,
    runner: str,
    mlflow_enabled_override: bool | None,
    mlflow_experiment_override: str,
    mlflow_tracking_uri_override: str,
) -> dict[str, Any]:
    updated = dict(request)
    updated["runner"] = str(runner)
    request_mlflow = dict(updated.get("mlflow", {}))
    if mlflow_enabled_override is not None:
        request_mlflow["enabled"] = bool(mlflow_enabled_override)
    if mlflow_experiment_override:
        request_mlflow["experiment_name"] = str(mlflow_experiment_override)
    if mlflow_tracking_uri_override:
        request_mlflow["tracking_uri"] = str(mlflow_tracking_uri_override)
    updated["mlflow"] = request_mlflow
    return updated


def _apply_azure_submission_metadata(request: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request)
    azure_meta = dict(updated.get("azure", {}))
    azure_meta.update(
        {
            "workspace": str(submission.get("workspace") or ""),
            "compute": str(submission.get("compute") or ""),
            "job_name": str(submission.get("job_name") or ""),
            "job_id": str(submission.get("job_id") or ""),
            "portal_url": str(submission.get("portal_url") or ""),
            "experiment_name": str(submission.get("experiment_name") or ""),
            "storage_container": str(submission.get("storage_container") or ""),
            "storage_prefix": str(submission.get("storage_prefix") or ""),
            "result_blob_uri": str(submission.get("result_blob_uri") or ""),
            "artifact_root_uri": str(submission.get("artifact_root_uri") or ""),
            "dashboard_uri": str(submission.get("dashboard_uri") or ""),
            "model_artifact_uri": str(submission.get("model_artifact_uri") or ""),
            "request_snapshot_blob_uri": str(submission.get("request_snapshot_blob_uri") or ""),
        }
    )
    updated["azure"] = azure_meta
    updated.setdefault("result", {})
    updated["result"].update(
        {
            "artifact_root": str(submission.get("artifact_root_uri") or ""),
            "model_artifact_uri": str(submission.get("model_artifact_uri") or ""),
            "dashboard_uri": str(submission.get("dashboard_uri") or ""),
        }
    )
    return updated


def _apply_azure_publish_metadata(request: dict[str, Any], publication: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request)
    azure_meta = dict(updated.get("azure", {}))
    published_at = _utc_now()
    azure_meta.update(
        {
            "storage_container": str(publication.get("storage_container") or ""),
            "storage_prefix": str(publication.get("storage_prefix") or ""),
            "artifact_root_uri": str(publication.get("artifact_root_uri") or ""),
            "dashboard_uri": str(publication.get("dashboard_uri") or ""),
            "model_artifact_uri": str(publication.get("model_artifact_uri") or ""),
            "request_snapshot_blob_uri": str(publication.get("request_snapshot_blob_uri") or ""),
            "result_blob_uri": str(publication.get("result_blob_uri") or ""),
            "published_at": published_at,
        }
    )
    updated["azure"] = azure_meta
    updated.setdefault("result", {})
    updated["result"].update(
        {
            "artifact_root": str(publication.get("artifact_root_uri") or updated["result"].get("artifact_root") or ""),
            "model_artifact_uri": str(publication.get("model_artifact_uri") or ""),
            "dashboard_uri": str(publication.get("dashboard_uri") or ""),
        }
    )
    return updated


def _sync_azure_request_state(request: dict[str, Any], request_path: Path) -> dict[str, Any]:
    snapshot = azure_ml.fetch_job_snapshot(request)
    updated = dict(request)
    updated["azure"] = {
        **(updated.get("azure", {}) if isinstance(updated.get("azure"), dict) else {}),
        "job_name": str(snapshot.get("job_name") or ""),
        "job_id": str(snapshot.get("job_id") or ""),
        "portal_url": str(snapshot.get("portal_url") or ""),
        "result_blob_uri": str(snapshot.get("result_blob_uri") or ""),
        "artifact_root_uri": str(snapshot.get("artifact_root_uri") or ""),
        "dashboard_uri": str(snapshot.get("dashboard_uri") or ""),
        "model_artifact_uri": str(snapshot.get("model_artifact_uri") or ""),
    }
    updated.setdefault("result", {})
    updated["result"].update(
        {
            "artifact_root": str(snapshot.get("artifact_root_uri") or ""),
            "model_artifact_uri": str(snapshot.get("model_artifact_uri") or ""),
            "dashboard_uri": str(snapshot.get("dashboard_uri") or ""),
        }
    )
    updated["status"] = str(snapshot.get("request_status") or "reconcile_required")

    if updated["status"] == "completed":
        result_blob_uri = str(snapshot.get("result_blob_uri") or "")
        if not result_blob_uri:
            updated["status"] = "reconcile_required"
            updated["result"]["error"] = "Azure job completed without a result blob URI."
        else:
            blob_payload = azure_ml.download_json_blob(result_blob_uri)
            updated["status"] = str(blob_payload.get("status") or "completed")
            updated["result"] = {
                **(updated.get("result", {}) if isinstance(updated.get("result"), dict) else {}),
                **blob_payload,
            }
            updated["result"]["artifact_root"] = str(
                blob_payload.get("artifact_root") or snapshot.get("artifact_root_uri") or ""
            )
            updated["result"]["model_artifact_uri"] = str(
                blob_payload.get("model_artifact_uri") or snapshot.get("model_artifact_uri") or ""
            )
            updated["result"]["dashboard_uri"] = str(
                blob_payload.get("dashboard_uri") or snapshot.get("dashboard_uri") or ""
            )
    elif updated["status"] == "failed":
        result_blob_uri = str(snapshot.get("result_blob_uri") or "")
        if result_blob_uri:
            try:
                blob_payload = azure_ml.download_json_blob(result_blob_uri)
            except Exception:
                blob_payload = {}
            if isinstance(blob_payload, dict):
                updated["result"] = {
                    **(updated.get("result", {}) if isinstance(updated.get("result"), dict) else {}),
                    **blob_payload,
                }

    _write_request_file(request_path, updated)
    return updated


def _publish_request_outputs(
    request_path: Path,
    request: dict[str, Any],
    *,
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    publication = azure_ml.publish_local_run(request_path, request, result_payload)
    updated = _apply_azure_publish_metadata(request, publication)
    _write_request_file(request_path, updated)
    return updated


def _needs_azure_publish(request: dict[str, Any]) -> bool:
    azure_meta = request.get("azure", {}) if isinstance(request.get("azure"), dict) else {}
    return not all(
        str(azure_meta.get(key) or "").strip()
        for key in ("artifact_root_uri", "dashboard_uri", "model_artifact_uri", "published_at")
    )


def _row_matches_request(row: dict[str, Any], request: dict[str, Any]) -> bool:
    return (
        str(row.get("family") or "").strip().upper() == request["family"]
        and str(row.get("train_length") or "").strip().upper() == request["train_length"]
        and str(row.get("window_length") or "").strip().upper() == request["window_length"]
        and str(row.get("model_variation") or "").strip().upper() == request["model_variation"]
        and str(row.get("time_variant") or "").strip().upper() in set(request["time_variants"])
        and str(row.get("env_version") or "ternary").strip().lower() == request["env"]
    )


def _matching_rows_for_request(output_root: Path, request: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_summary_rows(output_root)
    matches = [row for row in rows if _row_matches_request(row, request)]
    matches.sort(
        key=lambda row: (
            str(row.get("time_variant") or ""),
            int(row.get("threshold_pct") or 0),
        )
    )
    return matches


def _row_artifacts_exist(row: dict[str, Any]) -> bool:
    saved_model_path = str(row.get("saved_model_path") or row.get("source_saved_model_path") or "").strip()
    predictions_path = str(row.get("predictions_path") or row.get("source_predictions_path") or "").strip()
    if not saved_model_path or not Path(saved_model_path).exists():
        return False
    if not predictions_path or not Path(predictions_path).exists():
        return False
    return True


def _request_artifacts_exist(request: dict[str, Any]) -> bool:
    result = request.get("result", {})
    if not isinstance(result, dict):
        return False
    for key in ("summary_path", "manifest_path", "models_path"):
        path_value = str(result.get(key) or "").strip()
        if not path_value or not Path(path_value).exists():
            return False
    matched_rows = result.get("matched_rows", [])
    if not isinstance(matched_rows, list) or not matched_rows:
        return False
    for row in matched_rows:
        if not isinstance(row, dict):
            return False
        saved_model_path = str(row.get("saved_model_path") or "").strip()
        predictions_path = str(row.get("predictions_path") or "").strip()
        if not saved_model_path or not Path(saved_model_path).exists():
            return False
        if not predictions_path or not Path(predictions_path).exists():
            return False
    return True


def _machine_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    test_payload = row.get("test", {}) if isinstance(row.get("test"), dict) else {}
    return {
        "name": row.get("name"),
        "status": row.get("status"),
        "time_variant": row.get("time_variant"),
        "threshold_pct": row.get("threshold_pct"),
        "env_version": row.get("env_version"),
        "saved_model_path": row.get("saved_model_path") or row.get("source_saved_model_path"),
        "predictions_path": row.get("predictions_path") or row.get("source_predictions_path"),
        "test_accuracy": test_payload.get("accuracy"),
        "daily_net_wins": row.get("daily_net_wins"),
    }


def _result_payload_from_command(
    output_root: Path, command_result: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    models_path = summary_rows_path(output_root)
    matched_rows = [_machine_row_summary(row) for row in _matching_rows_for_request(output_root, request)]
    return {
        "summary_path": str(command_result.get("summary_path") or ""),
        "manifest_path": str(command_result.get("manifest_path") or ""),
        "models_path": str(models_path.resolve()),
        "artifact_root": str(output_root.resolve()),
        "matched_rows": matched_rows,
        "completed_at": _utc_now(),
    }


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "matched_row_count": float(len(rows)),
        "completed_row_count": float(sum(1 for row in rows if str(row.get("status") or "") == "completed")),
    }
    accuracies = []
    daily_net_wins = []
    for row in rows:
        test_payload = row.get("test", {}) if isinstance(row.get("test"), dict) else {}
        if test_payload.get("accuracy") is not None:
            accuracies.append(float(test_payload["accuracy"]))
        if row.get("daily_net_wins") is not None:
            daily_net_wins.append(float(row["daily_net_wins"]))
    if accuracies:
        metrics["best_test_accuracy"] = max(accuracies)
        metrics["mean_test_accuracy"] = sum(accuracies) / len(accuracies)
    if daily_net_wins:
        metrics["best_daily_net_wins"] = max(daily_net_wins)
        metrics["mean_daily_net_wins"] = sum(daily_net_wins) / len(daily_net_wins)
    return metrics


@contextmanager
def _mlflow_run_context(
    *,
    enabled: bool,
    experiment_name: str,
    tracking_uri: str,
    params: dict[str, Any],
    tags: dict[str, Any],
):
    if not enabled:
        yield None
        return
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name or DEFAULT_MLFLOW_EXPERIMENT)
    clean_params = {key: str(value) for key, value in params.items() if value is not None and str(value) != ""}
    clean_tags = {key: str(value) for key, value in tags.items() if value is not None and str(value) != ""}
    with mlflow.start_run(run_name=clean_params.get("request_id") or clean_params.get("command")) as run:
        if clean_tags:
            mlflow.set_tags(clean_tags)
        if clean_params:
            mlflow.log_params(clean_params)
        yield run


def _log_mlflow_result(
    run: Any,
    *,
    result_payload: dict[str, Any],
    request_snapshot: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    if run is None:
        return ""
    import mlflow

    if metrics:
        mlflow.log_metrics(metrics)
    mlflow.log_text(json.dumps(result_payload, indent=2), "result.json")
    if request_snapshot is not None:
        mlflow.log_text(json.dumps(request_snapshot, indent=2), "request.json")
    for key in ("summary_path", "manifest_path"):
        path_value = str(result_payload.get(key) or "").strip()
        if path_value and Path(path_value).exists():
            mlflow.log_artifact(path_value)
    return str(run.info.run_id)


def _training_namespace_from_request(request: dict[str, Any], runner: str) -> argparse.Namespace:
    return argparse.Namespace(
        command="train",
        experiment=int(request["experiment"]),
        output_dir=str(request["output_dir"]),
        runner=runner,
        emit_json=True,
        mlflow_enabled=bool(request["mlflow"]["enabled"]),
        mlflow_experiment=str(request["mlflow"]["experiment_name"]),
        mlflow_tracking_uri=str(request["mlflow"]["tracking_uri"]),
        selection_mode=str(request.get("selection_mode") or "all"),
        families=[request["family"]],
        train_lengths=[request["train_length"]],
        window_lengths=[request["window_length"]] if request["window_length"] else [],
        model_variations=[request["model_variation"]],
        time_variants=list(request["time_variants"]),
        envs=[request["env"]],
        device=str(request.get("device") or "auto"),
        overwrite=bool(request.get("overwrite", False)),
        max_workers=int(request.get("max_workers") or 4),
        verbose=int(request.get("verbose") or 1),
        experiments_root=str(request.get("experiments_root") or (PROJECT_ROOT / "EXPERIMENTS")),
        mamba_epochs=int(request.get("mamba_epochs") or 0),
        mamba_batch_size=int(request.get("mamba_batch_size") or 0),
        mamba_disable_early_stopping=bool(request.get("mamba_disable_early_stopping", False)),
        overfit=bool(request.get("overfit", False)),
        overfit_max_epochs=int(request.get("overfit_max_epochs") or 5000),
        visualize_test_acc=bool(request.get("visualize_test_acc", False)),
        bandit_strategy=str(request.get("bandit_strategy") or "ts"),
        post_base_total_updates=int(request.get("post_base_total_updates") or 0),
        post_base_learning_rate=float(request.get("post_base_learning_rate") or 0.0),
        post_base_clip_epsilon=float(request.get("post_base_clip_epsilon") or 0.0),
        post_base_entropy_coef=float(request.get("post_base_entropy_coef") or 0.0),
        post_base_value_loss_coef=float(request.get("post_base_value_loss_coef") or 0.0),
        post_base_ppo_epochs=int(request.get("post_base_ppo_epochs") or 0),
        post_base_minibatch_size=int(request.get("post_base_minibatch_size") or 0),
        retrain_epochs=int(request.get("retrain_epochs") or 0),
    )


def _result_envelope(command: str, *, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"command": command, "status": status, **payload}


def run_train_command(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args.experiment, args.output_dir)
    params = {
        "command": "train",
        "runner": args.runner,
        "experiment": args.experiment,
        "families": ",".join(args.families),
        "train_lengths": ",".join(args.train_lengths),
        "window_lengths": ",".join(args.window_lengths),
        "model_variations": ",".join(args.model_variations),
        "envs": ",".join(args.envs),
        "output_dir": str(output_root),
    }
    tags = {
        "runner": args.runner,
        "scope": DEFAULT_REQUEST_SCOPE,
        "workflow_source": "manual_cli",
    }
    with _mlflow_run_context(
        enabled=bool(args.mlflow_enabled),
        experiment_name=str(args.mlflow_experiment),
        tracking_uri=str(args.mlflow_tracking_uri),
        params=params,
        tags=tags,
    ) as run:
        command_result = unified_v2.run_train(
            argparse.Namespace(
                experiment=args.experiment,
                output_dir=str(output_root),
                selection_mode=args.selection_mode,
                families=list(args.families),
                train_lengths=list(args.train_lengths),
                window_lengths=list(args.window_lengths),
                model_variations=list(args.model_variations),
                time_variants=list(args.time_variants),
                envs=list(args.envs),
                device=args.device,
                overwrite=bool(args.overwrite),
                max_workers=int(args.max_workers),
                experiments_root=str(args.experiments_root),
                verbose=int(args.verbose),
                mamba_epochs=int(args.mamba_epochs),
                mamba_batch_size=int(args.mamba_batch_size),
                mamba_disable_early_stopping=bool(args.mamba_disable_early_stopping),
                overfit=bool(args.overfit),
                overfit_max_epochs=int(args.overfit_max_epochs),
                visualize_test_acc=bool(args.visualize_test_acc),
                bandit_strategy=str(args.bandit_strategy),
                post_base_total_updates=int(args.post_base_total_updates),
                post_base_learning_rate=float(args.post_base_learning_rate),
                post_base_clip_epsilon=float(args.post_base_clip_epsilon),
                post_base_entropy_coef=float(args.post_base_entropy_coef),
                post_base_value_loss_coef=float(args.post_base_value_loss_coef),
                post_base_ppo_epochs=int(args.post_base_ppo_epochs),
                post_base_minibatch_size=int(args.post_base_minibatch_size),
                retrain_epochs=int(args.retrain_epochs),
            )
        )
        rows = load_summary_rows(output_root)
        mlflow_run_id = _log_mlflow_result(
            run,
            result_payload=command_result,
            metrics=_metrics_from_rows(rows),
        )
    return _result_envelope(
        "train",
        status="completed",
        payload={
            **command_result,
            "output_dir": str(output_root),
            "models_path": str(summary_rows_path(output_root).resolve()),
            "mlflow_run_id": mlflow_run_id,
        },
    )


def run_reeval_command(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args.experiment, args.output_dir)
    params = {
        "command": "reeval",
        "runner": args.runner,
        "experiment": args.experiment,
        "families": ",".join(args.families),
        "train_lengths": ",".join(args.train_lengths),
        "envs": ",".join(args.envs),
        "output_dir": str(output_root),
    }
    tags = {
        "runner": args.runner,
        "scope": DEFAULT_REQUEST_SCOPE,
        "workflow_source": "manual_cli",
    }
    with _mlflow_run_context(
        enabled=bool(args.mlflow_enabled),
        experiment_name=str(args.mlflow_experiment),
        tracking_uri=str(args.mlflow_tracking_uri),
        params=params,
        tags=tags,
    ) as run:
        command_result = unified_v2.run_reeval(
            argparse.Namespace(
                experiment=args.experiment,
                output_dir=str(output_root),
                families=list(args.families),
                train_lengths=list(args.train_lengths),
                target_scope=str(args.target_scope),
                missing_only=bool(args.missing_only),
                max_workers=int(args.max_workers),
                envs=list(args.envs),
            )
        )
        rows = load_summary_rows(output_root)
        mlflow_run_id = _log_mlflow_result(run, result_payload=command_result, metrics=_metrics_from_rows(rows))
    return _result_envelope(
        "reeval",
        status="completed",
        payload={
            **command_result,
            "output_dir": str(output_root),
            "mlflow_run_id": mlflow_run_id,
        },
    )


def run_report_command(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args.experiment, args.output_dir)
    params = {
        "command": "report",
        "runner": args.runner,
        "experiment": args.experiment,
        "output_dir": str(output_root),
    }
    tags = {
        "runner": args.runner,
        "scope": DEFAULT_REQUEST_SCOPE,
        "workflow_source": "manual_cli",
    }
    with _mlflow_run_context(
        enabled=bool(args.mlflow_enabled),
        experiment_name=str(args.mlflow_experiment),
        tracking_uri=str(args.mlflow_tracking_uri),
        params=params,
        tags=tags,
    ) as run:
        command_result = unified_v2.run_report(
            argparse.Namespace(
                experiment=args.experiment,
                output_dir=str(output_root),
                rebuild_artifacts=bool(args.rebuild_artifacts),
            )
        )
        rows = load_summary_rows(output_root)
        mlflow_run_id = _log_mlflow_result(run, result_payload=command_result, metrics=_metrics_from_rows(rows))
    return _result_envelope(
        "report",
        status="completed",
        payload={
            **command_result,
            "output_dir": str(output_root),
            "mlflow_run_id": mlflow_run_id,
        },
    )


def run_migrate_command(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args.experiment, args.output_dir)
    params = {
        "command": "migrate",
        "runner": args.runner,
        "experiment": args.experiment,
        "output_dir": str(output_root),
        "rewrite_paths": bool(args.rewrite_paths),
    }
    tags = {
        "runner": args.runner,
        "scope": DEFAULT_REQUEST_SCOPE,
        "workflow_source": "manual_cli",
    }
    with _mlflow_run_context(
        enabled=bool(args.mlflow_enabled),
        experiment_name=str(args.mlflow_experiment),
        tracking_uri=str(args.mlflow_tracking_uri),
        params=params,
        tags=tags,
    ) as run:
        command_result = unified_v2.run_migrate(
            argparse.Namespace(
                experiment=args.experiment,
                output_dir=str(output_root),
                rewrite_paths=bool(args.rewrite_paths),
                rebuild_artifacts=bool(args.rebuild_artifacts),
            )
        )
        rows = load_summary_rows(output_root)
        mlflow_run_id = _log_mlflow_result(run, result_payload=command_result, metrics=_metrics_from_rows(rows))
    return _result_envelope(
        "migrate",
        status="completed",
        payload={
            **command_result,
            "output_dir": str(output_root),
            "mlflow_run_id": mlflow_run_id,
        },
    )


def _run_request_file(
    request_path: Path,
    *,
    runner: str,
    publish_azure: bool,
    mlflow_enabled_override: bool | None,
    mlflow_experiment_override: str,
    mlflow_tracking_uri_override: str,
    retry_failed: bool = False,
) -> dict[str, Any]:
    request, validation_error = _validate_request_file_for_execution(request_path)
    if request is None:
        return _result_envelope(
            "request-run",
            status="skipped",
            payload={
                "request_file": str(request_path),
                "request_id": request_path.stem,
                "reason": validation_error,
            },
        )
    status = str(request.get("status") or DEFAULT_REQUEST_STATUS).lower()
    if status == "failed" and not retry_failed:
        return _result_envelope(
            "request-run",
            status="skipped",
            payload={
                "request_file": str(request_path),
                "reason": "request already failed",
                "request_id": request["request_id"],
            },
        )
    if runner != "azure" and _request_artifacts_exist(request) and not bool(request.get("overwrite", False)):
        request["status"] = "completed"
        request.setdefault("result", {})
        request["result"]["completed_at"] = request["result"].get("completed_at") or _utc_now()
        if publish_azure:
            try:
                request = _publish_request_outputs(request_path, request, result_payload=dict(request["result"]))
                request["status"] = "completed"
            except Exception as exc:
                request["status"] = "reconcile_required"
                request["result"] = {
                    **(request.get("result", {}) if isinstance(request.get("result"), dict) else {}),
                    "error": f"Azure publish failed after local training: {exc}",
                    "published_to_azure": False,
                }
            _write_request_file(request_path, request)
        else:
            _write_request_file(request_path, request)
        return _result_envelope(
            "request-run",
            status=str(request["status"]),
            payload={
                "request_file": str(request_path),
                "request_id": request["request_id"],
                "skipped_training": True,
                "result": request["result"],
            },
        )

    request = _apply_request_runtime_overrides(
        request,
        runner=runner,
        mlflow_enabled_override=mlflow_enabled_override,
        mlflow_experiment_override=mlflow_experiment_override,
        mlflow_tracking_uri_override=mlflow_tracking_uri_override,
    )
    if not str(request.get("commit_sha") or "").strip():
        request["commit_sha"] = str(os.getenv("GITHUB_SHA") or "").strip()

    train_args = _training_namespace_from_request(request, runner)
    request_mlflow = dict(request.get("mlflow", {}))
    train_args.mlflow_enabled = bool(request_mlflow.get("enabled", True))
    train_args.mlflow_experiment = str(request_mlflow.get("experiment_name") or DEFAULT_MLFLOW_EXPERIMENT)
    train_args.mlflow_tracking_uri = str(request_mlflow.get("tracking_uri") or "")

    if runner == "azure":
        try:
            _write_request_file(request_path, request)
            submission = azure_ml.submit_train_request(request_path, request, vars(train_args))
            request = _apply_azure_submission_metadata(request, submission)
            request["status"] = "submitted"
            request["last_submitted_at"] = _utc_now()
            _write_request_file(request_path, request)
            return _result_envelope(
                "request-run",
                status="submitted",
                payload={
                    "request_file": str(request_path),
                    "request_id": request["request_id"],
                    "azure": request["azure"],
                    "result": request.get("result", {}),
                },
            )
        except Exception as exc:
            request["status"] = "failed"
            request["result"] = {
                **(request.get("result", {}) if isinstance(request.get("result"), dict) else {}),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": _utc_now(),
            }
            _write_request_file(request_path, request)
            return _result_envelope(
                "request-run",
                status="failed",
                payload={
                    "request_file": str(request_path),
                    "request_id": request["request_id"],
                    "result": request["result"],
                },
            )

    request["status"] = "running"
    request["last_started_at"] = _utc_now()
    _write_request_file(request_path, request)

    try:
        command_result = run_train_command(train_args)
        output_root = _resolve_output_dir(int(request["experiment"]), str(request["output_dir"]))
        result_payload = _result_payload_from_command(output_root, command_result, request)
        healthy = _request_artifacts_exist({"result": result_payload})
        request["result"] = {
            **result_payload,
            "healthy": healthy,
            "mlflow_run_id": command_result.get("mlflow_run_id") or "",
        }
        request["status"] = "completed" if healthy else "reconcile_required"
        if healthy and publish_azure:
            try:
                request = _publish_request_outputs(request_path, request, result_payload=request["result"])
                request["result"]["published_to_azure"] = True
                request["status"] = "completed"
            except Exception as exc:
                request["status"] = "reconcile_required"
                request["result"] = {
                    **(request.get("result", {}) if isinstance(request.get("result"), dict) else {}),
                    "error": f"Azure publish failed after local training: {exc}",
                    "published_to_azure": False,
                }
        _write_request_file(request_path, request)
        return _result_envelope(
            "request-run",
            status=request["status"],
            payload={
                "request_file": str(request_path),
                "request_id": request["request_id"],
                "result": request["result"],
            },
        )
    except Exception as exc:
        request["status"] = "failed"
        request["result"] = {
            **(request.get("result", {}) if isinstance(request.get("result"), dict) else {}),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": _utc_now(),
        }
        _write_request_file(request_path, request)
        raise


def run_request_run_command(args: argparse.Namespace) -> dict[str, Any]:
    request_path = Path(args.request_file).resolve()
    result = _run_request_file(
        request_path,
        runner=str(args.runner),
        publish_azure=bool(args.publish_azure),
        mlflow_enabled_override=(True if args.mlflow_enabled else None),
        mlflow_experiment_override=str(args.mlflow_experiment or ""),
        mlflow_tracking_uri_override=str(args.mlflow_tracking_uri or ""),
        retry_failed=True,
    )
    return result


def run_request_reconcile_command(args: argparse.Namespace) -> dict[str, Any]:
    requests_dir = Path(args.requests_dir).resolve()
    request_paths = sorted(requests_dir.rglob("*.json"))
    summary = {
        "requests_dir": str(requests_dir),
        "scanned": 0,
        "healthy": 0,
        "retrained": 0,
        "failed": 0,
        "skipped": 0,
        "results": [],
    }
    for request_path in request_paths:
        summary["scanned"] += 1
        request, validation_error = _validate_request_file_for_execution(request_path)
        if request is None:
            summary["skipped"] += 1
            summary["results"].append(
                {
                    "request_file": str(request_path),
                    "request_id": request_path.stem,
                    "status": "skipped",
                    "reason": validation_error,
                }
            )
            continue
        status = str(request.get("status") or DEFAULT_REQUEST_STATUS).lower()
        request_runner = str(request.get("runner") or args.runner or DEFAULT_RUNNER).lower()
        if request_runner == "azure":
            if status == "pending":
                result = _run_request_file(
                    request_path,
                    runner="azure",
                    publish_azure=False,
                    mlflow_enabled_override=(True if args.mlflow_enabled else None),
                    mlflow_experiment_override=str(args.mlflow_experiment or ""),
                    mlflow_tracking_uri_override=str(args.mlflow_tracking_uri or ""),
                    retry_failed=bool(args.retry_failed),
                )
                if result.get("status") in {"submitted", "running", "completed"}:
                    summary["retrained"] += 1
                elif result.get("status") == "failed":
                    summary["failed"] += 1
                else:
                    summary["skipped"] += 1
                summary["results"].append(result)
                continue
            if status in {"submitted", "running", "completed", "reconcile_required"} or (
                status == "failed" and bool(args.retry_failed)
            ):
                try:
                    synced = _sync_azure_request_state(request, request_path)
                    synced_status = str(synced.get("status") or "")
                    if synced_status == "completed":
                        summary["healthy"] += 1
                    elif synced_status in {"submitted", "running"}:
                        summary["retrained"] += 1
                    elif synced_status == "failed":
                        summary["failed"] += 1
                    else:
                        summary["skipped"] += 1
                    summary["results"].append(
                        {
                            "request_file": str(request_path),
                            "request_id": synced["request_id"],
                            "status": synced_status,
                            "azure": synced.get("azure", {}),
                            "result": synced.get("result", {}),
                        }
                    )
                except Exception as exc:
                    summary["failed"] += 1
                    summary["results"].append(
                        {
                            "request_file": str(request_path),
                            "request_id": request.get("request_id"),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                continue
        needs_train = False
        if status in {"pending", "running", "reconcile_required"}:
            needs_train = True
        elif status == "completed":
            needs_train = not _request_artifacts_exist(request)
        elif status == "failed":
            needs_train = bool(args.retry_failed)
        if not needs_train and _request_artifacts_exist(request):
            request["status"] = "completed"
            if bool(args.publish_azure) and _needs_azure_publish(request):
                try:
                    request = _publish_request_outputs(
                        request_path, request, result_payload=dict(request.get("result", {}))
                    )
                    request["result"]["published_to_azure"] = True
                except Exception as exc:
                    request["status"] = "reconcile_required"
                    request.setdefault("result", {})
                    request["result"]["error"] = f"Azure publish failed during reconcile: {exc}"
                    request["result"]["published_to_azure"] = False
            _write_request_file(request_path, request)
            if request["status"] == "completed":
                summary["healthy"] += 1
            else:
                summary["failed"] += 1
            summary["results"].append(
                {
                    "request_file": str(request_path),
                    "request_id": request["request_id"],
                    "status": "healthy" if request["status"] == "completed" else request["status"],
                }
            )
            continue
        if not needs_train:
            summary["skipped"] += 1
            summary["results"].append(
                {
                    "request_file": str(request_path),
                    "request_id": request["request_id"],
                    "status": "skipped",
                }
            )
            continue
        try:
            result = _run_request_file(
                request_path,
                runner=str(args.runner),
                publish_azure=bool(args.publish_azure),
                mlflow_enabled_override=(True if args.mlflow_enabled else None),
                mlflow_experiment_override=str(args.mlflow_experiment or ""),
                mlflow_tracking_uri_override=str(args.mlflow_tracking_uri or ""),
                retry_failed=bool(args.retry_failed),
            )
            if result.get("status") == "completed":
                summary["retrained"] += 1
            else:
                summary["failed"] += 1
            summary["results"].append(result)
        except Exception as exc:
            summary["failed"] += 1
            summary["results"].append(
                {
                    "request_file": str(request_path),
                    "request_id": request.get("request_id"),
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return _result_envelope("request-reconcile", status="completed", payload=summary)


def main() -> int:
    args = parse_args()
    if args.command == "train":
        result = run_train_command(args)
    elif args.command == "reeval":
        result = run_reeval_command(args)
    elif args.command == "report":
        result = run_report_command(args)
    elif args.command == "migrate":
        result = run_migrate_command(args)
    elif args.command == "request-run":
        result = run_request_run_command(args)
    elif args.command == "request-reconcile":
        result = run_request_reconcile_command(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
