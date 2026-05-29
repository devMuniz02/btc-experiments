from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train as train_cli
from src.utils import azure_ml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one fixed-5k request inside Azure ML and publish results.")
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--train-command-json", required=True)
    parser.add_argument("--report-command-json", required=True)
    return parser.parse_args()


def _load_command(command_json: str) -> list[str]:
    payload = json.loads(command_json)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("Expected command JSON to be a list of strings.")
    return list(payload)


def _run_json_command(argv: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise ValueError(f"Command produced no JSON output: {' '.join(argv)}")
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object output: {' '.join(argv)}")
    return payload


def _resolve_best_model_uri(result_payload: dict[str, Any]) -> str:
    rows = result_payload.get("matched_rows", [])
    if not isinstance(rows, list) or not rows:
        return str(result_payload.get("model_artifact_uri") or "")

    def ranking_key(row: dict[str, Any]) -> tuple[float, float, str]:
        test_accuracy = float(row.get("test_accuracy") or 0.0)
        daily_net_wins = float(row.get("daily_net_wins") or 0.0)
        return (-test_accuracy, -daily_net_wins, str(row.get("name") or ""))

    best_row = sorted((row for row in rows if isinstance(row, dict)), key=ranking_key)[0]
    return str(best_row.get("saved_model_path") or "")


def _blobify_result_paths(
    result_payload: dict[str, Any],
    *,
    output_root: Path,
    blob_root_uri: str,
) -> dict[str, Any]:
    converted = dict(result_payload)
    for key in ("summary_path", "manifest_path", "models_path", "artifact_root"):
        value = str(converted.get(key) or "").strip()
        if not value:
            continue
        if key == "artifact_root":
            converted[key] = blob_root_uri
            continue
        converted[key] = azure_ml.map_local_path_to_blob_uri(value, str(output_root), blob_root_uri)

    rows = converted.get("matched_rows", [])
    if isinstance(rows, list):
        converted_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            converted_row = dict(row)
            for key in ("saved_model_path", "predictions_path"):
                value = str(converted_row.get(key) or "").strip()
                if value:
                    converted_row[key] = azure_ml.map_local_path_to_blob_uri(value, str(output_root), blob_root_uri)
            converted_rows.append(converted_row)
        converted["matched_rows"] = converted_rows
    return converted


def main() -> int:
    args = parse_args()
    request_path = Path(args.request_file).resolve()
    request = train_cli._load_request_file(request_path)
    config = azure_ml.load_azure_config()
    azure_ml.validate_azure_config(config)

    request_snapshot_blob_uri = str(
        request.get("azure", {}).get("request_snapshot_blob_uri") if isinstance(request.get("azure"), dict) else ""
    ) or azure_ml.build_request_snapshot_blob_uri(config, request)
    result_blob_uri = str(
        request.get("azure", {}).get("result_blob_uri") if isinstance(request.get("azure"), dict) else ""
    ) or azure_ml.build_result_blob_uri(config, request)
    artifact_root_uri = str(
        request.get("azure", {}).get("artifact_root_uri") if isinstance(request.get("azure"), dict) else ""
    ) or azure_ml.build_request_artifact_root_uri(config, request)
    dashboard_uri = str(
        request.get("azure", {}).get("dashboard_uri") if isinstance(request.get("azure"), dict) else ""
    ) or azure_ml.build_dashboard_uri(config, request)

    azure_ml.upload_json_blob(request_snapshot_blob_uri, request, config=config)

    try:
        train_result = _run_json_command(_load_command(args.train_command_json))
        output_root = Path(str(train_result.get("output_dir") or request["output_dir"])).resolve()
        report_result = _run_json_command(_load_command(args.report_command_json))
        result_payload = train_cli._result_payload_from_command(output_root, train_result, request)
        result_payload["mlflow_run_id"] = str(train_result.get("mlflow_run_id") or "")
        result_payload["report_result"] = report_result

        azure_ml.upload_directory_to_blob(output_root, artifact_root_uri, config=config)
        blob_result = _blobify_result_paths(result_payload, output_root=output_root, blob_root_uri=artifact_root_uri)
        best_model_local_path = _resolve_best_model_uri(result_payload)
        model_artifact_uri = azure_ml.map_local_path_to_blob_uri(
            best_model_local_path,
            str(output_root),
            artifact_root_uri,
        )
        blob_result["artifact_root"] = artifact_root_uri
        blob_result["model_artifact_uri"] = model_artifact_uri
        blob_result["dashboard_uri"] = dashboard_uri
        blob_result["status"] = "completed" if bool(blob_result.get("matched_rows")) else "reconcile_required"
        blob_result["request_id"] = str(request.get("request_id") or request_path.stem)
        azure_ml.upload_json_blob(result_blob_uri, blob_result, config=config)
        return 0
    except Exception as exc:
        failure_payload = {
            "status": "failed",
            "request_id": str(request.get("request_id") or request_path.stem),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "artifact_root": artifact_root_uri,
            "dashboard_uri": dashboard_uri,
        }
        try:
            azure_ml.upload_json_blob(result_blob_uri, failure_payload, config=config)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
