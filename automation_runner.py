from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.ensemble import build_ensemble_predictions
from src.models.training import train_model
from automation.update_readme import update_readme
from src.utils import (
    QuantStreamPaths,
    canonicalize_payload,
    compute_model_id,
    ensure_quant_stream_layout,
    read_yaml_file,
    utc_now_iso,
    write_yaml_file,
)

SUPPORTED_MODEL_TYPES = {
    "actor_critic",
    "bc",
    "dagger",
    "ensemble",
    "lstm",
    "mamba",
    "mamba_post_base",
    "nn",
    "ppo",
    "ppo_continue",
    "rf",
    "transformer",
    "xgboost",
}
SUPPORTED_TRAIN_MODES = {
    "post_base",
    "reinforcement_ppo",
    "sliding_window_continue",
    "sliding_window_current_only",
    "sliding_window_retrain",
    "static_baseline",
    "windowed_continue",
    "windowed_isolated",
    "windowed_retrain",
}
REQUIRED_ROOT_KEYS = {"model_type", "variation_id", "train_mode", "hyperparameters"}


@dataclass(frozen=True)
class RequestOutcome:
    status: str
    source_path: str
    target_path: str
    model_id: str = ""
    reason: str = ""


def validate_request(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_ROOT_KEYS - set(payload))
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")
    model_type = str(payload["model_type"]).strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {payload['model_type']}")
    train_mode = str(payload["train_mode"]).strip().lower()
    if train_mode not in SUPPORTED_TRAIN_MODES:
        raise ValueError(f"Unsupported train_mode: {payload['train_mode']}")
    if int(payload["variation_id"]) <= 0:
        raise ValueError("variation_id must be positive.")
    hyperparameters = payload["hyperparameters"]
    if not isinstance(hyperparameters, dict):
        raise ValueError("hyperparameters must be a mapping.")
    if model_type == "ensemble":
        models_pool = hyperparameters.get("models_pool")
        mechanism = str(hyperparameters.get("voting_mechanism", "")).strip().lower()
        if not isinstance(models_pool, list) or not models_pool:
            raise ValueError("ensemble hyperparameters.models_pool must be a non-empty list.")
        if mechanism not in {"hard_majority", "soft_average", "unanimity"}:
            raise ValueError("ensemble voting_mechanism must be hard_majority, soft_average, or unanimity.")


def reject_request(paths: QuantStreamPaths, source_path: Path, message: str) -> RequestOutcome:
    target = paths.rejected_runs_dir / source_path.name
    comment = (
        f"# Quant-Stream validation failure\n"
        f"# timestamp: {utc_now_iso()}\n"
        f"# exception: {message}\n"
        f"# remediation: edit the request and place it back in automation/run_requests.\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(comment + source_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_path.unlink()
    return RequestOutcome("rejected", str(source_path), str(target), reason=message)


def has_tombstone(paths: QuantStreamPaths, payload: dict[str, Any]) -> bool:
    target = canonicalize_payload(
        {
            "model_type": str(payload["model_type"]).lower(),
            "variation_id": int(payload["variation_id"]),
            "train_mode": str(payload["train_mode"]).lower(),
            "hyperparameters": payload["hyperparameters"],
        }
    )
    for tombstone in paths.deleted_runs_dir.glob("*.y*ml"):
        try:
            deleted_payload = read_yaml_file(tombstone)
            candidate = canonicalize_payload(
                {
                    "model_type": str(deleted_payload.get("model_type", "")).lower(),
                    "variation_id": int(deleted_payload.get("variation_id", 0)),
                    "train_mode": str(deleted_payload.get("train_mode", "")).lower(),
                    "hyperparameters": deleted_payload.get("hyperparameters", {}),
                }
            )
        except Exception:
            continue
        if candidate == target:
            return True
    return False


def load_clean_frame(paths: QuantStreamPaths, variation_id: int) -> pd.DataFrame:
    clean_path = paths.clean_data_path(variation_id)
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean dataset for var_{variation_id}: {clean_path}")
    return pd.read_parquet(clean_path)


def split_train_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 105000:
        raise ValueError(f"Expected at least 105000 rows, found {len(frame)}.")
    return frame.iloc[:100000].reset_index(drop=True), frame.iloc[100000:105000].reset_index(drop=True)


def heuristic_predictions(test_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    close = test_frame["close"].to_numpy(dtype=float)
    open_ = test_frame["open"].to_numpy(dtype=float)
    delta = close - open_
    scale = np.maximum(np.abs(open_) * 0.001, 1e-9)
    confidence = np.clip(np.abs(delta) / scale, 0.0, 1.0)
    pred = np.where(confidence < 0.15, 0, np.where(delta >= 0, 1, -1)).astype(np.int8)
    return pred, confidence.astype(np.float32)


def read_global_results(paths: QuantStreamPaths, variation_id: int, test_frame: pd.DataFrame) -> pd.DataFrame:
    result_path = paths.global_results_path(variation_id)
    if result_path.exists():
        results = pd.read_parquet(result_path)
        if "actual" in results.columns and "target" not in results.columns:
            results = results.rename(columns={"actual": "target"})
        if "actual" in results.columns and "target" in results.columns:
            results = results.drop(columns=["actual"])
        if "target" not in results.columns and "target" in test_frame.columns:
            results["target"] = test_frame["target"].to_numpy()
        return results
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test_frame["timestamp"], utc=True).astype(str),
            "open": test_frame["open"].to_numpy(),
            "close": test_frame["close"].to_numpy(),
        }
    )
    if "target" in test_frame.columns:
        base["target"] = test_frame["target"].to_numpy()
    return base


def write_model_artifacts(
    paths: QuantStreamPaths,
    variation_id: int,
    model_id: str,
    payload: dict[str, Any],
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    model_dir = paths.variation_models_dir(variation_id) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "weights.bin"
    if not weights_path.exists():
        weights_path.write_bytes(
            json.dumps({"model_id": model_id, "model_type": payload["model_type"]}, sort_keys=True).encode("utf-8")
        )
    metadata = {**payload, "model_id": model_id, "finalized_at": utc_now_iso()}
    (model_dir / "hyperparameters.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    tracking_line = json.dumps(
        {
            "event": "model_finalized",
            "timestamp": utc_now_iso(),
            "model_id": model_id,
            "prediction_count": int(len(predictions)),
            "mean_probability": float(np.mean(probabilities)) if len(probabilities) else 0.0,
        },
        sort_keys=True,
    )
    (model_dir / "tracking_buffer.json").write_text(tracking_line + "\n", encoding="utf-8")


def active_scaler_references(paths: QuantStreamPaths) -> set[str]:
    references: set[str] = set()
    for metadata_path in paths.models_dev_dir.glob("var_*/*/hyperparameters.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scaler_ref = str(payload.get("scaler_path") or payload.get("hyperparameters", {}).get("scaler_path") or "")
        if scaler_ref:
            references.add(Path(scaler_ref).name)
    return references


def purge_unreferenced_scalers(paths: QuantStreamPaths, variation_id: int) -> None:
    references = active_scaler_references(paths)
    for scaler_path in paths.variation_scalers_dir(variation_id).glob("*.pkl"):
        if scaler_path.name not in references:
            scaler_path.unlink()


def process_run_request(paths: QuantStreamPaths, request_path: Path) -> RequestOutcome:
    try:
        payload = read_yaml_file(request_path)
        validate_request(payload)
        model_id = compute_model_id(payload)
        force_run = bool(payload.get("force_run", False))
        model_dir = paths.variation_models_dir(int(payload["variation_id"])) / model_id
        if not force_run and model_dir.exists():
            return reject_request(paths, request_path, f"Duplicate active model_id: {model_id}")
        if not force_run and has_tombstone(paths, payload):
            return reject_request(paths, request_path, "Request matches a deleted tombstone configuration.")
        variation_id = int(payload["variation_id"])
        clean_frame = load_clean_frame(paths, variation_id)
        _train_frame, test_frame = split_train_test(clean_frame)
        results = read_global_results(paths, variation_id, test_frame)
        if str(payload["model_type"]).lower() == "ensemble":
            predictions, probabilities = build_ensemble_predictions(results, payload["hyperparameters"])
            training_payload: dict[str, Any] = {"history": [], "backend": "ensemble"}
        else:
            model_dir = paths.variation_models_dir(variation_id) / model_id
            training_payload = train_model(
                model_type=str(payload["model_type"]).lower(),
                train_mode=str(payload["train_mode"]).lower(),
                clean_frame=clean_frame,
                hyperparameters=payload["hyperparameters"],
                output_dir=model_dir,
            )
            predictions = training_payload["predictions"]
            probabilities = training_payload["probabilities"]
        results[f"{model_id}_pred"] = predictions
        results[f"{model_id}_prob"] = probabilities
        paths.global_results_path(variation_id).parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(paths.global_results_path(variation_id), index=False)
        enriched_payload = {
            **payload,
            "training_backend": training_payload.get("backend", ""),
            "training_device": training_payload.get("device", ""),
            "scaler_path": training_payload.get("scaler_path", ""),
            "training_history": training_payload.get("history", []),
        }
        write_model_artifacts(paths, variation_id, model_id, enriched_payload, predictions, probabilities)
        done_payload = {**payload, "model_id": model_id, "status": "completed", "completed_at": utc_now_iso()}
        done_path = paths.runs_done_dir / f"{model_id}.yaml"
        write_yaml_file(done_path, done_payload)
        request_path.unlink()
        return RequestOutcome("completed", str(request_path), str(done_path), model_id=model_id)
    except Exception as exc:
        return reject_request(paths, request_path, str(exc))


def process_delete_request(paths: QuantStreamPaths, request_path: Path) -> RequestOutcome:
    try:
        payload = read_yaml_file(request_path)
        model_id = str(payload.get("model_id") or compute_model_id(payload))
        variation_id = int(payload["variation_id"])
        model_dir = paths.variation_models_dir(variation_id) / model_id
        hyperparameters_path = model_dir / "hyperparameters.json"
        if hyperparameters_path.exists():
            stored = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
            stored_hyperparameters = canonicalize_payload(stored.get("hyperparameters", {}))
            request_hyperparameters = canonicalize_payload(payload.get("hyperparameters", {}))
            if stored_hyperparameters != request_hyperparameters:
                raise ValueError("Delete request hyperparameters do not match stored model metadata.")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        results_path = paths.global_results_path(variation_id)
        if results_path.exists():
            results = pd.read_parquet(results_path)
            drop_columns = [column for column in results.columns if column in {f"{model_id}_pred", f"{model_id}_prob"}]
            if drop_columns:
                results.drop(columns=drop_columns).to_parquet(results_path, index=False)
        purge_unreferenced_scalers(paths, variation_id)
        tombstone = {**payload, "model_id": model_id, "deleted_at": utc_now_iso(), "status": "deleted"}
        tombstone_path = paths.deleted_runs_dir / f"{model_id}.yaml"
        write_yaml_file(tombstone_path, tombstone)
        request_path.unlink()
        return RequestOutcome("deleted", str(request_path), str(tombstone_path), model_id=model_id)
    except Exception as exc:
        return reject_request(paths, request_path, str(exc))


def run_once(root: Path | None = None) -> list[RequestOutcome]:
    paths = QuantStreamPaths(root=root or QuantStreamPaths().root)
    ensure_quant_stream_layout(paths)
    outcomes: list[RequestOutcome] = []
    for request_path in sorted(paths.run_requests_dir.glob("*.y*ml")):
        outcomes.append(process_run_request(paths, request_path))
    for request_path in sorted(paths.delete_requests_dir.glob("*.y*ml")):
        outcomes.append(process_delete_request(paths, request_path))
    update_readme(paths.root)
    return outcomes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant-Stream folder-state automation runner.")
    parser.add_argument("--once", action="store_true", help="Process pending requests once and exit.")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else None
    while True:
        outcomes = run_once(root=root)
        print(json.dumps([outcome.__dict__ for outcome in outcomes], indent=2), flush=True)
        if args.once:
            return 0
        time.sleep(float(args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
