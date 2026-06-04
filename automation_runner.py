from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.ensemble import train_stacked_logistic_ensemble
from src.models.backtest import BacktestConfig, run_binary_backtest
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
    "logistic_regression",
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
    worker_pid: int = 0
    rss_mb_before_cleanup: float | None = None
    rss_mb_after_cleanup: float | None = None


def current_rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def trim_windows_working_set() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        psapi.EmptyWorkingSet(handle)
    except Exception:
        pass


def release_runtime_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    trim_windows_working_set()


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
        ensemble_mode = str(hyperparameters.get("ensemble_mode", "stacked_logistic")).strip().lower()
        if ensemble_mode != "stacked_logistic":
            raise ValueError("automation ensemble requests only support ensemble_mode: stacked_logistic.")
        if not isinstance(models_pool, list) or not models_pool:
            raise ValueError("ensemble hyperparameters.models_pool must be a non-empty list.")
        if len(models_pool) < 2:
            raise ValueError("ensemble hyperparameters.models_pool must include at least two trained model ids.")
        if train_mode != "static_baseline":
            raise ValueError("stacked_logistic ensembles are only supported with train_mode: static_baseline.")


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
    metrics: dict[str, Any] | None = None,
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
            "signal_count": metrics.get("signal_count") if metrics else int(len(predictions)),
            "mean_probability": float(np.mean(probabilities)) if len(probabilities) else 0.0,
            "accuracy": metrics.get("accuracy") if metrics else None,
            "win_rate": metrics.get("win_rate") if metrics else None,
            "net_pnl": metrics.get("net_pnl") if metrics else None,
            "max_drawdown": metrics.get("max_drawdown") if metrics else None,
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
            model_dir = paths.variation_models_dir(variation_id) / model_id
            predictions, probabilities, training_payload = train_stacked_logistic_ensemble(
                clean_frame=clean_frame,
                results=results,
                models_root=paths.models_dev_dir,
                variation_id=variation_id,
                hyperparameters=payload["hyperparameters"],
                output_dir=model_dir,
            )
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
        expected_rows = len(test_frame)
        evaluated_rows = len(predictions)
        if evaluated_rows < expected_rows and str(payload.get("trial_source", "")).lower() == "optuna":
            predictions = np.pad(predictions, (0, expected_rows - evaluated_rows), constant_values=0)
            probabilities = np.pad(probabilities, (0, expected_rows - evaluated_rows), constant_values=0.0)
        if len(predictions) != expected_rows or len(probabilities) != expected_rows:
            raise ValueError(
                f"Model returned {len(predictions)} predictions and {len(probabilities)} probabilities; "
                f"expected {expected_rows} test rows."
            )
        results[f"{model_id}_pred"] = predictions
        results[f"{model_id}_prob"] = probabilities
        backtest = run_binary_backtest(
            results,
            model_id,
            config=BacktestConfig(confidence_threshold=0.0, bet_fraction=0.02, use_cuda="cpu"),
        )
        paths.global_results_path(variation_id).parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(paths.global_results_path(variation_id), index=False)
        enriched_payload = {
            **payload,
            "training_backend": training_payload.get("backend", ""),
            "training_device": training_payload.get("device", ""),
            "scaler_path": training_payload.get("scaler_path", ""),
            "training_history": training_payload.get("history", []),
            "resolved_train_length": training_payload.get("train_length", ""),
            "resolved_test_length": training_payload.get("test_length", evaluated_rows),
            "evaluated_rows": evaluated_rows,
        }
        write_model_artifacts(
            paths,
            variation_id,
            model_id,
            enriched_payload,
            predictions,
            probabilities,
            metrics=backtest.metrics,
        )
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
        done_path = paths.runs_done_dir / f"{model_id}.yaml"
        if done_path.exists():
            done_path.unlink()
        request_path.unlink()
        return RequestOutcome("deleted", str(request_path), str(tombstone_path), model_id=model_id)
    except Exception as exc:
        return reject_request(paths, request_path, str(exc))


def outcome_from_dict(payload: dict[str, Any]) -> RequestOutcome:
    return RequestOutcome(
        status=str(payload.get("status", "")),
        source_path=str(payload.get("source_path", "")),
        target_path=str(payload.get("target_path", "")),
        model_id=str(payload.get("model_id", "")),
        reason=str(payload.get("reason", "")),
        worker_pid=int(payload.get("worker_pid") or 0),
        rss_mb_before_cleanup=payload.get("rss_mb_before_cleanup"),
        rss_mb_after_cleanup=payload.get("rss_mb_after_cleanup"),
    )


def parse_worker_outcome(stdout: str) -> RequestOutcome | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "status" in payload and "source_path" in payload:
            return outcome_from_dict(payload)
    return None


def run_request_worker(kind: str, request_path: Path, root: Path | None = None) -> RequestOutcome:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--process-{kind}-request",
        str(request_path),
    ]
    if root is not None:
        command.extend(["--root", str(root)])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    outcome = parse_worker_outcome(completed.stdout)
    if outcome is not None:
        return outcome
    details = (completed.stderr or completed.stdout or "").strip().splitlines()
    reason = (
        "\n".join(details[-8:]) if details else f"Worker exited with code {completed.returncode} without JSON output."
    )
    return RequestOutcome(
        "worker_error",
        str(request_path),
        "",
        reason=reason,
        worker_pid=0,
    )


def finalize_worker_outcome(outcome: RequestOutcome, before_cleanup: float | None) -> RequestOutcome:
    release_runtime_memory()
    after_cleanup = current_rss_mb()
    return RequestOutcome(
        status=outcome.status,
        source_path=outcome.source_path,
        target_path=outcome.target_path,
        model_id=outcome.model_id,
        reason=outcome.reason,
        worker_pid=os.getpid(),
        rss_mb_before_cleanup=before_cleanup,
        rss_mb_after_cleanup=after_cleanup,
    )


def process_single_worker(kind: str, request_path: Path, root: Path | None = None) -> RequestOutcome:
    paths = QuantStreamPaths(root=root or QuantStreamPaths().root)
    ensure_quant_stream_layout(paths)
    before_cleanup: float | None = None
    try:
        if kind == "run":
            outcome = process_run_request(paths, request_path)
        elif kind == "delete":
            outcome = process_delete_request(paths, request_path)
        else:
            raise ValueError(f"Unsupported worker kind: {kind}")
        before_cleanup = current_rss_mb()
        return finalize_worker_outcome(outcome, before_cleanup)
    except Exception as exc:
        before_cleanup = current_rss_mb()
        outcome = RequestOutcome("worker_error", str(request_path), "", reason=str(exc))
        return finalize_worker_outcome(outcome, before_cleanup)


def run_once(root: Path | None = None) -> list[RequestOutcome]:
    paths = QuantStreamPaths(root=root or QuantStreamPaths().root)
    ensure_quant_stream_layout(paths)
    outcomes: list[RequestOutcome] = []
    for request_path in sorted(paths.run_requests_dir.glob("*.y*ml")):
        outcomes.append(run_request_worker("run", request_path, paths.root))
    for request_path in sorted(paths.delete_requests_dir.glob("*.y*ml")):
        outcomes.append(run_request_worker("delete", request_path, paths.root))
    update_readme(paths.root)
    release_runtime_memory()
    return outcomes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant-Stream folder-state automation runner.")
    parser.add_argument("--once", action="store_true", help="Process pending requests once and exit.")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--root", default="")
    parser.add_argument("--process-run-request", default="", help=argparse.SUPPRESS)
    parser.add_argument("--process-delete-request", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else None
    if args.process_run_request:
        outcome = process_single_worker("run", Path(args.process_run_request).resolve(), root)
        print(json.dumps(outcome.__dict__, separators=(",", ":")), flush=True)
        return 0 if outcome.status != "worker_error" else 1
    if args.process_delete_request:
        outcome = process_single_worker("delete", Path(args.process_delete_request).resolve(), root)
        print(json.dumps(outcome.__dict__, separators=(",", ":")), flush=True)
        return 0 if outcome.status != "worker_error" else 1
    while True:
        outcomes = run_once(root=root)
        print(json.dumps([outcome.__dict__ for outcome in outcomes], indent=2), flush=True)
        if args.once:
            return 0
        time.sleep(float(args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
