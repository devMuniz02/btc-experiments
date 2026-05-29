from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._experiment_eval_reuse import (
    build_fixed_bundle,
    build_prediction_stream_from_saved_model,
    concat_prediction_streams,
    locate_decision_indices_for_timestamps,
    make_subset_test_bundle,
)
from src.utils.experiment_support import (
    EXPERIMENTS_V2_ROOT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    THRESHOLDS,
    TIME_VARIANTS,
    DirectCombination,
    build_parallel_eval_rows,
    dedupe_rows_by_name,
    direct_target_combinations_from_rows,
    load_summary,
    load_summary_rows,
    normalize_window_length,
    is_target_completed,
    time_variant_predicate,
    target_row_saved_model_dir,
    V2TargetRow,
    target_rows_from_rows,
    write_combinations_markdown,
    write_manifest_and_summary,
    write_window_train_manifests_from_rows,
)
from scripts.run_experiment_train_once import (
    resolve_device,
    run_bc,
    run_dagger,
    run_lstm,
    run_mamba,
    run_nn,
    run_ppo,
    run_rf,
    run_transformer,
    run_xgboost,
)
from src.btc_direction_learning.config import FIXED_REGIME_NAME, FIXED_TEST_POOL_HOURS
from src.btc_direction_learning.dataset import (
    DirectionDatasetBundle,
    build_direction_dataset_bundle,
    build_direction_dataset_bundle_from_processed_with_feature_columns,
    ensure_shared_data_dir,
)
from src.btc_direction_learning.env import BTCDirectionEnv, ENV_VERSION_BINARY
from src.btc_direction_learning.evaluation import evaluate_policy, load_checkpoint
from src.btc_direction_learning.models import LSTMClassificationPolicy
from src.utils.market_data import set_seed


TARGET_FAMILY = "LSTM"
TARGET_TRAIN_LENGTH = "5K"
TARGET_WINDOW_LENGTH = "1K"
TARGET_MODEL_VARIATION = "WINDOW_RETRAIN"
TARGET_SOURCE_NAME = "LSTM_5K"
TEST_POOL_HOURS = FIXED_TEST_POOL_HOURS
WINDOW_HOURS = 1000
NUM_WINDOWS = 4
DEFAULT_RETRAIN_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 64
SUPPORTED_WINDOW_RETRAIN_FAMILIES = ("BC", "DAGGER", "NN", "RF", "XGBOOST", "LSTM", "TRANSFORMER", "MAMBA", "PPO")
SUPPORTED_WINDOW_MODEL_VARIATIONS = ("WINDOW_RETRAIN", "WINDOW_CONTINUE")
FAMILY_RUNNERS = {
    "BC": run_bc,
    "DAGGER": run_dagger,
    "NN": run_nn,
    "RF": run_rf,
    "XGBOOST": run_xgboost,
    "LSTM": run_lstm,
    "TRANSFORMER": run_transformer,
    "MAMBA": run_mamba,
    "PPO": run_ppo,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train canonical EXPERIMENTSV2 LSTM 5K 1K WINDOW_RETRAIN target natively.")
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / str(EXPERIMENT_ID)))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--raw-fetch-hours", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--window-hours", type=int, default=WINDOW_HOURS)
    parser.add_argument("--num-windows", type=int, default=NUM_WINDOWS)
    parser.add_argument("--retrain-epochs", type=int, default=DEFAULT_RETRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--target-model-variation", default=TARGET_MODEL_VARIATION)
    parser.add_argument("--mamba-epochs", type=int, default=0)
    parser.add_argument("--mamba-batch-size", type=int, default=0)
    parser.add_argument("--mamba-disable-early-stopping", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-max-epochs", type=int, default=5000)
    parser.add_argument("--visualize-test-acc", action="store_true")
    parser.add_argument("--time-variants", nargs="+", default=list(TIME_VARIANTS))
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def _log(args: argparse.Namespace, level: int, message: str) -> None:
    if int(getattr(args, "verbose", 1)) >= level:
        print(message, flush=True)


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _target_family(args: argparse.Namespace) -> str:
    family = str(getattr(args, "target_family", TARGET_FAMILY) or TARGET_FAMILY).strip().upper()
    if family not in SUPPORTED_WINDOW_RETRAIN_FAMILIES:
        raise ValueError(f"Unsupported WINDOW_RETRAIN family: {family}")
    return family


def _target_model_variation(args: argparse.Namespace) -> str:
    variation = str(getattr(args, "target_model_variation", TARGET_MODEL_VARIATION) or TARGET_MODEL_VARIATION).strip().upper()
    if variation not in SUPPORTED_WINDOW_MODEL_VARIATIONS:
        raise ValueError(f"Unsupported window model variation: {variation}")
    return variation


def _target_train_length(args: argparse.Namespace) -> str:
    return str(getattr(args, "target_train_length", TARGET_TRAIN_LENGTH) or TARGET_TRAIN_LENGTH).strip().upper()


def _target_source_name(args: argparse.Namespace) -> str:
    return str(getattr(args, "target_source_name", f"{_target_family(args)}_{_target_train_length(args)}") or "").strip().upper()


def _find_v2_base_row(output_root: Path) -> dict[str, Any]:
    raise RuntimeError("_find_v2_base_row requires args; use _find_v2_base_row_for_args instead.")


def _find_v2_base_row_for_args(output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = load_summary_rows(output_root)
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("family") or "").upper() == _target_family(args)
        and str(row.get("train_length") or "").upper() == _target_train_length(args)
        and not str(row.get("window_length") or "").strip()
        and str(row.get("model_variation") or "").upper() == "BASE"
        and str(row.get("time_variant") or "").upper() == "FULL"
        and int(row.get("threshold_pct") or 0) == 50
        and str(row.get("saved_model_path") or "").strip()
    ]
    if not candidates:
        raise RuntimeError(
            f"V2-native window training requires the canonical {_target_family(args)}_{_target_train_length(args)} FULL BASE 50 row in EXPERIMENTSV2."
        )
    return dict(candidates[0])


def _target_window_length(args: argparse.Namespace) -> str:
    explicit = normalize_window_length(getattr(args, "target_window_length", ""))
    if explicit:
        return explicit
    return normalize_window_length(getattr(args, "window_hours", TARGET_WINDOW_LENGTH))


def _window_length_token(args: argparse.Namespace) -> str:
    return _target_window_length(args).lower()


def _default_num_windows(window_hours: int) -> int:
    if int(window_hours) <= 0:
        raise ValueError("window_hours must be positive.")
    return max(0, (int(TEST_POOL_HOURS) // int(window_hours)) - 1)


def _base_train_end_date(base_row: dict[str, Any]) -> str:
    timestamps = ((base_row.get("train") or {}).get("timestamps") or [])
    return str(timestamps[-1]) if isinstance(timestamps, list) and timestamps else ""


def _base_train_rows(base_row: dict[str, Any]) -> int:
    return int(base_row.get("actual_train_rows") or base_row.get("requested_train_rows") or 5000)


def _load_torch_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Bad artifact format: {path}")
    return payload


def _save_result_artifact_with_name(path: Path, raw_result: dict[str, Any], source_name: str) -> str:
    artifact = raw_result.pop("__model_artifact__", None)
    if not isinstance(artifact, dict):
        raise RuntimeError("Training result is missing __model_artifact__.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.get("payload")
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["model_name"] = str(source_name)
        if artifact["kind"] == "torch":
            payload["series_name"] = path.stem
    if artifact["kind"] == "torch":
        torch.save(payload, path)
    elif artifact["kind"] == "pickle":
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
    else:
        raise ValueError(f"Unsupported model artifact kind: {artifact['kind']}")
    return str(path.resolve())


def _build_window_bundle(full_bundle: DirectionDatasetBundle, window_idx: int, window_hours: int) -> tuple[DirectionDatasetBundle, DirectionDatasetBundle]:
    orig_train_count = full_bundle.train_row_count
    accumulated_test = window_idx * window_hours
    orig_train_keep = orig_train_count - accumulated_test
    train_region_start = orig_train_count - orig_train_keep
    train_region_end = orig_train_count + accumulated_test
    eval_start_row = orig_train_count + accumulated_test
    eval_end_row = eval_start_row + window_hours
    window_frame = full_bundle.processed_df.iloc[train_region_start:eval_end_row].reset_index(drop=True)
    refit_bundle = build_direction_dataset_bundle_from_processed_with_feature_columns(
        window_frame,
        train_hours=int(train_region_end - train_region_start),
        test_hours=int(window_hours),
        feature_columns=list(full_bundle.feature_columns),
    )
    return refit_bundle, refit_bundle


def _timestamps_for_indices(bundle: DirectionDatasetBundle, indices: np.ndarray) -> list[str]:
    timestamps = pd.to_datetime(bundle.processed_df.iloc[indices]["timestamp"], utc=True)
    return [timestamp.isoformat() for timestamp in timestamps.tolist()]


def _canonical_group_dir(output_root: Path, args: argparse.Namespace) -> Path:
    target = V2TargetRow(
        family=_target_family(args),
        train_length=_target_train_length(args),
        window_length=_target_window_length(args),
        model_variation=_target_model_variation(args),
        time_variant="FULL",
    )
    group_dir = target_row_saved_model_dir(output_root, target)
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def _variant_slug(time_variant: str) -> str:
    normalized = str(time_variant).strip().upper()
    return "" if normalized == "FULL" else f"_{normalized.lower()}"


def _variant_group_dir(output_root: Path, time_variant: str, args: argparse.Namespace) -> Path:
    base_dir = _canonical_group_dir(output_root, args)
    slug = _variant_slug(time_variant)
    group_dir = base_dir if not slug else base_dir.parent / f"{base_dir.name}{slug}"
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def _canonical_paths(output_root: Path, num_windows: int, time_variant: str, args: argparse.Namespace, artifact_suffix: str) -> dict[str, Path]:
    group_dir = _variant_group_dir(output_root, time_variant, args)
    slug = _variant_slug(time_variant)
    window_token = _window_length_token(args)
    source_stub = _target_source_name(args).lower()
    variation_token = _target_model_variation(args).lower()
    extension = artifact_suffix if str(artifact_suffix).startswith(".") else f".{artifact_suffix}"
    result: dict[str, Path] = {
        "group_dir": group_dir,
        "source_copy": group_dir / f"{source_stub}{slug}{extension}",
        "base": group_dir / f"{source_stub}_{window_token}_{variation_token}_base{slug}{extension}",
        "final": group_dir / f"{source_stub}_{window_token}_{variation_token}{slug}{extension}",
    }
    for index in range(1, num_windows + 1):
        result[f"w{index}"] = group_dir / f"{source_stub}_{window_token}_{variation_token}_w{index}{slug}{extension}"
    return result


def _copy_if_needed(source: str | Path, destination: Path) -> str:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source_path, destination)
    return str(destination.resolve())


def _existing_window_count(paths: dict[str, Path], num_windows: int) -> int:
    count = 0
    for index in range(1, num_windows + 1):
        if paths[f"w{index}"].exists():
            count = index
        else:
            break
    return count


def _window_checkpoint_metadata(
    full_bundle: DirectionDatasetBundle,
    paths: dict[str, Path],
    num_windows: int,
    window_hours: int,
    time_variant: str,
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for index in range(1, num_windows + 1):
        path = paths[f"w{index}"]
        if not path.exists():
            break
        train_bundle, eval_bundle = _build_window_bundle(full_bundle, index, window_hours)
        train_bundle = _bundle_for_time_variant(train_bundle, time_variant)
        eval_bundle = _bundle_for_time_variant(eval_bundle, time_variant)
        train_timestamps = _timestamps_for_indices(train_bundle, train_bundle.train_decision_indices)
        eval_timestamps = _timestamps_for_indices(eval_bundle, eval_bundle.test_decision_indices)
        checkpoints.append(
            {
                "window_index": index,
                "window_name": f"W{index}",
                "normalized_window_name": f"W{index}",
                "saved_model_path": str(path.resolve()),
                "window_train_end": train_timestamps[-1] if train_timestamps else "",
                "test_timestamps": eval_timestamps,
                "is_base_window": False,
            }
        )
    return checkpoints


def _build_window_prediction_stream(
    base_saved_model_path: str,
    checkpoints: list[dict[str, Any]],
    train_rows: int,
    window_hours: int,
    time_variant: str,
) -> dict[str, Any]:
    full_bundle = build_fixed_bundle(data_variant="full", train_rows=int(train_rows), test_rows=int(TEST_POOL_HOURS))
    base_timestamps = _timestamps_for_indices(full_bundle, full_bundle.test_decision_indices[:window_hours])
    stream_parts = []
    base_indices = locate_decision_indices_for_timestamps(full_bundle, base_timestamps)
    base_bundle = make_subset_test_bundle(full_bundle, base_indices)
    stream_parts.append(
        build_prediction_stream_from_saved_model(
            saved_path=base_saved_model_path,
            eval_bundle=base_bundle,
            device=torch.device("cpu"),
            timestamps_override=base_timestamps,
        )
    )
    for checkpoint in checkpoints:
        test_timestamps = [str(value) for value in checkpoint.get("test_timestamps", [])]
        window_index = int(checkpoint.get("window_index") or 0)
        checkpoint_bundle, eval_bundle = _build_window_bundle(full_bundle, window_index, int(window_hours))
        del checkpoint_bundle
        checkpoint_bundle = _bundle_for_time_variant(eval_bundle, time_variant)
        stream_parts.append(
            build_prediction_stream_from_saved_model(
                saved_path=str(checkpoint["saved_model_path"]),
                eval_bundle=checkpoint_bundle,
                device=torch.device("cpu"),
                timestamps_override=test_timestamps,
            )
        )
    return concat_prediction_streams(stream_parts)


def _normalize_selected_time_variants(args: argparse.Namespace) -> list[str]:
    selected = [str(value).strip().upper() for value in getattr(args, "time_variants", TIME_VARIANTS) if str(value).strip()]
    return selected or list(TIME_VARIANTS)


def _filter_bundle_indices(bundle: DirectionDatasetBundle, indices: np.ndarray, predicate) -> np.ndarray:
    timestamps = pd.to_datetime(bundle.processed_df.iloc[indices]["timestamp"], utc=True)
    filtered = [int(index) for index, timestamp in zip(indices.tolist(), timestamps.tolist()) if predicate(timestamp)]
    return np.asarray(filtered, dtype=np.int64)


def _bundle_for_time_variant(bundle: DirectionDatasetBundle, time_variant: str) -> DirectionDatasetBundle:
    normalized = str(time_variant).strip().upper()
    if normalized == "FULL":
        return bundle
    predicate = time_variant_predicate(normalized)
    if predicate is None:
        return bundle
    filtered_train = _filter_bundle_indices(bundle, bundle.train_decision_indices, predicate)
    filtered_test = _filter_bundle_indices(bundle, bundle.test_decision_indices, predicate)
    return DirectionDatasetBundle(
        processed_df=bundle.processed_df,
        scaled_features=bundle.scaled_features,
        scaler=bundle.scaler,
        feature_columns=bundle.feature_columns,
        sequence_length=bundle.sequence_length,
        labels=bundle.labels,
        train_decision_indices=filtered_train,
        test_decision_indices=filtered_test,
        train_row_count=int(len(filtered_train)),
        test_row_count=int(len(filtered_test)),
    )


def _target_for_time_variant(time_variant: str, args: argparse.Namespace) -> V2TargetRow:
    return V2TargetRow(
        family=_target_family(args),
        train_length=_target_train_length(args),
        window_length=_target_window_length(args),
        model_variation=_target_model_variation(args),
        time_variant=str(time_variant).strip().upper(),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    previous_summary = load_summary(output_root)
    previous_rows = load_summary_rows(output_root)
    selected_time_variants = _normalize_selected_time_variants(args)
    base_row = _find_v2_base_row_for_args(output_root, args)
    base_saved_model_path = str(base_row.get("saved_model_path") or "").strip()
    base_train_rows = _base_train_rows(base_row)
    base_train_end_date = _base_train_end_date(base_row)
    source_name = _target_source_name(args)
    family = _target_family(args)
    target_model_variation = _target_model_variation(args)
    if target_model_variation == "WINDOW_CONTINUE" and family != "MAMBA":
        raise RuntimeError("WINDOW_CONTINUE is currently implemented only for MAMBA in the V2 native window trainer.")
    device = resolve_device(args.device)
    set_seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    shared_data_dir = ensure_shared_data_dir(PROJECT_ROOT / "artifacts" / "btc" / "direction_learning" / "_shared")
    full_bundle = build_direction_dataset_bundle(
        force_refresh=bool(args.force_refresh),
        train_hours=base_train_rows,
        test_hours=TEST_POOL_HOURS,
        raw_fetch_hours=int(args.raw_fetch_hours) or None,
        split_regime=FIXED_REGIME_NAME,
        data_variant="full",
        data_dir=shared_data_dir,
    )
    configured_num_windows = int(getattr(args, "num_windows", 0) or 0)
    if configured_num_windows <= 0:
        configured_num_windows = _default_num_windows(int(args.window_hours))
    artifact_suffix = Path(base_saved_model_path).suffix or ".pt"
    replacement_rows: list[dict[str, Any]] = []
    resume_by_variant: dict[str, int] = {}
    _log(args, 1, "[V2 train] freeing memory before train run")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for time_variant in selected_time_variants:
        target = _target_for_time_variant(time_variant, args)
        if not bool(args.overwrite) and is_target_completed(previous_rows, target):
            _log(args, 2, f"[V2 train] skipping completed {time_variant} window retrain target")
            continue

        paths = _canonical_paths(output_root, configured_num_windows, time_variant, args, artifact_suffix)
        _copy_if_needed(base_saved_model_path, paths["source_copy"])
        if bool(args.overwrite):
            for key, path in paths.items():
                if key == "group_dir":
                    continue
                if path.exists() and path.is_file():
                    path.unlink()
        if not paths["base"].exists():
            _copy_if_needed(base_saved_model_path, paths["base"])

        completed_windows = 0 if bool(args.overwrite) else _existing_window_count(paths, configured_num_windows)
        resume_by_variant[time_variant] = completed_windows
        if completed_windows > 0:
            _log(args, 2, f"[V2 train] resuming {time_variant} window retrain from W{completed_windows + 1}")

        completed_window_durations: list[float] = []
        for window_idx in range(completed_windows + 1, configured_num_windows + 1):
            _log(args, 1, f"[V2 train] {time_variant} W{window_idx} retrain/eval/save")
            train_bundle, eval_bundle = _build_window_bundle(full_bundle, window_idx, int(args.window_hours))
            train_bundle = _bundle_for_time_variant(train_bundle, time_variant)
            eval_bundle = _bundle_for_time_variant(eval_bundle, time_variant)
            train_timestamps = _timestamps_for_indices(train_bundle, train_bundle.train_decision_indices)
            eval_timestamps = _timestamps_for_indices(eval_bundle, eval_bundle.test_decision_indices)
            runner = FAMILY_RUNNERS[family]
            runner_kwargs: dict[str, Any] = {
                "bundle": train_bundle,
                "device": device,
                "env_version": "ternary",
                "threshold": 0.0,
                "train_timestamps": train_timestamps,
                "test_timestamps": eval_timestamps,
                "force_none_outside_market_hours": False,
            }
            window_start_time = time.perf_counter()
            if family == "PPO":
                runner_kwargs["ppo_total_updates"] = 5
                runner_kwargs["ppo_trajectories_per_update"] = 128
            if family == "MAMBA":
                if target_model_variation == "WINDOW_CONTINUE":
                    continue_source_path = paths["base"] if window_idx == 1 else paths[f"w{window_idx - 1}"]
                    continue_payload = _load_torch_artifact(continue_source_path)
                    runner_kwargs["initial_state_dict"] = continue_payload.get("state_dict")
                resolved_mamba_epochs = int(getattr(args, "mamba_epochs", 0) or 0)
                runner_kwargs["mamba_epochs"] = resolved_mamba_epochs if resolved_mamba_epochs > 0 else None
                mamba_batch_size = int(getattr(args, "mamba_batch_size", 0) or 0)
                if mamba_batch_size > 0:
                    runner_kwargs["batch_size_override"] = mamba_batch_size
                runner_kwargs["disable_early_stopping"] = bool(getattr(args, "mamba_disable_early_stopping", False))
                runner_kwargs["overfit"] = bool(getattr(args, "overfit", False))
                runner_kwargs["overfit_max_epochs"] = int(getattr(args, "overfit_max_epochs", 5000))
                total_windows = configured_num_windows
                next_window_test_env = BTCDirectionEnv(eval_bundle, "test", env_version="ternary")
                visualize_test_acc = bool(getattr(args, "visualize_test_acc", False))

                def _mamba_progress_callback(epoch_metrics: dict[str, Any], epoch: int, total_epochs: int) -> None:
                    if epoch <= 0:
                        label = str(epoch_metrics.get("label") or "").strip()
                        if label:
                            _log(args, 1, f"[V2 train] {time_variant} W{window_idx} {label}")
                        return
                    elapsed_window = max(0.0, time.perf_counter() - window_start_time)
                    avg_epoch_seconds = elapsed_window / max(1, epoch)
                    window_eta_seconds = avg_epoch_seconds * max(0, total_epochs - epoch)
                    projected_window_total = avg_epoch_seconds * max(1, total_epochs)
                    if completed_window_durations:
                        average_window_seconds = sum(completed_window_durations) / len(completed_window_durations)
                    else:
                        average_window_seconds = projected_window_total
                    remaining_full_windows = max(0, total_windows - window_idx)
                    all_windows_eta_seconds = window_eta_seconds + (average_window_seconds * remaining_full_windows)
                    loss_value = float(epoch_metrics.get("loss") or 0.0)
                    train_accuracy = float(epoch_metrics.get("train_accuracy") or 0.0)
                    val_loss = epoch_metrics.get("val_loss")
                    val_fragment = ""
                    if val_loss is not None:
                        val_fragment = f" val_loss={float(val_loss):.4f}"
                    test_fragment = ""
                    if visualize_test_acc:
                        policy = epoch_metrics.get("_policy")
                        if isinstance(policy, nn.Module):
                            next_window_result = evaluate_policy(next_window_test_env, policy, device, 0.0)
                            test_fragment = f" next_test_acc={float(next_window_result.get('accuracy') or 0.0):.4f}"
                    _log(
                        args,
                        1,
                        (
                            f"[V2 train] {time_variant} W{window_idx}/{total_windows} "
                            f"epoch {epoch}/{total_epochs} loss={loss_value:.4f} acc={train_accuracy:.4f}"
                            f"{test_fragment}{val_fragment} eta_window={_format_eta(window_eta_seconds)} "
                            f"eta_all={_format_eta(all_windows_eta_seconds)}"
                        ),
                    )

                runner_kwargs["progress_callback"] = _mamba_progress_callback
            raw_result = runner(
                **runner_kwargs,
            )
            completed_window_durations.append(time.perf_counter() - window_start_time)
            _log(
                args,
                2,
                f"[V2 train] {time_variant} W{window_idx} retrain train={float(((raw_result.get('train') or {}).get('accuracy')) or 0.0):.4f} "
                f"eval={float(((raw_result.get('test') or {}).get('accuracy')) or 0.0):.4f}",
            )
            _save_result_artifact_with_name(paths[f"w{window_idx}"], raw_result, source_name)

        final_source = paths["final"] if paths["final"].exists() else paths.get(f"w{configured_num_windows}")
        if final_source is None or not Path(final_source).exists():
            raise RuntimeError("WINDOW_RETRAIN did not produce a final checkpoint artifact.")
        if Path(final_source).resolve() != paths["final"].resolve():
            shutil.copy2(final_source, paths["final"])

        checkpoint_metadata = _window_checkpoint_metadata(full_bundle, paths, configured_num_windows, int(args.window_hours), time_variant)
        grouped_saved_model_paths = [str(paths["base"].resolve()), *[str(Path(item["saved_model_path"]).resolve()) for item in checkpoint_metadata]]
        prediction_stream = _build_window_prediction_stream(
            base_saved_model_path=str(paths["base"].resolve()),
            checkpoints=checkpoint_metadata,
            train_rows=base_train_rows,
            window_hours=int(args.window_hours),
            time_variant=time_variant,
        )
        variant_rows = build_parallel_eval_rows(
            output_root=output_root,
            combination=DirectCombination(family=_target_family(args), train_length=_target_train_length(args)),
            saved_model_path=str(paths["final"].resolve()),
            prediction_stream=prediction_stream,
            source_metadata={
                "source_experiment_id": EXPERIMENT_ID,
                "source_experiment_name": EXPERIMENT_NAME,
                "source_variant": "full",
                "window_checkpoints": checkpoint_metadata,
                "base_train_end_date": base_train_end_date,
            },
            train_payload={"timestamps": list((base_row.get("train") or {}).get("timestamps", []))},
            time_variants=[time_variant],
            thresholds=list(THRESHOLDS),
            model_variation=target_model_variation,
            window_length=_target_window_length(args),
            artifact_bucket="windowed",
            source_model=source_name,
            env_version="ternary",
        )
        group_dir = str(paths["group_dir"].resolve())
        for row in variant_rows:
            row["saved_model_path"] = str(paths["final"].resolve())
            row["source_saved_model_path"] = str(paths["final"].resolve())
            row["grouped_saved_model_dir"] = group_dir
            row["grouped_saved_model_paths"] = list(grouped_saved_model_paths)
            row["window_checkpoints"] = json.loads(json.dumps(checkpoint_metadata))
            row["base_train_end_date"] = base_train_end_date
            row["source_experiment_id"] = EXPERIMENT_ID
            row["source_experiment_name"] = EXPERIMENT_NAME
            row["source_variant"] = "full"
            row["source_model"] = source_name
        replacement_rows.extend(variant_rows)

    replacement_names = {str(row.get("name") or "") for row in replacement_rows if row.get("name")}
    preserved_rows = [row for row in previous_rows if str(row.get("name") or "") not in replacement_names]
    merged_rows = dedupe_rows_by_name(preserved_rows + replacement_rows)
    summary = dict(previous_summary)
    _log(args, 1, "[V2 train] freeing memory after train run")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log(args, 1, f"[V2 train] merging {len(replacement_rows)} retrained rows into EXPERIMENTSV2 summary")
    summary["experiment_id"] = EXPERIMENT_ID
    summary["name"] = EXPERIMENT_NAME
    summary["series_order"] = [row["name"] for row in merged_rows]
    summary["models"] = merged_rows
    summary["direct_target_combinations"] = direct_target_combinations_from_rows(merged_rows)
    summary["target_rows"] = target_rows_from_rows(merged_rows)
    _log(args, 1, "[V2 train] writing summary, manifest, combinations, and window manifests")
    summary_path, manifest_path = write_manifest_and_summary(output_root, summary)
    combinations_path = write_combinations_markdown(output_root, summary, merged_rows)
    write_window_train_manifests_from_rows(merged_rows)
    _log(
        args,
        1,
        (
            f"[V2 train] native window retrain finished: rows={len(merged_rows)} "
            f"summary={summary_path.resolve()} manifest={manifest_path.resolve()}"
        ),
    )
    return {
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combinations_path": str(combinations_path.resolve()),
        "retrained_row_names": [str(row.get("name")) for row in replacement_rows if row.get("name")],
        "row_count": len(merged_rows),
        "native_backend": "window_retrain_v2",
        "resumed_from_window": resume_by_variant,
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
