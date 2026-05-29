from __future__ import annotations

import argparse
import json
import shutil
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
from src.btc_direction_learning.algorithms import ppo_continue
from src.btc_direction_learning.config import FIXED_REGIME_NAME, FIXED_TEST_POOL_HOURS
from src.btc_direction_learning.continuation_models import LSTMContinuationActorCriticPolicy
from src.btc_direction_learning.dataset import (
    DirectionDatasetBundle,
    build_direction_dataset_bundle,
    build_direction_dataset_bundle_from_processed_with_feature_columns,
    ensure_shared_data_dir,
)
from src.btc_direction_learning.env import BTCDirectionEnv, ENV_VERSION_BINARY
from src.btc_direction_learning.evaluation import clone_state_dict, evaluate_policy, load_checkpoint
from src.btc_direction_learning.models import LSTMClassificationPolicy
from src.utils.market_data import set_seed


TARGET_FAMILY = "LSTM"
TARGET_TRAIN_LENGTH = "35K"
TARGET_WINDOW_LENGTH = "1K"
TARGET_MODEL_VARIATION = "PPO_WINDOW_CONTINUE"
TARGET_SOURCE_NAME = "LSTM_35K"
TEST_POOL_HOURS = FIXED_TEST_POOL_HOURS
WINDOW_HOURS = 1000
NUM_WINDOWS = 4
DEFAULT_PPO_TOTAL_UPDATES = 3
DEFAULT_PPO_TRAJECTORIES_PER_UPDATE = 32
DEFAULT_PPO_EPOCHS = 2
DEFAULT_PPO_MINIBATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train canonical EXPERIMENTSV2 window-model targets natively into the V2 output tree.")
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / str(EXPERIMENT_ID)))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--raw-fetch-hours", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--window-hours", type=int, default=WINDOW_HOURS)
    parser.add_argument("--num-windows", type=int, default=NUM_WINDOWS)
    parser.add_argument("--ppo-total-updates", type=int, default=DEFAULT_PPO_TOTAL_UPDATES)
    parser.add_argument("--ppo-trajectories-per-update", type=int, default=DEFAULT_PPO_TRAJECTORIES_PER_UPDATE)
    parser.add_argument("--ppo-learning-rate", type=float, default=2e-4)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.1)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-value-loss-coef", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=DEFAULT_PPO_EPOCHS)
    parser.add_argument("--ppo-minibatch-size", type=int, default=DEFAULT_PPO_MINIBATCH_SIZE)
    parser.add_argument("--previous-policy-kl-coef", type=float, default=0.1)
    parser.add_argument("--continual-lr-decay", type=float, default=1.0)
    parser.add_argument("--continual-clip-decay", type=float, default=1.0)
    parser.add_argument("--continual-entropy-decay", type=float, default=1.0)
    parser.add_argument("--time-variants", nargs="+", default=list(TIME_VARIANTS))
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def _log(args: argparse.Namespace, level: int, message: str) -> None:
    if int(getattr(args, "verbose", 1)) >= level:
        print(message, flush=True)


def _target_family(args: argparse.Namespace) -> str:
    return str(getattr(args, "target_family", TARGET_FAMILY) or TARGET_FAMILY).strip().upper()


def _target_train_length(args: argparse.Namespace) -> str:
    return str(getattr(args, "target_train_length", TARGET_TRAIN_LENGTH) or TARGET_TRAIN_LENGTH).strip().upper()


def _target_source_name(args: argparse.Namespace) -> str:
    return str(getattr(args, "target_source_name", f"{_target_family(args)}_{_target_train_length(args)}") or "").strip().upper()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return int(base_row.get("actual_train_rows") or base_row.get("requested_train_rows") or 35000)


def _load_torch_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Bad artifact format: {path}")
    return payload


def _instantiate_base_policy(payload: dict[str, Any]) -> nn.Module:
    hidden_dim = int(
        payload.get("hidden_dim")
        or payload["state_dict"]["input_proj.weight"].shape[0]
    )
    policy = LSTMClassificationPolicy(
        int(payload["sequence_length"]),
        int(payload["feature_dim"]),
        int(payload["action_dim"]),
        hidden_dim=hidden_dim,
    )
    load_checkpoint(policy, payload["state_dict"])
    return policy


def _build_continuation_policy(bundle: DirectionDatasetBundle) -> nn.Module:
    return LSTMContinuationActorCriticPolicy(bundle.sequence_length, bundle.feature_dim, action_dim=2)


def _transfer_weights(source: nn.Module, target: nn.Module) -> None:
    src = source.state_dict()
    tgt = target.state_dict()
    key_map = {
        "input_proj.weight": "input_proj.weight",
        "input_proj.bias": "input_proj.bias",
        "lstm.weight_ih_l0": "lstm.weight_ih_l0",
        "lstm.weight_hh_l0": "lstm.weight_hh_l0",
        "lstm.bias_ih_l0": "lstm.bias_ih_l0",
        "lstm.bias_hh_l0": "lstm.bias_hh_l0",
        "lstm.weight_ih_l1": "lstm.weight_ih_l1",
        "lstm.weight_hh_l1": "lstm.weight_hh_l1",
        "lstm.bias_ih_l1": "lstm.bias_ih_l1",
        "lstm.bias_hh_l1": "lstm.bias_hh_l1",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
        "head.weight": "policy_head.weight",
        "head.bias": "policy_head.bias",
    }
    for source_key, target_key in key_map.items():
        tgt[target_key] = src[source_key].detach().cpu().clone()
    target.load_state_dict(tgt)


def _save_policy_artifact(path: Path, policy: nn.Module, bundle: DirectionDatasetBundle) -> str:
    raise RuntimeError("_save_policy_artifact requires source_name; use _save_policy_artifact_with_name instead.")


def _save_policy_artifact_with_name(path: Path, policy: nn.Module, bundle: DirectionDatasetBundle, source_name: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": str(source_name),
        "series_name": path.stem,
        "policy_class": policy.__class__.__name__,
        "policy_family": "lstm",
        "sequence_length": int(bundle.sequence_length),
        "feature_dim": int(bundle.feature_dim),
        "action_dim": 2,
        "hidden_dim": int(policy.input_proj.out_features),
        "env_version": ENV_VERSION_BINARY,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()},
    }
    torch.save(payload, path)
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
        model_variation=TARGET_MODEL_VARIATION,
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


def _canonical_paths(output_root: Path, time_variant: str, args: argparse.Namespace) -> dict[str, Path]:
    group_dir = _variant_group_dir(output_root, time_variant, args)
    slug = _variant_slug(time_variant)
    window_token = _window_length_token(args)
    source_stub = _target_source_name(args).lower()
    return {
        "group_dir": group_dir,
        "source_copy": group_dir / f"{source_stub}{slug}.pt",
        "base": group_dir / f"{source_stub}_{window_token}_ppo_window_continue_base{slug}.pt",
        **{f"w{index}": group_dir / f"{source_stub}_{window_token}_ppo_window_continue_w{index}{slug}.pt" for index in range(1, NUM_WINDOWS + 1)},
    }


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
        model_variation=TARGET_MODEL_VARIATION,
        time_variant=str(time_variant).strip().upper(),
    )


def _window_checkpoint_metadata(
    full_bundle: DirectionDatasetBundle,
    paths: dict[str, Path],
    num_windows: int,
    time_variant: str,
    window_hours: int,
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for index in range(1, num_windows + 1):
        path = paths[f"w{index}"]
        if not path.exists():
            break
        train_bundle, eval_bundle = _build_window_bundle(full_bundle, index=index, window_hours=int(window_hours))
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
    base_timestamps = _timestamps_for_indices(full_bundle, full_bundle.test_decision_indices[: int(window_hours)])
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
        checkpoint_bundle, eval_bundle = _build_window_bundle(full_bundle, window_idx=window_index, window_hours=int(window_hours))
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

    source_payload = _load_torch_artifact(Path(base_saved_model_path))
    configured_num_windows = int(getattr(args, "num_windows", 0) or 0)
    if configured_num_windows <= 0:
        configured_num_windows = _default_num_windows(int(args.window_hours))
    replacement_rows: list[dict[str, Any]] = []
    resume_by_variant: dict[str, int] = {}
    for time_variant in selected_time_variants:
        target = _target_for_time_variant(time_variant, args)
        if not bool(args.overwrite) and is_target_completed(previous_rows, target):
            _log(args, 2, f"[V2 train] skipping completed {time_variant} window continue target")
            continue

        paths = _canonical_paths(output_root, time_variant, args)
        _copy_if_needed(base_saved_model_path, paths["source_copy"])
        if bool(args.overwrite):
            for key, path in paths.items():
                if key == "group_dir":
                    continue
                if path.exists() and path.is_file():
                    path.unlink()

        source_policy = _instantiate_base_policy(source_payload).to(device)
        continuation_policy = _build_continuation_policy(full_bundle).to(device)
        _transfer_weights(source_policy.cpu(), continuation_policy.cpu())
        continuation_policy = continuation_policy.to(device)
        if not paths["base"].exists():
            _save_policy_artifact_with_name(paths["base"], continuation_policy, full_bundle, source_name)

        resume_count = 0 if bool(args.overwrite) else _existing_window_count(paths, configured_num_windows)
        resume_by_variant[time_variant] = resume_count
        if resume_count > 0:
            _log(args, 2, f"[V2 train] resuming {time_variant} window training from W{resume_count + 1}")
            resume_payload = _load_torch_artifact(paths[f"w{resume_count}"])
            load_checkpoint(continuation_policy, resume_payload["state_dict"])
        else:
            _log(args, 2, f"[V2 train] starting {time_variant} window training from base checkpoint")

        for window_idx in range(resume_count + 1, configured_num_windows + 1):
            lr_value = float(args.ppo_learning_rate) * (float(args.continual_lr_decay) ** (window_idx - 1))
            clip_value = float(args.ppo_clip_epsilon) * (float(args.continual_clip_decay) ** (window_idx - 1))
            entropy_value = float(args.ppo_entropy_coef) * (float(args.continual_entropy_decay) ** (window_idx - 1))
            _log(args, 1, f"[V2 train] {time_variant} W{window_idx} train/eval/save")
            kl_ref_state_dict = clone_state_dict(continuation_policy) if float(args.previous_policy_kl_coef) > 0 else None
            train_bundle, eval_bundle = _build_window_bundle(full_bundle, window_idx=window_idx, window_hours=int(args.window_hours))
            train_bundle = _bundle_for_time_variant(train_bundle, time_variant)
            eval_bundle = _bundle_for_time_variant(eval_bundle, time_variant)
            train_env = BTCDirectionEnv(train_bundle, "train", env_version=ENV_VERSION_BINARY)
            progress_train_eval_env = BTCDirectionEnv(train_bundle, "train", env_version=ENV_VERSION_BINARY)
            progress_eval_env = BTCDirectionEnv(eval_bundle, "test", env_version=ENV_VERSION_BINARY)
            eval_timestamps = _timestamps_for_indices(eval_bundle, eval_bundle.test_decision_indices)
            history, checkpoints, _ = ppo_continue.train(
                env=train_env,
                policy=continuation_policy,
                device=device,
                policy_family="lstm",
                total_updates=int(args.ppo_total_updates),
                trajectories_per_update=int(args.ppo_trajectories_per_update),
                learning_rate=lr_value,
                clip_epsilon=clip_value,
                entropy_coef=entropy_value,
                value_loss_coef=float(args.ppo_value_loss_coef),
                ppo_epochs=int(args.ppo_epochs),
                minibatch_size=int(args.ppo_minibatch_size),
                previous_policy_kl_coef=float(args.previous_policy_kl_coef),
                reference_policy_state_dict=kl_ref_state_dict,
                progress_train_eval_env=progress_train_eval_env,
                progress_recent_eval_env=progress_eval_env,
                enable_progress=False,
            )
            del history
            load_checkpoint(continuation_policy, checkpoints["final"])
            _save_policy_artifact_with_name(paths[f"w{window_idx}"], continuation_policy, full_bundle, source_name)
            _log(args, 2, f"[V2 train] {time_variant} W{window_idx} saved eval window ending {eval_timestamps[-1] if eval_timestamps else ''}")

        checkpoint_metadata = _window_checkpoint_metadata(full_bundle, paths, configured_num_windows, time_variant, int(args.window_hours))
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
            saved_model_path=str(paths["base"].resolve()),
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
            model_variation=TARGET_MODEL_VARIATION,
            window_length=_target_window_length(args),
            artifact_bucket="windowed",
            source_model=source_name,
            env_version="ternary",
        )
        group_dir = str(paths["group_dir"].resolve())
        for row in variant_rows:
            row["saved_model_path"] = str(paths["base"].resolve())
            row["source_saved_model_path"] = str(paths["base"].resolve())
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
    summary["experiment_id"] = EXPERIMENT_ID
    summary["name"] = EXPERIMENT_NAME
    summary["series_order"] = [row["name"] for row in merged_rows]
    summary["models"] = merged_rows
    summary["direct_target_combinations"] = direct_target_combinations_from_rows(merged_rows)
    summary["target_rows"] = target_rows_from_rows(merged_rows)
    summary_path, manifest_path = write_manifest_and_summary(output_root, summary)
    combinations_path = write_combinations_markdown(output_root, summary, merged_rows)
    write_window_train_manifests_from_rows(merged_rows)
    return {
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combinations_path": str(combinations_path.resolve()),
        "retrained_row_names": [str(row.get("name")) for row in replacement_rows if row.get("name")],
        "row_count": len(merged_rows),
        "native_backend": "window_train_v2",
        "resumed_from_window": resume_by_variant,
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
