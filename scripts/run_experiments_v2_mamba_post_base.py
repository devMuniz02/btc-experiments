from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
import sys
from typing import Any, Callable

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
    V2TargetRow,
    build_parallel_eval_rows,
    dedupe_rows_by_name,
    direct_target_combinations_from_rows,
    is_target_completed,
    normalize_model_variation,
    normalize_train_length,
    normalize_v2_family,
    normalize_window_length,
    target_row_saved_model_dir,
    target_rows_from_rows,
    time_variant_predicate,
    write_combinations_markdown,
    write_manifest_and_summary,
    write_window_train_manifests_from_rows,
    load_summary,
    load_summary_rows,
)
from src.btc_direction_learning.algorithms.mamba_post_base import (
    train_actor_critic,
    train_bandit,
    train_ppo_full,
    train_ppo_hybrid,
)
from src.btc_direction_learning.config import FIXED_REGIME_NAME, FIXED_TEST_POOL_HOURS
from src.btc_direction_learning.continuation_models import MambaBanditPolicy, MambaContinuationActorCriticPolicy
from src.btc_direction_learning.dataset import (
    DirectionDatasetBundle,
    build_direction_dataset_bundle,
    build_direction_dataset_bundle_from_processed_with_feature_columns,
    ensure_shared_data_dir,
)
from src.btc_direction_learning.evaluation import clone_state_dict, load_checkpoint
from src.btc_direction_learning.models import MambaClassificationPolicy
from src.utils.market_data import set_seed
from scripts.run_experiment_train_once import resolve_device


SUPPORTED_VARIATIONS = {
    "PPO_FULL",
    "PPO_HYBRID",
    "ACTOR_CRITIC",
    "BANDITS",
    "BANDITS_TS",
    "BANDITS_UCB",
    "PPO_FULL_WINDOW_CONTINUE",
    "PPO_HYBRID_WINDOW_CONTINUE",
    "ACTOR_CRITIC_WINDOW_CONTINUE",
    "BANDITS_WINDOW_CONTINUE",
    "BANDITS_TS_WINDOW_CONTINUE",
    "BANDITS_UCB_WINDOW_CONTINUE",
}
DEFAULT_TOTAL_UPDATES = 12
DEFAULT_MINIBATCH_SIZE = 2048
DEFAULT_WINDOW_HOURS = 1000
TARGET_FAMILY = "MAMBA"
TEST_POOL_HOURS = FIXED_TEST_POOL_HOURS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mamba post-base window variants natively into EXPERIMENTSV2.")
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / str(EXPERIMENT_ID)))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--raw-fetch-hours", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-train-length", default="35K")
    parser.add_argument("--target-source-name", default="")
    parser.add_argument("--target-model-variation", choices=sorted(SUPPORTED_VARIATIONS), required=True)
    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--target-window-length", default="")
    parser.add_argument("--num-windows", type=int, default=0)
    parser.add_argument("--total-updates", type=int, default=DEFAULT_TOTAL_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--time-variants", nargs="+", default=list(TIME_VARIANTS))
    parser.add_argument("--bandit-strategy", choices=["ts", "ucb"], default="ts")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def _log(args: argparse.Namespace, level: int, message: str) -> None:
    if int(getattr(args, "verbose", 1)) >= level:
        print(message, flush=True)


def _target_train_length(args: argparse.Namespace) -> str:
    return normalize_train_length(getattr(args, "target_train_length", "35K"))


def _target_source_name(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "target_source_name", "") or "").strip().upper()
    if explicit:
        return explicit
    return f"{TARGET_FAMILY}_{_target_train_length(args)}"


def _target_window_length(args: argparse.Namespace) -> str:
    explicit = normalize_window_length(getattr(args, "target_window_length", ""))
    if explicit:
        return explicit
    return normalize_window_length(getattr(args, "window_hours", DEFAULT_WINDOW_HOURS))


def _window_length_token(args: argparse.Namespace) -> str:
    return _target_window_length(args).lower()


def _default_num_windows(window_hours: int) -> int:
    return max(0, (int(TEST_POOL_HOURS) // int(window_hours)) - 1)


def _target_for_time_variant(time_variant: str, args: argparse.Namespace) -> V2TargetRow:
    return V2TargetRow(
        family=TARGET_FAMILY,
        train_length=_target_train_length(args),
        window_length=_target_window_length(args),
        model_variation=_persisted_variation(args),
        time_variant=str(time_variant).strip().upper(),
    )


def _canonical_variation(variation: str) -> str:
    normalized = normalize_model_variation(variation)
    if normalized.endswith("_WINDOW_CONTINUE"):
        normalized = normalized[: -len("_WINDOW_CONTINUE")]
    if normalized in {"BANDITS_TS", "BANDITS_UCB"}:
        return "BANDITS"
    return normalized


def _resolved_bandit_strategy(args: argparse.Namespace) -> str:
    variation = normalize_model_variation(args.target_model_variation)
    if variation in {"BANDITS_TS", "BANDITS_TS_WINDOW_CONTINUE"}:
        return "ts"
    if variation in {"BANDITS_UCB", "BANDITS_UCB_WINDOW_CONTINUE"}:
        return "ucb"
    return str(getattr(args, "bandit_strategy", "ts")).strip().lower() or "ts"


def _persisted_variation(args: argparse.Namespace) -> str:
    variation = normalize_model_variation(args.target_model_variation)
    if _canonical_variation(variation) != "BANDITS":
        return variation
    strategy = _resolved_bandit_strategy(args).upper()
    if variation.endswith("_WINDOW_CONTINUE"):
        return f"BANDITS_{strategy}_WINDOW_CONTINUE"
    return f"BANDITS_{strategy}"


def _find_v2_base_row_for_args(output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = load_summary_rows(output_root)
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("family") or "").upper() == TARGET_FAMILY
        and str(row.get("train_length") or "").upper() == _target_train_length(args)
        and not str(row.get("window_length") or "").strip()
        and str(row.get("model_variation") or "").upper() == "BASE"
        and str(row.get("time_variant") or "").upper() == "FULL"
        and int(row.get("threshold_pct") or 0) == 50
        and str(row.get("saved_model_path") or "").strip()
    ]
    if not candidates:
        raise RuntimeError(
            f"Mamba post-base training requires the canonical {TARGET_FAMILY}_{_target_train_length(args)} FULL BASE 50 row in EXPERIMENTSV2."
        )
    return dict(candidates[0])


def _base_train_end_date(base_row: dict[str, Any]) -> str:
    timestamps = ((base_row.get("train") or {}).get("timestamps") or [])
    return str(timestamps[-1]) if isinstance(timestamps, list) and timestamps else ""


def _base_train_rows(base_row: dict[str, Any]) -> int:
    return int(base_row.get("actual_train_rows") or base_row.get("requested_train_rows") or 35000)


def _infer_num_layers(state_dict: dict[str, Any]) -> int:
    layer_indices = set()
    for key in state_dict.keys():
        if not key.startswith("layers."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            layer_indices.add(int(parts[1]))
    return (max(layer_indices) + 1) if layer_indices else 3


def _load_torch_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Bad artifact format: {path}")
    return payload


def _instantiate_base_policy(payload: dict[str, Any]) -> MambaClassificationPolicy:
    hidden_dim = int(payload.get("hidden_dim") or payload["state_dict"]["input_proj.weight"].shape[0])
    num_layers = int(payload.get("num_layers") or _infer_num_layers(payload["state_dict"]))
    policy = MambaClassificationPolicy(
        int(payload["sequence_length"]),
        int(payload["feature_dim"]),
        int(payload["action_dim"]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    load_checkpoint(policy, payload["state_dict"])
    return policy


def _instantiate_variant_policy(
    *,
    variation: str,
    bundle: DirectionDatasetBundle,
    hidden_dim: int,
    num_layers: int,
    bandit_strategy: str,
) -> nn.Module:
    if _canonical_variation(variation) == "BANDITS":
        return MambaBanditPolicy(
            bundle.sequence_length,
            bundle.feature_dim,
            action_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            bandit_strategy=bandit_strategy,
        )
    return MambaContinuationActorCriticPolicy(
        bundle.sequence_length,
        bundle.feature_dim,
        action_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    )


def _transfer_base_to_variant(source: MambaClassificationPolicy, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared_prefixes = ("input_proj.", "layers.", "norm.")
    for key, value in source_state.items():
        if key.startswith(shared_prefixes) and key in target_state:
            target_state[key] = value.detach().cpu().clone()
    if "head.weight" in source_state and "policy_head.weight" in target_state:
        target_state["policy_head.weight"][:2] = source_state["head.weight"].detach().cpu().clone()
        target_state["policy_head.weight"][2].zero_()
    if "head.bias" in source_state and "policy_head.bias" in target_state:
        target_state["policy_head.bias"][:2] = source_state["head.bias"].detach().cpu().clone()
        target_state["policy_head.bias"][2] = 0.0
    if "head.weight" in source_state and "head.weight" in target_state:
        target_state["head.weight"][:2] = source_state["head.weight"].detach().cpu().clone()
        target_state["head.weight"][2].zero_()
    if "head.bias" in source_state and "head.bias" in target_state:
        target_state["head.bias"][:2] = source_state["head.bias"].detach().cpu().clone()
        target_state["head.bias"][2] = 0.0
    target.load_state_dict(target_state, strict=False)


def _save_policy_artifact(
    path: Path,
    policy: nn.Module,
    bundle: DirectionDatasetBundle,
    *,
    source_name: str,
    policy_family: str,
    variation: str,
    bandit_strategy: str | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": str(source_name),
        "series_name": path.stem,
        "policy_class": policy.__class__.__name__,
        "policy_family": policy_family,
        "model_variation": variation,
        "sequence_length": int(bundle.sequence_length),
        "feature_dim": int(bundle.feature_dim),
        "action_dim": 3,
        "hidden_dim": int(getattr(policy, "hidden_dim", policy.input_proj.out_features)),
        "num_layers": int(getattr(policy, "num_layers", len(getattr(policy, "layers", [])))),
        "bandit_strategy": bandit_strategy,
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


def _target_row_saved_dir(output_root: Path, args: argparse.Namespace, time_variant: str) -> Path:
    target = _target_for_time_variant(time_variant, args)
    group_dir = target_row_saved_model_dir(output_root, target)
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def _canonical_paths(output_root: Path, time_variant: str, args: argparse.Namespace) -> dict[str, Path]:
    group_dir = _target_row_saved_dir(output_root, args, time_variant)
    slug = "" if str(time_variant).upper() == "FULL" else f"_{str(time_variant).lower()}"
    window_token = _window_length_token(args)
    source_stub = _target_source_name(args).lower()
    variation = _persisted_variation(args).lower()
    return {
        "group_dir": group_dir,
        "source_copy": group_dir / f"{source_stub}{slug}.pt",
        "base": group_dir / f"{source_stub}_{window_token}_{variation}_base{slug}.pt",
        **{f"w{index}": group_dir / f"{source_stub}_{window_token}_{variation}_w{index}{slug}.pt" for index in range(1, int(getattr(args, 'num_windows', 0) or 0) + 1)},
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
        if paths.get(f"w{index}") and paths[f"w{index}"].exists():
            count = index
        else:
            break
    return count


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
        train_bundle, eval_bundle = _build_window_bundle(full_bundle, index, int(window_hours))
        train_bundle = _bundle_for_time_variant(train_bundle, time_variant)
        eval_bundle = _bundle_for_time_variant(eval_bundle, time_variant)
        del train_bundle
        eval_timestamps = _timestamps_for_indices(eval_bundle, eval_bundle.test_decision_indices)
        checkpoints.append(
            {
                "window_index": index,
                "window_name": f"W{index}",
                "normalized_window_name": f"W{index}",
                "saved_model_path": str(path.resolve()),
                "window_train_end": eval_timestamps[0] if eval_timestamps else "",
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
        _, eval_bundle = _build_window_bundle(full_bundle, window_index, int(window_hours))
        eval_bundle = _bundle_for_time_variant(eval_bundle, time_variant)
        stream_parts.append(
            build_prediction_stream_from_saved_model(
                saved_path=str(checkpoint["saved_model_path"]),
                eval_bundle=eval_bundle,
                device=torch.device("cpu"),
                timestamps_override=test_timestamps,
            )
        )
    return concat_prediction_streams(stream_parts)


def _train_variant_for_window(
    *,
    args: argparse.Namespace,
    variation: str,
    bundle: DirectionDatasetBundle,
    policy: nn.Module,
    device: torch.device,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    common_kwargs = {
        "bundle": bundle,
        "policy": policy,
        "device": device,
        "progress_callback": progress_callback,
    }
    canonical_variation = _canonical_variation(variation)
    if canonical_variation == "PPO_FULL":
        return train_ppo_full(
            **common_kwargs,
            total_updates=int(args.total_updates),
            learning_rate=float(args.learning_rate),
            clip_epsilon=float(args.clip_epsilon),
            entropy_coef=float(args.entropy_coef),
            value_loss_coef=float(args.value_loss_coef),
            ppo_epochs=int(args.ppo_epochs),
            minibatch_size=int(args.minibatch_size),
        )
    if canonical_variation == "PPO_HYBRID":
        return train_ppo_hybrid(
            **common_kwargs,
            total_updates=int(args.total_updates),
            learning_rate=float(args.learning_rate),
            clip_epsilon=float(args.clip_epsilon),
            entropy_coef=float(args.entropy_coef) * 0.5,
            value_loss_coef=float(args.value_loss_coef),
            ppo_epochs=max(4, int(args.ppo_epochs)),
            minibatch_size=max(2048, int(args.minibatch_size)),
        )
    if canonical_variation == "ACTOR_CRITIC":
        return train_actor_critic(
            **common_kwargs,
            total_updates=int(args.total_updates),
            learning_rate=float(args.learning_rate),
            entropy_coef=float(args.entropy_coef),
            value_loss_coef=float(args.value_loss_coef),
            minibatch_size=int(args.minibatch_size),
        )
    if canonical_variation == "BANDITS":
        if not isinstance(policy, MambaBanditPolicy):
            raise TypeError("BANDITS requires MambaBanditPolicy")
        return train_bandit(
            **common_kwargs,
            strategy=_resolved_bandit_strategy(args),
            ucb_alpha=1.0,
        )
    raise ValueError(f"Unsupported Mamba post-base variation: {variation}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    previous_summary = load_summary(output_root)
    previous_rows = load_summary_rows(output_root)
    variation = normalize_model_variation(args.target_model_variation)
    canonical_variation = _canonical_variation(variation)
    persisted_variation = _persisted_variation(args)
    selected_time_variants = [str(value).strip().upper() for value in getattr(args, "time_variants", TIME_VARIANTS) if str(value).strip()] or list(TIME_VARIANTS)
    base_row = _find_v2_base_row_for_args(output_root, args)
    base_saved_model_path = str(base_row.get("saved_model_path") or "").strip()
    base_train_rows = _base_train_rows(base_row)
    base_train_end_date = _base_train_end_date(base_row)
    source_name = _target_source_name(args)
    device = resolve_device(args.device)
    set_seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    configured_num_windows = int(getattr(args, "num_windows", 0) or 0)
    if configured_num_windows <= 0:
        configured_num_windows = _default_num_windows(int(args.window_hours))
        args.num_windows = configured_num_windows

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
    base_policy = _instantiate_base_policy(source_payload)
    hidden_dim = int(getattr(base_policy, "head").in_features)
    num_layers = len(base_policy.layers)

    replacement_rows: list[dict[str, Any]] = []
    resume_by_variant: dict[str, int] = {}
    for time_variant in selected_time_variants:
        target = _target_for_time_variant(time_variant, args)
        if not bool(args.overwrite) and is_target_completed(previous_rows, target):
            _log(args, 2, f"[V2 train] skipping completed {time_variant} {variation} target")
            continue

        if int(getattr(args, "verbose", 1)) >= 1:
            _log(args, 1, "[V2 train] freeing memory before train run")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        paths = _canonical_paths(output_root, time_variant, args)
        _copy_if_needed(base_saved_model_path, paths["source_copy"])
        if bool(args.overwrite):
            for key, path in paths.items():
                if key == "group_dir":
                    continue
                if path.exists() and path.is_file():
                    path.unlink()

        variant_policy = _instantiate_variant_policy(
            variation=variation,
            bundle=full_bundle,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bandit_strategy=_resolved_bandit_strategy(args),
        )
        _transfer_base_to_variant(base_policy.cpu(), variant_policy.cpu())
        variant_policy = variant_policy.to(device)
        if not paths["base"].exists():
            _save_policy_artifact(
                paths["base"],
                variant_policy,
                full_bundle,
                source_name=source_name,
                policy_family="mamba",
                variation=persisted_variation,
                bandit_strategy=_resolved_bandit_strategy(args) if canonical_variation == "BANDITS" else None,
            )

        resume_count = 0 if bool(args.overwrite) else _existing_window_count(paths, configured_num_windows)
        resume_by_variant[time_variant] = resume_count
        if resume_count > 0:
            resume_payload = _load_torch_artifact(paths[f"w{resume_count}"])
            load_checkpoint(variant_policy, resume_payload["state_dict"])

        for window_idx in range(resume_count + 1, configured_num_windows + 1):
            _log(args, 1, f"[V2 train] {variation} {time_variant} W{window_idx} train/eval/save")
            train_bundle, eval_bundle = _build_window_bundle(full_bundle, window_idx, int(args.window_hours))
            train_bundle = _bundle_for_time_variant(train_bundle, time_variant)
            eval_bundle = _bundle_for_time_variant(eval_bundle, time_variant)
            total_windows = configured_num_windows

            def _progress_callback(metrics: dict[str, Any], epoch: int, total_epochs: int) -> None:
                if int(getattr(args, "verbose", 1)) < 1:
                    return
                print(
                    (
                        f"[V2 train] {variation} {time_variant} W{window_idx}/{total_windows} "
                        f"epoch {epoch}/{total_epochs} "
                        f"loss={float(metrics.get('loss', 0.0)):.6f} "
                        f"acc={float(metrics.get('train_accuracy', 0.0)):.4f} "
                        f"reward={float(metrics.get('mean_reward', 0.0)):.4f}"
                    ),
                    flush=True,
                )

            _, checkpoints, _ = _train_variant_for_window(
                args=args,
                variation=variation,
                bundle=train_bundle,
                policy=variant_policy,
                device=device,
                progress_callback=_progress_callback,
            )
            load_checkpoint(variant_policy, checkpoints["final"])
            _save_policy_artifact(
                paths[f"w{window_idx}"],
                variant_policy,
                full_bundle,
                source_name=source_name,
                policy_family="mamba",
                variation=persisted_variation,
                bandit_strategy=_resolved_bandit_strategy(args) if canonical_variation == "BANDITS" else None,
            )
            del train_bundle, eval_bundle

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
            combination=DirectCombination(family=TARGET_FAMILY, train_length=_target_train_length(args)),
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
            model_variation=persisted_variation,
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
            if canonical_variation == "BANDITS":
                row["bandit_strategy"] = _resolved_bandit_strategy(args)
        replacement_rows.extend(variant_rows)

        if int(getattr(args, "verbose", 1)) >= 1:
            _log(args, 1, "[V2 train] freeing memory after train run")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
        "trained_row_names": [str(row.get("name")) for row in replacement_rows if row.get("name")],
        "row_count": len(merged_rows),
        "native_backend": "mamba_post_base_v2",
        "resumed_from_window": resume_by_variant,
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
