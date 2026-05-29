from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_experiments_v2_direct_models as direct_v2
from scripts import run_experiments_v2_mamba_post_base as mamba_post_base_v2
from scripts import run_experiments_v2_stacked_ensemble as stacker_v2
from scripts import run_experiments_v2_window_retrain as window_retrain_v2
from scripts import run_experiments_v2_window_train as window_train_v2
from src.utils.experiment_support import (
    EXPERIMENTS_V2_ROOT,
    EXPERIMENT_ID,
    RetrainableTarget,
    current_source_backed_targets,
    is_combination_completed,
    is_target_completed,
    load_summary,
    load_summary_rows,
    manifest_path,
    normalize_model_variation,
    normalize_train_length,
    normalize_v2_family,
    normalize_window_length,
    target_rows,
    train_length_to_rows,
    write_combinations_markdown,
)


SUPPORTED_SELECTION_MODES = ("all", "same_as_current", "missing_only")
DIRECT_FAMILIES = {"BC", "DAGGER", "NN", "RF", "XGBOOST", "LSTM", "TRANSFORMER", "MAMBA", "PPO"}
STACKER_FAMILY = stacker_v2.STACKER_FAMILY
STACKER_TARGET = RetrainableTarget(
    source_experiment_id=EXPERIMENT_ID,
    source_experiment_name="DIRECT_FIXED_5K_V2",
    source_variant="full",
    source_name="STACKER_ALL",
    family=STACKER_FAMILY,
    train_length=stacker_v2.STACKER_TRAIN_LENGTH,
    window_length="",
    model_variation=stacker_v2.STACKER_VARIATION,
)
CANONICAL_WINDOW_TARGET = RetrainableTarget(
    source_experiment_id=EXPERIMENT_ID,
    source_experiment_name="DIRECT_FIXED_5K_V2",
    source_variant="full",
    source_name="LSTM_35K",
    family="LSTM",
    train_length="35K",
    window_length="1K",
    model_variation="PPO_WINDOW_CONTINUE",
)
CANONICAL_WINDOW_RETRAIN_TARGET = RetrainableTarget(
    source_experiment_id=EXPERIMENT_ID,
    source_experiment_name="DIRECT_FIXED_5K_V2",
    source_variant="full",
    source_name="LSTM_5K",
    family="LSTM",
    train_length="5K",
    window_length="1K",
    model_variation="WINDOW_RETRAIN",
)
STACKER_BASE_TARGETS = (
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "LSTM_5K", "LSTM", "5K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "LSTM_10K", "LSTM", "10K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "LSTM_20K", "LSTM", "20K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "LSTM_30K", "LSTM", "30K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "LSTM_35K", "LSTM", "35K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "TRANSFORMER_5K", "TRANSFORMER", "5K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "TRANSFORMER_10K", "TRANSFORMER", "10K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "TRANSFORMER_20K", "TRANSFORMER", "20K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "TRANSFORMER_30K", "TRANSFORMER", "30K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "TRANSFORMER_35K", "TRANSFORMER", "35K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "MAMBA_5K", "MAMBA", "5K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "MAMBA_10K", "MAMBA", "10K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "MAMBA_20K", "MAMBA", "20K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "MAMBA_30K", "MAMBA", "30K", "", "BASE"),
    RetrainableTarget(EXPERIMENT_ID, "DIRECT_FIXED_5K_V2", "full", "MAMBA_35K", "MAMBA", "35K", "", "BASE"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or retrain V2-native EXPERIMENTSV2 targets.")
    parser.add_argument("--selection-mode", choices=SUPPORTED_SELECTION_MODES, required=True)
    parser.add_argument("--families", nargs="+", default=[])
    parser.add_argument("--train-lengths", nargs="+", default=[])
    parser.add_argument("--window-lengths", nargs="+", default=[])
    parser.add_argument("--model-variations", nargs="+", default=[])
    parser.add_argument("--time-variants", nargs="+", default=["FULL", "MARKET_HOURS", "OUTSIDE_MARKET_HOURS"])
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / str(EXPERIMENT_ID)))
    parser.add_argument("--experiments-root", default=str(PROJECT_ROOT / "EXPERIMENTS"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--envs", nargs="+", default=["ternary"])
    parser.add_argument("--verbose", type=int, default=1)
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
    parser.add_argument("--retrain-epochs", type=int, default=window_retrain_v2.DEFAULT_RETRAIN_EPOCHS)
    return parser.parse_args()


def _normalized_filter_set(values: list[str]) -> set[str]:
    return {str(value).strip().upper() for value in values if str(value).strip()}


def _canonical_key(target: RetrainableTarget) -> tuple[str, str, str, str]:
    return (
        normalize_v2_family(target.family),
        normalize_train_length(target.train_length),
        normalize_window_length(target.window_length),
        normalize_model_variation(target.model_variation),
    )


def _window_length_to_hours(value: str) -> int:
    normalized = normalize_window_length(value)
    if not normalized:
        return 0
    text = str(normalized).strip().upper()
    if text.endswith("K"):
        return int(text[:-1]) * 1000
    return int(text)


def _matches_filters(target: RetrainableTarget, args: argparse.Namespace) -> bool:
    families = _normalized_filter_set(list(args.families or []))
    train_lengths = _normalized_filter_set(list(args.train_lengths or []))
    window_lengths = _normalized_filter_set(list(args.window_lengths or []))
    model_variations = _normalized_filter_set(list(args.model_variations or []))
    if families and normalize_v2_family(target.family) not in families:
        return False
    if train_lengths and normalize_train_length(target.train_length) not in train_lengths:
        return False
    if window_lengths and normalize_window_length(target.window_length) not in window_lengths:
        return False
    if model_variations and normalize_model_variation(target.model_variation) not in model_variations:
        return False
    return True


def _default_direct_targets() -> list[RetrainableTarget]:
    defaults: list[RetrainableTarget] = []
    for family in sorted(DIRECT_FAMILIES):
        for train_length in ("5K", "10K", "20K", "30K", "35K"):
            defaults.append(
                RetrainableTarget(
                    source_experiment_id=EXPERIMENT_ID,
                    source_experiment_name="DIRECT_FIXED_5K_V2",
                    source_variant="full",
                    source_name=f"{family}_{train_length}",
                    family=family,
                    train_length=train_length,
                    window_length="",
                    model_variation="BASE",
                )
            )
    return defaults


def _discover_targets(output_root: Path, args: argparse.Namespace) -> list[RetrainableTarget]:
    rows = load_summary_rows(output_root)
    current_targets = {
        _canonical_key(target): target
        for target in current_source_backed_targets(rows)
    }
    requested_direct_families = _normalized_filter_set(list(args.families or []))
    requested_model_variations = _normalized_filter_set(list(args.model_variations or []))
    requested_families = sorted(_normalized_filter_set(list(args.families or [])))
    requested_train_lengths = sorted(_normalized_filter_set(list(args.train_lengths or [])))
    requested_window_lengths = sorted(_normalized_filter_set(list(args.window_lengths or [])))
    requested_variations = sorted(_normalized_filter_set(list(args.model_variations or [])))

    defaults = _default_direct_targets() if not rows else []
    for target in defaults:
        current_targets.setdefault(_canonical_key(target), target)

    for row_target in target_rows(load_summary(output_root), rows):
        target = RetrainableTarget(
            source_experiment_id=EXPERIMENT_ID,
            source_experiment_name="DIRECT_FIXED_5K_V2",
            source_variant="full",
            source_name=f"{row_target['family']}_{row_target['train_length']}",
            family=row_target["family"],
            train_length=row_target["train_length"],
            window_length=row_target.get("window_length", ""),
            model_variation=row_target.get("model_variation", "BASE"),
        )
        current_targets.setdefault(_canonical_key(target), target)

    if (
        (not requested_direct_families or "LSTM" in requested_direct_families)
        and (not requested_model_variations or "PPO_WINDOW_CONTINUE" in requested_model_variations)
    ):
        current_targets.setdefault(_canonical_key(CANONICAL_WINDOW_TARGET), CANONICAL_WINDOW_TARGET)
    if (
        (not requested_direct_families or "LSTM" in requested_direct_families)
        and (not requested_model_variations or "WINDOW_RETRAIN" in requested_model_variations)
    ):
        current_targets.setdefault(_canonical_key(CANONICAL_WINDOW_RETRAIN_TARGET), CANONICAL_WINDOW_RETRAIN_TARGET)
    if (
        (not requested_families or STACKER_FAMILY in requested_families)
        and (not requested_train_lengths or stacker_v2.STACKER_TRAIN_LENGTH in requested_train_lengths)
        and (not requested_variations or stacker_v2.STACKER_VARIATION in requested_variations)
    ):
        current_targets.setdefault(_canonical_key(STACKER_TARGET), STACKER_TARGET)
    if requested_families and requested_train_lengths and (not requested_variations or "BASE" in requested_variations):
        for family in requested_families:
            if family not in DIRECT_FAMILIES:
                continue
            for train_length in requested_train_lengths:
                target = RetrainableTarget(
                    source_experiment_id=EXPERIMENT_ID,
                    source_experiment_name="DIRECT_FIXED_5K_V2",
                    source_variant="full",
                    source_name=f"{family}_{train_length}",
                    family=family,
                    train_length=train_length,
                    window_length="",
                    model_variation="BASE",
                )
                current_targets.setdefault(_canonical_key(target), target)

    if requested_families and requested_train_lengths and requested_variations:
        effective_window_lengths = requested_window_lengths or [""]
        for family in requested_families:
            for train_length in requested_train_lengths:
                for window_length in effective_window_lengths:
                    for model_variation in requested_variations:
                        target = RetrainableTarget(
                            source_experiment_id=EXPERIMENT_ID,
                            source_experiment_name="DIRECT_FIXED_5K_V2",
                            source_variant="full",
                            source_name=f"{family}_{train_length}",
                            family=family,
                            train_length=train_length,
                            window_length=window_length,
                            model_variation=model_variation,
                        )
                        current_targets.setdefault(_canonical_key(target), target)

    discovered = [target for target in current_targets.values() if _matches_filters(target, args)]
    return sorted(
        discovered,
        key=lambda item: (
            item.family,
            train_length_to_rows(item.train_length),
            item.window_length,
            item.model_variation,
        ),
    )


def _target_is_completed(summary_rows: list[dict[str, Any]], target: RetrainableTarget) -> bool:
    if not target.window_length and target.model_variation == "BASE":
        from src.utils.experiment_support import DirectCombination

        return is_combination_completed(
            summary_rows,
            DirectCombination(family=target.family, train_length=target.train_length),
        )
    from src.utils.experiment_support import V2TargetRow

    return is_target_completed(
        summary_rows,
        V2TargetRow(
            family=target.family,
            train_length=target.train_length,
            window_length=target.window_length,
            model_variation=target.model_variation,
            time_variant="FULL",
        ),
    )


def _select_targets(
    discoverable_targets: list[RetrainableTarget],
    current_targets: list[RetrainableTarget],
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[RetrainableTarget], list[RetrainableTarget]]:
    current_keys = {_canonical_key(target) for target in current_targets}
    completed_keys = {_canonical_key(target) for target in discoverable_targets if _target_is_completed(summary_rows, target)}
    if bool(args.overwrite):
        completed_keys = set()

    if args.selection_mode == "same_as_current":
        selected = [target for target in discoverable_targets if _canonical_key(target) in current_keys]
    elif args.selection_mode == "missing_only":
        selected = [target for target in discoverable_targets if _canonical_key(target) not in completed_keys]
    else:
        selected = list(discoverable_targets)
    skipped = [target for target in selected if _canonical_key(target) in completed_keys]
    return selected, skipped


def _route_target(target: RetrainableTarget) -> str:
    if not target.window_length and target.model_variation == "BASE" and target.family in DIRECT_FAMILIES:
        return "direct"
    if (
        normalize_v2_family(target.family) == "LSTM"
        and normalize_model_variation(target.model_variation) == "PPO_WINDOW_CONTINUE"
        and normalize_window_length(target.window_length)
    ):
        return "window_native"
    if (
        normalize_v2_family(target.family) == "MAMBA"
        and normalize_model_variation(target.model_variation) in {
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
        and normalize_window_length(target.window_length)
    ):
        return "mamba_post_base_native"
    if (
        normalize_v2_family(target.family) == STACKER_FAMILY
        and normalize_train_length(target.train_length) == stacker_v2.STACKER_TRAIN_LENGTH
        and not normalize_window_length(target.window_length)
        and normalize_model_variation(target.model_variation) == stacker_v2.STACKER_VARIATION
    ):
        return "stacked_ensemble_native"
    if (
        normalize_v2_family(target.family) in DIRECT_FAMILIES
        and normalize_model_variation(target.model_variation) in {"WINDOW_RETRAIN", "WINDOW_CONTINUE"}
        and normalize_window_length(target.window_length)
    ):
        return "window_retrain_native"
    return "unsupported"


def _run_one_target(target: RetrainableTarget, args: argparse.Namespace) -> dict[str, Any]:
    route = _route_target(target)
    if route == "direct":
        return direct_v2.run(
            argparse.Namespace(
                selection_mode="all",
                window_scope="non_window",
                families=[target.family],
                train_lengths=[target.train_length],
                time_variants=list(args.time_variants),
                output_dir=str(args.output_dir),
                device=args.device,
                overwrite=bool(args.overwrite),
                verbose=int(getattr(args, "verbose", 1)),
                mamba_epochs=int(getattr(args, "mamba_epochs", 0)),
                mamba_batch_size=int(getattr(args, "mamba_batch_size", 0)),
                mamba_disable_early_stopping=bool(getattr(args, "mamba_disable_early_stopping", False)),
                overfit=bool(getattr(args, "overfit", False)),
                overfit_max_epochs=int(getattr(args, "overfit_max_epochs", 5000)),
            )
        )
    if route == "window_native":
        window_length = normalize_window_length(target.window_length)
        window_hours = _window_length_to_hours(window_length)
        return window_train_v2.run(
            argparse.Namespace(
                output_dir=str(args.output_dir),
                device=args.device,
                seed=42,
                force_refresh=False,
                raw_fetch_hours=0,
                overwrite=bool(args.overwrite),
                target_family=normalize_v2_family(target.family),
                target_train_length=normalize_train_length(target.train_length),
                target_source_name=target.source_name,
                target_model_variation=normalize_model_variation(target.model_variation),
                window_hours=window_hours,
                target_window_length=window_length,
                num_windows=max(0, (5000 // window_hours) - 1),
                ppo_total_updates=window_train_v2.DEFAULT_PPO_TOTAL_UPDATES,
                ppo_trajectories_per_update=window_train_v2.DEFAULT_PPO_TRAJECTORIES_PER_UPDATE,
                ppo_learning_rate=2e-4,
                ppo_clip_epsilon=0.1,
                ppo_entropy_coef=0.01,
                ppo_value_loss_coef=0.5,
                ppo_epochs=window_train_v2.DEFAULT_PPO_EPOCHS,
                ppo_minibatch_size=window_train_v2.DEFAULT_PPO_MINIBATCH_SIZE,
                previous_policy_kl_coef=0.1,
                continual_lr_decay=1.0,
                continual_clip_decay=1.0,
                continual_entropy_decay=1.0,
                time_variants=list(args.time_variants),
                verbose=int(getattr(args, "verbose", 1)),
            )
        )
    if route == "window_retrain_native":
        window_length = normalize_window_length(target.window_length)
        window_hours = _window_length_to_hours(window_length)
        return window_retrain_v2.run(
            argparse.Namespace(
                output_dir=str(args.output_dir),
                device=args.device,
                seed=42,
                force_refresh=False,
                raw_fetch_hours=0,
                overwrite=bool(args.overwrite),
                target_family=normalize_v2_family(target.family),
                target_train_length=normalize_train_length(target.train_length),
                target_source_name=target.source_name,
                window_hours=window_hours,
                target_window_length=window_length,
                num_windows=max(0, (5000 // window_hours) - 1),
                retrain_epochs=int(getattr(args, "retrain_epochs", window_retrain_v2.DEFAULT_RETRAIN_EPOCHS)),
                learning_rate=window_retrain_v2.DEFAULT_LEARNING_RATE,
                batch_size=window_retrain_v2.DEFAULT_BATCH_SIZE,
                mamba_batch_size=int(getattr(args, "mamba_batch_size", 0)),
                mamba_disable_early_stopping=bool(getattr(args, "mamba_disable_early_stopping", False)),
                mamba_epochs=int(getattr(args, "mamba_epochs", 0)),
                overfit=bool(getattr(args, "overfit", False)),
                overfit_max_epochs=int(getattr(args, "overfit_max_epochs", 5000)),
                visualize_test_acc=bool(getattr(args, "visualize_test_acc", False)),
                time_variants=list(args.time_variants),
                verbose=int(getattr(args, "verbose", 1)),
            )
        )
    if route == "mamba_post_base_native":
        window_length = normalize_window_length(target.window_length)
        window_hours = _window_length_to_hours(window_length)
        return mamba_post_base_v2.run(
            argparse.Namespace(
                output_dir=str(args.output_dir),
                device=args.device,
                seed=42,
                force_refresh=False,
                raw_fetch_hours=0,
                overwrite=bool(args.overwrite),
                target_train_length=normalize_train_length(target.train_length),
                target_source_name=target.source_name,
                target_model_variation=normalize_model_variation(target.model_variation),
                window_hours=window_hours,
                target_window_length=window_length,
                num_windows=max(0, (5000 // window_hours) - 1),
                total_updates=int(getattr(args, "post_base_total_updates", 0) or 12),
                learning_rate=float(getattr(args, "post_base_learning_rate", 0.0) or 2e-4),
                clip_epsilon=float(getattr(args, "post_base_clip_epsilon", 0.0) or 0.1),
                entropy_coef=float(getattr(args, "post_base_entropy_coef", 0.0) or 0.01),
                value_loss_coef=float(getattr(args, "post_base_value_loss_coef", 0.0) or 0.5),
                ppo_epochs=int(getattr(args, "post_base_ppo_epochs", 0) or 4),
                minibatch_size=int(getattr(args, "post_base_minibatch_size", 0) or 2048),
                time_variants=list(args.time_variants),
                bandit_strategy=str(getattr(args, "bandit_strategy", "ts")),
                verbose=int(getattr(args, "verbose", 1)),
            )
        )
    if route == "stacked_ensemble_native":
        return stacker_v2.run(
            argparse.Namespace(
                output_dir=str(args.output_dir),
                device=args.device,
                overwrite=bool(args.overwrite),
            )
        )
    return {
        "status": "unsupported",
        "family": target.family,
        "train_length": target.train_length,
        "window_length": target.window_length,
        "model_variation": target.model_variation,
        "message": (
            f"Target {target.family} {target.train_length} {target.window_length or '-'} {target.model_variation} "
            "does not have a compatible V2-native trainer."
        ),
    }


def _expand_targets_with_dependencies(
    selected_targets: list[RetrainableTarget],
    summary_rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> list[RetrainableTarget]:
    expanded: list[RetrainableTarget] = []
    seen: set[tuple[str, str, str, str]] = set()
    for target in selected_targets:
        if _route_target(target) == "stacked_ensemble_native":
            for dependency in STACKER_BASE_TARGETS:
                dependency_key = _canonical_key(dependency)
                if dependency_key in seen:
                    continue
                if overwrite or not _target_is_completed(summary_rows, dependency):
                    expanded.append(dependency)
                    seen.add(dependency_key)
        target_key = _canonical_key(target)
        if target_key in seen:
            continue
        expanded.append(target)
        seen.add(target_key)
    return expanded


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    summary_rows = load_summary_rows(output_root)
    current_targets = current_source_backed_targets(summary_rows)
    discoverable_targets = _discover_targets(output_root, args)
    selected_targets, skipped_targets = _select_targets(
        discoverable_targets=discoverable_targets,
        current_targets=current_targets,
        summary_rows=summary_rows,
        args=args,
    )
    selected_targets = _expand_targets_with_dependencies(
        selected_targets,
        summary_rows,
        overwrite=bool(args.overwrite),
    )
    to_train = selected_targets if bool(args.overwrite) else [target for target in selected_targets if target not in skipped_targets]

    results: list[dict[str, Any]] = []
    unsupported_targets: list[dict[str, Any]] = []
    for target in to_train:
        result = _run_one_target(target, args)
        if str(result.get("status") or "") == "unsupported":
            unsupported_targets.append(
                {
                    "family": target.family,
                    "train_length": target.train_length,
                    "window_length": target.window_length,
                    "model_variation": target.model_variation,
                    "message": str(result.get("message") or ""),
                }
            )
            continue
        results.append(result)

    final_summary = load_summary(output_root)
    final_rows = load_summary_rows(output_root)
    combinations_path = write_combinations_markdown(output_root, final_summary, final_rows)
    return {
        "selected_targets": [
            {
                "family": target.family,
                "train_length": target.train_length,
                "window_length": target.window_length,
                "model_variation": target.model_variation,
                "source_name": target.source_name,
            }
            for target in selected_targets
        ],
        "trained_targets": [
            {
                "family": target.family,
                "train_length": target.train_length,
                "window_length": target.window_length,
                "model_variation": target.model_variation,
            }
            for target in to_train
        ],
        "skipped_existing": [
            {
                "family": target.family,
                "train_length": target.train_length,
                "window_length": target.window_length,
                "model_variation": target.model_variation,
            }
            for target in skipped_targets
        ],
        "results": results,
        "unsupported_targets": unsupported_targets,
        "row_count": len(final_rows),
        "summary_path": str((output_root / "full" / "summary.json").resolve()),
        "manifest_path": str(manifest_path(output_root).resolve()),
        "combinations_path": str(combinations_path.resolve()),
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
