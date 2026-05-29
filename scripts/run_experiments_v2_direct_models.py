from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._experiment_eval_reuse import build_fixed_bundle, build_prediction_stream_from_saved_model
from src.utils.experiment_support import (
    EXPERIMENTS_V2_ROOT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    TIME_VARIANTS,
    DirectCombination,
    build_parallel_eval_rows,
    dedupe_rows_by_name,
    direct_target_combinations_from_rows,
    family_saved_model_dir,
    is_combination_completed,
    load_summary,
    load_summary_rows,
    row_combination,
    target_rows_from_rows,
    train_length_to_rows,
    write_combinations_markdown,
    write_manifest_and_summary,
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


SUPPORTED_FAMILIES = ("BC", "DAGGER", "NN", "RF", "XGBOOST", "LSTM", "TRANSFORMER", "MAMBA", "PPO")
CANONICAL_TEST_ROWS = 5000

FAMILY_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
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
    parser = argparse.ArgumentParser(description="Train V2-native direct EXPERIMENTSV2 models from scratch.")
    parser.add_argument("--selection-mode", choices=["all", "same_as_current", "missing_only"], required=True)
    parser.add_argument("--window-scope", default="non_window", choices=["non_window", "windowed", "all"])
    parser.add_argument("--families", nargs="+", default=list(SUPPORTED_FAMILIES))
    parser.add_argument("--train-lengths", nargs="+", required=True)
    parser.add_argument("--time-variants", nargs="+", default=list(TIME_VARIANTS))
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / str(EXPERIMENT_ID)))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--mamba-epochs", type=int, default=0)
    parser.add_argument("--mamba-batch-size", type=int, default=0)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-max-epochs", type=int, default=5000)
    return parser.parse_args()


def _normalize_family(value: str) -> str:
    family = str(value).strip().upper()
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported family: {value}")
    return family


def _normalize_train_length(value: str | int) -> str:
    text = str(value).strip().upper()
    if not text:
        raise ValueError("Train length cannot be empty.")
    if text.endswith("K"):
        return text
    numeric = int(text)
    if numeric <= 0:
        raise ValueError(f"Invalid train length: {value}")
    return f"{numeric}K" if numeric < 1000 else f"{numeric // 1000}K"


def _normalize_time_variant(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in TIME_VARIANTS:
        raise ValueError(f"Unsupported time variant: {value}")
    return normalized


def _log(args: argparse.Namespace, level: int, message: str) -> None:
    if int(getattr(args, "verbose", 1)) >= level:
        print(message, flush=True)


def _save_model_artifact(result: dict[str, Any], target_dir: Path) -> str:
    artifact = result.pop("__model_artifact__", None)
    if not isinstance(artifact, dict):
        raise RuntimeError("Training result is missing __model_artifact__.")
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / str(artifact["filename"])
    if artifact["kind"] == "torch":
        torch.save(artifact["payload"], artifact_path)
    elif artifact["kind"] == "pickle":
        with artifact_path.open("wb") as handle:
            pickle.dump(artifact["payload"], handle)
    else:
        raise ValueError(f"Unsupported model artifact kind: {artifact['kind']}")
    return str(artifact_path.resolve())


def _existing_direct_combinations(rows: list[dict[str, Any]]) -> set[DirectCombination]:
    result: set[DirectCombination] = set()
    for row in rows:
        try:
            combination = row_combination(row)
        except Exception:
            continue
        if str(row.get("model_variation") or "").upper() != "BASE":
            continue
        if str(row.get("window_length") or "").strip():
            continue
        result.add(combination)
    return result


def _requested_combinations(families: list[str], train_lengths: list[str]) -> list[DirectCombination]:
    return sorted(
        [DirectCombination(family=family, train_length=train_length) for family in families for train_length in train_lengths],
        key=lambda item: (item.family, item.train_rows),
    )


def _resolve_target_combinations(
    *,
    requested: list[DirectCombination],
    selection_mode: str,
    summary_rows: list[dict[str, Any]],
    overwrite: bool,
) -> tuple[list[DirectCombination], list[DirectCombination]]:
    requested_set = set(requested)
    current = _existing_direct_combinations(summary_rows)
    completed = {item for item in requested if is_combination_completed(summary_rows, item)}
    if overwrite:
        completed = set()
    if selection_mode == "all":
        return requested, sorted(requested_set & completed, key=lambda item: (item.family, item.train_rows))
    if selection_mode == "same_as_current":
        selected = sorted(requested_set & current, key=lambda item: (item.family, item.train_rows))
        skipped = sorted(set(selected) & completed, key=lambda item: (item.family, item.train_rows))
        if overwrite:
            skipped = []
        return selected, skipped
    if selection_mode == "missing_only":
        selected = sorted(requested_set - completed, key=lambda item: (item.family, item.train_rows))
        skipped = sorted(requested_set & completed, key=lambda item: (item.family, item.train_rows))
        return selected, skipped
    raise ValueError(f"Unsupported selection mode: {selection_mode}")


def _train_direct_combination(
    *,
    args: argparse.Namespace,
    output_root: Path,
    combination: DirectCombination,
    time_variants: list[str],
) -> list[dict[str, Any]]:
    _log(args, 1, f"[V2 train] training direct {combination.family} {combination.train_length}")
    bundle = build_fixed_bundle(
        data_variant="full",
        train_rows=combination.train_rows,
        test_rows=CANONICAL_TEST_ROWS,
    )
    train_timestamps = [str(value) for value in bundle.get_split_timestamps("train")]
    test_timestamps = [str(value) for value in bundle.get_split_timestamps("test")]
    runner = FAMILY_RUNNERS[combination.family]
    runner_kwargs: dict[str, Any] = {
        "bundle": bundle,
        "device": resolve_device(args.device),
        "env_version": "ternary",
        "threshold": 0.0,
        "train_timestamps": train_timestamps,
        "test_timestamps": test_timestamps,
        "force_none_outside_market_hours": False,
    }
    if combination.family in {"NN", "LSTM", "TRANSFORMER", "MAMBA"} and int(getattr(args, "verbose", 1)) >= 1:
        def _progress_callback(epoch_metrics: dict[str, Any], epoch: int, total_epochs: int) -> None:
            print(
                (
                    f"[V2 train][{combination.family} {combination.train_length}] "
                    f"epoch {epoch}/{total_epochs} "
                    f"loss={float(epoch_metrics.get('loss', 0.0)):.6f} "
                    f"train_accuracy={float(epoch_metrics.get('train_accuracy', 0.0)):.4f}"
                ),
                flush=True,
            )
        runner_kwargs["progress_callback"] = _progress_callback
    if combination.family == "MAMBA":
        resolved_mamba_epochs = int(getattr(args, "mamba_epochs", 0) or 0)
        runner_kwargs["mamba_epochs"] = resolved_mamba_epochs if resolved_mamba_epochs > 0 else None
        resolved_mamba_batch_size = int(getattr(args, "mamba_batch_size", 0) or 0)
        if resolved_mamba_batch_size > 0:
            runner_kwargs["batch_size_override"] = resolved_mamba_batch_size
        runner_kwargs["disable_early_stopping"] = bool(getattr(args, "mamba_disable_early_stopping", False))
        runner_kwargs["overfit"] = bool(getattr(args, "overfit", False))
        runner_kwargs["overfit_max_epochs"] = int(getattr(args, "overfit_max_epochs", 5000))
    if combination.family == "PPO":
        runner_kwargs["ppo_total_updates"] = 5
        runner_kwargs["ppo_trajectories_per_update"] = 128
    raw_result = runner(**runner_kwargs)
    saved_model_path = _save_model_artifact(raw_result, family_saved_model_dir(output_root, combination))
    prediction_stream = build_prediction_stream_from_saved_model(
        saved_path=saved_model_path,
        eval_bundle=bundle,
        device=torch.device("cpu"),
    )
    return build_parallel_eval_rows(
        output_root=output_root,
        combination=combination,
        saved_model_path=saved_model_path,
        prediction_stream=prediction_stream,
        source_metadata={
            "source_experiment_id": EXPERIMENT_ID,
            "source_experiment_name": EXPERIMENT_NAME,
            "source_variant": "full",
        },
        train_payload=dict(raw_result.get("train") or {}),
        time_variants=time_variants,
        model_variation="BASE",
        window_length="",
        source_model=f"{combination.family}_{combination.train_length}",
        env_version="ternary",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if str(args.window_scope) not in {"non_window", "all"}:
        raise ValueError("V2 direct training only supports --window-scope non_window or all.")

    families = sorted({_normalize_family(value) for value in args.families})
    train_lengths = sorted({_normalize_train_length(value) for value in args.train_lengths}, key=train_length_to_rows)
    time_variants = []
    seen_variants: set[str] = set()
    for value in args.time_variants:
        normalized = _normalize_time_variant(value)
        if normalized not in seen_variants:
            seen_variants.add(normalized)
            time_variants.append(normalized)

    output_root = Path(args.output_dir)
    previous_summary = load_summary(output_root)
    previous_rows = load_summary_rows(output_root)
    requested = _requested_combinations(families, train_lengths)
    selected, skipped = _resolve_target_combinations(
        requested=requested,
        selection_mode=str(args.selection_mode),
        summary_rows=previous_rows,
        overwrite=bool(args.overwrite),
    )

    to_train = selected if bool(args.overwrite) else [item for item in selected if item not in set(skipped)]
    if not to_train:
        combinations_path = write_combinations_markdown(output_root, previous_summary, previous_rows)
        return {
            "summary_path": str((output_root / "full" / "summary.json").resolve()),
            "manifest_path": str((output_root / "manifest.json").resolve()),
            "combinations_path": str(combinations_path.resolve()),
            "trained_combinations": [],
            "skipped_existing": [f"{item.family}:{item.train_length}" for item in skipped],
        }

    replacement_rows: list[dict[str, Any]] = []
    for combination in to_train:
        replacement_rows.extend(
            _train_direct_combination(
                args=args,
                output_root=output_root,
                combination=combination,
                time_variants=time_variants,
            )
        )

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
    return {
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combinations_path": str(combinations_path.resolve()),
        "trained_combinations": [f"{item.family}:{item.train_length}" for item in to_train],
        "skipped_existing": [f"{item.family}:{item.train_length}" for item in skipped],
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
