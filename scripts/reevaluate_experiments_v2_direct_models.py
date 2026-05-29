from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._experiment_eval_reuse import (
    build_fixed_bundle,
    build_prediction_stream_from_saved_model,
    load_prediction_stream_from_file,
)
from src.utils.experiment_support import (
    CANONICAL_TEST_ROWS,
    EXPERIMENTS_V2_ROOT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    DirectCombination,
    TARGET_ROWS_KEY,
    V2TargetRow,
    build_parallel_eval_rows,
    dedupe_rows_by_name,
    direct_rows,
    extra_rows,
    load_summary,
    load_summary_rows,
    merge_target_combinations,
    merge_target_rows,
    normalize_rows_window_metadata,
    normalize_train_length,
    normalize_v2_family,
    row_combination,
    target_row_from_model,
    is_combination_completed,
    is_target_completed,
    WINDOWED_BUCKET,
    write_combinations_markdown,
    write_manifest_and_summary,
)
from src.utils.experiment_support import validate_window_train_manifest_for_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reevaluate all saved EXPERIMENTSV2/1 direct models.")
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / str(EXPERIMENT_ID)))
    parser.add_argument("--families", nargs="+", default=[])
    parser.add_argument("--train-lengths", nargs="+", default=[])
    parser.add_argument("--target-scope", choices=["all", "direct", "windowed"], default="all")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--envs", nargs="+", default=["ternary"])
    return parser.parse_args()


def _filters(args: argparse.Namespace) -> tuple[set[str], set[str]]:
    families = {normalize_v2_family(value) for value in args.families} if args.families else set()
    train_lengths = {normalize_train_length(value) for value in args.train_lengths} if args.train_lengths else set()
    return families, train_lengths


def _selected_combinations(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[DirectCombination]:
    if str(getattr(args, "target_scope", "all")).lower() == "windowed":
        return []
    families, train_lengths = _filters(args)
    combinations = []
    envs = {str(value).strip().lower() for value in (getattr(args, "envs", []) or []) if str(value).strip()}
    if "all" in envs:
        envs = {"ternary", "intensity11"}
    for row in direct_rows(rows):
        if envs and str(row.get("env_version") or "ternary").strip().lower() not in envs:
            continue
        combination = row_combination(row)
        if families and combination.family not in families:
            continue
        if train_lengths and combination.train_length not in train_lengths:
            continue
        if bool(getattr(args, "missing_only", False)) and is_combination_completed(rows, combination):
            continue
        combinations.append(combination)
    unique = sorted(set(combinations), key=lambda item: (item.family, item.train_rows))
    return unique


def _selected_extra_targets(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[V2TargetRow]:
    families, train_lengths = _filters(args)
    target_scope = str(getattr(args, "target_scope", "all")).lower()
    envs = {str(value).strip().lower() for value in (getattr(args, "envs", []) or []) if str(value).strip()}
    if "all" in envs:
        envs = {"ternary", "intensity11"}
    targets: list[V2TargetRow] = []
    for row in extra_rows(rows):
        if envs and str(row.get("env_version") or "ternary").strip().lower() not in envs:
            continue
        target = target_row_from_model(row)
        if target_scope == "direct":
            continue
        if target_scope == "windowed" and target.bucket != WINDOWED_BUCKET:
            continue
        if families and target.family not in families:
            continue
        if train_lengths and target.train_length not in train_lengths:
            continue
        if bool(getattr(args, "missing_only", False)) and is_target_completed(rows, target):
            continue
        targets.append(target)
    return sorted(
        set(targets),
        key=lambda item: (
            item.family,
            item.train_rows,
            item.window_length,
            item.model_variation,
            item.time_variant,
        ),
    )


def _representative_row(rows: list[dict[str, Any]], combination: DirectCombination) -> dict[str, Any] | None:
    for row in direct_rows(rows):
        if row_combination(row) == combination and str(row.get("saved_model_path") or "").strip():
            return row
    return None


def _representative_extra_row(rows: list[dict[str, Any]], target: V2TargetRow) -> dict[str, Any] | None:
    for row in extra_rows(rows):
        if target_row_from_model(row) != target:
            continue
        predictions_path = str(row.get("predictions_path") or row.get("source_predictions_path") or "").strip()
        saved_model_path = str(row.get("saved_model_path") or row.get("source_saved_model_path") or "").strip()
        if _requires_exact_prediction_artifact(target, row):
            if predictions_path and Path(predictions_path).exists():
                return row
            continue
        if predictions_path or saved_model_path:
            return row
    return None


def _reevaluate_one(output_root: Path, source_row: dict[str, Any], max_workers: int) -> tuple[DirectCombination, list[dict[str, Any]]]:
    combination = row_combination(source_row)
    saved_model_path = str(source_row.get("saved_model_path") or "").strip()
    bundle = build_fixed_bundle(
        data_variant="full",
        train_rows=combination.train_rows,
        test_rows=CANONICAL_TEST_ROWS,
    )
    prediction_stream = build_prediction_stream_from_saved_model(
        saved_path=saved_model_path,
        eval_bundle=bundle,
        device=torch.device("cpu"),
    )
    rows = build_parallel_eval_rows(
        output_root=output_root,
        combination=combination,
        saved_model_path=saved_model_path,
        prediction_stream=prediction_stream,
        source_metadata=source_row,
        train_payload=source_row.get("train") if isinstance(source_row.get("train"), dict) else {},
        max_workers=max_workers,
        env_version=str(source_row.get("env_version") or "ternary"),
    )
    return combination, rows


def _requires_exact_prediction_artifact(target: V2TargetRow, source_row: dict[str, Any]) -> bool:
    if target.bucket == WINDOWED_BUCKET:
        return True
    return bool(source_row.get("is_aggregate_series") or source_row.get("is_aggregate"))


def _load_prediction_stream_for_row(source_row: dict[str, Any], target: V2TargetRow) -> dict[str, Any]:
    if target.bucket == WINDOWED_BUCKET:
        validate_window_train_manifest_for_row(source_row)
    predictions_path = str(source_row.get("predictions_path") or "").strip()
    if predictions_path:
        try:
            return load_prediction_stream_from_file(predictions_path)
        except Exception:
            if _requires_exact_prediction_artifact(target, source_row):
                raise FileNotFoundError(
                    f"Exact prediction artifact is unreadable for ambiguous windowed target "
                    f"{target.family}:{target.train_length}:{target.window_length}:{target.model_variation}:{target.time_variant}"
                )
    saved_model_path = str(source_row.get("saved_model_path") or "").strip()
    if _requires_exact_prediction_artifact(target, source_row):
        raise FileNotFoundError(
            f"Exact prediction artifact is required for ambiguous windowed target "
            f"{target.family}:{target.train_length}:{target.window_length}:{target.model_variation}:{target.time_variant}"
        )
    if not saved_model_path:
        raise FileNotFoundError(f"No reusable prediction source for row {source_row.get('name')}")
    train_rows = int(source_row.get("actual_train_rows") or source_row.get("requested_train_rows") or CANONICAL_TEST_ROWS)
    bundle = build_fixed_bundle(
        data_variant="full",
        train_rows=train_rows,
        test_rows=CANONICAL_TEST_ROWS,
    )
    return build_prediction_stream_from_saved_model(
        saved_path=saved_model_path,
        eval_bundle=bundle,
        device=torch.device("cpu"),
    )


def _reevaluate_extra_one(output_root: Path, source_row: dict[str, Any], max_workers: int) -> tuple[V2TargetRow, list[dict[str, Any]]]:
    target = target_row_from_model(source_row)
    prediction_stream = _load_prediction_stream_for_row(source_row, target)
    saved_model_path = str(source_row.get("saved_model_path") or source_row.get("source_saved_model_path") or "").strip()
    rows = build_parallel_eval_rows(
        output_root=output_root,
        combination=DirectCombination(family=target.family, train_length=target.train_length),
        saved_model_path=saved_model_path,
        prediction_stream=prediction_stream,
        source_metadata=source_row,
        train_payload=source_row.get("train") if isinstance(source_row.get("train"), dict) else {},
        time_variants=[target.time_variant],
        max_workers=max_workers,
        model_variation=target.model_variation,
        window_length=target.window_length,
        artifact_bucket=target.bucket,
        source_model=str(source_row.get("source_model") or target.family),
        env_version=str(source_row.get("env_version") or "ternary"),
    )
    return target, rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    summary = load_summary(output_root)
    rows = load_summary_rows(output_root)
    combinations = _selected_combinations(rows, args)
    extra_targets = _selected_extra_targets(rows, args)
    skipped: list[str] = []
    replacement_rows: list[dict[str, Any]] = []

    print(
        f"[EXPV2] Reevaluating {len(combinations)} direct combinations and {len(extra_targets)} expanded targets from {output_root}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        future_map = {}
        for combination in combinations:
            source_row = _representative_row(rows, combination)
            if source_row is None:
                skipped.append(f"{combination.family}:{combination.train_length}")
                continue
            future = executor.submit(_reevaluate_one, output_root, source_row, int(args.max_workers))
            future_map[future] = combination
        for target in extra_targets:
            source_row = _representative_extra_row(rows, target)
            if source_row is None:
                skipped.append(f"{target.family}:{target.train_length}:{target.window_length}:{target.model_variation}:{target.time_variant}")
                continue
            future = executor.submit(_reevaluate_extra_one, output_root, source_row, int(args.max_workers))
            future_map[future] = target
        for future in future_map:
            try:
                target_or_combination, built_rows = future.result()
            except FileNotFoundError as exc:
                skipped.append(str(exc))
                continue
            if isinstance(target_or_combination, DirectCombination):
                print(
                    f"[EXPV2] Reevaluated {target_or_combination.family}:{target_or_combination.train_length} -> {len(built_rows)} rows",
                    flush=True,
                )
            else:
                print(
                    f"[EXPV2] Reevaluated {target_or_combination.family}:{target_or_combination.train_length}:{target_or_combination.window_length}:{target_or_combination.model_variation}:{target_or_combination.time_variant} -> {len(built_rows)} rows",
                    flush=True,
                )
            replacement_rows.extend(built_rows)

    replacement_names = {str(row.get("name") or "") for row in replacement_rows if row.get("name")}
    preserved_rows = [row for row in rows if str(row.get("name") or "") not in replacement_names]
    merged_rows = dedupe_rows_by_name(normalize_rows_window_metadata(preserved_rows + replacement_rows))
    summary["experiment_id"] = EXPERIMENT_ID
    summary["name"] = EXPERIMENT_NAME
    summary["series_order"] = [row["name"] for row in merged_rows]
    summary["models"] = merged_rows
    summary["direct_target_combinations"] = merge_target_combinations(
        summary.get("direct_target_combinations", []),
        combinations,
    )
    summary[TARGET_ROWS_KEY] = merge_target_rows(
        summary.get(TARGET_ROWS_KEY, []),
        extra_targets,
    )
    summary_path, manifest_path = write_manifest_and_summary(output_root, summary)
    combinations_path = write_combinations_markdown(output_root, summary, merged_rows)
    return {
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combinations_path": str(combinations_path.resolve()),
        "reevaluated_combinations": [f"{item.family}:{item.train_length}" for item in combinations]
        + [
            f"{item.family}:{item.train_length}:{item.window_length}:{item.model_variation}:{item.time_variant}"
            for item in extra_targets
        ],
        "skipped": skipped,
        "row_count": len(merged_rows),
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
