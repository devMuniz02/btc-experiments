from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import experiment_reporting as reporting
from scripts import reevaluate_experiments_v2_direct_models as reeval_v2
from scripts import run_experiment_v2_selective_ensemble as selective_ensemble_v2
from scripts import run_experiments_v2_retrain_all_models as retrain_v2
from src.utils.experiment_support import (
    EXPERIMENTS_V2_ROOT,
    ENSEMBLE_TARGET_KEY,
    TARGET_ROWS_KEY,
    TIME_VARIANTS,
    direct_target_combinations_from_rows,
    load_summary,
    load_summary_rows,
    normalize_summary_row,
    normalize_summary_rows,
    normalize_window_length,
    target_row_from_model,
    target_row_prediction_output_path,
    target_row_saved_model_dir,
    target_rows_from_rows,
    write_window_train_manifests_from_rows,
    window_timeline_markdown,
    write_combinations_markdown,
    write_manifest_and_summary,
)


INTENSITY11_VARIATIONS = {"INTENSITY11", "DELTA_INTENSITY11", "DELTA_PPO_INTENSITY11"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified EXPERIMENTSV2 migration, training, reevaluation, and reporting CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate_parser = subparsers.add_parser("migrate", help="Normalize EXPERIMENTSV2 artifacts and metadata in place.")
    _add_common_args(migrate_parser)
    migrate_parser.add_argument("--rewrite-paths", action="store_true")
    migrate_parser.add_argument("--rebuild-artifacts", action="store_true")

    report_parser = subparsers.add_parser("report", help="Regenerate EXPERIMENTSV2 reports and dashboard artifacts.")
    _add_common_args(report_parser)
    report_parser.add_argument("--rebuild-artifacts", action="store_true")

    purge_parser = subparsers.add_parser("purge-window", help="Delete all window-model rows and artifacts from EXPERIMENTSV2.")
    _add_common_args(purge_parser)
    purge_parser.add_argument("--rebuild-artifacts", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train or retrain EXPERIMENTSV2 targets through a unified CLI.")
    _add_common_args(train_parser)
    train_parser.add_argument("--selection-mode", choices=retrain_v2.SUPPORTED_SELECTION_MODES, default="all")
    train_parser.add_argument("--families", nargs="+", default=[])
    train_parser.add_argument("--train-lengths", nargs="+", default=[])
    train_parser.add_argument("--window-lengths", nargs="+", default=[])
    train_parser.add_argument("--model-variations", nargs="+", default=[])
    train_parser.add_argument("--time-variants", nargs="+", default=list(TIME_VARIANTS))
    train_parser.add_argument("--envs", nargs="+", default=["ternary"])
    train_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument("--max-workers", type=int, default=4)
    train_parser.add_argument("--experiments-root", default=str(PROJECT_ROOT / "EXPERIMENTS"))
    train_parser.add_argument("--verbose", type=int, default=1)
    train_parser.add_argument("--mamba-epochs", type=int, default=0)
    train_parser.add_argument("--mamba-batch-size", type=int, default=0)
    train_parser.add_argument("--mamba-disable-early-stopping", action="store_true")
    train_parser.add_argument("--overfit", action="store_true")
    train_parser.add_argument("--overfit-max-epochs", type=int, default=5000)
    train_parser.add_argument("--visualize-test-acc", action="store_true")
    train_parser.add_argument("--bandit-strategy", choices=["ts", "ucb"], default="ts")
    train_parser.add_argument("--post-base-total-updates", type=int, default=0)
    train_parser.add_argument("--post-base-learning-rate", type=float, default=0.0)
    train_parser.add_argument("--post-base-clip-epsilon", type=float, default=0.0)
    train_parser.add_argument("--post-base-entropy-coef", type=float, default=0.0)
    train_parser.add_argument("--post-base-value-loss-coef", type=float, default=0.0)
    train_parser.add_argument("--post-base-ppo-epochs", type=int, default=0)
    train_parser.add_argument("--post-base-minibatch-size", type=int, default=0)
    train_parser.add_argument("--retrain-epochs", type=int, default=retrain_v2.window_retrain_v2.DEFAULT_RETRAIN_EPOCHS)

    reeval_parser = subparsers.add_parser("reeval", help="Reevaluate EXPERIMENTSV2 targets through a unified CLI.")
    _add_common_args(reeval_parser)
    reeval_parser.add_argument("--families", nargs="+", default=[])
    reeval_parser.add_argument("--train-lengths", nargs="+", default=[])
    reeval_parser.add_argument("--target-scope", choices=["all", "direct", "windowed"], default="all")
    reeval_parser.add_argument("--missing-only", action="store_true")
    reeval_parser.add_argument("--envs", nargs="+", default=["ternary"])
    reeval_parser.add_argument("--max-workers", type=int, default=4)

    ensemble_parser = subparsers.add_parser("ensemble-select", help="Build selective-edge ensembles from EXPERIMENTSV2 rows.")
    _add_common_args(ensemble_parser)
    ensemble_parser.add_argument("--families", nargs="+", default=list(selective_ensemble_v2.DEFAULT_FAMILIES))
    ensemble_parser.add_argument("--model-variations", nargs="+", default=list(selective_ensemble_v2.DEFAULT_VARIATIONS))
    ensemble_parser.add_argument("--search-mode", choices=["rules", "stacker", "both"], default="both")
    ensemble_parser.add_argument("--max-candidates", type=int, default=6)
    ensemble_parser.add_argument("--max-members", type=int, default=3)
    ensemble_parser.add_argument("--min-coverage", type=float, default=0.05)
    ensemble_parser.add_argument("--correlation-threshold", type=float, default=0.985)
    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", type=int, default=1)
    parser.add_argument("--output-dir", default="")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    text = str(getattr(args, "output_dir", "") or "").strip()
    if text:
        return Path(text).resolve()
    return (EXPERIMENTS_V2_ROOT / str(int(getattr(args, "experiment", 1)))).resolve()


def _normalize_envs(values: list[str]) -> list[str]:
    requested = [str(value).strip().lower() for value in values if str(value).strip()]
    if not requested:
        return ["ternary"]
    if "all" in requested:
        return ["ternary", "intensity11"]
    unique: list[str] = []
    for value in requested:
        if value not in {"ternary", "intensity11"}:
            raise ValueError(f"Unsupported env: {value}")
        if value not in unique:
            unique.append(value)
    return unique


def _target_env_from_row(row: dict[str, Any]) -> str:
    env_version = str(row.get("env_version") or "").strip().lower()
    if env_version:
        return env_version
    model_variation = str(row.get("model_variation") or "").strip().upper()
    if model_variation in INTENSITY11_VARIATIONS:
        return "intensity11"
    return "ternary"


def _artifact_basename(path: Path, row: dict[str, Any], checkpoint_name: str | None = None) -> str:
    base_name = path.stem.lower()
    base_name = re.sub(r"(^|[_-])1000(?=$|[_-])", lambda match: f"{match.group(1)}1k", base_name)
    if checkpoint_name:
        base_name = re.sub(r"_(?:w|m)\d+$", "", base_name)
        base_name = f"{base_name}_{checkpoint_name.lower()}"
    return f"{base_name}{path.suffix.lower()}"


def _move_path(source: str, destination: Path, moved_paths: dict[str, str]) -> str:
    source_path = Path(source)
    destination = destination.resolve()
    if not source_path.exists():
        return str(destination if destination.exists() else source_path)
    if str(source_path.resolve()) == str(destination):
        return str(destination)
    existing = moved_paths.get(str(source_path.resolve()))
    if existing:
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    shutil.move(str(source_path), str(destination))
    moved_paths[str(source_path.resolve())] = str(destination)
    return str(destination)


def _rewrite_row_paths(output_root: Path, row: dict[str, Any], moved_paths: dict[str, str]) -> dict[str, Any]:
    normalized = dict(row)
    target = target_row_from_model(normalized)
    threshold = int(normalized.get("threshold_pct") or 50)
    if normalized.get("predictions_path"):
        prediction_destination = target_row_prediction_output_path(output_root, target, threshold=threshold)
        rewritten = _move_path(str(normalized["predictions_path"]), prediction_destination, moved_paths)
        normalized["predictions_path"] = rewritten
        normalized["source_predictions_path"] = rewritten
    if normalized.get("saved_model_path"):
        saved_model_source = Path(str(normalized["saved_model_path"]))
        saved_model_destination = target_row_saved_model_dir(output_root, target) / _artifact_basename(saved_model_source, normalized)
        rewritten = _move_path(str(saved_model_source), saved_model_destination, moved_paths)
        normalized["saved_model_path"] = rewritten
        normalized["source_saved_model_path"] = rewritten
        normalized["grouped_saved_model_dir"] = str(saved_model_destination.parent.resolve())
    grouped_paths: list[str] = []
    checkpoints = normalized.get("window_checkpoints")
    if isinstance(checkpoints, list):
        rewritten_checkpoints = []
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            rewritten_checkpoint = dict(checkpoint)
            checkpoint_saved = str(checkpoint.get("saved_model_path") or "").strip()
            checkpoint_name = str(checkpoint.get("normalized_window_name") or checkpoint.get("window_name") or "").strip()
            if checkpoint_saved:
                checkpoint_destination = target_row_saved_model_dir(output_root, target) / _artifact_basename(
                    Path(checkpoint_saved),
                    normalized,
                    checkpoint_name=checkpoint_name or None,
                )
                rewritten_path = _move_path(checkpoint_saved, checkpoint_destination, moved_paths)
                rewritten_checkpoint["saved_model_path"] = rewritten_path
                grouped_paths.append(rewritten_path)
            rewritten_checkpoints.append(rewritten_checkpoint)
        normalized["window_checkpoints"] = rewritten_checkpoints
    if grouped_paths:
        normalized["grouped_saved_model_paths"] = grouped_paths
    elif normalized.get("saved_model_path"):
        normalized["grouped_saved_model_paths"] = [str(normalized["saved_model_path"])]
    return normalized


def _write_reports(output_root: Path, summary: dict[str, Any], rows: list[dict[str, Any]], rebuild_artifacts: bool) -> dict[str, str]:
    write_window_train_manifests_from_rows(rows)
    summary_path, manifest_file = write_manifest_and_summary(output_root, summary)
    combinations_path = write_combinations_markdown(output_root, summary, rows)
    timeline_path = output_root / "windowed_model_timeline.md"
    timeline_path.write_text(window_timeline_markdown(summary, rows), encoding="utf-8")
    table_path = output_root / "full_interactive_table.html"
    table_rows = reporting.build_table_rows(summary)
    table_html = reporting.render_html(table_rows, sort_by="daily_net_wins", title=f"EXPERIMENTSV2/{output_root.name}")
    table_path.write_text(table_html, encoding="utf-8")
    if rebuild_artifacts:
        reporting.build_v2_dashboard_artifacts(
            experiments_root=output_root.parent,
            experiments={str(output_root.name)},
            force=True,
        )
    return {
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_file.resolve()),
        "combinations_path": str(combinations_path.resolve()),
        "timeline_path": str(timeline_path.resolve()),
        "interactive_table_path": str(table_path.resolve()),
    }


def _normalized_summary_payload(output_root: Path, *, rewrite_paths: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    existing_summary = load_summary(output_root)
    existing_rows = load_summary_rows(output_root)
    moved_paths: dict[str, str] = {}
    normalized_rows: list[dict[str, Any]] = []
    for row in normalize_summary_rows(existing_rows):
        normalized = dict(row)
        if rewrite_paths:
            normalized = _rewrite_row_paths(output_root, normalized, moved_paths)
        normalized_rows.append(normalized)
    summary = dict(existing_summary)
    summary["models"] = normalized_rows
    summary["series_order"] = [row["name"] for row in normalized_rows]
    summary["direct_target_combinations"] = direct_target_combinations_from_rows(normalized_rows)
    summary[TARGET_ROWS_KEY] = target_rows_from_rows(normalized_rows)
    if ENSEMBLE_TARGET_KEY in existing_summary:
        summary[ENSEMBLE_TARGET_KEY] = json.loads(json.dumps(existing_summary.get(ENSEMBLE_TARGET_KEY, [])))
    return summary, normalized_rows, {"moved_paths": moved_paths}


def _is_window_row(row: dict[str, Any]) -> bool:
    if str(row.get("window_length") or "").strip():
        return True
    return "WINDOW" in str(row.get("model_variation") or "").upper()


def _is_window_artifact_dir(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    if "windowed" in lower_parts:
        return True
    return bool(path.is_dir() and "window" in path.name.lower())


def _purge_window_artifacts(output_root: Path) -> list[str]:
    removed_paths: list[str] = []
    full_root = output_root / "full"
    for relative in (
        Path("saved_models") / "windowed",
        Path("predictions") / "windowed",
    ):
        target = full_root / relative
        if target.exists():
            shutil.rmtree(target)
            removed_paths.append(str(target.resolve()))
    for relative in (
        Path("saved_models") / "non_window",
        Path("predictions") / "non_window",
    ):
        root_dir = full_root / relative
        if not root_dir.exists():
            continue
        candidates = sorted(
            [path for path in root_dir.rglob("*") if _is_window_artifact_dir(path)],
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            removed_paths.append(str(candidate.resolve()))
        for manifest_path in root_dir.rglob("train_window_manifest.json"):
            if not manifest_path.exists():
                continue
            manifest_path.unlink()
            removed_paths.append(str(manifest_path.resolve()))
    return sorted(set(removed_paths))


def run_purge_window(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args)
    existing_summary = load_summary(output_root)
    existing_rows = load_summary_rows(output_root)
    kept_rows = [row for row in normalize_summary_rows(existing_rows) if not _is_window_row(row)]
    removed_row_names = sorted(
        str(row.get("name") or "")
        for row in normalize_summary_rows(existing_rows)
        if _is_window_row(row) and str(row.get("name") or "").strip()
    )
    removed_artifact_paths = _purge_window_artifacts(output_root)
    summary = dict(existing_summary)
    summary["models"] = kept_rows
    summary["series_order"] = [row["name"] for row in kept_rows]
    summary["direct_target_combinations"] = direct_target_combinations_from_rows(kept_rows)
    summary[TARGET_ROWS_KEY] = target_rows_from_rows(kept_rows)
    if ENSEMBLE_TARGET_KEY in existing_summary:
        summary[ENSEMBLE_TARGET_KEY] = json.loads(json.dumps(existing_summary.get(ENSEMBLE_TARGET_KEY, [])))
    report_paths = _write_reports(output_root, summary, kept_rows, bool(args.rebuild_artifacts))
    purge_report_path = output_root / "purge_window_report.json"
    purge_report = {
        "experiment_id": int(args.experiment),
        "removed_row_count": len(removed_row_names),
        "removed_row_names": removed_row_names,
        "removed_artifact_paths": removed_artifact_paths,
        "remaining_row_count": len(kept_rows),
    }
    purge_report_path.write_text(json.dumps(purge_report, indent=2), encoding="utf-8")
    return {
        **report_paths,
        "purge_report_path": str(purge_report_path.resolve()),
        "removed_row_count": len(removed_row_names),
        "remaining_row_count": len(kept_rows),
    }


def run_migrate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args)
    summary, rows, migration_meta = _normalized_summary_payload(output_root, rewrite_paths=bool(args.rewrite_paths))
    report_paths = _write_reports(output_root, summary, rows, bool(args.rebuild_artifacts))
    migration_report_path = output_root / "migration_report.json"
    migration_report = {
        "experiment_id": int(args.experiment),
        "row_count": len(rows),
        "rewrote_paths": bool(args.rewrite_paths),
        "moved_paths": migration_meta["moved_paths"],
    }
    migration_report_path.write_text(json.dumps(migration_report, indent=2), encoding="utf-8")
    return {**report_paths, "migration_report_path": str(migration_report_path.resolve()), "row_count": len(rows)}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args)
    summary, rows, _ = _normalized_summary_payload(output_root, rewrite_paths=False)
    return _write_reports(output_root, summary, rows, bool(args.rebuild_artifacts))


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args)
    envs = _normalize_envs(args.envs)
    model_variations = [str(value).strip().upper() for value in args.model_variations if str(value).strip()]
    if envs == ["intensity11"] and not model_variations:
        model_variations = sorted(INTENSITY11_VARIATIONS)
    result = retrain_v2.run(
        argparse.Namespace(
            selection_mode=args.selection_mode,
            families=args.families,
            train_lengths=args.train_lengths,
            window_lengths=[normalize_window_length(value) for value in args.window_lengths],
            model_variations=model_variations,
            time_variants=args.time_variants,
            output_dir=str(output_root),
            experiments_root=args.experiments_root,
            device=args.device,
            overwrite=bool(args.overwrite),
            max_workers=int(args.max_workers),
            envs=envs,
            verbose=int(getattr(args, "verbose", 1)),
            mamba_epochs=int(getattr(args, "mamba_epochs", 0)),
            mamba_batch_size=int(getattr(args, "mamba_batch_size", 0)),
            mamba_disable_early_stopping=bool(getattr(args, "mamba_disable_early_stopping", False)),
            overfit=bool(getattr(args, "overfit", False)),
            overfit_max_epochs=int(getattr(args, "overfit_max_epochs", 5000)),
            visualize_test_acc=bool(getattr(args, "visualize_test_acc", False)),
            bandit_strategy=str(getattr(args, "bandit_strategy", "ts")),
            post_base_total_updates=int(getattr(args, "post_base_total_updates", 0)),
            post_base_learning_rate=float(getattr(args, "post_base_learning_rate", 0.0)),
            post_base_clip_epsilon=float(getattr(args, "post_base_clip_epsilon", 0.0)),
            post_base_entropy_coef=float(getattr(args, "post_base_entropy_coef", 0.0)),
            post_base_value_loss_coef=float(getattr(args, "post_base_value_loss_coef", 0.0)),
            post_base_ppo_epochs=int(getattr(args, "post_base_ppo_epochs", 0)),
            post_base_minibatch_size=int(getattr(args, "post_base_minibatch_size", 0)),
            retrain_epochs=int(getattr(args, "retrain_epochs", retrain_v2.window_retrain_v2.DEFAULT_RETRAIN_EPOCHS)),
        )
    )
    summary, rows, _ = _normalized_summary_payload(output_root, rewrite_paths=False)
    report_paths = _write_reports(output_root, summary, rows, rebuild_artifacts=True)
    return {**result, **report_paths}


def run_reeval(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args)
    envs = _normalize_envs(args.envs)
    result = reeval_v2.run(
        argparse.Namespace(
            output_dir=str(output_root),
            families=args.families,
            train_lengths=args.train_lengths,
            target_scope=args.target_scope,
            missing_only=bool(args.missing_only),
            max_workers=int(args.max_workers),
            envs=envs,
        )
    )
    summary, rows, _ = _normalized_summary_payload(output_root, rewrite_paths=False)
    report_paths = _write_reports(output_root, summary, rows, rebuild_artifacts=True)
    return {**result, **report_paths}


def run_ensemble_select(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_output_dir(args)
    return selective_ensemble_v2.run(
        argparse.Namespace(
            output_dir=str(output_root),
            time_variant="FULL",
            families=list(args.families),
            model_variations=list(args.model_variations),
            search_mode=str(args.search_mode),
            max_candidates=int(args.max_candidates),
            max_members=int(args.max_members),
            min_coverage=float(args.min_coverage),
            correlation_threshold=float(args.correlation_threshold),
        )
    )


def main() -> int:
    args = parse_args()
    if args.command == "migrate":
        result = run_migrate(args)
    elif args.command == "report":
        result = run_report(args)
    elif args.command == "purge-window":
        result = run_purge_window(args)
    elif args.command == "train":
        result = run_train(args)
    elif args.command == "reeval":
        result = run_reeval(args)
    elif args.command == "ensemble-select":
        result = run_ensemble_select(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
