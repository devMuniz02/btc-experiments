from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._experiment_eval_reuse import (
    build_absolute_bundle,
    build_fixed_bundle,
    build_prediction_stream_from_saved_model,
    evaluate_prediction_stream,
)
from src.utils.experiment_support import (
    CANONICAL_TEST_ROWS,
    EXPERIMENTS_V2_ROOT,
    THRESHOLDS,
    DirectCombination,
    V2TargetRow,
    build_parallel_eval_rows,
    dedupe_rows_by_name,
    direct_target_combinations_from_rows,
    load_summary,
    load_summary_rows,
    normalize_summary_rows,
    read_prediction_stream_parquet,
    target_rows_from_rows,
    target_row_saved_model_dir,
    write_prediction_stream_parquet,
    write_combinations_markdown,
    write_manifest_and_summary,
)
from scripts._helpers import write_json_atomic
from src.btc_direction_learning.dataset import load_unified_dataset_frame
from src.btc_direction_learning.env import ENV_VERSION_TERNARY


STACKER_FAMILY = "STACKER"
STACKER_TRAIN_LENGTH = "ALL"
STACKER_VARIATION = "STACKED_ENSEMBLE"
STACKER_TIME_VARIANT = "FULL"
SELECTION_SPLIT = "PRE_HOLDOUT_5K"
HOLDOUT_SPLIT = "FIXED_5K_HOLDOUT"
META_MODEL_KIND = "logistic_stacker"

DEFAULT_BASE_POOL = (
    {"family": "LSTM", "train_length": "5K", "window_length": "", "model_variation": "BASE"},
    {"family": "LSTM", "train_length": "10K", "window_length": "", "model_variation": "BASE"},
    {"family": "LSTM", "train_length": "20K", "window_length": "", "model_variation": "BASE"},
    {"family": "LSTM", "train_length": "30K", "window_length": "", "model_variation": "BASE"},
    {"family": "LSTM", "train_length": "35K", "window_length": "", "model_variation": "BASE"},
    {"family": "TRANSFORMER", "train_length": "5K", "window_length": "", "model_variation": "BASE"},
    {"family": "TRANSFORMER", "train_length": "10K", "window_length": "", "model_variation": "BASE"},
    {"family": "TRANSFORMER", "train_length": "20K", "window_length": "", "model_variation": "BASE"},
    {"family": "TRANSFORMER", "train_length": "30K", "window_length": "", "model_variation": "BASE"},
    {"family": "TRANSFORMER", "train_length": "35K", "window_length": "", "model_variation": "BASE"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned stacked ensemble inside EXPERIMENTSV2.")
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / "1"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _canonical_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("family") or "").upper(),
        str(row.get("train_length") or "").upper(),
        str(row.get("window_length") or "").upper(),
        str(row.get("model_variation") or "").upper(),
        str(row.get("time_variant") or "").upper(),
    )


def _base_spec_key(spec: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(spec["family"]).upper(),
        str(spec["train_length"]).upper(),
        str(spec["window_length"]).upper(),
        str(spec["model_variation"]).upper(),
        STACKER_TIME_VARIANT,
    )


def _representative_base_row(rows: list[dict[str, Any]], spec: dict[str, str]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in normalize_summary_rows(rows)
        if str(row.get("status") or "") == "completed" and _canonical_key(row) == _base_spec_key(spec)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: int(row.get("threshold_pct") or 9999))
    return candidates[0]


def _train_rows_for_row(row: dict[str, Any]) -> int:
    actual = int(row.get("actual_train_rows") or row.get("requested_train_rows") or 0)
    if actual > 0:
        return actual
    train_length = str(row.get("train_length") or "").strip().upper()
    if train_length.endswith("K"):
        return int(train_length[:-1]) * 1000
    return int(train_length or 0)


def _validation_train_start(train_rows: int) -> int:
    frame = load_unified_dataset_frame("full")
    validation_test_start = len(frame) - (CANONICAL_TEST_ROWS * 2)
    train_start = validation_test_start - int(train_rows)
    if train_start < 0:
        raise RuntimeError(f"Cannot build pre-holdout validation split for train_rows={train_rows}.")
    return train_start


def _prediction_stream_path(base_dir: Path, row: dict[str, Any], split_name: str) -> Path:
    base_name = str(row.get("normalized_base_name") or row.get("name") or "base").lower()
    return base_dir / f"{base_name}_{split_name}_predictions.parquet"


def _write_probability_stream(path: Path, stream: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prediction_stream_parquet(path, stream)
    return str(path.resolve())


def _load_probability_stream(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".parquet":
        return read_prediction_stream_parquet(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "timestamps": [str(value) for value in payload.get("timestamps", [])],
        "labels": [int(value) for value in payload.get("labels", [])],
        "probability_up": [float(value) for value in payload.get("probability_up", [])],
        "probability_down": [float(value) for value in payload.get("probability_down", [])],
        "probability_none": [float(value) for value in payload.get("probability_none", [])],
        "source_kind": str(payload.get("source_kind") or "saved_model"),
    }


def _build_or_reuse_stream(
    *,
    saved_model_path: str,
    bundle: Any,
    cache_path: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], str]:
    if cache_path.exists() and not overwrite:
        return _load_probability_stream(cache_path), str(cache_path.resolve())
    stream = build_prediction_stream_from_saved_model(
        saved_path=saved_model_path,
        eval_bundle=bundle,
        device=torch.device("cpu"),
    )
    return stream, _write_probability_stream(cache_path, stream)


def _assert_aligned_streams(streams: list[dict[str, Any]]) -> None:
    if not streams:
        raise RuntimeError("No streams were provided to the stacker.")
    expected_timestamps = list(streams[0].get("timestamps", []))
    expected_labels = list(streams[0].get("labels", []))
    for index, stream in enumerate(streams[1:], start=2):
        if list(stream.get("timestamps", [])) != expected_timestamps:
            raise RuntimeError(f"Stacker feature streams are timestamp-misaligned at member {index}.")
        if list(stream.get("labels", [])) != expected_labels:
            raise RuntimeError(f"Stacker feature streams are label-misaligned at member {index}.")


def _feature_schema(member_names: list[str]) -> list[str]:
    columns: list[str] = []
    for member_name in member_names:
        prefix = member_name.lower()
        columns.extend(
            [
                f"{prefix}_probability_up",
                f"{prefix}_probability_down",
                f"{prefix}_confidence_spread",
                f"{prefix}_directional_max",
            ]
        )
    columns.extend(["agreement_count", "up_vote_count", "down_vote_count"])
    return columns


def _feature_matrix(streams: list[dict[str, Any]]) -> np.ndarray:
    _assert_aligned_streams(streams)
    rows: list[list[float]] = []
    size = len(streams[0].get("timestamps", []))
    for index in range(size):
        row: list[float] = []
        up_votes = 0.0
        down_votes = 0.0
        for stream in streams:
            up_prob = float(stream["probability_up"][index])
            down_prob = float(stream["probability_down"][index])
            row.extend([up_prob, down_prob, up_prob - down_prob, max(up_prob, down_prob)])
            if up_prob > down_prob:
                up_votes += 1.0
            elif down_prob > up_prob:
                down_votes += 1.0
        row.extend([max(up_votes, down_votes), up_votes, down_votes])
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _fit_logistic_stacker(validation_streams: list[dict[str, Any]]) -> LogisticRegression:
    features = _feature_matrix(validation_streams)
    labels = np.asarray(validation_streams[0].get("labels", []), dtype=np.int64)
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(features, labels)
    return model


def _stacker_prediction_stream(model: LogisticRegression, streams: list[dict[str, Any]]) -> dict[str, Any]:
    features = _feature_matrix(streams)
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)
    aligned = np.zeros((features.shape[0], 2), dtype=np.float64)
    for index, klass in enumerate(classes):
        aligned[:, int(klass)] = probabilities[:, index]
    return {
        "timestamps": [str(value) for value in streams[0].get("timestamps", [])],
        "labels": [int(value) for value in streams[0].get("labels", [])],
        "probability_up": [float(value) for value in aligned[:, 1].tolist()],
        "probability_down": [float(value) for value in aligned[:, 0].tolist()],
        "probability_none": [0.0] * aligned.shape[0],
        "source_kind": META_MODEL_KIND,
    }


def _selection_metrics(stream: dict[str, Any]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        evaluated = evaluate_prediction_stream(
            prediction_stream=stream,
            threshold_pct=int(threshold),
            env_version=ENV_VERSION_TERNARY,
        )
        results[int(threshold)] = {
            "test": evaluated["test"],
            "portfolio": evaluated["portfolio"],
        }
    return results


def _metadata_payload(
    *,
    member_rows: list[dict[str, Any]],
    validation_paths: dict[str, str],
    feature_schema: list[str],
    requested_member_names: list[str],
    missing_member_names: list[str],
) -> dict[str, Any]:
    return {
        "family": STACKER_FAMILY,
        "train_length": STACKER_TRAIN_LENGTH,
        "model_variation": STACKER_VARIATION,
        "time_variant": STACKER_TIME_VARIANT,
        "selection_split": SELECTION_SPLIT,
        "holdout_split": HOLDOUT_SPLIT,
        "meta_model_kind": META_MODEL_KIND,
        "source_member_names": [str(row.get("normalized_base_name") or row.get("name") or "") for row in member_rows],
        "source_member_paths": [str(row.get("saved_model_path") or row.get("source_saved_model_path") or "") for row in member_rows],
        "requested_member_names": requested_member_names,
        "missing_member_names": missing_member_names,
        "validation_predictions_path": validation_paths,
        "feature_schema": feature_schema,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    summary = load_summary(output_root)
    rows = load_summary_rows(output_root)
    member_rows: list[dict[str, Any]] = []
    requested_member_names: list[str] = []
    missing_member_names: list[str] = []
    for spec in DEFAULT_BASE_POOL:
        requested_name = (
            f"{str(spec['family']).upper()}_{str(spec['train_length']).upper()}"
            f"{'_' + str(spec['window_length']).upper() if str(spec['window_length']).strip() else ''}"
            f"_{STACKER_TIME_VARIANT}_{str(spec['model_variation']).upper()}"
        )
        requested_member_names.append(requested_name)
        row = _representative_base_row(rows, spec)
        if row is None:
            missing_member_names.append(requested_name)
            continue
        member_rows.append(row)
    if len(member_rows) < 2:
        raise RuntimeError(
            "Stacker requires at least 2 completed base members. "
            f"Found {len(member_rows)} available members and {len(missing_member_names)} missing shortlist members."
        )

    stacker_target_dir = target_row_saved_model_dir(
        output_root,
        V2TargetRow(
            family=STACKER_FAMILY,
            train_length=STACKER_TRAIN_LENGTH,
            window_length="",
            model_variation=STACKER_VARIATION,
            time_variant=STACKER_TIME_VARIANT,
        ),
    )
    validation_cache_dir = stacker_target_dir / "validation_streams"
    feature_columns = _feature_schema([str(row.get("normalized_base_name") or row.get("name") or "") for row in member_rows])

    validation_streams: list[dict[str, Any]] = []
    holdout_streams: list[dict[str, Any]] = []
    validation_paths: dict[str, str] = {}
    for row in member_rows:
        train_rows = _train_rows_for_row(row)
        validation_bundle = build_absolute_bundle(
            variant="full",
            train_start_row=_validation_train_start(train_rows),
            train_rows=train_rows,
            test_rows=CANONICAL_TEST_ROWS,
        )
        holdout_bundle = build_fixed_bundle(data_variant="full", train_rows=train_rows, test_rows=CANONICAL_TEST_ROWS)
        saved_model_path = str(row.get("saved_model_path") or row.get("source_saved_model_path") or "").strip()
        if not saved_model_path:
            raise RuntimeError(f"Base member {row.get('name')} is missing saved_model_path.")
        validation_stream, cached_validation_path = _build_or_reuse_stream(
            saved_model_path=saved_model_path,
            bundle=validation_bundle,
            cache_path=_prediction_stream_path(validation_cache_dir, row, "validation"),
            overwrite=bool(args.overwrite),
        )
        holdout_stream, _ = _build_or_reuse_stream(
            saved_model_path=saved_model_path,
            bundle=holdout_bundle,
            cache_path=_prediction_stream_path(validation_cache_dir, row, "holdout"),
            overwrite=bool(args.overwrite),
        )
        member_name = str(row.get("normalized_base_name") or row.get("name") or "")
        validation_paths[member_name] = cached_validation_path
        validation_streams.append(validation_stream)
        holdout_streams.append(holdout_stream)

    model = _fit_logistic_stacker(validation_streams)
    validation_prediction_stream = _stacker_prediction_stream(model, validation_streams)
    holdout_prediction_stream = _stacker_prediction_stream(model, holdout_streams)
    validation_metrics_by_threshold = _selection_metrics(validation_prediction_stream)

    stacker_model_dir = stacker_target_dir
    stacker_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = stacker_model_dir / "stacker.pkl"
    metadata_path = stacker_model_dir / "stacker.metadata.json"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "estimator": model,
                "feature_schema": feature_columns,
                "source_member_names": [str(row.get("normalized_base_name") or row.get("name") or "") for row in member_rows],
                "meta_model_kind": META_MODEL_KIND,
            },
            handle,
        )
    metadata_payload = _metadata_payload(
        member_rows=member_rows,
        validation_paths=validation_paths,
        feature_schema=feature_columns,
        requested_member_names=requested_member_names,
        missing_member_names=missing_member_names,
    )
    write_json_atomic(metadata_path, metadata_payload)

    built_rows = build_parallel_eval_rows(
        output_root=output_root,
        combination=DirectCombination(family=STACKER_FAMILY, train_length=STACKER_TRAIN_LENGTH),
        saved_model_path=str(model_path.resolve()),
        prediction_stream=holdout_prediction_stream,
        source_metadata={
            "source_experiment_id": 1,
            "source_experiment_name": "DIRECT_FIXED_5K_V2",
            "source_variant": "full",
        },
        time_variants=[STACKER_TIME_VARIANT],
        model_variation=STACKER_VARIATION,
        window_length="",
        source_model="STACKER_LOGISTIC",
        env_version=ENV_VERSION_TERNARY,
    )

    enriched_rows: list[dict[str, Any]] = []
    for row in built_rows:
        threshold = int(row.get("threshold_pct") or 50)
        enriched = dict(row)
        enriched["train"] = dict(validation_metrics_by_threshold[threshold]["test"])
        enriched["selection_split"] = SELECTION_SPLIT
        enriched["source_member_names"] = metadata_payload["source_member_names"]
        enriched["source_member_paths"] = metadata_payload["source_member_paths"]
        enriched["requested_member_names"] = list(requested_member_names)
        enriched["missing_member_names"] = list(missing_member_names)
        enriched["validation_predictions_path"] = dict(validation_paths)
        enriched["feature_schema"] = list(feature_columns)
        enriched["meta_model_kind"] = META_MODEL_KIND
        enriched["stacker_metadata_path"] = str(metadata_path.resolve())
        enriched["grouped_saved_model_paths"] = [str(model_path.resolve()), str(metadata_path.resolve())]
        enriched["validation"] = {
            "split": SELECTION_SPLIT,
            "test": validation_metrics_by_threshold[threshold]["test"],
            "portfolio": validation_metrics_by_threshold[threshold]["portfolio"],
        }
        enriched_rows.append(enriched)

    refreshed_names = {str(row.get("name") or "") for row in enriched_rows}
    retained_rows = [row for row in rows if str(row.get("name") or "") not in refreshed_names]
    merged_rows = dedupe_rows_by_name(retained_rows + enriched_rows)
    summary["models"] = merged_rows
    summary["series_order"] = [str(row.get("name") or "") for row in merged_rows]
    summary["direct_target_combinations"] = direct_target_combinations_from_rows(merged_rows)
    summary["target_rows"] = target_rows_from_rows(merged_rows)
    summary.setdefault("selection_reports", {})
    summary["selection_reports"]["stacker"] = {
        "selection_split": SELECTION_SPLIT,
        "holdout_split": HOLDOUT_SPLIT,
        "meta_model_kind": META_MODEL_KIND,
        "source_member_names": metadata_payload["source_member_names"],
        "requested_member_names": requested_member_names,
        "missing_member_names": missing_member_names,
    }
    summary_path, manifest_path = write_manifest_and_summary(output_root, summary)
    combinations_path = write_combinations_markdown(output_root, summary, merged_rows)
    return {
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combinations_path": str(combinations_path.resolve()),
        "model_path": str(model_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "source_member_names": metadata_payload["source_member_names"],
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
