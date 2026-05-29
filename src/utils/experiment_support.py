from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts._helpers import load_json, write_json_atomic
from scripts._experiment_eval_reuse import evaluate_prediction_stream


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_V2_ROOT = PROJECT_ROOT / "EXPERIMENTSV2"
EXPERIMENT_ID = 1
EXPERIMENT_NAME = "DIRECT_FIXED_5K_V2"
VARIANT_NAME = "full"
SUPPORTED_FAMILIES = ("BC", "DAGGER", "NN", "RF", "XGBOOST", "LSTM", "TRANSFORMER", "MAMBA")
BASELINE_FAMILIES = {"RANDOM", "ALWAYS_UP", "ALWAYS_DOWN"}
TARGET_ROWS_KEY = "target_rows"
IGNORED_SOURCE_EXPERIMENT_IDS = {5, 14, 15, 17, 18, 21}
EXP23_SOURCE_EXPERIMENT_ID = 23
TIME_VARIANTS = ("FULL", "MARKET_HOURS", "OUTSIDE_MARKET_HOURS")
THRESHOLDS = (50, 90, 95, 99, 995)
CANONICAL_TEST_ROWS = 5000
SOURCE_PROVENANCE_BY_TIME_VARIANT = {
    "FULL": "FULL_SOURCE",
    "MARKET_HOURS": "DERIVED_MARKET_HOURS_FROM_FULL",
    "OUTSIDE_MARKET_HOURS": "DERIVED_OUTSIDE_MARKET_HOURS_FROM_FULL",
}

WINDOW_TRAIN_MANIFEST_FILENAME = "train_window_manifest.json"


def parse_timestamp_to_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text[:10]


def group_dir_from_row(row: dict[str, Any]) -> Path | None:
    group_dir = str(row.get("grouped_saved_model_dir") or "").strip()
    if group_dir:
        return Path(group_dir).resolve()
    saved_model_path = str(row.get("saved_model_path") or row.get("source_saved_model_path") or "").strip()
    if saved_model_path:
        return Path(saved_model_path).resolve().parent
    return None


def manifest_path_for_group_dir(group_dir: str | Path) -> Path:
    return Path(group_dir).resolve() / WINDOW_TRAIN_MANIFEST_FILENAME


def normalize_window_checkpoints(checkpoints: Any) -> list[dict[str, Any]]:
    if not isinstance(checkpoints, list):
        return []

    def sort_key(entry: dict[str, Any]) -> tuple[str, int]:
        timestamps = entry.get("test_timestamps") if isinstance(entry, dict) else []
        if isinstance(timestamps, list) and timestamps:
            first_timestamp = str(timestamps[0] or "")
        else:
            first_timestamp = str(
                entry.get("window_test_start")
                or entry.get("test_start")
                or entry.get("start")
                or ""
            )
        try:
            numeric_index = int(entry.get("window_index"))
        except Exception:
            numeric_index = 10**9
        return (first_timestamp, numeric_index)

    normalized_rows: list[dict[str, Any]] = []
    for normalized_index, checkpoint in enumerate(
        sorted((item for item in checkpoints if isinstance(item, dict)), key=sort_key),
        start=1,
    ):
        normalized_window_name = f"W{normalized_index}"
        normalized_rows.append(
            {
                **checkpoint,
                "window_index": normalized_index,
                "normalized_window_index": normalized_index,
                "window_name": normalized_window_name,
                "normalized_window_name": normalized_window_name,
            }
        )
    return normalized_rows


def window_eval_bounds(checkpoint: dict[str, Any]) -> tuple[str, str]:
    timestamps = checkpoint.get("test_timestamps")
    if isinstance(timestamps, list) and timestamps:
        return parse_timestamp_to_date(timestamps[0]), parse_timestamp_to_date(timestamps[-1])
    start = checkpoint.get("window_test_start") or checkpoint.get("test_start") or checkpoint.get("start")
    end = checkpoint.get("window_test_end") or checkpoint.get("test_end") or checkpoint.get("end") or start
    return parse_timestamp_to_date(start), parse_timestamp_to_date(end)


def window_train_end_date(checkpoint: dict[str, Any]) -> str:
    return parse_timestamp_to_date(
        checkpoint.get("window_train_end")
        or checkpoint.get("train_end")
        or checkpoint.get("train_end_timestamp")
    )


def base_train_end_date_from_row(row: dict[str, Any]) -> str:
    if str(row.get("base_train_end_date") or "").strip():
        return parse_timestamp_to_date(row.get("base_train_end_date"))
    if str(row.get("model_used_train_end_date") or "").strip():
        return parse_timestamp_to_date(row.get("model_used_train_end_date"))
    train_timestamps = ((row.get("train") or {}).get("timestamps") or [])
    if isinstance(train_timestamps, list) and train_timestamps:
        return parse_timestamp_to_date(train_timestamps[-1])
    return ""


def build_window_train_manifest(
    *,
    source_model: str,
    family: str,
    train_length: str,
    window_length: str,
    model_variation: str,
    entries: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "source_model": source_model,
        "family": family,
        "train_length": train_length,
        "window_length": window_length,
        "model_variation": model_variation,
        "entries": entries,
    }


def build_window_train_manifest_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    checkpoints = normalize_window_checkpoints(row.get("window_checkpoints"))
    if not checkpoints:
        return None
    base_saved_model_path = str(row.get("saved_model_path") or row.get("source_saved_model_path") or "").strip()
    base_train_end_date = base_train_end_date_from_row(row)
    if not base_saved_model_path or not base_train_end_date:
        return None
    entries = [
        {
            "role": "BASE",
            "saved_model_path": str(Path(base_saved_model_path).resolve()),
            "train_end_date": base_train_end_date,
            "window_name": "",
        }
    ]
    for checkpoint in checkpoints:
        saved_model_path = str(checkpoint.get("saved_model_path") or "").strip()
        train_end_date = window_train_end_date(checkpoint)
        window_name = str(checkpoint.get("normalized_window_name") or checkpoint.get("window_name") or "").strip()
        if not saved_model_path or not train_end_date or not window_name:
            return None
        entries.append(
            {
                "role": window_name,
                "saved_model_path": str(Path(saved_model_path).resolve()),
                "train_end_date": train_end_date,
                "window_name": window_name,
            }
        )
    return build_window_train_manifest(
        source_model=str(row.get("source_model") or row.get("family") or ""),
        family=str(row.get("family") or ""),
        train_length=str(row.get("train_length") or ""),
        window_length=str(row.get("window_length") or ""),
        model_variation=str(row.get("model_variation") or ""),
        entries=entries,
    )


def write_window_train_manifest(group_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path_for_group_dir(group_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, manifest)
    return path


def write_window_train_manifest_from_row(row: dict[str, Any]) -> Path | None:
    group_dir = group_dir_from_row(row)
    if group_dir is None:
        return None
    manifest = build_window_train_manifest_from_row(row)
    if manifest is None:
        return None
    return write_window_train_manifest(group_dir, manifest)


def backfill_window_train_manifests(rows: list[dict[str, Any]]) -> list[str]:
    written_paths: list[str] = []
    seen_group_dirs: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_dir = group_dir_from_row(row)
        if group_dir is None:
            continue
        key = str(group_dir)
        if key in seen_group_dirs:
            continue
        seen_group_dirs.add(key)
        manifest_path = write_window_train_manifest_from_row(row)
        if manifest_path is not None:
            written_paths.append(str(manifest_path.resolve()))
    return written_paths


def load_window_train_manifest_for_row(row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    group_dir = group_dir_from_row(row)
    if group_dir is None:
        raise RuntimeError(f"Window model row {row.get('name')} is missing grouped saved model metadata.")
    manifest_path = manifest_path_for_group_dir(group_dir)
    if not manifest_path.exists():
        raise RuntimeError(f"Window train manifest is missing for {row.get('name')} at {manifest_path}.")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Window train manifest at {manifest_path} is invalid.")
    return manifest, manifest_path


def validate_window_train_manifest_for_row(row: dict[str, Any]) -> Path:
    manifest, manifest_path = load_window_train_manifest_for_row(row)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"Window train manifest at {manifest_path} has no entries.")
    entry_by_role = {
        str(entry.get("role") or "").strip().upper(): entry
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("role") or "").strip()
    }
    checkpoints = normalize_window_checkpoints(row.get("window_checkpoints"))
    if not checkpoints:
        raise RuntimeError(f"Window model row {row.get('name')} is missing normalized checkpoints.")
    for index, checkpoint in enumerate(checkpoints, start=1):
        role = "BASE" if index == 1 else f"W{index - 1}"
        entry = entry_by_role.get(role)
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Window train manifest at {manifest_path} is missing the {role} entry for {row.get('name')}."
            )
        saved_model_path = str(entry.get("saved_model_path") or "").strip()
        train_end_date = parse_timestamp_to_date(entry.get("train_end_date"))
        eval_start_date, _ = window_eval_bounds(checkpoint)
        if not saved_model_path:
            raise RuntimeError(f"Window train manifest at {manifest_path} is missing saved_model_path for {role}.")
        if not Path(saved_model_path).exists():
            raise RuntimeError(
                f"Window train manifest at {manifest_path} points to a missing model for {role}: {saved_model_path}"
            )
        if not train_end_date:
            raise RuntimeError(f"Window train manifest at {manifest_path} is missing train_end_date for {role}.")
        if not eval_start_date:
            raise RuntimeError(f"Window checkpoint metadata is missing test start date for {row.get('name')}.")
        if eval_start_date <= train_end_date:
            raise RuntimeError(
                f"Window leakage guard failed for {row.get('name')}: {role} train_end_date={train_end_date} "
                f"is not earlier than eval_start_date={eval_start_date}."
            )
    return manifest_path
ENSEMBLE_TRAIN_LENGTH = "ALL"
ENSEMBLE_TARGET_KEY = "ensemble_target_families"
WINDOWED_BUCKET = "windowed"
NON_WINDOW_BUCKET = "non_window"
EXTRA_VARIATION_TOKENS = (
    "PPO",
    "PPO_FULL",
    "PPO_HYBRID",
    "PPO_FULL_WINDOW_CONTINUE",
    "PPO_HYBRID_WINDOW_CONTINUE",
    "WINDOW",
    "RETRAIN",
    "CONTINUE",
    "ACTOR_CRITIC",
    "BANDITS",
    "BANDITS_TS",
    "BANDITS_UCB",
    "ACTOR_CRITIC_WINDOW_CONTINUE",
    "BANDITS_WINDOW_CONTINUE",
    "BANDITS_TS_WINDOW_CONTINUE",
    "BANDITS_UCB_WINDOW_CONTINUE",
    "INTENSITY11",
    "DELTA_INTENSITY11",
    "DELTA_PPO_INTENSITY11",
    "STACKED_ENSEMBLE",
)
PRESERVED_WINDOW_METADATA_KEYS = (
    "window_checkpoints",
    "base_train_end_date",
    "normalized_window_name",
    "window_name",
    "normalized_window_index",
    "original_window_index",
    "original_series_name",
    "normalized_series_name",
)
INTEGRITY_INVALID_EXIT_CODE = 10
SUMMARY_ROWS_FILE_NAME = "summary_rows.parquet"
PREDICTION_STREAM_COLUMNS = (
    "timestamp",
    "label",
    "action",
    "chosen_action_probability",
    "probability_up",
    "probability_down",
    "probability_none",
    "source_kind",
)


@dataclass(frozen=True)
class DirectCombination:
    family: str
    train_length: str

    @property
    def train_rows(self) -> int:
        return train_length_to_rows(self.train_length)

    @property
    def folder_name(self) -> str:
        return f"{self.family}_{self.train_length}_BASE".lower()


@dataclass(frozen=True)
class V2TargetRow:
    family: str
    train_length: str
    window_length: str
    model_variation: str
    time_variant: str

    @property
    def bucket(self) -> str:
        return WINDOWED_BUCKET if self.window_length or "WINDOW" in self.model_variation else NON_WINDOW_BUCKET

    @property
    def train_rows(self) -> int:
        return train_length_to_rows(self.train_length)

    @property
    def folder_name(self) -> str:
        parts = [self.family, self.train_length]
        if self.window_length:
            parts.append(self.window_length)
        parts.append(self.model_variation)
        return "_".join(parts).lower()

    @property
    def base_name(self) -> str:
        parts = [self.family, self.train_length]
        if self.window_length:
            parts.append(self.window_length)
        parts.append(self.time_variant)
        parts.append(self.model_variation)
        return "_".join(parts)


@dataclass(frozen=True)
class RetrainableTarget:
    source_experiment_id: int
    source_experiment_name: str
    source_variant: str
    source_name: str
    family: str
    train_length: str
    window_length: str
    model_variation: str

    @property
    def bucket(self) -> str:
        if self.window_length or "WINDOW" in self.model_variation:
            return WINDOWED_BUCKET
        return NON_WINDOW_BUCKET

    @property
    def train_rows(self) -> int:
        return train_length_to_rows(self.train_length)

    @property
    def folder_name(self) -> str:
        parts = [self.family, self.train_length]
        if self.window_length:
            parts.append(self.window_length)
        parts.append(self.model_variation)
        return "_".join(parts).lower()

    @property
    def key(self) -> tuple[int, str, str, str, str, str]:
        return (
            self.source_experiment_id,
            self.family,
            self.train_length,
            self.window_length,
            self.model_variation,
            self.source_name,
        )


def normalize_train_length(value: str | int) -> str:
    text = str(value).strip().upper()
    if not text:
        raise ValueError("Train length cannot be empty.")
    if text == ENSEMBLE_TRAIN_LENGTH:
        return ENSEMBLE_TRAIN_LENGTH
    if text.endswith("K"):
        number = int(text[:-1])
        if number <= 0:
            raise ValueError(f"Invalid train length: {value}")
        return f"{number}K"
    number = int(text)
    if number <= 0:
        raise ValueError(f"Invalid train length: {value}")
    if number < 1000:
        return f"{number}K"
    if number % 1000 == 0:
        return f"{number // 1000}K"
    raise ValueError(f"Unsupported train length format: {value}")


def train_length_to_rows(train_length: str) -> int:
    normalized = normalize_train_length(train_length)
    if normalized == ENSEMBLE_TRAIN_LENGTH:
        return 10**12
    return int(normalized[:-1]) * 1000


def normalize_family(value: str) -> str:
    family = str(value).strip().upper()
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported v2 family: {value}")
    return family


def normalize_v2_family(value: str) -> str:
    family = str(value).strip().upper()
    if not family:
        raise ValueError("V2 family cannot be empty.")
    return family


def normalize_ensemble_family(value: str) -> str:
    text = str(value).strip().upper()
    if text.startswith("ENSEMBLE"):
        base = text[len("ENSEMBLE") :]
    else:
        base = text
    base_family = normalize_family(base)
    return f"ENSEMBLE{base_family}"


def normalize_window_length(value: str | int | None) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.endswith("K"):
        return normalize_train_length(text)
    try:
        numeric = int(float(text))
    except (TypeError, ValueError):
        return text
    if numeric <= 0:
        return text
    if numeric >= 1000 and numeric % 1000 == 0:
        return f"{numeric // 1000}K"
    return str(numeric)


def normalize_model_variation(value: str | None) -> str:
    text = str(value or "BASE").strip().upper()
    return text or "BASE"


def time_variant_predicate(time_variant: str):
    normalized = str(time_variant or "FULL").strip().upper()
    if normalized == "FULL":
        return None
    if normalized == "MARKET_HOURS":
        from scripts._experiment_eval_reuse import is_weekday_market_hours_timestamp

        return is_weekday_market_hours_timestamp
    if normalized == "OUTSIDE_MARKET_HOURS":
        from scripts._experiment_eval_reuse import is_outside_weekday_market_hours_timestamp

        return is_outside_weekday_market_hours_timestamp
    raise ValueError(f"Unsupported time variant: {time_variant}")


def is_ignored_source_row(row: dict[str, Any]) -> bool:
    return int(row.get("source_experiment_id") or 0) in IGNORED_SOURCE_EXPERIMENT_IDS


def is_derived_v2_row(row: dict[str, Any]) -> bool:
    family = str(row.get("family") or "").strip().upper()
    if family.startswith("ENSEMBLE"):
        return True
    source_experiment_id = int(row.get("source_experiment_id") or 0)
    return source_experiment_id <= 0


def source_backed_v2_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and not is_derived_v2_row(row)]


def derived_v2_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and is_derived_v2_row(row)]


def is_baseline_family_name(value: str | None) -> bool:
    return str(value or "").strip().upper() in BASELINE_FAMILIES


def infer_window_length_from_row(row: dict[str, Any]) -> str:
    existing = normalize_window_length(row.get("window_length"))
    if existing:
        return existing
    checkpoints = row.get("window_checkpoints")
    if isinstance(checkpoints, list) and checkpoints:
        lengths = set()
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            explicit = normalize_window_length(
                checkpoint.get("window_size")
                or checkpoint.get("dataset_metadata", {}).get("test_rows")
                if isinstance(checkpoint.get("dataset_metadata"), dict)
                else checkpoint.get("window_size")
            )
            if explicit:
                lengths.add(explicit)
                continue
            timestamps = checkpoint.get("test_timestamps")
            if isinstance(timestamps, list) and timestamps:
                lengths.add(normalize_window_length(len(timestamps)))
        lengths.discard("")
        if len(lengths) == 1:
            return next(iter(lengths))
    candidates = [
        str(row.get("name") or ""),
        str(row.get("normalized_base_name") or ""),
        str(row.get("source_model") or ""),
    ]
    for raw_text in candidates:
        text = raw_text.strip().upper()
        if not text:
            continue
        match = re.search(r"WINDOW(\d+K?|\d+)", text)
        if match:
            return normalize_window_length(match.group(1))
        match = re.match(
            r"^[A-Z0-9]+_[0-9]+K?_([0-9]+K?|\d+)_(FULL|MARKET_HOURS|OUTSIDE_MARKET_HOURS)_",
            text,
        )
        if match:
            return normalize_window_length(match.group(1))
    source_experiment_id = int(row.get("source_experiment_id") or 0)
    if source_experiment_id in {3, 6}:
        return "1K"
    if source_experiment_id in {10, 11}:
        return "500"
    return ""


def infer_window_name_from_row(row: dict[str, Any]) -> str:
    existing = str(row.get("normalized_window_name") or row.get("window_name") or "").strip()
    if existing and existing.upper() != "W1":
        return existing
    source_experiment_id = int(row.get("source_experiment_id") or 0)
    model_variation = normalize_model_variation(row.get("model_variation"))
    window_length = infer_window_length_from_row(row)
    if window_length == "1K" and model_variation == "PPO_WINDOW_CONTINUE" and source_experiment_id in {3, 6, 7}:
        return "W1-W5"
    return ""


def _normalized_window_name_from_checkpoints(checkpoints: list[dict[str, Any]]) -> str:
    if not checkpoints:
        return ""
    first = str(checkpoints[0].get("normalized_window_name") or checkpoints[0].get("window_name") or "").strip()
    last = str(checkpoints[-1].get("normalized_window_name") or checkpoints[-1].get("window_name") or "").strip()
    if not first:
        return ""
    if not last or last == first:
        return ""
    return f"{first}-{last}"


def normalize_windowed_row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    target = target_row_from_model(normalized)
    is_windowed = target.bucket == WINDOWED_BUCKET
    had_window_name = bool(
        str(normalized.get("normalized_window_name") or normalized.get("window_name") or "").strip()
    )
    had_window_checkpoints = False
    checkpoints = normalized.get("window_checkpoints")
    if isinstance(checkpoints, list) and checkpoints:
        had_window_checkpoints = True
        renormalized = normalize_window_checkpoints(checkpoints)
        if renormalized:
            normalized["window_checkpoints"] = renormalized
            normalized_name = _normalized_window_name_from_checkpoints(renormalized)
            if normalized_name:
                normalized["normalized_window_name"] = normalized_name
                normalized["window_name"] = normalized_name
                is_windowed = True
    if is_windowed:
        window_length = infer_window_length_from_row(normalized)
        if window_length:
            normalized["window_length"] = window_length
        allow_default_window_name = True
        if target.model_variation == "BASE" and not had_window_checkpoints:
            normalized.pop("normalized_window_name", None)
            normalized.pop("window_name", None)
            allow_default_window_name = False
        window_name = str(
            normalized.get("normalized_window_name")
            or normalized.get("window_name")
            or ""
        ).strip()
        if window_name.upper() == "W1":
            window_name = ""
        if not window_name:
            window_name = infer_window_name_from_row(normalized)
        if window_name:
            normalized["normalized_window_name"] = window_name
            normalized["window_name"] = window_name
        else:
            normalized.pop("normalized_window_name", None)
            normalized.pop("window_name", None)
    return normalized


def normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    had_normalized_base_name = "normalized_base_name" in row
    normalized = normalize_windowed_row_metadata(row)
    normalized["family"] = normalize_v2_family(str(normalized.get("family") or ""))
    normalized["train_length"] = normalize_train_length(str(normalized.get("train_length") or ""))
    normalized["window_length"] = normalize_window_length(normalized.get("window_length"))
    normalized["model_variation"] = normalize_model_variation(normalized.get("model_variation"))
    normalized["time_variant"] = str(normalized.get("time_variant") or "FULL").strip().upper()
    target = target_row_from_model(normalized)
    threshold = int(normalized.get("threshold_pct") or 50)
    normalized_base_name = target.base_name
    normalized["normalized_base_name"] = normalized_base_name
    existing_name = str(normalized.get("name") or "").strip()
    structured_name = bool(
        existing_name
        and re.match(
            r"^[A-Z0-9]+_[0-9]+K?(?:_[0-9]+K?)?_(?:FULL|MARKET_HOURS|OUTSIDE_MARKET_HOURS)_[A-Z0-9_]+_\d+$",
            existing_name.upper(),
        )
    )
    if not existing_name or structured_name or had_normalized_base_name:
        normalized["name"] = f"{normalized_base_name}_{threshold}"
    checkpoints = normalized.get("window_checkpoints")
    if isinstance(checkpoints, list) and checkpoints:
        normalized["window_checkpoints"] = normalize_window_checkpoints(checkpoints)
        normalized_name = _normalized_window_name_from_checkpoints(normalized["window_checkpoints"])
        if normalized_name:
            normalized["window_name"] = normalized_name
            normalized["normalized_window_name"] = normalized_name
    return normalized


def normalize_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows = [normalize_summary_row(row) for row in rows if isinstance(row, dict)]
    return dedupe_rows_by_name(finalized_v2_rows(normalized_rows))


def has_invalid_single_window_name(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    target = target_row_from_model(row)
    if target.bucket != WINDOWED_BUCKET:
        return False
    window_name = str(row.get("normalized_window_name") or row.get("window_name") or "").strip().upper()
    return window_name == "W1"


def is_invalid_static_base_window_row(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    family = str(row.get("family") or "").strip().upper()
    if not family or family.startswith("ENSEMBLE"):
        return False
    if normalize_model_variation(row.get("model_variation")) != "BASE":
        return False
    if not normalize_window_length(row.get("window_length")):
        return False
    checkpoints = row.get("window_checkpoints")
    return not (isinstance(checkpoints, list) and checkpoints)


def is_source_backed_exp23_row(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict) or is_derived_v2_row(row):
        return False
    return int(row.get("source_experiment_id") or 0) == EXP23_SOURCE_EXPERIMENT_ID


def normalized_run_identity(row: dict[str, Any]) -> tuple[str, str, str, str, str, int]:
    return (
        normalize_v2_family(str(row.get("family") or "")),
        normalize_train_length(str(row.get("train_length") or "")),
        normalize_window_length(row.get("window_length")),
        normalize_model_variation(row.get("model_variation")),
        str(row.get("time_variant") or "FULL").strip().upper(),
        int(row.get("threshold_pct") or 0),
    )


def normalized_run_source_identity(row: dict[str, Any]) -> tuple[str, str, str, str, str, int, int, str]:
    source_name = str(
        row.get("source_model")
        or row.get("source_name")
        or row.get("original_series_name")
        or row.get("name")
        or ""
    ).strip().upper()
    return (
        *normalized_run_identity(row),
        int(row.get("source_experiment_id") or 0),
        source_name,
    )


def has_mixed_legacy_window_naming(row: dict[str, Any]) -> bool:
    checkpoints = row.get("window_checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        return False
    for index, checkpoint in enumerate(checkpoints, start=1):
        if not isinstance(checkpoint, dict):
            return True
        expected = f"W{index}"
        actual = str(
            checkpoint.get("normalized_window_name")
            or checkpoint.get("window_name")
            or ""
        ).strip().upper()
        if actual != expected:
            return True
    return False


def finalized_v2_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if isinstance(row, dict) and not is_invalid_static_base_window_row(row)
    ]


def normalize_rows_window_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_summary_row(row) if isinstance(row, dict) else row for row in rows]

def window_timeline_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[str]] = {}
    seen_sections: set[str] = set()
    for row in normalize_summary_rows(rows):
        target = target_row_from_model(row)
        checkpoints = row.get("window_checkpoints")
        if target.bucket != WINDOWED_BUCKET or not isinstance(checkpoints, list) or not checkpoints:
            continue
        section_name = " | ".join(
            [
                target.family,
                target.train_length,
                target.window_length or "-",
                target.model_variation,
                str(row.get("env_version") or "ternary"),
            ]
        )
        if section_name in seen_sections:
            continue
        seen_sections.add(section_name)
        lines = []
        sorted_checkpoints = normalize_window_checkpoints(checkpoints)
        prior_model_train_end = base_train_end_date_from_row(row)
        for index, checkpoint in enumerate(sorted_checkpoints, start=1):
            eval_start, eval_end = window_eval_bounds(checkpoint)
            model_used = "BASE" if index == 1 else f"W{index - 1}"
            train_end = prior_model_train_end or eval_start
            lines.append(
                f"- EVAL {index} start_date={eval_start} end_date={eval_end} model_used={model_used} model_used_train_end_date={train_end}"
            )
            prior_model_train_end = window_train_end_date(checkpoint)
        grouped[section_name] = lines
    output = ["# EXPERIMENTSV2 Window Timeline", ""]
    for section_name in sorted(grouped):
        output.append(f"## {section_name}")
        output.append("")
        output.extend(grouped[section_name])
        output.append("")
    return "\n".join(output).rstrip() + "\n"


def summary_path(output_root: Path) -> Path:
    return output_root / VARIANT_NAME / "summary.json"


def summary_rows_path(output_root: Path) -> Path:
    return output_root / VARIANT_NAME / SUMMARY_ROWS_FILE_NAME


def manifest_path(output_root: Path) -> Path:
    return output_root / "manifest.json"


def saved_models_root(output_root: Path, bucket: str = NON_WINDOW_BUCKET) -> Path:
    return output_root / VARIANT_NAME / "saved_models" / bucket


def predictions_root(output_root: Path, bucket: str = NON_WINDOW_BUCKET) -> Path:
    return output_root / VARIANT_NAME / "predictions" / bucket


def family_saved_model_dir(output_root: Path, combination: DirectCombination) -> Path:
    return saved_models_root(output_root, NON_WINDOW_BUCKET) / combination.family.lower() / combination.folder_name


def family_predictions_dir(output_root: Path, combination: DirectCombination) -> Path:
    return predictions_root(output_root, NON_WINDOW_BUCKET) / combination.family.lower() / combination.folder_name


def prediction_output_path(
    output_root: Path,
    combination: DirectCombination,
    time_variant: str,
    *,
    threshold: int | None = None,
) -> Path:
    base_name = f"{combination.family}_{combination.train_length}_{time_variant}_BASE".lower()
    suffix = f"_{int(threshold)}" if threshold is not None else ""
    return family_predictions_dir(output_root, combination) / f"{base_name}{suffix}_predictions.parquet"


def target_row_from_model(row: dict[str, Any]) -> V2TargetRow:
    return V2TargetRow(
        family=normalize_v2_family(str(row.get("family") or "")),
        train_length=normalize_train_length(str(row.get("train_length") or "")),
        window_length=normalize_window_length(row.get("window_length")),
        model_variation=normalize_model_variation(row.get("model_variation")),
        time_variant=str(row.get("time_variant") or "FULL").strip().upper(),
    )


def target_row_saved_model_dir(output_root: Path, target: V2TargetRow) -> Path:
    return saved_models_root(output_root, target.bucket) / target.family.lower() / target.folder_name


def target_row_predictions_dir(output_root: Path, target: V2TargetRow) -> Path:
    return predictions_root(output_root, target.bucket) / target.family.lower() / target.folder_name


def target_row_prediction_output_path(output_root: Path, target: V2TargetRow, *, threshold: int | None = None) -> Path:
    suffix = f"_{int(threshold)}" if threshold is not None else ""
    return target_row_predictions_dir(output_root, target) / f"{target.base_name.lower()}{suffix}_predictions.parquet"


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, engine="pyarrow", index=False)
    except ImportError:
        # Keep local development functional when pyarrow is not yet installed.
        frame.to_pickle(path)


def _read_dataframe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except ImportError:
        return pd.read_pickle(path)
    except Exception:
        return pd.read_pickle(path)


def _json_dump_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def summary_metadata_only(summary: dict[str, Any], *, models_path: Path | None = None) -> dict[str, Any]:
    metadata = dict(summary)
    metadata.pop("models", None)
    if models_path is not None:
        metadata["models_path"] = str(models_path.resolve())
    return metadata


def write_summary_rows_parquet(path: Path, rows: list[dict[str, Any]]) -> Path:
    frame = pd.DataFrame(
        [
            {
                "name": str(row.get("name") or ""),
                "family": str(row.get("family") or ""),
                "train_length": str(row.get("train_length") or ""),
                "window_length": str(row.get("window_length") or ""),
                "model_variation": str(row.get("model_variation") or ""),
                "time_variant": str(row.get("time_variant") or ""),
                "threshold_pct": int(row.get("threshold_pct") or 0),
                "status": str(row.get("status") or ""),
                "row_json": _json_dump_compact(row),
            }
            for row in rows
            if isinstance(row, dict)
        ]
    )
    _write_dataframe(path, frame)
    return path


def read_summary_rows_parquet(path: str | Path) -> list[dict[str, Any]]:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return []
    frame = _read_dataframe(parquet_path)
    rows: list[dict[str, Any]] = []
    if "row_json" not in frame.columns:
        return rows
    for value in frame["row_json"].tolist():
        if not isinstance(value, str) or not value.strip():
            continue
        payload = json.loads(value)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_prediction_stream_parquet(path: Path, prediction_stream: dict[str, Any]) -> Path:
    timestamps = [str(value) for value in prediction_stream.get("timestamps", [])]
    labels = list(prediction_stream.get("labels", []))
    size = max(len(timestamps), len(labels))
    if isinstance(prediction_stream.get("actions"), list):
        size = max(
            size,
            len(prediction_stream.get("actions", [])),
            len(prediction_stream.get("chosen_action_probabilities", [])),
        )
    else:
        size = max(
            size,
            len(prediction_stream.get("probability_up", [])),
            len(prediction_stream.get("probability_down", [])),
            len(prediction_stream.get("probability_none", [])),
        )
    rows: list[dict[str, Any]] = []
    for index in range(size):
        row = {
            "timestamp": timestamps[index] if index < len(timestamps) else None,
            "label": labels[index] if index < len(labels) else None,
            "action": None,
            "chosen_action_probability": None,
            "probability_up": None,
            "probability_down": None,
            "probability_none": None,
            "source_kind": prediction_stream.get("source_kind", "saved_model"),
        }
        if isinstance(prediction_stream.get("actions"), list):
            actions = prediction_stream.get("actions", [])
            probabilities = prediction_stream.get("chosen_action_probabilities", [])
            row["action"] = actions[index] if index < len(actions) else None
            row["chosen_action_probability"] = probabilities[index] if index < len(probabilities) else None
        else:
            for key in ("probability_up", "probability_down", "probability_none"):
                values = prediction_stream.get(key, [])
                row[key] = values[index] if index < len(values) else None
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(PREDICTION_STREAM_COLUMNS))
    _write_dataframe(path, frame)
    return path


def read_prediction_stream_parquet(path: str | Path) -> dict[str, Any]:
    parquet_path = Path(path)
    frame = _read_dataframe(parquet_path)
    for column in PREDICTION_STREAM_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[list(PREDICTION_STREAM_COLUMNS)]
    timestamps = [str(value) for value in frame["timestamp"].tolist() if value is not None and str(value) != "nan"]
    labels_raw = frame["label"].tolist()
    labels = [int(value) for value in labels_raw[: len(timestamps)]]
    action_values = frame["action"].tolist()
    has_actions = any(value is not None and str(value) != "nan" for value in action_values)
    source_kind_values = [value for value in frame["source_kind"].tolist() if isinstance(value, str) and value.strip()]
    source_kind = source_kind_values[0] if source_kind_values else "saved_model"
    if has_actions:
        probabilities = frame["chosen_action_probability"].tolist()
        return {
            "timestamps": timestamps,
            "actions": [int(value) for value in action_values[: len(timestamps)]],
            "labels": labels,
            "chosen_action_probabilities": [
                None if value is None or str(value) == "nan" else float(value)
                for value in probabilities[: len(timestamps)]
            ],
            "source_kind": source_kind,
            "predictions_path": str(parquet_path.resolve()),
        }
    return {
        "timestamps": timestamps,
        "labels": labels,
        "probability_up": [float(value) for value in frame["probability_up"].tolist()[: len(timestamps)]],
        "probability_down": [float(value) for value in frame["probability_down"].tolist()[: len(timestamps)]],
        "probability_none": [
            0.0 if value is None or str(value) == "nan" else float(value)
            for value in frame["probability_none"].tolist()[: len(timestamps)]
        ],
        "source_kind": source_kind,
        "predictions_path": str(parquet_path.resolve()),
    }


def is_supported_direct_row(row: dict[str, Any]) -> bool:
    family = str(row.get("family") or "").upper()
    if family not in SUPPORTED_FAMILIES:
        return False
    if str(row.get("model_variation") or "").upper() != "BASE":
        return False
    if str(row.get("window_length") or "").strip():
        return False
    if is_ignored_source_row(row):
        return False
    return True


def is_supported_v2_extra_row(row: dict[str, Any]) -> bool:
    if is_ignored_source_row(row):
        return False
    family = normalize_v2_family(str(row.get("family") or ""))
    if family.startswith("ENSEMBLE") or is_baseline_family_name(family):
        return False
    if family in SUPPORTED_FAMILIES and normalize_model_variation(row.get("model_variation")) == "BASE" and not normalize_window_length(row.get("window_length")):
        return False
    model_variation = normalize_model_variation(row.get("model_variation"))
    window_length = normalize_window_length(row.get("window_length"))
    return bool(window_length) or any(token in model_variation for token in EXTRA_VARIATION_TOKENS) or "PPO" in family


def supported_v2_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if isinstance(row, dict) and (is_supported_direct_row(row) or is_supported_v2_extra_row(row))
    ]


def direct_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and is_supported_direct_row(row)]


def extra_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and is_supported_v2_extra_row(row)]


def row_combination(row: dict[str, Any]) -> DirectCombination:
    return DirectCombination(
        family=normalize_family(str(row.get("family") or "")),
        train_length=normalize_train_length(str(row.get("train_length") or "")),
    )


def dedupe_rows_by_name(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name") or "")
        if not name:
            continue
        current = deduped.get(name)
        if current is None:
            deduped[name] = row
            continue
        current_completed = 1 if str(current.get("status") or "") == "completed" else 0
        row_completed = 1 if str(row.get("status") or "") == "completed" else 0
        current_has_model = 1 if current.get("saved_model_path") else 0
        row_has_model = 1 if row.get("saved_model_path") else 0
        if (row_completed, row_has_model) > (current_completed, current_has_model):
            deduped[name] = row
    return sorted(
        deduped.values(),
        key=lambda row: (
            str(row.get("family") or ""),
            train_length_to_rows(str(row.get("train_length") or "1K")) if str(row.get("train_length") or "").strip() else 0,
            str(row.get("window_length") or ""),
            str(row.get("model_variation") or ""),
            str(row.get("time_variant") or ""),
            int(row.get("threshold_pct") or 0),
            str(row.get("name") or ""),
        ),
    )


def load_summary(output_root: Path) -> dict[str, Any]:
    path = summary_path(output_root)
    if not path.exists():
        return {}
    summary = load_json(path)
    if not isinstance(summary, dict):
        return {}
    rows = summary.get("models")
    if isinstance(rows, list):
        return summary
    models_path_raw = summary.get("models_path")
    models_path = Path(str(models_path_raw)) if models_path_raw else summary_rows_path(output_root)
    if models_path.exists():
        summary["models"] = read_summary_rows_parquet(models_path)
    else:
        summary["models"] = []
    return summary


def load_summary_rows(output_root: Path) -> list[dict[str, Any]]:
    parquet_path = summary_rows_path(output_root)
    if parquet_path.exists():
        return [row for row in read_summary_rows_parquet(parquet_path) if isinstance(row, dict)]
    summary = load_summary(output_root)
    rows = summary.get("models", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def direct_target_combinations_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    combinations = sorted(
        {row_combination(row) for row in direct_rows(rows)},
        key=lambda item: (item.family, item.train_rows),
    )
    return [{"family": item.family, "train_length": item.train_length} for item in combinations]


def target_rows_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    targets = sorted(
        {target_row_from_model(row) for row in extra_rows(rows)},
        key=lambda item: (
            item.family,
            train_length_to_rows(item.train_length),
            item.window_length,
            item.model_variation,
            item.time_variant,
        ),
    )
    return [
        {
            "family": item.family,
            "train_length": item.train_length,
            "window_length": item.window_length,
            "model_variation": item.model_variation,
            "time_variant": item.time_variant,
        }
        for item in targets
    ]


def direct_target_combinations(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    stored = summary.get("direct_target_combinations", [])
    if isinstance(stored, list) and stored:
        normalized: set[DirectCombination] = set()
        for item in stored:
            if not isinstance(item, dict):
                continue
            family = item.get("family")
            train_length = item.get("train_length")
            if not family or not train_length:
                continue
            normalized.add(
                DirectCombination(
                    family=normalize_family(str(family)),
                    train_length=normalize_train_length(str(train_length)),
                )
            )
        if normalized:
            return [
                {"family": item.family, "train_length": item.train_length}
                for item in sorted(normalized, key=lambda value: (value.family, value.train_rows))
            ]
    return direct_target_combinations_from_rows(rows)


def target_rows(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    stored = summary.get(TARGET_ROWS_KEY, [])
    if isinstance(stored, list) and stored:
        normalized: set[V2TargetRow] = set()
        for item in stored:
            if not isinstance(item, dict):
                continue
            family = item.get("family")
            train_length = item.get("train_length")
            if not family or not train_length:
                continue
            normalized.add(
                V2TargetRow(
                    family=normalize_v2_family(str(family)),
                    train_length=normalize_train_length(str(train_length)),
                    window_length=normalize_window_length(item.get("window_length")),
                    model_variation=normalize_model_variation(item.get("model_variation")),
                    time_variant=str(item.get("time_variant") or "FULL").strip().upper(),
                )
            )
        if normalized:
            return [
                {
                    "family": item.family,
                    "train_length": item.train_length,
                    "window_length": item.window_length,
                    "model_variation": item.model_variation,
                    "time_variant": item.time_variant,
                }
                for item in sorted(
                    normalized,
                    key=lambda value: (
                        value.family,
                        train_length_to_rows(value.train_length),
                        value.window_length,
                        value.model_variation,
                        value.time_variant,
                    ),
                )
            ]
    return target_rows_from_rows(rows)


def retrainable_target_from_candidate(candidate: Any) -> RetrainableTarget:
    inferred_window_length = normalize_window_length(candidate.window_label)
    if not inferred_window_length and "WINDOW" in normalize_model_variation(candidate.model_variation):
        if int(candidate.experiment_id) in {3, 6, 7}:
            inferred_window_length = "1K"
        elif int(candidate.experiment_id) in {10, 11, 13}:
            inferred_window_length = "500"
        elif int(candidate.experiment_id) == 12:
            inferred_window_length = "250"
    return RetrainableTarget(
        source_experiment_id=int(candidate.experiment_id),
        source_experiment_name=str(candidate.experiment_name),
        source_variant=str(candidate.variant_name),
        source_name=str(candidate.source_name),
        family=normalize_v2_family(candidate.family),
        train_length=normalize_train_length(candidate.train_label),
        window_length=inferred_window_length,
        model_variation=normalize_model_variation(candidate.model_variation),
    )


def is_retrainable_source_candidate(candidate: Any) -> bool:
    if int(candidate.experiment_id) in {EXPERIMENT_ID, EXP23_SOURCE_EXPERIMENT_ID}:
        return False
    family = normalize_v2_family(candidate.family)
    if family.startswith("ENSEMBLE") or is_baseline_family_name(family):
        return False
    return True


def retrainable_targets_from_candidates(candidates: list[Any]) -> list[RetrainableTarget]:
    targets = {
        retrainable_target_from_candidate(candidate)
        for candidate in candidates
        if is_retrainable_source_candidate(candidate)
    }
    return sorted(
        targets,
        key=lambda item: (
            item.family,
            item.train_rows,
            item.window_length,
            item.model_variation,
            item.source_experiment_id,
            item.source_name,
        ),
    )


def current_source_backed_targets(rows: list[dict[str, Any]]) -> list[RetrainableTarget]:
    targets: set[RetrainableTarget] = set()
    for row in source_backed_v2_rows(rows):
        if not isinstance(row, dict):
            continue
        family = normalize_v2_family(str(row.get("family") or ""))
        if family.startswith("ENSEMBLE") or is_baseline_family_name(family):
            continue
        source_name = str(row.get("source_model") or "").strip()
        train_length = str(row.get("train_length") or "").strip()
        if not source_name or not train_length:
            continue
        targets.add(
            RetrainableTarget(
                source_experiment_id=int(row.get("source_experiment_id") or 0),
                source_experiment_name=str(row.get("source_experiment_name") or ""),
                source_variant=str(row.get("source_variant") or ""),
                source_name=source_name,
                family=family,
                train_length=normalize_train_length(train_length),
                window_length=normalize_window_length(row.get("window_length")),
                model_variation=normalize_model_variation(row.get("model_variation")),
            )
        )
    return sorted(
        targets,
        key=lambda item: (
            item.family,
            item.train_rows,
            item.window_length,
            item.model_variation,
            item.source_experiment_id,
            item.source_name,
        ),
    )


def merge_target_combinations(
    existing: list[dict[str, str]],
    additions: list[DirectCombination],
) -> list[dict[str, str]]:
    merged: set[DirectCombination] = set()
    for item in existing:
        if not isinstance(item, dict):
            continue
        family = item.get("family")
        train_length = item.get("train_length")
        if family and train_length:
            merged.add(
                DirectCombination(
                    family=normalize_family(str(family)),
                    train_length=normalize_train_length(str(train_length)),
                )
            )
    merged.update(additions)
    return [
        {"family": item.family, "train_length": item.train_length}
        for item in sorted(merged, key=lambda value: (value.family, value.train_rows))
    ]


def merge_target_rows(existing: list[dict[str, str]], additions: list[V2TargetRow]) -> list[dict[str, str]]:
    merged: set[V2TargetRow] = set()
    for item in existing:
        if not isinstance(item, dict):
            continue
        family = item.get("family")
        train_length = item.get("train_length")
        if not family or not train_length:
            continue
        merged.add(
            V2TargetRow(
                family=normalize_v2_family(str(family)),
                train_length=normalize_train_length(str(train_length)),
                window_length=normalize_window_length(item.get("window_length")),
                model_variation=normalize_model_variation(item.get("model_variation")),
                time_variant=str(item.get("time_variant") or "FULL").strip().upper(),
            )
        )
    merged.update(additions)
    return [
        {
            "family": item.family,
            "train_length": item.train_length,
            "window_length": item.window_length,
            "model_variation": item.model_variation,
            "time_variant": item.time_variant,
        }
        for item in sorted(
            merged,
            key=lambda value: (
                value.family,
                train_length_to_rows(value.train_length),
                value.window_length,
                value.model_variation,
                value.time_variant,
            ),
        )
    ]


def ensemble_target_families(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    stored = summary.get(ENSEMBLE_TARGET_KEY, [])
    if isinstance(stored, list) and stored:
        normalized = sorted({normalize_ensemble_family(item) for item in stored if str(item).strip()})
        if normalized:
            return normalized
    derived = sorted(
        {
            normalize_ensemble_family(str(row.get("family") or ""))
            for row in rows
            if isinstance(row, dict) and str(row.get("family") or "").upper().startswith("ENSEMBLE")
        }
    )
    return derived


def merge_ensemble_target_families(existing: list[str], additions: list[str]) -> list[str]:
    merged = {
        normalize_ensemble_family(value)
        for value in list(existing or []) + list(additions or [])
        if str(value).strip()
    }
    return sorted(merged)


def copy_file_if_present(source_path: str | Path | None, destination_path: Path) -> str:
    if not source_path:
        return ""
    source = Path(str(source_path))
    if not source.exists():
        return ""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination_path.resolve():
        shutil.copy2(source, destination_path)
    return str(destination_path.resolve())


def ensure_prediction_copy(
    output_root: Path,
    row: dict[str, Any],
) -> str:
    source_predictions = str(row.get("predictions_path") or row.get("source_predictions_path") or "").strip()
    if not source_predictions:
        return ""
    destination = target_row_prediction_output_path(output_root, target_row_from_model(row))
    return copy_file_if_present(source_predictions, destination)


def ensure_saved_model_copy(
    output_root: Path,
    row: dict[str, Any],
) -> tuple[str, str, list[str]]:
    source_saved_model = str(row.get("saved_model_path") or row.get("source_saved_model_path") or "").strip()
    if not source_saved_model:
        return "", "", []
    group_dir = target_row_saved_model_dir(output_root, target_row_from_model(row))
    destination = group_dir / Path(source_saved_model).name
    copied = copy_file_if_present(source_saved_model, destination)
    if not copied:
        return "", "", []
    return copied, str(group_dir.resolve()), [copied]


def current_row_names_for_combination(combination: DirectCombination) -> list[str]:
    names: list[str] = []
    for time_variant in TIME_VARIANTS:
        base_name = f"{combination.family}_{combination.train_length}_{time_variant}_BASE"
        for threshold in THRESHOLDS:
            names.append(f"{base_name}_{threshold}")
    return names


def is_combination_completed(summary_rows: list[dict[str, Any]], combination: DirectCombination) -> bool:
    row_map = {str(row.get("name") or ""): row for row in summary_rows}
    for row_name in current_row_names_for_combination(combination):
        row = row_map.get(row_name)
        if not isinstance(row, dict):
            return False
        if str(row.get("status") or "") != "completed":
            return False
        if not row_has_reusable_source(row):
            return False
    return True


def is_target_completed(summary_rows: list[dict[str, Any]], target: V2TargetRow) -> bool:
    row_map = {str(row.get("name") or ""): row for row in summary_rows}
    for threshold in THRESHOLDS:
        row_name = f"{target.base_name}_{threshold}"
        row = row_map.get(row_name)
        if not isinstance(row, dict):
            return False
        if str(row.get("status") or "") != "completed":
            return False
        if not row_has_reusable_source(row):
            return False
    return True


def row_has_reusable_source(row: dict[str, Any]) -> bool:
    return any(
        str(row.get(key) or "").strip()
        for key in ("saved_model_path", "source_saved_model_path", "predictions_path", "source_predictions_path")
    )


def build_combinations_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    target_items = direct_target_combinations(summary, rows)
    targets_by_family: dict[str, list[DirectCombination]] = {}
    for item in target_items:
        combination = DirectCombination(
            family=normalize_family(item["family"]),
            train_length=normalize_train_length(item["train_length"]),
        )
        targets_by_family.setdefault(combination.family, []).append(combination)

    row_map = {str(row.get("name") or ""): row for row in rows}
    grouped_entries: dict[str, list[tuple[str, list[str]]]] = {}
    for family in SUPPORTED_FAMILIES:
        combinations = sorted(targets_by_family.get(family, []), key=lambda item: item.train_rows)
        for combination in combinations:
            for time_variant in TIME_VARIANTS:
                row_names = [
                    f"{combination.family}_{combination.train_length}_{time_variant}_BASE_{threshold}"
                    for threshold in THRESHOLDS
                ]
                grouped_entries.setdefault(family, []).append(
                    (f"{combination.family}:{combination.train_length}:{time_variant}", row_names)
                )
    for item in target_rows(summary, rows):
        target = V2TargetRow(
            family=normalize_v2_family(item["family"]),
            train_length=normalize_train_length(item["train_length"]),
            window_length=normalize_window_length(item.get("window_length")),
            model_variation=normalize_model_variation(item.get("model_variation")),
            time_variant=str(item.get("time_variant") or "FULL").strip().upper(),
        )
        row_names = [f"{target.base_name}_{threshold}" for threshold in THRESHOLDS]
        label_parts = [target.family, target.train_length]
        if target.window_length:
            label_parts.append(target.window_length)
        label_parts.extend([target.model_variation, target.time_variant])
        grouped_entries.setdefault(target.family, []).append((":".join(label_parts), row_names))

    lines = ["# ExperimentSV2/1 Combinations", ""]
    for family in sorted(grouped_entries.keys()):
        entries = grouped_entries[family]
        completed = sum(
            1
            for _, row_names in entries
            if all(
                isinstance(row_map.get(row_name), dict)
                and str(row_map[row_name].get("status") or "") == "completed"
                and row_has_reusable_source(row_map[row_name])
                for row_name in row_names
            )
        )
        lines.append(f"## {family}")
        lines.append("")
        lines.append(f"{completed}/{len(entries)} combinations completed")
        lines.append("")
        for label, row_names in entries:
            status = "completed" if all(
                isinstance(row_map.get(row_name), dict)
                and str(row_map[row_name].get("status") or "") == "completed"
                and row_has_reusable_source(row_map[row_name])
                for row_name in row_names
            ) else "missing"
            lines.append(f"- {label} [{status}]")
        lines.append("")
    ensemble_families = ensemble_target_families(summary, rows)
    if ensemble_families:
        for ensemble_family in ensemble_families:
            expanded_targets = [(ENSEMBLE_TRAIN_LENGTH, time_variant) for time_variant in TIME_VARIANTS]
            completed = 0
            for train_length, time_variant in expanded_targets:
                row_names = [
                    f"{ensemble_family}_{train_length}_{time_variant}_BASE_{threshold}"
                    for threshold in THRESHOLDS
                ]
                if all(
                    isinstance(row_map.get(row_name), dict)
                    and str(row_map[row_name].get("status") or "") == "completed"
                    and row_has_reusable_source(row_map[row_name])
                    for row_name in row_names
                ):
                    completed += 1
            lines.append(f"## {ensemble_family}")
            lines.append("")
            lines.append(f"{completed}/{len(expanded_targets)} combinations completed")
            lines.append("")
            for train_length, time_variant in expanded_targets:
                row_names = [
                    f"{ensemble_family}_{train_length}_{time_variant}_BASE_{threshold}"
                    for threshold in THRESHOLDS
                ]
                status = "completed" if all(
                    isinstance(row_map.get(row_name), dict)
                    and str(row_map[row_name].get("status") or "") == "completed"
                    and row_has_reusable_source(row_map[row_name])
                    for row_name in row_names
                ) else "missing"
                lines.append(f"- {ensemble_family}:{train_length}:{time_variant} [{status}]")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_combinations_markdown(output_root: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    markdown_path = output_root / "combinations.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_combinations_markdown(summary, rows), encoding="utf-8")
    return markdown_path


def write_manifest_and_summary(output_root: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    variant_dir = output_root / VARIANT_NAME
    variant_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in summary.get("models", []) if isinstance(row, dict)]
    models_file = summary_rows_path(output_root)
    write_summary_rows_parquet(models_file, rows)
    summary_file = summary_path(output_root)
    summary_metadata = summary_metadata_only(summary, models_path=models_file)
    write_json_atomic(summary_file, summary_metadata)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "root_dir": str(output_root.resolve()),
        "variants": {
            VARIANT_NAME: {
                "summary_path": str(summary_file.resolve()),
                "models_path": str(models_file.resolve()),
                "summary": summary_metadata,
            }
        },
    }
    manifest_file = manifest_path(output_root)
    write_json_atomic(manifest_file, manifest)
    return summary_file, manifest_file


def build_parallel_eval_rows(
    *,
    output_root: Path,
    combination: DirectCombination,
    saved_model_path: str,
    prediction_stream: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
    train_payload: dict[str, Any] | None = None,
    time_variants: list[str] | None = None,
    thresholds: list[int] | None = None,
    max_workers: int = 8,
    model_variation: str = "BASE",
    window_length: str = "",
    artifact_bucket: str = NON_WINDOW_BUCKET,
    source_model: str | None = None,
    env_version: str = "ternary",
) -> list[dict[str, Any]]:
    selected_time_variants = list(time_variants or TIME_VARIANTS)
    selected_thresholds = list(thresholds or THRESHOLDS)
    normalized_target = V2TargetRow(
        family=combination.family,
        train_length=combination.train_length,
        window_length=normalize_window_length(window_length),
        model_variation=normalize_model_variation(model_variation),
        time_variant="FULL",
    )
    saved_model_dir = target_row_saved_model_dir(output_root, normalized_target if artifact_bucket == WINDOWED_BUCKET else normalized_target)
    grouped_saved_model_dir = str(saved_model_dir.resolve())
    grouped_saved_model_paths = [str(Path(saved_model_path).resolve())]

    for time_variant in selected_time_variants:
        prediction_path = target_row_prediction_output_path(
            output_root,
            V2TargetRow(
                family=combination.family,
                train_length=combination.train_length,
                window_length=normalize_window_length(window_length),
                model_variation=normalize_model_variation(model_variation),
                time_variant=time_variant,
            ),
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        write_prediction_stream_parquet(prediction_path, prediction_stream)

    def build_one(time_variant: str, threshold: int) -> dict[str, Any]:
        predicate = time_variant_predicate(time_variant)
        evaluated = evaluate_prediction_stream(
            prediction_stream=prediction_stream,
            threshold_pct=threshold,
            active_timestamp_predicate=predicate,
            env_version=env_version,
        )
        row_target = V2TargetRow(
            family=combination.family,
            train_length=combination.train_length,
            window_length=normalize_window_length(window_length),
            model_variation=normalize_model_variation(model_variation),
            time_variant=time_variant,
        )
        base_name = row_target.base_name
        prediction_path = target_row_prediction_output_path(output_root, row_target)
        row = {
            "name": f"{base_name}_{threshold}",
            "source_model": source_model or combination.family,
            "normalized_base_name": base_name,
            "source_provenance": SOURCE_PROVENANCE_BY_TIME_VARIANT[time_variant],
            "family": combination.family,
            "train_length": combination.train_length,
            "window_length": row_target.window_length,
            "time_variant": time_variant,
            "model_variation": row_target.model_variation,
            "threshold_pct": int(threshold),
            "env_version": env_version,
            "requested_train_rows": combination.train_rows,
            "actual_train_rows": combination.train_rows,
            "requested_test_rows": CANONICAL_TEST_ROWS,
            "actual_test_rows": CANONICAL_TEST_ROWS,
            "saved_model_path": str(Path(saved_model_path).resolve()),
            "grouped_saved_model_dir": grouped_saved_model_dir,
            "grouped_saved_model_paths": grouped_saved_model_paths,
            "source_saved_model_path": str(Path(saved_model_path).resolve()),
            "predictions_path": str(prediction_path.resolve()),
            "source_predictions_path": str(prediction_path.resolve()),
            "source_experiment_id": int((source_metadata or {}).get("source_experiment_id") or EXPERIMENT_ID),
            "source_experiment_name": str((source_metadata or {}).get("source_experiment_name") or EXPERIMENT_NAME),
            "source_variant": str((source_metadata or {}).get("source_variant") or VARIANT_NAME),
            "status": "completed",
        }
        if isinstance(source_metadata, dict):
            for key in PRESERVED_WINDOW_METADATA_KEYS:
                if key in source_metadata:
                    row[key] = source_metadata.get(key)
        row.update(evaluated)
        if isinstance(train_payload, dict) and train_payload:
            row["train"] = train_payload
        return row

    futures: list[dict[str, Any]] = []
    tasks = [(time_variant, threshold) for time_variant in selected_time_variants for threshold in selected_thresholds]
    worker_count = max(1, min(max_workers, len(tasks)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(build_one, time_variant, threshold): (time_variant, threshold)
            for time_variant, threshold in tasks
        }
        for future in future_map:
            futures.append(future.result())
    return dedupe_rows_by_name(normalize_rows_window_metadata(futures))


def write_window_train_manifests_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return backfill_window_train_manifests(normalize_summary_rows(rows))
