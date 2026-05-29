from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.btc_direction_learning.config import (
    DEFAULT_TEST_HOURS,
    DEFAULT_TOTAL_LABELED_HOURS,
    DEFAULT_TRAIN_HOURS,
    DEFAULT_SPLIT_REGIME,
    FEATURE_WARMUP_PADDING,
    FIXED_REGIME_NAME,
    FIXED_TOTAL_LABELED_HOURS,
    FIXED_TEST_START_TIMESTAMP,
    LEGACY_RL_PPO_DIR,
    MARKET_HOURS_MAX_FETCH_RETRIES,
    SHARED_DATA_DIR,
    VARIABLE_REGIME_NAME,
    build_regime_split_slug,
    infer_raw_fetch_hours,
)
from src.utils.market_hours import is_allowed_prediction_target_timestamp
from src.utils.market_data import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    add_features,
    fetch_ohlcv,
)


RAW_CACHE_NAME = "btc_ohlcv_cache.csv"
PROCESSED_CACHE_NAME = "btc_features_cache.csv"
SPLIT_METADATA_NAME = "split_metadata.json"
FULL_CANDLES_PATH = Path("data/btc/btc_100k_train_5k_test_candles.csv")
MARKET_HOURS_CANDLES_PATH = Path("data/btc/btc_100k_train_5k_test_market_hours_candles.csv")
EXPERIMENT_TEMPORAL_FEATURE_COLUMNS = [
    "day_of_week",
    "hour_of_day",
    "day_of_month_sin",
    "day_of_month_cos",
]
TEMPORAL_FEATURE_COLUMNS = EXPERIMENT_TEMPORAL_FEATURE_COLUMNS
CANONICAL_TEST_ROWS = 5000


@dataclass
class DirectionDatasetBundle:
    processed_df: pd.DataFrame
    scaled_features: np.ndarray
    scaler: StandardScaler
    feature_columns: list[str]
    sequence_length: int
    labels: np.ndarray
    train_decision_indices: np.ndarray
    test_decision_indices: np.ndarray
    train_row_count: int
    test_row_count: int

    @property
    def feature_dim(self) -> int:
        return len(self.feature_columns)

    def build_observation(self, global_index: int) -> np.ndarray:
        start = global_index - self.sequence_length + 1
        end = global_index + 1
        return self.scaled_features[start:end].astype(np.float32, copy=True)

    def build_observation_batch(self, indices: np.ndarray) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64)
        offsets = np.arange(self.sequence_length, dtype=np.int64)
        start_indices = index_array - (self.sequence_length - 1)
        window_indices = start_indices[:, None] + offsets[None, :]
        return self.scaled_features[window_indices].astype(np.float32, copy=False)

    def get_split_data(self, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self._get_split_indices(split_name)
        observations = self.build_observation_batch(indices)
        labels = self.labels[indices].astype(np.int64)
        return observations, labels, indices

    def get_expert_actions(self, split_name: str) -> np.ndarray:
        _, labels, _ = self.get_split_data(split_name)
        return labels.copy()

    def get_all_metadata(self) -> dict[str, int]:
        return {
            "processed_rows": int(len(self.processed_df)),
            "train_rows": self.train_row_count,
            "test_rows": self.test_row_count,
            "train_decisions": int(len(self.train_decision_indices)),
            "test_decisions": int(len(self.test_decision_indices)),
            "sequence_length": self.sequence_length,
            "feature_count": self.feature_dim,
        }

    def get_split_timestamps(self, split_name: str) -> list[str]:
        indices = self._get_split_indices(split_name)
        timestamps = pd.to_datetime(self.processed_df.iloc[indices]["timestamp"], utc=True)
        return [timestamp.isoformat() for timestamp in timestamps.tolist()]

    def get_split_contract(
        self,
        *,
        data_variant: str,
        dataset_source_path: str,
        split_mode: str,
        source_variant: str | None = None,
    ) -> dict[str, object]:
        train_timestamps = self.get_split_timestamps("train")
        test_timestamps = self.get_split_timestamps("test")
        return {
            "data_variant": data_variant,
            "source_variant": source_variant or data_variant,
            "dataset_source_path": dataset_source_path,
            "split_mode": split_mode,
            "dataset_metadata": self.get_all_metadata(),
            "source_test_window": {
                "start": test_timestamps[0] if test_timestamps else None,
                "end": test_timestamps[-1] if test_timestamps else None,
                "count": len(test_timestamps),
            },
            "plotting": {
                "x_axis_mode": "index_with_7_day_date_ticks",
                "train_timestamps": train_timestamps,
                "test_timestamps": test_timestamps,
            },
        }

    def _get_split_indices(self, split_name: str) -> np.ndarray:
        if split_name == "train":
            return self.train_decision_indices
        if split_name == "test":
            return self.test_decision_indices
        raise ValueError(f"Unknown split name: {split_name}")


@dataclass
class RegressionDatasetBundle:
    processed_df: pd.DataFrame
    scaled_features: np.ndarray
    scaler: StandardScaler
    feature_columns: list[str]
    sequence_length: int
    targets: np.ndarray
    direction_labels: np.ndarray
    train_decision_indices: np.ndarray
    test_decision_indices: np.ndarray
    train_row_count: int
    test_row_count: int

    @property
    def feature_dim(self) -> int:
        return len(self.feature_columns)

    def build_observation(self, global_index: int) -> np.ndarray:
        start = global_index - self.sequence_length + 1
        end = global_index + 1
        return self.scaled_features[start:end].astype(np.float32, copy=True)

    def build_observation_batch(self, indices: np.ndarray) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64)
        offsets = np.arange(self.sequence_length, dtype=np.int64)
        start_indices = index_array - (self.sequence_length - 1)
        window_indices = start_indices[:, None] + offsets[None, :]
        return self.scaled_features[window_indices].astype(np.float32, copy=False)

    def get_split_data(self, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self._get_split_indices(split_name)
        observations = self.build_observation_batch(indices)
        targets = self.targets[indices].astype(np.float32)
        return observations, targets, indices

    def get_split_direction_labels(self, split_name: str) -> np.ndarray:
        indices = self._get_split_indices(split_name)
        return self.direction_labels[indices].astype(np.int64)

    def get_all_metadata(self) -> dict[str, int]:
        return {
            "processed_rows": int(len(self.processed_df)),
            "train_rows": self.train_row_count,
            "test_rows": self.test_row_count,
            "train_decisions": int(len(self.train_decision_indices)),
            "test_decisions": int(len(self.test_decision_indices)),
            "sequence_length": self.sequence_length,
            "feature_count": self.feature_dim,
        }

    def get_split_timestamps(self, split_name: str) -> list[str]:
        indices = self._get_split_indices(split_name)
        timestamps = pd.to_datetime(self.processed_df.iloc[indices]["timestamp"], utc=True)
        return [timestamp.isoformat() for timestamp in timestamps.tolist()]

    def get_split_contract(
        self,
        *,
        data_variant: str,
        dataset_source_path: str,
        split_mode: str,
        source_variant: str | None = None,
    ) -> dict[str, object]:
        train_timestamps = self.get_split_timestamps("train")
        test_timestamps = self.get_split_timestamps("test")
        return {
            "data_variant": data_variant,
            "source_variant": source_variant or data_variant,
            "dataset_source_path": dataset_source_path,
            "split_mode": split_mode,
            "dataset_metadata": self.get_all_metadata(),
            "source_test_window": {
                "start": test_timestamps[0] if test_timestamps else None,
                "end": test_timestamps[-1] if test_timestamps else None,
                "count": len(test_timestamps),
            },
            "plotting": {
                "x_axis_mode": "index_with_7_day_date_ticks",
                "train_timestamps": train_timestamps,
                "test_timestamps": test_timestamps,
            },
        }

    def _get_split_indices(self, split_name: str) -> np.ndarray:
        if split_name == "train":
            return self.train_decision_indices
        if split_name == "test":
            return self.test_decision_indices
        raise ValueError(f"Unknown split name: {split_name}")


def ensure_shared_data_dir(data_dir: Path | None = None) -> Path:
    base_dir = data_dir or SHARED_DATA_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def build_shared_data_dir(
    train_hours: int,
    test_hours: int,
    split_regime: str = DEFAULT_SPLIT_REGIME,
    data_variant: str = "full",
    data_dir: Path | None = None,
) -> Path:
    base_dir = data_dir or SHARED_DATA_DIR
    target_dir = base_dir / f"{build_regime_split_slug(split_regime, train_hours, test_hours)}_data_{data_variant}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _raw_cache_path(data_dir: Path) -> Path:
    return data_dir / RAW_CACHE_NAME


def _processed_cache_path(data_dir: Path) -> Path:
    return data_dir / PROCESSED_CACHE_NAME


def _split_metadata_path(data_dir: Path) -> Path:
    return data_dir / SPLIT_METADATA_NAME


def _legacy_processed_cache_path() -> Path:
    return LEGACY_RL_PPO_DIR / PROCESSED_CACHE_NAME


def _legacy_raw_cache_path() -> Path:
    return LEGACY_RL_PPO_DIR / RAW_CACHE_NAME


def _shared_legacy_processed_cache_path() -> Path:
    return SHARED_DATA_DIR / PROCESSED_CACHE_NAME


def _shared_legacy_raw_cache_path() -> Path:
    return SHARED_DATA_DIR / RAW_CACHE_NAME


def _validate_processed_frame(frame: pd.DataFrame, total_labeled_hours: int) -> None:
    required_columns = ["timestamp", "target", *FEATURE_COLUMNS]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Processed direction-learning dataset is missing columns: {missing}")
    if len(frame) != total_labeled_hours:
        raise RuntimeError(
            f"Processed direction-learning dataset must contain exactly {total_labeled_hours} rows, "
            f"but has {len(frame)}."
        )


def _filter_market_hours_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    mask = timestamps.apply(is_allowed_prediction_target_timestamp)
    return frame.loc[mask].reset_index(drop=True)


def add_experiment_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = pd.to_datetime(enriched["timestamp"], utc=True)
    enriched["day_of_week"] = timestamps.dt.dayofweek.astype(np.float32)
    enriched["hour_of_day"] = timestamps.dt.hour.astype(np.float32)
    day_of_month = timestamps.dt.day.astype(np.float32)
    enriched["day_of_month_sin"] = np.sin((2.0 * np.pi * day_of_month) / 31.0).astype(np.float32)
    enriched["day_of_month_cos"] = np.cos((2.0 * np.pi * day_of_month) / 31.0).astype(np.float32)
    return enriched


def _add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    return add_experiment_temporal_features(frame)


def build_experiment_model_feature_columns(feature_columns: list[str] | None = None) -> list[str]:
    if feature_columns is not None:
        return list(feature_columns)
    return [*FEATURE_COLUMNS, *EXPERIMENT_TEMPORAL_FEATURE_COLUMNS]


def fit_experiment_style_scaler(
    processed: pd.DataFrame,
    *,
    model_feature_columns: list[str],
    train_row_count: int,
) -> tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    enriched = add_experiment_temporal_features(processed)
    scaler = StandardScaler().fit(
        enriched.iloc[:train_row_count][model_feature_columns].to_numpy(dtype=np.float32)
    )
    scaled_features = scaler.transform(
        enriched[model_feature_columns].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    return enriched, scaler, scaled_features


def _build_model_feature_columns() -> list[str]:
    return build_experiment_model_feature_columns()


def resolve_dataset_source_path(data_variant: str) -> Path:
    if data_variant == "market_hours":
        return MARKET_HOURS_CANDLES_PATH
    return FULL_CANDLES_PATH


def load_unified_dataset_frame(data_variant: str) -> pd.DataFrame:
    unified_path = resolve_dataset_source_path(data_variant)
    if not unified_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {unified_path}\nRun scripts/expand_train_only_dataset_to_100k.py first."
        )
    return pd.read_csv(unified_path, parse_dates=["timestamp"])


def slice_latest_train_test_window(
    frame: pd.DataFrame,
    *,
    train_hours: int,
    test_hours: int = CANONICAL_TEST_ROWS,
) -> pd.DataFrame:
    required_rows = int(train_hours) + int(test_hours)
    if len(frame) < required_rows:
        raise ValueError(
            f"Dataset has {len(frame)} rows; need >= {required_rows} "
            f"(train_hours={train_hours} + test_hours={test_hours})."
        )

    train_pool = frame.iloc[: -test_hours].reset_index(drop=True)
    test_set = frame.iloc[-test_hours:].reset_index(drop=True)
    if train_hours > len(train_pool):
        raise ValueError(f"Requested train_hours ({train_hours}) exceeds available ({len(train_pool)})")

    train_set = train_pool.iloc[-train_hours:].reset_index(drop=True)
    return pd.concat([train_set, test_set], ignore_index=True)


def slice_absolute_train_test_window(
    frame: pd.DataFrame,
    *,
    train_start_row: int,
    train_hours: int,
    test_hours: int,
) -> pd.DataFrame:
    train_start_row = int(train_start_row)
    train_hours = int(train_hours)
    test_hours = int(test_hours)
    if train_start_row < 0:
        raise ValueError(f"train_start_row must be >= 0, got {train_start_row}.")
    if train_hours <= 0:
        raise ValueError(f"train_hours must be positive, got {train_hours}.")
    if test_hours <= 0:
        raise ValueError(f"test_hours must be positive, got {test_hours}.")

    train_end_row = train_start_row + train_hours
    test_end_row = train_end_row + test_hours
    if test_end_row > len(frame):
        raise ValueError(
            f"Dataset has {len(frame)} rows; need rows through {test_end_row} "
            f"for train_start_row={train_start_row}, train_hours={train_hours}, test_hours={test_hours}."
        )

    train_set = frame.iloc[train_start_row:train_end_row].reset_index(drop=True)
    test_set = frame.iloc[train_end_row:test_end_row].reset_index(drop=True)
    return pd.concat([train_set, test_set], ignore_index=True)


def _seed_from_legacy_cache(
    data_dir: Path,
    train_hours: int,
    test_hours: int,
    raw_fetch_hours: int,
) -> pd.DataFrame | None:
    total_labeled_hours = train_hours + test_hours

    shared_legacy_processed = _shared_legacy_processed_cache_path()
    if shared_legacy_processed.exists():
        cached = pd.read_csv(shared_legacy_processed, parse_dates=["timestamp"])
        if len(cached) == total_labeled_hours:
            _validate_processed_frame(cached, total_labeled_hours)
            cached.to_csv(_processed_cache_path(data_dir), index=False)

            shared_legacy_raw = _shared_legacy_raw_cache_path()
            if shared_legacy_raw.exists():
                pd.read_csv(shared_legacy_raw, parse_dates=["timestamp"]).to_csv(_raw_cache_path(data_dir), index=False)

            metadata = {
                "raw_fetch_hours": raw_fetch_hours,
                "sequence_length": SEQUENCE_LENGTH,
                "feature_columns": FEATURE_COLUMNS,
                "processed_rows": len(cached),
                "train_rows": train_hours,
                "test_rows": test_hours,
                "train_decisions": int(max(0, train_hours - (SEQUENCE_LENGTH - 1))),
                "test_decisions": test_hours,
                "seeded_from_legacy_rl_ppo_cache": False,
                "seeded_from_previous_shared_cache": True,
            }
            _split_metadata_path(data_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return cached

    if train_hours != DEFAULT_TRAIN_HOURS or test_hours != DEFAULT_TEST_HOURS:
        return None

    processed_path = _legacy_processed_cache_path()
    if not processed_path.exists():
        return None

    cached = pd.read_csv(processed_path, parse_dates=["timestamp"])
    if len(cached) != DEFAULT_TOTAL_LABELED_HOURS:
        return None
    _validate_processed_frame(cached, DEFAULT_TOTAL_LABELED_HOURS)
    cached.to_csv(_processed_cache_path(data_dir), index=False)

    legacy_raw = _legacy_raw_cache_path()
    if legacy_raw.exists():
        pd.read_csv(legacy_raw, parse_dates=["timestamp"]).to_csv(_raw_cache_path(data_dir), index=False)

    metadata = {
        "raw_fetch_hours": raw_fetch_hours,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "processed_rows": len(cached),
        "train_rows": train_hours,
        "test_rows": test_hours,
        "train_decisions": int(max(0, train_hours - (SEQUENCE_LENGTH - 1))),
        "test_decisions": test_hours,
        "seeded_from_legacy_rl_ppo_cache": True,
        "seeded_from_previous_shared_cache": False,
    }
    _split_metadata_path(data_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cached


def fetch_and_prepare_dataset(
    force_refresh: bool = False,
    train_hours: int = DEFAULT_TRAIN_HOURS,
    test_hours: int = DEFAULT_TEST_HOURS,
    raw_fetch_hours: int | None = None,
    split_regime: str = DEFAULT_SPLIT_REGIME,
    data_variant: str = "full",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Loads a unified candle CSV, slices train/test, and returns the processed DataFrame.

    - ``full`` and ``derived_market_hours`` (built from full in callers): ``btc_100k_train_5k_test_candles.csv``.
    - ``market_hours``: ``btc_100k_train_5k_test_market_hours_candles.csv`` (ET prediction hours only).

    Always uses the last 5k rows of the chosen CSV as test; train is the last ``train_hours``
    rows of the pool before that test block.
    """
    df = load_unified_dataset_frame(data_variant)
    return slice_latest_train_test_window(
        df,
        train_hours=int(train_hours),
        test_hours=CANONICAL_TEST_ROWS,
    )


def build_direction_dataset_bundle(
    force_refresh: bool = False,
    train_hours: int = DEFAULT_TRAIN_HOURS,
    test_hours: int = DEFAULT_TEST_HOURS,
    raw_fetch_hours: int | None = None,
    split_regime: str = DEFAULT_SPLIT_REGIME,
    data_variant: str = "full",
    data_dir: Path | None = None,
) -> DirectionDatasetBundle:
    processed = fetch_and_prepare_dataset(
        force_refresh=force_refresh,
        train_hours=train_hours,
        test_hours=test_hours,
        raw_fetch_hours=raw_fetch_hours,
        split_regime=split_regime,
        data_variant=data_variant,
        data_dir=data_dir,
    )
    # Always use test_hours=5000 for canonical split
    return build_direction_dataset_bundle_from_processed(
        processed=processed,
        train_hours=train_hours,
        test_hours=5000,
    )


def build_direction_dataset_bundle_from_processed(
    processed: pd.DataFrame,
    train_hours: int,
    test_hours: int,
) -> DirectionDatasetBundle:
    return build_direction_dataset_bundle_from_processed_with_feature_columns(
        processed,
        train_hours=train_hours,
        test_hours=test_hours,
    )


def build_direction_dataset_bundle_from_processed_with_feature_columns(
    processed: pd.DataFrame,
    train_hours: int,
    test_hours: int,
    *,
    feature_columns: list[str] | None = None,
    label_column: str = "target",
) -> DirectionDatasetBundle:
    total_labeled_hours = train_hours + test_hours
    _validate_processed_frame(processed, total_labeled_hours)
    model_feature_columns = build_experiment_model_feature_columns(feature_columns)
    processed, scaler, scaled_features = fit_experiment_style_scaler(
        processed,
        model_feature_columns=model_feature_columns,
        train_row_count=int(train_hours),
    )
    labels = processed[label_column].to_numpy(dtype=np.int64)

    train_decision_indices = np.arange(SEQUENCE_LENGTH - 1, train_hours, dtype=np.int64)
    test_decision_indices = np.arange(train_hours, train_hours + test_hours, dtype=np.int64)

    return DirectionDatasetBundle(
        processed_df=processed,
        scaled_features=scaled_features,
        scaler=scaler,
        feature_columns=model_feature_columns,
        sequence_length=SEQUENCE_LENGTH,
        labels=labels,
        train_decision_indices=train_decision_indices,
        test_decision_indices=test_decision_indices,
        train_row_count=train_hours,
        test_row_count=test_hours,
    )


def build_direction_dataset_bundle_from_absolute_window(
    frame: pd.DataFrame,
    *,
    train_start_row: int,
    train_hours: int,
    test_hours: int,
) -> DirectionDatasetBundle:
    processed = slice_absolute_train_test_window(
        frame,
        train_start_row=train_start_row,
        train_hours=train_hours,
        test_hours=test_hours,
    )
    return build_direction_dataset_bundle_from_processed(
        processed=processed,
        train_hours=int(train_hours),
        test_hours=int(test_hours),
    )


def build_regression_dataset_bundle_from_processed(
    processed: pd.DataFrame,
    train_hours: int,
    test_hours: int,
    *,
    target_column: str = "target_delta",
    direction_label_column: str = "target_direction",
    feature_columns: list[str] | None = None,
) -> RegressionDatasetBundle:
    total_labeled_hours = train_hours + test_hours
    _validate_processed_frame(processed, total_labeled_hours)
    required_columns = [target_column, direction_label_column]
    missing = [column for column in required_columns if column not in processed.columns]
    if missing:
        raise RuntimeError(f"Processed regression dataset is missing columns: {missing}")

    model_feature_columns = build_experiment_model_feature_columns(feature_columns)
    processed, scaler, scaled_features = fit_experiment_style_scaler(
        processed,
        model_feature_columns=model_feature_columns,
        train_row_count=int(train_hours),
    )
    targets = processed[target_column].to_numpy(dtype=np.float32)
    direction_labels = processed[direction_label_column].to_numpy(dtype=np.int64)

    train_decision_indices = np.arange(SEQUENCE_LENGTH - 1, train_hours, dtype=np.int64)
    test_decision_indices = np.arange(train_hours, train_hours + test_hours, dtype=np.int64)

    return RegressionDatasetBundle(
        processed_df=processed,
        scaled_features=scaled_features,
        scaler=scaler,
        feature_columns=model_feature_columns,
        sequence_length=SEQUENCE_LENGTH,
        targets=targets,
        direction_labels=direction_labels,
        train_decision_indices=train_decision_indices,
        test_decision_indices=test_decision_indices,
        train_row_count=train_hours,
        test_row_count=test_hours,
    )
