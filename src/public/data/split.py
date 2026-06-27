from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FEATURE_WARMUP_ROWS = 64


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    warmup: pd.DataFrame | None = None


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_length: int,
    validation_length: int,
    test_length: int,
    warmup_length: int = 0,
) -> SplitFrames:
    total = train_length + validation_length + test_length
    required = total + warmup_length
    if len(frame) < required:
        raise ValueError(f"Not enough rows for configured split: need {required}, got {len(frame)}")
    data = frame.tail(required).reset_index(drop=True)
    train_start = warmup_length
    validation_start = train_start + train_length
    test_start = validation_start + validation_length
    warmup = data.iloc[:train_start].reset_index(drop=True) if warmup_length else None
    train = data.iloc[train_start:validation_start].reset_index(drop=True)
    validation = data.iloc[validation_start:test_start].reset_index(drop=True)
    test = data.iloc[test_start : test_start + test_length].reset_index(drop=True)
    if not (train["timestamp"].max() < validation["timestamp"].min() < test["timestamp"].min()):
        raise ValueError("Chronological split order failed.")
    return SplitFrames(train=train, validation=validation, test=test, warmup=warmup)


def production_split(
    frame: pd.DataFrame,
    *,
    production_start: str,
    train_length: int,
    validation_length: int,
    warmup_length: int = FEATURE_WARMUP_ROWS,
) -> SplitFrames:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    start = pd.Timestamp(production_start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    history_length = warmup_length + train_length + validation_length
    history = frame.loc[timestamps < start].tail(history_length).reset_index(drop=True)
    test = frame.loc[timestamps >= start].reset_index(drop=True)
    if len(history) < history_length:
        raise ValueError("Not enough pre-production rows for warm-up, train, and validation splits")
    if test.empty:
        raise ValueError("No production rows exist at or after production.start_utc")
    train_start = warmup_length
    validation_start = train_start + train_length
    warmup = history.iloc[:train_start].reset_index(drop=True)
    train = history.iloc[train_start:validation_start].reset_index(drop=True)
    validation = history.iloc[validation_start:].reset_index(drop=True)
    if not (train["timestamp"].max() < validation["timestamp"].min() < test["timestamp"].min()):
        raise ValueError("Production split order failed")
    return SplitFrames(train=train, validation=validation, test=test, warmup=warmup)
