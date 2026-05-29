#!/usr/bin/env python3
"""
Shared ET market-hours gating helpers for isolated BTC workflows.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

EASTERN_TZ = ZoneInfo("America/New_York")
TRAIN_START_HOUR_ET = 7
TRAIN_END_HOUR_ET = 19
PREDICTION_START_HOUR_ET = 8
PREDICTION_END_HOUR_ET = 20


def normalize_timestamp(value: pd.Timestamp | str | None = None) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if current.tzinfo is None:
        return current.tz_localize("UTC")
    return current.tz_convert("UTC")


def current_et_timestamp(value: pd.Timestamp | str | None = None) -> pd.Timestamp:
    return normalize_timestamp(value).tz_convert(EASTERN_TZ)


def next_target_timestamp_utc(value: pd.Timestamp | str | None = None) -> pd.Timestamp:
    current_utc = normalize_timestamp(value)
    return current_utc.floor("h") + pd.Timedelta(hours=1)


def next_target_timestamp_et(value: pd.Timestamp | str | None = None) -> pd.Timestamp:
    return next_target_timestamp_utc(value).tz_convert(EASTERN_TZ)


def is_allowed_prediction_target_timestamp(value: pd.Timestamp | str) -> bool:
    target = pd.Timestamp(value)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    target_et = target.tz_convert(EASTERN_TZ)
    return PREDICTION_START_HOUR_ET <= target_et.hour <= PREDICTION_END_HOUR_ET


def should_run_prediction_window(value: pd.Timestamp | str | None = None) -> bool:
    return is_allowed_prediction_target_timestamp(next_target_timestamp_utc(value))


def should_run_training_window(value: pd.Timestamp | str | None = None) -> bool:
    current_et = current_et_timestamp(value)
    return TRAIN_START_HOUR_ET <= current_et.hour <= TRAIN_END_HOUR_ET


def describe_window(value: pd.Timestamp | str | None = None) -> str:
    current_et = current_et_timestamp(value)
    target_et = next_target_timestamp_et(value)
    return (
        f"current_et={current_et.isoformat()} "
        f"target_et={target_et.isoformat()} "
        f"train_hours={TRAIN_START_HOUR_ET:02d}:00-{TRAIN_END_HOUR_ET:02d}:59 "
        f"prediction_target_hours={PREDICTION_START_HOUR_ET:02d}:00-{PREDICTION_END_HOUR_ET:02d}:59"
    )
