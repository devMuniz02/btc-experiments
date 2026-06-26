from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np

from src.public.evaluation.metrics import classification_metrics


def _utc_timestamp(value: Any) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def select_top_candidate_ids(ranked_rows: Iterable[dict[str, Any]], limit: int = 3) -> list[str]:
    """Return the validation-ranked candidate IDs without consulting test metrics."""
    if limit < 1:
        return []
    return [str(row["candidate_id"]) for row in ranked_rows if row.get("candidate_id")][:limit]


def build_public_prediction_result(
    *,
    timestamps: Iterable[Any],
    y_true: Iterable[int],
    predictions: Iterable[int],
    probabilities: Iterable[float],
    returns: Iterable[float],
    latest_public_window: str,
    earliest_public_window: str | None = None,
) -> dict[str, Any] | None:
    """Build metrics and a zero-based cumulative-return series for the public window."""
    timestamp_values = list(timestamps)
    true = np.asarray(list(y_true), dtype=int)
    predicted = np.asarray(list(predictions), dtype=int)
    probability = np.asarray(list(probabilities), dtype=float)
    returns_array = np.asarray(list(returns), dtype=float)
    lengths = {len(timestamp_values), len(true), len(predicted), len(probability), len(returns_array)}
    if len(lengths) != 1:
        raise ValueError("Prediction arrays and timestamps must have identical lengths")
    cutoff = _utc_timestamp(latest_public_window)
    start = _utc_timestamp(earliest_public_window) if earliest_public_window else None
    mask = np.asarray(
        [
            (start is None or _utc_timestamp(value) >= start) and _utc_timestamp(value) <= cutoff
            for value in timestamp_values
        ],
        dtype=bool,
    )
    if not mask.any():
        return None

    selected_timestamps = [value for value, include in zip(timestamp_values, mask, strict=True) if include]
    selected_true = true[mask]
    selected_predicted = predicted[mask]
    selected_probability = probability[mask]
    selected_returns = returns_array[mask]
    metrics = classification_metrics(
        selected_true,
        selected_predicted,
        returns=selected_returns,
        probabilities=selected_probability,
    )
    direction_scores = np.where(selected_predicted == selected_true, 1.0, -1.0)
    cumulative = np.cumsum(direction_scores)
    metrics["cumulative_return"] = float(cumulative[-1])
    if len(selected_timestamps) > 1:
        step = _utc_timestamp(selected_timestamps[1]) - _utc_timestamp(selected_timestamps[0])
    else:
        step = timedelta(seconds=1)
    baseline_timestamp = start if start is not None else _utc_timestamp(selected_timestamps[0]) - step
    series = [{"timestamp": baseline_timestamp.isoformat(), "performance": 0.0}] + [
        {
            "timestamp": _utc_timestamp(timestamp).isoformat(),
            "performance": round(float(value), 10),
        }
        for timestamp, value in zip(selected_timestamps, cumulative, strict=True)
    ]
    return {"metrics": metrics, "prediction_series": series}
