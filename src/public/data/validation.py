from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SchemaValidationResult:
    row_count: int
    columns: tuple[str, ...]
    start_timestamp: str
    end_timestamp: str


def validate_ohlcv_schema(frame: pd.DataFrame, *, min_rows: int = 32) -> SchemaValidationResult:
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV data missing required columns: {', '.join(missing)}")
    if len(frame) < min_rows:
        raise ValueError(f"OHLCV data needs at least {min_rows} rows, got {len(frame)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be chronological.")
    if data["timestamp"].duplicated().any():
        raise ValueError("OHLCV timestamps must be unique.")
    for column in ("open", "high", "low", "close", "volume"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{column} contains non-numeric values.")
    return SchemaValidationResult(
        row_count=len(data),
        columns=tuple(str(column) for column in data.columns),
        start_timestamp=data["timestamp"].iloc[0].isoformat(),
        end_timestamp=data["timestamp"].iloc[-1].isoformat(),
    )
