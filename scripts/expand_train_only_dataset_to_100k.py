"""
Expand the directional-learning BTC dataset by prepending older train rows only.

This script preserves the current 40k dataset tail exactly:
- the existing 35k train tail is kept unchanged
- the existing latest 5k test block is kept unchanged

It writes a new expanded dataset that reaches 100k train rows + 5k test rows
without modifying the current 40k source files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.market_data import add_features, get_exchange_candidates
from src.utils.market_hours import is_allowed_prediction_target_timestamp


DATA_DIR = REPO_ROOT / "data" / "btc"
CURRENT_FULL_PATH = DATA_DIR / "btc_40k_candles.csv"
CURRENT_MARKET_HOURS_PATH = DATA_DIR / "btc_40k_market_hours_candles.csv"
DEFAULT_OUTPUT_FULL_PATH = DATA_DIR / "btc_100k_train_5k_test_candles.csv"
DEFAULT_OUTPUT_MARKET_HOURS_PATH = DATA_DIR / "btc_100k_train_5k_test_market_hours_candles.csv"
DEFAULT_METADATA_PATH = DATA_DIR / "btc_100k_train_5k_test_metadata.json"
RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
TIMEFRAME_MS = 60 * 60 * 1000
DEFAULT_TEST_ROWS = 5000
DEFAULT_TARGET_TRAIN_ROWS = 100000
DEFAULT_FEATURE_BUFFER_ROWS = 256
DEFAULT_BATCH_LIMIT = 1000
HISTORICAL_EXCHANGE_CANDIDATES: list[tuple[str, str, dict[str, Any]]] = [
    ("bitstamp", "BTC/USD", {}),
    ("bitfinex", "BTC/USD", {}),
    ("kraken", "BTC/USD", {}),
    ("gemini", "BTC/USD", {}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepend older BTC candles to extend the train-only side while preserving the current 5k test block."
    )
    parser.add_argument("--target-train-rows", type=int, default=DEFAULT_TARGET_TRAIN_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--feature-buffer-rows", type=int, default=DEFAULT_FEATURE_BUFFER_ROWS)
    parser.add_argument("--batch-limit", type=int, default=DEFAULT_BATCH_LIMIT)
    parser.add_argument("--output-full-path", default=str(DEFAULT_OUTPUT_FULL_PATH))
    parser.add_argument("--output-market-hours-path", default=str(DEFAULT_OUTPUT_MARKET_HOURS_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--skip-market-hours", action="store_true")
    return parser.parse_args()


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True)
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize("UTC")
    return timestamps.dt.tz_convert("UTC")


def _load_existing_full_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Current full dataset not found: {path}")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = _normalize_timestamp(frame["timestamp"])
    return frame.reset_index(drop=True)


def _load_existing_market_hours_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = _normalize_timestamp(frame["timestamp"])
    return frame.reset_index(drop=True)


def _raw_overlap_frame(featured_frame: pd.DataFrame) -> pd.DataFrame:
    overlap = featured_frame.loc[:, RAW_COLUMNS].copy()
    overlap["timestamp"] = _normalize_timestamp(overlap["timestamp"])
    return overlap


def _try_exchange_fetch(
    exchange_id: str,
    symbol: str,
    extra_config: dict[str, Any],
    *,
    since_ms: int,
    limit: int,
) -> list[list[float]]:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "timeout": 30000,
            **extra_config,
        }
    )
    return exchange.fetch_ohlcv(symbol, timeframe="1h", since=since_ms, limit=limit)


def _historical_candidates() -> list[tuple[str, str, dict[str, Any]]]:
    primary_candidates, secondary_candidate, fallback_candidates = get_exchange_candidates()
    ordered: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()

    for exchange_id, symbol, extra_config in [
        *HISTORICAL_EXCHANGE_CANDIDATES,
        *primary_candidates,
        *([secondary_candidate] if secondary_candidate is not None else []),
        *fallback_candidates,
    ]:
        key = (exchange_id, symbol, json.dumps(extra_config, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        ordered.append((exchange_id, symbol, extra_config))
    return ordered


def _fetch_backward_batches(
    exchange_id: str,
    symbol: str,
    extra_config: dict[str, Any],
    *,
    end_exclusive: pd.Timestamp,
    min_rows: int,
    batch_limit: int,
) -> pd.DataFrame:
    target_end_ms = int(end_exclusive.timestamp() * 1000)
    cursor_end_ms = target_end_ms
    rows: list[list[float]] = []
    max_batches = math.ceil(min_rows / batch_limit) + 8

    for _ in range(max_batches):
        since_ms = cursor_end_ms - (batch_limit * TIMEFRAME_MS)
        candles = _try_exchange_fetch(
            exchange_id,
            symbol,
            extra_config,
            since_ms=since_ms,
            limit=batch_limit,
        )
        if not candles:
            break

        filtered = [candle for candle in candles if int(candle[0]) < cursor_end_ms]
        if not filtered:
            break

        rows.extend(filtered)
        earliest_ts = int(filtered[0][0])
        if earliest_ts >= cursor_end_ms:
            break
        cursor_end_ms = earliest_ts

        deduped_count = len({int(row[0]) for row in rows})
        if deduped_count >= min_rows:
            break

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame.loc[frame["timestamp"] < end_exclusive].reset_index(drop=True)


def fetch_historical_ohlcv_before(
    *,
    end_exclusive: pd.Timestamp,
    min_rows: int,
    batch_limit: int,
) -> pd.DataFrame:
    end_exclusive = pd.Timestamp(end_exclusive).tz_convert("UTC")
    failures: list[str] = []

    for exchange_id, symbol, extra_config in _historical_candidates():
        try:
            frame = _fetch_backward_batches(
                exchange_id,
                symbol,
                extra_config,
                end_exclusive=end_exclusive,
                min_rows=min_rows,
                batch_limit=batch_limit,
            )
            if len(frame) < min_rows:
                raise RuntimeError(
                    f"{exchange_id} produced only {len(frame)} pre-anchor candles after backward batching; need {min_rows}."
                )
            print(f"[INFO] Fetched {len(frame)} historical raw candles from {exchange_id} using {symbol}.")
            return frame
        except Exception as exc:
            failures.append(f"{exchange_id}:{symbol}: {exc}")
            print(f"[WARN] Exchange attempt failed for {exchange_id} {symbol}: {exc}")

    raise RuntimeError("Unable to fetch historical OHLCV before anchor.\n" + "\n".join(failures))


def build_market_hours_frame(full_frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = _normalize_timestamp(full_frame["timestamp"])
    mask = timestamps.apply(is_allowed_prediction_target_timestamp)
    return full_frame.loc[mask].reset_index(drop=True)


def verify_tail_preserved(expanded_frame: pd.DataFrame, existing_frame: pd.DataFrame) -> None:
    expanded_tail = expanded_frame.tail(len(existing_frame)).reset_index(drop=True)
    existing_tail = existing_frame.reset_index(drop=True)
    if not expanded_tail.equals(existing_tail):
        raise ValueError("Expanded dataset does not preserve the existing 40k tail exactly.")


def verify_test_block_preserved(expanded_frame: pd.DataFrame, existing_frame: pd.DataFrame, test_rows: int) -> None:
    expanded_test = expanded_frame.tail(test_rows).reset_index(drop=True)
    existing_test = existing_frame.tail(test_rows).reset_index(drop=True)
    if not expanded_test.equals(existing_test):
        raise ValueError("Expanded dataset does not preserve the current 5k test block exactly.")


def verify_market_hours_tail_preserved(expanded_market_hours: pd.DataFrame, existing_market_hours: pd.DataFrame) -> None:
    expanded_tail = expanded_market_hours.tail(len(existing_market_hours)).reset_index(drop=True)
    existing_tail = existing_market_hours.reset_index(drop=True)
    if not expanded_tail.equals(existing_tail):
        raise ValueError("Expanded market-hours dataset does not preserve the existing filtered tail exactly.")


def main() -> None:
    args = parse_args()
    target_train_rows = int(args.target_train_rows)
    test_rows = int(args.test_rows)
    feature_buffer_rows = int(args.feature_buffer_rows)
    batch_limit = int(args.batch_limit)

    output_full_path = Path(args.output_full_path)
    output_market_hours_path = Path(args.output_market_hours_path)
    metadata_path = Path(args.metadata_path)

    existing_full = _load_existing_full_dataset(CURRENT_FULL_PATH)
    existing_market_hours = None if args.skip_market_hours else _load_existing_market_hours_dataset(CURRENT_MARKET_HOURS_PATH)

    current_total_rows = len(existing_full)
    current_train_rows = current_total_rows - test_rows
    if current_train_rows <= 0:
        raise ValueError(
            f"Current dataset has {current_total_rows} rows with test_rows={test_rows}; no train rows remain."
        )
    if target_train_rows <= current_train_rows:
        raise ValueError(
            f"target_train_rows={target_train_rows} must exceed current_train_rows={current_train_rows}."
        )

    prepend_rows_needed = target_train_rows - current_train_rows
    anchor_timestamp = pd.Timestamp(existing_full["timestamp"].iloc[0]).tz_convert("UTC")
    print(f"[INFO] Current dataset rows: total={current_total_rows} train={current_train_rows} test={test_rows}")
    print(f"[INFO] Current train starts at {anchor_timestamp.isoformat()}")
    print(f"[INFO] Need to prepend {prepend_rows_needed} older featured rows.")

    historical_raw = fetch_historical_ohlcv_before(
        end_exclusive=anchor_timestamp,
        min_rows=prepend_rows_needed + feature_buffer_rows,
        batch_limit=batch_limit,
    )

    overlap_raw = _raw_overlap_frame(existing_full).head(1)
    historical_plus_overlap = pd.concat([historical_raw.loc[:, RAW_COLUMNS], overlap_raw], ignore_index=True)
    historical_plus_overlap["timestamp"] = _normalize_timestamp(historical_plus_overlap["timestamp"])
    historical_plus_overlap = historical_plus_overlap.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    print("[INFO] Applying feature engineering to prepended history...")
    historical_featured = add_features(historical_plus_overlap)
    historical_featured["timestamp"] = _normalize_timestamp(historical_featured["timestamp"])
    prepend_featured = historical_featured.loc[historical_featured["timestamp"] < anchor_timestamp].reset_index(drop=True)
    if len(prepend_featured) < prepend_rows_needed:
        raise ValueError(
            f"Only produced {len(prepend_featured)} prependable featured rows; need {prepend_rows_needed}. "
            "Increase feature buffer or fetch depth."
        )

    prepend_featured = prepend_featured.tail(prepend_rows_needed).reset_index(drop=True)
    expanded_full = pd.concat([prepend_featured, existing_full], ignore_index=True)
    if len(expanded_full) != target_train_rows + test_rows:
        raise ValueError(
            f"Expanded full dataset has {len(expanded_full)} rows; expected {target_train_rows + test_rows}."
        )

    verify_tail_preserved(expanded_full, existing_full)
    verify_test_block_preserved(expanded_full, existing_full, test_rows)

    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_full.to_csv(output_full_path, index=False)
    print(f"[INFO] Saved expanded full dataset to {output_full_path}")

    metadata: dict[str, Any] = {
        "source_full_path": str(CURRENT_FULL_PATH.resolve()),
        "output_full_path": str(output_full_path.resolve()),
        "target_train_rows": target_train_rows,
        "test_rows": test_rows,
        "previous_train_rows": current_train_rows,
        "prepended_train_rows": prepend_rows_needed,
        "total_rows": len(expanded_full),
        "current_anchor_timestamp": anchor_timestamp.isoformat(),
        "test_window_start": pd.Timestamp(expanded_full["timestamp"].iloc[-test_rows]).isoformat(),
        "test_window_end": pd.Timestamp(expanded_full["timestamp"].iloc[-1]).isoformat(),
        "preserved_current_40k_tail": True,
        "preserved_test_block": True,
    }

    if not args.skip_market_hours:
        if existing_market_hours is not None:
            prepend_market_hours = build_market_hours_frame(prepend_featured)
            expanded_market_hours = pd.concat([prepend_market_hours, existing_market_hours], ignore_index=True)
            verify_market_hours_tail_preserved(expanded_market_hours, existing_market_hours)
            metadata["preserved_market_hours_tail"] = True
            metadata["prepended_market_hours_rows"] = len(prepend_market_hours)
        else:
            expanded_market_hours = build_market_hours_frame(expanded_full)
        output_market_hours_path.parent.mkdir(parents=True, exist_ok=True)
        expanded_market_hours.to_csv(output_market_hours_path, index=False)
        print(f"[INFO] Saved expanded market-hours dataset to {output_market_hours_path}")
        metadata["output_market_hours_path"] = str(output_market_hours_path.resolve())
        metadata["market_hours_rows"] = len(expanded_market_hours)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
