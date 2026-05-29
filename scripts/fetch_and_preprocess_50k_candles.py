"""
fetch_and_preprocess_50k_candles.py

Fetches BTC hourly candles, applies feature engineering, and saves:
- data/btc/btc_40k_candles.csv (40k rows, full hourly series)
- data/btc/btc_40k_market_hours_candles.csv (same features, filtered to ET prediction market hours)

If the full CSV already exists, the script can still backfill the market-hours CSV from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path for src imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.market_data import add_features, fetch_ohlcv
from src.utils.market_hours import is_allowed_prediction_target_timestamp

import pandas as pd

DATA_DIR = Path("data/btc")
CSV_PATH = DATA_DIR / "btc_40k_candles.csv"
MARKET_HOURS_CSV_PATH = DATA_DIR / "btc_40k_market_hours_candles.csv"
CANDLE_LIMIT = 40000
# Fetch extra to account for feature-engineering loss (e.g., rolling windows)
EXTRA_CANDLES = 100
FETCH_LIMIT = CANDLE_LIMIT + EXTRA_CANDLES


def _write_market_hours_csv(full_df: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(full_df["timestamp"], utc=True)
    mask = timestamps.apply(is_allowed_prediction_target_timestamp)
    filtered = full_df.loc[mask].reset_index(drop=True)
    print(f"[INFO] Market-hours rows retained: {len(filtered)} / {len(full_df)}")
    if len(filtered) < 5000 + 5000:
        raise ValueError(
            f"Market-hours filtered frame has only {len(filtered)} rows; need >=10,000 "
            "(5k train pool + 5k test)."
        )
    print(f"[INFO] Saving to {MARKET_HOURS_CSV_PATH} ...")
    filtered.to_csv(MARKET_HOURS_CSV_PATH, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists() and MARKET_HOURS_CSV_PATH.exists():
        print(f"[INFO] {CSV_PATH} and {MARKET_HOURS_CSV_PATH} already exist. Skipping.")
        return

    if CSV_PATH.exists():
        print(f"[INFO] Reusing existing {CSV_PATH} to build market-hours CSV.")
        df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    else:
        print(f"[INFO] Fetching {FETCH_LIMIT} BTC hourly candles...")
        df = fetch_ohlcv(limit=FETCH_LIMIT)
        print(f"[INFO] Fetched {len(df)} rows.")

        print("[INFO] Applying feature engineering...")
        df = add_features(df)
        print(f"[INFO] After feature engineering: {len(df)} rows.")

        if len(df) < CANDLE_LIMIT:
            raise ValueError(f"Not enough rows after feature engineering. Needed {CANDLE_LIMIT}, got {len(df)}.")
        df = df.tail(CANDLE_LIMIT).reset_index(drop=True)

        print(f"[INFO] Saving to {CSV_PATH} ...")
        df.to_csv(CSV_PATH, index=False)

    _write_market_hours_csv(df)
    print("[DONE] Dataset ready.")


if __name__ == "__main__":
    main()
