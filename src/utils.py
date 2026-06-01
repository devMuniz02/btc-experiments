from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIATION_ID = 1
TRAIN_CANDLES = 100000
TEST_CANDLES = 5000
FEATURE_LOOKBACK_PADDING = 200
EXCHANGE_PAGE_LIMIT = 1000
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_8_gap",
    "ema_21_gap",
    "ema_50_gap",
    "bollinger_zscore",
    "bollinger_bandwidth",
    "stochastic_k",
    "stochastic_d",
    "atr_14",
    "atr_pct",
    "volume_delta",
    "volume_zscore_20",
    "price_change_1h",
    "price_change_3h",
    "price_change_6h",
    "price_change_12h",
    "price_change_24h",
    "volatility_6h",
    "volatility_24h",
]
TEMPORAL_FEATURE_COLUMNS = [
    "day_of_week",
    "hour_of_day",
    "day_of_month_sin",
    "day_of_month_cos",
]


@dataclass(frozen=True)
class QuantStreamPaths:
    root: Path = PROJECT_ROOT

    @property
    def automation_dir(self) -> Path:
        return self.root / "automation"

    @property
    def run_requests_dir(self) -> Path:
        return self.automation_dir / "run_requests"

    @property
    def runs_done_dir(self) -> Path:
        return self.automation_dir / "runs_done"

    @property
    def delete_requests_dir(self) -> Path:
        return self.automation_dir / "delete_requests"

    @property
    def deleted_runs_dir(self) -> Path:
        return self.automation_dir / "deleted_runs"

    @property
    def rejected_runs_dir(self) -> Path:
        return self.automation_dir / "rejected_runs"

    @property
    def models_dev_dir(self) -> Path:
        return self.root / "models" / "dev"

    @property
    def models_prod_dir(self) -> Path:
        return self.root / "models" / "prod"

    @property
    def scalers_dir(self) -> Path:
        return self.root / "scalers"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    def variation_name(self, variation_id: int | str) -> str:
        text = str(variation_id).strip()
        return text if text.startswith("var_") else f"var_{int(text)}"

    def variation_data_dir(self, variation_id: int | str) -> Path:
        return self.data_dir / self.variation_name(variation_id)

    def variation_models_dir(self, variation_id: int | str) -> Path:
        return self.models_dev_dir / self.variation_name(variation_id)

    def variation_scalers_dir(self, variation_id: int | str) -> Path:
        return self.scalers_dir / self.variation_name(variation_id)

    def global_results_path(self, variation_id: int | str) -> Path:
        return self.variation_data_dir(variation_id) / "global_results.parquet"

    def clean_data_path(self, variation_id: int | str) -> Path:
        return self.variation_data_dir(variation_id) / "btc_1h_clean.parquet"


def ensure_quant_stream_layout(paths: QuantStreamPaths | None = None) -> None:
    resolved = paths or QuantStreamPaths()
    directories = [
        resolved.run_requests_dir,
        resolved.runs_done_dir,
        resolved.delete_requests_dir,
        resolved.deleted_runs_dir,
        resolved.rejected_runs_dir,
        resolved.variation_data_dir(DEFAULT_VARIATION_ID),
        resolved.variation_models_dir(DEFAULT_VARIATION_ID),
        resolved.variation_scalers_dir(DEFAULT_VARIATION_ID),
        resolved.models_prod_dir,
        resolved.scalers_dir / "prod",
    ]
    directories.extend(resolved.models_prod_dir / f"model_slot_{index}" for index in range(1, 6))
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_model_id(payload: dict[str, Any]) -> str:
    identity = {
        "model_type": str(payload["model_type"]).lower(),
        "variation_id": int(payload["variation_id"]),
        "train_mode": str(payload["train_mode"]).lower(),
        "data": payload.get("data", {}),
        "hyperparameters": payload["hyperparameters"],
    }
    return hashlib.sha256(canonicalize_payload(identity).encode("utf-8")).hexdigest()[:24]


def read_yaml_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Request YAML must contain a mapping at the document root.")
    return payload


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_timestamp(value: str | None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value) if value else pd.Timestamp.now(tz="UTC").floor("h")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def requested_bounds(test_start_date: str | None) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    anchor = parse_timestamp(test_start_date)
    padded_start = anchor - pd.Timedelta(hours=TRAIN_CANDLES + FEATURE_LOOKBACK_PADDING)
    end = anchor + pd.Timedelta(hours=TEST_CANDLES - 1)
    return anchor, padded_start, end


def fetch_ohlcv_range(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange_id: str = "binance",
    symbol: str = SYMBOL,
) -> pd.DataFrame:
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30000})
    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[Any]] = []
    while since_ms <= end_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=EXCHANGE_PAGE_LIMIT)
        if not candles:
            break
        rows.extend(candles)
        last_ms = int(candles[-1][0])
        next_ms = last_ms + int(pd.Timedelta(hours=1).total_seconds() * 1000)
        if next_ms <= since_ms:
            break
        since_ms = next_ms
        if last_ms >= end_ms:
            break
    if not rows:
        raise RuntimeError(f"No candles fetched for {symbol} from {start.isoformat()} to {end.isoformat()}.")
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    mask = (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
    return frame.loc[mask].reset_index(drop=True)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    return 100 - (100 / (1 + rs))


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = pd.to_datetime(enriched["timestamp"], utc=True)
    enriched["day_of_week"] = timestamps.dt.dayofweek.astype(np.float32)
    enriched["hour_of_day"] = timestamps.dt.hour.astype(np.float32)
    day_of_month = timestamps.dt.day.astype(np.float32)
    enriched["day_of_month_sin"] = np.sin((2.0 * np.pi * day_of_month) / 31.0).astype(np.float32)
    enriched["day_of_month_cos"] = np.cos((2.0 * np.pi * day_of_month) / 31.0).astype(np.float32)
    return enriched


def add_quant_stream_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]
    volume = df["volume"]
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    rolling_mean_20 = close.rolling(window=20, min_periods=20).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=20).std()
    lowest_low_14 = df["low"].rolling(window=14, min_periods=14).min()
    highest_high_14 = df["high"].rolling(window=14, min_periods=14).max()
    stochastic_k = 100 * (close - lowest_low_14) / (highest_high_14 - lowest_low_14)
    stochastic_d = stochastic_k.rolling(window=3, min_periods=3).mean()
    atr_14 = compute_atr(df, 14)
    volume_mean_20 = volume.rolling(window=20, min_periods=20).mean()
    volume_std_20 = volume.rolling(window=20, min_periods=20).std()
    df["rsi_14"] = compute_rsi(close, 14)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal
    df["ema_8_gap"] = (close / ema_8) - 1.0
    df["ema_21_gap"] = (close / ema_21) - 1.0
    df["ema_50_gap"] = (close / ema_50) - 1.0
    df["bollinger_zscore"] = (close - rolling_mean_20) / rolling_std_20
    df["bollinger_bandwidth"] = (2 * rolling_std_20) / rolling_mean_20
    df["stochastic_k"] = stochastic_k
    df["stochastic_d"] = stochastic_d
    df["atr_14"] = atr_14
    df["atr_pct"] = atr_14 / close
    df["volume_delta"] = volume.diff()
    df["volume_zscore_20"] = (volume - volume_mean_20) / volume_std_20
    df["price_change_1h"] = close.pct_change(1)
    df["price_change_3h"] = close.pct_change(3)
    df["price_change_6h"] = close.pct_change(6)
    df["price_change_12h"] = close.pct_change(12)
    df["price_change_24h"] = close.pct_change(24)
    df["volatility_6h"] = df["price_change_1h"].rolling(window=6, min_periods=6).std()
    df["volatility_24h"] = df["price_change_1h"].rolling(window=24, min_periods=24).std()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = add_temporal_features(df)
    required_columns = ["timestamp", *FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS, "target"]
    return df[required_columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def build_clean_matrix(raw_frame: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    featured = add_quant_stream_features(raw_frame)
    featured["timestamp"] = pd.to_datetime(featured["timestamp"], utc=True)
    start = test_start - pd.Timedelta(hours=TRAIN_CANDLES)
    end = test_start + pd.Timedelta(hours=TEST_CANDLES - 1)
    matrix = featured.loc[(featured["timestamp"] >= start) & (featured["timestamp"] <= end)].reset_index(drop=True)
    expected = TRAIN_CANDLES + TEST_CANDLES
    if len(matrix) != expected:
        raise RuntimeError(f"Expected {expected} clean rows for anchor {test_start.isoformat()}, found {len(matrix)}.")
    return matrix


def ingest_variation(
    *,
    variation_id: int,
    test_start_date: str | None,
    root: Path | None = None,
    exchange_id: str = "binance",
) -> Path:
    paths = QuantStreamPaths(root=root or PROJECT_ROOT)
    ensure_quant_stream_layout(paths)
    test_start, padded_start, end = requested_bounds(test_start_date)
    raw = fetch_ohlcv_range(start=padded_start, end=end, exchange_id=exchange_id)
    clean = build_clean_matrix(raw, test_start)
    output_path = paths.clean_data_path(variation_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False, compression="zstd")
    return output_path
