from __future__ import annotations

import hashlib
import json
import math
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
BACKFILL_EXTRA_ROWS = 1000
EXCHANGE_PAGE_LIMIT = 1000
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LEGACY_VAR_1_CSV = PROJECT_ROOT / "legacy_unused" / "data" / "btc" / "btc_100k_train_5k_test_candles.csv"
TIMEFRAME_MS = 60 * 60 * 1000
HISTORICAL_BACKFILL_CANDIDATES: list[tuple[str, str]] = [
    ("bitstamp", "BTC/USD"),
    ("bitfinex", "BTC/USD"),
    ("kraken", "BTC/USD"),
    ("gemini", "BTC/USD"),
]

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
    try:
        import ccxt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'ccxt'. Install it with "
            "`conda run -n btc-quant-stream python -m pip install ccxt` "
            "or rerun `powershell -ExecutionPolicy Bypass -File .\\setup_repo.ps1`."
        ) from exc

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


def fetch_ohlcv_backward_before(
    *,
    end_exclusive: pd.Timestamp,
    min_rows: int,
    exchange_id: str,
    symbol: str,
    batch_limit: int = EXCHANGE_PAGE_LIMIT,
) -> pd.DataFrame:
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30000})
    cursor_end_ms = int(pd.Timestamp(end_exclusive).timestamp() * 1000)
    rows: list[list[Any]] = []
    max_batches = math.ceil(int(min_rows) / int(batch_limit)) + 8

    for _ in range(max_batches):
        since_ms = cursor_end_ms - (int(batch_limit) * TIMEFRAME_MS)
        candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=int(batch_limit))
        if not candles:
            break
        filtered = [candle for candle in candles if int(candle[0]) < cursor_end_ms]
        if not filtered:
            break
        rows.extend(filtered)
        earliest_ms = int(filtered[0][0])
        if earliest_ms >= cursor_end_ms:
            break
        cursor_end_ms = earliest_ms
        if len({int(row[0]) for row in rows}) >= int(min_rows):
            break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame.loc[frame["timestamp"] < end_exclusive].reset_index(drop=True)


def fetch_historical_backfill_before(
    *,
    end_exclusive: pd.Timestamp,
    min_rows: int,
    preferred_exchange_id: str,
    preferred_symbol: str,
) -> pd.DataFrame:
    candidates = [(preferred_exchange_id, preferred_symbol), *HISTORICAL_BACKFILL_CANDIDATES]
    seen: set[tuple[str, str]] = set()
    failures: list[str] = []
    for exchange_id, symbol in candidates:
        key = (exchange_id, symbol)
        if key in seen:
            continue
        seen.add(key)
        try:
            frame = fetch_ohlcv_backward_before(
                end_exclusive=end_exclusive,
                min_rows=min_rows,
                exchange_id=exchange_id,
                symbol=symbol,
            )
            if len(frame) < min_rows:
                raise RuntimeError(
                    f"{exchange_id} produced only {len(frame)} pre-anchor candles after backward batching; "
                    f"need {min_rows}."
                )
            print(f"Fetched {len(frame)} historical backfill candles from {exchange_id} using {symbol}.")
            return frame
        except Exception as exc:
            failures.append(f"{exchange_id}:{symbol}: {exc}")
            print(f"Backfill exchange attempt failed for {exchange_id} {symbol}: {exc}")
    raise RuntimeError("Unable to fetch historical backfill candles before anchor.\n" + "\n".join(failures))


def fetch_ohlcv_range_with_backfill(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    primary_exchange_id: str,
    primary_symbol: str,
    backfill_exchange_id: str,
    backfill_symbol: str,
) -> pd.DataFrame:
    primary = fetch_ohlcv_range(
        start=start,
        end=end,
        exchange_id=primary_exchange_id,
        symbol=primary_symbol,
    )
    if primary.empty:
        return fetch_ohlcv_range(
            start=start,
            end=end,
            exchange_id=backfill_exchange_id,
            symbol=backfill_symbol,
        )

    primary_start = pd.Timestamp(primary["timestamp"].min()).tz_convert("UTC")
    if primary_start <= start:
        return primary

    required_backfill_rows = int((primary_start - start) / pd.Timedelta(hours=1)) + BACKFILL_EXTRA_ROWS
    backfill = fetch_historical_backfill_before(
        end_exclusive=primary_start,
        min_rows=required_backfill_rows,
        preferred_exchange_id=backfill_exchange_id,
        preferred_symbol=backfill_symbol,
    )
    combined = pd.concat([backfill, primary], ignore_index=True)
    return combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


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
    expected = TRAIN_CANDLES + TEST_CANDLES
    anchor_matches = featured.index[featured["timestamp"] == test_start].to_numpy()
    if len(anchor_matches) == 0:
        earliest = featured["timestamp"].min()
        latest = featured["timestamp"].max()
        raise RuntimeError(
            f"Fetched data does not contain test anchor {test_start.isoformat()}. "
            f"Fetched exchange history after feature creation spans {earliest.isoformat()} to {latest.isoformat()}."
        )
    anchor_index = int(anchor_matches[0])
    start_index = anchor_index - TRAIN_CANDLES
    end_index = anchor_index + TEST_CANDLES
    if start_index < 0 or end_index > len(featured):
        earliest = featured["timestamp"].min()
        latest = featured["timestamp"].max()
        available_train = max(0, anchor_index)
        available_test = max(0, len(featured) - anchor_index)
        raise RuntimeError(
            f"Expected {TRAIN_CANDLES} train rows before anchor and {TEST_CANDLES} test rows from anchor, "
            f"but found {available_train} train rows and {available_test} test rows. "
            f"Fetched exchange history after feature creation spans {earliest.isoformat()} to {latest.isoformat()}."
        )
    matrix = featured.iloc[start_index:end_index].reset_index(drop=True)
    if len(matrix) != expected:
        earliest = featured["timestamp"].min()
        latest = featured["timestamp"].max()
        raise RuntimeError(
            f"Expected {expected} clean rows for anchor {test_start.isoformat()}, found {len(matrix)}. "
            f"The fetched exchange history after feature creation spans {earliest.isoformat()} "
            f"to {latest.isoformat()}. "
            "Use an exchange/symbol with enough BTC candles around the anchor, or pass a different anchor."
        )
    observed_anchor = pd.Timestamp(matrix["timestamp"].iloc[TRAIN_CANDLES]).tz_convert("UTC")
    if observed_anchor != test_start:
        raise RuntimeError(
            f"Clean matrix anchor mismatch: expected {test_start.isoformat()}, found {observed_anchor.isoformat()}."
        )
    return matrix


def load_legacy_var_1_matrix(*, source_path: Path, test_start: pd.Timestamp) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Legacy reproducibility CSV not found: {source_path}")
    frame = pd.read_csv(source_path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    expected = TRAIN_CANDLES + TEST_CANDLES
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} legacy rows in {source_path}, found {len(frame)}.")
    observed_anchor = pd.Timestamp(frame["timestamp"].iloc[TRAIN_CANDLES]).tz_convert("UTC")
    if observed_anchor != test_start:
        raise RuntimeError(
            f"Legacy CSV test anchor mismatch: expected {test_start.isoformat()}, "
            f"found {observed_anchor.isoformat()} at row {TRAIN_CANDLES}."
        )
    frame = add_temporal_features(frame)
    required_columns = ["timestamp", *FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS, "target"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Legacy CSV is missing required columns after temporal feature enrichment: {missing}")
    return frame[required_columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def ingest_variation(
    *,
    variation_id: int,
    test_start_date: str | None,
    root: Path | None = None,
    exchange_id: str = "binance",
    symbol: str = SYMBOL,
    backfill_exchange_id: str = "kraken",
    backfill_symbol: str = "BTC/USD",
    source: str = "auto",
    source_path: Path | None = None,
) -> Path:
    paths = QuantStreamPaths(root=root or PROJECT_ROOT)
    ensure_quant_stream_layout(paths)
    test_start, padded_start, end = requested_bounds(test_start_date)
    resolved_source = str(source or "auto").strip().lower()
    legacy_source_path = source_path or LEGACY_VAR_1_CSV
    if resolved_source not in {"auto", "exchange", "legacy"}:
        raise ValueError("source must be one of: auto, exchange, legacy.")
    if resolved_source == "legacy":
        clean = load_legacy_var_1_matrix(source_path=legacy_source_path, test_start=test_start)
    else:
        raw = fetch_ohlcv_range_with_backfill(
            start=padded_start,
            end=end,
            primary_exchange_id=exchange_id,
            primary_symbol=symbol,
            backfill_exchange_id=backfill_exchange_id,
            backfill_symbol=backfill_symbol,
        )
        clean = build_clean_matrix(raw, test_start)
    output_path = paths.clean_data_path(variation_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False, compression="zstd")
    return output_path


def parse_ingest_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a Quant-Stream clean BTC variation dataset.")
    parser.add_argument("command", choices=["ingest"], help="Dataset utility command.")
    parser.add_argument("--variation_id", type=int, default=DEFAULT_VARIATION_ID)
    parser.add_argument("--test_start_date", default="2025-10-08 15:00:00+00:00")
    parser.add_argument("--exchange_id", default="binance")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--backfill_exchange_id", default="kraken")
    parser.add_argument("--backfill_symbol", default="BTC/USD")
    parser.add_argument("--source", choices=["auto", "exchange", "legacy"], default="auto")
    parser.add_argument("--source_path", default="")
    parser.add_argument("--root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_ingest_args()
    if args.command == "ingest":
        output_path = ingest_variation(
            variation_id=int(args.variation_id),
            test_start_date=str(args.test_start_date),
            root=Path(args.root).resolve() if args.root else None,
            exchange_id=str(args.exchange_id),
            symbol=str(args.symbol),
            backfill_exchange_id=str(args.backfill_exchange_id),
            backfill_symbol=str(args.backfill_symbol),
            source=str(args.source),
            source_path=Path(args.source_path).resolve() if args.source_path else None,
        )
        print(str(output_path))
        return 0
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
