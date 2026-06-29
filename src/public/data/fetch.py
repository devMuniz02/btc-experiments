from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.public.data.split import FEATURE_WARMUP_ROWS

LIVE_DATA_ROOT = "private_hf_cache/live_data"
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

BINANCE_CANDIDATES = (
    ("binance", "binance", {}),
    ("binance_futures", "binance", {"options": {"defaultType": "future"}}),
)
FALLBACK_EXCHANGE_CANDIDATES = (
    ("binanceus", "binanceus", {}),
    ("kraken", "kraken", {}),
    ("okx", "okx", {}),
    ("kucoin", "kucoin", {}),
    ("bitfinex", "bitfinex", {}),
)


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    provider_used: str
    fallback_reasons: tuple[str, ...]
    cache_path: Path | None = None
    cache_status: str = "miss"


def _rows_needed(config: dict[str, Any]) -> int:
    split = config["split"]
    return (
        historical_context_rows(config)
        + int(split["train_length"])
        + int(split["validation_length"])
        + int(split["test_length"])
        + max_target_horizon(config)
        + 1
    )


def max_sequence_length(config: dict[str, Any]) -> int:
    configured = [int(value) for value in config.get("experiments", {}).get("sequence_lengths") or [1]]
    if config.get("experiments", {}).get("workflow_profile") == "exhaustive_v1":
        configured.extend([1, 12, 24, 48, 72, 96, 168, 240])
    return max(configured)


def max_feature_window(config: dict[str, Any]) -> int:
    windows = [FEATURE_WARMUP_ROWS, 6, 24]
    experiments = config.get("experiments", {})
    if experiments.get("workflow_profile") == "exhaustive_v1":
        windows.extend([96, 168])
    if any(str(value).lower() in {"ema", "ema_low_pass"} for value in experiments.get("denoising") or []):
        windows.append(FEATURE_WARMUP_ROWS)
    return max(windows)


def max_target_horizon(config: dict[str, Any]) -> int:
    if config.get("experiments", {}).get("workflow_profile") == "exhaustive_v1":
        return 24
    return 0


def historical_context_rows(config: dict[str, Any]) -> int:
    return max(max_sequence_length(config), max_feature_window(config))


def _market_id(config: dict[str, Any]) -> str:
    return str(config["market"]["market_id"]).strip()


def _timeframe(config: dict[str, Any]) -> str:
    return str(config["market"]["timeframe"]).strip()


def _timeframe_delta(config: dict[str, Any]) -> pd.Timedelta:
    values = {"5min": "5min", "15min": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
    return pd.Timedelta(values[_timeframe(config)])


def _spot_symbol(config: dict[str, Any]) -> str:
    configured = str(config["market"].get("symbol") or "BTCUSDT").strip().upper()
    if "/" in configured:
        return configured
    if configured.endswith("USDT") and len(configured) > 4:
        return f"{configured[:-4]}/USDT"
    return "BTC/USDT"


def _symbol_for_provider(provider_used: str, config: dict[str, Any]) -> str:
    spot = _spot_symbol(config)
    if provider_used == "binance_futures" and spot.endswith("/USDT"):
        return f"{spot}:USDT"
    return spot


def live_cache_path(root: Path, *, market_id: str, timeframe: str) -> Path:
    return root / LIVE_DATA_ROOT / market_id / f"{timeframe}.parquet"


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.loc[:, OHLCV_COLUMNS].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
    return normalized.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _read_cache(cache_path: Path, *, rows_needed: int, fallback_reasons: list[str]) -> FetchResult | None:
    if not cache_path.exists():
        fallback_reasons.append(f"live_cache_missing:{cache_path.as_posix()}")
        return None
    try:
        frame = _normalize_ohlcv_frame(pd.read_parquet(cache_path))
    except Exception as exc:
        fallback_reasons.append(f"live_cache_unreadable:{type(exc).__name__}:{exc}")
        return None
    if len(frame) < rows_needed:
        fallback_reasons.append(f"live_cache_too_short:{len(frame)}<{rows_needed}")
        return None

    provider_used = "binance"
    meta_path = _cache_meta_path(cache_path)
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            provider_used = str(metadata.get("provider_used") or provider_used)
        except json.JSONDecodeError as exc:
            fallback_reasons.append(f"live_cache_metadata_unreadable:{type(exc).__name__}:{exc}")
    else:
        fallback_reasons.append("live_cache_metadata_missing_assumed_binance")
    return FetchResult(
        frame.tail(rows_needed).reset_index(drop=True),
        provider_used,
        tuple(fallback_reasons),
        cache_path=cache_path,
        cache_status="hit",
    )


def _write_cache(cache_path: Path, frame: pd.DataFrame, *, provider_used: str, config: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_ohlcv_frame(frame)
    normalized.to_parquet(cache_path, index=False)
    metadata = {
        "market_id": _market_id(config),
        "timeframe": _timeframe(config),
        "provider_used": provider_used,
        "rows": len(normalized),
        "first_timestamp": str(normalized["timestamp"].iloc[0]) if len(normalized) else None,
        "last_timestamp": str(normalized["timestamp"].iloc[-1]) if len(normalized) else None,
        "public_safe": False,
    }
    _cache_meta_path(cache_path).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _exchange_ohlcv_frame(
    exchange: Any,
    *,
    symbol: str,
    timeframe: str,
    limit: int,
    since_ms: int | None = None,
) -> pd.DataFrame:
    timeframe_ms = int(pd.Timedelta(timeframe).total_seconds() * 1000)
    current_since = since_ms
    if current_since is None and limit > 1000:
        current_since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(timeframe) * (limit + 5)).timestamp() * 1000)
    rows: list[list[Any]] = []
    seen: set[int] = set()
    per_call_limit = min(1000, max(1, limit))
    while len(rows) < limit:
        batch_limit = min(per_call_limit, limit - len(rows))
        kwargs = {"since": current_since} if current_since is not None else {}
        no_since_fallback = False
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=batch_limit, params={}, **kwargs)
        except TypeError:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params={})
            no_since_fallback = True
        if not batch:
            break
        added = 0
        last_timestamp = current_since or 0
        for row in batch:
            timestamp = int(row[0])
            last_timestamp = max(last_timestamp, timestamp)
            if timestamp not in seen:
                seen.add(timestamp)
                rows.append(row)
                added += 1
        if current_since is None or added == 0 or no_since_fallback:
            break
        next_since = last_timestamp + timeframe_ms
        if next_since <= current_since:
            break
        current_since = next_since
        rate_limit = float(getattr(exchange, "rateLimit", 0) or 0)
        if rate_limit > 0:
            time.sleep(rate_limit / 1000.0)
    if not rows:
        raise RuntimeError("empty OHLCV response")
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return _normalize_ohlcv_frame(frame)


def _fetch_provider(
    *,
    ccxt_module: Any,
    provider_used: str,
    exchange_id: str,
    exchange_config: dict[str, Any],
    config: dict[str, Any],
    limit: int,
    since_ms: int | None = None,
) -> pd.DataFrame:
    exchange_class = getattr(ccxt_module, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30000, **exchange_config})
    try:
        frame = _exchange_ohlcv_frame(
            exchange,
            symbol=_symbol_for_provider(provider_used, config),
            timeframe=_timeframe(config),
            limit=limit,
            since_ms=since_ms,
        )
    finally:
        close = getattr(exchange, "close", None)
        if callable(close):
            close()
    if len(frame) < limit:
        raise RuntimeError(f"insufficient live candles:{len(frame)}<{limit}")
    return frame.tail(limit).reset_index(drop=True)


def _attempt_provider(
    *,
    ccxt_module: Any,
    provider_used: str,
    exchange_id: str,
    exchange_config: dict[str, Any],
    config: dict[str, Any],
    limit: int,
    since_ms: int | None,
    attempts: int,
    backoff_seconds: float,
    fallback_reasons: list[str],
) -> pd.DataFrame | None:
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return _fetch_provider(
                ccxt_module=ccxt_module,
                provider_used=provider_used,
                exchange_id=exchange_id,
                exchange_config=exchange_config,
                config=config,
                limit=limit,
                since_ms=since_ms,
            )
        except Exception as exc:
            fallback_reasons.append(f"{provider_used}:attempt_{attempt}:{type(exc).__name__}:{exc}")
            if attempt < attempts and backoff_seconds > 0:
                time.sleep(backoff_seconds)
    return None


def fetch_market_data(
    config: dict[str, Any],
    *,
    root: Path,
    cache_namespace: str | None = None,
    end_utc: str | None = None,
    production_start_utc: str | None = None,
    post_end_rows: int = 0,
    pre_end_rows: int = 0,
) -> FetchResult:
    rows_needed = _rows_needed(config)
    extra_rows = max(0, int(post_end_rows))
    pre_rows = max(0, int(pre_end_rows))
    cache_path = live_cache_path(root, market_id=cache_namespace or _market_id(config), timeframe=_timeframe(config))
    fallback_reasons: list[str] = []

    cached_frame: pd.DataFrame | None = None
    cached_provider = "binance"
    if cache_path.exists():
        try:
            cached_frame = _normalize_ohlcv_frame(pd.read_parquet(cache_path))
            meta_path = _cache_meta_path(cache_path)
            if meta_path.exists():
                cached_provider = str(json.loads(meta_path.read_text(encoding="utf-8")).get("provider_used") or cached_provider)
        except Exception as exc:
            fallback_reasons.append(f"live_cache_unreadable:{type(exc).__name__}:{exc}")
            cached_frame = None

    if end_utc:
        end = pd.Timestamp(end_utc)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        delta = _timeframe_delta(config)
        base_end = end.floor(delta)
        if pre_rows:
            return_end = base_end
            required_rows = rows_needed + pre_rows
            fetch_end = return_end
            live_gap_rows = 0
            total_fetch_rows = required_rows
        else:
            now = pd.Timestamp.now(tz="UTC").floor(delta)
            available_extra_rows = min(extra_rows, max(0, int((now - base_end) / delta)))
            return_end = base_end + delta * available_extra_rows
            required_rows = rows_needed + available_extra_rows
            fetch_end = max(return_end, now)
            live_gap_rows = max(0, int((fetch_end - return_end) / delta))
            total_fetch_rows = required_rows + live_gap_rows
        bounded = cached_frame.loc[cached_frame["timestamp"] <= return_end] if cached_frame is not None else None
        cache_covers_live_gap = (
            cached_frame is not None
            and not cached_frame.empty
            and cached_frame["timestamp"].max() >= fetch_end - delta
        )
        if bounded is not None and len(bounded) >= required_rows and (
            extra_rows == 0 or live_gap_rows == 0 or cache_covers_live_gap
        ):
            return FetchResult(
                bounded.tail(required_rows).reset_index(drop=True),
                cached_provider,
                tuple(fallback_reasons),
                cache_path=cache_path,
                cache_status="hit",
            )
        fetch_start = fetch_end - delta * (total_fetch_rows - 1)
        fetch_limit = total_fetch_rows
    elif production_start_utc:
        start = pd.Timestamp(production_start_utc)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        history_rows = (
            historical_context_rows(config)
            + int(config["split"]["train_length"])
            + int(config["split"]["validation_length"])
        )
        required_start = start - _timeframe_delta(config) * history_rows
        now = pd.Timestamp.now(tz="UTC").floor(_timeframe_delta(config))
        if cached_frame is not None and not cached_frame.empty and cached_frame["timestamp"].min() <= required_start:
            fetch_start = cached_frame["timestamp"].max()
            if fetch_start > now:
                return FetchResult(
                    cached_frame.loc[cached_frame["timestamp"] >= required_start].reset_index(drop=True),
                    cached_provider,
                    tuple(fallback_reasons),
                    cache_path=cache_path,
                    cache_status="hit",
                )
        else:
            fetch_start = required_start
        fetch_limit = max(1, int((now - fetch_start) / _timeframe_delta(config)) + 1)
    else:
        cached = _read_cache(cache_path, rows_needed=rows_needed, fallback_reasons=fallback_reasons)
        if cached is not None:
            return cached
        fetch_start = None
        fetch_limit = rows_needed

    fetch = config.get("fetch", {})
    max_retries = int(fetch.get("max_retries", 3))
    backoff_seconds = float(fetch.get("retry_backoff_seconds", 0))
    try:
        import ccxt
    except ModuleNotFoundError as exc:
        fallback_reasons.append("ccxt_missing")
        raise RuntimeError("Real OHLCV fetch requires ccxt and no usable private live cache was found.") from exc

    candidates = (*BINANCE_CANDIDATES, *FALLBACK_EXCHANGE_CANDIDATES)
    for provider_used, exchange_id, exchange_config in candidates:
        attempts = max_retries if provider_used == "binanceus" else 1
        frame = _attempt_provider(
            ccxt_module=ccxt,
            provider_used=provider_used,
            exchange_id=exchange_id,
            exchange_config=exchange_config,
            config=config,
            limit=fetch_limit,
            since_ms=int(fetch_start.timestamp() * 1000) if fetch_start is not None else None,
            attempts=attempts,
            backoff_seconds=backoff_seconds,
            fallback_reasons=fallback_reasons,
        )
        if frame is None:
            continue
        if cached_frame is not None:
            frame = _normalize_ohlcv_frame(pd.concat([cached_frame, frame], ignore_index=True))
        if end_utc:
            cached_full_frame = frame
            frame = frame.loc[frame["timestamp"] <= return_end].tail(required_rows).reset_index(drop=True)
        elif production_start_utc:
            cached_full_frame = frame
            frame = frame.loc[frame["timestamp"] >= required_start].reset_index(drop=True)
        else:
            cached_full_frame = frame
        _write_cache(cache_path, cached_full_frame, provider_used=provider_used, config=config)
        return FetchResult(
            frame,
            provider_used,
            tuple(fallback_reasons),
            cache_path=cache_path,
            cache_status="appended" if cached_frame is not None else "written",
        )

    raise RuntimeError(
        "All live OHLCV providers failed and no usable private live cache exists: " + "; ".join(fallback_reasons)
    )
