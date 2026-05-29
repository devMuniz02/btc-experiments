#!/usr/bin/env python3
"""
Directional BTC/USDT model tournament with MLflow registry promotion on DagsHub.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import sys

import ccxt
import joblib
import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.path_config import HOURLY_LAST_PREDICTION_PATH


SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LOOKBACK_HOURS = 5000
VALIDATION_HOURS = 48
SEQUENCE_LENGTH = 24
CROSS_VALIDATION_FOLDS = 5
SEED = 42
DEVICE = torch.device("cpu")
FEATURE_PREPROCESSOR_NAME = "standard_scaler"
DEFAULT_EXPERIMENT_PREFIX = "btc"
DEFAULT_MODEL_NAME = "btc-usdt-directional-classifier"
ARTIFACT_SUBDIR = "packaged_model"
MODEL_ARTIFACT_NAME = "model"
LAST_PREDICTION_PATH = HOURLY_LAST_PREDICTION_PATH
CHAMPION_ALIAS = "champion"
CHAMPION_CACHE_METADATA_FILENAME = "_champion_cache.json"
BINANCE_CANDIDATES = [
    ("binance", "BTC/USDT", {}),
    ("binance", "BTC/USDT:USDT", {"options": {"defaultType": "future"}}),
]
BINANCE_US_CANDIDATE = ("binanceus", "BTC/USDT", {})
FALLBACK_EXCHANGE_CANDIDATES = [
    ("kraken", "BTC/USDT", {}),
    ("okx", "BTC/USDT", {}),
    ("kucoin", "BTC/USDT", {}),
    ("bitfinex", "BTC/USDT", {}),
]


def log_step(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_current_utc_timestamp(
    value: datetime | pd.Timestamp | str | None = None,
) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if current.tzinfo is None:
        return current.tz_localize("UTC")
    return current.tz_convert("UTC")


def build_experiment_name(now_utc: datetime | pd.Timestamp | str | None = None) -> str:
    current_utc = resolve_current_utc_timestamp(now_utc)
    return (
        f"{DEFAULT_EXPERIMENT_PREFIX}-"
        f"{current_utc.hour:02d}-00_{current_utc.day}_{current_utc.month}"
    )


def get_exchange_candidates() -> tuple[
    list[tuple[str, str, dict[str, Any]]],
    tuple[str, str, dict[str, Any]] | None,
    list[tuple[str, str, dict[str, Any]]],
]:
    exchange_mode = (get_env_str("BTC_EXCHANGE_MODE") or "").lower()
    if exchange_mode == "binance":
        return [("binance", "BTC/USDT", {})], None, []
    return BINANCE_CANDIDATES, BINANCE_US_CANDIDATE, FALLBACK_EXCHANGE_CANDIDATES


def configure_tracking(now_utc: datetime | pd.Timestamp | str | None = None) -> str:
    tracking_uri = get_env_str("MLFLOW_TRACKING_URI")
    username = get_env_str("MLFLOW_TRACKING_USERNAME")
    password = get_env_str("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required.")
    if not username:
        raise RuntimeError("MLFLOW_TRACKING_USERNAME is required.")
    if not password:
        raise RuntimeError("MLFLOW_TRACKING_PASSWORD is required.")

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = build_experiment_name(now_utc)
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: {experiment_name} (UTC)")
    return get_env_str("MLFLOW_MODEL_NAME") or DEFAULT_MODEL_NAME


def fetch_ohlcv(
    limit: int = LOOKBACK_HOURS,
    min_candles: int | None = None,
    retry_binanceus: bool = False,
    retry_binanceus_attempts: int = 3,
    retry_interval_seconds: int = 60,
) -> pd.DataFrame:
    failures: list[str] = []
    required_candles = min_candles if min_candles is not None else limit
    primary_candidates, secondary_candidate, fallback_candidates = get_exchange_candidates()

    def try_exchange(
        exchange_id: str,
        symbol: str,
        extra_config: dict[str, Any],
    ) -> pd.DataFrame:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(
            {
                "enableRateLimit": True,
                "timeout": 30000,
                **extra_config,
            }
        )
        fetch_params: dict[str, Any] = {}
        if limit > 1000:
            fetch_params = {
                "paginate": True,
                "paginationCalls": math.ceil(limit / 1000),
            }
        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=TIMEFRAME,
            limit=limit,
            params=fetch_params,
        )
        if len(candles) < required_candles:
            raise RuntimeError(
                f"{exchange_id} returned too few candles ({len(candles)})."
            )
        frame = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        print(f"Fetched {len(frame)} candles from {exchange_id} using {symbol}.")
        return frame

    for exchange_id, symbol, extra_config in primary_candidates:
        try:
            return try_exchange(exchange_id, symbol, extra_config)
        except Exception as exc:
            failures.append(f"{exchange_id}:{symbol}: {exc}")
            print(f"Exchange attempt failed for {exchange_id} {symbol}: {exc}")

    if secondary_candidate is not None:
        exchange_id, symbol, extra_config = secondary_candidate
        attempts = 0
        while True:
            try:
                return try_exchange(exchange_id, symbol, extra_config)
            except Exception as exc:
                attempts += 1
                failures.append(f"{exchange_id}:{symbol}: {exc}")
                print(f"Exchange attempt failed for {exchange_id} {symbol}: {exc}")
                if not retry_binanceus or attempts >= retry_binanceus_attempts:
                    break
                print(
                    f"Retrying {exchange_id} {symbol} in {retry_interval_seconds} seconds "
                    f"(attempt {attempts + 1} of {retry_binanceus_attempts})."
                )
                time.sleep(retry_interval_seconds)

    for exchange_id, symbol, extra_config in fallback_candidates:
        try:
            return try_exchange(exchange_id, symbol, extra_config)
        except Exception as exc:
            failures.append(f"{exchange_id}:{symbol}: {exc}")
            print(f"Exchange attempt failed for {exchange_id} {symbol}: {exc}")

    raise RuntimeError(
        "Could not fetch BTC/USDT candles from Binance variants or fallback exchanges. "
        + " | ".join(failures)
    )


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]
    volume = df["volume"]

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    ema_8 = close.ewm(span=8, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    macd_signal = (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()

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
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = macd_signal
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["ema_8_gap"] = (close / ema_8) - 1.0
    df["ema_21_gap"] = (close / ema_21) - 1.0
    df["ema_50_gap"] = (close / ema_50) - 1.0
    df["bollinger_zscore"] = (close - rolling_mean_20) / rolling_std_20
    df["bollinger_bandwidth"] = (2 * rolling_std_20) / rolling_mean_20
    df["stochastic_k"] = stochastic_k
    df["stochastic_d"] = stochastic_d
    df["atr_14"] = atr_14
    df["atr_pct"] = atr_14 / close
    df["volume_delta"] = df["volume"].diff()
    df["volume_zscore_20"] = (volume - volume_mean_20) / volume_std_20
    df["price_change_1h"] = close.pct_change(1)
    df["price_change_3h"] = close.pct_change(3)
    df["price_change_6h"] = close.pct_change(6)
    df["price_change_12h"] = close.pct_change(12)
    df["price_change_24h"] = close.pct_change(24)
    df["volatility_6h"] = df["price_change_1h"].rolling(window=6, min_periods=6).std()
    df["volatility_24h"] = df["price_change_1h"].rolling(window=24, min_periods=24).std()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


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


def split_dataset(
    df: pd.DataFrame,
    validation_hours: int = VALIDATION_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = df.iloc[:-1].copy()
    future_row = df.iloc[[-1]].copy()
    if len(labeled) <= validation_hours + SEQUENCE_LENGTH:
        raise RuntimeError("Not enough samples to build train and validation splits.")

    train_df = labeled.iloc[:-validation_hours].copy()
    valid_df = labeled.iloc[-validation_hours:].copy()
    return train_df, valid_df, future_row


def filter_ohlcv_to_effective_now(
    frame: pd.DataFrame,
    effective_now: datetime | pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if effective_now is None:
        return frame.copy()
    cutoff = resolve_current_utc_timestamp(effective_now).floor("h")
    filtered = frame.loc[frame["timestamp"] <= cutoff].copy()
    if filtered.empty:
        raise RuntimeError(
            f"No OHLCV rows were available at or before simulated timestamp {cutoff.isoformat()}."
        )
    return filtered.reset_index(drop=True)


def build_sequence_dataset(
    features: np.ndarray,
    labels: np.ndarray | None,
    indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_seq: list[np.ndarray] = []
    y_seq: list[float] = []

    for idx in indices:
        if idx < seq_len - 1:
            continue
        x_seq.append(features[idx - seq_len + 1 : idx + 1])
        if labels is not None:
            y_seq.append(labels[idx])

    x_array = np.asarray(x_seq, dtype=np.float32)
    if labels is None:
        return x_array, None
    return x_array, np.asarray(y_seq, dtype=np.float32)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class SequenceMLPClassifier(nn.Module):
    def __init__(self, seq_len: int, input_dim: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim
        self.network = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.network(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, SEQUENCE_LENGTH, model_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.input_proj(x) + self.position_embedding[:, :seq_len, :]
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


def flatten_sequence_features(sequence_features: np.ndarray) -> np.ndarray:
    return sequence_features.reshape(sequence_features.shape[0], -1)


def prepare_sequence_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
) -> dict[str, np.ndarray | StandardScaler]:
    train_features = train_df[feature_columns].to_numpy(dtype=np.float32)
    sequence_frame = pd.concat([train_df, valid_df], ignore_index=True)
    scaler = build_feature_scaler(train_features)
    scaled_sequence_features = scaler.transform(
        sequence_frame[feature_columns].to_numpy(dtype=np.float32)
    )
    sequence_labels = sequence_frame["target"].to_numpy(dtype=np.float32)

    train_indices = np.arange(len(train_df))
    valid_indices = np.arange(len(train_df), len(train_df) + len(valid_df))
    train_seq_x, train_seq_y = build_sequence_dataset(
        features=scaled_sequence_features,
        labels=sequence_labels,
        indices=train_indices,
        seq_len=seq_len,
    )
    valid_seq_x, valid_seq_y = build_sequence_dataset(
        features=scaled_sequence_features,
        labels=sequence_labels,
        indices=valid_indices,
        seq_len=seq_len,
    )
    if len(train_seq_x) == 0 or len(valid_seq_x) == 0:
        raise RuntimeError("Could not build sequence datasets for the challenger models.")
    return {
        "scaler": scaler,
        "train_seq_x": train_seq_x,
        "train_seq_y": train_seq_y,
        "valid_seq_x": valid_seq_x,
        "valid_seq_y": valid_seq_y,
        "train_flat_x": flatten_sequence_features(train_seq_x),
        "valid_flat_x": flatten_sequence_features(valid_seq_x),
    }


def train_torch_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int = 32,
    patience: int = 8,
) -> nn.Module:
    model.to(DEVICE)
    positive_count = max(float(train_y.sum()), 1.0)
    negative_count = max(float(len(train_y) - train_y.sum()), 1.0)
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32, device=DEVICE)
    valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32, device=DEVICE)

    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x_tensor)
            valid_loss = criterion(valid_logits, valid_y_tensor).item()

        if valid_loss < best_loss:
            best_loss = valid_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    return model


def torch_predict_proba(model: nn.Module, data: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(data, dtype=torch.float32)
        return torch.sigmoid(model(tensor)).cpu().numpy()


@dataclass
class TournamentCandidate:
    name: str
    family: str
    model: Any
    feature_columns: list[str]
    sequence_length: int
    scaler: StandardScaler | None = None


@dataclass
class TournamentExecutionResult:
    registered_model_name: str
    raw: pd.DataFrame
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    future_row: pd.DataFrame
    challenger_results: list[dict[str, Any]]
    all_results: list[dict[str, Any]]
    family_decisions: list[dict[str, Any]]
    active_results_by_family: dict[str, dict[str, Any]]
    active_result: dict[str, Any]
    validation_start: str
    validation_end: str
    phase_timings: dict[str, float]
    benchmark_metadata: dict[str, Any]


def build_feature_scaler(features: np.ndarray) -> StandardScaler:
    # Normalize every feature dimension from train-only statistics to avoid one input scale dominating training.
    return StandardScaler().fit(features)


def prepare_full_sequence_training_data(
    labeled_df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
) -> dict[str, np.ndarray | StandardScaler]:
    features = labeled_df[feature_columns].to_numpy(dtype=np.float32)
    scaler = build_feature_scaler(features)
    scaled_features = scaler.transform(features)
    labels = labeled_df["target"].to_numpy(dtype=np.float32)
    indices = np.arange(len(labeled_df))
    seq_x, seq_y = build_sequence_dataset(
        features=scaled_features,
        labels=labels,
        indices=indices,
        seq_len=seq_len,
    )
    if len(seq_x) == 0:
        raise RuntimeError("Could not build sequence dataset from labeled data.")
    return {
        "scaler": scaler,
        "seq_x": seq_x,
        "seq_y": seq_y,
        "flat_x": flatten_sequence_features(seq_x),
    }


def build_challenger_candidates(train_seq_y: np.ndarray, scaler: StandardScaler) -> list[TournamentCandidate]:
    scale_pos_weight = max(
        float((len(train_seq_y) - train_seq_y.sum()) / max(train_seq_y.sum(), 1.0)),
        1.0,
    )
    return [
        TournamentCandidate(
            name="RandomForest",
            family="rf",
            model=RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=scaler,
        ),
        TournamentCandidate(
            name="XGBoost",
            family="xgb",
            model=XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=SEED,
                n_jobs=2,
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=scaler,
        ),
        TournamentCandidate(
            name="MLPClassifier",
            family="mlp_sklearn",
            model=MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=32,
                learning_rate_init=6e-4,
                max_iter=500,
                random_state=SEED,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=scaler,
        ),
        TournamentCandidate(
            name="LSTM",
            family="lstm",
            model=LSTMClassifier(input_dim=len(FEATURE_COLUMNS), hidden_dim=64),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=scaler,
        ),
        TournamentCandidate(
            name="Transformer",
            family="transformer",
            model=TransformerClassifier(
                input_dim=len(FEATURE_COLUMNS),
                model_dim=48,
                num_heads=4,
                num_layers=3,
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=scaler,
        ),
        TournamentCandidate(
            name="NN",
            family="nn",
            model=SequenceMLPClassifier(
                seq_len=SEQUENCE_LENGTH,
                input_dim=len(FEATURE_COLUMNS),
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=scaler,
        ),
    ]


def fit_candidate(
    candidate: TournamentCandidate,
    train_flat_x: np.ndarray,
    train_seq_x: np.ndarray,
    train_seq_y: np.ndarray,
    valid_flat_x: np.ndarray | None = None,
    valid_seq_x: np.ndarray | None = None,
    valid_seq_y: np.ndarray | None = None,
) -> TournamentCandidate:
    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        candidate.model.fit(train_flat_x, train_seq_y)
        return candidate
    if valid_seq_x is None or valid_seq_y is None:
        valid_seq_x = train_seq_x
        valid_seq_y = train_seq_y
    if candidate.family == "lstm":
        candidate.model = train_torch_classifier(
            candidate.model,
            train_seq_x,
            train_seq_y,
            valid_seq_x,
            valid_seq_y,
            epochs=40,
            learning_rate=6e-4,
        )
    elif candidate.family == "transformer":
        candidate.model = train_torch_classifier(
            candidate.model,
            train_seq_x,
            train_seq_y,
            valid_seq_x,
            valid_seq_y,
            epochs=36,
            learning_rate=5e-4,
        )
    elif candidate.family == "nn":
        candidate.model = train_torch_classifier(
            candidate.model,
            train_seq_x,
            train_seq_y,
            valid_seq_x,
            valid_seq_y,
            epochs=48,
            learning_rate=7e-4,
        )
    else:
        raise ValueError(f"Unsupported challenger family: {candidate.family}")
    return candidate


def run_challenger_cross_validation(
    labeled_df: pd.DataFrame,
    folds: int = CROSS_VALIDATION_FOLDS,
) -> dict[str, dict[str, float]]:
    log_step(f"Run {folds}-fold time-series cross-validation")
    splitter = TimeSeriesSplit(n_splits=folds)
    fold_metrics: dict[str, list[dict[str, float]]] = {}

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(labeled_df), start=1):
        train_fold = labeled_df.iloc[train_idx].reset_index(drop=True)
        valid_fold = labeled_df.iloc[valid_idx].reset_index(drop=True)
        if len(train_fold) <= SEQUENCE_LENGTH or len(valid_fold) == 0:
            raise RuntimeError("Cross-validation fold is too small for the sequence models.")

        fold_splits = prepare_sequence_splits(train_fold, valid_fold, FEATURE_COLUMNS, SEQUENCE_LENGTH)
        candidates = build_challenger_candidates(
            train_seq_y=fold_splits["train_seq_y"],
            scaler=fold_splits["scaler"],
        )
        print(f"Cross-validation fold {fold_index}/{folds}", flush=True)
        for candidate in candidates:
            print(f"  Fold {fold_index}: training {candidate.name}", flush=True)
            fit_candidate(
                candidate,
                train_flat_x=fold_splits["train_flat_x"],
                train_seq_x=fold_splits["train_seq_x"],
                train_seq_y=fold_splits["train_seq_y"],
                valid_flat_x=fold_splits["valid_flat_x"],
                valid_seq_x=fold_splits["valid_seq_x"],
                valid_seq_y=fold_splits["valid_seq_y"],
            )

            eval_frame = pd.concat([train_fold, valid_fold], ignore_index=True)
            valid_start = len(train_fold)
            valid_labels = valid_fold["target"].to_numpy(dtype=np.int32)
            fold_probs = predict_candidate_probabilities(candidate, eval_frame)[valid_start:]
            if np.isnan(fold_probs).any():
                raise RuntimeError(
                    f"{candidate.name} produced incomplete cross-validation output on fold {fold_index}."
                )
            metrics = evaluate_probabilities(fold_probs, valid_labels)
            fold_metrics.setdefault(candidate.family, []).append(metrics)

    summary: dict[str, dict[str, float]] = {}
    for family, metrics_list in fold_metrics.items():
        summary[family] = {
            "cv_accuracy": float(np.mean([metrics["accuracy"] for metrics in metrics_list])),
            "cv_f1": float(np.mean([metrics["f1"] for metrics in metrics_list])),
        }
    return summary


def prediction_to_signal(probability: float) -> str:
    return "UP" if probability >= 0.5 else "DOWN"


def evaluate_probabilities(probabilities: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    labels = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(actual, labels)),
        "f1": float(f1_score(actual, labels, zero_division=0)),
    }


def ranking_key(result: dict[str, Any]) -> tuple[float, float]:
    return (-result["f1"], -result["accuracy"])


def registered_model_name_for_family(base_name: str, family: str) -> str:
    return f"{base_name}-{family}"


def get_cv_metric(result: dict[str, Any], key: str) -> float:
    value = result.get(key)
    if value is None:
        return 0.0
    return float(value)


def serialize_result(
    result: dict[str, Any],
    *,
    include_registry_version: bool = True,
) -> dict[str, Any]:
    payload = {
        "name": result["candidate"].name,
        "family": result["family"],
        "source": result["source"],
        "accuracy": float(result["accuracy"]),
        "f1": float(result["f1"]),
        "cv_accuracy": get_cv_metric(result, "cv_accuracy"),
        "cv_f1": get_cv_metric(result, "cv_f1"),
        "probability_up": float(result["next_probability"]),
        "predicted_signal": result["next_signal"],
        "predicted_label": int(result["next_probability"] >= 0.5),
    }
    if include_registry_version:
        payload["registry_version"] = result.get("registry_version")
    return payload


def predict_candidate_probabilities(
    candidate: TournamentCandidate,
    feature_rows: pd.DataFrame,
) -> np.ndarray:
    features = feature_rows[candidate.feature_columns].to_numpy(dtype=np.float32)
    scaled = candidate.scaler.transform(features) if candidate.scaler is not None else features
    sequence_indices = np.arange(len(feature_rows))
    seq_x, _ = build_sequence_dataset(
        features=scaled,
        labels=None,
        indices=sequence_indices,
        seq_len=candidate.sequence_length,
    )
    output = np.full(len(feature_rows), np.nan, dtype=np.float32)
    if len(seq_x) == 0:
        return output
    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        seq_features = flatten_sequence_features(seq_x)
        output[candidate.sequence_length - 1 :] = candidate.model.predict_proba(seq_features)[:, 1]
        return output
    output[candidate.sequence_length - 1 :] = torch_predict_proba(candidate.model, seq_x)
    return output


def train_challengers(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> tuple[list[TournamentCandidate], dict[str, dict[str, float]]]:
    log_step("Prepare challenger datasets")
    sequence_splits = prepare_sequence_splits(train_df, valid_df, FEATURE_COLUMNS, SEQUENCE_LENGTH)
    challengers = build_challenger_candidates(
        train_seq_y=sequence_splits["train_seq_y"],
        scaler=sequence_splits["scaler"],
    )

    for candidate in challengers:
        print(f"Training challenger: {candidate.name}", flush=True)
        fit_candidate(
            candidate,
            train_flat_x=sequence_splits["train_flat_x"],
            train_seq_x=sequence_splits["train_seq_x"],
            train_seq_y=sequence_splits["train_seq_y"],
            valid_flat_x=sequence_splits["valid_flat_x"],
            valid_seq_x=sequence_splits["valid_seq_x"],
            valid_seq_y=sequence_splits["valid_seq_y"],
        )
        print(f"Finished challenger: {candidate.name}", flush=True)

    return challengers, {}


def retrain_challengers_on_full_data(
    labeled_df: pd.DataFrame,
) -> list[TournamentCandidate]:
    log_step("Retrain challengers on all labeled data")
    training_data = prepare_full_sequence_training_data(
        labeled_df,
        FEATURE_COLUMNS,
        SEQUENCE_LENGTH,
    )
    challengers = build_challenger_candidates(
        train_seq_y=training_data["seq_y"],
        scaler=training_data["scaler"],
    )
    for candidate in challengers:
        print(f"Refitting challenger on full data: {candidate.name}", flush=True)
        fit_candidate(
            candidate,
            train_flat_x=training_data["flat_x"],
            train_seq_x=training_data["seq_x"],
            train_seq_y=training_data["seq_y"],
        )
    return challengers


def build_results(
    candidates: list[TournamentCandidate],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
    cv_summary: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    log_step("Evaluate challengers on validation window")
    eval_frame = pd.concat([train_df, valid_df], ignore_index=True)
    prediction_frame = pd.concat([eval_frame, future_row], ignore_index=True)
    val_start = len(train_df)
    actual_valid = valid_df["target"].to_numpy(dtype=np.int32)

    results = []
    for candidate in candidates:
        prediction_probs = predict_candidate_probabilities(candidate, prediction_frame)
        valid_probs = prediction_probs[val_start:-1]
        if np.isnan(valid_probs).any():
            raise RuntimeError(f"{candidate.name} produced incomplete validation output.")
        metrics = evaluate_probabilities(valid_probs, actual_valid)
        next_probability = float(prediction_probs[-1])
        cv_metrics = (cv_summary or {}).get(candidate.family, {})
        results.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "source": "challenger",
                "candidate": candidate,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "cv_accuracy": float(cv_metrics.get("cv_accuracy", 0.0)),
                "cv_f1": float(cv_metrics.get("cv_f1", 0.0)),
                "next_probability": next_probability,
                "next_signal": prediction_to_signal(next_probability),
            }
        )
    return results


def save_candidate_package(candidate: TournamentCandidate, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": candidate.name,
        "family": candidate.family,
        "feature_columns": candidate.feature_columns,
        "sequence_length": candidate.sequence_length,
        "preprocessing": {
            "feature_scaler": FEATURE_PREPROCESSOR_NAME,
            "normalized_features": True,
        },
    }

    if candidate.scaler is not None:
        joblib.dump(candidate.scaler, output_dir / "scaler.joblib")
    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        joblib.dump(candidate.model, output_dir / "model.joblib")
    else:
        if candidate.family == "lstm":
            config["model_kwargs"] = {
                "input_dim": candidate.model.input_dim,
                "hidden_dim": candidate.model.hidden_dim,
            }
        elif candidate.family == "transformer":
            config["model_kwargs"] = {
                "input_dim": candidate.model.input_dim,
                "model_dim": candidate.model.model_dim,
                "num_heads": candidate.model.num_heads,
                "num_layers": candidate.model.num_layers,
            }
        elif candidate.family == "nn":
            config["model_kwargs"] = {
                "seq_len": candidate.model.seq_len,
                "input_dim": candidate.model.input_dim,
            }
        else:
            raise ValueError(f"Unsupported candidate family: {candidate.family}")
        torch.save({"state_dict": candidate.model.state_dict()}, output_dir / "model.pt")

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_candidate_package(model_dir: str | Path) -> TournamentCandidate:
    package_dir = Path(model_dir)
    with (package_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    candidate = TournamentCandidate(
        name=config["name"],
        family=config["family"],
        model=None,
        feature_columns=config["feature_columns"],
        sequence_length=config["sequence_length"],
        scaler=None,
    )

    scaler_path = package_dir / "scaler.joblib"
    if scaler_path.exists():
        candidate.scaler = joblib.load(scaler_path)
    elif config.get("preprocessing", {}).get("normalized_features"):
        print(
            f"Warning: normalized model package for {candidate.name} is missing scaler.joblib. "
            "Predictions may not match training-time preprocessing."
        )

    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        candidate.model = joblib.load(package_dir / "model.joblib")
        return candidate

    kwargs = config["model_kwargs"]
    if candidate.family == "lstm":
        candidate.model = LSTMClassifier(**kwargs)
    elif candidate.family == "transformer":
        candidate.model = TransformerClassifier(**kwargs)
    elif candidate.family == "nn":
        candidate.model = SequenceMLPClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported packaged family: {candidate.family}")

    state = torch.load(package_dir / "model.pt", map_location="cpu")
    candidate.model.load_state_dict(state["state_dict"])
    candidate.model.eval()
    return candidate


class TournamentPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.candidate = load_candidate_package(context.artifacts["model_dir"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        probabilities = predict_candidate_probabilities(self.candidate, model_input)
        return pd.DataFrame({"prob_up": probabilities})


def resolve_candidate_package_dir(model_root: str | Path) -> Path:
    root = Path(model_root)
    candidate_dirs = [
        root,
        root / ARTIFACT_SUBDIR,
        root / "artifacts" / ARTIFACT_SUBDIR,
    ]
    for candidate_dir in candidate_dirs:
        if (candidate_dir / "config.json").exists():
            return candidate_dir
    raise FileNotFoundError(
        "Could not find packaged champion model under "
        f"{root}. Checked: {', '.join(str(path) for path in candidate_dirs)}"
    )


def read_champion_cache_metadata(download_root: str | Path) -> dict[str, str] | None:
    metadata_path = Path(download_root) / CHAMPION_CACHE_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    run_id = payload.get("run_id")
    if not version or not run_id:
        return None
    return {"version": str(version), "run_id": str(run_id)}


def write_champion_cache_metadata(download_root: str | Path, *, version: str, run_id: str) -> None:
    metadata_path = Path(download_root) / CHAMPION_CACHE_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps({"version": str(version), "run_id": str(run_id)}, indent=2),
        encoding="utf-8",
    )


def download_model_root(version: Any, dst_path: str) -> str:
    artifact_uris = [
        getattr(version, "source", None),
        f"models:/{version.name}/{version.version}",
        f"runs:/{version.run_id}/{MODEL_ARTIFACT_NAME}",
    ]
    last_error: Exception | None = None
    for artifact_uri in artifact_uris:
        if not artifact_uri:
            continue
        try:
            return mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=dst_path,
            )
        except Exception as exc:
            last_error = exc

    try:
        client = MlflowClient()
        return client.download_artifacts(version.run_id, MODEL_ARTIFACT_NAME, dst_path)
    except Exception as exc:
        last_error = exc

    if last_error is None:
        raise RuntimeError("No artifact URI was available for the champion model.")
    raise last_error


def build_model_logging_inputs(
    candidate: TournamentCandidate,
    feature_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, Any]:
    sample_size = max(candidate.sequence_length, min(len(feature_rows), candidate.sequence_length * 2))
    input_example = feature_rows[candidate.feature_columns].tail(sample_size).copy()
    predictions = pd.DataFrame(
        {"prob_up": predict_candidate_probabilities(candidate, input_example)}
    )
    signature = infer_signature(input_example, predictions)
    return input_example, signature


def build_model_pip_requirements() -> list[str]:
    return [
        "mlflow==3.10.1",
        "pandas",
        "numpy",
        "joblib",
        "scikit-learn==1.8.0",
        "xgboost",
        "torch==2.10.0",
    ]


def get_current_champion(
    client: MlflowClient,
    registered_model_name: str,
    alias: str = CHAMPION_ALIAS,
    download_root: str | Path | None = None,
) -> tuple[TournamentCandidate | None, dict[str, str] | None]:
    try:
        version = client.get_model_version_by_alias(registered_model_name, alias)
    except Exception:
        return None, None

    if download_root is None:
        temp_context = tempfile.TemporaryDirectory()
        temp_dir_obj = temp_context.__enter__()
    else:
        temp_context = None
        temp_dir_obj = Path(download_root)
        temp_dir_obj.mkdir(parents=True, exist_ok=True)

    try:
        try:
            if download_root is not None:
                cached_metadata = read_champion_cache_metadata(temp_dir_obj)
                if (
                    cached_metadata is not None
                    and cached_metadata.get("version") == str(version.version)
                    and cached_metadata.get("run_id") == str(version.run_id)
                ):
                    package_dir = resolve_candidate_package_dir(temp_dir_obj)
                    candidate = load_candidate_package(package_dir)
                    metadata = {"version": version.version, "run_id": version.run_id, "download_root": str(temp_dir_obj)}
                    return candidate, metadata
            log_step(f"Load current champion from MLflow ({alias})")
            local_model_root = download_model_root(version, str(temp_dir_obj))
            package_dir = resolve_candidate_package_dir(local_model_root)
            candidate = load_candidate_package(package_dir)
            if download_root is not None:
                write_champion_cache_metadata(
                    temp_dir_obj,
                    version=str(version.version),
                    run_id=str(version.run_id),
                )
        except Exception as exc:
            print(
                "Champion alias exists but its artifacts could not be loaded. "
                f"Proceeding without champion. Last error: {exc}"
            )
            return None, None
        metadata = {"version": version.version, "run_id": version.run_id}
        if download_root is not None:
            metadata["download_root"] = str(temp_dir_obj)
        return candidate, metadata
    finally:
        if temp_context is not None:
            temp_context.__exit__(None, None, None)


def evaluate_champion(
    champion: TournamentCandidate,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
) -> dict[str, Any]:
    eval_frame = pd.concat([train_df, valid_df], ignore_index=True)
    prediction_frame = pd.concat([eval_frame, future_row], ignore_index=True)
    val_start = len(train_df)
    valid_labels = valid_df["target"].to_numpy(dtype=np.int32)

    prediction_probs = predict_candidate_probabilities(champion, prediction_frame)
    valid_probs = prediction_probs[val_start:-1]
    if np.isnan(valid_probs).any():
        raise RuntimeError("Champion produced incomplete validation output.")

    metrics = evaluate_probabilities(valid_probs, valid_labels)
    next_probability = float(prediction_probs[-1])
    return {
        "name": champion.name,
        "family": champion.family,
        "source": "champion",
        "candidate": champion,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "next_probability": next_probability,
        "next_signal": prediction_to_signal(next_probability),
    }


def log_challenger_summary(
    challenger_results: list[dict[str, Any]],
) -> None:
    summary = []
    for result in challenger_results:
        metric_prefix = result["name"].lower().replace(" ", "_")
        mlflow.log_metrics(
            {
                f"{metric_prefix}_accuracy": result["accuracy"],
                f"{metric_prefix}_f1": result["f1"],
                f"{metric_prefix}_cv_accuracy": get_cv_metric(result, "cv_accuracy"),
                f"{metric_prefix}_cv_f1": get_cv_metric(result, "cv_f1"),
                f"{metric_prefix}_next_probability": result["next_probability"],
            }
        )
        summary.append(
            {
                "name": result["name"],
                "family": result["family"],
                "accuracy": result["accuracy"],
                "f1": result["f1"],
                "cv_accuracy": get_cv_metric(result, "cv_accuracy"),
                "cv_f1": get_cv_metric(result, "cv_f1"),
                "next_probability": result["next_probability"],
                "next_signal": result["next_signal"],
            }
        )
    mlflow.log_text(json.dumps(summary, indent=2), "challenger_summary.json")


def log_comparison_metrics(
    active_results_by_family: dict[str, dict[str, Any]],
    active_result: dict[str, Any],
) -> None:
    metrics: dict[str, float] = {
        "best_accuracy": float(active_result["accuracy"]),
        "best_f1": float(active_result["f1"]),
        "best_cv_accuracy": get_cv_metric(active_result, "cv_accuracy"),
        "best_cv_f1": get_cv_metric(active_result, "cv_f1"),
        "best_probability_up": float(active_result["next_probability"]),
        "best_predicted_label": float(active_result["next_probability"] >= 0.5),
    }
    tags = {
        "best_model_name": active_result["candidate"].name,
        "best_model_family": active_result["family"],
        "best_model_source": active_result["source"],
        "best_predicted_signal": active_result["next_signal"],
        "feature_preprocessor": FEATURE_PREPROCESSOR_NAME,
        "features_normalized": "true",
    }

    ordered_results = sorted(active_results_by_family.values(), key=ranking_key)
    family_rank_map = {
        result["family"]: rank for rank, result in enumerate(ordered_results, start=1)
    }

    for family, result in sorted(active_results_by_family.items()):
        family_prefix = family.replace("-", "_")
        metrics.update(
            {
                f"{family_prefix}_accuracy": float(result["accuracy"]),
                f"{family_prefix}_f1": float(result["f1"]),
                f"{family_prefix}_cv_accuracy": get_cv_metric(result, "cv_accuracy"),
                f"{family_prefix}_cv_f1": get_cv_metric(result, "cv_f1"),
                f"{family_prefix}_probability_up": float(result["next_probability"]),
                f"{family_prefix}_predicted_label": float(result["next_probability"] >= 0.5),
                f"{family_prefix}_is_best": float(family == active_result["family"]),
                f"{family_prefix}_rank": float(family_rank_map[family]),
            }
        )
        if result.get("registry_version") is not None:
            metrics[f"{family_prefix}_registry_version"] = float(result["registry_version"])
        tags[f"{family_prefix}_model_name"] = result["candidate"].name
        tags[f"{family_prefix}_source"] = result["source"]
        tags[f"{family_prefix}_predicted_signal"] = result["next_signal"]

    mlflow.log_metrics(metrics)
    mlflow.set_tags(tags)


def try_mlflow_call(step_label: str, callback: Callable[[], Any]) -> Any | None:
    try:
        return callback()
    except Exception as exc:
        print(f"Warning: skipped MLflow step '{step_label}': {type(exc).__name__}: {exc}", flush=True)
        return None


@contextmanager
def start_tournament_mlflow_run() -> Any:
    primary_error: Exception | None = None
    try:
        with mlflow.start_run(run_name="btc-directional-tournament") as run:
            yield run
            return
    except Exception as exc:
        primary_error = exc
        print(
            f"Warning: failed to open named MLflow run 'btc-directional-tournament': {type(exc).__name__}: {exc}",
            flush=True,
        )

    try:
        with mlflow.start_run() as run:
            yield run
            return
    except Exception:
        if primary_error is not None:
            raise primary_error
        raise


def promote_champion(
    client: MlflowClient,
    registered_model_name: str,
    winner: dict[str, Any],
    validation_start: str,
    validation_end: str,
    feature_rows: pd.DataFrame,
    alias: str,
) -> str:
    log_step(f"Promote new champion: {winner['candidate'].name} ({alias})")
    candidate: TournamentCandidate = winner["candidate"]
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir) / ARTIFACT_SUBDIR
        save_candidate_package(candidate, package_dir)
        input_example, signature = build_model_logging_inputs(candidate, feature_rows)
        model_code_path = Path(__file__).with_name("mlflow_tournament_model.py")
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Champion promotion requires an active MLflow run.")

        metric_prefix = candidate.family.replace("-", "_")
        mlflow.set_tags(
            {
                f"{metric_prefix}_promotion_role": "champion_candidate",
                f"{metric_prefix}_model_name": candidate.name,
                f"{metric_prefix}_model_family": candidate.family,
                f"{metric_prefix}_feature_preprocessor": FEATURE_PREPROCESSOR_NAME,
                f"{metric_prefix}_features_normalized": "true",
                f"{metric_prefix}_champion_alias": alias,
                f"{metric_prefix}_validation_start": validation_start,
                f"{metric_prefix}_validation_end": validation_end,
            }
        )
        mlflow.log_metrics(
            {
                f"{metric_prefix}_promotion_accuracy": winner["accuracy"],
                f"{metric_prefix}_promotion_f1": winner["f1"],
                f"{metric_prefix}_promotion_next_probability": winner["next_probability"],
            }
        )
        artifact_name = f"{MODEL_ARTIFACT_NAME}_{candidate.family}"
        mlflow.pyfunc.log_model(
            name=artifact_name,
            python_model=str(model_code_path),
            artifacts={"model_dir": str(package_dir)},
            registered_model_name=registered_model_name,
            input_example=input_example,
            signature=signature,
            pip_requirements=build_model_pip_requirements(),
        )

        versions = client.search_model_versions(f"run_id = '{active_run.info.run_id}'")
        matching_versions = [
            version
            for version in versions
            if getattr(version, "name", "") == registered_model_name
        ]
        if not matching_versions:
            raise RuntimeError("Registered model version was not found for the promoted run.")

        version = sorted(matching_versions, key=lambda item: int(item.version))[-1]
        client.set_model_version_tag(
            registered_model_name,
            version.version,
            "champion_alias",
            alias,
        )
        client.set_model_version_tag(
            registered_model_name,
            version.version,
            "tournament_model_name",
            candidate.name,
        )
        client.set_model_version_tag(
            registered_model_name,
            version.version,
            "feature_preprocessor",
            FEATURE_PREPROCESSOR_NAME,
        )
        client.set_model_version_tag(
            registered_model_name,
            version.version,
            "features_normalized",
            "true",
        )
        client.set_registered_model_alias(
            registered_model_name,
            alias,
            version.version,
        )
        return version.version


def print_scoreboard(results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=ranking_key)
    print("Directional tournament scoreboard:")
    for index, result in enumerate(ordered, start=1):
        print(
            f"{index}. {result['name']} [{result['source']}] "
            f"F1={result['f1']:.3f} "
            f"Accuracy={result['accuracy']:.3f} "
            f"Next={result['next_probability']:.3f} {result['next_signal']}"
        )


def build_prediction_record(
    active_result: dict[str, Any],
    active_results_by_family: dict[str, dict[str, Any]],
    future_row: pd.DataFrame,
    registered_model_name: str,
    generated_at: datetime | pd.Timestamp | str | None = None,
) -> dict[str, Any]:
    reference_timestamp = pd.Timestamp(future_row["timestamp"].iloc[0])
    target_timestamp = reference_timestamp + pd.Timedelta(hours=1)
    model_predictions = {
        family: {
            **serialize_result(result),
            "registered_model_name": registered_model_name_for_family(registered_model_name, family),
            "is_current_best": family == active_result["family"],
        }
        for family, result in sorted(active_results_by_family.items())
    }
    return {
        "status": "success",
        "generated_at": resolve_current_utc_timestamp(generated_at).isoformat(),
        "registered_model_name": registered_model_name,
        "model_name": active_result["candidate"].name,
        "model_family": active_result["family"],
        "model_accuracy": float(active_result["accuracy"]),
        "model_f1": float(active_result["f1"]),
        "probability_up": float(active_result["next_probability"]),
        "predicted_signal": active_result["next_signal"],
        "predicted_label": int(active_result["next_probability"] >= 0.5),
        "best_champion_name": active_result["candidate"].name,
        "best_champion_family": active_result["family"],
        "best_champion_version": active_result.get("registry_version"),
        "feature_preprocessor": FEATURE_PREPROCESSOR_NAME,
        "features_normalized": True,
        "model_predictions": model_predictions,
        "reference_candle_timestamp": reference_timestamp.isoformat(),
        "target_candle_timestamp": target_timestamp.isoformat(),
        "reference_open": float(future_row["open"].iloc[0]),
        "reference_close": float(future_row["close"].iloc[0]),
        "price_to_beat": float(future_row["open"].iloc[0]),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
    }


def write_failed_prediction_record(
    exc: Exception,
    generated_at: datetime | pd.Timestamp | str | None = None,
) -> None:
    generated_timestamp = resolve_current_utc_timestamp(generated_at)
    target_timestamp = (generated_timestamp.floor("h") + pd.Timedelta(hours=1)).isoformat()
    failure_record = {
        "status": "failed",
        "generated_at": generated_timestamp.isoformat(),
        "registered_model_name": get_env_str("MLFLOW_MODEL_NAME") or DEFAULT_MODEL_NAME,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "target_candle_timestamp": target_timestamp,
        "error": str(exc),
    }
    LAST_PREDICTION_PATH.write_text(
        json.dumps(failure_record, indent=2),
        encoding="utf-8",
    )


def execute_hourly_tournament(
    *,
    client: MlflowClient,
    registered_model_name: str,
    reset_champion_from_challenger: bool = False,
    effective_now: datetime | pd.Timestamp | str | None = None,
    champion_download_root: str | Path | None = None,
    enable_champion_lookup: bool = True,
) -> TournamentExecutionResult:
    phase_timings: dict[str, float] = {}
    benchmark_metadata: dict[str, Any] = {
        "effective_now": (
            resolve_current_utc_timestamp(effective_now).isoformat()
            if effective_now is not None
            else None
        ),
        "champion_lookup_enabled": bool(enable_champion_lookup),
    }

    phase_started = time.perf_counter()
    log_step("Fetch BTC/USDT market data")
    raw = fetch_ohlcv(
        limit=LOOKBACK_HOURS,
        min_candles=5000,
        retry_binanceus=True,
        retry_binanceus_attempts=3,
    )
    filtered_raw = filter_ohlcv_to_effective_now(raw, effective_now)
    log_step("Build features and dataset splits")
    featured = add_features(filtered_raw)
    train_df, valid_df, future_row = split_dataset(featured, VALIDATION_HOURS)
    phase_timings["GETTING_DATA"] = time.perf_counter() - phase_started

    validation_start = valid_df["timestamp"].iloc[0].isoformat()
    validation_end = valid_df["timestamp"].iloc[-1].isoformat()

    phase_started = time.perf_counter()
    log_step("Train challenger zoo")
    challengers, cv_summary = train_challengers(train_df, valid_df)
    full_labeled_df = pd.concat([train_df, valid_df], ignore_index=True)
    refit_challengers = retrain_challengers_on_full_data(full_labeled_df)
    refit_by_family = {candidate.family: candidate for candidate in refit_challengers}
    phase_timings["TRAINING"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    challenger_results = build_results(
        challengers,
        train_df,
        valid_df,
        future_row,
        cv_summary=cv_summary,
    )
    full_prediction_frame = pd.concat([full_labeled_df, future_row], ignore_index=True)
    for result in challenger_results:
        refit_candidate = refit_by_family[result["family"]]
        result["candidate"] = refit_candidate
        result["next_probability"] = float(
            predict_candidate_probabilities(refit_candidate, full_prediction_frame)[-1]
        )
        result["next_signal"] = prediction_to_signal(result["next_probability"])
    phase_timings["EVAL"] = time.perf_counter() - phase_started

    all_results = list(challenger_results)
    challenger_by_family = {result["family"]: result for result in challenger_results}
    family_decisions: list[dict[str, Any]] = []
    active_results_by_family: dict[str, dict[str, Any]] = {}
    champion_downloads = 0

    if reset_champion_from_challenger:
        print("Champion comparison disabled. Selecting from the current challenger leaderboard only.")

    phase_started = time.perf_counter()
    for family, challenger_result in sorted(challenger_by_family.items()):
        champion_result: dict[str, Any] | None = None
        champion_meta: dict[str, str] | None = None
        family_registered_model_name = registered_model_name_for_family(
            registered_model_name,
            family,
        )
        if enable_champion_lookup and not reset_champion_from_challenger:
            family_download_root = None
            if champion_download_root is not None:
                family_download_root = Path(champion_download_root) / family
            champion_candidate, champion_meta = get_current_champion(
                client,
                family_registered_model_name,
                alias=CHAMPION_ALIAS,
                download_root=family_download_root,
            )
            if champion_candidate is not None and champion_meta is not None:
                champion_result = evaluate_champion(
                    champion_candidate,
                    train_df,
                    valid_df,
                    future_row,
                )
                champion_result["registry_version"] = champion_meta["version"]
                all_results.append(champion_result)
                champion_downloads += 1
        null_model_block = (
            challenger_result["f1"] <= 0.5 or challenger_result["accuracy"] <= 0.5
        )
        if champion_result is None:
            should_promote = True
            active_family_result = challenger_result
        else:
            should_promote = (
                challenger_result["f1"] > champion_result["f1"] and not null_model_block
            )
            active_family_result = challenger_result if should_promote else champion_result
        active_results_by_family[family] = active_family_result
        family_decisions.append(
            {
                "family": family,
                "registered_model_name": family_registered_model_name,
                "challenger": challenger_result,
                "champion": champion_result,
                "champion_meta": champion_meta,
                "should_promote": should_promote,
                "null_model_block": null_model_block,
                "active_result": active_family_result,
            }
        )
    phase_timings["GETTING_MODELS_FROM_DAGSHUB"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    active_result = sorted(active_results_by_family.values(), key=ranking_key)[0]
    benchmark_metadata["champion_downloads"] = champion_downloads
    benchmark_metadata["future_reference_timestamp"] = pd.Timestamp(
        future_row["timestamp"].iloc[0]
    ).isoformat()
    phase_timings["PRED"] = time.perf_counter() - phase_started

    return TournamentExecutionResult(
        registered_model_name=registered_model_name,
        raw=filtered_raw,
        train_df=train_df,
        valid_df=valid_df,
        future_row=future_row,
        challenger_results=challenger_results,
        all_results=all_results,
        family_decisions=family_decisions,
        active_results_by_family=active_results_by_family,
        active_result=active_result,
        validation_start=validation_start,
        validation_end=validation_end,
        phase_timings=phase_timings,
        benchmark_metadata=benchmark_metadata,
    )


def publish_tournament_execution(
    execution: TournamentExecutionResult,
    *,
    client: MlflowClient,
) -> dict[str, Any]:
    active_result = execution.active_result
    active_results_by_family = execution.active_results_by_family
    challenger_results = execution.challenger_results
    all_results = execution.all_results
    family_decisions = execution.family_decisions
    future_row = execution.future_row
    prediction_record_updates = dict(
        execution.benchmark_metadata.get("prediction_record_updates") or {}
    )
    mlflow_tags = dict(execution.benchmark_metadata.get("mlflow_tags") or {})

    print_scoreboard(all_results)
    try:
        with start_tournament_mlflow_run():
            try_mlflow_call(
                "set_run_tags",
                lambda: mlflow.set_tags(
                    {
                        "asset": SYMBOL,
                        "timeframe": TIMEFRAME,
                        "validation_hours": str(VALIDATION_HOURS),
                        **mlflow_tags,
                    }
                ),
            )
            try_mlflow_call(
                "log_run_params",
                lambda: mlflow.log_params(
                    {
                        "lookback_hours": LOOKBACK_HOURS,
                        "validation_hours": VALIDATION_HOURS,
                        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
                        "sequence_length": SEQUENCE_LENGTH,
                        "rf_estimators": 400,
                        "xgb_estimators": 500,
                        "lstm_epochs": 40,
                        "transformer_epochs": 36,
                        "nn_epochs": 48,
                    }
                ),
            )

            for decision in family_decisions:
                challenger_result = decision["challenger"]
                champion_result = decision["champion"]
                if decision["null_model_block"] and champion_result is not None:
                    print(
                        f"Promotion blocked for {challenger_result['name']}: "
                        f"F1={challenger_result['f1']:.3f}, Accuracy={challenger_result['accuracy']:.3f}"
                    )
                elif decision["null_model_block"] and champion_result is None:
                    print(
                        f"Bootstrapping missing {decision['registered_model_name']} because no incumbent champion exists. "
                        "The null-model guard only blocks replacing an existing champion: "
                        f"F1={challenger_result['f1']:.3f}, Accuracy={challenger_result['accuracy']:.3f}"
                    )

                if decision["should_promote"]:
                    new_version = try_mlflow_call(
                        f"promote_{decision['family']}",
                        lambda: promote_champion(
                            client=client,
                            registered_model_name=decision["registered_model_name"],
                            winner=challenger_result,
                            validation_start=execution.validation_start,
                            validation_end=execution.validation_end,
                            feature_rows=pd.concat([execution.train_df, execution.valid_df], ignore_index=True),
                            alias=CHAMPION_ALIAS,
                        ),
                    )
                    if new_version is not None:
                        decision["active_result"]["registry_version"] = new_version
                        decision["active_result"]["source"] = "champion"
                        print(
                            f"{challenger_result['name']} -> promoted to {decision['registered_model_name']} version {new_version}"
                        )
                    elif champion_result is not None and decision.get("champion_meta") is not None:
                        decision["active_result"] = champion_result
                        active_results_by_family[decision["family"]] = champion_result
                        decision["active_result"]["registry_version"] = decision["champion_meta"]["version"]
                        print(
                            f"{challenger_result['name']} -> promotion failed, retained existing "
                            f"{decision['registered_model_name']} version {decision['champion_meta']['version']}"
                        )
                    else:
                        print(
                            f"{challenger_result['name']} -> promotion failed and no existing "
                            f"{decision['registered_model_name']} champion is available"
                        )
                elif champion_result is not None:
                    decision["active_result"]["registry_version"] = decision["champion_meta"]["version"]
                    print(
                        f"{champion_result['name']} -> retained as {decision['registered_model_name']} "
                        f"version {decision['champion_meta']['version']}"
                    )
                else:
                    print(
                        f"{challenger_result['name']} -> no existing {decision['registered_model_name']} and not promoted"
                    )

            best_registered_result = next(
                (
                    result
                    for result in sorted(active_results_by_family.values(), key=ranking_key)
                    if result.get("registry_version") is not None
                ),
                None,
            )
            active_result = sorted(active_results_by_family.values(), key=ranking_key)[0]
            if best_registered_result is not None:
                active_result["best_overall_registry_version"] = best_registered_result["registry_version"]
                print(
                    f"Current best across champions: {best_registered_result['candidate'].name} "
                    f"({best_registered_result['family']}) version {best_registered_result['registry_version']}"
                )
            else:
                print("Current best across champions: no registered family champion available yet")

            prediction_record = build_prediction_record(
                active_result=active_result,
                active_results_by_family=active_results_by_family,
                future_row=future_row,
                registered_model_name=execution.registered_model_name,
            )
            if prediction_record_updates:
                prediction_record.update(prediction_record_updates)
            log_step("Write latest prediction metadata")
            LAST_PREDICTION_PATH.write_text(
                json.dumps(prediction_record, indent=2),
                encoding="utf-8",
            )

            log_step("Log tournament results to MLflow")
            try_mlflow_call(
                "log_challenger_summary",
                lambda: log_challenger_summary(challenger_results),
            )
            try_mlflow_call(
                "log_comparison_metrics",
                lambda: log_comparison_metrics(
                    active_results_by_family=active_results_by_family,
                    active_result=active_result,
                ),
            )
            try_mlflow_call(
                "log_tournament_results_text",
                lambda: mlflow.log_text(
                    json.dumps(
                        [
                            serialize_result(row)
                            for row in sorted(all_results, key=ranking_key)
                        ],
                        indent=2,
                    ),
                    "tournament_results.json",
                ),
            )
            try_mlflow_call(
                "log_prediction_record_text",
                lambda: mlflow.log_text(
                    json.dumps(prediction_record, indent=2),
                    LAST_PREDICTION_PATH.name,
                ),
            )
    except Exception as exc:
        print(f"Warning: skipped MLflow run logging after execution: {type(exc).__name__}: {exc}", flush=True)
        prediction_record = build_prediction_record(
            active_result=active_result,
            active_results_by_family=active_results_by_family,
            future_row=future_row,
            registered_model_name=execution.registered_model_name,
        )
        if prediction_record_updates:
            prediction_record.update(prediction_record_updates)
        log_step("Write latest prediction metadata")
        LAST_PREDICTION_PATH.write_text(
            json.dumps(prediction_record, indent=2),
            encoding="utf-8",
        )

    return prediction_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BTC directional tournament and optionally reset the champion from challengers."
    )
    parser.add_argument(
        "--reset-champion-from-challenger",
        action="store_true",
        help="Ignore the current champion comparison and choose from the top challenger only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_step("Initialize tournament")
    set_seed()
    registered_model_name = configure_tracking()
    client = MlflowClient()
    execution = execute_hourly_tournament(
        client=client,
        registered_model_name=registered_model_name,
        reset_champion_from_challenger=args.reset_champion_from_challenger,
    )
    publish_tournament_execution(execution, client=client)

    print(
        f"Upcoming hour probability: {execution.active_result['next_probability']:.1%} chance of UP"
    )
    print(f"Final signal: {execution.active_result['next_signal']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_failed_prediction_record(exc)
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        raise
