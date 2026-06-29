from __future__ import annotations

import numpy as np
import pandas as pd


def build_basic_features(frame: pd.DataFrame, target_frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    data = frame.copy()
    close = data["close"].astype(float)
    target_source = target_frame if target_frame is not None else frame
    target_close = target_source["close"].astype(float).reset_index(drop=True)
    open_ = data["open"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)

    data["return"] = close.pct_change().fillna(0.0)
    data["log_return"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    data["hl_range"] = ((high - low) / close.replace(0, np.nan)).fillna(0.0)
    data["oc_body"] = ((close - open_) / open_.replace(0, np.nan)).fillna(0.0)
    data["upper_wick"] = ((high - np.maximum(open_, close)) / close.replace(0, np.nan)).fillna(0.0)
    data["lower_wick"] = ((np.minimum(open_, close) - low) / close.replace(0, np.nan)).fillna(0.0)
    data["rolling_return_mean_6"] = data["return"].rolling(6, min_periods=1).mean()
    data["rolling_return_std_6"] = data["return"].rolling(6, min_periods=1).std().fillna(0.0)
    data["rolling_return_mean_24"] = data["return"].rolling(24, min_periods=1).mean()
    data["rolling_return_std_24"] = data["return"].rolling(24, min_periods=1).std().fillna(0.0)
    data["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    data["volume_z_24"] = (
        (volume - volume.rolling(24, min_periods=1).mean()) / volume.rolling(24, min_periods=1).std()
    ).fillna(0.0)
    if "timestamp" in data:
        ts = pd.to_datetime(data["timestamp"], utc=True)
        data["hour"] = ts.dt.hour / 23.0
        data["dayofweek"] = ts.dt.dayofweek / 6.0
    else:
        data["hour"] = 0.0
        data["dayofweek"] = 0.0
    data["target"] = (target_close.shift(-1) > target_close).astype(int)
    data = data.iloc[:-1].reset_index(drop=True)
    features = [
        "return",
        "log_return",
        "hl_range",
        "oc_body",
        "upper_wick",
        "lower_wick",
        "rolling_return_mean_6",
        "rolling_return_std_6",
        "rolling_return_mean_24",
        "rolling_return_std_24",
        "volume_change",
        "volume_z_24",
        "hour",
        "dayofweek",
    ]
    return data, features
