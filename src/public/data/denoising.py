from __future__ import annotations

import math

import pandas as pd

SUPPORTED_DENOISING_METHODS = (
    "none",
    "raw",
    "rolling_mean",
    "rolling_median",
    "rolling_gaussian",
    "gaussian",
    "ema",
    "ema_low_pass",
    "fourier_low_pass",
    "wavelet",
    "kalman",
    "volatility_normalized",
    "feature_only",
    "price_only",
    "volume_only",
)
TIME_OR_TARGET_COLUMNS = {"timestamp", "target", "hour", "dayofweek", "day_of_week", "hour_of_day"}


def denoisable_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = list(frame.select_dtypes(include="number").columns)
    return [column for column in numeric_columns if column not in TIME_OR_TARGET_COLUMNS]


def _causal_gaussian_weights(window: int, sigma: float) -> list[float]:
    raw = [math.exp(-0.5 * ((window - 1 - index) / sigma) ** 2) for index in range(window)]
    total = sum(raw)
    return [value / total for value in raw]


def _causal_weighted_average(values: list[float], weights: list[float], index: int) -> float:
    available = min(index + 1, len(weights))
    active_values = values[index - available + 1 : index + 1]
    active_weights = weights[-available:]
    total = sum(active_weights)
    return float(sum(value * weight for value, weight in zip(active_values, active_weights)) / total)


def causal_denoise_frame(
    frame: pd.DataFrame,
    *,
    method: str,
    window: int = 6,
    sigma: float = 2.0,
    alpha: float = 0.25,
) -> pd.DataFrame:
    normalized = method.strip().lower()
    if normalized not in SUPPORTED_DENOISING_METHODS:
        raise ValueError(f"Unsupported denoising method: {method}")
    denoised = frame.copy(deep=True)
    if normalized in {"none", "raw", "feature_only"}:
        return denoised
    columns = denoisable_columns(frame)
    if normalized == "price_only":
        columns = [column for column in columns if column in {"open", "high", "low", "close"}]
        normalized = "ema"
    elif normalized == "volume_only":
        columns = [column for column in columns if column == "volume"]
        normalized = "ema"
    if normalized == "rolling_mean":
        denoised[columns] = frame[columns].rolling(window=window, min_periods=1).mean()
    elif normalized == "rolling_median":
        denoised[columns] = frame[columns].rolling(window=window, min_periods=1).median()
    elif normalized in {"rolling_gaussian", "gaussian"}:
        weights = _causal_gaussian_weights(window, sigma)
        for column in columns:
            values = [float(value) for value in frame[column].tolist()]
            denoised[column] = [_causal_weighted_average(values, weights, index) for index in range(len(values))]
    elif normalized in {"ema", "ema_low_pass"}:
        denoised[columns] = frame[columns].ewm(alpha=alpha, adjust=False).mean()
    elif normalized == "fourier_low_pass":
        import numpy as np

        for column in columns:
            values = frame[column].to_numpy(dtype=float)
            filtered: list[float] = []
            for index in range(len(values)):
                history = values[max(0, index - 31) : index + 1].copy()
                spectrum = np.fft.rfft(history)
                cutoff = max(1, len(spectrum) // 4)
                spectrum[cutoff:] = 0
                filtered.append(float(np.fft.irfft(spectrum, n=len(history))[-1]))
            denoised[column] = filtered
    elif normalized == "wavelet":
        import numpy as np
        import pywt

        for column in columns:
            values = frame[column].to_numpy(dtype=float)
            output: list[float] = []
            for index in range(len(values)):
                history = values[max(0, index - 31) : index + 1].copy()
                if len(history) < 8:
                    output.append(float(history[-1]))
                    continue
                level = min(2, pywt.dwt_max_level(len(history), pywt.Wavelet("db2").dec_len))
                if level < 1:
                    output.append(float(history[-1]))
                    continue
                coeffs = pywt.wavedec(history, "db2", level=level)
                threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(max(2, len(history))))
                filtered = [coeffs[0], *(pywt.threshold(value, threshold, mode="soft") for value in coeffs[1:])]
                output.append(float(pywt.waverec(filtered, "db2")[: len(history)][-1]))
            denoised[column] = output
    elif normalized == "kalman":
        for column in columns:
            values = [float(value) for value in frame[column]]
            estimate = values[0]
            error = 1.0
            output: list[float] = []
            for value in values:
                error += 1e-5
                gain = error / (error + 1e-2)
                estimate += gain * (value - estimate)
                error *= 1.0 - gain
                output.append(estimate)
            denoised[column] = output
    elif normalized == "volatility_normalized":
        for column in columns:
            rolling_std = frame[column].rolling(window, min_periods=2).std().replace(0, 1.0).fillna(1.0)
            denoised[column] = frame[column] / rolling_std
    return denoised


def variation_from_config(config: dict) -> str:
    data_variation = config.get("data_variation") or {}
    variation = str(data_variation.get("variation") or data_variation.get("single_dataset_variation") or "none")
    return variation.strip().lower() or "none"
