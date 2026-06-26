from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd


@dataclass
class TrainOnlyStandardScaler:
    columns: list[str]
    mean_: dict[str, float]
    scale_: dict[str, float]
    fit_split: str = "train"

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Iterable[str]) -> "TrainOnlyStandardScaler":
        selected = list(columns)
        means = frame[selected].mean().to_dict()
        scales = frame[selected].std(ddof=0).replace(0, 1.0).fillna(1.0).to_dict()
        return cls(
            columns=selected,
            mean_={key: float(value) for key, value in means.items()},
            scale_={key: float(value) for key, value in scales.items()},
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in self.columns:
            result[column] = (result[column].astype(float) - self.mean_[column]) / self.scale_[column]
        result[self.columns] = result[self.columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return result

    def metadata(self) -> dict[str, object]:
        return {"fit_split": self.fit_split, "columns": self.columns, "mean": self.mean_, "scale": self.scale_}


class FrameTransform(Protocol):
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...
    def metadata(self) -> dict[str, object]: ...


@dataclass
class TrainOnlyTransform:
    columns: list[str]
    method: str
    estimator: Any = None
    lower_: dict[str, float] | None = None
    upper_: dict[str, float] | None = None
    scale_: dict[str, float] | None = None
    fit_split: str = "train"

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Iterable[str], method: str) -> "TrainOnlyTransform":
        selected = list(columns)
        normalized = method.strip().lower()
        estimator = None
        if normalized == "standard":
            from sklearn.preprocessing import StandardScaler

            estimator = StandardScaler().fit(frame[selected])
        elif normalized == "robust":
            from sklearn.preprocessing import RobustScaler

            estimator = RobustScaler().fit(frame[selected])
        elif normalized == "min_max":
            from sklearn.preprocessing import MinMaxScaler

            estimator = MinMaxScaler().fit(frame[selected])
        elif normalized in {"quantile", "rank"}:
            from sklearn.preprocessing import QuantileTransformer

            estimator = QuantileTransformer(
                n_quantiles=min(100, len(frame)),
                output_distribution="normal" if normalized == "quantile" else "uniform",
                random_state=42,
            ).fit(frame[selected])
        lower = upper = scale = None
        if normalized == "winsorization":
            lower = {key: float(value) for key, value in frame[selected].quantile(0.01).items()}
            upper = {key: float(value) for key, value in frame[selected].quantile(0.99).items()}
        elif normalized == "clipping":
            means = frame[selected].mean()
            std = frame[selected].std(ddof=0).replace(0, 1.0).fillna(1.0)
            lower = {key: float(value) for key, value in (means - 5.0 * std).items()}
            upper = {key: float(value) for key, value in (means + 5.0 * std).items()}
        if normalized == "volatility_normalization":
            scale = {key: float(value) or 1.0 for key, value in frame[selected].std(ddof=0).fillna(1.0).items()}
        return cls(selected, normalized, estimator, lower, upper, scale)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if self.estimator is not None:
            result[self.columns] = self.estimator.transform(result[self.columns])
        elif self.method in {"winsorization", "clipping"}:
            for column in self.columns:
                result[column] = result[column].clip((self.lower_ or {})[column], (self.upper_ or {})[column])
        elif self.method == "rolling_zscore":
            for column in self.columns:
                mean = result[column].rolling(24, min_periods=1).mean()
                std = result[column].rolling(24, min_periods=2).std().replace(0, 1.0).fillna(1.0)
                result[column] = (result[column] - mean) / std
        elif self.method == "volatility_normalization":
            for column in self.columns:
                result[column] = result[column] / (self.scale_ or {})[column]
        elif self.method == "stationarity":
            signed_log = np.sign(result[self.columns]) * np.log1p(np.abs(result[self.columns]))
            result[self.columns] = signed_log.diff().fillna(0.0)
        elif self.method == "differencing":
            result[self.columns] = result[self.columns].diff().fillna(0.0)
        elif self.method == "percentage_change":
            result[self.columns] = result[self.columns].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def metadata(self) -> dict[str, object]:
        return {"fit_split": self.fit_split, "columns": self.columns, "method": self.method}


def fit_train_only_transform(frame: pd.DataFrame, columns: Iterable[str], method: str) -> FrameTransform:
    normalized = method.strip().lower()
    if normalized in {"none", "no_scaler"}:
        return TrainOnlyTransform.fit(frame, columns, "none")
    return TrainOnlyTransform.fit(frame, columns, normalized)
