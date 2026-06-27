from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.public.config.loader import stable_hash


REQUIRED_DRIFT_ARTIFACTS = [
    "train_to_validation",
    "train_to_test",
    "validation_to_test",
    "rolling_validation_test",
    "raw_vs_denoised",
]


@dataclass(frozen=True)
class DriftMethodConfig:
    method: str = "population_summary"
    statistics: tuple[str, ...] = ("mean", "std", "missing_rate")
    rolling_windows: tuple[int, ...] = (24, 168)
    diagnostic_by_default: bool = True
    affects_ranking: bool = False
    affects_retraining: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftCacheKey:
    market_id: str
    dataset_fingerprint: str
    feature_set_hash: str
    split_definition: dict[str, Any]
    method_config: DriftMethodConfig = field(default_factory=DriftMethodConfig)

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "market_id": self.market_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_set_hash": self.feature_set_hash,
            "split_definition": self.split_definition,
            "method_config": self.method_config.as_dict(),
        }
        if include_fingerprint:
            payload["cache_key"] = self.fingerprint
        return payload


def _summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "missing_rate": float(values.isna().mean()),
        }
    return result


def compare_population(source: pd.DataFrame, comparison: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    source_summary = _summary(source, columns)
    comparison_summary = _summary(comparison, columns)
    distances: dict[str, float] = {}
    for column in columns:
        denom = abs(source_summary[column]["std"]) or 1.0
        distances[column] = abs(comparison_summary[column]["mean"] - source_summary[column]["mean"]) / denom
    max_distance = max(distances.values()) if distances else 0.0
    return {
        "method": "population_summary",
        "source_summary": source_summary,
        "comparison_summary": comparison_summary,
        "z_distance_by_column": distances,
        "max_z_distance": float(max_distance),
        "status": "critical" if max_distance >= 2.5 else "warning" if max_distance >= 1.5 else "ok",
        "affects_ranking": False,
    }


def drift_artifact_manifest(cache_key: DriftCacheKey) -> dict[str, Any]:
    return {
        "cache_key": cache_key.fingerprint,
        "key_inputs": cache_key.as_dict(),
        "required_artifacts": list(REQUIRED_DRIFT_ARTIFACTS),
        "artifact_paths": {name: f"drift/{name}.json" for name in REQUIRED_DRIFT_ARTIFACTS},
        "selection_policy": "drift diagnostics do not affect validation ranking unless explicitly enabled",
    }
