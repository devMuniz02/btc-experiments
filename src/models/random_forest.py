from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.training import train_model


@dataclass(frozen=True)
class RandomForestModelSpec:
    n_estimators: int = 300
    max_depth: int = 10
    min_samples_leaf: int = 5
    n_jobs: int = -1


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> RandomForestModelSpec:
    return RandomForestModelSpec(
        n_estimators=int(hyperparameters.get("n_estimators", 300)),
        max_depth=int(hyperparameters.get("max_depth", 10)),
        min_samples_leaf=int(hyperparameters.get("min_samples_leaf", 5)),
        n_jobs=int(hyperparameters.get("n_jobs", -1)),
    )


__all__ = ["RandomForestModelSpec", "train_model", "validate_hyperparameters"]
