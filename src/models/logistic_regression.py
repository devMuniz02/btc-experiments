from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.training import train_model


@dataclass(frozen=True)
class LogisticRegressionModelSpec:
    penalty: str = "l2"
    C: float = 1.0
    solver: str = "saga"
    max_iter: int = 1000
    tol: float = 1e-4
    class_weight: str | None = None
    fit_intercept: bool = True
    n_jobs: int = -1
    device: str = "auto"


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> LogisticRegressionModelSpec:
    return LogisticRegressionModelSpec(
        penalty=str(hyperparameters.get("penalty", "l2")),
        C=float(hyperparameters.get("C", 1.0)),
        solver=str(hyperparameters.get("solver", "saga")),
        max_iter=int(hyperparameters.get("max_iter", 1000)),
        tol=float(hyperparameters.get("tol", 1e-4)),
        class_weight=hyperparameters.get("class_weight", None),
        fit_intercept=bool(hyperparameters.get("fit_intercept", True)),
        n_jobs=int(hyperparameters.get("n_jobs", -1)),
        device=str(hyperparameters.get("device", "auto")),
    )


__all__ = ["LogisticRegressionModelSpec", "train_model", "validate_hyperparameters"]
