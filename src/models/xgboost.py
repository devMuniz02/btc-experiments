from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class XGBoostModelSpec:
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.03


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> XGBoostModelSpec:
    return XGBoostModelSpec(
        n_estimators=int(hyperparameters.get("n_estimators", 500)),
        max_depth=int(hyperparameters.get("max_depth", 4)),
        learning_rate=float(hyperparameters.get("learning_rate", 0.03)),
    )
