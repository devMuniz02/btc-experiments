from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.training import ClassificationPolicy, train_model


@dataclass(frozen=True)
class NNModelSpec:
    sequence_length: int = 24
    hidden_dim: int = 256
    epochs: int = 20
    learning_rate: float = 0.001


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> NNModelSpec:
    return NNModelSpec(
        sequence_length=int(hyperparameters.get("sequence_length", 24)),
        hidden_dim=int(hyperparameters.get("hidden_dim", 256)),
        epochs=int(hyperparameters.get("epochs", 20)),
        learning_rate=float(hyperparameters.get("learning_rate", 0.001)),
    )


__all__ = ["ClassificationPolicy", "NNModelSpec", "train_model", "validate_hyperparameters"]
