from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LSTMModelSpec:
    sequence_length: int = 24
    hidden_dim: int = 128
    epochs: int = 20
    learning_rate: float = 0.001


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> LSTMModelSpec:
    return LSTMModelSpec(
        sequence_length=int(hyperparameters.get("sequence_length", 24)),
        hidden_dim=int(hyperparameters.get("hidden_dim", 128)),
        epochs=int(hyperparameters.get("epochs", 20)),
        learning_rate=float(hyperparameters.get("learning_rate", 0.001)),
    )
