from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.training import MambaClassificationPolicy, MambaContinuationActorCriticPolicy, train_model


@dataclass(frozen=True)
class MambaModelSpec:
    sequence_length: int = 24
    hidden_dim: int = 192
    num_layers: int = 3
    epochs: int = 20
    learning_rate: float = 0.001


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> MambaModelSpec:
    return MambaModelSpec(
        sequence_length=int(hyperparameters.get("sequence_length", 24)),
        hidden_dim=int(hyperparameters.get("hidden_dim", 192)),
        num_layers=int(hyperparameters.get("num_layers", 3)),
        epochs=int(hyperparameters.get("epochs", 20)),
        learning_rate=float(hyperparameters.get("learning_rate", 0.001)),
    )


__all__ = [
    "MambaClassificationPolicy",
    "MambaContinuationActorCriticPolicy",
    "MambaModelSpec",
    "train_model",
    "validate_hyperparameters",
]
