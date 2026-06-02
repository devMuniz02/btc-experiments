from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.training import ActorCriticPolicy, MambaContinuationActorCriticPolicy, train_model


@dataclass(frozen=True)
class RLModelSpec:
    total_updates: int = 5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 0.0003
    entropy_coef: float = 0.01


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> RLModelSpec:
    return RLModelSpec(
        total_updates=int(hyperparameters.get("total_updates", 5)),
        ppo_epochs=int(hyperparameters.get("ppo_epochs", 4)),
        minibatch_size=int(hyperparameters.get("minibatch_size", 256)),
        learning_rate=float(hyperparameters.get("learning_rate", 0.0003)),
        entropy_coef=float(hyperparameters.get("entropy_coef", 0.01)),
    )


__all__ = [
    "ActorCriticPolicy",
    "MambaContinuationActorCriticPolicy",
    "RLModelSpec",
    "train_model",
    "validate_hyperparameters",
]
