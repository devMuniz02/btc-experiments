from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.btc_direction_learning.dataset import DirectionDatasetBundle
from src.btc_direction_learning.evaluation import clone_state_dict
from src.btc_direction_learning.models import ClassificationPolicy
from src.btc_direction_learning.progress import ProgressLogger


def build_policy(sequence_length: int, feature_dim: int, action_dim: int = 2) -> ClassificationPolicy:
    return ClassificationPolicy(sequence_length=sequence_length, feature_dim=feature_dim, action_dim=action_dim)


def _build_loader(observations: np.ndarray, actions: np.ndarray, batch_size: int = 16) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)


def train(
    dataset_bundle: DirectionDatasetBundle,
    policy: nn.Module,
    device: torch.device,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
) -> tuple[list[dict], dict[str, dict[str, torch.Tensor]], dict]:
    observations, expert_actions, _ = dataset_bundle.get_split_data("train")
    data_loader = _build_loader(observations, expert_actions, batch_size=batch_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    checkpoints = {"initial": clone_state_dict(policy)}
    midpoint_epoch = max(1, epochs // 2)
    progress = ProgressLogger(label="BC", total_steps=epochs)

    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_observations, batch_actions in data_loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)
            logits = policy(batch_observations)
            loss = criterion(logits, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * batch_actions.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += int((predictions == batch_actions).sum().item())
            total += int(batch_actions.size(0))

        history.append(
            {
                "epoch": epoch,
                "step": epoch,
                "loss": epoch_loss / max(1, total),
                "train_accuracy": correct / max(1, total),
            }
        )
        progress.log(
            epoch,
            f"loss={history[-1]['loss']:.6f}, train_accuracy={history[-1]['train_accuracy']:.4f}",
        )
        if epoch == midpoint_epoch:
            checkpoints["midpoint"] = clone_state_dict(policy)

    checkpoints["final"] = clone_state_dict(policy)
    if "midpoint" not in checkpoints:
        checkpoints["midpoint"] = clone_state_dict(policy)

    plot_config = {
        "title": "Behavior Cloning Training Curve",
        "series": [
            {"key": "loss", "label": "Classification loss", "color": "#d62728"},
            {"key": "train_accuracy", "label": "Train imitation accuracy", "color": "#1f77b4", "alpha": 0.8},
        ],
    }
    return history, checkpoints, plot_config
