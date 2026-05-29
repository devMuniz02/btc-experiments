from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.btc_direction_learning.dataset import DirectionDatasetBundle
from src.btc_direction_learning.env import BTCDirectionEnv, ENV_VERSION_BINARY
from src.btc_direction_learning.evaluation import clone_state_dict
from src.btc_direction_learning.models import ClassificationPolicy, extract_policy_logits, select_actions_from_logits
from src.btc_direction_learning.progress import ProgressLogger


def build_policy(sequence_length: int, feature_dim: int, action_dim: int = 2) -> ClassificationPolicy:
    return ClassificationPolicy(sequence_length=sequence_length, feature_dim=feature_dim, action_dim=action_dim)


def _build_loader(observations: np.ndarray, actions: np.ndarray, batch_size: int = 16) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)


def _fit_policy(
    policy: nn.Module,
    observations: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[float, float]:
    loader = _build_loader(observations, actions, batch_size=batch_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    final_loss = 0.0
    final_accuracy = 0.0

    for _ in range(epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for batch_observations, batch_actions in loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)
            logits = policy(batch_observations)
            loss = criterion(logits, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch_actions.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += int((predictions == batch_actions).sum().item())
            total += int(batch_actions.size(0))

        final_loss = total_loss / max(1, total)
        final_accuracy = correct / max(1, total)

    return final_loss, final_accuracy


def _collect_policy_rollout(
    policy: nn.Module,
    env: BTCDirectionEnv,
    device: torch.device,
    ternary_confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    observation = env.reset()
    done = False
    observations = []
    expert_actions = []
    rewards = []

    while not done:
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = extract_policy_logits(policy, obs_tensor)
            action = int(
                select_actions_from_logits(
                    logits,
                    env_version=env.env_version,
                    ternary_confidence_threshold=ternary_confidence_threshold,
                ).item()
            )
        next_observation, reward, done, info = env.step(action)
        observations.append(observation)
        expert_actions.append(int(info["expert_action"]))
        rewards.append(float(reward))
        observation = next_observation

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(expert_actions, dtype=np.int64),
        float(np.mean(rewards)) if rewards else 0.0,
    )


def train(
    dataset_bundle: DirectionDatasetBundle,
    policy: nn.Module,
    device: torch.device,
    env_version: str = ENV_VERSION_BINARY,
    ternary_confidence_threshold: float = 0.0,
    rounds: int = 12,
    fit_epochs_per_round: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
) -> tuple[list[dict], dict[str, dict[str, torch.Tensor]], dict]:
    base_observations, base_actions, _ = dataset_bundle.get_split_data("train")
    aggregated_observations = [base_observations]
    aggregated_actions = [base_actions]
    history: list[dict] = []
    checkpoints = {"initial": clone_state_dict(policy)}
    midpoint_round = max(1, rounds // 2)
    train_env = BTCDirectionEnv(dataset_bundle=dataset_bundle, split_name="train", env_version=env_version)
    progress = ProgressLogger(label="DAgger", total_steps=rounds)

    for round_idx in range(1, rounds + 1):
        rollout_observations, rollout_expert_actions, rollout_mean_reward = _collect_policy_rollout(
            policy=policy,
            env=train_env,
            device=device,
            ternary_confidence_threshold=ternary_confidence_threshold,
        )
        aggregated_observations.append(rollout_observations)
        aggregated_actions.append(rollout_expert_actions)

        fit_observations = np.concatenate(aggregated_observations, axis=0)
        fit_actions = np.concatenate(aggregated_actions, axis=0)
        loss, accuracy = _fit_policy(
            policy=policy,
            observations=fit_observations,
            actions=fit_actions,
            device=device,
            epochs=fit_epochs_per_round,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        history.append(
            {
                "round": round_idx,
                "step": round_idx,
                "loss": loss,
                "train_accuracy": accuracy,
                "rollout_mean_reward": rollout_mean_reward,
                "dataset_size": int(len(fit_actions)),
            }
        )
        progress.log(
            round_idx,
            (
                f"loss={history[-1]['loss']:.6f}, "
                f"train_accuracy={history[-1]['train_accuracy']:.4f}, "
                f"rollout_mean_reward={history[-1]['rollout_mean_reward']:.4f}, "
                f"dataset_size={history[-1]['dataset_size']}"
            ),
        )
        if round_idx == midpoint_round:
            checkpoints["midpoint"] = clone_state_dict(policy)

    checkpoints["final"] = clone_state_dict(policy)
    if "midpoint" not in checkpoints:
        checkpoints["midpoint"] = clone_state_dict(policy)

    plot_config = {
        "title": "DAgger Training Curve",
        "series": [
            {"key": "loss", "label": "Imitation loss", "color": "#9467bd"},
            {"key": "rollout_mean_reward", "label": "Rollout mean reward", "color": "#2ca02c", "alpha": 0.8},
        ],
    }
    return history, checkpoints, plot_config
