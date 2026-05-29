from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from src.btc_direction_learning.continuation_models import (
    LSTMContinuationActorCriticPolicy,
    MLPContinuationActorCriticPolicy,
    TransformerContinuationActorCriticPolicy,
)
from src.btc_direction_learning.env import BTCDirectionEnv
from src.btc_direction_learning.evaluation import clone_state_dict, evaluate_policy
from src.btc_direction_learning.progress import ProgressLogger


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def build_policy(
    sequence_length: int,
    feature_dim: int,
    action_dim: int = 2,
    policy_family: str = "mlp",
) -> nn.Module:
    if policy_family == "mlp":
        return MLPContinuationActorCriticPolicy(sequence_length=sequence_length, feature_dim=feature_dim, action_dim=action_dim)
    if policy_family == "lstm":
        return LSTMContinuationActorCriticPolicy(sequence_length=sequence_length, feature_dim=feature_dim, action_dim=action_dim)
    if policy_family == "transformer":
        return TransformerContinuationActorCriticPolicy(sequence_length=sequence_length, feature_dim=feature_dim, action_dim=action_dim)
    raise ValueError(f"Unknown continuation PPO policy family: {policy_family}")


def _build_rollout_batch(
    observations: list[np.ndarray],
    actions: list[int],
    log_probs: list[float],
    rewards: list[float],
    dones: list[float],
    values: list[float],
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> tuple[RolloutBatch, dict]:
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards_arr)
    returns = np.zeros_like(rewards_arr)
    next_value = 0.0
    next_advantage = 0.0

    for step in reversed(range(len(rewards_arr))):
        mask = 1.0 - dones_arr[step]
        delta = rewards_arr[step] + gamma * next_value * mask - values_arr[step]
        next_advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[step] = next_advantage
        returns[step] = advantages[step] + values_arr[step]
        next_value = values_arr[step]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch = RolloutBatch(
        observations=torch.tensor(np.asarray(observations), dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.int64),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        rewards=torch.tensor(rewards_arr, dtype=torch.float32),
        dones=torch.tensor(dones_arr, dtype=torch.float32),
        values=torch.tensor(values_arr, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    metrics = {
        "trajectory_return": float(rewards_arr.sum()),
        "mean_reward": float(rewards_arr.mean()),
        "steps": int(len(rewards_arr)),
    }
    return batch, metrics


def collect_vectorized_rollouts(
    env: BTCDirectionEnv,
    num_envs: int,
    policy: nn.Module,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[RolloutBatch], list[dict]]:
    envs = [
        BTCDirectionEnv(dataset_bundle=env.dataset_bundle, split_name=env.split_name, env_version=env.env_version)
        for _ in range(num_envs)
    ]
    active_observations = [sub_env.reset() for sub_env in envs]
    trajectory_buffers = [
        {"observations": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}
        for _ in envs
    ]
    completed = [False for _ in envs]

    while not all(completed):
        batch_observations = np.stack(active_observations, axis=0)
        obs_tensor = torch.tensor(batch_observations, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, values = policy(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        for env_index, sub_env in enumerate(envs):
            if completed[env_index]:
                continue
            next_observation, reward, done, _ = sub_env.step(int(actions[env_index].item()))
            buffer = trajectory_buffers[env_index]
            buffer["observations"].append(active_observations[env_index])
            buffer["actions"].append(int(actions[env_index].item()))
            buffer["log_probs"].append(float(log_probs[env_index].item()))
            buffer["rewards"].append(float(reward))
            buffer["dones"].append(float(done))
            buffer["values"].append(float(values[env_index].item()))
            active_observations[env_index] = next_observation
            completed[env_index] = done

    rollout_batches: list[RolloutBatch] = []
    rollout_metrics: list[dict] = []
    for buffer in trajectory_buffers:
        rollout_batch, metrics = _build_rollout_batch(
            observations=buffer["observations"],
            actions=buffer["actions"],
            log_probs=buffer["log_probs"],
            rewards=buffer["rewards"],
            dones=buffer["dones"],
            values=buffer["values"],
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        rollout_batches.append(rollout_batch)
        rollout_metrics.append(metrics)

    return rollout_batches, rollout_metrics


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float = 1.0,
    reference_policy: nn.Module | None = None,
    previous_policy_kl_coef: float = 0.0,
) -> dict:
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl_loss = 0.0
    total_updates = 0
    sample_count = batch.observations.shape[0]

    policy_device = next(policy.parameters()).device

    for _ in range(epochs):
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            obs = batch.observations[idx].to(policy_device, non_blocking=True)
            actions = batch.actions[idx].to(policy_device, non_blocking=True)
            old_log_probs = batch.log_probs[idx].to(policy_device, non_blocking=True)
            advantages = batch.advantages[idx].to(policy_device, non_blocking=True)
            returns = batch.returns[idx].to(policy_device, non_blocking=True)

            logits, values = policy(obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            kl_loss = torch.tensor(0.0, device=obs.device)
            if reference_policy is not None and previous_policy_kl_coef > 0.0:
                with torch.no_grad():
                    reference_logits, _ = reference_policy(obs)
                    reference_dist = Categorical(logits=reference_logits)
                kl_loss = kl_divergence(reference_dist, dist).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns)
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy + (previous_policy_kl_coef * kl_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None and float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_kl_loss += float(kl_loss.item())
            total_updates += 1

    return {
        "policy_loss": total_policy_loss / max(1, total_updates),
        "value_loss": total_value_loss / max(1, total_updates),
        "entropy": total_entropy / max(1, total_updates),
        "kl_loss": total_kl_loss / max(1, total_updates),
    }


def train(
    env: BTCDirectionEnv,
    policy: nn.Module,
    device: torch.device,
    policy_family: str,
    total_updates: int = 5,
    trajectories_per_update: int = 128,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    learning_rate: float = 3e-4,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    max_grad_norm: float = 1.0,
    previous_policy_kl_coef: float = 0.0,
    reference_policy_state_dict: dict[str, torch.Tensor] | None = None,
    progress_train_eval_env: BTCDirectionEnv | None = None,
    progress_recent_eval_env: BTCDirectionEnv | None = None,
    progress_test_env: BTCDirectionEnv | None = None,
    progress_base_train_accuracy: float | None = None,
    progress_base_recent_accuracy: float | None = None,
    progress_base_test_accuracy: float | None = None,
    enable_progress: bool = True,
    progress_eval_env: BTCDirectionEnv | None = None,
    progress_base_eval_accuracy: float | None = None,
) -> tuple[list[dict], dict[str, dict[str, torch.Tensor]], dict]:
    if progress_eval_env is not None and progress_test_env is None:
        progress_test_env = progress_eval_env
    if progress_base_eval_accuracy is not None and progress_base_test_accuracy is None:
        progress_base_test_accuracy = progress_base_eval_accuracy

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict] = []
    checkpoints = {"initial": clone_state_dict(policy)}
    midpoint_update = max(1, total_updates // 2)
    progress = ProgressLogger(label="PPO-CONTINUE", total_steps=total_updates) if enable_progress else None

    reference_policy: nn.Module | None = None
    if reference_policy_state_dict is not None and previous_policy_kl_coef > 0.0:
        reference_policy = build_policy(
            sequence_length=env.dataset_bundle.sequence_length,
            feature_dim=env.dataset_bundle.feature_dim,
            action_dim=env.action_dim,
            policy_family=policy_family,
        ).to(device)
        reference_policy.load_state_dict(reference_policy_state_dict)
        reference_policy.eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)

    for update_idx in range(1, total_updates + 1):
        rollout_batches, rollout_metrics = collect_vectorized_rollouts(
            env=env,
            num_envs=trajectories_per_update,
            policy=policy,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        merged_batch = RolloutBatch(
            observations=torch.cat([rb.observations for rb in rollout_batches], dim=0),
            actions=torch.cat([rb.actions for rb in rollout_batches], dim=0),
            log_probs=torch.cat([rb.log_probs for rb in rollout_batches], dim=0),
            rewards=torch.cat([rb.rewards for rb in rollout_batches], dim=0),
            dones=torch.cat([rb.dones for rb in rollout_batches], dim=0),
            values=torch.cat([rb.values for rb in rollout_batches], dim=0),
            returns=torch.cat([rb.returns for rb in rollout_batches], dim=0),
            advantages=torch.cat([rb.advantages for rb in rollout_batches], dim=0),
        )
        update_metrics = ppo_update(
            policy=policy,
            optimizer=optimizer,
            batch=merged_batch,
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            reference_policy=reference_policy,
            previous_policy_kl_coef=previous_policy_kl_coef,
        )
        mean_reward = float(np.mean([metric["mean_reward"] for metric in rollout_metrics])) if rollout_metrics else 0.0
        mean_return = float(np.mean([metric["trajectory_return"] for metric in rollout_metrics])) if rollout_metrics else 0.0
        train_accuracy = None
        recent_accuracy = None
        test_accuracy = None
        if progress_train_eval_env is not None:
            train_result = evaluate_policy(progress_train_eval_env, policy, device=device)
            train_accuracy = float(train_result["accuracy"])
        if progress_recent_eval_env is not None:
            recent_result = evaluate_policy(progress_recent_eval_env, policy, device=device)
            recent_accuracy = float(recent_result["accuracy"])
        if progress_test_env is not None:
            test_result = evaluate_policy(progress_test_env, policy, device=device)
            test_accuracy = float(test_result["accuracy"])
        history.append(
            {
                "update": update_idx,
                "step": update_idx,
                "mean_reward": mean_reward,
                "mean_return": mean_return,
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
                "kl_loss": update_metrics["kl_loss"],
                "train_accuracy": train_accuracy,
                "recent_accuracy": recent_accuracy,
                "test_accuracy": test_accuracy,
                "base_train_accuracy": progress_base_train_accuracy,
                "base_recent_accuracy": progress_base_recent_accuracy,
                "base_test_accuracy": progress_base_test_accuracy,
                "ppo_train_accuracy": train_accuracy,
                "ppo_recent_accuracy": recent_accuracy,
                "ppo_test_accuracy": test_accuracy,
            }
        )
        if progress is not None:
            accuracy_parts: list[str] = []
            if train_accuracy is not None:
                accuracy_parts.append(f"ppo_train_accuracy={train_accuracy:.4f}")
            if progress_base_train_accuracy is not None:
                accuracy_parts.append(f"base_train_accuracy={progress_base_train_accuracy:.4f}")
            if recent_accuracy is not None:
                accuracy_parts.append(f"ppo_recent_accuracy={recent_accuracy:.4f}")
            if progress_base_recent_accuracy is not None:
                accuracy_parts.append(f"base_recent_accuracy={progress_base_recent_accuracy:.4f}")
            if test_accuracy is not None:
                accuracy_parts.append(f"ppo_test_accuracy={test_accuracy:.4f}")
            if progress_base_test_accuracy is not None:
                accuracy_parts.append(f"base_test_accuracy={progress_base_test_accuracy:.4f}")
            accuracy_part = f", {', '.join(accuracy_parts)}" if accuracy_parts else ""
            progress.log(
                update_idx,
                (
                    f"mean_reward={mean_reward:.4f}, mean_return={mean_return:.4f}, "
                    f"policy_loss={update_metrics['policy_loss']:.4f}, value_loss={update_metrics['value_loss']:.4f}"
                    f"{accuracy_part}"
                ),
            )
        if update_idx == midpoint_update:
            checkpoints["midpoint"] = clone_state_dict(policy)

    checkpoints["final"] = clone_state_dict(policy)
    if "midpoint" not in checkpoints:
        checkpoints["midpoint"] = clone_state_dict(policy)

    plot_config = {
        "title": "PPO Continuation Mean Rewards Over Trajectories",
        "series": [
            {"key": "mean_reward", "label": "Mean reward", "color": "#1f77b4"},
            {"key": "mean_return", "label": "Mean return", "color": "#2ca02c"},
        ],
    }
    return history, checkpoints, plot_config
