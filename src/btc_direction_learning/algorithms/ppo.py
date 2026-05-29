from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from src.btc_direction_learning.env import BTCDirectionEnv
from src.btc_direction_learning.evaluation import clone_state_dict, evaluate_policy
from src.btc_direction_learning.models import ActorCriticPolicy
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


def build_policy(sequence_length: int, feature_dim: int, action_dim: int = 2) -> ActorCriticPolicy:
    return ActorCriticPolicy(sequence_length=sequence_length, feature_dim=feature_dim, action_dim=action_dim)


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
        observations=torch.tensor(np.asarray(observations), dtype=torch.float32, device=device),
        actions=torch.tensor(actions, dtype=torch.int64, device=device),
        log_probs=torch.tensor(log_probs, dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards_arr, dtype=torch.float32, device=device),
        dones=torch.tensor(dones_arr, dtype=torch.float32, device=device),
        values=torch.tensor(values_arr, dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
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
        BTCDirectionEnv(
            dataset_bundle=env.dataset_bundle,
            split_name=env.split_name,
            env_version=env.env_version,
        )
        for _ in range(num_envs)
    ]
    active_observations = [sub_env.reset() for sub_env in envs]
    trajectory_buffers = [
        {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }
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


def collect_sequential_rollouts(
    env: BTCDirectionEnv,
    num_envs: int,
    policy: nn.Module,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[RolloutBatch], list[dict]]:
    rollout_batches: list[RolloutBatch] = []
    rollout_metrics: list[dict] = []

    for _ in range(num_envs):
        sub_env = BTCDirectionEnv(
            dataset_bundle=env.dataset_bundle,
            split_name=env.split_name,
            env_version=env.env_version,
        )
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        observation = sub_env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = policy(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_observation, reward, done, _ = sub_env.step(int(action.item()))
            observations.append(observation)
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.item()))
            rewards.append(float(reward))
            dones.append(float(done))
            values.append(float(value.item()))
            observation = next_observation

        rollout_batch, metrics = _build_rollout_batch(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
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
    reference_policy: nn.Module | None = None,
    previous_policy_kl_coef: float = 0.0,
    epoch_callback: Callable[[dict], None] | None = None,
) -> dict:
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl_loss = 0.0
    total_updates = 0
    sample_count = batch.observations.shape[0]

    for epoch_idx in range(1, epochs + 1):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_kl_loss = 0.0
        epoch_updates = 0
        permutation = torch.randperm(sample_count, device=batch.observations.device)
        for start in range(0, sample_count, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            obs = batch.observations[idx]
            actions = batch.actions[idx]
            old_log_probs = batch.log_probs[idx]
            advantages = batch.advantages[idx]
            returns = batch.returns[idx]

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
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_kl_loss += float(kl_loss.item())
            epoch_policy_loss += float(policy_loss.item())
            epoch_value_loss += float(value_loss.item())
            epoch_entropy += float(entropy.item())
            epoch_kl_loss += float(kl_loss.item())
            total_updates += 1
            epoch_updates += 1

        if epoch_callback is not None:
            epoch_callback(
                {
                    "epoch": int(epoch_idx),
                    "epochs": int(epochs),
                    "policy_loss": epoch_policy_loss / max(1, epoch_updates),
                    "value_loss": epoch_value_loss / max(1, epoch_updates),
                    "entropy": epoch_entropy / max(1, epoch_updates),
                    "kl_loss": epoch_kl_loss / max(1, epoch_updates),
                }
            )

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
    total_updates: int = 40,
    trajectories_per_update: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.1,
    learning_rate: float = 1e-4,
    ppo_epochs: int = 3,
    minibatch_size: int = 128,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    previous_policy_kl_coef: float = 0.0,
    reference_policy_state_dict: dict[str, torch.Tensor] | None = None,
    vectorized_rollouts: bool = True,
    enable_progress: bool = True,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    early_stopping_restore_best: bool = False,
    accuracy_stop_threshold: float | None = None,
    accuracy_eval_env: BTCDirectionEnv | None = None,
    accuracy_eval_threshold: float = 0.0,
) -> tuple[list[dict], dict[str, dict[str, torch.Tensor]], dict]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict] = []
    checkpoints: dict[str, dict[str, torch.Tensor]] = {
        "initial": clone_state_dict(policy),
    }
    bounded_updates = total_updates is not None and int(total_updates) > 0
    midpoint_update = max(1, int(total_updates) // 2) if bounded_updates else None
    progress = ProgressLogger(label="PPO", total_steps=int(total_updates) if bounded_updates else None) if enable_progress else None
    reference_policy: nn.Module | None = None
    if reference_policy_state_dict is not None and previous_policy_kl_coef > 0.0:
        reference_policy = build_policy(
            sequence_length=env.dataset_bundle.sequence_length,
            feature_dim=env.dataset_bundle.feature_dim,
            action_dim=env.action_dim,
        ).to(device)
        reference_policy.load_state_dict(reference_policy_state_dict)
        reference_policy.eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)
    best_mean_return = float("-inf")
    best_update = 0
    best_state_dict = clone_state_dict(policy)
    no_improvement_count = 0
    early_stopped = False
    accuracy_stopped = False

    update_idx = 0
    while True:
        update_idx += 1
        collector = collect_vectorized_rollouts if vectorized_rollouts else collect_sequential_rollouts
        rollout_batches, trajectory_metrics = collector(
            env=env,
            num_envs=trajectories_per_update,
            policy=policy,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        batch = RolloutBatch(
            observations=torch.cat([rb.observations for rb in rollout_batches], dim=0),
            actions=torch.cat([rb.actions for rb in rollout_batches], dim=0),
            log_probs=torch.cat([rb.log_probs for rb in rollout_batches], dim=0),
            rewards=torch.cat([rb.rewards for rb in rollout_batches], dim=0),
            dones=torch.cat([rb.dones for rb in rollout_batches], dim=0),
            values=torch.cat([rb.values for rb in rollout_batches], dim=0),
            returns=torch.cat([rb.returns for rb in rollout_batches], dim=0),
            advantages=torch.cat([rb.advantages for rb in rollout_batches], dim=0),
        )

        losses = ppo_update(
            policy=policy,
            optimizer=optimizer,
            batch=batch,
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            epochs=ppo_epochs,
            minibatch_size=min(minibatch_size, batch.observations.shape[0]),
            reference_policy=reference_policy,
            previous_policy_kl_coef=previous_policy_kl_coef,
            epoch_callback=(
                (lambda epoch_metrics: progress.log(
                    update_idx,
                    (
                        f"update={update_idx}, epoch={epoch_metrics['epoch']}/{epoch_metrics['epochs']}, "
                        f"policy_loss={epoch_metrics['policy_loss']:.4f}, "
                        f"value_loss={epoch_metrics['value_loss']:.4f}, "
                        f"kl_loss={epoch_metrics['kl_loss']:.4f}"
                    ),
                ))
                if progress is not None else None
            ),
        )
        mean_rewards = [metric["mean_reward"] for metric in trajectory_metrics]
        mean_returns = [metric["trajectory_return"] for metric in trajectory_metrics]
        history.append(
            {
                "update": update_idx,
                "step": update_idx,
                "mean_reward": float(np.mean(mean_rewards)),
                "mean_return": float(np.mean(mean_returns)),
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "kl_loss": losses["kl_loss"],
            }
        )
        if accuracy_eval_env is not None:
            accuracy_eval_result = evaluate_policy(
                accuracy_eval_env,
                policy,
                device=device,
                ternary_confidence_threshold=accuracy_eval_threshold,
            )
            history[-1]["train_accuracy"] = float(accuracy_eval_result.get("accuracy", 0.0))
        current_mean_return = history[-1]["mean_return"]
        improved = current_mean_return > (best_mean_return + early_stopping_min_delta)
        if improved:
            best_mean_return = current_mean_return
            best_update = update_idx
            best_state_dict = clone_state_dict(policy)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if progress is not None:
            progress.log(
                update_idx,
                (
                    f"mean_reward={history[-1]['mean_reward']:.4f}, "
                    f"mean_return={history[-1]['mean_return']:.4f}, "
                    f"train_accuracy={history[-1].get('train_accuracy', 0.0):.4f}, "
                    f"policy_loss={history[-1]['policy_loss']:.4f}, "
                    f"value_loss={history[-1]['value_loss']:.4f}, "
                    f"kl_loss={history[-1]['kl_loss']:.4f}"
                ),
            )

        if midpoint_update is not None and update_idx == midpoint_update:
            checkpoints["midpoint"] = clone_state_dict(policy)
        if (
            accuracy_stop_threshold is not None
            and history[-1].get("train_accuracy") is not None
            and float(history[-1]["train_accuracy"]) >= float(accuracy_stop_threshold)
        ):
            accuracy_stopped = True
            best_state_dict = clone_state_dict(policy)
            best_update = update_idx
            best_mean_return = max(best_mean_return, current_mean_return)
            break
        if bounded_updates and update_idx >= int(total_updates):
            break
        if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
            early_stopped = True
            if early_stopping_restore_best:
                policy.load_state_dict(best_state_dict)
            break

    checkpoints["final"] = clone_state_dict(policy)
    if "midpoint" not in checkpoints:
        checkpoints["midpoint"] = clone_state_dict(policy)

    plot_config = {
        "title": "PPO Training Mean Rewards Over Trajectories",
        "series": [
            {"key": "mean_reward", "label": "Mean reward / trajectory", "color": "#1f77b4"},
            {"key": "mean_return", "label": "Mean return / trajectory", "color": "#ff7f0e", "alpha": 0.8},
        ],
        "early_stopping": {
            "enabled": bool(early_stopping_patience > 0),
            "triggered": bool(early_stopped),
            "patience": int(early_stopping_patience),
            "min_delta": float(early_stopping_min_delta),
            "restore_best": bool(early_stopping_restore_best),
            "best_update": int(best_update),
            "best_mean_return": float(best_mean_return if best_mean_return != float("-inf") else 0.0),
            "stopped_update": int(history[-1]["update"]) if history else 0,
        },
        "accuracy_stopping": {
            "enabled": bool(accuracy_stop_threshold is not None),
            "triggered": bool(accuracy_stopped),
            "threshold": float(accuracy_stop_threshold) if accuracy_stop_threshold is not None else None,
            "stopped_update": int(history[-1]["update"]) if history else 0,
            "stopped_accuracy": float(history[-1].get("train_accuracy", 0.0)) if history else 0.0,
        },
    }
    return history, checkpoints, plot_config
