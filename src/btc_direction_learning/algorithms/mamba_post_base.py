from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.distributions import Categorical

from src.btc_direction_learning.continuation_models import MambaBanditPolicy, MambaContinuationActorCriticPolicy
from src.btc_direction_learning.dataset import DirectionDatasetBundle
from src.btc_direction_learning.evaluation import clone_state_dict
from src.btc_direction_learning.models import extract_policy_logits, select_actions_from_logits


@dataclass
class PostBaseBatch:
    observations: torch.Tensor
    labels: torch.Tensor
    features: torch.Tensor | None = None


def _reward_tensor(actions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rewards = torch.zeros_like(actions, dtype=torch.float32)
    directional_mask = actions != 2
    rewards = torch.where(directional_mask, torch.full_like(rewards, -1.0), rewards)
    correct_mask = directional_mask & (actions == labels)
    rewards = torch.where(correct_mask, torch.ones_like(rewards), rewards)
    return rewards


def _classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = select_actions_from_logits(logits, env_version="ternary", ternary_confidence_threshold=0.0)
    return float((predictions == labels).float().mean().item())


def _batched_indices(sample_count: int, batch_size: int, device: torch.device):
    permutation = torch.randperm(sample_count, device=device)
    for start in range(0, sample_count, batch_size):
        yield permutation[start : start + batch_size]


def _prepare_batch(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    *,
    preload_to_device: bool = True,
) -> PostBaseBatch:
    observations, labels, _ = bundle.get_split_data("train")
    if preload_to_device and device.type == "cuda":
        return PostBaseBatch(
            observations=torch.tensor(observations, dtype=torch.float32, device=device),
            labels=torch.tensor(labels, dtype=torch.long, device=device),
        )
    return PostBaseBatch(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        labels=torch.tensor(labels, dtype=torch.long, device=device),
    )


def _maybe_cache_features(
    policy: MambaContinuationActorCriticPolicy | MambaBanditPolicy,
    batch: PostBaseBatch,
) -> PostBaseBatch:
    with torch.no_grad():
        features = policy.extract_features(batch.observations)
    return PostBaseBatch(observations=batch.observations, labels=batch.labels, features=features)


def _ppo_like_train(
    *,
    bundle: DirectionDatasetBundle,
    policy: MambaContinuationActorCriticPolicy,
    device: torch.device,
    total_updates: int,
    learning_rate: float,
    clip_epsilon: float,
    entropy_coef: float,
    value_loss_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    variant_label: str,
    freeze_backbone: bool = False,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    if freeze_backbone:
        policy.freeze_backbone()
        cached_batch = _maybe_cache_features(policy, _prepare_batch(bundle, device))
    else:
        policy.unfreeze_all()
        cached_batch = _prepare_batch(bundle, device)

    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    history: list[dict[str, Any]] = []
    checkpoints = {"initial": clone_state_dict(policy)}
    sample_count = int(cached_batch.labels.shape[0])

    for update_idx in range(1, total_updates + 1):
        policy.train()
        with torch.no_grad():
            if cached_batch.features is not None:
                rollout_logits, rollout_values = policy.forward_from_features(cached_batch.features)
            else:
                rollout_logits, rollout_values = policy(cached_batch.observations)
            rollout_dist = Categorical(logits=rollout_logits)
            rollout_actions = rollout_dist.sample()
            old_log_probs = rollout_dist.log_prob(rollout_actions)
            rewards = _reward_tensor(rollout_actions, cached_batch.labels)
            advantages = rewards - rollout_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))

        total_loss = 0.0
        update_steps = 0
        for _ in range(ppo_epochs):
            for idx in _batched_indices(sample_count, minibatch_size, device):
                batch_labels = cached_batch.labels.index_select(0, idx)
                batch_actions = rollout_actions.index_select(0, idx)
                batch_old_log_probs = old_log_probs.index_select(0, idx)
                batch_advantages = advantages.index_select(0, idx)
                batch_rewards = rewards.index_select(0, idx)
                if cached_batch.features is not None:
                    batch_features = cached_batch.features.index_select(0, idx)
                    logits, values = policy.forward_from_features(batch_features)
                else:
                    batch_obs = cached_batch.observations.index_select(0, idx)
                    logits, values = policy(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, batch_rewards)
                loss = policy_loss + (value_loss_coef * value_loss) - (entropy_coef * entropy)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                update_steps += 1

        policy.eval()
        with torch.no_grad():
            if cached_batch.features is not None:
                eval_logits, eval_values = policy.forward_from_features(cached_batch.features)
            else:
                eval_logits, eval_values = policy(cached_batch.observations)
        epoch_metrics = {
            "epoch": update_idx,
            "step": update_idx,
            "loss": total_loss / max(1, update_steps),
            "train_accuracy": _classification_accuracy(eval_logits, cached_batch.labels),
            "mean_reward": float(_reward_tensor(torch.argmax(eval_logits, dim=-1), cached_batch.labels).mean().item()),
            "mean_value": float(eval_values.mean().item()),
            "label": variant_label,
        }
        history.append(epoch_metrics)
        if progress_callback is not None:
            callback_metrics = dict(epoch_metrics)
            callback_metrics["_policy"] = policy
            progress_callback(callback_metrics, update_idx, total_updates)

    checkpoints["final"] = clone_state_dict(policy)
    metadata = {"cached_features": bool(cached_batch.features is not None)}
    return history, checkpoints, metadata


def train_actor_critic(
    *,
    bundle: DirectionDatasetBundle,
    policy: MambaContinuationActorCriticPolicy,
    device: torch.device,
    total_updates: int = 12,
    learning_rate: float = 2e-4,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    minibatch_size: int = 2048,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    batch = _prepare_batch(bundle, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    checkpoints = {"initial": clone_state_dict(policy)}
    sample_count = int(batch.labels.shape[0])

    for update_idx in range(1, total_updates + 1):
        total_loss = 0.0
        steps = 0
        for idx in _batched_indices(sample_count, minibatch_size, device):
            obs = batch.observations.index_select(0, idx)
            labels = batch.labels.index_select(0, idx)
            logits, values = policy(obs)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            rewards = _reward_tensor(actions, labels)
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))
            policy_loss = -(dist.log_prob(actions) * advantages).mean()
            value_loss = torch.nn.functional.mse_loss(values, rewards)
            entropy = dist.entropy().mean()
            loss = policy_loss + (value_loss_coef * value_loss) - (entropy_coef * entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        with torch.no_grad():
            eval_logits, _ = policy(batch.observations)
        metrics = {
            "epoch": update_idx,
            "step": update_idx,
            "loss": total_loss / max(1, steps),
            "train_accuracy": _classification_accuracy(eval_logits, batch.labels),
            "mean_reward": float(_reward_tensor(torch.argmax(eval_logits, dim=-1), batch.labels).mean().item()),
            "label": "ACTOR_CRITIC",
        }
        history.append(metrics)
        if progress_callback is not None:
            callback_metrics = dict(metrics)
            callback_metrics["_policy"] = policy
            progress_callback(callback_metrics, update_idx, total_updates)

    checkpoints["final"] = clone_state_dict(policy)
    return history, checkpoints, {}


def train_ppo_full(
    *,
    bundle: DirectionDatasetBundle,
    policy: MambaContinuationActorCriticPolicy,
    device: torch.device,
    total_updates: int = 12,
    learning_rate: float = 2e-4,
    clip_epsilon: float = 0.1,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    return _ppo_like_train(
        bundle=bundle,
        policy=policy,
        device=device,
        total_updates=total_updates,
        learning_rate=learning_rate,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        variant_label="PPO_FULL",
        freeze_backbone=False,
        progress_callback=progress_callback,
    )


def train_ppo_hybrid(
    *,
    bundle: DirectionDatasetBundle,
    policy: MambaContinuationActorCriticPolicy,
    device: torch.device,
    total_updates: int = 12,
    learning_rate: float = 3e-4,
    clip_epsilon: float = 0.1,
    entropy_coef: float = 0.005,
    value_loss_coef: float = 0.5,
    ppo_epochs: int = 6,
    minibatch_size: int = 4096,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    return _ppo_like_train(
        bundle=bundle,
        policy=policy,
        device=device,
        total_updates=total_updates,
        learning_rate=learning_rate,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        variant_label="PPO_HYBRID",
        freeze_backbone=True,
        progress_callback=progress_callback,
    )


def train_bandit(
    *,
    bundle: DirectionDatasetBundle,
    policy: MambaBanditPolicy,
    device: torch.device,
    strategy: str = "ts",
    ridge_lambda: float = 1.0,
    ucb_alpha: float = 1.0,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    batch = _maybe_cache_features(policy, _prepare_batch(bundle, device))
    features = batch.features
    if features is None:
        raise RuntimeError("Bandit training requires cached features.")
    labels = batch.labels
    policy.freeze_backbone()
    feature_dim = int(features.shape[1])
    identity = torch.eye(feature_dim, device=device, dtype=features.dtype) * float(ridge_lambda)
    weights = []
    covariance = []
    sampled_weights = []
    targets = torch.stack(
        (
            torch.where(labels == 0, torch.ones_like(labels, dtype=torch.float32), -torch.ones_like(labels, dtype=torch.float32)),
            torch.where(labels == 1, torch.ones_like(labels, dtype=torch.float32), -torch.ones_like(labels, dtype=torch.float32)),
            torch.zeros_like(labels, dtype=torch.float32),
        ),
        dim=-1,
    )
    for action_index in range(3):
        if action_index == 0:
            subset = features[labels == 0]
        elif action_index == 1:
            subset = features[labels == 1]
        else:
            subset = features
        if subset.shape[0] == 0:
            subset = features
        precision = identity + subset.T @ subset
        covariance_action = torch.linalg.pinv(precision)
        mean_weight = covariance_action @ (features.T @ targets[:, action_index])
        weights.append(mean_weight)
        covariance.append(covariance_action)
        if str(strategy).strip().lower() == "ts":
            diagonal_noise = torch.sqrt(torch.diag(covariance_action).clamp_min(1e-8))
            sampled_weights.append(mean_weight + (torch.randn_like(mean_weight) * diagonal_noise))
        else:
            sampled_weights.append(mean_weight)
    weight_tensor = torch.stack(weights, dim=-1)
    covariance_tensor = torch.stack(covariance, dim=0)
    sampled_tensor = torch.stack(sampled_weights, dim=-1)
    policy.bandit_strategy = str(strategy).strip().lower()
    policy.ucb_alpha = float(ucb_alpha)
    policy.set_posterior(
        weights=weight_tensor,
        bias=torch.zeros(3, device=device, dtype=features.dtype),
        covariance=covariance_tensor,
        sampled_weights=sampled_tensor,
    )
    with torch.no_grad():
        logits = policy.forward(batch.observations)
    metrics = {
        "epoch": 1,
        "step": 1,
        "loss": 0.0,
        "train_accuracy": _classification_accuracy(logits, labels),
        "mean_reward": float(_reward_tensor(torch.argmax(logits, dim=-1), labels).mean().item()),
        "label": f"BANDITS_{policy.bandit_strategy.upper()}",
    }
    if progress_callback is not None:
        callback_metrics = dict(metrics)
        callback_metrics["_policy"] = policy
        progress_callback(callback_metrics, 1, 1)
    return [metrics], {"initial": clone_state_dict(policy), "final": clone_state_dict(policy)}, {"cached_features": True}


__all__ = [
    "train_actor_critic",
    "train_bandit",
    "train_ppo_full",
    "train_ppo_hybrid",
]
