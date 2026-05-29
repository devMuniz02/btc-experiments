from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from src.btc_direction_learning.env import (
    BTCDirectionEnv,
    ENV_VERSION_INTENSITY11,
    ENV_VERSION_TERNARY,
    NONE_ACTION,
    action_direction_label,
    action_intensity,
    compute_action_reward,
    is_none_action,
    none_action_for_env,
)
from src.btc_direction_learning.models import extract_policy_logits, select_actions_from_logits
from src.utils.market_hours import is_allowed_prediction_target_timestamp


def clone_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}


def load_checkpoint(policy: nn.Module, state_dict: dict[str, torch.Tensor]) -> nn.Module:
    policy.load_state_dict(copy.deepcopy(state_dict))
    return policy


def chosen_action_probability_from_logits(
    logits: torch.Tensor,
    *,
    action: int,
    env_version: str,
) -> float | None:
    if logits.ndim == 0:
        return None
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[0] < 1:
        return None

    if env_version == ENV_VERSION_INTENSITY11 and logits.shape[-1] == 2:
        directional_probs = torch.softmax(logits[0], dim=-1)
        probability_up = float(directional_probs[1].item())
        action_index = int(action)
        if is_none_action(action_index, env_version):
            return None
        directional_probability = probability_up if action_direction_label(action_index, env_version) == 1 else (1.0 - probability_up)
        return max(0.0, min(1.0, directional_probability))

    if env_version == ENV_VERSION_TERNARY and logits.shape[-1] == 2:
        directional_probs = torch.softmax(logits[0], dim=-1)
        probability_up = float(directional_probs[1].item())
        action_index = int(action)
        if action_index == NONE_ACTION:
            return None
        directional_probability = probability_up if action_index == 1 else (1.0 - probability_up)
        return max(0.0, min(1.0, directional_probability))

    action_index = int(action)
    if action_index < 0 or action_index >= logits.shape[-1]:
        return None
    action_probs = torch.softmax(logits, dim=-1)
    probability = action_probs[0, action_index].item()

    numeric_probability = float(probability)
    if not np.isfinite(numeric_probability):
        return None
    return max(0.0, min(1.0, numeric_probability))


def chosen_action_probabilities_from_logits(
    logits: torch.Tensor,
    *,
    actions: torch.Tensor,
    env_version: str,
) -> list[float | None]:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if actions.ndim == 0:
        actions = actions.unsqueeze(0)

    if env_version == ENV_VERSION_INTENSITY11 and logits.shape[-1] == 2:
        directional_probs = torch.softmax(logits, dim=-1)[:, 1]
        probabilities: list[float | None] = []
        for probability_up, action in zip(directional_probs.detach().cpu().tolist(), actions.detach().cpu().tolist()):
            action_index = int(action)
            if is_none_action(action_index, env_version):
                probabilities.append(None)
                continue
            directional_probability = probability_up if action_direction_label(action_index, env_version) == 1 else (1.0 - probability_up)
            probabilities.append(max(0.0, min(1.0, float(directional_probability))))
        return probabilities

    if env_version == ENV_VERSION_TERNARY and logits.shape[-1] == 2:
        directional_probs = torch.softmax(logits, dim=-1)[:, 1]
        probabilities = []
        for probability_up, action in zip(directional_probs.detach().cpu().tolist(), actions.detach().cpu().tolist()):
            action_index = int(action)
            if action_index == NONE_ACTION:
                probabilities.append(None)
                continue
            directional_probability = probability_up if action_index == 1 else (1.0 - probability_up)
            probabilities.append(max(0.0, min(1.0, float(directional_probability))))
        return probabilities

    action_probs = torch.softmax(logits, dim=-1)
    probabilities = []
    for row_probs, action in zip(action_probs.detach().cpu(), actions.detach().cpu().tolist()):
        action_index = int(action)
        if action_index < 0 or action_index >= row_probs.shape[0]:
            probabilities.append(None)
            continue
        numeric_probability = float(row_probs[action_index].item())
        if not np.isfinite(numeric_probability):
            probabilities.append(None)
            continue
        probabilities.append(max(0.0, min(1.0, numeric_probability)))
    return probabilities


def _scored_action_mask(actions: np.ndarray, env_version: str) -> np.ndarray:
    return np.asarray([not is_none_action(int(action), env_version) for action in actions], dtype=bool)


def _directional_accuracy(actions: list[int], labels: list[int], env_version: str) -> tuple[float, int]:
    if not labels:
        return 0.0, 0
    action_arr = np.asarray(actions, dtype=np.int64)
    label_arr = np.asarray(labels, dtype=np.int64)
    scored_mask = _scored_action_mask(action_arr, env_version)
    scored_count = int(scored_mask.sum())
    if scored_count == 0:
        return 0.0, 0
    scored_actions = action_arr[scored_mask]
    scored_labels = label_arr[scored_mask]
    scored_directions = np.asarray(
        [action_direction_label(int(action), env_version) for action in scored_actions],
        dtype=np.int64,
    )
    return float(np.mean(scored_directions == scored_labels)), scored_count


def _batched_policy_outputs(
    *,
    observations: np.ndarray,
    policy: nn.Module,
    device: torch.device,
    env_version: str,
    ternary_confidence_threshold: float,
    batch_size: int = 512,
) -> tuple[list[int], list[float | None]]:
    actions: list[int] = []
    chosen_action_probabilities: list[float | None] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.tensor(observations[start:start + batch_size], dtype=torch.float32, device=device)
            logits = extract_policy_logits(policy, batch)
            batch_actions = select_actions_from_logits(
                logits,
                env_version=env_version,
                ternary_confidence_threshold=ternary_confidence_threshold,
            )
            actions.extend(int(value) for value in batch_actions.detach().cpu().tolist())
            chosen_action_probabilities.extend(
                chosen_action_probabilities_from_logits(
                    logits,
                    actions=batch_actions,
                    env_version=env_version,
                )
            )
    return actions, chosen_action_probabilities


def evaluate_policy(
    env: BTCDirectionEnv,
    policy: nn.Module,
    device: torch.device,
    ternary_confidence_threshold: float = 0.0,
) -> dict:
    was_training = policy.training
    policy.eval()
    observations, labels_array, _ = env.dataset_bundle.get_split_data(env.split_name)
    labels = [int(value) for value in labels_array.tolist()]
    actions, chosen_action_probabilities = _batched_policy_outputs(
        observations=observations,
        policy=policy,
        device=device,
        env_version=env.env_version,
        ternary_confidence_threshold=ternary_confidence_threshold,
    )
    result = summarize_actions_against_labels(
        actions,
        labels,
        env.env_version,
        chosen_action_probabilities=chosen_action_probabilities,
    )
    if was_training:
        policy.train()
    return result


def evaluate_policy_predictions(
    *,
    observations: np.ndarray,
    labels: list[int] | np.ndarray,
    policy: nn.Module,
    device: torch.device,
    action_mapper: Callable[[float], int],
) -> dict:
    was_training = policy.training
    policy.eval()
    actions: list[int] = []
    predictions: list[float] = []
    label_list = [int(value) for value in np.asarray(labels, dtype=np.int64).tolist()]

    with torch.no_grad():
        for observation in observations:
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            prediction = float(policy(obs_tensor).reshape(-1)[0].item())
            predictions.append(prediction)
            actions.append(int(action_mapper(prediction)))

    result = summarize_actions_against_labels(actions, label_list, ENV_VERSION_INTENSITY11)
    result["predictions"] = predictions
    if was_training:
        policy.train()
    return result


def evaluate_policy_with_market_hours_none(
    env: BTCDirectionEnv,
    policy: nn.Module,
    device: torch.device,
    timestamps: list[str],
    ternary_confidence_threshold: float = 0.0,
) -> dict:
    if env.env_version != ENV_VERSION_TERNARY:
        raise ValueError("Market-hours-derived evaluation requires the ternary env.")

    was_training = policy.training
    policy.eval()
    observations, labels_array, _ = env.dataset_bundle.get_split_data(env.split_name)
    labels = [int(value) for value in labels_array.tolist()]
    actions, chosen_action_probabilities = _batched_policy_outputs(
        observations=observations,
        policy=policy,
        device=device,
        env_version=env.env_version,
        ternary_confidence_threshold=ternary_confidence_threshold,
    )
    none_action = none_action_for_env(env.env_version)
    if none_action is None:
        raise ValueError(f"Env version {env.env_version} does not support none-action gating.")
    for index, timestamp in enumerate(timestamps[: len(actions)]):
        if not is_allowed_prediction_target_timestamp(timestamp):
            actions[index] = none_action
            chosen_action_probabilities[index] = None
    result = summarize_actions_against_labels(
        actions,
        labels,
        env.env_version,
        chosen_action_probabilities=chosen_action_probabilities,
    )
    if was_training:
        policy.train()
    return result


def summarize_actions_against_labels(
    actions: list[int],
    labels: list[int],
    env_version: str,
    chosen_action_probabilities: list[float | None] | None = None,
) -> dict:
    rewards: list[float] = []
    cumulative_rewards = [0.0]
    scored_count = 0
    correct_count = 0
    normalized_probabilities = list(chosen_action_probabilities or [])
    if len(normalized_probabilities) < len(actions):
        normalized_probabilities.extend([None] * (len(actions) - len(normalized_probabilities)))
    elif len(normalized_probabilities) > len(actions):
        normalized_probabilities = normalized_probabilities[: len(actions)]

    for action, label in zip(actions, labels):
        reward = compute_action_reward(int(action), int(label), env_version)
        if not is_none_action(int(action), env_version):
            scored_count += 1
            if action_direction_label(int(action), env_version) == int(label):
                correct_count += 1
        rewards.append(float(reward))
        cumulative_rewards.append(cumulative_rewards[-1] + float(reward))

    accuracy = float(correct_count / scored_count) if scored_count > 0 else 0.0
    return {
        "rewards": rewards,
        "actions": list(actions),
        "labels": list(labels),
        "chosen_action_probabilities": normalized_probabilities,
        "cumulative_rewards": cumulative_rewards,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "accuracy": accuracy,
        "accuracy_scored_count": int(scored_count),
    }


def simulate_portfolio(
    actions: list[int],
    labels: list[int],
    env_version: str,
    starting_balance: float = 100.0,
    move_fraction: float = 0.05,
    sizing_mode: str = "peak_fraction",
) -> dict:
    balances = [float(starting_balance)]
    peak_balance = float(starting_balance)
    trade_pnls: list[float] = []
    max_drawdown_pct = 0.0
    max_drawdown_peak_index = 0
    max_drawdown_trough_index = 0
    max_drawdown_segment_loss_score = 0
    longest_win_streak = 0
    longest_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    full_loss_count = 0
    in_zero_zone = starting_balance <= 0.0
    peak_index = 0

    def compute_trade_pnl(balance: float, peak: float, action: int, label: int) -> float:
        action_int = int(action)
        if is_none_action(action_int, env_version):
            return 0.0
        intensity = float(action_intensity(action_int, env_version))
        if env_version == ENV_VERSION_INTENSITY11:
            if sizing_mode == "fixed_dollar":
                pnl = intensity
            elif sizing_mode == "current_fraction":
                pnl = balance * (intensity / 100.0)
            elif sizing_mode == "peak_fraction":
                pnl = peak * (intensity / 100.0)
            else:
                raise ValueError(f"Unknown sizing mode: {sizing_mode}")
        else:
            if sizing_mode == "fixed_dollar":
                pnl = 1.0
            elif sizing_mode == "current_fraction":
                pnl = max(1.0, move_fraction * balance)
            elif sizing_mode == "peak_fraction":
                pnl = max(1.0, move_fraction * peak)
            else:
                raise ValueError(f"Unknown sizing mode: {sizing_mode}")
        if action_direction_label(action_int, env_version) != int(label):
            pnl *= -1.0
        return float(pnl)

    for step_index, (action, label) in enumerate(zip(actions, labels), start=1):
        balance = balances[-1]
        pnl = compute_trade_pnl(balance, peak_balance, action, label)

        next_balance = balance + pnl
        balances.append(float(next_balance))
        if float(next_balance) > peak_balance:
            peak_balance = float(next_balance)
            peak_index = step_index
        trade_pnls.append(float(pnl))
        if next_balance <= 0.0:
            if not in_zero_zone:
                full_loss_count += 1
                in_zero_zone = True
        else:
            in_zero_zone = False
        if peak_balance > 0.0:
            drawdown_pct = max(0.0, (peak_balance - float(next_balance)) / peak_balance)
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
                max_drawdown_peak_index = peak_index
                max_drawdown_trough_index = step_index

        if pnl > 0.0:
            current_win_streak += 1
            current_loss_streak = 0
        elif pnl < 0.0:
            current_loss_streak += 1
            current_win_streak = 0
        else:
            current_win_streak = 0
            current_loss_streak = 0

        longest_win_streak = max(longest_win_streak, current_win_streak)
        longest_loss_streak = max(longest_loss_streak, current_loss_streak)

    if max_drawdown_trough_index > max_drawdown_peak_index:
        segment_actions = actions[max_drawdown_peak_index:max_drawdown_trough_index]
        segment_labels = labels[max_drawdown_peak_index:max_drawdown_trough_index]
        loss_count = 0
        win_count = 0
        for action, label in zip(segment_actions, segment_labels):
            pnl = compute_trade_pnl(float(starting_balance), float(starting_balance), action, label)
            if pnl < 0.0:
                loss_count += 1
            elif pnl > 0.0:
                win_count += 1
        max_drawdown_segment_loss_score = int(loss_count - win_count)

    return {
        "starting_balance": float(starting_balance),
        "balances": balances,
        "trade_pnls": trade_pnls,
        "final_balance": float(balances[-1]),
        "pnl": float(balances[-1] - starting_balance),
        "pnl_pct": float((balances[-1] - starting_balance) / starting_balance) if starting_balance != 0 else 0.0,
        "max_balance": float(max(balances)),
        "min_balance": float(min(balances)),
        "max_drawdown_pct": float(max_drawdown_pct),
        "max_drawdown_peak_index": int(max_drawdown_peak_index),
        "max_drawdown_trough_index": int(max_drawdown_trough_index),
        "max_drawdown_segment_loss_score": int(max_drawdown_segment_loss_score),
        "longest_win_streak": int(longest_win_streak),
        "longest_loss_streak": int(longest_loss_streak),
        "full_loss_count": int(full_loss_count),
        "sizing_mode": sizing_mode,
    }
