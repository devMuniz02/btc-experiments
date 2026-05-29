from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.btc_direction_learning.dataset import DirectionDatasetBundle

ENV_VERSION_BINARY = "binary"
ENV_VERSION_TERNARY = "ternary"
ENV_VERSION_INTENSITY11 = "intensity11"
NONE_ACTION = 2
INTENSITY11_NONE_ACTION = 5
DOWN_DIRECTION = "down"
UP_DIRECTION = "up"
NONE_DIRECTION = "none"


@dataclass
class StepRecord:
    global_index: int
    action: int
    label: int
    reward: float


def none_action_for_env(env_version: str) -> int | None:
    if env_version == ENV_VERSION_TERNARY:
        return NONE_ACTION
    if env_version == ENV_VERSION_INTENSITY11:
        return INTENSITY11_NONE_ACTION
    return None


def is_none_action(action: int, env_version: str) -> bool:
    none_action = none_action_for_env(env_version)
    return none_action is not None and int(action) == none_action


def action_direction(action: int, env_version: str) -> str:
    action_int = int(action)
    if env_version == ENV_VERSION_BINARY:
        if action_int == 0:
            return DOWN_DIRECTION
        if action_int == 1:
            return UP_DIRECTION
    elif env_version == ENV_VERSION_TERNARY:
        if action_int == 0:
            return DOWN_DIRECTION
        if action_int == 1:
            return UP_DIRECTION
        if action_int == NONE_ACTION:
            return NONE_DIRECTION
    elif env_version == ENV_VERSION_INTENSITY11:
        if 0 <= action_int <= 4:
            return DOWN_DIRECTION
        if action_int == INTENSITY11_NONE_ACTION:
            return NONE_DIRECTION
        if 6 <= action_int <= 10:
            return UP_DIRECTION
    raise ValueError(f"Unsupported action {action_int} for env version {env_version}.")


def action_direction_label(action: int, env_version: str) -> int | None:
    direction = action_direction(action, env_version)
    if direction == DOWN_DIRECTION:
        return 0
    if direction == UP_DIRECTION:
        return 1
    return None


def action_intensity(action: int, env_version: str) -> int:
    action_int = int(action)
    if env_version in {ENV_VERSION_BINARY, ENV_VERSION_TERNARY}:
        return 0 if is_none_action(action_int, env_version) else 1
    if env_version == ENV_VERSION_INTENSITY11:
        if 0 <= action_int <= 4:
            return action_int + 1
        if action_int == INTENSITY11_NONE_ACTION:
            return 0
        if 6 <= action_int <= 10:
            return action_int - 5
    raise ValueError(f"Unsupported action {action_int} for env version {env_version}.")


def expert_action_for_label(label: int, env_version: str) -> int:
    label_int = int(label)
    if env_version == ENV_VERSION_TERNARY and label_int == NONE_ACTION:
        return NONE_ACTION
    if env_version == ENV_VERSION_INTENSITY11 and label_int == INTENSITY11_NONE_ACTION:
        return INTENSITY11_NONE_ACTION
    if label_int not in (0, 1):
        raise ValueError(f"Unsupported label: {label_int}")
    if env_version == ENV_VERSION_INTENSITY11:
        return 0 if label_int == 0 else 6
    return label_int


def compute_action_reward(action: int, label: int, env_version: str) -> float:
    if is_none_action(action, env_version):
        return 0.0
    reward_scale = float(action_intensity(action, env_version))
    return reward_scale if action_direction_label(action, env_version) == int(label) else -reward_scale


def action_dim_for_env(env_version: str) -> int:
    if env_version == ENV_VERSION_BINARY:
        return 2
    if env_version == ENV_VERSION_TERNARY:
        return 3
    if env_version == ENV_VERSION_INTENSITY11:
        return 11
    raise ValueError(f"Unknown env version: {env_version}")


def baseline_down_action(env_version: str) -> int:
    return expert_action_for_label(0, env_version)


def baseline_up_action(env_version: str) -> int:
    return expert_action_for_label(1, env_version)


def intensity11_action_from_probability(probability_up: float) -> int:
    probability = max(0.0, min(1.0, float(probability_up)))
    if probability < 0.10:
        return 4
    if probability < 0.20:
        return 3
    if probability < 0.30:
        return 2
    if probability < 0.40:
        return 1
    if probability < 0.45:
        return 0
    if probability < 0.55:
        return INTENSITY11_NONE_ACTION
    if probability < 0.60:
        return 6
    if probability < 0.70:
        return 7
    if probability < 0.80:
        return 8
    if probability < 0.90:
        return 9
    return 10


def delta_magnitude_to_intensity(delta_magnitude: float) -> int:
    magnitude = abs(float(delta_magnitude))
    if magnitude < 50.0:
        return 0
    if magnitude < 100.0:
        return 1
    if magnitude < 150.0:
        return 2
    if magnitude < 200.0:
        return 3
    if magnitude < 250.0:
        return 4
    return 5


def intensity11_action_from_delta(delta: float) -> int:
    intensity = delta_magnitude_to_intensity(delta)
    if intensity == 0:
        return INTENSITY11_NONE_ACTION
    if float(delta) < 0.0:
        return 5 - intensity
    return 5 + intensity


class BTCDirectionEnv:
    def __init__(
        self,
        dataset_bundle: DirectionDatasetBundle,
        split_name: str,
        env_version: str = ENV_VERSION_BINARY,
    ) -> None:
        self.dataset_bundle = dataset_bundle
        self.split_name = split_name
        self.env_version = env_version
        if split_name == "train":
            self.decision_indices = dataset_bundle.train_decision_indices.astype(np.int64)
        elif split_name == "test":
            self.decision_indices = dataset_bundle.test_decision_indices.astype(np.int64)
        else:
            raise ValueError(f"Unknown split name: {split_name}")
        if env_version not in {ENV_VERSION_BINARY, ENV_VERSION_TERNARY, ENV_VERSION_INTENSITY11}:
            raise ValueError(f"Unknown env version: {env_version}")

        if len(self.decision_indices) == 0:
            raise RuntimeError("BTCDirectionEnv needs at least one decision index.")

        self._cursor = 0
        self._trajectory: list[StepRecord] = []

    def reset(self) -> np.ndarray:
        self._cursor = 0
        self._trajectory = []
        return self.dataset_bundle.build_observation(int(self.decision_indices[self._cursor]))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        global_index = int(self.decision_indices[self._cursor])
        label = int(self.dataset_bundle.labels[global_index])
        action_int = int(action)
        reward = self._compute_reward(action_int, label)
        self._trajectory.append(
            StepRecord(
                global_index=global_index,
                action=action_int,
                label=label,
                reward=reward,
            )
        )

        self._cursor += 1
        done = self._cursor >= len(self.decision_indices)
        if done:
            next_observation = np.zeros(
                (self.dataset_bundle.sequence_length, self.dataset_bundle.feature_dim),
                dtype=np.float32,
            )
        else:
            next_observation = self.dataset_bundle.build_observation(int(self.decision_indices[self._cursor]))

        info = {
            "global_index": global_index,
            "label": label,
            "trajectory_length": len(self._trajectory),
            "split_name": self.split_name,
            "env_version": self.env_version,
            "expert_action": self._expert_action_for_label(label),
        }
        return next_observation, reward, done, info

    def collect_expert_demonstrations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations, expert_actions, decision_indices = self.dataset_bundle.get_split_data(self.split_name)
        return observations, expert_actions, decision_indices

    def _expert_action_for_label(self, label: int) -> int:
        return expert_action_for_label(label, self.env_version)

    def _compute_reward(self, action: int, label: int) -> float:
        return compute_action_reward(action, label, self.env_version)

    @property
    def horizon(self) -> int:
        return len(self.decision_indices)

    @property
    def trajectory(self) -> list[StepRecord]:
        return list(self._trajectory)

    @property
    def action_dim(self) -> int:
        return action_dim_for_env(self.env_version)
