from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import FEATURE_COLUMNS, TEMPORAL_FEATURE_COLUMNS, TEST_CANDLES, TRAIN_CANDLES

import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

MODEL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS]
SEQUENCE_LENGTH = 24
NONE_ACTION = 2


def release_torch_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


class LocalStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "LocalStandardScaler":
        matrix = np.asarray(values, dtype=np.float32)
        self.mean_ = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit.")
        return (np.asarray(values, dtype=np.float32) - self.mean_) / self.scale_


try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    StandardScaler = LocalStandardScaler


@dataclass
class DirectionDatasetBundle:
    frame: pd.DataFrame
    scaled_features: np.ndarray
    scaler: StandardScaler
    feature_columns: list[str]
    sequence_length: int
    labels: np.ndarray
    train_decision_indices: np.ndarray
    test_decision_indices: np.ndarray
    train_row_count: int
    test_row_count: int

    @property
    def feature_dim(self) -> int:
        return len(self.feature_columns)

    def build_observation(self, global_index: int) -> np.ndarray:
        start = int(global_index) - self.sequence_length + 1
        end = int(global_index) + 1
        return self.scaled_features[start:end].astype(np.float32, copy=True)

    def build_observation_batch(self, indices: np.ndarray) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64)
        offsets = np.arange(self.sequence_length, dtype=np.int64)
        start_indices = index_array - (self.sequence_length - 1)
        window_indices = start_indices[:, None] + offsets[None, :]
        return self.scaled_features[window_indices].astype(np.float32, copy=False)

    def get_split_data(self, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self.train_decision_indices if split_name == "train" else self.test_decision_indices
        return self.build_observation_batch(indices), self.labels[indices].astype(np.int64), indices


class BTCDirectionEnv:
    def __init__(
        self,
        dataset_bundle: DirectionDatasetBundle,
        split_name: str,
        action_dim: int = 2,
        reward_value: float = 1.0,
    ) -> None:
        self.dataset_bundle = dataset_bundle
        self.split_name = split_name
        self.action_dim = int(action_dim)
        self.reward_value = float(reward_value)
        self.decision_indices = (
            dataset_bundle.train_decision_indices if split_name == "train" else dataset_bundle.test_decision_indices
        )
        if len(self.decision_indices) == 0:
            raise RuntimeError("BTCDirectionEnv needs at least one decision index.")
        self._cursor = 0

    def reset(self) -> np.ndarray:
        self._cursor = 0
        return self.dataset_bundle.build_observation(int(self.decision_indices[self._cursor]))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        global_index = int(self.decision_indices[self._cursor])
        label = int(self.dataset_bundle.labels[global_index])
        action_int = int(action)
        if self.action_dim == 3 and action_int == NONE_ACTION:
            reward = 0.0
        else:
            reward = self.reward_value if action_int == label else -self.reward_value
        self._cursor += 1
        done = self._cursor >= len(self.decision_indices)
        if done:
            next_observation = np.zeros(
                (self.dataset_bundle.sequence_length, self.dataset_bundle.feature_dim),
                dtype=np.float32,
            )
        else:
            next_observation = self.dataset_bundle.build_observation(int(self.decision_indices[self._cursor]))
        return next_observation, reward, done, {"label": label, "expert_action": label, "global_index": global_index}


def require_torch() -> None:
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto"):
    require_torch()
    value = str(name or "auto").lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset_bundle(
    frame: pd.DataFrame,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    train_rows: int = TRAIN_CANDLES,
    test_rows: int = TEST_CANDLES,
) -> DirectionDatasetBundle:
    missing = [column for column in ["timestamp", "target", *MODEL_FEATURE_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"Clean dataset is missing required model columns: {missing}")
    if train_rows < sequence_length:
        raise ValueError(f"train_length must be at least sequence_length ({sequence_length}), found {train_rows}.")
    if test_rows <= 0:
        raise ValueError("test_length must be positive.")
    if train_rows > TRAIN_CANDLES:
        raise ValueError(f"train_length cannot exceed the canonical {TRAIN_CANDLES} training rows.")
    if len(frame) < TRAIN_CANDLES + test_rows:
        raise ValueError(f"Expected at least {TRAIN_CANDLES + test_rows} rows, found {len(frame)}.")
    train_start = TRAIN_CANDLES - int(train_rows)
    train_end = TRAIN_CANDLES
    test_end = TRAIN_CANDLES + int(test_rows)
    working = pd.concat(
        [frame.iloc[train_start:train_end], frame.iloc[TRAIN_CANDLES:test_end]],
        ignore_index=True,
    ).copy()
    scaler = StandardScaler().fit(working.iloc[:train_rows][MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    scaled_features = scaler.transform(working[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)).astype(np.float32)
    labels = working["target"].to_numpy(dtype=np.int64)
    train_indices = np.arange(sequence_length - 1, train_rows, dtype=np.int64)
    test_indices = np.arange(train_rows, train_rows + test_rows, dtype=np.int64)
    return DirectionDatasetBundle(
        frame=working,
        scaled_features=scaled_features,
        scaler=scaler,
        feature_columns=list(MODEL_FEATURE_COLUMNS),
        sequence_length=int(sequence_length),
        labels=labels,
        train_decision_indices=train_indices,
        test_decision_indices=test_indices,
        train_row_count=int(train_rows),
        test_row_count=int(test_rows),
    )


class SequenceBackbone(nn.Module):
    def __init__(self, sequence_length: int, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(sequence_length * feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, observations):
        return self.network(observations.reshape(observations.shape[0], -1))


class LSTMClassificationPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=int(num_layers),
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations):
        projected = self.input_proj(observations)
        outputs, _ = self.lstm(projected)
        return self.head(self.norm(outputs[:, -1, :]))


class TransformerClassificationPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim or hidden_dim * 4),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations):
        hidden = self.input_proj(observations) + self.position_embedding[:, : observations.shape[1], :]
        encoded = self.encoder(hidden)
        return self.head(self.norm(encoded[:, -1, :]))


class MambaBlock(nn.Module):
    def __init__(self, hidden_dim: int, conv_kernel_size: int = 4, dropout: float = 0.1) -> None:
        require_torch()
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel_size,
            groups=hidden_dim,
            padding=conv_kernel_size - 1,
        )
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_log = nn.Parameter(torch.linspace(-1.0, 0.0, hidden_dim))
        self.d = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        hidden = self.norm(inputs)
        x_branch, gate = self.in_proj(hidden).chunk(2, dim=-1)
        sequence_length = x_branch.shape[1]
        mixed = self.depthwise_conv(x_branch.transpose(1, 2))[:, :, :sequence_length].transpose(1, 2)
        mixed = torch.nn.functional.silu(mixed)
        mixed = self.dropout(mixed)
        delta = torch.nn.functional.softplus(self.dt_proj(mixed))
        b_term = torch.tanh(self.b_proj(mixed))
        c_term = torch.tanh(self.c_proj(mixed))
        a_term = -torch.exp(self.a_log).view(1, 1, -1)
        decay = torch.exp(a_term * delta).clamp_min(1e-12)
        input_term = delta * b_term * mixed
        prefix = torch.cumprod(decay, dim=1)
        state = prefix * torch.cumsum(input_term / prefix.clamp_min(1e-12), dim=1)
        stacked = c_term * state + self.d.view(1, 1, -1) * mixed
        return residual + self.dropout(self.out_proj(stacked * torch.sigmoid(gate)))


class MambaClassificationPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def extract_features(self, observations):
        hidden = self.input_dropout(self.input_proj(observations))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden[:, -1, :])

    def forward(self, observations):
        return self.head(self.extract_features(observations))


class ClassificationPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        self.backbone = SequenceBackbone(sequence_length, feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations):
        return self.policy_head(self.backbone(observations))


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        feature_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        self.backbone = SequenceBackbone(sequence_length, feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations):
        hidden = self.backbone(observations)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


class MambaContinuationActorCriticPolicy(MambaClassificationPolicy):
    def __init__(self, sequence_length: int, feature_dim: int, action_dim: int = 3, **kwargs: Any) -> None:
        super().__init__(sequence_length, feature_dim, action_dim=action_dim, **kwargs)
        hidden_dim = self.head.in_features
        self.policy_head = self.head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward_from_features(self, features):
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def forward(self, observations):
        return self.forward_from_features(self.extract_features(observations))

    def freeze_backbone(self) -> None:
        for module in (self.input_proj, self.input_dropout, self.layers, self.norm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)


class SklearnPolicyWrapper:
    def __init__(self, estimator: Any, action_dim: int = 2) -> None:
        self.estimator = estimator
        self.action_dim = int(action_dim)

    def predict_logits(self, observations: np.ndarray) -> np.ndarray:
        flat = observations.reshape(observations.shape[0], -1)
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(flat)
            if probabilities.shape[1] == 1:
                probabilities = np.column_stack([1.0 - probabilities[:, 0], probabilities[:, 0]])
            return np.log(np.clip(probabilities, 1e-8, 1.0))
        predictions = self.estimator.predict(flat)
        logits = np.zeros((len(predictions), self.action_dim), dtype=np.float32)
        logits[np.arange(len(predictions)), predictions.astype(int)] = 1.0
        return logits


def build_torch_policy(
    model_type: str,
    bundle: DirectionDatasetBundle,
    hyperparameters: dict[str, Any],
    action_dim: int = 2,
):
    require_torch()
    hidden_dim = int(hyperparameters.get("hidden_dim", 128 if model_type in {"lstm", "transformer"} else 256))
    if model_type == "lstm":
        return LSTMClassificationPolicy(
            bundle.sequence_length,
            bundle.feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=int(hyperparameters.get("num_layers", 2)),
            dropout=float(hyperparameters.get("dropout", 0.1)),
        )
    if model_type == "transformer":
        return TransformerClassificationPolicy(
            bundle.sequence_length,
            bundle.feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=int(hyperparameters.get("num_layers", 2)),
            num_heads=int(hyperparameters.get("num_heads", 4)),
            feedforward_dim=int(hyperparameters.get("feedforward_dim", hidden_dim * 4)),
            dropout=float(hyperparameters.get("dropout", 0.1)),
        )
    if model_type == "mamba":
        return MambaClassificationPolicy(
            bundle.sequence_length,
            bundle.feature_dim,
            action_dim=action_dim,
            hidden_dim=int(hyperparameters.get("hidden_dim", 192)),
            num_layers=int(hyperparameters.get("num_layers", 3)),
            dropout=float(hyperparameters.get("dropout", 0.1)),
        )
    if model_type in {"nn", "bc", "dagger"}:
        return ClassificationPolicy(
            bundle.sequence_length,
            bundle.feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout=float(hyperparameters.get("dropout", 0.1)),
        )
    if model_type in {"ppo", "ppo_continue", "actor_critic"}:
        return ActorCriticPolicy(
            bundle.sequence_length,
            bundle.feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout=float(hyperparameters.get("dropout", 0.1)),
        )
    if model_type == "mamba_post_base":
        return MambaContinuationActorCriticPolicy(
            bundle.sequence_length,
            bundle.feature_dim,
            action_dim=int(hyperparameters.get("action_dim", 3)),
            hidden_dim=int(hyperparameters.get("hidden_dim", 192)),
            num_layers=int(hyperparameters.get("num_layers", 3)),
            dropout=float(hyperparameters.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported torch model_type: {model_type}")


def make_loader(observations: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, pin_memory: bool):
    dataset = TensorDataset(torch.tensor(observations, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=shuffle, pin_memory=pin_memory)


def train_torch_classifier(
    policy,
    bundle: DirectionDatasetBundle,
    *,
    device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    preload_to_device: bool = False,
    weight_decay: float = 0.0,
    gradient_clip_norm: float = 0.0,
) -> list[dict[str, Any]]:
    require_torch()
    observations, labels, _ = bundle.get_split_data("train")
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    pin_memory = device.type == "cuda"
    use_preloaded = bool(preload_to_device and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    x_device = torch.tensor(observations, dtype=torch.float32, device=device) if use_preloaded else None
    y_device = torch.tensor(labels, dtype=torch.long, device=device) if use_preloaded else None
    loader = None
    if not use_preloaded:
        loader = make_loader(observations, labels, batch_size, shuffle=True, pin_memory=pin_memory)
    autocast_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
    for epoch in range(1, int(epochs) + 1):
        policy.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batches: Any
        if use_preloaded and x_device is not None and y_device is not None:
            permutation = torch.randperm(x_device.shape[0], device=device)
            batches = []
            for start in range(0, x_device.shape[0], batch_size):
                end = min(start + batch_size, x_device.shape[0])
                batch_indices = permutation.narrow(0, start, end - start)
                batches.append(
                    (
                        x_device.index_select(0, batch_indices),
                        y_device.index_select(0, batch_indices),
                    )
                )
        else:
            batches = loader
        for batch_observations, batch_labels in batches:
            if not use_preloaded:
                batch_observations = batch_observations.to(device, non_blocking=pin_memory)
                batch_labels = batch_labels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                logits = policy(batch_observations)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(gradient_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * int(batch_labels.size(0))
            predictions = torch.argmax(logits.detach(), dim=-1)
            correct += int((predictions == batch_labels).sum().item())
            total += int(batch_labels.size(0))
        history.append({"epoch": epoch, "loss": total_loss / max(1, total), "train_accuracy": correct / max(1, total)})
    return history


def train_dagger(
    policy,
    bundle: DirectionDatasetBundle,
    *,
    device,
    hyperparameters: dict[str, Any],
) -> list[dict[str, Any]]:
    require_torch()
    base_observations, base_actions, _ = bundle.get_split_data("train")
    observations = [base_observations]
    actions = [base_actions]
    rounds = int(hyperparameters.get("rounds", 4))
    fit_epochs = int(hyperparameters.get("fit_epochs_per_round", 2))
    history: list[dict[str, Any]] = []
    env = BTCDirectionEnv(
        bundle,
        "train",
        action_dim=int(hyperparameters.get("action_dim", 2)),
        reward_value=float(hyperparameters.get("reward_value", 1.0)),
    )
    for round_index in range(1, rounds + 1):
        rollout_observations: list[np.ndarray] = []
        expert_actions: list[int] = []
        observation = env.reset()
        done = False
        policy.eval()
        while not done:
            with torch.inference_mode():
                obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(obs_tensor)
                action = int(torch.argmax(logits, dim=-1).item())
            next_observation, _, done, info = env.step(action)
            rollout_observations.append(observation)
            expert_actions.append(int(info["expert_action"]))
            observation = next_observation
        observations.append(np.asarray(rollout_observations, dtype=np.float32))
        actions.append(np.asarray(expert_actions, dtype=np.int64))
        fit_observations = np.concatenate(observations, axis=0)
        fit_actions = np.concatenate(actions, axis=0)
        loader = make_loader(
            fit_observations,
            fit_actions,
            int(hyperparameters.get("batch_size", 64)),
            shuffle=True,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(hyperparameters.get("learning_rate", 1e-3)),
            weight_decay=float(hyperparameters.get("weight_decay", 0.0)),
        )
        criterion = nn.CrossEntropyLoss()
        final_loss = 0.0
        final_accuracy = 0.0
        for _ in range(fit_epochs):
            total = 0
            correct = 0
            total_loss = 0.0
            for batch_observations, batch_labels in loader:
                batch_observations = batch_observations.to(device, non_blocking=device.type == "cuda")
                batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
                logits = policy(batch_observations)
                loss = criterion(logits, batch_labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_clip_norm = float(hyperparameters.get("gradient_clip_norm", 0.0))
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
                total_loss += float(loss.item()) * int(batch_labels.size(0))
                correct += int((torch.argmax(logits, dim=-1) == batch_labels).sum().item())
                total += int(batch_labels.size(0))
            final_loss = total_loss / max(1, total)
            final_accuracy = correct / max(1, total)
        history.append({"round": round_index, "loss": final_loss, "train_accuracy": final_accuracy})
    return history


def ppo_update(
    policy,
    optimizer,
    observations,
    actions,
    old_log_probs,
    advantages,
    returns,
    *,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    entropy_coef: float,
    value_loss_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    sample_count = int(observations.shape[0])
    total_loss = 0.0
    updates = 0
    for _ in range(int(epochs)):
        permutation = torch.randperm(sample_count, device=observations.device)
        for start in range(0, sample_count, int(minibatch_size)):
            end = min(start + int(minibatch_size), sample_count)
            idx = permutation.narrow(0, start, end - start)
            logits, values = policy(observations[idx])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns[idx])
            loss = policy_loss + (value_loss_coef * value_loss) - (entropy_coef * entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()
            total_loss += float(loss.item())
            updates += 1
    return {"loss": total_loss / max(1, updates)}


def train_actor_critic_or_ppo(
    policy,
    bundle: DirectionDatasetBundle,
    *,
    device,
    hyperparameters: dict[str, Any],
) -> list[dict[str, Any]]:
    require_torch()
    observations, labels, _ = bundle.get_split_data("train")
    obs = torch.tensor(observations, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    policy.to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=float(hyperparameters.get("learning_rate", 3e-4)),
        weight_decay=float(hyperparameters.get("weight_decay", 0.0)),
    )
    total_updates = int(hyperparameters.get("total_updates", 5))
    ppo_epochs = int(hyperparameters.get("ppo_epochs", 4))
    minibatch_size = int(hyperparameters.get("minibatch_size", 256))
    clip_epsilon = float(hyperparameters.get("clip_epsilon", 0.2))
    entropy_coef = float(hyperparameters.get("entropy_coef", 0.01))
    value_loss_coef = float(hyperparameters.get("value_loss_coef", 0.5))
    max_grad_norm = float(hyperparameters.get("max_grad_norm", 1.0))
    reward_value = float(hyperparameters.get("reward_value", 1.0))
    history: list[dict[str, Any]] = []
    for update in range(1, total_updates + 1):
        policy.eval()
        with torch.no_grad():
            logits, values = policy(obs)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            old_log_probs = dist.log_prob(actions)
            reward_scale = torch.full_like(values, reward_value)
            rewards = torch.where(actions == label_tensor, reward_scale, -reward_scale)
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        metrics = ppo_update(
            policy,
            optimizer,
            obs,
            actions,
            old_log_probs,
            advantages,
            rewards,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
        )
        with torch.no_grad():
            eval_logits, _ = policy(obs)
            accuracy = float((torch.argmax(eval_logits, dim=-1) == label_tensor).float().mean().item())
        history.append({"update": update, "loss": metrics["loss"], "train_accuracy": accuracy})
    return history


def evaluate_torch_policy(
    policy,
    bundle: DirectionDatasetBundle,
    *,
    device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    require_torch()
    observations, _, _ = bundle.get_split_data("test")
    loader = make_loader(
        observations,
        np.zeros(len(observations), dtype=np.int64),
        batch_size,
        False,
        device.type == "cuda",
    )
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    policy.eval()
    with torch.inference_mode():
        for batch_observations, _ in loader:
            batch_observations = batch_observations.to(device, non_blocking=device.type == "cuda")
            logits = policy(batch_observations)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)
            predictions.append(pred.detach().cpu().numpy().astype(np.int8))
            probabilities.append(confidence.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(predictions), np.concatenate(probabilities)


def array_to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())
    if hasattr(values, "get"):
        return np.asarray(values.get())
    return np.asarray(values)


def cuda_requested_or_auto(hyperparameters: dict[str, Any]) -> bool:
    device = str(hyperparameters.get("device", "auto")).strip().lower()
    if device == "cpu":
        return False
    if device == "cuda":
        return True
    return bool(torch.cuda.is_available())


def train_sklearn_model(
    model_type: str,
    bundle: DirectionDatasetBundle,
    hyperparameters: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]], str]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    observations, labels, _ = bundle.get_split_data("train")
    flat = observations.reshape(observations.shape[0], -1)
    use_cuda = cuda_requested_or_auto(hyperparameters)
    requested_device = str(hyperparameters.get("device", "auto")).strip().lower()
    if use_cuda and model_type == "logistic_regression":
        try:
            from cuml.linear_model import LogisticRegression as CuMLLogisticRegression

            estimator = CuMLLogisticRegression(
                penalty=str(hyperparameters.get("penalty", "l2")),
                C=float(hyperparameters.get("C", 1.0)),
                max_iter=int(hyperparameters.get("max_iter", 1000)),
                tol=float(hyperparameters.get("tol", 1e-4)),
                fit_intercept=bool(hyperparameters.get("fit_intercept", True)),
            )
            estimator.fit(flat, labels)
            predictions = array_to_numpy(estimator.predict(flat)).astype(np.int64)
            accuracy = float(np.mean(predictions == labels))
            return estimator, [{"train_accuracy": accuracy, "backend": "cuml_cuda"}], "cuml_cuda"
        except Exception as exc:
            if requested_device == "cuda":
                raise RuntimeError("CUDA logistic_regression requested, but cuML training failed.") from exc
    if use_cuda and model_type == "rf":
        try:
            from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier

            estimator = CuMLRandomForestClassifier(
                n_estimators=int(hyperparameters.get("n_estimators", 300)),
                max_depth=int(hyperparameters.get("max_depth", 10)),
                max_features=hyperparameters.get("max_features", "sqrt"),
                bootstrap=bool(hyperparameters.get("bootstrap", True)),
                random_state=int(hyperparameters.get("seed", 42)),
                n_streams=int(hyperparameters.get("n_streams", 4)),
            )
            estimator.fit(flat, labels)
            predictions = array_to_numpy(estimator.predict(flat)).astype(np.int64)
            accuracy = float(np.mean(predictions == labels))
            return estimator, [{"train_accuracy": accuracy, "backend": "cuml_cuda"}], "cuml_cuda"
        except Exception as exc:
            if requested_device == "cuda":
                raise RuntimeError("CUDA random forest requested, but cuML training failed.") from exc
    if model_type == "logistic_regression":
        estimator = LogisticRegression(
            penalty=str(hyperparameters.get("penalty", "l2")),
            C=float(hyperparameters.get("C", 1.0)),
            solver=str(hyperparameters.get("solver", "saga")),
            max_iter=int(hyperparameters.get("max_iter", 1000)),
            tol=float(hyperparameters.get("tol", 1e-4)),
            class_weight=hyperparameters.get("class_weight", None),
            fit_intercept=bool(hyperparameters.get("fit_intercept", True)),
            random_state=int(hyperparameters.get("seed", 42)),
            n_jobs=int(hyperparameters.get("n_jobs", -1)),
        )
    elif model_type == "rf":
        estimator = RandomForestClassifier(
            n_estimators=int(hyperparameters.get("n_estimators", 300)),
            max_depth=int(hyperparameters.get("max_depth", 10)),
            min_samples_leaf=int(hyperparameters.get("min_samples_leaf", 5)),
            min_samples_split=int(hyperparameters.get("min_samples_split", 2)),
            max_features=hyperparameters.get("max_features", "sqrt"),
            class_weight=hyperparameters.get("class_weight", None),
            bootstrap=bool(hyperparameters.get("bootstrap", True)),
            random_state=int(hyperparameters.get("seed", 42)),
            n_jobs=int(hyperparameters.get("n_jobs", -1)),
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        xgb_params = {
            "n_estimators": int(hyperparameters.get("n_estimators", 500)),
            "max_depth": int(hyperparameters.get("max_depth", 4)),
            "learning_rate": float(hyperparameters.get("learning_rate", 0.03)),
            "subsample": float(hyperparameters.get("subsample", 0.9)),
            "colsample_bytree": float(hyperparameters.get("colsample_bytree", 0.9)),
            "min_child_weight": float(hyperparameters.get("min_child_weight", 1.0)),
            "reg_alpha": float(hyperparameters.get("reg_alpha", 0.0)),
            "reg_lambda": float(hyperparameters.get("reg_lambda", 1.0)),
            "objective": str(hyperparameters.get("objective", "binary:logistic")),
            "eval_metric": str(hyperparameters.get("eval_metric", "logloss")),
            "tree_method": str(hyperparameters.get("tree_method", "hist")),
            "n_jobs": int(hyperparameters.get("n_jobs", -1)),
            "random_state": int(hyperparameters.get("seed", 42)),
        }
        if use_cuda:
            gpu_params = {**xgb_params, "device": "cuda", "tree_method": "hist"}
            estimator = XGBClassifier(**gpu_params)
            try:
                estimator.fit(flat, labels)
                accuracy = float(estimator.score(flat, labels))
                return estimator, [{"train_accuracy": accuracy, "backend": "xgboost_cuda"}], "xgboost_cuda"
            except Exception as exc:
                if requested_device == "cuda":
                    raise RuntimeError("CUDA XGBoost requested, but GPU training failed.") from exc
        estimator = XGBClassifier(**xgb_params)
    else:
        raise ValueError(f"Unsupported sklearn model_type: {model_type}")
    estimator.fit(flat, labels)
    accuracy = float(estimator.score(flat, labels))
    return estimator, [{"train_accuracy": accuracy, "backend": "sklearn_cpu"}], "sklearn_cpu"


def evaluate_sklearn_model(estimator: Any, bundle: DirectionDatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    observations, _, _ = bundle.get_split_data("test")
    flat = observations.reshape(observations.shape[0], -1)
    prediction_input = flat
    try:
        params = estimator.get_xgb_params()
        if str(params.get("device", "")).startswith("cuda"):
            import cupy as cp

            prediction_input = cp.asarray(flat)
    except Exception:
        prediction_input = flat
    try:
        predictions = array_to_numpy(estimator.predict(prediction_input)).astype(np.int8)
        if hasattr(estimator, "predict_proba"):
            probabilities = array_to_numpy(estimator.predict_proba(prediction_input))
            if probabilities.ndim == 2:
                confidence = np.max(probabilities, axis=1).astype(np.float32)
            else:
                confidence = probabilities.astype(np.float32)
        else:
            confidence = np.ones(len(predictions), dtype=np.float32)
    except Exception:
        predictions = array_to_numpy(estimator.predict(flat)).astype(np.int8)
        if hasattr(estimator, "predict_proba"):
            probabilities = array_to_numpy(estimator.predict_proba(flat))
            if probabilities.ndim == 2:
                confidence = np.max(probabilities, axis=1).astype(np.float32)
            else:
                confidence = probabilities.astype(np.float32)
        else:
            confidence = np.ones(len(predictions), dtype=np.float32)
    return predictions, confidence


def train_model(
    *,
    model_type: str,
    train_mode: str,
    clean_frame: pd.DataFrame,
    hyperparameters: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import pickle

    model_key = model_type.strip().lower()
    seed = int(hyperparameters.get("seed", 42))
    set_seed(seed)
    sequence_length = int(hyperparameters.get("sequence_length", SEQUENCE_LENGTH))
    train_rows = int(hyperparameters.get("train_length", TRAIN_CANDLES))
    test_rows = int(hyperparameters.get("test_length", TEST_CANDLES))
    bundle = build_dataset_bundle(
        clean_frame,
        sequence_length=sequence_length,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = output_dir / "scaler.pkl"
    try:
        import joblib

        joblib.dump(bundle.scaler, scaler_path)
    except Exception:
        scaler_path.write_bytes(pickle.dumps(bundle.scaler))
    if model_key in {"logistic_regression", "rf", "xgboost"}:
        estimator, history, backend = train_sklearn_model(model_key, bundle, hyperparameters)
        predictions, probabilities = evaluate_sklearn_model(estimator, bundle)
        weights_path = output_dir / "weights.bin"
        try:
            import joblib

            joblib.dump(estimator, weights_path)
        except Exception:
            weights_path.write_bytes(pickle.dumps(estimator))
        del estimator
        gc.collect()
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "history": history,
            "scaler_path": str(output_dir / "scaler.pkl"),
            "backend": backend,
            "device": "cuda" if backend.endswith("_cuda") else "cpu",
            "train_length": bundle.train_row_count,
            "test_length": bundle.test_row_count,
        }

    require_torch()
    device = resolve_device(str(hyperparameters.get("device", "auto")))
    action_dim = int(hyperparameters.get("action_dim", 3 if model_key in {"mamba_post_base"} else 2))
    policy = build_torch_policy(model_key, bundle, hyperparameters, action_dim=action_dim).to(device)
    if model_key == "dagger":
        history = train_dagger(policy, bundle, device=device, hyperparameters=hyperparameters)
    elif model_key in {"ppo", "ppo_continue", "actor_critic", "mamba_post_base"} or train_mode in {
        "reinforcement_ppo",
        "post_base",
    }:
        history = train_actor_critic_or_ppo(policy, bundle, device=device, hyperparameters=hyperparameters)
    else:
        epochs = int(hyperparameters.get("epochs", 20 if model_key != "bc" else 120))
        batch_size = int(hyperparameters.get("batch_size", 128 if model_key == "mamba" else 64))
        if model_key == "mamba":
            candidate_batch_sizes = []
            current = min(batch_size, bundle.train_row_count)
            while current >= 1:
                candidate_batch_sizes.append(current)
                if current == 1:
                    break
                current = max(1, current // 2)
        else:
            candidate_batch_sizes = [batch_size]
        for attempt_batch_size in candidate_batch_sizes:
            try:
                history = train_torch_classifier(
                    policy,
                    bundle,
                    device=device,
                    epochs=epochs,
                    learning_rate=float(hyperparameters.get("learning_rate", 1e-3)),
                    batch_size=attempt_batch_size,
                    preload_to_device=bool(hyperparameters.get("preload_to_device", model_key == "mamba")),
                    weight_decay=float(hyperparameters.get("weight_decay", 0.0)),
                    gradient_clip_norm=float(hyperparameters.get("gradient_clip_norm", 0.0)),
                )
                break
            except RuntimeError as exc:
                if (
                    device.type != "cuda"
                    or "out of memory" not in str(exc).lower()
                    or attempt_batch_size == candidate_batch_sizes[-1]
                ):
                    raise
                torch.cuda.empty_cache()
        else:
            history = []
    predictions, probabilities = evaluate_torch_policy(
        policy,
        bundle,
        device=device,
        batch_size=int(hyperparameters.get("eval_batch_size", 1024)),
    )
    torch.save(
        {
            "model_type": model_key,
            "train_mode": train_mode,
            "state_dict": policy.state_dict(),
            "feature_columns": bundle.feature_columns,
            "sequence_length": bundle.sequence_length,
            "hyperparameters": hyperparameters,
        },
        output_dir / "weights.bin",
    )
    result = {
        "predictions": predictions,
        "probabilities": probabilities,
        "history": history,
        "scaler_path": str(output_dir / "scaler.pkl"),
        "backend": "torch",
        "device": str(device),
        "train_length": bundle.train_row_count,
        "test_length": bundle.test_row_count,
    }
    policy.to("cpu")
    del policy
    release_torch_memory()
    return result
