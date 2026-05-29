from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.btc_direction_learning.algorithms import behavior_cloning, dagger, ppo
from src.btc_direction_learning.config import (
    FIXED_REGIME_NAME,
    FIXED_TEST_POOL_HOURS,
    FIXED_TRAIN_POOL_HOURS,
)
from scripts.generate_experiment_tables import (
    extract_metrics_from_summary,
    generate_markdown_table,
    update_experiments_md,
)
from scripts._helpers import build_summary_contract
from src.btc_direction_learning.dataset import (
    DirectionDatasetBundle,
    build_direction_dataset_bundle,
    build_shared_data_dir,
    ensure_shared_data_dir,
    resolve_dataset_source_path,
)
from src.btc_direction_learning.env import BTCDirectionEnv, ENV_VERSION_BINARY, ENV_VERSION_TERNARY, NONE_ACTION
from src.btc_direction_learning.evaluation import clone_state_dict, evaluate_policy, evaluate_policy_with_market_hours_none, load_checkpoint, simulate_portfolio, summarize_actions_against_labels
from src.btc_direction_learning.models import (
    ClassificationPolicy,
    LSTMClassificationPolicy,
    MambaClassificationPolicy,
    TransformerClassificationPolicy,
    TransformerClassificationPolicyV2,
)
from src.utils.market_hours import is_allowed_prediction_target_timestamp
from src.utils.market_data import set_seed


EXPERIMENT_ROOT = PROJECT_ROOT / "EXPERIMENTS" / "1"
EXPERIMENT_NAME = "TRAIN_ONCE"
DATA_VARIANTS = ["full", "market_hours", "derived_market_hours"]
BASELINE_ORDER = ["RANDOM", "ALWAYS_UP", "ALWAYS_DOWN"]
MODEL_ORDER = ["BC", "DAGGER", "PPO", "NN", "RF", "XGBOOST", "LSTM", "TRANSFORMER", "MAMBA"]
ALL_SERIES_ORDER = [*BASELINE_ORDER, *MODEL_ORDER]
MODEL_ARTIFACTS_DIRNAME = "saved_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment #1 TRAIN_ONCE across full and market-hours datasets.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output-dir", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-version", choices=[ENV_VERSION_BINARY, ENV_VERSION_TERNARY], default=ENV_VERSION_BINARY)
    parser.add_argument("--ternary-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--raw-fetch-hours", type=int, default=0)
    parser.add_argument("--ppo-total-updates", type=int, default=5)
    parser.add_argument("--ppo-trajectories-per-update", type=int, default=128)
    parser.add_argument("--mamba-epochs", type=int, default=60)
    parser.add_argument(
        "--model-artifacts-dirname",
        default=MODEL_ARTIFACTS_DIRNAME,
        help="Directory name under each variant where trained model artifacts are saved.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DATA_VARIANTS,
        choices=DATA_VARIANTS,
        help="Optional subset of experiment variants to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        choices=MODEL_ORDER,
        help="Optional subset of trained models to run. Baselines are always included.",
    )
    parser.add_argument(
        "--overwrite-models",
        nargs="+",
        default=[],
        choices=MODEL_ORDER,
        help="Optional subset of trained models to retrain even if results already exist.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_portfolio_scenarios(actions: list[int], labels: list[int], env_version: str) -> dict[str, Any]:
    return {
        "fixed_dollar": simulate_portfolio(
            actions=actions,
            labels=labels,
            env_version=env_version,
            starting_balance=10.0,
            move_fraction=0.05,
            sizing_mode="fixed_dollar",
        ),
        "current_fraction": simulate_portfolio(
            actions=actions,
            labels=labels,
            env_version=env_version,
            starting_balance=10.0,
            move_fraction=0.05,
            sizing_mode="current_fraction",
        ),
        "peak_fraction": simulate_portfolio(
            actions=actions,
            labels=labels,
            env_version=env_version,
            starting_balance=10.0,
            move_fraction=0.05,
            sizing_mode="peak_fraction",
        ),
    }


def result_has_required_fields(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    train = result.get("train")
    test = result.get("test")
    portfolio = result.get("portfolio")
    return isinstance(train, dict) and isinstance(test, dict) and isinstance(portfolio, dict)


def variant_summary_is_complete(
    summary: dict[str, Any] | None,
    selected_models: list[str],
    expected_training_data_mode: str,
    overwrite_models: set[str] | None = None,
) -> bool:
    if not isinstance(summary, dict):
        return False
    if str(summary.get("training_data_mode", "native")) != expected_training_data_mode:
        return False
    if overwrite_models and any(model_name in overwrite_models for model_name in selected_models):
        return False
    expected_names = [*BASELINE_ORDER, *selected_models]
    model_map = {
        str(entry.get("name")): entry
        for entry in summary.get("models", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    return all(result_has_required_fields(model_map.get(name)) for name in expected_names)


def serialize_result(result: dict[str, Any], timestamps: list[str] | None = None) -> dict[str, Any]:
    payload = {
        "rewards": [float(value) for value in result["rewards"]],
        "actions": [int(value) for value in result["actions"]],
        "labels": [int(value) for value in result["labels"]],
        "chosen_action_probabilities": [
            (None if value is None else float(value))
            for value in result.get("chosen_action_probabilities", [])
        ],
        "cumulative_rewards": [float(value) for value in result["cumulative_rewards"]],
        "mean_reward": float(result["mean_reward"]),
        "total_reward": float(result["total_reward"]),
        "accuracy": float(result["accuracy"]),
        "accuracy_scored_count": int(result["accuracy_scored_count"]),
    }
    if timestamps is not None:
        payload["timestamps"] = [str(value) for value in timestamps]
    return payload


def split_timestamps(bundle: DirectionDatasetBundle, split_name: str) -> list[str]:
    _, _, indices = bundle.get_split_data(split_name)
    timestamps = pd.to_datetime(bundle.processed_df.iloc[indices]["timestamp"], utc=True)
    return [timestamp.isoformat() for timestamp in timestamps.tolist()]


def gate_actions_outside_market_hours(actions: list[int], timestamps: list[str], env_version: str) -> list[int]:
    if env_version != ENV_VERSION_TERNARY:
        return list(actions)
    gated: list[int] = []
    for action, timestamp in zip(actions, timestamps):
        gated.append(NONE_ACTION if not is_allowed_prediction_target_timestamp(timestamp) else int(action))
    return gated


class SklearnPolicyWrapper(nn.Module):
    def __init__(self, estimator: Any, action_dim: int) -> None:
        super().__init__()
        self.estimator = estimator
        self.action_dim = action_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        np_obs = observations.detach().cpu().numpy().reshape(observations.shape[0], -1)
        probabilities = self.estimator.predict_proba(np_obs)
        if probabilities.shape[1] < self.action_dim:
            padded = np.zeros((probabilities.shape[0], self.action_dim), dtype=np.float32)
            padded[:, : probabilities.shape[1]] = probabilities
            probabilities = padded
        probabilities = np.clip(probabilities, 1e-8, 1.0)
        logits = np.log(probabilities)
        return torch.tensor(logits, dtype=torch.float32, device=observations.device)


def build_loader(
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int = 64,
    *,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
    )
    worker_count = 0 if os.name == "nt" else min(4, max(0, (os.cpu_count() or 0) // 2))
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
    )


def split_train_validation(
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    validation_ratio: float,
    min_validation_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    total_rows = int(len(labels))
    if total_rows <= (min_validation_size * 2):
        return None
    validation_size = max(min_validation_size, int(total_rows * validation_ratio))
    validation_size = min(validation_size, total_rows // 5)
    if validation_size <= 0 or validation_size >= total_rows:
        return None
    split_index = total_rows - validation_size
    return (
        observations[:split_index],
        labels[:split_index],
        observations[split_index:],
        labels[split_index:],
    )


def train_torch_classifier(
    policy: nn.Module,
    observations: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    label: str,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    grad_clip_norm: float | None = None,
    val_observations: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    autocast_dtype: torch.dtype | None = None,
    use_cuda_autocast: bool = True,
    preload_to_device: bool = False,
    overfit: bool = False,
    overfit_target_accuracy: float = 0.99,
    overfit_max_epochs: int = 5000,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> list[dict[str, Any]]:
    pin_memory = device.type == "cuda"
    loader = build_loader(observations, labels, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = None
    if val_observations is not None and val_labels is not None and len(val_labels) > 0:
        val_loader = build_loader(val_observations, val_labels, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    optimizer_kwargs: dict[str, Any] = {"lr": learning_rate, "weight_decay": weight_decay}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer_class = torch.optim.Adam
    try:
        optimizer = optimizer_class(policy.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = optimizer_class(policy.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    resolved_autocast_dtype = autocast_dtype
    if resolved_autocast_dtype is None and device.type == "cuda" and use_cuda_autocast:
        resolved_autocast_dtype = torch.float16
    autocast_enabled = device.type == "cuda" and use_cuda_autocast and resolved_autocast_dtype is not None
    scaler_enabled = autocast_enabled and resolved_autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    history: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_loss: float | None = None
    stagnant_epochs = 0
    overfit_reached_epoch: int | None = None

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    x_device = None
    y_device = None
    use_preloaded_path = bool(preload_to_device and device.type == "cuda" and val_loader is None)
    if use_preloaded_path:
        x_device = torch.tensor(observations, dtype=torch.float32, device=device)
        y_device = torch.tensor(labels, dtype=torch.long, device=device)

    effective_epochs = max(int(epochs), int(overfit_max_epochs)) if overfit else int(epochs)

    for epoch in range(1, effective_epochs + 1):
        policy.train()
        total_loss = 0.0
        total = 0
        correct = 0
        if use_preloaded_path and x_device is not None and y_device is not None:
            permutation = torch.randperm(x_device.shape[0], device=device)
            effective_batch_size = min(batch_size, x_device.shape[0])
            for start in range(0, x_device.shape[0], effective_batch_size):
                batch_indices = permutation[start:start + effective_batch_size]
                batch_observations = x_device.index_select(0, batch_indices)
                batch_labels = y_device.index_select(0, batch_indices)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=resolved_autocast_dtype):
                    logits = policy(batch_observations)
                    loss = criterion(logits, batch_labels)
                if scaler_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                    optimizer.step()

                total_loss += float(loss.item()) * batch_labels.size(0)
                predictions = torch.argmax(logits, dim=-1)
                correct += int((predictions == batch_labels).sum().item())
                total += int(batch_labels.size(0))
        else:
            for batch_observations, batch_labels in loader:
                batch_observations = batch_observations.to(device, non_blocking=pin_memory)
                batch_labels = batch_labels.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=resolved_autocast_dtype):
                    logits = policy(batch_observations)
                    loss = criterion(logits, batch_labels)
                if scaler_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                    optimizer.step()

                total_loss += float(loss.item()) * batch_labels.size(0)
                predictions = torch.argmax(logits, dim=-1)
                correct += int((predictions == batch_labels).sum().item())
                total += int(batch_labels.size(0))

        epoch_metrics = {
            "epoch": epoch,
            "step": epoch,
            "loss": total_loss / max(1, total),
            "train_accuracy": correct / max(1, total),
            "label": label,
        }

        if val_loader is not None:
            policy.eval()
            val_loss_total = 0.0
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for batch_observations, batch_labels in val_loader:
                    batch_observations = batch_observations.to(device, non_blocking=pin_memory)
                    batch_labels = batch_labels.to(device, non_blocking=pin_memory)
                    with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=resolved_autocast_dtype):
                        logits = policy(batch_observations)
                        loss = criterion(logits, batch_labels)
                    val_loss_total += float(loss.item()) * batch_labels.size(0)
                    predictions = torch.argmax(logits, dim=-1)
                    val_correct += int((predictions == batch_labels).sum().item())
                    val_total += int(batch_labels.size(0))
            epoch_metrics["val_loss"] = val_loss_total / max(1, val_total)
            epoch_metrics["val_accuracy"] = val_correct / max(1, val_total)

            improved = (
                best_val_loss is None
                or epoch_metrics["val_loss"] < (best_val_loss - float(early_stopping_min_delta))
            )
            if improved:
                best_val_loss = float(epoch_metrics["val_loss"])
                best_state_dict = cpu_state_dict(policy)
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1

        history.append(epoch_metrics)
        if progress_callback is not None:
            callback_metrics = dict(epoch_metrics)
            callback_metrics["_policy"] = policy
            progress_callback(callback_metrics, epoch, effective_epochs)

        if overfit:
            train_accuracy_value = float(epoch_metrics["train_accuracy"])
            if overfit_reached_epoch is None and train_accuracy_value >= float(overfit_target_accuracy):
                overfit_reached_epoch = epoch
            if train_accuracy_value >= 1.0:
                break
            if overfit_reached_epoch is not None and epoch >= (overfit_reached_epoch + 100):
                break

        if (
            val_loader is not None
            and early_stopping_patience is not None
            and early_stopping_patience > 0
            and stagnant_epochs >= early_stopping_patience
        ):
            break

    if best_state_dict is not None:
        policy.load_state_dict(best_state_dict)
    return history


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def classifier_action_dim_for_env(env_version: str) -> int:
    if env_version == ENV_VERSION_TERNARY:
        return 2
    return 2


def training_env_version_for_ternary_eval(env_version: str) -> str:
    if env_version == ENV_VERSION_TERNARY:
        return ENV_VERSION_BINARY
    return env_version


def infer_policy_action_dim(policy: nn.Module) -> int:
    if hasattr(policy, "head") and hasattr(policy.head, "out_features"):
        return int(policy.head.out_features)
    if hasattr(policy, "policy_head") and hasattr(policy.policy_head, "out_features"):
        return int(policy.policy_head.out_features)
    action_dim = getattr(policy, "action_dim", None)
    if action_dim is not None:
        return int(action_dim)
    raise ValueError(f"Unable to infer action dimension for policy {policy.__class__.__name__}")


def infer_policy_hidden_dim(policy: nn.Module) -> int | None:
    if hasattr(policy, "input_proj") and hasattr(policy.input_proj, "out_features"):
        return int(policy.input_proj.out_features)
    if hasattr(policy, "policy_head") and hasattr(policy.policy_head, "in_features"):
        return int(policy.policy_head.in_features)
    if hasattr(policy, "head") and hasattr(policy.head, "in_features"):
        return int(policy.head.in_features)
    return None


def build_torch_artifact(
    model_name: str,
    policy: nn.Module,
    bundle: DirectionDatasetBundle,
    env_version: str,
) -> dict[str, Any]:
    action_dim = infer_policy_action_dim(policy)
    return {
        "kind": "torch",
        "filename": f"{model_name.lower()}.pt",
        "payload": {
            "model_name": model_name,
            "policy_class": policy.__class__.__name__,
            "sequence_length": int(bundle.sequence_length),
            "feature_dim": int(bundle.feature_dim),
            "action_dim": int(action_dim),
            "hidden_dim": infer_policy_hidden_dim(policy),
            "env_version": env_version,
            "state_dict": cpu_state_dict(policy),
        },
    }


def build_sklearn_artifact(model_name: str, estimator: Any, env_version: str) -> dict[str, Any]:
    return {
        "kind": "pickle",
        "filename": f"{model_name.lower()}.pkl",
        "payload": {
            "model_name": model_name,
            "env_version": env_version,
            "estimator": estimator,
        },
    }


def save_model_artifact(variant_dir: Path, result: dict[str, Any], artifacts_dirname: str) -> dict[str, Any]:
    artifact = result.pop("__model_artifact__", None)
    if not isinstance(artifact, dict):
        return result
    artifacts_dir = variant_dir / str(artifacts_dirname)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / str(artifact["filename"])
    if artifact["kind"] == "torch":
        torch.save(artifact["payload"], artifact_path)
    elif artifact["kind"] == "pickle":
        with artifact_path.open("wb") as handle:
            pickle.dump(artifact["payload"], handle)
    else:
        raise ValueError(f"Unsupported model artifact kind: {artifact['kind']}")
    result["saved_model_path"] = str(artifact_path.resolve())
    return result


def load_existing_variant_summary(variant_dir: Path) -> dict[str, Any] | None:
    summary_path = variant_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def prune_existing_summary_models(summary: dict[str, Any] | None, overwrite_models: set[str]) -> dict[str, Any] | None:
    if not isinstance(summary, dict) or not overwrite_models:
        return summary
    models = summary.get("models", [])
    if not isinstance(models, list):
        return summary
    cloned = dict(summary)
    cloned["models"] = [
        entry
        for entry in models
        if not (isinstance(entry, dict) and str(entry.get("name")) in overwrite_models)
    ]
    return cloned


def experiment_dataset_force_refresh(shared_data_dir: Path, data_variant: str, requested_force_refresh: bool) -> bool:
    if not requested_force_refresh:
        return False
    cache_dir = build_shared_data_dir(
        train_hours=FIXED_TRAIN_POOL_HOURS,
        test_hours=FIXED_TEST_POOL_HOURS,
        split_regime=FIXED_REGIME_NAME,
        data_variant=data_variant,
        data_dir=shared_data_dir,
    )
    processed_cache = cache_dir / "btc_features_cache.csv"
    metadata_cache = cache_dir / "split_metadata.json"
    return not (processed_cache.exists() and metadata_cache.exists())


def attach_timestamps_to_result(result: dict[str, Any], train_timestamps: list[str], test_timestamps: list[str]) -> dict[str, Any]:
    train_payload = result.get("train")
    if isinstance(train_payload, dict):
        train_payload["timestamps"] = [str(value) for value in train_timestamps]
    test_payload = result.get("test")
    if isinstance(test_payload, dict):
        test_payload["timestamps"] = [str(value) for value in test_timestamps]
    return result


def derive_result_with_market_hours_none(
    result: dict[str, Any],
    train_timestamps: list[str],
    test_timestamps: list[str],
) -> dict[str, Any]:
    derived = {
        "name": result.get("name"),
        "history": list(result.get("history", [])),
    }
    source_train = result.get("train", {})
    source_test = result.get("test", {})

    gated_train_actions = gate_actions_outside_market_hours(source_train.get("actions", []), train_timestamps, ENV_VERSION_TERNARY)
    gated_test_actions = gate_actions_outside_market_hours(source_test.get("actions", []), test_timestamps, ENV_VERSION_TERNARY)
    gated_train_probabilities = [
        None if int(action) == NONE_ACTION else value
        for action, value in zip(gated_train_actions, source_train.get("chosen_action_probabilities", []))
    ]
    gated_test_probabilities = [
        None if int(action) == NONE_ACTION else value
        for action, value in zip(gated_test_actions, source_test.get("chosen_action_probabilities", []))
    ]

    derived_train = serialize_result(
        summarize_actions_against_labels(
            gated_train_actions,
            source_train.get("labels", []),
            ENV_VERSION_TERNARY,
            chosen_action_probabilities=gated_train_probabilities,
        ),
        train_timestamps,
    )
    derived_test = serialize_result(
        summarize_actions_against_labels(
            gated_test_actions,
            source_test.get("labels", []),
            ENV_VERSION_TERNARY,
            chosen_action_probabilities=gated_test_probabilities,
        ),
        test_timestamps,
    )
    derived["train"] = derived_train
    derived["test"] = derived_test
    derived["portfolio"] = build_portfolio_scenarios(derived_test["actions"], derived_test["labels"], ENV_VERSION_TERNARY)
    return derived


def build_derived_market_hours_summary_from_full(full_summary: dict[str, Any]) -> dict[str, Any]:
    plotting = full_summary.get("plotting", {})
    train_timestamps = [str(value) for value in plotting.get("train_timestamps", [])]
    test_timestamps = [str(value) for value in plotting.get("test_timestamps", [])]
    models = [
        derive_result_with_market_hours_none(model, train_timestamps, test_timestamps)
        for model in full_summary.get("models", [])
        if isinstance(model, dict) and model.get("name")
    ]
    model_map = {entry["name"]: entry for entry in models}
    ordered_models = [model_map[name] for name in ALL_SERIES_ORDER if name in model_map]
    return {
        "variant": "derived_market_hours",
        "source_variant": "full",
        "env_version": ENV_VERSION_TERNARY,
        "ternary_confidence_threshold": float(full_summary.get("ternary_confidence_threshold", 0.0)),
        "force_none_outside_market_hours": True,
        "training_data_mode": "derived_market_hours",
        "data_variant": "derived_market_hours",
        "dataset_source_path": str(resolve_dataset_source_path("full").resolve()),
        "split_mode": "latest_5000_fixed_test",
        "dataset_metadata": dict(full_summary.get("dataset_metadata", {})),
        "source_test_window": {
            "start": test_timestamps[0] if test_timestamps else None,
            "end": test_timestamps[-1] if test_timestamps else None,
            "count": len(test_timestamps),
        },
        "plotting": {
            "x_axis_mode": "index_with_7_day_date_ticks",
            "train_timestamps": train_timestamps,
            "test_timestamps": test_timestamps,
        },
        "series_order": [entry["name"] for entry in ordered_models],
        "models": ordered_models,
    }


def build_derived_market_hours_bundle(full_bundle: DirectionDatasetBundle) -> DirectionDatasetBundle:
    processed_df = full_bundle.processed_df.copy()
    labels = full_bundle.labels.copy()
    timestamps = pd.to_datetime(processed_df["timestamp"], utc=True)
    decision_indices = np.concatenate([full_bundle.train_decision_indices, full_bundle.test_decision_indices])
    for decision_index in decision_indices:
        if not is_allowed_prediction_target_timestamp(timestamps.iloc[int(decision_index)]):
            labels[int(decision_index)] = NONE_ACTION
    return DirectionDatasetBundle(
        processed_df=processed_df,
        scaled_features=full_bundle.scaled_features.copy(),
        scaler=full_bundle.scaler,
        feature_columns=list(full_bundle.feature_columns),
        sequence_length=full_bundle.sequence_length,
        labels=labels,
        train_decision_indices=full_bundle.train_decision_indices.copy(),
        test_decision_indices=full_bundle.test_decision_indices.copy(),
        train_row_count=full_bundle.train_row_count,
        test_row_count=full_bundle.test_row_count,
    )


def evaluate_model(
    model_name: str,
    policy: nn.Module,
    train_env: BTCDirectionEnv,
    test_env: BTCDirectionEnv,
    device: torch.device,
    ternary_confidence_threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool = False,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if force_none_outside_market_hours:
        train_raw = evaluate_policy_with_market_hours_none(
            train_env,
            policy,
            device=device,
            timestamps=train_timestamps,
            ternary_confidence_threshold=ternary_confidence_threshold,
        )
        test_raw = evaluate_policy_with_market_hours_none(
            test_env,
            policy,
            device=device,
            timestamps=test_timestamps,
            ternary_confidence_threshold=ternary_confidence_threshold,
        )
    else:
        train_raw = evaluate_policy(
            train_env,
            policy,
            device=device,
            ternary_confidence_threshold=ternary_confidence_threshold,
        )
        test_raw = evaluate_policy(
            test_env,
            policy,
            device=device,
            ternary_confidence_threshold=ternary_confidence_threshold,
        )

    train_result = serialize_result(train_raw, train_timestamps)
    test_result = serialize_result(test_raw, test_timestamps)
    return {
        "name": model_name,
        "history": history or [],
        "train": train_result,
        "test": test_result,
        "portfolio": build_portfolio_scenarios(test_result["actions"], test_result["labels"], test_env.env_version),
    }


def run_bc(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
) -> dict[str, Any]:
    policy = behavior_cloning.build_policy(
        bundle.sequence_length,
        bundle.feature_dim,
        action_dim=classifier_action_dim_for_env(env_version),
    ).to(device)
    history, checkpoints, _ = behavior_cloning.train(bundle, policy, device)
    load_checkpoint(policy, checkpoints["final"])
    train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    result = evaluate_model(
        "BC",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_torch_artifact("BC", policy, bundle, env_version)
    return result


def run_dagger(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
) -> dict[str, Any]:
    policy = dagger.build_policy(
        bundle.sequence_length,
        bundle.feature_dim,
        action_dim=classifier_action_dim_for_env(env_version),
    ).to(device)
    history, checkpoints, _ = dagger.train(
        bundle,
        policy,
        device,
        env_version=training_env_version_for_ternary_eval(env_version),
        ternary_confidence_threshold=threshold,
    )
    load_checkpoint(policy, checkpoints["final"])
    train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    result = evaluate_model(
        "DAGGER",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_torch_artifact("DAGGER", policy, bundle, env_version)
    return result


def run_ppo(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
    ppo_total_updates: int,
    ppo_trajectories_per_update: int,
    ppo_accuracy_stop_threshold: float | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    ppo_previous_policy_kl_coef: float = 0.0,
) -> dict[str, Any]:
    rollout_env_version = training_env_version_for_ternary_eval(env_version)
    train_env = BTCDirectionEnv(bundle, "train", env_version=rollout_env_version)
    accuracy_eval_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    policy = ppo.build_policy(bundle.sequence_length, bundle.feature_dim, action_dim=train_env.action_dim).to(device)
    if initial_state_dict is not None:
        policy.load_state_dict(initial_state_dict)
    reference_policy_state_dict = clone_state_dict(policy) if ppo_previous_policy_kl_coef > 0.0 else None
    history, checkpoints, _ = ppo.train(
        train_env,
        policy,
        device,
        total_updates=0 if ppo_accuracy_stop_threshold is not None else ppo_total_updates,
        trajectories_per_update=ppo_trajectories_per_update,
        enable_progress=True,
        accuracy_stop_threshold=ppo_accuracy_stop_threshold,
        accuracy_eval_env=accuracy_eval_env,
        accuracy_eval_threshold=threshold,
        previous_policy_kl_coef=ppo_previous_policy_kl_coef,
        reference_policy_state_dict=reference_policy_state_dict,
    )
    load_checkpoint(policy, checkpoints["final"])
    result = evaluate_model(
        "PPO",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_torch_artifact("PPO", policy, bundle, env_version)
    return result


def run_nn(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    train_observations, train_labels, _ = bundle.get_split_data("train")
    policy = ClassificationPolicy(
        bundle.sequence_length,
        bundle.feature_dim,
        action_dim=classifier_action_dim_for_env(env_version),
    ).to(device)
    history = train_torch_classifier(
        policy,
        train_observations,
        train_labels,
        device,
        epochs=80,
        learning_rate=1e-3,
        batch_size=64,
        label="NN",
        progress_callback=progress_callback,
    )
    train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    result = evaluate_model(
        "NN",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_torch_artifact("NN", policy, bundle, env_version)
    return result


def run_lstm(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    train_observations, train_labels, _ = bundle.get_split_data("train")
    policy = LSTMClassificationPolicy(
        bundle.sequence_length,
        bundle.feature_dim,
        action_dim=classifier_action_dim_for_env(env_version),
        hidden_dim=64,
    ).to(device)
    history = train_torch_classifier(
        policy,
        train_observations,
        train_labels,
        device,
        epochs=60,
        learning_rate=5e-4,
        batch_size=64,
        label="LSTM",
        progress_callback=progress_callback,
    )
    train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    result = evaluate_model(
        "LSTM",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_torch_artifact("LSTM", policy, bundle, env_version)
    return result


def run_transformer(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    train_observations, train_labels, _ = bundle.get_split_data("train")
    train_rows = int(len(train_labels))
    if train_rows >= 30000:
        hidden_dim = 256
        epochs = 220
        learning_rate = 2e-4
        batch_size = 128
    elif train_rows >= 20000:
        hidden_dim = 256
        epochs = 180
        learning_rate = 2.5e-4
        batch_size = 128
    elif train_rows >= 10000:
        hidden_dim = 192
        epochs = 140
        learning_rate = 3e-4
        batch_size = 96
    else:
        hidden_dim = 128
        epochs = 100
        learning_rate = 5e-4
        batch_size = 64

    policy = TransformerClassificationPolicyV2(
        bundle.sequence_length,
        bundle.feature_dim,
        action_dim=classifier_action_dim_for_env(env_version),
        hidden_dim=hidden_dim,
    ).to(device)
    history = train_torch_classifier(
        policy,
        train_observations,
        train_labels,
        device,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        label="TRANSFORMER",
        progress_callback=progress_callback,
    )
    train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    result = evaluate_model(
        "TRANSFORMER",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_torch_artifact("TRANSFORMER", policy, bundle, env_version)
    return result


def run_mamba(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
    mamba_epochs: int | None = None,
    batch_size_override: int | None = None,
    disable_early_stopping: bool = False,
    overfit: bool = False,
    overfit_max_epochs: int = 5000,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    train_observations, train_labels, _ = bundle.get_split_data("train")
    train_rows = int(len(train_labels))
    split_data = None
    if train_rows >= 10000 and not disable_early_stopping:
        split_data = split_train_validation(
            train_observations,
            train_labels,
            validation_ratio=0.1,
            min_validation_size=512,
        )
    if split_data is None:
        fit_observations, fit_labels = train_observations, train_labels
        val_observations = None
        val_labels = None
    else:
        fit_observations, fit_labels, val_observations, val_labels = split_data

    if train_rows >= 30000:
        hidden_dim = 256
        learning_rate = 2e-4
        batch_size = 128
        weight_decay = 1e-2
        label_smoothing = 0.0
        early_stopping_patience = 8
        dropout = 0.0
    elif train_rows >= 20000:
        hidden_dim = 224
        learning_rate = 2.5e-4
        batch_size = 128
        weight_decay = 8e-3
        label_smoothing = 0.0
        early_stopping_patience = 8
        dropout = 0.0
    elif train_rows >= 10000:
        hidden_dim = 192
        learning_rate = 3e-4
        batch_size = 96
        weight_decay = 5e-3
        label_smoothing = 0.0
        early_stopping_patience = 7
        dropout = 0.0
    else:
        hidden_dim = 160
        learning_rate = 4e-4
        batch_size = 64
        weight_decay = 0.0
        label_smoothing = 0.0
        early_stopping_patience = 0
        dropout = 0.0
    if train_rows >= 30000:
        default_epochs = 180
    elif train_rows >= 20000:
        default_epochs = 150
    elif train_rows >= 10000:
        default_epochs = 120
    else:
        default_epochs = 90
    epochs = int(mamba_epochs) if mamba_epochs is not None else default_epochs
    if batch_size_override is not None:
        batch_size = max(1, min(int(batch_size_override), train_rows))
    if disable_early_stopping:
        early_stopping_patience = 0
    if overfit:
        weight_decay = 0.0
        label_smoothing = 0.0
        dropout = 0.0
        early_stopping_patience = 0
        val_observations = None
        val_labels = None
        fit_observations, fit_labels = train_observations, train_labels
    use_preloaded_cuda_fast_path = (
        device.type == "cuda"
        and val_observations is None
    )

    candidate_batch_sizes: list[int] = []
    current_batch_size = min(batch_size, max(1, train_rows))
    while current_batch_size >= 1:
        if current_batch_size not in candidate_batch_sizes:
            candidate_batch_sizes.append(current_batch_size)
        if current_batch_size == 1:
            break
        current_batch_size = max(1, current_batch_size // 2)

    last_oom_error: BaseException | None = None
    for attempt_batch_size in candidate_batch_sizes:
        policy = None
        try:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            policy = MambaClassificationPolicy(
                bundle.sequence_length,
                bundle.feature_dim,
                action_dim=classifier_action_dim_for_env(env_version),
                hidden_dim=hidden_dim,
                dropout=dropout,
            ).to(device)
            if initial_state_dict is not None:
                policy.load_state_dict(initial_state_dict)
            history = train_torch_classifier(
                policy,
                fit_observations,
                fit_labels,
                device,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=attempt_batch_size,
                label="MAMBA",
                weight_decay=weight_decay,
                label_smoothing=label_smoothing,
                grad_clip_norm=1.0,
                val_observations=val_observations,
                val_labels=val_labels,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=1e-4,
                autocast_dtype=(torch.bfloat16 if use_preloaded_cuda_fast_path else None),
                use_cuda_autocast=not overfit,
                preload_to_device=use_preloaded_cuda_fast_path,
                overfit=overfit,
                overfit_max_epochs=overfit_max_epochs,
                progress_callback=progress_callback,
            )
            train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
            test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
            result = evaluate_model(
                "MAMBA",
                policy,
                train_env,
                test_env,
                device,
                threshold,
                train_timestamps,
                test_timestamps,
                force_none_outside_market_hours=force_none_outside_market_hours,
                history=history,
            )
            result["__model_artifact__"] = build_torch_artifact("MAMBA", policy, bundle, env_version)
            return result
        except Exception as exc:
            message = str(exc).lower()
            is_cuda_oom = device.type == "cuda" and "out of memory" in message
            if not is_cuda_oom:
                raise
            last_oom_error = exc
            if progress_callback is not None:
                progress_callback(
                    {
                        "epoch": 0,
                        "step": 0,
                        "loss": 0.0,
                        "train_accuracy": 0.0,
                        "label": f"MAMBA_OOM_RETRY_BS_{attempt_batch_size}",
                        "_policy": None,
                    },
                    0,
                    epochs,
                )
        finally:
            if policy is not None:
                del policy
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if last_oom_error is not None:
        raise RuntimeError(
            f"MAMBA training exhausted all CUDA batch-size retries up to batch_size=1 for {train_rows} rows."
        ) from last_oom_error
    raise RuntimeError("MAMBA training failed before starting.")


def run_rf(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
) -> dict[str, Any]:
    train_observations, train_labels, _ = bundle.get_split_data("train")
    estimator = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=1)
    estimator.fit(train_observations.reshape(train_observations.shape[0], -1), train_labels)
    policy = SklearnPolicyWrapper(estimator, action_dim=classifier_action_dim_for_env(env_version)).to(device)
    train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    history = [{"step": 1, "label": "RF", "fit_rows": int(len(train_labels))}]
    result = evaluate_model(
        "RF",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_sklearn_artifact("RF", estimator, env_version)
    return result


def run_xgboost(
    bundle: DirectionDatasetBundle,
    device: torch.device,
    env_version: str,
    threshold: float,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool,
) -> dict[str, Any]:
    train_observations, train_labels, _ = bundle.get_split_data("train")
    action_dim = classifier_action_dim_for_env(env_version)
    estimator_kwargs = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 1,
    }
    if action_dim > 2:
        estimator_kwargs["objective"] = "multi:softprob"
        estimator_kwargs["num_class"] = action_dim
    else:
        estimator_kwargs["objective"] = "binary:logistic"
    estimator = XGBClassifier(**estimator_kwargs)
    estimator.fit(train_observations.reshape(train_observations.shape[0], -1), train_labels)
    policy = SklearnPolicyWrapper(estimator, action_dim=action_dim).to(device)
    train_env = BTCDirectionEnv(bundle, "train", env_version=env_version)
    test_env = BTCDirectionEnv(bundle, "test", env_version=env_version)
    history = [{"step": 1, "label": "XGBOOST", "fit_rows": int(len(train_labels))}]
    result = evaluate_model(
        "XGBOOST",
        policy,
        train_env,
        test_env,
        device,
        threshold,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
        history=history,
    )
    result["__model_artifact__"] = build_sklearn_artifact("XGBOOST", estimator, env_version)
    return result


def build_baseline_results(
    bundle: DirectionDatasetBundle,
    env_version: str,
    seed: int,
    train_timestamps: list[str],
    test_timestamps: list[str],
    force_none_outside_market_hours: bool = False,
) -> list[dict[str, Any]]:
    labels = bundle.get_split_data("test")[1].tolist()
    train_labels = bundle.get_split_data("train")[1].tolist()
    action_dim = 3 if env_version == ENV_VERSION_TERNARY else 2
    rng = np.random.default_rng(seed)
    baseline_specs = {
        "RANDOM": (
            rng.integers(low=0, high=action_dim, size=len(labels), dtype=np.int64).tolist(),
            rng.integers(low=0, high=action_dim, size=len(train_labels), dtype=np.int64).tolist(),
        ),
        "ALWAYS_UP": ([1] * len(labels), [1] * len(train_labels)),
        "ALWAYS_DOWN": ([0] * len(labels), [0] * len(train_labels)),
    }
    results = []
    for name, (test_actions, train_actions) in baseline_specs.items():
        if force_none_outside_market_hours:
            train_actions = gate_actions_outside_market_hours(train_actions, train_timestamps, env_version)
            test_actions = gate_actions_outside_market_hours(test_actions, test_timestamps, env_version)
        train_result = serialize_result(
            summarize_actions_against_labels(
                train_actions,
                train_labels,
                env_version,
                chosen_action_probabilities=[None] * len(train_actions),
            ),
            train_timestamps,
        )
        test_result = serialize_result(
            summarize_actions_against_labels(
                test_actions,
                labels,
                env_version,
                chosen_action_probabilities=[None] * len(test_actions),
            ),
            test_timestamps,
        )
        results.append(
            {
                "name": name,
                "history": [],
                "train": train_result,
                "test": test_result,
                "portfolio": build_portfolio_scenarios(test_result["actions"], test_result["labels"], env_version),
            }
        )
    return results


def build_variant_summary(
    variant_name: str,
    bundle: DirectionDatasetBundle,
    env_version: str,
    threshold: float,
    seed: int,
    device: torch.device,
    selected_models: list[str],
    ppo_total_updates: int,
    ppo_trajectories_per_update: int,
    source_variant: str | None = None,
    force_none_outside_market_hours: bool = False,
    existing_summary: dict[str, Any] | None = None,
    variant_dir: Path | None = None,
    model_artifacts_dirname: str = MODEL_ARTIFACTS_DIRNAME,
) -> dict[str, Any]:
    train_timestamps = split_timestamps(bundle, "train")
    test_timestamps = split_timestamps(bundle, "test")
    existing_models = {}
    if isinstance(existing_summary, dict):
        for entry in existing_summary.get("models", []):
            if isinstance(entry, dict) and entry.get("name"):
                existing_models[str(entry["name"])] = entry

    baselines = build_baseline_results(
        bundle,
        env_version,
        seed,
        train_timestamps,
        test_timestamps,
        force_none_outside_market_hours=force_none_outside_market_hours,
    )
    baseline_map = {entry["name"]: entry for entry in baselines}
    for baseline_name in BASELINE_ORDER:
        existing_entry = existing_models.get(baseline_name)
        if result_has_required_fields(existing_entry):
            baseline_map[baseline_name] = attach_timestamps_to_result(existing_entry, train_timestamps, test_timestamps)
    baselines = [baseline_map[name] for name in BASELINE_ORDER if name in baseline_map]

    model_runners = {
        "BC": run_bc,
        "DAGGER": run_dagger,
        "PPO": run_ppo,
        "NN": run_nn,
        "RF": run_rf,
        "XGBOOST": run_xgboost,
        "LSTM": run_lstm,
        "TRANSFORMER": run_transformer,
        "MAMBA": run_mamba,
    }
    preserved_trained_results = [
        attach_timestamps_to_result(existing_models[name], train_timestamps, test_timestamps)
        for name in MODEL_ORDER
        if name not in selected_models and result_has_required_fields(existing_models.get(name))
    ]

    def assemble_summary(current_trained_results: list[dict[str, Any]]) -> dict[str, Any]:
        models = {entry["name"]: entry for entry in [*baselines, *preserved_trained_results, *current_trained_results]}
        ordered_models = [models[name] for name in ALL_SERIES_ORDER if name in models]
        summary_contract = build_summary_contract(
            bundle,
            data_variant=variant_name,
            dataset_source_path=resolve_dataset_source_path("market_hours" if variant_name == "market_hours" else "full"),
            split_mode="latest_5000_fixed_test",
            source_variant=source_variant or variant_name,
        )
        return {
            "variant": variant_name,
            "env_version": env_version,
            "ternary_confidence_threshold": threshold,
            "force_none_outside_market_hours": force_none_outside_market_hours,
            "training_data_mode": "derived_market_hours" if variant_name == "derived_market_hours" else "native",
            **summary_contract,
            "series_order": [entry["name"] for entry in ordered_models],
            "models": ordered_models,
        }

    trained_results = []
    if variant_dir is not None:
        write_variant_artifacts(variant_dir, assemble_summary(trained_results))
    for name in selected_models:
        existing_entry = existing_models.get(name)
        if result_has_required_fields(existing_entry):
            trained_results.append(attach_timestamps_to_result(existing_entry, train_timestamps, test_timestamps))
            if variant_dir is not None:
                write_variant_artifacts(variant_dir, assemble_summary(trained_results))
            continue
        if name == "PPO":
            result = (
                model_runners[name](
                    bundle,
                    device,
                    env_version,
                    threshold,
                    train_timestamps,
                    test_timestamps,
                    force_none_outside_market_hours,
                    ppo_total_updates,
                    ppo_trajectories_per_update,
                )
            )
        else:
            result = (
                model_runners[name](
                    bundle,
                    device,
                    env_version,
                    threshold,
                    train_timestamps,
                    test_timestamps,
                    force_none_outside_market_hours,
                )
            )
        if variant_dir is not None:
            result = save_model_artifact(variant_dir, result, model_artifacts_dirname)
        trained_results.append(result)
        if variant_dir is not None:
            write_variant_artifacts(variant_dir, assemble_summary(trained_results))
    return assemble_summary(trained_results)


def write_variant_artifacts(variant_dir: Path, summary: dict[str, Any]) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for model in summary["models"]:
        history = model.get("history", [])
        if history:
            save_history_csv(variant_dir / f"{model['name'].lower()}_training_metrics.csv", history)


def update_experiment_markdown(root_dir: Path) -> Path:
    return PROJECT_ROOT / "EXPERIMENTS" / "EXPERIMENTS.md"


def main() -> None:
    args = parse_args()
    overwrite_models = set(args.overwrite_models)
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    variant_summaries: dict[str, Any] = {}
    shared_data_dir = ensure_shared_data_dir(PROJECT_ROOT / "artifacts" / "btc" / "direction_learning" / "_shared")
    selected_variants = list(dict.fromkeys(args.variants))
    existing_variant_summaries = {
        variant_name: load_existing_variant_summary(output_root / variant_name)
        for variant_name in selected_variants
    }

    full_bundle: DirectionDatasetBundle | None = None
    market_hours_bundle: DirectionDatasetBundle | None = None
    derived_market_hours_bundle: DirectionDatasetBundle | None = None

    def get_full_bundle() -> DirectionDatasetBundle:
        nonlocal full_bundle
        if full_bundle is None:
            full_bundle = build_direction_dataset_bundle(
                force_refresh=experiment_dataset_force_refresh(shared_data_dir, "full", args.force_refresh),
                train_hours=FIXED_TRAIN_POOL_HOURS,
                test_hours=FIXED_TEST_POOL_HOURS,
                raw_fetch_hours=args.raw_fetch_hours or None,
                split_regime=FIXED_REGIME_NAME,
                data_variant="full",
                data_dir=shared_data_dir,
            )
        return full_bundle

    def get_market_hours_bundle() -> DirectionDatasetBundle:
        nonlocal market_hours_bundle
        if market_hours_bundle is None:
            market_hours_bundle = build_direction_dataset_bundle(
                force_refresh=experiment_dataset_force_refresh(shared_data_dir, "market_hours", args.force_refresh),
                train_hours=FIXED_TRAIN_POOL_HOURS,
                test_hours=FIXED_TEST_POOL_HOURS,
                raw_fetch_hours=args.raw_fetch_hours or None,
                split_regime=FIXED_REGIME_NAME,
                data_variant="market_hours",
                data_dir=shared_data_dir,
            )
        return market_hours_bundle

    def get_derived_market_hours_bundle() -> DirectionDatasetBundle:
        nonlocal derived_market_hours_bundle
        if derived_market_hours_bundle is None:
            derived_market_hours_bundle = build_derived_market_hours_bundle(get_full_bundle())
        return derived_market_hours_bundle

    for variant_name in selected_variants:
        existing_summary_raw = existing_variant_summaries.get(variant_name)
        existing_summary = prune_existing_summary_models(existing_summary_raw, overwrite_models)
        expected_training_data_mode = "derived_market_hours" if variant_name == "derived_market_hours" else "native"
        existing_mode_matches = isinstance(existing_summary_raw, dict) and str(existing_summary_raw.get("training_data_mode", "native")) == expected_training_data_mode
        if variant_summary_is_complete(existing_summary, args.models, expected_training_data_mode, overwrite_models):
            summary = existing_summary
        else:
            if variant_name == "full":
                bundle = get_full_bundle()
                env_version = args.env_version
                threshold = args.ternary_confidence_threshold
                source_variant = "full"
                force_none_outside_market_hours = False
            elif variant_name == "market_hours":
                bundle = get_market_hours_bundle()
                env_version = args.env_version
                threshold = args.ternary_confidence_threshold
                source_variant = "market_hours"
                force_none_outside_market_hours = False
            else:
                bundle = get_derived_market_hours_bundle()
                env_version = ENV_VERSION_TERNARY
                threshold = max(args.ternary_confidence_threshold, 0.0)
                source_variant = "full"
                force_none_outside_market_hours = True
            summary = build_variant_summary(
                variant_name=variant_name,
                bundle=bundle,
                env_version=env_version,
                threshold=threshold,
                seed=args.seed,
                device=device,
                selected_models=args.models,
                ppo_total_updates=args.ppo_total_updates,
                ppo_trajectories_per_update=args.ppo_trajectories_per_update,
                source_variant=source_variant,
                force_none_outside_market_hours=force_none_outside_market_hours,
                existing_summary=existing_summary if existing_mode_matches else None,
                variant_dir=output_root / variant_name,
                model_artifacts_dirname=args.model_artifacts_dirname,
            )
        write_variant_artifacts(output_root / variant_name, summary)
        variant_summaries[variant_name] = summary

    markdown_path = update_experiment_markdown(output_root.parent)
    manifest = {
        "experiment_id": 1,
        "name": EXPERIMENT_NAME,
        "root_dir": str(output_root.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "variants": {
            variant_name: {
                "summary_path": str((output_root / variant_name / "summary.json").resolve()),
                "summary": variant_summaries[variant_name],
            }
            for variant_name in selected_variants
            if variant_name in variant_summaries
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate and update experiment markdown tables
    try:
        variant_tables: dict[str, str] = {}
        for variant_name in selected_variants:
            if variant_name in variant_summaries:
                summary = variant_summaries[variant_name]
                metrics = extract_metrics_from_summary(summary)
                if metrics:
                    variant_display_name = variant_name.replace("_", " ").title()
                    table = generate_markdown_table(metrics, variant_display_name)
                    variant_tables[variant_name] = table
        if variant_tables:
            update_experiments_md(1, variant_tables, dry_run=False)
    except Exception as e:
        print(f"Warning: Failed to generate experiment tables: {e}", file=sys.stderr)

    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "name": manifest["name"],
                "output_dir": str(output_root.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "variants": list(variant_summaries.keys()),
                "selected_variants": selected_variants,
                "models": args.models,
                "overwrite_models": args.overwrite_models,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
