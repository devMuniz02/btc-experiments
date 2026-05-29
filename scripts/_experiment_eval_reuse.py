from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.btc_direction_learning.continuation_models import (
    LSTMContinuationActorCriticPolicy,
    MambaBanditPolicy,
    MambaContinuationActorCriticPolicy,
    MLPContinuationActorCriticPolicy,
    TransformerContinuationActorCriticPolicy,
)
from src.btc_direction_learning.dataset import (
    DirectionDatasetBundle,
    build_direction_dataset_bundle,
    build_direction_dataset_bundle_from_absolute_window,
    load_unified_dataset_frame,
)
from src.btc_direction_learning.env import ENV_VERSION_INTENSITY11, ENV_VERSION_TERNARY, NONE_ACTION, none_action_for_env
from src.btc_direction_learning.evaluation import load_checkpoint, summarize_actions_against_labels
from src.btc_direction_learning.models import (
    ActorCriticPolicy,
    ClassificationPolicy,
    LSTMClassificationPolicy,
    MambaClassificationPolicy,
    TransformerClassificationPolicy,
    TransformerClassificationPolicyV2,
    extract_policy_logits,
)
from src.btc_direction_learning.train import build_portfolio_scenarios
from src.utils.market_hours import EASTERN_TZ, is_allowed_prediction_target_timestamp


POLICY_CLASSES: dict[str, type] = {
    "ClassificationPolicy": ClassificationPolicy,
    "LSTMClassificationPolicy": LSTMClassificationPolicy,
    "MambaClassificationPolicy": MambaClassificationPolicy,
    "TransformerClassificationPolicy": TransformerClassificationPolicy,
    "TransformerClassificationPolicyV2": TransformerClassificationPolicyV2,
    "ActorCriticPolicy": ActorCriticPolicy,
    "MLPContinuationActorCriticPolicy": MLPContinuationActorCriticPolicy,
    "LSTMContinuationActorCriticPolicy": LSTMContinuationActorCriticPolicy,
    "MambaContinuationActorCriticPolicy": MambaContinuationActorCriticPolicy,
    "MambaBanditPolicy": MambaBanditPolicy,
    "TransformerContinuationActorCriticPolicy": TransformerContinuationActorCriticPolicy,
}


class UnsupportedPredictionSource(RuntimeError):
    """Raised when a saved artifact or prediction file cannot support ternary fan-out."""


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


def normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def is_weekday_market_hours_timestamp(value: pd.Timestamp | str) -> bool:
    timestamp_utc = normalize_timestamp(value)
    timestamp_et = timestamp_utc.tz_convert(EASTERN_TZ)
    return timestamp_et.weekday() < 5 and is_allowed_prediction_target_timestamp(timestamp_utc)


def is_outside_weekday_market_hours_timestamp(value: pd.Timestamp | str) -> bool:
    return not is_weekday_market_hours_timestamp(value)


def threshold_pct_to_probability(threshold_pct: float | int) -> float:
    return max(0.0, min(1.0, float(threshold_pct) / 100.0))


def serialize_eval_result(eval_result: dict[str, Any], timestamps: list[str]) -> dict[str, Any]:
    return {
        "rewards": [float(value) for value in eval_result["rewards"]],
        "actions": [int(value) for value in eval_result["actions"]],
        "labels": [int(value) for value in eval_result["labels"]],
        "chosen_action_probabilities": [
            None if value is None else float(value)
            for value in eval_result.get("chosen_action_probabilities", [])
        ],
        "cumulative_rewards": [float(value) for value in eval_result["cumulative_rewards"]],
        "mean_reward": float(eval_result["mean_reward"]),
        "total_reward": float(eval_result["total_reward"]),
        "accuracy": float(eval_result["accuracy"]),
        "accuracy_scored_count": int(eval_result["accuracy_scored_count"]),
        "timestamps": [str(value) for value in timestamps],
    }


def _heuristic_policy_from_state_dict(
    state_dict: dict[str, Any],
    dataset_bundle: DirectionDatasetBundle,
    action_dim: int,
):
    keys = set(state_dict.keys())
    seq_len = dataset_bundle.sequence_length
    feat_dim = dataset_bundle.feature_dim
    if any("value_head" in key for key in keys):
        if any("position_embedding" in key for key in keys) or any("encoder." in key for key in keys):
            return TransformerContinuationActorCriticPolicy(
                sequence_length=seq_len,
                feature_dim=feat_dim,
                action_dim=action_dim,
            )
        if any("a_log" in key for key in keys) or any("depthwise_conv" in key for key in keys):
            return MambaContinuationActorCriticPolicy(
                sequence_length=seq_len,
                feature_dim=feat_dim,
                action_dim=action_dim,
            )
        if any("lstm." in key for key in keys):
            return LSTMContinuationActorCriticPolicy(
                sequence_length=seq_len,
                feature_dim=feat_dim,
                action_dim=action_dim,
            )
        return ActorCriticPolicy(sequence_length=seq_len, feature_dim=feat_dim, action_dim=action_dim)
    if any("posterior_weights" in key for key in keys) or any("sampled_weights" in key for key in keys):
        return MambaBanditPolicy(
            sequence_length=seq_len,
            feature_dim=feat_dim,
            action_dim=action_dim,
        )
    if any("lstm." in key for key in keys):
        return LSTMClassificationPolicy(sequence_length=seq_len, feature_dim=feat_dim, action_dim=action_dim)
    if any("a_log" in key for key in keys) or any("depthwise_conv" in key for key in keys):
        return MambaClassificationPolicy(sequence_length=seq_len, feature_dim=feat_dim, action_dim=action_dim)
    if any("position_embedding" in key for key in keys) or any("encoder." in key for key in keys):
        return TransformerClassificationPolicy(sequence_length=seq_len, feature_dim=feat_dim, action_dim=action_dim)
    return ClassificationPolicy(sequence_length=seq_len, feature_dim=feat_dim, action_dim=action_dim)


def _infer_hidden_dim_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    input_proj_weight = state_dict.get("input_proj.weight")
    if hasattr(input_proj_weight, "shape") and len(input_proj_weight.shape) >= 1:
        return int(input_proj_weight.shape[0])
    head_weight = state_dict.get("head.weight")
    if hasattr(head_weight, "shape") and len(head_weight.shape) >= 2:
        return int(head_weight.shape[1])
    policy_head_weight = state_dict.get("policy_head.weight")
    if hasattr(policy_head_weight, "shape") and len(policy_head_weight.shape) >= 2:
        return int(policy_head_weight.shape[1])
    return None


def build_policy_from_payload(payload: dict[str, Any], dataset_bundle: DirectionDatasetBundle, action_dim: int):
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("payload must contain a state_dict mapping")
    cls_name = payload.get("policy_class")
    cls = POLICY_CLASSES.get(str(cls_name)) if cls_name else None
    if cls is None:
        return _heuristic_policy_from_state_dict(state_dict, dataset_bundle, action_dim)
    init_kwargs = {
        "sequence_length": int(payload.get("sequence_length", dataset_bundle.sequence_length)),
        "feature_dim": int(payload.get("feature_dim", dataset_bundle.feature_dim)),
        "action_dim": int(payload.get("action_dim", action_dim)),
    }
    hidden_dim = payload.get("hidden_dim")
    if hidden_dim is None:
        hidden_dim = _infer_hidden_dim_from_state_dict(state_dict)
    if hidden_dim is not None and str(cls_name) in {
        "LSTMClassificationPolicy",
        "MambaClassificationPolicy",
        "TransformerClassificationPolicy",
        "TransformerClassificationPolicyV2",
        "ClassificationPolicy",
        "ActorCriticPolicy",
        "LSTMContinuationActorCriticPolicy",
        "MambaContinuationActorCriticPolicy",
        "MambaBanditPolicy",
        "TransformerContinuationActorCriticPolicy",
        "MLPContinuationActorCriticPolicy",
    }:
        init_kwargs["hidden_dim"] = int(hidden_dim)
    if str(cls_name) in {"MambaContinuationActorCriticPolicy", "MambaBanditPolicy"}:
        num_layers = payload.get("num_layers")
        if num_layers is not None:
            init_kwargs["num_layers"] = int(num_layers)
    if str(cls_name) == "MambaBanditPolicy":
        init_kwargs["bandit_strategy"] = str(payload.get("bandit_strategy") or "ts")
        init_kwargs["ucb_alpha"] = float(payload.get("ucb_alpha") or 1.0)
    return cls(**init_kwargs)


def model_train_rows(model_entry: dict[str, Any], summary: dict[str, Any]) -> int:
    dataset_meta = model_entry.get("dataset_metadata", {}) if isinstance(model_entry, dict) else {}
    summary_dataset_meta = summary.get("dataset_metadata", {}) if isinstance(summary, dict) else {}
    return int(
        model_entry.get("actual_train_rows")
        or model_entry.get("requested_train_rows")
        or dataset_meta.get("train_rows")
        or summary_dataset_meta.get("train_rows")
        or summary.get("train_hours")
        or 5000
    )


def model_test_rows(model_entry: dict[str, Any], summary: dict[str, Any]) -> int:
    dataset_meta = model_entry.get("dataset_metadata", {}) if isinstance(model_entry, dict) else {}
    summary_dataset_meta = summary.get("dataset_metadata", {}) if isinstance(summary, dict) else {}
    return int(
        dataset_meta.get("test_rows")
        or summary_dataset_meta.get("test_rows")
        or summary.get("test_hours")
        or 5000
    )


def build_fixed_bundle(
    *,
    data_variant: str,
    train_rows: int,
    test_rows: int,
) -> DirectionDatasetBundle:
    variant = data_variant if data_variant in {"full", "market_hours"} else "full"
    return build_direction_dataset_bundle(
        train_hours=int(train_rows),
        test_hours=int(test_rows),
        split_regime="fixed_5k_5k",
        data_variant=variant,
    )


def build_absolute_bundle(
    *,
    variant: str,
    train_start_row: int,
    train_rows: int,
    test_rows: int,
) -> DirectionDatasetBundle:
    source_variant = "market_hours" if variant == "market_hours" else "full"
    frame = load_unified_dataset_frame(source_variant)
    return build_direction_dataset_bundle_from_absolute_window(
        frame,
        train_start_row=int(train_start_row),
        train_hours=int(train_rows),
        test_hours=int(test_rows),
    )


def make_subset_test_bundle(parent_bundle: DirectionDatasetBundle, test_indices: np.ndarray) -> DirectionDatasetBundle:
    return DirectionDatasetBundle(
        processed_df=parent_bundle.processed_df,
        scaled_features=parent_bundle.scaled_features,
        scaler=parent_bundle.scaler,
        feature_columns=parent_bundle.feature_columns,
        sequence_length=parent_bundle.sequence_length,
        labels=parent_bundle.labels,
        train_decision_indices=parent_bundle.train_decision_indices,
        test_decision_indices=np.asarray(test_indices, dtype=np.int64),
        train_row_count=parent_bundle.train_row_count,
        test_row_count=int(len(test_indices)),
    )


def locate_decision_indices_for_timestamps(bundle: DirectionDatasetBundle, timestamps: list[str]) -> np.ndarray:
    target_timestamps = [str(value) for value in timestamps]
    bundle_timestamps = bundle.get_split_timestamps("test")
    if not target_timestamps:
        return np.asarray([], dtype=np.int64)
    if target_timestamps == bundle_timestamps[: len(target_timestamps)]:
        return bundle.test_decision_indices[: len(target_timestamps)]
    lookup = {str(ts): int(idx) for ts, idx in zip(bundle_timestamps, bundle.test_decision_indices)}
    indices: list[int] = []
    for timestamp in target_timestamps:
        if timestamp not in lookup:
            raise KeyError(f"Timestamp {timestamp} not found in bundle test split.")
        indices.append(lookup[timestamp])
    return np.asarray(indices, dtype=np.int64)


def family_prefix(name: str) -> str:
    text = str(name)
    if "_AGGREGATE" in text:
        return text.split("_AGGREGATE", 1)[0]
    if "_M" in text:
        return text.split("_M", 1)[0]
    return text


def window_series_suffix(name: str) -> str:
    return "_1K" if str(name).endswith("_1K") else ""


def find_forward_window_children(summary: dict[str, Any], aggregate_name: str) -> list[dict[str, Any]]:
    prefix = family_prefix(aggregate_name)
    suffix = window_series_suffix(aggregate_name)
    children = [
        entry
        for entry in summary.get("models", [])
        if isinstance(entry, dict)
        and str(entry.get("name", "")).startswith(f"{prefix}_M")
        and str(entry.get("name", "")).endswith(suffix)
        and entry.get("window_index") is not None
    ]
    children.sort(key=lambda entry: int(entry.get("window_index", 0)))
    return children


def build_baseline_prediction_stream(
    *,
    labels: list[int],
    timestamps: list[str],
    baseline_name: str,
) -> dict[str, Any]:
    label_values = [int(value) for value in labels]
    if baseline_name == "RANDOM":
        rng = np.random.default_rng(42)
        probability_up = [float(value) for value in rng.random(len(label_values))]
        probability_down = [1.0 - value for value in probability_up]
    elif baseline_name == "ALWAYS_UP":
        probability_up = [1.0] * len(label_values)
        probability_down = [0.0] * len(label_values)
    elif baseline_name == "ALWAYS_DOWN":
        probability_up = [0.0] * len(label_values)
        probability_down = [1.0] * len(label_values)
    else:
        raise UnsupportedPredictionSource(f"Unknown baseline {baseline_name}")
    return {
        "timestamps": [str(value) for value in timestamps],
        "labels": label_values,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "probability_none": [0.0] * len(label_values),
        "source_kind": "baseline",
    }


def _probabilities_from_logits(logits: torch.Tensor) -> tuple[float, float, float]:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[-1] < 2:
        raise UnsupportedPredictionSource(f"Unsupported logits shape {tuple(logits.shape)} for ternary reevaluation")
    probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
    if logits.shape[-1] == 2:
        return float(probs[0]), float(probs[1]), 0.0
    return float(probs[0]), float(probs[1]), float(probs[2])


def build_prediction_stream_from_saved_model(
    *,
    saved_path: str,
    eval_bundle: DirectionDatasetBundle,
    device: torch.device,
    timestamps_override: list[str] | None = None,
) -> dict[str, Any]:
    timestamps = [str(value) for value in (timestamps_override or eval_bundle.get_split_timestamps("test"))]
    observations, labels, _ = eval_bundle.get_split_data("test")
    label_list = [int(value) for value in labels.tolist()]
    probability_up: list[float] = []
    probability_down: list[float] = []
    probability_none: list[float] = []

    path = str(saved_path)
    if path.endswith(".pt"):
        try:
            raw = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location=device)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict checkpoint in {saved_path}")
        payload = raw if "state_dict" in raw else {"state_dict": raw, "policy_class": None}
        action_dim = int(payload.get("action_dim", 2))
        if action_dim not in (2, 3):
            raise UnsupportedPredictionSource(
                f"{Path(path).name} cannot be fanned out into ternary thresholds (action_dim={action_dim})"
            )
        policy = build_policy_from_payload(payload, eval_bundle, action_dim=action_dim).to(device)
        load_checkpoint(policy, payload["state_dict"])
        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            batch_size = 1024 if device.type == "cuda" else 256
            for start in range(0, len(observations), batch_size):
                obs_tensor = torch.tensor(observations[start:start + batch_size], dtype=torch.float32, device=device)
                logits = extract_policy_logits(policy, obs_tensor)
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                if probs.shape[-1] == 2:
                    probability_down.extend(float(value) for value in probs[:, 0].tolist())
                    probability_up.extend(float(value) for value in probs[:, 1].tolist())
                    probability_none.extend([0.0] * probs.shape[0])
                else:
                    probability_down.extend(float(value) for value in probs[:, 0].tolist())
                    probability_up.extend(float(value) for value in probs[:, 1].tolist())
                    probability_none.extend(float(value) for value in probs[:, 2].tolist())
        if was_training:
            policy.train()
    elif path.endswith(".pkl"):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        estimator = payload["estimator"] if isinstance(payload, dict) and "estimator" in payload else payload
        policy = SklearnPolicyWrapper(estimator, action_dim=2).to(device)
        with torch.no_grad():
            batch_size = 1024 if device.type == "cuda" else 256
            for start in range(0, len(observations), batch_size):
                obs_tensor = torch.tensor(observations[start:start + batch_size], dtype=torch.float32, device=device)
                logits = policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                probability_down.extend(float(value) for value in probs[:, 0].tolist())
                probability_up.extend(float(value) for value in probs[:, 1].tolist())
                probability_none.extend([0.0] * probs.shape[0])
    else:
        raise UnsupportedPredictionSource(f"Unsupported saved model path: {saved_path}")

    return {
        "timestamps": timestamps,
        "labels": label_list,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "probability_none": probability_none,
        "source_kind": "saved_model",
        "saved_model_path": str(Path(saved_path).resolve()),
    }


def load_prediction_stream_from_file(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".parquet":
        from src.utils.experiment_support import read_prediction_stream_parquet

        return read_prediction_stream_parquet(source_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("timestamps"), list) and isinstance(payload.get("actions"), list) and isinstance(payload.get("labels"), list):
        return {
            "timestamps": [str(value) for value in payload.get("timestamps", [])],
            "actions": [int(value) for value in payload.get("actions", [])],
            "labels": [int(value) for value in payload.get("labels", [])],
            "chosen_action_probabilities": [
                None if value is None else float(value) for value in payload.get("chosen_action_probabilities", [])
            ],
            "source_kind": "action_stream_file",
            "predictions_path": str(source_path.resolve()),
        }
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        if all(isinstance(row, dict) and "action" in row and "label" in row for row in rows):
            return {
                "timestamps": [str(row["timestamp"]) for row in rows],
                "actions": [int(row["action"]) for row in rows],
                "labels": [int(row["label"]) for row in rows],
                "chosen_action_probabilities": [
                    None if row.get("prediction") is None else float(row.get("prediction")) for row in rows
                ],
                "source_kind": "action_rows_file",
                "predictions_path": str(source_path.resolve()),
            }
        predictions: list[float] = []
        labels: list[int] = []
        timestamps: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                raise UnsupportedPredictionSource(f"Unsupported predictions payload in {path}")
            timestamps.append(str(row["timestamp"]))
            labels.append(int(row["label"]))
            predictions.append(float(row["prediction"]))
        if all(0.0 <= value <= 1.0 for value in predictions):
            probability_up = predictions
            probability_down = [1.0 - value for value in predictions]
            return {
                "timestamps": timestamps,
                "labels": labels,
                "probability_up": probability_up,
                "probability_down": probability_down,
                "probability_none": [0.0] * len(labels),
                "source_kind": "predictions_file",
                "predictions_path": str(source_path.resolve()),
            }
    raise UnsupportedPredictionSource(f"Prediction file {path} is not reusable for ternary threshold fan-out")


def concat_prediction_streams(streams: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps: list[str] = []
    labels: list[int] = []
    probability_up: list[float] = []
    probability_down: list[float] = []
    probability_none: list[float] = []
    for stream in streams:
        timestamps.extend([str(value) for value in stream.get("timestamps", [])])
        labels.extend([int(value) for value in stream.get("labels", [])])
        probability_up.extend([float(value) for value in stream.get("probability_up", [])])
        probability_down.extend([float(value) for value in stream.get("probability_down", [])])
        probability_none.extend([float(value) for value in stream.get("probability_none", [])])
    return {
        "timestamps": timestamps,
        "labels": labels,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "probability_none": probability_none,
        "source_kind": "aggregate_stream",
    }


def evaluate_prediction_stream(
    *,
    prediction_stream: dict[str, Any],
    threshold_pct: float | int,
    active_timestamp_predicate: Any | None = None,
    env_version: str = ENV_VERSION_TERNARY,
) -> dict[str, Any]:
    explicit_actions = prediction_stream.get("actions")
    if isinstance(explicit_actions, list):
        timestamps = [str(value) for value in prediction_stream.get("timestamps", [])]
        labels = [int(value) for value in prediction_stream.get("labels", [])]
        raw_probabilities = list(prediction_stream.get("chosen_action_probabilities", []))
        action_none = none_action_for_env(env_version)
        if action_none is None:
            raise UnsupportedPredictionSource(f"Env version {env_version} does not support action-stream reevaluation.")
        actions: list[int] = []
        chosen_action_probabilities: list[float | None] = []
        for index, action in enumerate(explicit_actions):
            timestamp = timestamps[index] if index < len(timestamps) else ""
            if active_timestamp_predicate is not None and not bool(active_timestamp_predicate(timestamp)):
                actions.append(int(action_none))
                chosen_action_probabilities.append(None)
                continue
            actions.append(int(action))
            chosen_action_probabilities.append(
                None if index >= len(raw_probabilities) or raw_probabilities[index] is None else float(raw_probabilities[index])
            )
        summarized = summarize_actions_against_labels(
            actions,
            labels,
            env_version,
            chosen_action_probabilities=chosen_action_probabilities,
        )
        serialized = serialize_eval_result(summarized, timestamps)
        portfolio = build_portfolio_scenarios(serialized["actions"], serialized["labels"], env_version)
        return {"test": serialized, "portfolio": portfolio}

    threshold_probability = threshold_pct_to_probability(threshold_pct)
    timestamps = [str(value) for value in prediction_stream.get("timestamps", [])]
    labels = [int(value) for value in prediction_stream.get("labels", [])]
    down_probs = [float(value) for value in prediction_stream.get("probability_down", [])]
    up_probs = [float(value) for value in prediction_stream.get("probability_up", [])]
    none_probs = [float(value) for value in prediction_stream.get("probability_none", [])]
    actions: list[int] = []
    chosen_action_probabilities: list[float | None] = []

    for timestamp, down_prob, up_prob, none_prob in zip(timestamps, down_probs, up_probs, none_probs):
        if active_timestamp_predicate is not None and not bool(active_timestamp_predicate(timestamp)):
            actions.append(NONE_ACTION)
            chosen_action_probabilities.append(None)
            continue
        if threshold_probability <= 0.5:
            if none_prob > max(down_prob, up_prob):
                actions.append(NONE_ACTION)
                chosen_action_probabilities.append(None)
            elif up_prob >= down_prob:
                actions.append(1)
                chosen_action_probabilities.append(float(up_prob))
            else:
                actions.append(0)
                chosen_action_probabilities.append(float(down_prob))
            continue
        if up_prob >= down_prob and up_prob >= threshold_probability:
            actions.append(1)
            chosen_action_probabilities.append(float(up_prob))
        elif down_prob > up_prob and down_prob >= threshold_probability:
            actions.append(0)
            chosen_action_probabilities.append(float(down_prob))
        else:
            actions.append(NONE_ACTION)
            chosen_action_probabilities.append(None)

    summarized = summarize_actions_against_labels(
        actions,
        labels,
        env_version,
        chosen_action_probabilities=chosen_action_probabilities,
    )
    serialized = serialize_eval_result(summarized, timestamps)
    portfolio = build_portfolio_scenarios(serialized["actions"], serialized["labels"], env_version)
    return {"test": serialized, "portfolio": portfolio}
