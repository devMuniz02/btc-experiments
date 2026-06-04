from __future__ import annotations

from collections import Counter
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.training import (
    build_dataset_bundle,
    build_torch_policy,
    evaluate_sklearn_model,
    evaluate_torch_policy,
    release_torch_memory,
    resolve_device,
)


def _pool_columns(frame: pd.DataFrame, model_id: str) -> tuple[str, str]:
    pred_column = f"{model_id}_pred"
    prob_column = f"{model_id}_prob"
    missing = [column for column in (pred_column, prob_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing ensemble input columns for {model_id}: {', '.join(missing)}")
    return pred_column, prob_column


def hard_majority_vote(predictions: np.ndarray) -> np.ndarray:
    output: list[int] = []
    for row in predictions:
        counts = Counter(int(value) for value in row)
        top_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == top_count]
        output.append(winners[0] if len(winners) == 1 else 0)
    return np.asarray(output, dtype=np.int8)


def unanimity_vote(predictions: np.ndarray) -> np.ndarray:
    output = []
    for row in predictions:
        unique = set(int(value) for value in row)
        if len(unique) == 1:
            output.append(next(iter(unique)))
        else:
            output.append(0)
    return np.asarray(output, dtype=np.int8)


def soft_average_vote(predictions: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = {0: np.zeros(len(predictions)), 1: np.zeros(len(predictions))}
    counts = {0: np.zeros(len(predictions)), 1: np.zeros(len(predictions))}
    for member_index in range(predictions.shape[1]):
        member_predictions = predictions[:, member_index].astype(int)
        member_probabilities = probabilities[:, member_index].astype(float)
        for label in (0, 1):
            mask = member_predictions == label
            scores[label][mask] += member_probabilities[mask]
            counts[label][mask] += 1
    averaged = np.vstack([scores[0], scores[1]]).T / np.maximum(
        np.vstack([counts[0], counts[1]]).T,
        1.0,
    )
    labels = np.asarray([0, 1], dtype=np.int8)
    chosen_indices = np.argmax(averaged, axis=1)
    return labels[chosen_indices], np.max(averaged, axis=1).astype(np.float32)


def ensemble_model_ids(hyperparameters: dict[str, Any]) -> list[str]:
    models_pool = hyperparameters.get("models_pool")
    if models_pool is None:
        models_pool = hyperparameters.get("model_ids", [])
    return [str(model_id) for model_id in models_pool]


def build_ensemble_predictions(frame: pd.DataFrame, hyperparameters: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    models_pool = ensemble_model_ids(hyperparameters)
    mechanism = str(hyperparameters["voting_mechanism"]).strip().lower()
    if mechanism == "soft_probability":
        mechanism = "soft_average"
    pred_columns: list[str] = []
    prob_columns: list[str] = []
    for model_id in models_pool:
        pred_column, prob_column = _pool_columns(frame, model_id)
        pred_columns.append(pred_column)
        prob_columns.append(prob_column)
    predictions = frame[pred_columns].to_numpy(dtype=np.int8)
    probabilities = frame[prob_columns].to_numpy(dtype=np.float32)
    if mechanism == "hard_majority":
        voted = hard_majority_vote(predictions)
        confidence = np.mean(predictions == voted[:, None], axis=1).astype(np.float32)
    elif mechanism == "soft_average":
        voted, confidence = soft_average_vote(predictions, probabilities)
    elif mechanism == "unanimity":
        voted = unanimity_vote(predictions)
        agreement = np.all(predictions == voted[:, None], axis=1)
        confidence = np.where(agreement, np.min(probabilities, axis=1), 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported voting_mechanism: {mechanism}")
    min_member_probability = float(hyperparameters.get("min_member_probability", 0.0))
    if min_member_probability > 0:
        member_pass = probabilities >= min_member_probability
        if bool(hyperparameters.get("require_unanimous_confidence", False)):
            confidence = np.where(np.all(member_pass, axis=1), confidence, 0.0).astype(np.float32)
        else:
            confidence = np.where(np.any(member_pass, axis=1), confidence, 0.0).astype(np.float32)
    return voted, confidence


def _load_joblib_or_pickle(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return pickle.loads(path.read_bytes())


def _evaluate_sklearn_split(estimator: Any, bundle, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    observations, _, _ = bundle.get_split_data(split_name)
    flat = observations.reshape(observations.shape[0], -1)
    predictions = estimator.predict(flat).astype(np.int8)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(flat)
        if probabilities.ndim == 2:
            confidence = np.max(probabilities, axis=1).astype(np.float32)
        else:
            confidence = probabilities.astype(np.float32)
    else:
        confidence = np.ones(len(predictions), dtype=np.float32)
    return predictions, confidence


def _evaluate_torch_split(
    policy: Any,
    bundle: Any,
    split_name: str,
    *,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    observations, _, _ = bundle.get_split_data(split_name)
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    policy.eval()
    with torch.inference_mode():
        for start in range(0, len(observations), int(batch_size)):
            end = start + int(batch_size)
            batch = torch.as_tensor(observations[start:end], dtype=torch.float32, device=device)
            logits = policy(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)
            predictions.append(pred.detach().cpu().numpy().astype(np.int8))
            probabilities.append(confidence.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(predictions), np.concatenate(probabilities)


def _load_base_model_outputs(
    *,
    model_dir: Path,
    clean_frame: pd.DataFrame,
    fallback_results: pd.DataFrame,
    model_id: str,
    default_hyperparameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata_path = model_dir / "hyperparameters.json"
    weights_path = model_dir / "weights.bin"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing stored artifacts for ensemble member {model_id}.")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_type = str(metadata.get("model_type") or "").strip().lower()
    hyperparameters = dict(metadata.get("hyperparameters") or {})
    sequence_length = int(hyperparameters.get("sequence_length", default_hyperparameters.get("sequence_length", 24)))
    train_rows = int(hyperparameters.get("train_length", default_hyperparameters.get("train_length", 100000)))
    test_rows = int(hyperparameters.get("test_length", default_hyperparameters.get("test_length", 5000)))
    bundle = build_dataset_bundle(
        clean_frame,
        sequence_length=sequence_length,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = _load_joblib_or_pickle(scaler_path)
        feature_values = bundle.frame[bundle.feature_columns].to_numpy(dtype=np.float32)
        bundle.scaled_features = scaler.transform(feature_values).astype(np.float32)
        bundle.scaler = scaler
    if model_type in {"rf", "xgboost"}:
        estimator = _load_joblib_or_pickle(weights_path)
        train_pred, train_prob = _evaluate_sklearn_split(estimator, bundle, "train")
        test_pred, test_prob = evaluate_sklearn_model(estimator, bundle)
    elif model_type in {"lstm", "transformer", "mamba", "nn", "bc", "dagger", "ppo", "ppo_continue", "actor_critic"}:
        import torch

        checkpoint = torch.load(weights_path, map_location="cpu")
        checkpoint_hyperparameters = dict(checkpoint.get("hyperparameters") or hyperparameters)
        action_dim = int(checkpoint_hyperparameters.get("action_dim", 2))
        device = resolve_device(str(default_hyperparameters.get("device", "auto")))
        policy = build_torch_policy(model_type, bundle, checkpoint_hyperparameters, action_dim=action_dim).to(device)
        policy.load_state_dict(checkpoint["state_dict"])
        raw_batch_size = default_hyperparameters.get("eval_batch_size") or checkpoint_hyperparameters.get(
            "eval_batch_size",
            1024,
        )
        batch_size = int(raw_batch_size)
        train_pred, train_prob = _evaluate_torch_split(policy, bundle, "train", device=device, batch_size=batch_size)
        test_pred, test_prob = evaluate_torch_policy(policy, bundle, device=device, batch_size=batch_size)
        policy.to("cpu")
        del policy
        release_torch_memory()
    elif model_type == "ensemble":
        raise ValueError(
            f"Ensemble member {model_id} is another ensemble; stacked training needs loadable base models, "
            "not derived ensemble prediction columns."
        )
    else:
        raise ValueError(f"Unsupported stacked ensemble member model_type: {model_type}")
    train_labels = bundle.labels[bundle.train_decision_indices].astype(np.int8)
    return train_pred, train_prob, test_pred, test_prob, train_labels


def train_stacked_logistic_ensemble(
    *,
    clean_frame: pd.DataFrame,
    results: pd.DataFrame,
    models_root: Path,
    variation_id: int,
    hyperparameters: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    models_pool = ensemble_model_ids(hyperparameters)
    if len(models_pool) < 2:
        raise ValueError("stacked_logistic ensembles require at least two trained model ids.")
    train_features: list[np.ndarray] = []
    test_features: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for model_id in models_pool:
        train_pred, train_prob, test_pred, test_prob, member_labels = _load_base_model_outputs(
            model_dir=models_root / f"var_{int(variation_id)}" / model_id,
            clean_frame=clean_frame,
            fallback_results=results,
            model_id=model_id,
            default_hyperparameters=hyperparameters,
        )
        if labels is None:
            labels = member_labels
        elif len(labels) != len(member_labels):
            min_len = min(len(labels), len(member_labels))
            labels = labels[-min_len:]
            train_features = [feature[-min_len:] for feature in train_features]
            train_pred = train_pred[-min_len:]
            train_prob = train_prob[-min_len:]
            member_labels = member_labels[-min_len:]
        elif len(train_pred) != len(labels):
            min_len = min(len(labels), len(train_pred))
            labels = labels[-min_len:]
            train_features = [feature[-min_len:] for feature in train_features]
            train_pred = train_pred[-min_len:]
            train_prob = train_prob[-min_len:]
            member_labels = member_labels[-min_len:]
        if not np.array_equal(labels, member_labels):
            raise ValueError(f"Training labels do not align for stacked ensemble member {model_id}.")
        train_features.append(np.column_stack([train_pred.astype(np.float32), train_prob.astype(np.float32)]))
        test_features.append(np.column_stack([test_pred.astype(np.float32), test_prob.astype(np.float32)]))
    if labels is None:
        raise ValueError("No ensemble member labels were available.")
    x_train = np.concatenate(train_features, axis=1)
    x_test = np.concatenate(test_features, axis=1)
    stacker = LogisticRegression(
        C=float(hyperparameters.get("stack_c", 1.0)),
        max_iter=int(hyperparameters.get("stack_max_iter", 1000)),
        solver=str(hyperparameters.get("stack_solver", "lbfgs")),
        random_state=int(hyperparameters.get("seed", 42)),
    )
    stacker.fit(x_train, labels)
    predictions = stacker.predict(x_test).astype(np.int8)
    probabilities = np.max(stacker.predict_proba(x_test), axis=1).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(stacker, output_dir / "weights.bin")
    except Exception:
        (output_dir / "weights.bin").write_bytes(pickle.dumps(stacker))
    history = [{"train_accuracy": float(stacker.score(x_train, labels)), "members": len(models_pool)}]
    del stacker
    release_torch_memory()
    return predictions, probabilities, {"history": history, "backend": "stacked_logistic", "device": "cpu"}
