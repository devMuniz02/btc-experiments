from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


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
        if len(unique) == 1 and 0 not in unique:
            output.append(next(iter(unique)))
        else:
            output.append(0)
    return np.asarray(output, dtype=np.int8)


def soft_average_vote(predictions: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = {-1: np.zeros(len(predictions)), 0: np.zeros(len(predictions)), 1: np.zeros(len(predictions))}
    counts = {-1: np.zeros(len(predictions)), 0: np.zeros(len(predictions)), 1: np.zeros(len(predictions))}
    for member_index in range(predictions.shape[1]):
        member_predictions = predictions[:, member_index].astype(int)
        member_probabilities = probabilities[:, member_index].astype(float)
        for label in (-1, 0, 1):
            mask = member_predictions == label
            scores[label][mask] += member_probabilities[mask]
            counts[label][mask] += 1
    averaged = np.vstack([scores[-1], scores[0], scores[1]]).T / np.maximum(
        np.vstack([counts[-1], counts[0], counts[1]]).T,
        1.0,
    )
    labels = np.asarray([-1, 0, 1], dtype=np.int8)
    chosen_indices = np.argmax(averaged, axis=1)
    return labels[chosen_indices], np.max(averaged, axis=1).astype(np.float32)


def build_ensemble_predictions(frame: pd.DataFrame, hyperparameters: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    models_pool = [str(model_id) for model_id in hyperparameters["models_pool"]]
    mechanism = str(hyperparameters["voting_mechanism"]).strip().lower()
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
        confidence[voted == 0] = 0.0
        return voted, confidence
    if mechanism == "soft_average":
        return soft_average_vote(predictions, probabilities)
    if mechanism == "unanimity":
        voted = unanimity_vote(predictions)
        confidence = np.where(voted == 0, 0.0, np.min(probabilities, axis=1)).astype(np.float32)
        return voted, confidence
    raise ValueError(f"Unsupported voting_mechanism: {mechanism}")
