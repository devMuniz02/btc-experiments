from __future__ import annotations

import numpy as np


def _binary_auc(true: np.ndarray, probability: np.ndarray) -> float:
    positive = probability[true == 1]
    negative = probability[true == 0]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def classification_metrics(y_true, y_pred, returns=None, probabilities=None) -> dict[str, float]:
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    returns_arr = np.asarray(returns if returns is not None else np.zeros_like(true), dtype=float)
    accuracy = float((true == pred).mean()) if len(true) else 0.0
    tp = float(((true == 1) & (pred == 1)).sum())
    fp = float(((true == 0) & (pred == 1)).sum())
    fn = float(((true == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tn = float(((true == 0) & (pred == 0)).sum())
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    probability = np.asarray(
        probabilities if probabilities is not None else np.where(pred == 1, 0.55, 0.45), dtype=float
    )
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    loss = float(-np.mean(true * np.log(probability) + (1 - true) * np.log(1 - probability))) if len(true) else 0.0
    auc = _binary_auc(true, probability)
    calibration_error = float(abs(probability.mean() - true.mean())) if len(true) else 0.0
    signed = np.where(pred == 1, 1.0, -1.0)
    pnl = signed * returns_arr
    std = float(np.std(pnl)) or 1.0
    sharpe = float(np.mean(pnl) / std * np.sqrt(252))
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity) if len(equity) else equity
    max_drawdown = float(np.min(equity - peak)) if len(equity) else 0.0
    return {
        "accuracy": accuracy,
        "direction_accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "precision_up": float(precision),
        "recall": float(recall),
        "recall_up": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "auc": float(auc),
        "loss": loss,
        "calibration_error": calibration_error,
        "trade_count": float(len(true)),
        "cumulative_return": float(np.sum(pnl)),
        "trading_proxy_score": float(sharpe),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "weighted_score": float(accuracy + 0.1 * sharpe),
    }
