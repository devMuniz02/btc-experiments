from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.training import (
    TEST_CANDLES,
    TRAIN_CANDLES,
    build_dataset_bundle,
    build_torch_policy,
    evaluate_sklearn_model,
    evaluate_torch_policy,
    release_torch_memory,
    resolve_device,
)

SKLEARN_MODEL_TYPES = {"logistic_regression", "rf", "xgboost"}


def load_pickle_or_joblib(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return pickle.loads(path.read_bytes())


def evaluate_saved_model(
    *,
    model_type: str,
    train_mode: str,
    clean_frame: pd.DataFrame,
    hyperparameters: dict[str, Any],
    model_dir: Path,
    test_length: int = TEST_CANDLES,
) -> tuple[np.ndarray, np.ndarray]:
    model_key = str(model_type).strip().lower()
    resolved_hyperparameters = {**hyperparameters, "test_length": int(test_length)}
    bundle = build_dataset_bundle(
        clean_frame,
        sequence_length=int(resolved_hyperparameters.get("sequence_length", 24)),
        train_rows=int(resolved_hyperparameters.get("train_length", TRAIN_CANDLES)),
        test_rows=int(test_length),
    )
    weights_path = model_dir / "weights.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")
    if model_key in SKLEARN_MODEL_TYPES:
        estimator = load_pickle_or_joblib(weights_path)
        return evaluate_sklearn_model(estimator, bundle)

    import torch

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    action_dim = int(resolved_hyperparameters.get("action_dim", 3 if model_key in {"mamba_post_base"} else 2))
    device = resolve_device(str(resolved_hyperparameters.get("device", "auto")))
    policy = build_torch_policy(model_key, bundle, resolved_hyperparameters, action_dim=action_dim)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.to(device)
    try:
        return evaluate_torch_policy(
            policy,
            bundle,
            device=device,
            batch_size=int(resolved_hyperparameters.get("eval_batch_size", 1024)),
        )
    finally:
        policy.to("cpu")
        del policy
        release_torch_memory()
