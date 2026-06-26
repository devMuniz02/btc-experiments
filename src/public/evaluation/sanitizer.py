from __future__ import annotations

from pathlib import Path


FORBIDDEN_PUBLIC_PATTERNS = (
    "privateexperiments",
    "src/private",
    "model weights",
    "scaler.pkl",
    ".joblib",
    ".pkl",
    ".safetensors",
    ".onnx",
    "HF_TOKEN",
    "token=",
    "live prediction",
    "recent prediction",
    '"data_variation"',
    "rolling_mean",
    "rolling_gaussian",
    "ema_low_pass",
    '"axis":',
    "target_horizon",
    "feature_set",
    "scaling_transform",
    "lookback_window",
    "model_family",
    "training_hyperparams",
    "objective_loss",
    "sampling_imbalance",
    "regime_modeling",
    "final_validation_lock",
)


def public_safety_violations(text: str) -> list[str]:
    lowered = text.lower()
    return [pattern for pattern in FORBIDDEN_PUBLIC_PATTERNS if pattern.lower() in lowered]


def assert_public_safe_text(text: str) -> None:
    violations = public_safety_violations(text)
    if violations:
        raise ValueError(f"Public text contains forbidden patterns: {', '.join(violations)}")


def assert_public_safe_file(path: Path) -> None:
    assert_public_safe_text(path.read_text(encoding="utf-8"))
