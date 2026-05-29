from __future__ import annotations

from pathlib import Path

from src.utils.path_config import ARTIFACTS_DIR


DEFAULT_TRAIN_HOURS = 450
DEFAULT_TEST_HOURS = 50
DEFAULT_TOTAL_LABELED_HOURS = DEFAULT_TRAIN_HOURS + DEFAULT_TEST_HOURS
DEFAULT_RAW_FETCH_HOURS = 700
FIXED_REGIME_NAME = "fixed_5k_5k"
VARIABLE_REGIME_NAME = "variable"
DEFAULT_SPLIT_REGIME = VARIABLE_REGIME_NAME
FIXED_TRAIN_POOL_HOURS = 5000
FIXED_TEST_POOL_HOURS = 5000
FIXED_TOTAL_LABELED_HOURS = FIXED_TRAIN_POOL_HOURS + FIXED_TEST_POOL_HOURS
FIXED_TEST_START_TIMESTAMP = "2025-10-06T16:00:00+00:00"
FEATURE_WARMUP_PADDING = 200
MARKET_HOURS_MAX_FETCH_RETRIES = 5

ARTIFACT_ROOT = ARTIFACTS_DIR / "btc" / "direction_learning"
SHARED_DATA_DIR = ARTIFACT_ROOT / "_shared"
LEGACY_RL_PPO_DIR = ARTIFACTS_DIR / "btc" / "rl_ppo"


def get_algorithm_output_dir(algorithm_name: str, output_dir: Path | None = None) -> Path:
    base_dir = output_dir or (ARTIFACT_ROOT / algorithm_name)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def build_split_slug(train_hours: int, test_hours: int) -> str:
    return f"train_{train_hours}_test_{test_hours}"


def build_regime_split_slug(split_regime: str, train_hours: int, test_hours: int) -> str:
    if split_regime == FIXED_REGIME_NAME:
        return f"{split_regime}_train_{train_hours}_test_{test_hours}"
    return build_split_slug(train_hours, test_hours)


def build_env_slug(env_version: str) -> str:
    return f"env_{env_version}"


def build_window_slug(observation_window: int) -> str:
    return f"window_{observation_window}"


def infer_raw_fetch_hours(train_hours: int, test_hours: int, raw_fetch_hours: int | None = None) -> int:
    if raw_fetch_hours is not None:
        return raw_fetch_hours

    total_labeled_hours = train_hours + test_hours
    return max(DEFAULT_RAW_FETCH_HOURS, total_labeled_hours + FEATURE_WARMUP_PADDING)
