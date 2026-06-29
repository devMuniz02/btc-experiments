from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SUPPORTED_MARKET_TYPES = {"crypto", "stocks"}
SUPPORTED_TIMEFRAMES = {"5min", "15min", "1h", "4h", "1d"}
EXHAUSTIVE_PHASES = [f"phase{index:02d}" for index in range(1, 17)]
SUPPORTED_RANKING_METRICS = {
    "accuracy",
    "auc",
    "balanced_accuracy",
    "cumulative_return",
    "direction_accuracy",
    "f1",
    "mcc",
    "precision",
    "precision_up",
    "recall",
    "recall_up",
    "sharpe",
    "trade_count",
    "trading_proxy_score",
    "weighted_score",
}
SUPPORTED_PHASE_INHERITANCE = {"fixed", "unfixed"}
MIN_PRODUCTION_START_UTC = datetime(2026, 6, 1, tzinfo=timezone.utc)
REQUIRED_TOP_LEVEL = (
    "project",
    "current_phase",
    "market",
    "fetch",
    "schema",
    "split",
    "features",
    "experiments",
    "production",
    "reporting",
)


def _normalize_phase(value: Any) -> str:
    text = "none" if value is None else str(value).strip()
    if text in {"none", "done"}:
        return text
    if text.startswith("phase"):
        text = text.replace("phase", "", 1)
    if not text.isdigit():
        raise ValueError("current_phase must be none, done, or a non-negative phase number")
    return text


def _validate_model_length_consistency(config: dict[str, Any]) -> None:
    split = config["split"]
    experiments = config["experiments"]
    train_length = int(split["train_length"])
    sequence_lengths = [int(value) for value in experiments.get("sequence_lengths", [])]

    for model in experiments.get("models") or []:
        if not isinstance(model, dict):
            continue
        model_id = model.get("model_id", model.get("id", "<unknown>"))
        if "train_length" in model and int(model["train_length"]) != train_length:
            raise ValueError(f"Model {model_id} train_length must match split.train_length")
        if "sequence_length" in model and [int(model["sequence_length"])] != sequence_lengths:
            raise ValueError(f"Model {model_id} sequence_length must match experiments.sequence_lengths")
        if "sequence_lengths" in model and [int(value) for value in model["sequence_lengths"]] != sequence_lengths:
            raise ValueError(f"Model {model_id} sequence_lengths must match experiments.sequence_lengths")

    overrides = experiments.get("model_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("experiments.model_overrides must be a mapping when provided")
    for model_id, override in overrides.items():
        if not isinstance(override, dict):
            continue
        if "train_length" in override and int(override["train_length"]) != train_length:
            raise ValueError(f"Model {model_id} train_length must match split.train_length")
        if "sequence_length" in override and [int(override["sequence_length"])] != sequence_lengths:
            raise ValueError(f"Model {model_id} sequence_length must match experiments.sequence_lengths")
        if (
            "sequence_lengths" in override
            and [int(value) for value in override["sequence_lengths"]] != sequence_lengths
        ):
            raise ValueError(f"Model {model_id} sequence_lengths must match experiments.sequence_lengths")


def validate_config(config: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in config]
    if missing:
        raise ValueError(f"Config missing top-level sections: {', '.join(missing)}")
    _normalize_phase(config["current_phase"])
    market = config["market"]
    if market.get("market_type") not in SUPPORTED_MARKET_TYPES:
        raise ValueError("market.market_type must be crypto or stocks")
    if market.get("timeframe") not in SUPPORTED_TIMEFRAMES:
        raise ValueError("market.timeframe must be one of 5min, 15min, 1h, 4h, 1d")
    if not str(market.get("market_id", "")).strip():
        raise ValueError("market.market_id is required")
    split = config["split"]
    for key in ("train_length", "validation_length", "test_length"):
        if int(split.get(key, 0)) <= 0:
            raise ValueError(f"split.{key} must be positive")
    sequence_lengths = list(config["experiments"].get("sequence_lengths") or [])
    if not sequence_lengths:
        raise ValueError("experiments.sequence_lengths must not be empty")
    smallest_split = min(int(split["train_length"]), int(split["validation_length"]), int(split["test_length"]))
    if max(map(int, sequence_lengths)) >= smallest_split:
        raise ValueError("All sequence lengths must be smaller than each split size")
    _validate_model_length_consistency(config)
    profile = str(config["experiments"].get("workflow_profile") or "legacy_v1")
    if profile not in {"legacy_v1", "exhaustive_v1"}:
        raise ValueError("experiments.workflow_profile must be legacy_v1 or exhaustive_v1")
    if profile == "exhaustive_v1":
        phases = [str(value) for value in config["experiments"].get("phases") or []]
        if phases != EXHAUSTIVE_PHASES:
            raise ValueError("exhaustive_v1 requires experiments.phases phase01 through phase16 in order")
        ranking_metric = str(config["experiments"].get("ranking_metric") or "direction_accuracy")
        if ranking_metric not in SUPPORTED_RANKING_METRICS:
            supported = ", ".join(sorted(SUPPORTED_RANKING_METRICS))
            raise ValueError(f"experiments.ranking_metric must be one of: {supported}")
        inheritance = str(config["experiments"].get("phase5_to_phase6_inheritance") or "unfixed")
        if inheritance not in SUPPORTED_PHASE_INHERITANCE:
            raise ValueError("experiments.phase5_to_phase6_inheritance must be fixed or unfixed")
        if min(int(split[key]) for key in ("train_length", "validation_length", "test_length")) <= 240:
            raise ValueError("exhaustive_v1 requires every split length to exceed the maximum 240-step lookback")
        normalized_phase = _normalize_phase(config["current_phase"])
        if normalized_phase not in {"none", "done"} and not 1 <= int(normalized_phase) <= 16:
            raise ValueError("exhaustive_v1 current_phase must be none, done, or 1 through 16")
    delay = int(config.get("project", {}).get("public_delay_hours", 24))
    if delay < 24:
        raise ValueError("project.public_delay_hours must be at least 24")
    experiment_end = datetime.fromisoformat(str(config["experiments"].get("test_end_utc", "")).replace("Z", "+00:00"))
    production_start = datetime.fromisoformat(str(config["production"].get("start_utc", "")).replace("Z", "+00:00"))
    if experiment_end.tzinfo is None or production_start.tzinfo is None:
        raise ValueError("experiments.test_end_utc and production.start_utc must include a UTC offset")
    if experiment_end.astimezone(timezone.utc) >= production_start.astimezone(timezone.utc):
        raise ValueError("experiments.test_end_utc must be before production.start_utc")
    if production_start.astimezone(timezone.utc) < MIN_PRODUCTION_START_UTC:
        raise ValueError("production.start_utc must be on or after 2026-06-01T00:00:00+00:00")
    top_k = int(config["production"].get("top_k", 3))
    if top_k < 1 or top_k > 3:
        raise ValueError("production.top_k must be between 1 and 3")
    if str(config["production"].get("drift_trigger", "critical")) != "critical":
        raise ValueError("production.drift_trigger must be critical")
