from __future__ import annotations

import copy
from typing import Any


def active_version(model: dict[str, Any] | None) -> dict[str, Any] | None:
    versions = list((model or {}).get("versions") or [])
    active = [version for version in versions if not version.get("deactivated_at_utc")]
    return active[-1] if active else (versions[-1] if versions else None)


def update_model_version(
    *,
    previous: dict[str, Any] | None,
    public_id: str,
    rank: int,
    train: dict[str, Any],
    validation: dict[str, Any],
    result: dict[str, Any] | None,
    generated_at_utc: str,
    trained_through_utc: str,
    activate_new_version: bool,
    refresh_active_metrics: bool,
    activated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Append an immutable version or refresh metrics within the active version's own window."""
    prior = copy.deepcopy(previous or {})
    versions = list(prior.get("versions") or [])
    current = active_version(prior)
    if current is None:
        activate_new_version = True

    if activate_new_version:
        if current is not None:
            current["deactivated_at_utc"] = generated_at_utc
        version_number = max((int(version.get("version", 0)) for version in versions), default=0) + 1
        activation_time = activated_at_utc or generated_at_utc
        current = {
            "version": version_number,
            "activated_at_utc": activation_time,
            "deactivated_at_utc": None,
            "trained_through_utc": trained_through_utc,
            "train": copy.deepcopy(train),
            "validation": copy.deepcopy(validation),
            "production_metrics": copy.deepcopy((result or {}).get("metrics") or {}),
            "prediction_series": copy.deepcopy((result or {}).get("prediction_series") or []),
        }
        versions.append(current)
    elif current is not None and refresh_active_metrics and result is not None:
        if activated_at_utc and not current.get("production_metrics") and not current.get("prediction_series"):
            current["activated_at_utc"] = activated_at_utc
        current["production_metrics"] = copy.deepcopy(result.get("metrics") or {})
        current["prediction_series"] = copy.deepcopy(result.get("prediction_series") or [])

    current = active_version({"versions": versions}) or {}
    return {
        "rank": rank,
        "public_id": public_id,
        "current_version": int(current.get("version", 1)),
        "train": copy.deepcopy(current.get("train") or train),
        "validation": copy.deepcopy(current.get("validation") or validation),
        "delayed_metrics": copy.deepcopy(current.get("production_metrics") or {}),
        "prediction_series": copy.deepcopy(current.get("prediction_series") or []),
        "versions": versions,
    }
