from __future__ import annotations

from typing import Any


def _eligible(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("status") == "worked"
        and candidate.get("passed_leakage_checks", True)
        and candidate.get("passed_privacy_checks", True)
        and candidate.get("decision_eligible", True)
    )


def validation_quality_key(
    candidate: dict[str, Any],
    primary_metric: str = "direction_accuracy",
) -> tuple[Any, ...]:
    """Return a descending-quality tuple that never consults test metrics."""
    validation = candidate.get("validation") or candidate.get("validation_metrics") or {}
    diagnostics = candidate.get("overfit_diagnostics") or {}
    secondary = "balanced_accuracy" if primary_metric == "direction_accuracy" else "direction_accuracy"
    return (
        float(validation.get(primary_metric, 0.0)),
        float(validation.get(secondary, 0.0)),
        float(validation.get("mcc", 0.0)),
        float(validation.get("f1", 0.0)),
        float(validation.get("stability_score", 0.0)),
        -abs(float(diagnostics.get("train_valid_gap", 0.0))),
        -float(candidate.get("complexity_score", 0.0)),
        -float(candidate.get("runtime_seconds", 0.0)),
    )


def rank_validation_candidates(candidates: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    eligible = [candidate for candidate in candidates if _eligible(candidate)]
    return sorted(
        eligible,
        key=lambda candidate: (
            tuple(-value for value in validation_quality_key(candidate, metric)),
            str(candidate.get("candidate_id", "")),
        ),
    )


def deterministic_discovery_rank(
    candidates: list[dict[str, Any]],
    metric: str = "direction_accuracy",
) -> list[dict[str, Any]]:
    """Rank without consulting test metrics; deterministic IDs are the final tie-breaker."""
    eligible = [candidate for candidate in candidates if _eligible(candidate)]

    def key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        return (
            *(-value for value in validation_quality_key(candidate, metric)),
            str(candidate.get("candidate_id", "")),
        )

    return sorted(eligible, key=key)
