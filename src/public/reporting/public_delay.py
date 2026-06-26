from __future__ import annotations

from datetime import datetime, timedelta, timezone


def parse_utc_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def public_cutoff(latest_timestamp: str, delay_hours: int) -> datetime:
    if int(delay_hours) < 24:
        raise ValueError("public delay must be at least 24 hours")
    return parse_utc_timestamp(latest_timestamp) - timedelta(hours=int(delay_hours))


def public_delay_policy(delay_hours: int) -> dict[str, object]:
    if int(delay_hours) < 24:
        raise ValueError("public delay must be at least 24 hours")
    return {
        "delay_hours": int(delay_hours),
        "enforced": True,
        "recent_predictions_excluded": True,
        "live_signals_excluded": True,
        "policy": "public outputs exclude the latest live window by at least the configured delay",
    }
