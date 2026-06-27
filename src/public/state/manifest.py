from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunManifest:
    market_id: str
    request_hash: str
    dataset_fingerprint: str
    phases: dict[str, str]
    artifacts: dict[str, str] = field(default_factory=dict)
    public_delay: dict[str, Any] = field(default_factory=dict)
    provider_used: str = ""
    selection_policy: str = "validation_only"
    test_policy: str = "frozen_winner_only"
    created_timestamp: str = field(default_factory=utc_now_iso)
    updated_timestamp: str = field(default_factory=utc_now_iso)

    def as_dict(self) -> dict[str, Any]:
        self.updated_timestamp = utc_now_iso()
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
