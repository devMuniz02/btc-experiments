from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.public.evaluation.sanitizer import public_safety_violations


@dataclass(frozen=True)
class ArtifactRegistryEntry:
    model_id: str
    public_id: str
    market_id: str
    backend: str
    artifact_path: str
    export_format: str
    public_safe: bool = False
    production_promotion: bool = False
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.public_safe:
            violations = public_safety_violations(str(payload))
            if violations:
                raise PermissionError(f"Public registry entry is unsafe: {', '.join(violations)}")
        return payload


def registry_entry(**kwargs: Any) -> dict[str, Any]:
    return ArtifactRegistryEntry(**kwargs).as_dict()
