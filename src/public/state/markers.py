from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.public.state.manifest import utc_now_iso


def phase_marker_payload(
    *, phase: str, status: str, input_hashes: dict[str, str], output_artifact_paths: dict[str, str], details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": status,
        "input_hashes": dict(input_hashes),
        "output_artifact_paths": dict(output_artifact_paths),
        "timestamp": utc_now_iso(),
        "details": dict(details or {}),
    }


def write_phase_marker(run_dir: Path, phase: str, payload: dict[str, Any]) -> Path:
    path = run_dir / "phases" / f"{phase}.done.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
