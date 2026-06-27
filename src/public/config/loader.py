from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.public.config.schema import validate_config

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal local runtimes.
    yaml = None


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip()
            current = {}
            root[key] = current
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            root[key.strip()] = _parse_scalar(value)
            current = None
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        current[key.strip()] = _parse_scalar(value)
    return root


def load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml is not None else _simple_yaml_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    validate_config(payload)
    return payload, stable_hash(payload)
