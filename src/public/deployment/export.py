from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.public.evaluation.sanitizer import assert_public_safe_text


class ExportFormat(str, Enum):
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    JSON_BUNDLE = "json_bundle"


@dataclass(frozen=True)
class ExportMetadata:
    model_id: str
    model_family: str
    market_id: str
    export_path: str
    export_format: str
    public_safe: bool
    contains_weights: bool
    contains_scaler: bool
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_plan(
    *,
    model_id: str,
    model_family: str,
    market_id: str,
    export_path: Path,
    export_format: str = "json_bundle",
    public_safe: bool = False,
    contains_weights: bool = True,
    contains_scaler: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if public_safe and (contains_weights or contains_scaler):
        raise PermissionError("Public-safe export plan cannot contain weights or scalers.")
    payload = ExportMetadata(
        model_id=model_id,
        model_family=model_family,
        market_id=market_id,
        export_path=export_path.as_posix(),
        export_format=ExportFormat(export_format).value,
        public_safe=bool(public_safe),
        contains_weights=bool(contains_weights),
        contains_scaler=bool(contains_scaler),
        metadata=dict(metadata or {}),
    ).as_dict()
    if public_safe:
        assert_public_safe_text(str(payload))
    payload["export_executed"] = False
    payload["private_hf_required"] = bool(contains_weights or contains_scaler)
    payload["runtime_requirements"] = {
        "torchscript": ["torch"],
        "onnx": ["onnx"],
        "safetensors": ["safetensors"],
        "json_bundle": [],
    }[payload["export_format"]]
    return payload
