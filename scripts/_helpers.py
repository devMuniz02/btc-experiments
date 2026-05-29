from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from src.btc_direction_learning.dataset import DirectionDatasetBundle


def load_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    else:
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
    return json.loads(text)


def write_json_atomic(path: str | Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    last_error: PermissionError | None = None
    for attempt in range(30):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.25 * (attempt + 1), 2.0))
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    if last_error is not None:
        raise last_error


def list_experiment_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def split_timestamps(bundle: DirectionDatasetBundle, split_name: str) -> list[str]:
    return bundle.get_split_timestamps(split_name)


def build_summary_contract(
    bundle: DirectionDatasetBundle,
    *,
    data_variant: str,
    dataset_source_path: str | Path,
    split_mode: str,
    source_variant: str | None = None,
) -> dict[str, Any]:
    return bundle.get_split_contract(
        data_variant=data_variant,
        dataset_source_path=str(Path(dataset_source_path).resolve()),
        split_mode=split_mode,
        source_variant=source_variant,
    )
