from __future__ import annotations

from pathlib import Path


def repo_root_from(path: str | Path) -> Path:
    current = Path(path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "FINALTEMPLATE.MD").exists():
            return candidate
    return current
