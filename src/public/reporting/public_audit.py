from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.public.evaluation.sanitizer import public_safety_violations
from src.public.security.repo_hygiene import audit_tracked_files


PUBLIC_SCAN_PREFIXES = (
    "README.md",
    "docs",
    "experiments",
    "prod",
)

PUBLIC_SCAN_SUFFIXES = (
    ".json",
    ".html",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
)


def _is_public_scan_target(root: Path, path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in PUBLIC_SCAN_SUFFIXES:
        return False
    relative = path.relative_to(root).as_posix()
    if any(part.startswith(".") for part in Path(relative).parts):
        return False
    return any(relative == prefix or relative.startswith(f"{prefix}/") for prefix in PUBLIC_SCAN_PREFIXES)


def scan_public_files(root: Path) -> dict[str, Any]:
    scanned: list[str] = []
    findings: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not _is_public_scan_target(root, path):
            continue
        relative = path.relative_to(root).as_posix()
        scanned.append(relative)
        text = path.read_text(encoding="utf-8", errors="replace")
        violations = public_safety_violations(text)
        if violations:
            findings.append({"path": relative, "violations": violations})
    return {
        "status": "ok" if not findings else "failed",
        "scanned_file_count": len(scanned),
        "scanned_files": scanned,
        "finding_count": len(findings),
        "findings": findings,
    }


def run_public_audit(root: Path | str = ".") -> dict[str, Any]:
    resolved = Path(root).resolve()
    hygiene = audit_tracked_files(resolved)
    safety = scan_public_files(resolved)
    return {
        "status": "ok" if hygiene["status"] == "ok" and safety["status"] == "ok" else "failed",
        "repo_hygiene": hygiene,
        "public_safety": safety,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public-safe repository audits.")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_public_audit(Path(args.root).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
