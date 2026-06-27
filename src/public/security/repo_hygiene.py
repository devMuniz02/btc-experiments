from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


FORBIDDEN_TRACKED_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".feather",
    ".h5",
    ".hdf5",
    ".joblib",
    ".onnx",
    ".parquet",
    ".pem",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
)

FORBIDDEN_TRACKED_PARTS = (
    ".env",
    "artifacts/",
    "checkpoints/",
    "datasets/",
    "features_cache/",
    "live_predictions/",
    "logs_private/",
    "metrics_private/",
    "mlruns/",
    "private_artifacts/",
    "private_configs/",
    "private_predictions/",
    "privateexperiments/",
    "processed_data/",
    "raw_data/",
    "recent_predictions/",
    "runs/",
    "secrets/",
    "src/private/",
    "tensorboard/",
    "wandb/",
)

ALLOWED_TRACKED_PATHS = {
    "data/prod/.gitkeep",
    "data/var_1/.gitkeep",
    "models/dev/var_1/.gitkeep",
    "models/prod/model_slot_1/.gitkeep",
    "models/prod/model_slot_2/.gitkeep",
    "models/prod/model_slot_3/.gitkeep",
    "models/prod/model_slot_4/.gitkeep",
    "models/prod/model_slot_5/.gitkeep",
    "scalers/prod/.gitkeep",
    "scalers/var_1/.gitkeep",
}


@dataclass(frozen=True)
class HygieneFinding:
    path: str
    reason: str


def normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _matches_forbidden_part(lower_path: str, forbidden_part: str) -> bool:
    if forbidden_part.endswith("/"):
        return lower_path.startswith(forbidden_part) or f"/{forbidden_part}" in lower_path
    return forbidden_part in lower_path


def find_tracked_hygiene_violations(paths: list[str]) -> list[HygieneFinding]:
    findings: list[HygieneFinding] = []
    for raw_path in paths:
        path = normalize_repo_path(raw_path)
        lower_path = path.lower()
        if path in ALLOWED_TRACKED_PATHS:
            continue
        if lower_path.endswith(FORBIDDEN_TRACKED_SUFFIXES):
            findings.append(HygieneFinding(path, "forbidden private/heavy tracked suffix"))
            continue
        for forbidden_part in FORBIDDEN_TRACKED_PARTS:
            if _matches_forbidden_part(lower_path, forbidden_part):
                findings.append(HygieneFinding(path, f"forbidden tracked path pattern: {forbidden_part}"))
                break
    return findings


def git_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def git_status_candidate_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        if status == " D":
            continue
        raw_path = line[3:].strip()
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1].strip()
        paths.append(raw_path)
    return paths


def audit_tracked_files(root: Path) -> dict[str, object]:
    tracked = git_tracked_files(root)
    candidates = sorted(set(tracked + git_status_candidate_files(root)))
    findings = find_tracked_hygiene_violations(candidates)
    return {
        "status": "ok" if not findings else "failed",
        "candidate_file_count": len(candidates),
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check public repo hygiene for tracked private/heavy artifacts.")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_tracked_files(Path(args.root).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
