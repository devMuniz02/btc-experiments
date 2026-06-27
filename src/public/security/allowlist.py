from __future__ import annotations

from pathlib import PurePosixPath


ALLOWLIST_PREFIXES = (
    "README.md",
    "requirements.txt",
    "environment.yaml",
    "privatesetup.ps1",
    "privatesetup.sh",
    ".gitignore",
    ".github/workflows/",
    "config/config.example.yaml",
    "src/public/",
    "src/workflows/",
    "docs/",
    "tests/",
)
FORBIDDEN_PARTS = (
    "src/private/",
    "privateexperiments/",
    ".env",
    "token",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
)


def is_public_git_allowed(path: str) -> bool:
    normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
    lowered = normalized.lower()
    if any(part in lowered for part in FORBIDDEN_PARTS):
        return False
    if normalized in ALLOWLIST_PREFIXES:
        return True
    if any(normalized.startswith(prefix) for prefix in ALLOWLIST_PREFIXES if prefix.endswith("/")):
        return True
    if normalized.startswith("experiments/") and (
        "public" in lowered or normalized.endswith("/experiment-report.md") or "anonymized" in lowered
    ):
        return True
    if normalized.startswith("prod/") and ("public" in lowered or "delayed" in lowered):
        return True
    return False
