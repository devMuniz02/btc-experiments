from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PRIVATE_PATTERNS = (
    "models/dev",
    "models/prod",
    "scalers",
    "checkpoint",
    "checkpoints",
    "weights",
    "private",
    ".env",
    "secret",
    "secrets",
)
SECRET_KEYS = ("token", "secret", "password", "api_key", "apikey", "credential")
PUBLIC_CONTEXTS = ("public", "report_only", "public_report", "ci")
ARTIFACT_TOKEN_ENV_VAR = "HF_ARTIFACT_TOKEN"
LEGACY_CANDIDATE_TYPES = {
    "legacy_display_only",
    "legacy_archive_count_only",
    "legacy_archive_metadata_only",
    "legacy_archive_unreadable",
}
ELIGIBILITY_FLAGS = (
    "decision_eligible",
    "top_k_eligible",
    "production_eligible",
    "what_worked_eligible",
    "test_audit_eligible",
)


def _normalized_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").lower()


def assert_public_safe_artifact(path: Path | str) -> None:
    text = str(path).replace("\\", "/").lower()
    matches = [pattern for pattern in PRIVATE_PATTERNS if pattern in text]
    if matches:
        raise PermissionError(f"Artifact path is not public-safe: {path}")


def is_public_context(context: str) -> bool:
    return str(context).strip().lower() in PUBLIC_CONTEXTS


def assert_can_load_artifact(path: Path | str, *, context: str = "private") -> None:
    if is_public_context(context):
        assert_public_safe_artifact(path)


def require_env_token(token_name: str = ARTIFACT_TOKEN_ENV_VAR) -> str:
    token = os.environ.get(token_name, "")
    if not token:
        raise PermissionError(
            f"Missing required environment token: {token_name}. In PowerShell set it with: `$Env:{token_name} = '...'`"
        )
    return token


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if any(secret_key in str(key).lower() for secret_key in SECRET_KEYS):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def assert_no_serialized_secrets(payload: dict[str, Any]) -> None:
    redacted = redact_secrets(payload)
    if redacted != payload:
        raise ValueError("Deployment metadata contains serializable secret fields.")


def assert_modern_candidate_eligible(
    candidate: dict[str, Any],
    *,
    require_production: bool = False,
    require_test_audit: bool = False,
    target: str = "deployment",
) -> None:
    candidate_type = str(candidate.get("candidate_type") or "")
    candidate_source = str(candidate.get("candidate_source") or "")
    model_id = str(candidate.get("model_id") or candidate.get("candidate_id") or "unknown")
    if candidate_type in LEGACY_CANDIDATE_TYPES or candidate_source.startswith("historical_"):
        raise PermissionError(f"Legacy/display-only candidate cannot be used for {target}: {model_id}")
    required_flags = ["decision_eligible", "top_k_eligible"]
    if require_production:
        required_flags.append("production_eligible")
    if require_test_audit:
        required_flags.append("test_audit_eligible")
    failed = [flag for flag in required_flags if candidate.get(flag) is False]
    if failed:
        raise PermissionError(f"Candidate is not eligible for {target}: {model_id} ({', '.join(failed)})")
