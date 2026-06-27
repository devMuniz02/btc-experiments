from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _value(explicit: str, env_name: str) -> str:
    return (explicit or os.environ.get(env_name, "")).strip()


def _retry_after_seconds(exc: Exception, default: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw:
        try:
            return max(float(raw), default)
        except ValueError:
            return default
    return default


def _with_hf_retries(action, *, attempts: int = 4, base_sleep: float = 20.0):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:  # huggingface_hub wraps 429s in version-specific exception classes.
            message = str(exc)
            if "429" not in message and "Too Many Requests" not in message:
                raise
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(_retry_after_seconds(exc, base_sleep * attempt))
    assert last_error is not None
    raise last_error


def _list_private_source_files(api: Any, *, repo_id: str, repo_type: str, prefix: str) -> list[str]:
    def list_prefix() -> list[str]:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=prefix,
            recursive=True,
            expand=False,
        )
        paths = [str(getattr(entry, "path", "")).strip("/") for entry in entries]
        return sorted(path for path in paths if path and (path == prefix or path.startswith(f"{prefix}/")))

    return _with_hf_retries(list_prefix)


def pull_private_src(
    *,
    root: Path,
    repo_id: str,
    token: str,
    repo_type: str,
    prefix: str,
    required: bool,
    clean: bool,
) -> dict[str, Any]:
    repo_id = repo_id.strip()
    token = token.strip()
    prefix = prefix.strip("/")
    if not repo_id or not token:
        if required:
            raise RuntimeError("HF_TOKEN and HF_REPO_ID are required to pull private source.")
        return {"status": "skipped", "reason": "missing HF_TOKEN or HF_REPO_ID", "prefix": prefix}

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = _list_private_source_files(api, repo_id=repo_id, repo_type=repo_type, prefix=prefix)
    if not files:
        if required:
            raise RuntimeError(f"No private source files found in HF repo under {prefix!r}.")
        return {"status": "skipped", "reason": "no matching files", "prefix": prefix}

    target_root = (root / prefix).resolve()
    root_resolved = root.resolve()
    if root_resolved not in (target_root, *target_root.parents):
        raise RuntimeError(f"Refusing to hydrate private source outside repository root: {target_root}")
    if clean and target_root.exists():
        shutil.rmtree(target_root)

    written: list[str] = []
    for hf_path in files:
        if hf_path.endswith("/"):
            continue
        downloaded = _with_hf_retries(
            lambda hf_path=hf_path: hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                filename=hf_path,
            )
        )
        local_path = (root / hf_path).resolve()
        if root_resolved not in (local_path, *local_path.parents):
            raise RuntimeError(f"Refusing to write outside repository root: {hf_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(Path(downloaded).read_bytes())
        written.append(local_path.relative_to(root).as_posix())

    return {"status": "pulled", "repo_id": repo_id, "repo_type": repo_type, "prefix": prefix, "clean": clean, "files": written}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull private source files from a private Hugging Face repo.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--repo-type", default=os.environ.get("HF_REPO_TYPE", "model"), choices=["model", "dataset", "space"])
    parser.add_argument("--prefix", default=os.environ.get("HF_PRIVATE_SRC_PREFIX", "src/private"))
    parser.add_argument("--required", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = pull_private_src(
        root=Path(args.root).resolve(),
        repo_id=_value(str(args.repo_id), "HF_REPO_ID"),
        token=_value(str(args.token), "HF_TOKEN"),
        repo_type=str(args.repo_type),
        prefix=str(args.prefix),
        required=bool(args.required),
        clean=bool(args.clean),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
