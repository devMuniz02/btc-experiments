from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _value(explicit: str, env_name: str) -> str:
    return (explicit or os.environ.get(env_name, "")).strip()


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
    files = sorted(
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        if path == prefix or path.startswith(f"{prefix}/")
    )
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
        downloaded = hf_hub_download(repo_id=repo_id, repo_type=repo_type, token=token, filename=hf_path)
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
