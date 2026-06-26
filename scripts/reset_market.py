from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from src.workflows.hf_state import (
    HfSettings,
    HfStateClient,
    LIVE_DATA_ROOT,
    MARKETS_ROOT,
    REQUEST_ROOT,
    load_yaml,
    market_id_from_config,
)
from src.workflows.update_pages import update_pages

MARKET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_market_id(market: str) -> str:
    normalized = market.strip()
    if not normalized or not MARKET_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Market must be a simple id containing only letters, numbers, dot, underscore, or dash.")
    return normalized


def local_reset_paths(root: Path, *, market: str, cache_root: Path) -> list[Path]:
    cache_root = cache_root if cache_root.is_absolute() else root / cache_root
    return [
        root / "experiments" / market,
        root / "prod" / market,
        root / "privateexperiments" / market,
        cache_root / LIVE_DATA_ROOT / market,
        cache_root / "experiment_state" / market,
    ]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _delete_local_paths(root: Path, paths: list[Path]) -> list[str]:
    deleted: list[str] = []
    allowed_roots = [
        root / "experiments",
        root / "prod",
        root / "privateexperiments",
        root / "private_hf_cache" / LIVE_DATA_ROOT,
        root / "private_hf_cache" / "experiment_state",
    ]
    for path in paths:
        resolved = path.resolve()
        if not any(
            _is_within(resolved, allowed.resolve()) or resolved == allowed.resolve() for allowed in allowed_roots
        ):
            raise RuntimeError(f"Refusing to delete path outside reset roots: {path}")
        if resolved.exists():
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            deleted.append(str(resolved))
    return deleted


def _request_config_matches(client: Any, *, hf_path: str, market: str, scratch_root: Path) -> bool:
    scratch_path = scratch_root / hf_path
    try:
        client.download_file(hf_path, scratch_path)
        return market_id_from_config(load_yaml(scratch_path)) == market
    except Exception:
        return market in Path(hf_path).name


def hf_reset_paths(client: Any, *, market: str, scratch_root: Path) -> list[str]:
    paths: set[str] = set()
    for hf_path in client.list_files(""):
        normalized = hf_path.strip("/")
        if normalized == f"{MARKETS_ROOT}/{market}.yaml":
            paths.add(normalized)
        elif normalized.startswith(f"{MARKETS_ROOT}/{market}/"):
            paths.add(normalized)
        elif normalized.startswith(f"models/{market}/"):
            paths.add(normalized)
        elif normalized.startswith(f"{LIVE_DATA_ROOT}/{market}/"):
            paths.add(normalized)
        elif normalized.startswith(f"experiment_state/{market}/"):
            paths.add(normalized)
        elif (
            normalized.startswith(f"{REQUEST_ROOT}/phase")
            and normalized.endswith(".yaml")
            and market in Path(normalized).name
        ):
            paths.add(normalized)
        elif normalized == f"{REQUEST_ROOT}/config.yaml" or (
            normalized.startswith(f"{REQUEST_ROOT}/configs/") and normalized.endswith((".yaml", ".yml"))
        ):
            if _request_config_matches(client, hf_path=normalized, market=market, scratch_root=scratch_root):
                paths.add(normalized)
    return sorted(paths)


def delete_hf_paths(client: Any, *, market: str, paths: list[str]) -> list[str]:
    if not paths:
        return []
    if hasattr(client, "delete_files"):
        client.delete_files(paths, commit_message=f"Reset {market}: delete generated outputs")
        return list(paths)
    deleted: list[str] = []
    for hf_path in paths:
        client.delete_file(hf_path, commit_message=f"Reset {market}: delete {hf_path}")
        deleted.append(hf_path)
    return deleted


def reset_market(
    *,
    root: Path,
    cache_root: Path,
    market: str,
    client: Any | None,
    dry_run: bool,
    force: bool,
    local_only: bool = False,
    hf_only: bool = False,
) -> dict[str, Any]:
    market = _validate_market_id(market)
    if force == dry_run:
        raise ValueError("Choose exactly one of --dry-run or --force.")

    local_paths = [] if hf_only else local_reset_paths(root, market=market, cache_root=cache_root)
    hf_paths: list[str] = []
    if not local_only:
        if client is None:
            raise RuntimeError("HF credentials are required unless --local-only is set.")
        scratch_root = (cache_root if cache_root.is_absolute() else root / cache_root) / "_reset_scan"
        hf_paths = hf_reset_paths(client, market=market, scratch_root=scratch_root)

    result: dict[str, Any] = {
        "market": market,
        "dry_run": dry_run,
        "force": force,
        "local_paths": [str(path) for path in local_paths],
        "hf_paths": hf_paths,
        "deleted_local_paths": [],
        "deleted_hf_paths": [],
        "regenerated_pages": [],
    }
    if dry_run:
        return result

    if not hf_only:
        result["deleted_local_paths"] = _delete_local_paths(root, local_paths)
        result["regenerated_pages"] = [str(path) for path in update_pages(root)]
    if not local_only and client is not None:
        result["deleted_hf_paths"] = delete_hf_paths(client, market=market, paths=hf_paths)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete all local and HF state for one market so it can start fresh.")
    parser.add_argument("--market", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--cache-root", default="private_hf_cache")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--repo-type", default=os.environ.get("HF_REPO_TYPE", "model"), choices=["model", "dataset", "space"]
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--hf-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.local_only and args.hf_only:
        raise SystemExit("--local-only and --hf-only cannot be used together.")
    client = None
    if not args.local_only:
        client = HfStateClient(
            HfSettings.from_env_or_args(repo_id=str(args.repo_id), token=str(args.token), repo_type=str(args.repo_type))
        )
    result = reset_market(
        root=Path(args.root).resolve(),
        cache_root=Path(args.cache_root),
        market=str(args.market),
        client=client,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        local_only=bool(args.local_only),
        hf_only=bool(args.hf_only),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
