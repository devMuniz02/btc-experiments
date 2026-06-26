from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from src.workflows.hf_state import HfSettings, HfStateClient, LIVE_DATA_ROOT, MARKETS_ROOT, REQUEST_ROOT
from src.workflows.update_pages import update_pages

MARKET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
LOCAL_OUTPUT_ROOTS = ("experiments", "prod", "privateexperiments")
LOCAL_CACHE_ROOTS = (REQUEST_ROOT, MARKETS_ROOT, "models", LIVE_DATA_ROOT, "experiment_state")
HF_OUTPUT_PREFIXES = (REQUEST_ROOT, MARKETS_ROOT, "models", LIVE_DATA_ROOT, "experiment_state")


def validate_market_id(market: str) -> str:
    normalized = market.strip()
    if not normalized or not MARKET_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Market must be a simple id containing only letters, numbers, dot, underscore, or dash.")
    return normalized


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _market_child(root: Path, market: str) -> Path:
    return root / market


def local_output_paths(root: Path, *, cache_root: Path, market: str | None = None) -> list[Path]:
    resolved_cache = cache_root if cache_root.is_absolute() else root / cache_root
    if market:
        market = validate_market_id(market)
        paths = [_market_child(root / name, market) for name in LOCAL_OUTPUT_ROOTS]
        paths.extend(_market_child(resolved_cache / name, market) for name in LOCAL_CACHE_ROOTS)
        paths.append(resolved_cache / REQUEST_ROOT / "config.yaml")
        paths.append(resolved_cache / REQUEST_ROOT / "configs")
        return paths
    paths = [root / name for name in LOCAL_OUTPUT_ROOTS]
    paths.extend(resolved_cache / name for name in LOCAL_CACHE_ROOTS)
    return paths


def _allowed_local_roots(root: Path, cache_root: Path) -> list[Path]:
    resolved_cache = cache_root if cache_root.is_absolute() else root / cache_root
    return [root / name for name in LOCAL_OUTPUT_ROOTS] + [resolved_cache / name for name in LOCAL_CACHE_ROOTS]


def delete_local_outputs(root: Path, *, cache_root: Path, paths: list[Path]) -> list[str]:
    deleted: list[str] = []
    allowed = [path.resolve() for path in _allowed_local_roots(root, cache_root)]
    for path in paths:
        resolved = path.resolve()
        if not any(resolved == base or _is_within(resolved, base) for base in allowed):
            raise RuntimeError(f"Refusing to delete path outside generated output roots: {path}")
        if not resolved.exists():
            continue
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        deleted.append(str(resolved))
    return deleted


def hf_output_paths(client: Any, *, market: str | None = None) -> list[str]:
    normalized_market = validate_market_id(market) if market else None
    paths: list[str] = []
    for hf_path in client.list_files(""):
        normalized = hf_path.strip("/")
        if not normalized:
            continue
        if normalized_market is None:
            if any(normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in HF_OUTPUT_PREFIXES):
                paths.append(normalized)
            continue
        if normalized == f"{REQUEST_ROOT}/config.yaml":
            paths.append(normalized)
        elif normalized.startswith(f"{REQUEST_ROOT}/configs/"):
            paths.append(normalized)
        elif normalized.startswith(f"{REQUEST_ROOT}/phase") and normalized_market in Path(normalized).name:
            paths.append(normalized)
        elif normalized == f"{MARKETS_ROOT}/{normalized_market}.yaml":
            paths.append(normalized)
        elif normalized.startswith(f"{MARKETS_ROOT}/{normalized_market}/"):
            paths.append(normalized)
        elif normalized.startswith(f"models/{normalized_market}/"):
            paths.append(normalized)
        elif normalized.startswith(f"{LIVE_DATA_ROOT}/{normalized_market}/"):
            paths.append(normalized)
        elif normalized.startswith(f"experiment_state/{normalized_market}/"):
            paths.append(normalized)
    return sorted(set(paths))


def delete_hf_outputs(client: Any, *, paths: list[str], market: str | None = None) -> list[str]:
    label = market or "all generated outputs"
    if not paths:
        return []
    if hasattr(client, "delete_files"):
        client.delete_files(paths, commit_message=f"Reset {label}: delete generated outputs")
        return list(paths)
    deleted: list[str] = []
    for hf_path in paths:
        client.delete_file(hf_path, commit_message=f"Reset {label}: delete {hf_path}")
        deleted.append(hf_path)
    return deleted


def reset_all_outputs(
    *,
    root: Path,
    cache_root: Path,
    client: Any | None,
    dry_run: bool,
    force: bool,
    market: str | None = None,
    local_only: bool = False,
    hf_only: bool = False,
) -> dict[str, Any]:
    if force == dry_run:
        raise ValueError("Choose exactly one of --dry-run or --force.")
    if local_only and hf_only:
        raise ValueError("--local-only and --hf-only cannot be used together.")
    if market:
        market = validate_market_id(market)

    local_paths = [] if hf_only else local_output_paths(root, cache_root=cache_root, market=market)
    hf_paths: list[str] = []
    if not local_only:
        if client is None:
            raise RuntimeError("HF credentials are required unless --local-only is set.")
        hf_paths = hf_output_paths(client, market=market)

    result: dict[str, Any] = {
        "market": market or "all",
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
        result["deleted_local_paths"] = delete_local_outputs(root, cache_root=cache_root, paths=local_paths)
        result["regenerated_pages"] = [str(path) for path in update_pages(root)]
    if not local_only and client is not None:
        result["deleted_hf_paths"] = delete_hf_outputs(client, paths=hf_paths, market=market)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete generated experiment/prod outputs and HF workflow state from previous runs."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--cache-root", default="private_hf_cache")
    parser.add_argument("--market", default="", help="Optional market id. Omit to reset all generated outputs.")
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
    client = None
    if not args.local_only:
        client = HfStateClient(
            HfSettings.from_env_or_args(repo_id=str(args.repo_id), token=str(args.token), repo_type=str(args.repo_type))
        )
    result = reset_all_outputs(
        root=Path(args.root).resolve(),
        cache_root=Path(args.cache_root),
        client=client,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        market=str(args.market).strip() or None,
        local_only=bool(args.local_only),
        hf_only=bool(args.hf_only),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
