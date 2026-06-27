from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.public.config.loader import load_config, stable_hash
from src.workflows.hf_state import (
    HfSettings,
    HfStateClient,
    LIVE_DATA_ROOT,
    MARKETS_ROOT,
    explicit_git_push,
    load_yaml,
    market_id_from_config,
    pull_hf_folder,
    workflow_lock,
)
from src.workflows.orchestrate_experiments import _live_cache_files

TOP_K_ERROR_PATTERN = re.compile(r"Production requires exactly (?P<requested>\d+) Top K models; found (?P<available>\d+)")


def _require_pipeline() -> Any:
    try:
        from src.private.training import pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Private training source is required for production runs. "
            "Run scripts/pull_private_src.py with HF_TOKEN/HF_REPO_ID before executing production."
        ) from exc
    return pipeline


def _top_k_failure(message: str) -> dict[str, int] | None:
    match = TOP_K_ERROR_PATTERN.search(str(message))
    if not match:
        return None
    return {"requested": int(match.group("requested")), "available": int(match.group("available"))}


def _prediction_is_current(path: Path, *, hours: int = 24) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = payload.get("prediction_series") or []
    timestamps = [row.get("timestamp") for row in series if isinstance(row, dict) and row.get("timestamp")]
    timestamp = max(timestamps) if timestamps else (payload.get("latest_public_window") or payload.get("generated_at_utc"))
    if not timestamp:
        return False
    try:
        value = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - value <= timedelta(hours=hours)


def _ensure_anonymized_dataset_marker(root: Path, market_id: str, config: dict[str, Any]) -> Path:
    dataset_id = "prod_ds_" + stable_hash(
        {
            "market_id": market_id,
            "symbol": config.get("market", {}).get("symbol"),
            "timeframe": config.get("market", {}).get("timeframe"),
            "single_dataset_variation": True,
        }
    )[:16]
    path = root / "prod" / market_id / "data" / f"{dataset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "market_id": market_id,
        "single_dataset_variation": True,
        "anonymized": True,
        "raw_symbol_saved": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        payload["previous_dataset_id"] = previous.get("dataset_id", dataset_id)
        payload["append_policy"] = "append_only_new_rows_since_last_saved_point"
        payload["resume_from"] = previous.get("last_saved_point")
    else:
        payload["append_policy"] = "initial_full_history_fetch"
        payload["resume_from"] = None
    payload["last_saved_point"] = payload["updated_at_utc"]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_prod_workflow(
    *,
    root: Path,
    cache_root: Path,
    repo_id: str,
    token: str,
    repo_type: str,
    push_github: bool,
    github_branch: str | None,
    push_hf: bool,
    max_markets: int,
) -> dict[str, Any]:
    cache_root = cache_root if cache_root.is_absolute() else root / cache_root
    with workflow_lock(root, "prod"):
        client = HfStateClient(HfSettings.from_env_or_args(repo_id=repo_id, token=token, repo_type=repo_type))
        if cache_root.exists():
            shutil.rmtree(cache_root)
        pulled = pull_hf_folder(client, prefix=MARKETS_ROOT, local_dir=cache_root)
        pulled_live_cache = pull_hf_folder(client, prefix=LIVE_DATA_ROOT, local_dir=cache_root)
        market_configs = sorted((cache_root / MARKETS_ROOT).glob("*.yaml"), key=lambda path: path.name.lower())
        results: list[dict[str, Any]] = []
        for config_path in market_configs[:max_markets]:
            config = load_yaml(config_path)
            market_id = market_id_from_config(config)
            local_config = root / "config" / f"_prod_{config_path.name}"
            local_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, local_config)
            load_config(local_config)
            prediction_path = root / "prod" / market_id / "results" / "production_public.json"
            if _prediction_is_current(prediction_path, hours=24):
                results.append({"market_id": market_id, "status": "skipped_current_prediction"})
                continue
            dataset_marker = _ensure_anonymized_dataset_marker(root, market_id, config)
            pipeline = _require_pipeline()
            try:
                manifest = pipeline.run_pipeline(
                    pipeline.PipelineOptions(
                        request=local_config,
                        root=root,
                        market=market_id,
                        phase="prod",
                        push_hf=push_hf,
                        push_public=push_github,
                    )
                )
            except RuntimeError as exc:
                top_k_failure = _top_k_failure(str(exc))
                if top_k_failure is None:
                    raise
                results.append(
                    {
                        "market_id": market_id,
                        "status": "blocked_missing_top_k",
                        "reason": str(exc),
                        "top_k_requested": top_k_failure["requested"],
                        "top_k_available": top_k_failure["available"],
                        "dataset_marker": str(dataset_marker),
                    }
                )
                continue
            live_cache_files = _live_cache_files(cache_root, market_id) if push_hf else []
            live_cache_push = {
                "status": "queued" if live_cache_files else "skipped",
                "reason": "" if push_hf else "push_hf flag not set",
                "paths": [remote for _, remote in live_cache_files],
            }
            github_push = explicit_git_push(
                root=root,
                paths=[f"prod/{market_id}"],
                message=f"Update anonymized production artifacts for {market_id}",
                enabled=push_github,
                target_branch=github_branch,
            )
            hf_results: list[tuple[Path, str]] = [(local_config, f"{MARKETS_ROOT}/{market_id}.yaml")]
            if prediction_path.exists():
                hf_results.append((prediction_path, f"{MARKETS_ROOT}/{market_id}/results/production_public.json"))
            private_state = root / "privateexperiments" / market_id / "results" / "production_state_private.json"
            if private_state.exists():
                hf_results.append(
                    (private_state, f"{MARKETS_ROOT}/{market_id}/results/production_state_private.json")
                )
            if dataset_marker.exists():
                hf_results.append((dataset_marker, f"{MARKETS_ROOT}/{market_id}/data/{dataset_marker.name}"))
            hf_results.extend(live_cache_files)
            client.commit_files(hf_results, commit_message=f"Update production state for {market_id}")
            results.append(
                {
                    "market_id": market_id,
                    "status": manifest.get("status", "completed"),
                    "dataset_marker": str(dataset_marker),
                    "github_push": github_push,
                    "live_cache_push": live_cache_push,
                    "hf_paths": [remote for _, remote in hf_results],
                }
            )
        return {
            "status": "completed",
            "pulled_market_files": len(pulled),
            "pulled_live_cache": len(pulled_live_cache),
            "processed": len(results),
            "results": results,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HF-backed production workflow with local/Actions parity.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--cache-root", default="private_hf_cache")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--repo-type", default=os.environ.get("HF_REPO_TYPE", "model"), choices=["model", "dataset", "space"])
    parser.add_argument("--push-github", action="store_true")
    parser.add_argument("--github-branch", default=os.environ.get("GITHUB_ARTIFACT_BRANCH", ""))
    parser.add_argument("--push-hf", action="store_true")
    parser.add_argument("--max-markets", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_prod_workflow(
        root=Path(args.root).resolve(),
        cache_root=Path(args.cache_root),
        repo_id=str(args.repo_id),
        token=str(args.token),
        repo_type=str(args.repo_type),
        push_github=bool(args.push_github),
        github_branch=str(args.github_branch or ""),
        push_hf=bool(args.push_hf),
        max_markets=int(args.max_markets),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
