from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _download_file(
    *,
    api: object,
    repo_id: str,
    repo_type: str,
    token: str,
    hf_path: str,
    local_path: Path,
) -> dict[str, object]:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        filename=hf_path,
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(downloaded).read_bytes())
    return {"hf_path": hf_path, "local_path": str(local_path), "status": "downloaded"}


def _download_first_available(
    *,
    repo_id: str,
    repo_type: str,
    token: str,
    candidates: list[str],
    local_path: Path,
    optional: bool,
) -> dict[str, object]:
    errors: list[str] = []
    for hf_path in candidates:
        if not hf_path:
            continue
        try:
            return _download_file(
                api=None,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                hf_path=hf_path,
                local_path=local_path,
            )
        except Exception as exc:
            errors.append(f"{hf_path}: {exc}")
    if optional:
        return {"status": "missing_optional", "local_path": str(local_path), "attempted": candidates, "errors": errors}
    raise RuntimeError(f"Unable to download required HF file. Attempted {candidates}. Errors: {errors}")


def _pull_prod_bundle(*, repo_id: str, repo_type: str, token: str, market_id: str, output_dir: Path) -> list[dict[str, object]]:
    files = [
        "metadata.json", "scaler_metadata.json", "production_bundle.json", "model_card_private.md",
        "recipe_private.json",
    ]
    results: list[dict[str, object]] = []
    for name in files:
        results.append(
            _download_first_available(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                candidates=[f"models/{market_id}/{name}", name],
                local_path=output_dir / market_id / name,
                optional=True,
            )
        )
    root_bundle = output_dir / market_id / "production_bundle.json"
    if root_bundle.exists():
        bundle = json.loads(root_bundle.read_text(encoding="utf-8"))
        for name in (str(bundle.get("model") or ""), str(bundle.get("scaler") or "")):
            if name:
                results.append(
                    _download_first_available(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                        candidates=[f"models/{market_id}/{name}", name],
                        local_path=output_dir / market_id / name,
                        optional=False,
                    )
                )

    index_path = output_dir / market_id / "top_models_private.json"
    index_result = _download_first_available(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        candidates=[f"models/{market_id}/top_models_private.json"],
        local_path=index_path,
        optional=True,
    )
    results.append(index_result)
    if index_result.get("status") != "downloaded":
        return results
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for model in index.get("models") or []:
        directory = str(model.get("directory") or "").strip("/")
        if not directory:
            continue
        local_dir = output_dir / market_id / directory
        for name in files:
            results.append(
                _download_first_available(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    candidates=[f"models/{market_id}/{directory}/{name}"],
                    local_path=local_dir / name,
                    optional=name in {"model_card_private.md", "recipe_private.json"},
                )
            )
        for version in model.get("versions") or []:
            version_directory = str(version.get("directory") or "").strip("/")
            if not version_directory:
                continue
            version_local_dir = output_dir / market_id / version_directory
            for name in files:
                results.append(
                    _download_first_available(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                        candidates=[f"models/{market_id}/{version_directory}/{name}"],
                        local_path=version_local_dir / name,
                        optional=name in {"model_card_private.md", "recipe_private.json"},
                    )
                )
            version_bundle = version_local_dir / "production_bundle.json"
            if version_bundle.exists():
                payload = json.loads(version_bundle.read_text(encoding="utf-8"))
                for name in (str(payload.get("model") or ""), str(payload.get("scaler") or "")):
                    if name:
                        results.append(
                            _download_first_available(
                                repo_id=repo_id,
                                repo_type=repo_type,
                                token=token,
                                candidates=[f"models/{market_id}/{version_directory}/{name}"],
                                local_path=version_local_dir / name,
                                optional=False,
                            )
                        )
        bundle_path = local_dir / "production_bundle.json"
        if not bundle_path.exists():
            continue
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        for name in (str(bundle.get("model") or ""), str(bundle.get("scaler") or "")):
            if name:
                results.append(
                    _download_first_available(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                        candidates=[f"models/{market_id}/{directory}/{name}"],
                        local_path=local_dir / name,
                        optional=False,
                    )
                )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull private HF configs/artifacts for Actions or local runs.")
    parser.add_argument("--repo-id", default="", help="Env fallback: HF_REPO_ID")
    parser.add_argument("--token", default="", help="Env fallback: HF_TOKEN")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--hf-path", default="", help="Primary HF file path to download.")
    parser.add_argument("--fallback-hf-path", default="", help="Fallback HF file path if --hf-path is missing.")
    parser.add_argument("--local-path", default="", help="Local destination for downloaded file.")
    parser.add_argument("--market", default="btc_1h")
    parser.add_argument("--pull-prod-bundle", action="store_true")
    parser.add_argument("--bundle-output-dir", default="private_hf_cache")
    parser.add_argument("--optional", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = str(args.token or _env("HF_TOKEN")).strip()
    repo_id = str(args.repo_id or _env("HF_REPO_ID")).strip()
    if not token or not repo_id:
        print(json.dumps({"status": "blocked", "required": {"HF_TOKEN": bool(token), "HF_REPO_ID": bool(repo_id)}}, indent=2))
        return 2

    results: list[dict[str, object]] = []
    if args.hf_path and args.local_path:
        results.append(
            _download_first_available(
                repo_id=repo_id,
                repo_type=str(args.repo_type),
                token=token,
                candidates=[str(args.hf_path), str(args.fallback_hf_path)],
                local_path=Path(args.local_path),
                optional=bool(args.optional),
            )
        )
    if args.pull_prod_bundle:
        results.extend(
            _pull_prod_bundle(
                repo_id=repo_id,
                repo_type=str(args.repo_type),
                token=token,
                market_id=str(args.market),
                output_dir=Path(args.bundle_output_dir),
            )
        )
    if not results:
        results.append({"status": "ready", "reason": "HF_TOKEN and HF_REPO_ID are set; no download requested"})
    print(json.dumps({"status": "completed", "repo_id": repo_id, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
