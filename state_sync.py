from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from automation.update_readme import update_readme
from src.utils import QuantStreamPaths, ensure_quant_stream_layout


def active_model_ids(paths: QuantStreamPaths, variation_name: str) -> set[str]:
    variation_dir = paths.models_dev_dir / variation_name
    if not variation_dir.exists():
        return set()
    return {path.name for path in variation_dir.iterdir() if path.is_dir()}


def sync_global_results(paths: QuantStreamPaths, variation_name: str, *, check: bool) -> list[str]:
    messages: list[str] = []
    variation_id = variation_name.removeprefix("var_")
    results_path = paths.global_results_path(variation_id)
    if not results_path.exists():
        return messages
    results = pd.read_parquet(results_path)
    changed = False
    if "actual" in results.columns and "target" not in results.columns:
        messages.append(f"{variation_name}: rename actual column to target")
        if not check:
            results = results.rename(columns={"actual": "target"})
            changed = True
    if "actual" in results.columns and "target" in results.columns:
        messages.append(f"{variation_name}: drop duplicate actual column")
        if not check:
            results = results.drop(columns=["actual"])
            changed = True
    active = active_model_ids(paths, variation_name)
    stale_columns = []
    for column in results.columns:
        if column.endswith("_pred") or column.endswith("_prob"):
            model_id = column.rsplit("_", 1)[0]
            if model_id not in active:
                stale_columns.append(column)
    if stale_columns:
        messages.append(f"{variation_name}: stale result columns: {', '.join(stale_columns)}")
        if not check:
            results = results.drop(columns=stale_columns)
            changed = True
    if changed and not check:
        results.to_parquet(results_path, index=False)
    for model_id in sorted(active):
        if f"{model_id}_pred" not in results.columns or f"{model_id}_prob" not in results.columns:
            messages.append(f"{variation_name}: model folder lacks result columns: {model_id}")
    return messages


def referenced_scalers(paths: QuantStreamPaths) -> set[str]:
    references: set[str] = set()
    for metadata_path in paths.models_dev_dir.glob("var_*/*/hyperparameters.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scaler_ref = str(payload.get("scaler_path") or payload.get("hyperparameters", {}).get("scaler_path") or "")
        if scaler_ref:
            references.add(Path(scaler_ref).name)
    return references


def sync_scalers(paths: QuantStreamPaths, *, check: bool) -> list[str]:
    messages: list[str] = []
    references = referenced_scalers(paths)
    for scaler_path in paths.scalers_dir.glob("**/*.pkl"):
        if scaler_path.name not in references:
            messages.append(f"unreferenced scaler: {scaler_path}")
            if not check:
                scaler_path.unlink()
    return messages


def flush_tracking_buffers(paths: QuantStreamPaths, *, check: bool) -> list[str]:
    config_path = paths.automation_dir / "config.yaml"
    if not config_path.exists():
        return []
    text = config_path.read_text(encoding="utf-8").lower()
    if "mlflow_enabled: true" not in text and "azure_enabled: true" not in text:
        return []
    messages: list[str] = []
    for buffer_path in paths.models_dev_dir.glob("var_*/*/tracking_buffer.json"):
        if buffer_path.stat().st_size > 0:
            messages.append(f"pending tracking buffer: {buffer_path}")
            if not check:
                buffer_path.write_text("", encoding="utf-8")
    return messages


def run_sync(root: Path | None = None, *, check: bool = False) -> list[str]:
    paths = QuantStreamPaths(root=root or QuantStreamPaths().root)
    ensure_quant_stream_layout(paths)
    messages: list[str] = []
    for variation_dir in sorted(paths.models_dev_dir.glob("var_*")):
        if variation_dir.is_dir():
            messages.extend(sync_global_results(paths, variation_dir.name, check=check))
    messages.extend(sync_scalers(paths, check=check))
    messages.extend(flush_tracking_buffers(paths, check=check))
    update_readme(paths.root)
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant-Stream filesystem state synchronizer.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    messages = run_sync(root=Path(args.root).resolve() if args.root else None, check=bool(args.check))
    print(json.dumps({"status": "ok", "check": bool(args.check), "messages": messages}, indent=2))
    return 0 if not messages or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
