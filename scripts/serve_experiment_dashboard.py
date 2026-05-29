from __future__ import annotations

import argparse
import functools
import html
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.experiment_support import read_summary_rows_parquet

DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "EXPERIMENTS"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
BASELINE_MODELS = {"RANDOM", "ALWAYS_UP", "ALWAYS_DOWN"}
CATALOG_FILE_NAME = "dashboard_catalog.json"
SNAPSHOT_FILE_NAME = "dashboard_snapshot.html"
EXPERIMENT_16_ID = 16


class ArtifactNotBuiltError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a localhost dashboard for comparing direction-learning experiments.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    return parser.parse_args()


def safe_load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    else:
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
    return json.loads(text)


def load_variant_summary(variant_payload: dict[str, Any]) -> dict[str, Any]:
    summary_payload = variant_payload.get("summary", {})
    summary_path_raw = variant_payload.get("summary_path")
    if not summary_path_raw:
        if isinstance(summary_payload, dict) and isinstance(summary_payload.get("models"), list):
            return summary_payload
        return summary_payload if isinstance(summary_payload, dict) else {}
    summary = safe_load_json(Path(str(summary_path_raw)))
    if not isinstance(summary, dict):
        if isinstance(summary_payload, dict) and isinstance(summary_payload.get("models"), list):
            return summary_payload
        return {}
    if isinstance(summary.get("models"), list):
        return summary
    models_path_raw = variant_payload.get("models_path") or summary.get("models_path")
    if models_path_raw:
        summary["models"] = read_summary_rows_parquet(models_path_raw)
    else:
        summary["models"] = []
    return summary


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def normalize_curve(split_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": maybe_float(split_data.get("accuracy")),
        "total_reward": maybe_float(split_data.get("total_reward")),
        "mean_reward": maybe_float(split_data.get("mean_reward")),
        "accuracy_scored_count": maybe_int(split_data.get("accuracy_scored_count")),
        "threshold_action_count": maybe_int(split_data.get("threshold_action_count")),
        "threshold_correct_count": maybe_int(split_data.get("threshold_correct_count")),
        "cumulative_rewards": split_data.get("cumulative_rewards", []),
        "actions": split_data.get("actions", []),
        "labels": split_data.get("labels", []),
        "chosen_action_probabilities": split_data.get("chosen_action_probabilities", []),
        "cumulative_accuracy": split_data.get("cumulative_accuracy", []),
        "rolling24_accuracy": split_data.get("rolling24_accuracy", {}),
        "timestamps": split_data.get("timestamps", []),
    }


def normalize_portfolio(model_data: dict[str, Any]) -> dict[str, Any]:
    portfolio_data = model_data.get("portfolio", {})
    if not isinstance(portfolio_data, dict):
        portfolio_data = {}
    
    portfolios = {}
    for portfolio_name in ["fixed_dollar", "peak_fraction", "current_fraction"]:
        portfolio = portfolio_data.get(portfolio_name, {})
        if not isinstance(portfolio, dict):
            portfolio = {}
        portfolios[portfolio_name] = {
            "balances": portfolio.get("balances", []),
            "starting_balance": maybe_float(portfolio.get("starting_balance")),
            "final_balance": maybe_float(portfolio.get("final_balance")),
            "pnl": maybe_float(portfolio.get("pnl")),
            "pnl_pct": maybe_float(portfolio.get("pnl_pct")),
            "zero_crossing": portfolio.get("zero_crossing", {}),
        }
    
    return portfolios


def normalize_model(model_data: dict[str, Any]) -> dict[str, Any]:
    name = str(model_data.get("name", "UNKNOWN"))
    train = model_data.get("train", {})
    test = model_data.get("test", {})
    if not isinstance(train, dict):
        train = {}
    if not isinstance(test, dict):
        test = {}
    return {
        "name": name,
        "env_version": model_data.get("env_version"),
        "family": model_data.get("family"),
        "model_variation": model_data.get("model_variation"),
        "time_variant": model_data.get("time_variant"),
        "threshold_pct": maybe_int(model_data.get("threshold_pct")),
        "train_length": model_data.get("train_length"),
        "window_length": model_data.get("window_length"),
        "is_baseline": name in BASELINE_MODELS,
        "window_size": model_data.get("window_size"),
        "series_window_scale": model_data.get("series_window_scale"),
        "is_aggregate_series": bool(model_data.get("is_aggregate_series")),
        "train": normalize_curve(train),
        "test": normalize_curve(test),
        "portfolio": normalize_portfolio(model_data),
    }


def maybe_int(value: Any) -> int | None:
    numeric = maybe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def model_daily_net_wins(model_data: dict[str, Any], source_test_window: dict[str, Any] | None) -> float | None:
    portfolio = model_data.get("portfolio", {})
    if not isinstance(portfolio, dict):
        return None
    fixed_dollar = portfolio.get("fixed_dollar", {})
    if not isinstance(fixed_dollar, dict):
        return None
    trade_pnls = fixed_dollar.get("trade_pnls", [])
    if not isinstance(trade_pnls, list):
        trade_pnls = []
    wins = sum(1 for pnl in trade_pnls if maybe_float(pnl) is not None and float(pnl) > 0)
    losses = sum(1 for pnl in trade_pnls if maybe_float(pnl) is not None and float(pnl) < 0)
    net_wins = wins - losses
    count = maybe_int((source_test_window or {}).get("count"))
    if count is not None and count > 0:
        active_days = count / 24.0
        return net_wins / active_days if active_days > 0 else 0.0
    total_trades = len(trade_pnls)
    return (net_wins * 24.0 / total_trades) if total_trades > 0 else 0.0


def build_fixed_5k_runs_payload(manifest_paths: dict[str, Path]) -> dict[str, Any]:
    grouped_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for experiment_id, manifest_path in manifest_paths.items():
        manifest = safe_load_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        experiment_name = str(manifest.get("name", f"Experiment {experiment_id}"))
        variants = manifest.get("variants", {})
        if not isinstance(variants, dict):
            continue
        for variant_name, variant_payload in variants.items():
            if not isinstance(variant_payload, dict):
                continue
            summary = load_variant_summary(variant_payload)
            summary_path_raw = variant_payload.get("summary_path")
            if (
                summary_path_raw
                and (
                    "source_test_window" not in summary
                    or "split_mode" not in summary
                    or not isinstance(summary.get("models"), list)
                    or not summary.get("models")
                )
            ):
                try:
                    loaded_summary = load_variant_summary(variant_payload)
                except Exception:
                    loaded_summary = None
                if isinstance(loaded_summary, dict):
                    summary = loaded_summary
            split_mode = str(summary.get("split_mode", ""))
            source_test_window = summary.get("source_test_window", {})
            if not isinstance(source_test_window, dict):
                continue
            start = source_test_window.get("start")
            end = source_test_window.get("end")
            count = maybe_int(source_test_window.get("count"))
            if not start or not end or count != 5000:
                continue
            if "fixed" not in split_mode:
                continue
            models = summary.get("models", [])
            if not isinstance(models, list):
                continue
            key = (str(start), str(end), int(count))
            rows = grouped_rows.setdefault(key, [])
            for model in models:
                if not isinstance(model, dict):
                    continue
                model_name = str(model.get("name", "")).strip()
                if not model_name:
                    continue
                test = model.get("test", {})
                if not isinstance(test, dict):
                    test = {}
                accuracy = maybe_float(test.get("accuracy"))
                daily_net_wins = model_daily_net_wins(model, source_test_window)
                fixed_dollar = model.get("portfolio", {}).get("fixed_dollar", {}) if isinstance(model.get("portfolio"), dict) else {}
                pnl_pct = maybe_float(fixed_dollar.get("pnl_pct"))
                rows.append(
                    {
                        "experiment_id": maybe_int(manifest.get("experiment_id")) or maybe_int(experiment_id) or 0,
                        "experiment_name": experiment_name,
                        "variant": str(variant_name),
                        "model_name": model_name,
                        "accuracy": accuracy,
                        "daily_net_wins": daily_net_wins,
                        "pnl_pct": pnl_pct,
                        "window_start": str(start),
                        "window_end": str(end),
                        "window_count": int(count),
                    }
                )

    if not grouped_rows:
        return {"window": None, "rows": []}

    best_key = max(grouped_rows.items(), key=lambda item: (len(item[1]), item[0][0], item[0][1]))[0]
    rows = [row for row in grouped_rows[best_key] if row.get("accuracy") is not None]
    rows.sort(
        key=lambda row: (
            -(row["daily_net_wins"] if row["daily_net_wins"] is not None else float("-inf")),
            -(row["accuracy"] if row["accuracy"] is not None else float("-inf")),
            row["experiment_id"],
            row["model_name"],
        )
    )
    return {
        "window": {
            "start": best_key[0],
            "end": best_key[1],
            "count": best_key[2],
        },
        "rows": rows,
    }


def is_aggregate_series_model_name(model_name: str) -> bool:
    return str(model_name or "").endswith("_AGGREGATE")


def include_model_in_dashboard_lists(experiment_id: int, model_entry: dict[str, Any] | None) -> bool:
    if not isinstance(model_entry, dict):
        return False
    model_name = str(model_entry.get("name", ""))
    if not model_name:
        return False
    if int(experiment_id) != EXPERIMENT_16_ID:
        return True
    if bool(model_entry.get("is_aggregate_series")):
        return True
    return is_aggregate_series_model_name(model_name)


def normalize_variant_summary(summary: dict[str, Any]) -> dict[str, Any]:
    plotting = summary.get("plotting", {})
    if not isinstance(plotting, dict):
        plotting = {}
    models = summary.get("models", [])
    if not isinstance(models, list):
        models = []
    normalized_models = [normalize_model(model) for model in models if isinstance(model, dict)]
    return {
        "variant": summary.get("variant"),
        "source_variant": summary.get("source_variant"),
        "env_version": summary.get("env_version"),
        "training_data_mode": summary.get("training_data_mode"),
        "force_none_outside_market_hours": bool(summary.get("force_none_outside_market_hours")),
        "dataset_metadata": summary.get("dataset_metadata", {}),
        "plotting": {
            "x_axis_mode": plotting.get("x_axis_mode"),
            "train_timestamps": plotting.get("train_timestamps", []),
            "test_timestamps": plotting.get("test_timestamps", []),
        },
        "models": normalized_models,
    }


def build_manifest_variant_catalog(variant_name: str, variant_payload: dict[str, Any]) -> dict[str, Any]:
    summary = variant_payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    models = summary.get("models", [])
    if not isinstance(models, list):
        models = []
    model_catalog = []
    for model in models:
        if not isinstance(model, dict) or not model.get("name"):
            continue
        model_name = str(model["name"])
        model_catalog.append(
            {
                "name": model_name,
                "is_baseline": model_name in BASELINE_MODELS,
                "family": None,
                "model_variation": None,
                "time_variant": None,
                "threshold_pct": None,
                "train_length": None,
                "window_length": None,
                "window_size": None,
                "series_window_scale": None,
                "train_accuracy": None,
                "test_accuracy": None,
                "portfolio": {},
            }
        )
    return {
        "variant": summary.get("variant", variant_name),
        "source_variant": None,
        "env_version": None,
        "training_data_mode": None,
        "force_none_outside_market_hours": False,
        "dataset_metadata": {},
        "plotting": {},
        "models": model_catalog,
    }


class ExperimentStore:
    def __init__(self, experiments_root: Path) -> None:
        self.experiments_root = experiments_root
        self._lock = threading.RLock()
        self._manifest_paths: dict[str, Path] = {}
        self._manifest_cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        self._catalog_cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        self._series_cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        self._experiment_index_cache: list[dict[str, Any]] = []
        self._payload_cache: dict[str, dict[str, Any]] = {}
        self.refresh_index()

    @staticmethod
    def _get_file_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _load_cached_json(
        self,
        path: Path,
        cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]],
    ) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
        signature = self._get_file_signature(path)
        if signature is None:
            cache.pop(path, None)
            return None, None
        cached = cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1], signature
        payload = safe_load_json(path)
        if not isinstance(payload, dict):
            cache.pop(path, None)
            return None, signature
        cache[path] = (signature, payload)
        return payload, signature

    def _load_manifest(self, manifest_path: Path) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
        return self._load_cached_json(manifest_path, self._manifest_cache)

    @staticmethod
    def _variant_dir_from_payload(variant_payload: dict[str, Any]) -> Path | None:
        summary_path_raw = variant_payload.get("summary_path")
        if not summary_path_raw:
            return None
        return Path(str(summary_path_raw)).resolve().parent

    def _catalog_path_from_payload(self, variant_payload: dict[str, Any]) -> Path | None:
        variant_dir = self._variant_dir_from_payload(variant_payload)
        if variant_dir is None:
            return None
        return variant_dir / CATALOG_FILE_NAME

    def _snapshot_path_from_payload(self, variant_payload: dict[str, Any]) -> Path | None:
        variant_dir = self._variant_dir_from_payload(variant_payload)
        if variant_dir is None:
            return None
        return variant_dir / SNAPSHOT_FILE_NAME

    def _load_dashboard_catalog(self, catalog_path: Path) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
        return self._load_cached_json(catalog_path, self._catalog_cache)

    def _load_series_artifact(self, series_path: Path) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
        return self._load_cached_json(series_path, self._series_cache)

    @staticmethod
    def _build_experiment_summary(experiment_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        variants = manifest.get("variants", {})
        if not isinstance(variants, dict):
            variants = {}
        numeric_experiment_id = int(experiment_id)
        model_names: set[str] = set()
        for variant_payload in variants.values():
            summary = load_variant_summary(variant_payload) if isinstance(variant_payload, dict) else {}
            models = summary.get("models", []) if isinstance(summary, dict) else []
            if not isinstance(models, list):
                continue
            for model in models:
                if include_model_in_dashboard_lists(numeric_experiment_id, model):
                    model_names.add(str(model["name"]))
        return {
            "experiment_id": numeric_experiment_id,
            "name": manifest.get("name", f"Experiment {experiment_id}"),
            "available_variants": list(variants.keys()),
            "available_models": sorted(model_names),
        }

    def refresh_index(self) -> None:
        with self._lock:
            manifest_paths: dict[str, Path] = {}
            experiment_index: list[dict[str, Any]] = []
            active_manifests: set[Path] = set()
            if self.experiments_root.exists():
                for manifest_path in sorted(self.experiments_root.rglob("manifest.json")):
                    if "_retrain_sources" in manifest_path.parts:
                        continue
                    try:
                        manifest, _ = self._load_manifest(manifest_path)
                    except Exception:
                        continue
                    if manifest is None or "experiment_id" not in manifest:
                        continue
                    experiment_id = str(manifest["experiment_id"])
                    manifest_paths[experiment_id] = manifest_path
                    experiment_index.append(self._build_experiment_summary(experiment_id, manifest))
                    active_manifests.add(manifest_path)
            experiment_index.sort(key=lambda item: int(item.get("experiment_id", 0)))
            self._manifest_paths = manifest_paths
            self._experiment_index_cache = experiment_index
            self._payload_cache = {
                experiment_id: payload_cache
                for experiment_id, payload_cache in self._payload_cache.items()
                if experiment_id in manifest_paths
            }
            self._manifest_cache = {
                path: cached_payload
                for path, cached_payload in self._manifest_cache.items()
                if path in active_manifests
            }

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._experiment_index_cache)

    def list_fixed_5k_runs(self) -> dict[str, Any]:
        with self._lock:
            return build_fixed_5k_runs_payload(self._manifest_paths)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._lock:
            manifest_path = self._manifest_paths.get(str(experiment_id))
            if manifest_path is None:
                return None
            manifest, manifest_signature = self._load_manifest(manifest_path)
            if manifest is None or manifest_signature is None:
                return None
            variants = manifest.get("variants", {})
            if not isinstance(variants, dict):
                variants = {}

            cached_payload = self._payload_cache.get(str(experiment_id))
            if (
                cached_payload is not None
                and cached_payload.get("manifest_signature") == manifest_signature
            ):
                return cached_payload["payload"]

            payload_variants: dict[str, Any] = {}
            model_names: set[str] = set()
            for variant_name, variant_payload in variants.items():
                if not isinstance(variant_payload, dict):
                    continue
                catalog_path = self._catalog_path_from_payload(variant_payload)
                if catalog_path is None or not catalog_path.exists():
                    raise ArtifactNotBuiltError(
                        f"Missing dashboard catalog for experiment {experiment_id} variant {variant_name}. "
                        f"Run `python -m src.utils.experiment_reporting --experiments {experiment_id}` first."
                    )
                variant_catalog, _ = self._load_dashboard_catalog(catalog_path)
                if variant_catalog is None:
                    raise ArtifactNotBuiltError(f"Failed to read dashboard catalog: {catalog_path}")
                snapshot_path = self._snapshot_path_from_payload(variant_payload)
                payload_variants[variant_name] = {
                    **variant_catalog,
                    "snapshot_url": (
                        f"/experiments/{experiment_id}/{variant_name}/snapshot"
                        if snapshot_path is not None and snapshot_path.exists()
                        else None
                    ),
                }
                for model in payload_variants[variant_name].get("models", []):
                    if include_model_in_dashboard_lists(int(manifest.get("experiment_id")), model):
                        model_names.add(str(model["name"]))

                payload_variants[variant_name]["models"] = [
                    model
                    for model in payload_variants[variant_name].get("models", [])
                    if include_model_in_dashboard_lists(int(manifest.get("experiment_id")), model)
                ]

            payload = {
                "experiment_id": int(manifest.get("experiment_id")),
                "name": manifest.get("name", f"Experiment {experiment_id}"),
                "root_dir": manifest.get("root_dir", str(manifest_path.parent)),
                "available_variants": list(payload_variants.keys()),
                "available_models": sorted(model_names),
                "variants": payload_variants,
            }
            self._payload_cache[str(experiment_id)] = {
                "manifest_signature": manifest_signature,
                "payload": payload,
            }
            return payload

    def get_experiment_series(
        self,
        experiment_id: str,
        variant_names: list[str],
        model_names: list[str],
    ) -> dict[str, Any] | None:
        with self._lock:
            manifest_path = self._manifest_paths.get(str(experiment_id))
            if manifest_path is None:
                return None
            manifest, _ = self._load_manifest(manifest_path)
            if manifest is None:
                return None
            variants = manifest.get("variants", {})
            if not isinstance(variants, dict):
                variants = {}

            allowed_variants = set(variant_names)
            allowed_models = set(model_names)
            payload_variants: dict[str, Any] = {}
            returned_models: set[str] = set()

            for variant_name, variant_payload in variants.items():
                if allowed_variants and variant_name not in allowed_variants:
                    continue
                if not isinstance(variant_payload, dict):
                    continue
                catalog_path = self._catalog_path_from_payload(variant_payload)
                if catalog_path is None or not catalog_path.exists():
                    raise ArtifactNotBuiltError(
                        f"Missing dashboard catalog for experiment {experiment_id} variant {variant_name}. "
                        f"Run `python -m src.utils.experiment_reporting --experiments {experiment_id}` first."
                    )
                variant_catalog, _ = self._load_dashboard_catalog(catalog_path)
                if variant_catalog is None:
                    raise ArtifactNotBuiltError(f"Failed to read dashboard catalog: {catalog_path}")

                plotting = variant_catalog.get("plotting", {})
                if not isinstance(plotting, dict):
                    plotting = {}
                selected_models: list[dict[str, Any]] = []
                for model_entry in variant_catalog.get("models", []):
                    if not isinstance(model_entry, dict):
                        continue
                    if not include_model_in_dashboard_lists(int(manifest.get("experiment_id")), model_entry):
                        continue
                    model_name = str(model_entry.get("name", ""))
                    if not model_name or (allowed_models and model_name not in allowed_models):
                        continue
                    series_file = model_entry.get("series_file")
                    if not series_file:
                        continue
                    series_path = catalog_path.parent / str(series_file)
                    if not series_path.exists():
                        raise ArtifactNotBuiltError(f"Missing dashboard series artifact: {series_path}")
                    series_payload, _ = self._load_series_artifact(series_path)
                    if series_payload is None or not isinstance(series_payload, dict):
                        raise ArtifactNotBuiltError(f"Failed to read dashboard series artifact: {series_path}")
                    model_payload = series_payload.get("model")
                    if not isinstance(model_payload, dict):
                        raise ArtifactNotBuiltError(f"Invalid dashboard series artifact payload: {series_path}")
                    selected_models.append(model_payload)
                    returned_models.add(model_name)

                if not selected_models:
                    continue
                payload_variants[variant_name] = {
                    "variant": variant_catalog.get("variant", variant_name),
                    "source_variant": variant_catalog.get("source_variant"),
                    "env_version": variant_catalog.get("env_version"),
                    "training_data_mode": variant_catalog.get("training_data_mode"),
                    "force_none_outside_market_hours": bool(variant_catalog.get("force_none_outside_market_hours")),
                    "dataset_metadata": variant_catalog.get("dataset_metadata", {}),
                    "plotting": plotting,
                    "models": selected_models,
                }

            return {
                "experiment_id": int(manifest.get("experiment_id")),
                "name": manifest.get("name", f"Experiment {experiment_id}"),
                "requested_variants": variant_names,
                "requested_models": model_names,
                "returned_models": sorted(returned_models),
                "variants": payload_variants,
            }


@functools.lru_cache(maxsize=1)
def build_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Direction Learning Experiment Compare</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #151b23;
      --panel-2: #1c2430;
      --panel-3: #10161f;
      --line: #2c394a;
      --ink: #e8edf5;
      --muted: #9fb0c3;
      --accent: #5ab0ff;
      --accent-2: #f6b26b;
      --good: #52d273;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    html, body {
      height: 100%;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", "Aptos", system-ui, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(90,176,255,0.16), transparent 26%),
        radial-gradient(circle at top left, rgba(246,178,107,0.12), transparent 24%),
        linear-gradient(180deg, #0b1016 0%, var(--bg) 100%);
      color: var(--ink);
      overflow: hidden;
    }
    .layout {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 20px;
      height: 100vh;
      min-height: 100vh;
      padding: 18px;
      overflow: hidden;
    }
    .card {
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-3) 100%);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }
    .sidebar {
      position: sticky;
      top: 18px;
      align-self: start;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: calc(100vh - 36px);
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 6px;
      scrollbar-gutter: stable;
      overscroll-behavior: contain;
    }
    .sidebar .card {
      padding: 16px;
    }
    h1, h2, h3 { margin: 0; }
    .subtle {
      color: var(--muted);
      line-height: 1.45;
    }
    select {
      width: 100%;
      margin-top: 10px;
      padding: 10px 12px;
      color: var(--ink);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
    }
    .section-label {
      font-size: 0.84rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
      font-weight: 800;
    }
    .toggle-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .toggle {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--ink);
      border-radius: 999px;
      padding: 7px 12px;
      cursor: pointer;
      font-weight: 700;
      font-size: 0.88rem;
    }
    .toggle.active {
      background: rgba(90,176,255,0.16);
      border-color: rgba(90,176,255,0.7);
      color: #d9efff;
    }
    .portfolio-toggle {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--ink);
      border-radius: 999px;
      padding: 7px 12px;
      cursor: pointer;
      font-weight: 700;
      font-size: 0.88rem;
    }
    .portfolio-toggle.active {
      background: rgba(246,178,107,0.16);
      border-color: rgba(246,178,107,0.7);
      color: #ffd9b0;
    }
    .quick-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .action-btn {
      border: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      border-radius: 12px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 700;
    }
    .content {
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-width: 0;
      min-height: 0;
      max-height: calc(100vh - 36px);
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 6px;
      scrollbar-gutter: stable;
      overscroll-behavior: contain;
    }
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .summary-strip .card {
      padding: 14px 16px;
    }
    .control-card {
      padding: 14px 16px;
    }
    .summary-label {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.75rem;
      margin-bottom: 8px;
      font-weight: 800;
    }
    .summary-value {
      font-size: 1.4rem;
      font-weight: 800;
    }
    .slider-row {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-top: 10px;
    }
    .slider-row input[type="range"] {
      flex: 1 1 auto;
      accent-color: var(--accent);
    }
    .slider-value {
      min-width: 56px;
      text-align: right;
      font-size: 1.05rem;
      font-weight: 800;
      color: #d9efff;
    }
    .chart-card {
      padding: 16px;
    }
    .chart-card h2 {
      font-size: 1.08rem;
      margin-bottom: 6px;
    }
    .table-wrap {
      width: 100%;
      overflow-x: auto;
      margin-top: 10px;
    }
    .compare-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
      font-size: 0.94rem;
    }
    .compare-table th,
    .compare-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    .compare-table th {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(255,255,255,0.02);
    }
    .compare-table tbody tr:hover {
      background: rgba(255,255,255,0.03);
    }
    .metric-good {
      color: var(--good);
      font-weight: 700;
    }
    .metric-bad {
      color: #ff8f8f;
      font-weight: 700;
    }
    .chart-host {
      min-height: 360px;
      width: 100%;
      overflow-x: auto;
      margin-top: 10px;
    }
    .chart-empty {
      min-height: 220px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
    }
    svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .axis {
      fill: #c7d3df;
      font-size: 12px;
    }
    .axis-line {
      stroke: #4b5a6f;
      stroke-width: 1;
    }
    .grid-line {
      stroke: #243142;
      stroke-width: 1;
      stroke-dasharray: 4 5;
    }
    .series-line {
      fill: none;
      stroke-width: 1.7;
    }
    .series-point {
      cursor: default;
    }
    .series-bar {
      cursor: default;
    }
    .legend {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .legend-item {
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: 0;
      color: var(--muted);
      font-size: 0.98rem;
    }
    .legend-swatch {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex: 0 0 auto;
    }
    .legend-label {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    code {
      background: rgba(255,255,255,0.06);
      padding: 2px 6px;
      border-radius: 6px;
    }
    @media (max-width: 1040px) {
      body {
        overflow: auto;
      }
      .layout {
        grid-template-columns: 1fr;
        height: auto;
        min-height: 100vh;
        overflow: visible;
      }
      .sidebar {
        position: static;
        max-height: none;
        overflow: visible;
        padding-right: 0;
      }
      .content {
        max-height: none;
        overflow: visible;
        padding-right: 0;
      }
      .summary-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 680px) {
      .summary-strip {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <section class="card">
        <h1>Experiment Compare</h1>
        <p class="subtle">Dark-mode localhost viewer for aligned experiment comparisons. Data loads on demand per experiment so large reward and portfolio curves stay responsive.</p>
        <div class="section-label">Experiment</div>
        <select id="experimentSelect"></select>
      </section>
      <section class="card">
        <div class="section-label">Dataset Variants</div>
        <div class="toggle-row" id="variantToggles"></div>
      </section>
        <section class="card">
          <div class="section-label">Filters</div>
          <div class="quick-actions">
            <button type="button" class="toggle" data-action="select-all-models">Select All Models</button>
            <button type="button" class="toggle" data-action="clear-models">Clear Models</button>
          </div>
          <div style="margin-top:12px;">
            <div class="summary-label">Thresholds</div>
            <div class="quick-actions" id="thresholdQuickActions"></div>
          </div>
          <div style="margin-top:12px;">
            <div class="summary-label">Train Sizes</div>
            <div class="quick-actions" id="trainSizeQuickActions"></div>
          </div>
          <div style="margin-top:12px;">
            <div class="summary-label">Strategy Groups</div>
            <div class="quick-actions" id="strategyQuickActions"></div>
          </div>
        <div style="margin-top:12px;">
          <div class="summary-label">Families</div>
          <div class="quick-actions" id="familyQuickActions"></div>
        </div>
      </section>
      <section class="card">
        <div class="section-label">Models</div>
        <div class="toggle-row" id="modelToggles" style="margin-top:12px;"></div>
      </section>
    </aside>
    <main class="content">
      <section class="summary-strip">
        <div class="card">
          <div class="summary-label">Selected Experiment</div>
          <div class="summary-value" id="summaryExperiment">-</div>
        </div>
        <div class="card">
          <div class="summary-label">Active Variants</div>
          <div class="summary-value" id="summaryVariants">0</div>
        </div>
        <div class="card">
          <div class="summary-label">Active Models</div>
          <div class="summary-value" id="summaryModels">0</div>
        </div>
        <div class="card">
          <div class="summary-label">Series on Graphs</div>
          <div class="summary-value" id="summarySeries">0</div>
        </div>
      </section>

      <section class="card chart-card">
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 8px;">
          <div>
            <h2>Comparable Fixed 5K Runs</h2>
            <div class="subtle">Runs that use the shared fixed 5,000-row test window, ordered by net wins per day.</div>
          </div>
          <button type="button" class="action-btn" id="fixed5kRunsComputeButton">Compute</button>
        </div>
        <div class="chart-host" id="fixed5kRunsTable"></div>
      </section>

      <section class="card control-card">
        <div class="summary-label">Visible X-Axis Range</div>
        <div class="subtle">Choose how much of each time-series chart to draw from the start of the run.</div>
        <div class="slider-row">
          <input type="range" id="xAxisPercentRange" min="0" max="100" step="1" value="100" />
          <div class="slider-value" id="xAxisPercentValue">100%</div>
        </div>
        <div style="margin-top: 14px;">
          <div class="summary-label">Timeline Mode</div>
          <div class="subtle">Use each series' real dates by default, or align starts when you want to compare curve shapes.</div>
          <div class="toggle-row" id="timelineModeToggles" style="margin-top:12px;"></div>
        </div>
        <div style="margin-top: 14px;">
          <div class="summary-label">Portfolio Zero-Crossing</div>
          <div class="subtle">Choose whether portfolio curves stop when they hit $0 or continue showing the full path after the threshold is crossed.</div>
          <div class="toggle-row" id="portfolioZeroCrossingToggles" style="margin-top:12px;"></div>
        </div>
        <div style="margin-top: 14px;">
          <div class="summary-label">Ensemble</div>
          <div class="subtle">Add an ENSEMBLE series that takes the majority action of the currently selected models within each selected variant.</div>
          <div class="toggle-row" id="ensembleToggles" style="margin-top:12px;"></div>
        </div>
      </section>

      <section class="card chart-card">
        <h2>Accuracy</h2>
        <div class="subtle">Overall test accuracy plus best and lowest rolling 24-point test accuracy for every selected <code>&lt;variant&gt; / &lt;model&gt;</code> pair.</div>
        <div class="chart-host" id="accuracyChart"></div>
        <div class="legend" id="accuracyLegend"></div>
      </section>

      <section class="card chart-card">
        <h2>Cumulative Accuracy Over Test</h2>
        <div class="subtle">Running test accuracy over time for every selected <code>&lt;variant&gt; / &lt;model&gt;</code> pair.</div>
        <div class="chart-host" id="cumulativeAccuracyChart"></div>
        <div class="legend" id="cumulativeAccuracyLegend"></div>
      </section>

      <section class="card chart-card">
        <h2>Threshold Actions</h2>
        <div class="subtle">Counts of above-threshold directional actions taken and how many of those actions were correct for every selected <code>&lt;variant&gt; / &lt;model&gt;</code> pair.</div>
        <div class="chart-host" id="thresholdActionChart"></div>
        <div class="legend" id="thresholdActionLegend"></div>
      </section>

      <section class="card chart-card">
        <h2>Rewards Over Train</h2>
        <div class="subtle">Cumulative train reward curves across the selected variants and models.</div>
        <div class="chart-host" id="trainRewardChart"></div>
        <div class="legend" id="trainRewardLegend"></div>
      </section>

      <section class="card chart-card">
        <h2>Rewards Over Test</h2>
        <div class="subtle">Cumulative test reward curves with dynamic monthly date ticks.</div>
        <div class="chart-host" id="testRewardChart"></div>
        <div class="legend" id="testRewardLegend"></div>
      </section>

      <section class="card chart-card">
        <h2>Test Net Wins Per Day</h2>
        <div class="subtle">Net test wins divided by the number of days covered by each test slice, so full and market-hours runs can be compared on the same daily basis.</div>
        <div class="chart-host" id="testNetWinsPerDayChart"></div>
        <div class="legend" id="testNetWinsPerDayLegend"></div>
      </section>

      <section class="card chart-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div>
            <h2>Portfolio Strategies</h2>
            <div class="subtle">Portfolio balance curves across the selected variants and models, recomputed live from the test actions and labels.</div>
          </div>
          <div class="portfolio-selector" style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;">
            <button type="button" class="portfolio-toggle active" data-portfolio-mode="fixed_dollar">Fixed Dollar</button>
            <button type="button" class="portfolio-toggle" data-portfolio-mode="current_percent">Current Percent</button>
            <button type="button" class="portfolio-toggle" data-portfolio-mode="peak_percent">Peak Percent</button>
          </div>
        </div>
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom: 10px;">
          <label style="min-width:180px; display:grid; gap:6px;">
            <span class="summary-label">Starting Balance</span>
            <input id="portfolioStartingBalanceInput" type="number" min="0" step="0.01" value="10" style="background:#111923; color:#dbe7f3; border:1px solid #334155; border-radius:10px; padding:10px 12px;" />
          </label>
          <label style="min-width:180px; display:grid; gap:6px;">
            <span class="summary-label">Increment Value</span>
            <input id="portfolioIncrementValueInput" type="number" min="0" step="0.01" value="1" style="background:#111923; color:#dbe7f3; border:1px solid #334155; border-radius:10px; padding:10px 12px;" />
          </label>
        </div>
        <div class="chart-host" id="portfolioChart"></div>
        <div class="legend" id="portfolioLegend"></div>
      </section>

      <section class="card chart-card">
        <h2>Test Outcome Matrix</h2>
        <div class="subtle">Day-by-hour test outcome grid for a single selected series. Green is correct, red is wrong, and yellow is skipped or outside allowed hours.</div>
        <div class="chart-host" id="testOutcomeMatrixChart"></div>
      </section>

      <section class="card chart-card">
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 8px;">
          <div>
            <h2>Threshold Accuracy</h2>
            <div class="subtle">Accuracy for actions whose chosen probability is at or above the selected threshold. Thresholds are limited to 50% through 100%.</div>
          </div>
          <div style="display: flex; gap: 14px; flex-wrap: wrap; min-width: min(100%, 720px);">
            <div style="min-width: 200px; flex: 1 1 200px;">
              <div class="summary-label">Start %</div>
              <div class="slider-row">
                <input type="range" id="thresholdAccuracyRangeStart" min="50" max="100" step="0.25" value="50" />
                <div class="slider-value" id="thresholdAccuracyRangeStartValue">50.00%</div>
              </div>
            </div>
            <div style="min-width: 200px; flex: 1 1 200px;">
              <div class="summary-label">End %</div>
              <div class="slider-row">
                <input type="range" id="thresholdAccuracyRangeEnd" min="50" max="100" step="0.25" value="100" />
                <div class="slider-value" id="thresholdAccuracyRangeEndValue">100.00%</div>
              </div>
            </div>
            <div style="min-width: 200px; flex: 1 1 200px;">
              <div class="summary-label">Step %</div>
              <div class="slider-row">
                <input type="range" id="thresholdAccuracyStep" min="0.25" max="10" step="0.25" value="1" />
                <div class="slider-value" id="thresholdAccuracyStepValue">1.00%</div>
              </div>
            </div>
          </div>
        </div>
        <div class="chart-host" id="thresholdAccuracyChart"></div>
        <div class="legend" id="thresholdAccuracyLegend"></div>
      </section>
    </main>
  </div>
  <script src="/app.js"></script>
</body>
</html>
"""


@functools.lru_cache(maxsize=1)
def build_app_js() -> str:
    return r"""
const BASELINE_MODELS = new Set(["RANDOM", "ALWAYS_UP", "ALWAYS_DOWN"]);
const MAX_RENDERED_SERIES = Number.POSITIVE_INFINITY;
const MAX_RENDERED_AGGREGATE_SERIES = Number.POSITIVE_INFINITY;
const DEFAULT_LEARNED_SELECTION_LIMIT = 12;
const LARGE_EXPERIMENT_MODEL_COUNT = 40;
const LARGE_EXPERIMENT_DEFAULT_SELECTION_LIMIT = 6;
const MAX_RENDER_PATH_POINTS = 1200;
const MAX_HOVER_POINTS = 180;
const FAST_CANVAS_SERIES_THRESHOLD = 24;
const FAST_CANVAS_POINT_THRESHOLD = 12000;
const SEARCH_PARAMS = new URLSearchParams(window.location.search);
const COLOR_PALETTE = [
  "#5ab0ff", "#f6b26b", "#6fe3a4", "#e98eff", "#ff7f7f", "#95a4ff",
  "#33d0c9", "#ffd166", "#7ad67a", "#ff9f68", "#9ec1ff", "#d6a4ff",
];

const state = {
  experimentIndex: [],
  fixed5kRuns: { window: null, rows: [] },
  fixed5kRunsState: "idle",
  fixed5kRunsError: "",
  experimentCache: new Map(),
  activeExperimentId: null,
  selectedVariants: new Set(),
  selectedModels: new Set(),
  activeModelFilters: {
    timeVariants: new Set(),
    thresholds: new Set(),
    trainSizes: new Set(),
    strategies: new Set(),
    families: new Set(),
  },
  userClearedModels: false,
  portfolioSizingMode: "fixed_dollar",
  portfolioStartingBalance: 10,
  portfolioIncrementValue: 1,
  xAxisPercent: 100,
  timelineMode: "actual_dates",
  showFullPortfolioAfterZero: false,
  showEnsemble: false,
  thresholdAccuracyRangeStart: 50,
  thresholdAccuracyRangeEnd: 100,
  thresholdAccuracyStep: 1,
  renderToken: 0,
  pendingInitialExperimentId: window.__DASHBOARD_INITIAL_EXPERIMENT_ID__ || SEARCH_PARAMS.get("experiment_id") || null,
  pendingInitialVariants: (
    Array.isArray(window.__DASHBOARD_INITIAL_VARIANTS__)
      ? [...window.__DASHBOARD_INITIAL_VARIANTS__]
      : SEARCH_PARAMS.getAll("variant").filter(Boolean)
  ),
};

const colorMap = new Map();
const parsedTimestampCache = new Map();
const portfolioCurveCache = new Map();

function getSeriesColor(key) {
  if (!colorMap.has(key)) {
    colorMap.set(key, COLOR_PALETTE[colorMap.size % COLOR_PALETTE.length]);
  }
  return colorMap.get(key);
}

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function createEmptyState(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "chart-empty";
  wrapper.textContent = message;
  return wrapper;
}

function formatPercentLabel(value) {
  return `${Number(value).toFixed(2)}%`;
}

function formatProbabilityBucketLabel(value) {
  return Number(value).toFixed(2).replace(/\.?0+$/, "");
}

function truncateLabelMiddle(value, maxLength = 24) {
  const text = String(value || "");
  if (text.length <= maxLength) return text;
  const available = Math.max(3, maxLength - 1);
  const leftLength = Math.ceil(available * 0.55);
  const rightLength = Math.max(3, available - leftLength);
  return `${text.slice(0, leftLength)}…${text.slice(text.length - rightLength)}`;
}

function formatYAxisTickLabel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  if (Math.abs(numeric) >= 1000) return numeric.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (Math.abs(numeric) >= 100) return numeric.toFixed(0);
  if (Math.abs(numeric) >= 10) return numeric.toFixed(1);
  return numeric.toFixed(2);
}

function createSvg(width, height) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  return svg;
}

function appendSvg(parent, tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
  parent.appendChild(node);
  return node;
}

function computeFiniteBounds(values) {
  let min = Infinity;
  let max = -Infinity;
  values.forEach((value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return;
    if (numeric < min) min = numeric;
    if (numeric > max) max = numeric;
  });
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }
  return { min, max };
}

function buildMonthlyTicks(timestamps) {
  if (!Array.isArray(timestamps) || !timestamps.length) return [];
  const ticks = [];
  let lastMonthKey = null;
  timestamps.forEach((timestamp, index) => {
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return;
    const monthKey = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
    if (monthKey === lastMonthKey) return;
    lastMonthKey = monthKey;
    ticks.push({
      x: index,
      label: date.toISOString().slice(0, 7),
    });
  });
  const lastIndex = timestamps.length - 1;
  const lastDate = new Date(timestamps[lastIndex]);
  if (!Number.isNaN(lastDate.getTime())) {
    const lastLabel = lastDate.toISOString().slice(0, 7);
    if (!ticks.length || ticks[ticks.length - 1].x !== lastIndex) {
      ticks.push({ x: lastIndex, label: lastLabel });
    }
  }
  return ticks;
}

function buildMonthlyTicksFromValues(values) {
  if (!Array.isArray(values) || !values.length) return [];
  const uniqueValues = [...new Set(
    values
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b)
  )];
  if (!uniqueValues.length) return [];
  const ticks = [];
  let lastMonthKey = null;
  uniqueValues.forEach((value) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return;
    const monthKey = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
    if (monthKey === lastMonthKey) return;
    lastMonthKey = monthKey;
    ticks.push({
      x: value,
      label: date.toISOString().slice(0, 7),
    });
  });
  const lastValue = uniqueValues[uniqueValues.length - 1];
  const lastDate = new Date(lastValue);
  if (!Number.isNaN(lastDate.getTime())) {
    const lastLabel = lastDate.toISOString().slice(0, 7);
    if (!ticks.length || ticks[ticks.length - 1].x !== lastValue) {
      ticks.push({ x: lastValue, label: lastLabel });
    }
  }
  return ticks;
}

function fitTickLabels(ticks, width, pixelsPerLabel = 110) {
  if (!ticks.length) return [];
  const maxLabels = Math.max(2, Math.floor(width / pixelsPerLabel));
  if (ticks.length <= maxLabels) return ticks;
  const stride = Math.ceil(ticks.length / maxLabels);
  const fitted = ticks.filter((_, index) => index % stride === 0);
  const last = ticks[ticks.length - 1];
  if (!fitted.length || fitted[fitted.length - 1].x !== last.x) {
    fitted.push(last);
  }
  return fitted;
}

function buildReferenceLineSpecs(referenceLines) {
  return (Array.isArray(referenceLines) ? referenceLines : [])
    .map((line, index) => {
      const value = Number(line?.value);
      if (!Number.isFinite(value)) return null;
      return {
        key: line?.key || `reference-${index}`,
        value,
        color: line?.color || "#7dd3fc",
        dasharray: line?.dasharray || "6 4",
        width: Number.isFinite(Number(line?.width)) ? Number(line.width) : 1.2,
        label: line?.label ? String(line.label) : null,
      };
    })
    .filter(Boolean);
}

function clampXAxisPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 100;
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function trimPointsByPercent(points, percent) {
  if (!Array.isArray(points) || !points.length) return [];
  const clampedPercent = clampXAxisPercent(percent);
  if (clampedPercent >= 100) return points;
  if (clampedPercent <= 0) return [];
  const visibleCount = Math.max(1, Math.ceil(points.length * (clampedPercent / 100)));
  return points.slice(0, visibleCount);
}

function trimTimestampsByPercent(timestamps, percent) {
  if (!Array.isArray(timestamps) || !timestamps.length) return [];
  const clampedPercent = clampXAxisPercent(percent);
  if (clampedPercent >= 100) return timestamps;
  if (clampedPercent <= 0) return [];
  const visibleCount = Math.max(1, Math.ceil(timestamps.length * (clampedPercent / 100)));
  return timestamps.slice(0, visibleCount);
}

function actionDirectionForEnv(action, envVersion) {
  const numeric = Number(action);
  if (!Number.isFinite(numeric)) return null;
  if (envVersion === "ternary") {
    if (numeric === 0) return 0;
    if (numeric === 1) return 1;
    return null;
  }
  if (envVersion === "intensity11") {
    if (numeric >= 0 && numeric <= 4) return 0;
    if (numeric >= 6 && numeric <= 10) return 1;
    return null;
  }
  if (numeric === 0 || numeric === 1) return numeric;
  return null;
}

function countActiveTestDays(timestamps, fallbackCount) {
  const seenDays = new Set();
  (Array.isArray(timestamps) ? timestamps : []).forEach((timestamp) => {
    const parsed = parseTimestampMs(timestamp);
    if (!Number.isFinite(parsed)) return;
    const dateKey = new Date(parsed).toISOString().slice(0, 10);
    if (dateKey) seenDays.add(dateKey);
  });
  if (seenDays.size > 0) return seenDays.size;
  const fallback = Number(fallbackCount);
  if (Number.isFinite(fallback) && fallback > 0) {
    return Math.max(1, Math.ceil(fallback / 24));
  }
  return 1;
}

function computeNetWinsPerDay(actions, labels, timestamps, envVersion) {
  const actionList = Array.isArray(actions) ? actions : [];
  const labelList = Array.isArray(labels) ? labels : [];
  let netWins = 0;
  const limit = Math.min(actionList.length, labelList.length);
  for (let index = 0; index < limit; index += 1) {
    const direction = actionDirectionForEnv(actionList[index], envVersion);
    const label = Number(labelList[index]);
    if (!Number.isFinite(label) || (label !== 0 && label !== 1)) continue;
    if (direction === null) continue;
    netWins += direction === label ? 1 : -1;
  }
  const activeDays = countActiveTestDays(timestamps, limit);
  return netWins / Math.max(1, activeDays);
}

function parseTimestampMs(timestamp) {
  if (!timestamp) return null;
  const key = String(timestamp);
  if (parsedTimestampCache.has(key)) {
    return parsedTimestampCache.get(key);
  }
  const parsed = Date.parse(key);
  const value = Number.isFinite(parsed) ? parsed : null;
  parsedTimestampCache.set(key, value);
  return value;
}

function alignTimestampsToPointCount(values, timestamps) {
  const pointCount = Array.isArray(values) ? values.length : 0;
  if (!pointCount) return [];
  const rawTimestamps = Array.isArray(timestamps) ? [...timestamps] : [];
  if (!rawTimestamps.length) return [];
  if (rawTimestamps.length === pointCount) return rawTimestamps;
  if (rawTimestamps.length === pointCount - 1) {
    return [rawTimestamps[0], ...rawTimestamps];
  }
  if (rawTimestamps.length > pointCount) {
    return rawTimestamps.slice(rawTimestamps.length - pointCount);
  }
  const padded = [...rawTimestamps];
  while (padded.length < pointCount) {
    padded.unshift(padded[0]);
  }
  return padded;
}

function buildCurvePoints(values, timestamps, mode) {
  const alignedTimestamps = alignTimestampsToPointCount(values, timestamps);
  const hasValidTimestamp = alignedTimestamps.some((timestamp) => Number.isFinite(parseTimestampMs(timestamp)));
  const points = (Array.isArray(values) ? values : []).map((value, index) => {
    const dateLabel = alignedTimestamps[index] || null;
    const timestampMs = parseTimestampMs(dateLabel);
    if (value == null || value === "") {
      return {
        x: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs) ? timestampMs : index,
        y: null,
        dateLabel,
        stepIndex: index,
        usesTimestamp: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs),
      };
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return {
        x: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs) ? timestampMs : index,
        y: null,
        dateLabel,
        stepIndex: index,
        usesTimestamp: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs),
      };
    }
    return {
      x: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs) ? timestampMs : index,
      y: numeric,
      dateLabel,
      stepIndex: index,
      usesTimestamp: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs),
    };
  }).filter(Boolean);
  if (mode === "actual_dates" && hasValidTimestamp) {
    points.sort((left, right) => {
      const leftX = Number(left.x);
      const rightX = Number(right.x);
      if (leftX !== rightX) return leftX - rightX;
      return Number(left.stepIndex ?? 0) - Number(right.stepIndex ?? 0);
    });
  }
  return points;
}

function splitDrawablePointSegments(points, yToPx) {
  const drawablePoints = (Array.isArray(points) ? points : []).filter((point) => (
    point
    && Number.isFinite(point.x)
    && Number.isFinite(point.y)
    && Number.isFinite(yToPx(point.y))
  ));
  const timestampGaps = [];
  for (let index = 1; index < drawablePoints.length; index += 1) {
    const previousPoint = drawablePoints[index - 1];
    const point = drawablePoints[index];
    if (!previousPoint?.usesTimestamp || !point?.usesTimestamp) continue;
    const gap = Number(point.x) - Number(previousPoint.x);
    if (Number.isFinite(gap) && gap > 0) {
      timestampGaps.push(gap);
    }
  }
  timestampGaps.sort((left, right) => left - right);
  const medianTimestampGap = timestampGaps.length
    ? timestampGaps[Math.floor(timestampGaps.length / 2)]
    : null;
  const largeGapThreshold = Number.isFinite(medianTimestampGap) ? (medianTimestampGap * 3) : null;

  const segments = [];
  let current = [];
  (Array.isArray(points) ? points : []).forEach((point) => {
    if (
      !point
      || !Number.isFinite(point.x)
      || !Number.isFinite(point.y)
      || !Number.isFinite(yToPx(point.y))
    ) {
      if (current.length) {
        segments.push(current);
        current = [];
      }
      return;
    }
    const previousPoint = current.length ? current[current.length - 1] : null;
    if (previousPoint && Number(point.x) <= Number(previousPoint.x)) {
      segments.push(current);
      current = [];
    }
    if (
      previousPoint
      && previousPoint.usesTimestamp
      && point.usesTimestamp
      && Number.isFinite(largeGapThreshold)
      && (Number(point.x) - Number(previousPoint.x)) > largeGapThreshold
    ) {
      segments.push(current);
      current = [];
    }
    current.push(point);
  });
  if (current.length) {
    segments.push(current);
  }
  return segments;
}

function downsamplePoints(points, maxPoints, alwaysIncludeLast = true) {
  if (!Array.isArray(points) || points.length <= maxPoints) return points;
  const limit = Math.max(2, Number(maxPoints) || 2);
  const stride = Math.ceil(points.length / limit);
  const sampled = points.filter((_, index) => index % stride === 0);
  const last = points[points.length - 1];
  if (alwaysIncludeLast && sampled[sampled.length - 1] !== last) {
    sampled.push(last);
  }
  return sampled;
}

function renderLegend(hostId, items) {
  const host = byId(hostId);
  host.innerHTML = "";
  if (!items.length) return;
  items.forEach((item) => {
    const legendLabel = item.displayLabel || item.label;
    const wrapper = document.createElement("div");
    wrapper.className = "legend-item";
    wrapper.innerHTML = `
      <span class="legend-swatch" style="background:${escapeHtml(item.color)};"></span>
      <span class="legend-label" title="${escapeHtml(legendLabel)}">${escapeHtml(legendLabel)}</span>
    `;
    host.appendChild(wrapper);
  });
}

function drawLineChart(hostId, legendId, series, yLabel, tickTimestamps, options = {}) {
  const host = byId(hostId);
  host.innerHTML = "";
  if (!series.length) {
    host.appendChild(createEmptyState(options.emptyMessage || "Select at least one model and one variant to render this chart."));
    renderLegend(legendId, []);
    return;
  }

  const width = Math.max(980, host.clientWidth || 980);
  const height = 360;
  const margin = { top: 20, right: 24, bottom: 60, left: 64 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const allPoints = [];
  series.forEach((item) => item.points.forEach((point) => allPoints.push(point)));
  const referenceLines = buildReferenceLineSpecs(options.referenceLines);
  const xBounds = computeFiniteBounds(allPoints.map((point) => point.x));
  const yBounds = computeFiniteBounds([
    ...allPoints.map((point) => point.y),
    ...referenceLines.map((line) => line.value),
  ]);
  if (!xBounds || !yBounds) {
    host.appendChild(createEmptyState("No numeric values available for this chart."));
    renderLegend(legendId, []);
    return;
  }
  const requestedYScale = String(options.yScale || "linear");
  const canUsePiecewiseLogScale = requestedYScale === "piecewise_log_over_1000" && yBounds.max > 1000;
  let yMin = yBounds.min;
  let yMax = yBounds.max;
  if (yMin === yMax && !canUsePiecewiseLogScale) {
    yMin -= 1;
    yMax += 1;
  }
  const xRange = (xBounds.max - xBounds.min) || 1;
  const xToPx = (value) => margin.left + ((value - xBounds.min) / xRange) * plotWidth;
  let yToPx;
  let yTickValues;
  let axisLabelText = yLabel;
  let chartReferenceLines = referenceLines;
  if (canUsePiecewiseLogScale) {
    const logThreshold = 1000;
    const piecewiseTransform = (value) => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return null;
      if (numeric <= logThreshold) return numeric;
      return logThreshold + (logThreshold * Math.log10(numeric / logThreshold));
    };
    const transformedYMin = piecewiseTransform(yMin);
    const transformedYMax = piecewiseTransform(yMax);
    const transformedRange = ((transformedYMax ?? 0) - (transformedYMin ?? 0)) || 1;
    yToPx = (value) => {
      const transformed = piecewiseTransform(value);
      if (!Number.isFinite(transformed)) return null;
      return margin.top + plotHeight - ((transformed - transformedYMin) / transformedRange) * plotHeight;
    };
    const tickValues = [];
    const linearUpperBound = Math.min(logThreshold, yMax);
    if (linearUpperBound > yMin) {
      tickValues.push(yMin);
      tickValues.push(yMin + ((linearUpperBound - yMin) * 0.5));
      tickValues.push(linearUpperBound);
    } else {
      tickValues.push(yMin);
    }
    if (yMax > logThreshold) {
      const logTickCandidates = [1000, 2000, 5000];
      const maxExponent = Math.ceil(Math.log10(yMax));
      for (let exponent = 4; exponent <= maxExponent; exponent += 1) {
        logTickCandidates.push(10 ** exponent);
        logTickCandidates.push(2 * (10 ** exponent));
        logTickCandidates.push(5 * (10 ** exponent));
      }
      logTickCandidates.forEach((candidate) => {
        if (candidate > logThreshold && candidate < yMax) tickValues.push(candidate);
      });
      tickValues.push(yMax);
    }
    yTickValues = [...new Set(
      tickValues
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value))
    )].sort((a, b) => a - b);
    axisLabelText = `${yLabel} (linear <= 1k, log > 1k)`;
  } else {
    const yRange = (yMax - yMin) || 1;
    yToPx = (value) => margin.top + plotHeight - ((value - yMin) / yRange) * plotHeight;
    yTickValues = Array.from({ length: 5 }, (_, index) => yMin + (1 - (index / 4)) * yRange);
  }

  const svg = createSvg(width, height);
  appendSvg(svg, "rect", {
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
    fill: "#0f1620",
    stroke: "#334155",
    rx: 14,
  });

  yTickValues.forEach((yValue) => {
    const y = yToPx(yValue);
    if (!Number.isFinite(y)) return;
    appendSvg(svg, "line", {
      x1: margin.left,
      y1: y,
      x2: margin.left + plotWidth,
      y2: y,
      class: "grid-line",
    });
    const label = appendSvg(svg, "text", {
      x: margin.left - 10,
      y: y + 4,
      "text-anchor": "end",
      class: "axis",
    });
    label.textContent = formatYAxisTickLabel(yValue);
  });

  chartReferenceLines.forEach((line) => {
    const y = yToPx(line.value);
    if (!Number.isFinite(y)) return;
    appendSvg(svg, "line", {
      x1: margin.left,
      y1: y,
      x2: margin.left + plotWidth,
      y2: y,
      stroke: line.color,
      "stroke-width": line.width,
      "stroke-dasharray": line.dasharray,
      opacity: 0.9,
    });
    if (line.label) {
      const label = appendSvg(svg, "text", {
        x: margin.left + plotWidth - 8,
        y: y - 6,
        "text-anchor": "end",
        class: "axis",
        fill: line.color,
      });
      label.textContent = line.label;
    }
  });

  appendSvg(svg, "line", {
    x1: margin.left,
    y1: margin.top + plotHeight,
    x2: margin.left + plotWidth,
    y2: margin.top + plotHeight,
    class: "axis-line",
  });
  appendSvg(svg, "line", {
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + plotHeight,
    class: "axis-line",
  });

  const candidateTicks = Array.isArray(options.xTicks)
    ? options.xTicks
    : (
      options.tickMode === "actual_dates"
        ? buildMonthlyTicksFromValues(tickTimestamps)
        : buildMonthlyTicks(tickTimestamps)
    );
  const tickSpecs = fitTickLabels(candidateTicks.length ? candidateTicks : [
    { x: xBounds.min, label: String(xBounds.min) },
    { x: xBounds.max, label: String(xBounds.max) },
  ], plotWidth);
  const rotateTicks = tickSpecs.length > Math.max(4, Math.floor(plotWidth / 150));

  tickSpecs.forEach((tickSpec) => {
    const x = xToPx(tickSpec.x);
    appendSvg(svg, "line", {
      x1: x,
      y1: margin.top + plotHeight,
      x2: x,
      y2: margin.top + plotHeight + 6,
      class: "axis-line",
    });
    const label = appendSvg(svg, "text", {
      x,
      y: margin.top + plotHeight + 22,
      "text-anchor": rotateTicks ? "end" : "middle",
      class: "axis",
      transform: rotateTicks ? `rotate(-32 ${x} ${margin.top + plotHeight + 22})` : "",
    });
    label.textContent = tickSpec.label;
  });

  const axisLabel = appendSvg(svg, "text", {
    x: 16,
    y: margin.top - 4,
    class: "axis",
  });
  axisLabel.textContent = axisLabelText;
  if (options.xLabel) {
    const xAxisLabel = appendSvg(svg, "text", {
      x: margin.left + (plotWidth / 2),
      y: height - 8,
      "text-anchor": "middle",
      class: "axis",
    });
    xAxisLabel.textContent = String(options.xLabel);
  }

  series.forEach((item) => {
    const pointSegments = splitDrawablePointSegments(item.points, yToPx)
      .map((segment) => downsamplePoints(segment, MAX_RENDER_PATH_POINTS))
      .filter((segment) => segment.length);
    if (!pointSegments.length) return;
    const hoverPoints = downsamplePoints(
      pointSegments.flatMap((segment) => segment),
      MAX_HOVER_POINTS,
    );
    pointSegments.forEach((segment) => {
      const pathData = segment
        .map((point, index) => `${index === 0 ? "M" : "L"}${xToPx(point.x)},${yToPx(point.y)}`)
        .join(" ");
      if (!pathData) return;
      appendSvg(svg, "path", {
        d: pathData,
        stroke: item.color,
        class: "series-line",
      });
    });
    hoverPoints.forEach((point) => {
      const pointY = yToPx(point.y);
      if (!Number.isFinite(point.y) || !Number.isFinite(pointY)) return;
      const circle = appendSvg(svg, "circle", {
        cx: xToPx(point.x),
        cy: pointY,
        r: hoverPoints.length <= MAX_HOVER_POINTS ? 3 : 4,
        fill: item.color,
        class: "series-point",
      });
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      const xLabel = options.tickMode === "actual_dates" && point.usesTimestamp
        ? (point.dateLabel || new Date(point.x).toISOString())
        : `Step: ${point.stepIndex ?? point.x}`;
      title.textContent = point.tooltip || `${item.label}\n${options.tickMode === "actual_dates" && point.usesTimestamp ? `Date: ${xLabel}` : xLabel}\nValue: ${Number(point.y).toFixed(4)}`;
      circle.appendChild(title);
    });

    if (item.stopMarker && Number.isFinite(item.stopMarker.x) && Number.isFinite(item.stopMarker.y)) {
      const markerX = xToPx(item.stopMarker.x);
      const markerY = yToPx(item.stopMarker.y);
      if (!Number.isFinite(markerY)) return;
      const markerSize = 7;
      appendSvg(svg, "line", {
        x1: markerX - markerSize,
        y1: markerY - markerSize,
        x2: markerX + markerSize,
        y2: markerY + markerSize,
        stroke: "#ff5252",
        "stroke-width": 3,
        "stroke-linecap": "round",
      });
      appendSvg(svg, "line", {
        x1: markerX - markerSize,
        y1: markerY + markerSize,
        x2: markerX + markerSize,
        y2: markerY - markerSize,
        stroke: "#ff5252",
        "stroke-width": 3,
        "stroke-linecap": "round",
      });
      const markerHitbox = appendSvg(svg, "circle", {
        cx: markerX,
        cy: markerY,
        r: 11,
        fill: "transparent",
      });
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      const stopLabel = options.tickMode === "actual_dates" && item.stopMarker.usesTimestamp
        ? (item.stopMarker.dateLabel || new Date(item.stopMarker.x).toISOString())
        : `Step: ${item.stopMarker.stepIndex ?? item.stopMarker.x}`;
      title.textContent = `${item.label}\nStopped at zero crossing\n${options.tickMode === "actual_dates" && item.stopMarker.usesTimestamp ? `Date: ${stopLabel}` : stopLabel}\nValue: ${Number(item.stopMarker.y).toFixed(4)}`;
      markerHitbox.appendChild(title);
    }
  });

  host.appendChild(svg);
  renderLegend(legendId, series.map((item) => ({
    label: item.label,
    displayLabel: item.legendLabel || item.label,
    color: item.color,
  })));
}

function drawBarChart(hostId, legendId, bars, yLabel) {
  const host = byId(hostId);
  host.innerHTML = "";
  if (!bars.length) {
    host.appendChild(createEmptyState("Select at least one model and one variant to render this chart."));
    renderLegend(legendId, []);
    return;
  }
  const width = Math.max(980, host.clientWidth || 980, bars.length * 90);
  const height = 420;
  const margin = { top: 20, right: 24, bottom: 120, left: 64 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const bounds = computeFiniteBounds(bars.map((bar) => bar.value));
  if (!bounds) {
    host.appendChild(createEmptyState("No numeric values available for this chart."));
    renderLegend(legendId, []);
    return;
  }
  const yMin = Math.min(0, bounds.min);
  const yMax = Math.max(0, bounds.max);
  const yRange = (yMax - yMin) || 1;
  const yToPx = (value) => margin.top + plotHeight - ((value - yMin) / yRange) * plotHeight;
  const slotWidth = plotWidth / bars.length;
  const barWidth = Math.max(14, slotWidth * 0.68);

  const svg = createSvg(width, height);
  appendSvg(svg, "rect", {
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
    fill: "#0f1620",
    stroke: "#334155",
    rx: 14,
  });

  for (let i = 0; i <= 4; i += 1) {
    const ratio = i / 4;
    const yValue = yMin + (1 - ratio) * yRange;
    const y = margin.top + ratio * plotHeight;
    appendSvg(svg, "line", {
      x1: margin.left,
      y1: y,
      x2: margin.left + plotWidth,
      y2: y,
      class: "grid-line",
    });
    const label = appendSvg(svg, "text", {
      x: margin.left - 10,
      y: y + 4,
      "text-anchor": "end",
      class: "axis",
    });
    label.textContent = Number(yValue).toFixed(2);
  }

  appendSvg(svg, "line", {
    x1: margin.left,
    y1: yToPx(0),
    x2: margin.left + plotWidth,
    y2: yToPx(0),
    class: "axis-line",
  });
  appendSvg(svg, "line", {
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + plotHeight,
    class: "axis-line",
  });

  bars.forEach((bar, index) => {
    const xCenter = margin.left + (index + 0.5) * slotWidth;
    const x = xCenter - barWidth / 2;
    const y = Math.min(yToPx(bar.value), yToPx(0));
    const h = Math.max(1, Math.abs(yToPx(bar.value) - yToPx(0)));
    const rect = appendSvg(svg, "rect", {
      x,
      y,
      width: barWidth,
      height: h,
      fill: bar.color,
      rx: 8,
      class: "series-bar",
    });
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${bar.label}\n${yLabel}: ${Number(bar.value).toFixed(3)}`;
    rect.appendChild(title);

    const label = appendSvg(svg, "text", {
      x: xCenter,
      y: margin.top + plotHeight + 18,
      "text-anchor": "end",
      class: "axis",
      transform: `rotate(-35 ${xCenter} ${margin.top + plotHeight + 18})`,
    });
    const shortLabel = truncateLabelMiddle(bar.label, 28);
    label.textContent = shortLabel;
  });

  const axisLabel = appendSvg(svg, "text", {
    x: 16,
    y: margin.top - 4,
    class: "axis",
  });
  axisLabel.textContent = yLabel;

  host.appendChild(svg);
  renderLegend(legendId, bars.map((bar) => ({ label: bar.label, color: bar.color })));
}

function chooseRenderer(chartKind, seriesCount, pointCount) {
  const normalizedSeriesCount = Number(seriesCount) || 0;
  const normalizedPointCount = Number(pointCount) || 0;
  if (chartKind === "line" || chartKind === "bar") {
    if (normalizedSeriesCount >= FAST_CANVAS_SERIES_THRESHOLD || normalizedPointCount >= FAST_CANVAS_POINT_THRESHOLD) {
      return "canvas";
    }
  }
  return "svg";
}

function attachCanvasTooltip(canvas, hitTargets, formatter) {
  if (!canvas) return;
  canvas.onmousemove = (event) => {
    if (!Array.isArray(hitTargets) || !hitTargets.length) {
      canvas.title = "";
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let best = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    hitTargets.forEach((target) => {
      const dx = Number(target.x) - x;
      const dy = Number(target.y) - y;
      const distance = Math.sqrt((dx * dx) + (dy * dy));
      if (distance < bestDistance) {
        bestDistance = distance;
        best = target;
      }
    });
    if (best && bestDistance <= 20) {
      canvas.title = formatter(best);
    } else {
      canvas.title = "";
    }
  };
}

function drawLineChartCanvas(hostId, legendId, series, yLabel, tickTimestamps, options = {}) {
  const host = byId(hostId);
  host.innerHTML = "";
  if (!series.length) {
    host.appendChild(createEmptyState(options.emptyMessage || "Select at least one model and one variant to render this chart."));
    renderLegend(legendId, []);
    return;
  }
  const width = Math.max(980, host.clientWidth || 980);
  const height = 360;
  const margin = { top: 20, right: 24, bottom: 60, left: 64 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const allPoints = [];
  series.forEach((item) => item.points.forEach((point) => allPoints.push(point)));
  const referenceLines = buildReferenceLineSpecs(options.referenceLines);
  const xBounds = computeFiniteBounds(allPoints.map((point) => point.x));
  const yBounds = computeFiniteBounds([
    ...allPoints.map((point) => point.y),
    ...referenceLines.map((line) => line.value),
  ]);
  if (!xBounds || !yBounds) {
    host.appendChild(createEmptyState("No numeric values available for this chart."));
    renderLegend(legendId, []);
    return;
  }

  const requestedYScale = String(options.yScale || "linear");
  const canUsePiecewiseLogScale = requestedYScale === "piecewise_log_over_1000" && yBounds.max > 1000;
  let yMin = yBounds.min;
  let yMax = yBounds.max;
  if (yMin === yMax && !canUsePiecewiseLogScale) {
    yMin -= 1;
    yMax += 1;
  }
  const xRange = (xBounds.max - xBounds.min) || 1;
  const xToPx = (value) => margin.left + ((value - xBounds.min) / xRange) * plotWidth;
  let yToPx;
  let yTickValues;
  let axisLabelText = yLabel;
  if (canUsePiecewiseLogScale) {
    const logThreshold = 1000;
    const piecewiseTransform = (value) => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return null;
      if (numeric <= logThreshold) return numeric;
      return logThreshold + (logThreshold * Math.log10(numeric / logThreshold));
    };
    const transformedYMin = piecewiseTransform(yMin);
    const transformedYMax = piecewiseTransform(yMax);
    const transformedRange = ((transformedYMax ?? 0) - (transformedYMin ?? 0)) || 1;
    yToPx = (value) => {
      const transformed = piecewiseTransform(value);
      if (!Number.isFinite(transformed)) return null;
      return margin.top + plotHeight - ((transformed - transformedYMin) / transformedRange) * plotHeight;
    };
    yTickValues = [...new Set([yMin, Math.min(logThreshold, yMax), yMax].map((value) => Number(value)).filter((value) => Number.isFinite(value)))].sort((a, b) => a - b);
    axisLabelText = `${yLabel} (linear <= 1k, log > 1k)`;
  } else {
    const yRange = (yMax - yMin) || 1;
    yToPx = (value) => margin.top + plotHeight - ((value - yMin) / yRange) * plotHeight;
    yTickValues = Array.from({ length: 5 }, (_, index) => yMin + (1 - (index / 4)) * yRange);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    host.appendChild(createEmptyState("Canvas rendering is not available in this browser."));
    renderLegend(legendId, []);
    return;
  }

  ctx.fillStyle = "#0f1620";
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(margin.left, margin.top, plotWidth, plotHeight, 14);
  ctx.fill();
  ctx.stroke();

  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#c7d3df";
  ctx.strokeStyle = "#243142";
  yTickValues.forEach((yValue) => {
    const y = yToPx(yValue);
    if (!Number.isFinite(y)) return;
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText(formatYAxisTickLabel(yValue), margin.left - 52, y + 4);
  });
  referenceLines.forEach((line) => {
    const y = yToPx(line.value);
    if (!Number.isFinite(y)) return;
    ctx.save();
    ctx.strokeStyle = line.color;
    ctx.lineWidth = line.width;
    ctx.setLineDash(String(line.dasharray || "6 4").split(" ").map((value) => Number(value) || 0));
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
    ctx.restore();
  });
  ctx.strokeStyle = "#4b5a6f";
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.stroke();
  ctx.fillText(axisLabelText, 16, margin.top - 4);

  const candidateTicks = Array.isArray(options.xTicks)
    ? options.xTicks
    : (
      options.tickMode === "actual_dates"
        ? buildMonthlyTicksFromValues(tickTimestamps)
        : buildMonthlyTicks(tickTimestamps)
    );
  const tickSpecs = fitTickLabels(candidateTicks.length ? candidateTicks : [
    { x: xBounds.min, label: String(xBounds.min) },
    { x: xBounds.max, label: String(xBounds.max) },
  ], plotWidth);
  tickSpecs.forEach((tickSpec) => {
    const x = xToPx(tickSpec.x);
    ctx.beginPath();
    ctx.moveTo(x, margin.top + plotHeight);
    ctx.lineTo(x, margin.top + plotHeight + 6);
    ctx.stroke();
    ctx.save();
    ctx.translate(x, margin.top + plotHeight + 22);
    ctx.rotate(-0.55);
    ctx.fillText(String(tickSpec.label), 0, 0);
    ctx.restore();
  });
  if (options.xLabel) {
    ctx.save();
    ctx.textAlign = "center";
    ctx.fillText(String(options.xLabel), margin.left + (plotWidth / 2), height - 8);
    ctx.restore();
  }

  const hitTargets = [];
  series.forEach((item) => {
    const pointSegments = splitDrawablePointSegments(item.points, yToPx)
      .map((segment) => downsamplePoints(segment, MAX_RENDER_PATH_POINTS))
      .filter((segment) => segment.length);
    if (!pointSegments.length) return;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 1.5;
    pointSegments.forEach((segment) => {
      ctx.beginPath();
      segment.forEach((point, index) => {
        const px = xToPx(point.x);
        const py = yToPx(point.y);
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
        hitTargets.push({ x: px, y: py, item, point });
      });
      ctx.stroke();
    });
    if (item.stopMarker && Number.isFinite(item.stopMarker.x) && Number.isFinite(item.stopMarker.y)) {
      const markerX = xToPx(item.stopMarker.x);
      const markerY = yToPx(item.stopMarker.y);
      if (Number.isFinite(markerY)) {
        ctx.strokeStyle = "#ff5252";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(markerX - 7, markerY - 7);
        ctx.lineTo(markerX + 7, markerY + 7);
        ctx.moveTo(markerX - 7, markerY + 7);
        ctx.lineTo(markerX + 7, markerY - 7);
        ctx.stroke();
      }
    }
  });

  attachCanvasTooltip(canvas, hitTargets, (target) => {
    const point = target.point;
    const xLabel = options.tickMode === "actual_dates" && point.usesTimestamp
      ? (point.dateLabel || new Date(point.x).toISOString())
      : `Step: ${point.stepIndex ?? point.x}`;
    return point.tooltip || `${target.item.label}\n${options.tickMode === "actual_dates" && point.usesTimestamp ? `Date: ${xLabel}` : xLabel}\nValue: ${Number(point.y).toFixed(4)}`;
  });

  host.appendChild(canvas);
  renderLegend(legendId, series.map((item) => ({
    label: item.label,
    displayLabel: item.legendLabel || item.label,
    color: item.color,
  })));
}

function drawBarChartCanvas(hostId, legendId, bars, yLabel) {
  const host = byId(hostId);
  host.innerHTML = "";
  if (!bars.length) {
    host.appendChild(createEmptyState("Select at least one model and one variant to render this chart."));
    renderLegend(legendId, []);
    return;
  }
  const width = Math.max(980, host.clientWidth || 980, bars.length * 50);
  const height = 420;
  const margin = { top: 20, right: 24, bottom: 120, left: 64 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const bounds = computeFiniteBounds(bars.map((bar) => bar.value));
  if (!bounds) {
    host.appendChild(createEmptyState("No numeric values available for this chart."));
    renderLegend(legendId, []);
    return;
  }
  const yMin = Math.min(0, bounds.min);
  const yMax = Math.max(0, bounds.max);
  const yRange = (yMax - yMin) || 1;
  const yToPx = (value) => margin.top + plotHeight - ((value - yMin) / yRange) * plotHeight;
  const slotWidth = plotWidth / bars.length;
  const barWidth = Math.max(8, slotWidth * 0.7);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    host.appendChild(createEmptyState("Canvas rendering is not available in this browser."));
    renderLegend(legendId, []);
    return;
  }
  ctx.fillStyle = "#0f1620";
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(margin.left, margin.top, plotWidth, plotHeight, 14);
  ctx.fill();
  ctx.stroke();
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#c7d3df";
  for (let i = 0; i <= 4; i += 1) {
    const ratio = i / 4;
    const yValue = yMin + (1 - ratio) * yRange;
    const y = margin.top + ratio * plotHeight;
    ctx.strokeStyle = "#243142";
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText(Number(yValue).toFixed(2), margin.left - 52, y + 4);
  }
  const zeroY = yToPx(0);
  ctx.strokeStyle = "#4b5a6f";
  ctx.beginPath();
  ctx.moveTo(margin.left, zeroY);
  ctx.lineTo(margin.left + plotWidth, zeroY);
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.stroke();
  const hitTargets = [];
  bars.forEach((bar, index) => {
    const xCenter = margin.left + (index + 0.5) * slotWidth;
    const x = xCenter - barWidth / 2;
    const y = Math.min(yToPx(bar.value), zeroY);
    const h = Math.max(1, Math.abs(yToPx(bar.value) - zeroY));
    ctx.fillStyle = bar.color;
    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, h, 8);
    ctx.fill();
    hitTargets.push({ x: xCenter, y, bar });
  });
  attachCanvasTooltip(canvas, hitTargets, (target) => `${target.bar.label}\n${yLabel}: ${Number(target.bar.value).toFixed(3)}`);
  host.appendChild(canvas);
  renderLegend(legendId, bars.map((bar) => ({ label: bar.label, color: bar.color })));
}

function drawGroupedBarChart(hostId, legendId, groups, yLabel, metrics, options = {}) {
  const host = byId(hostId);
  host.innerHTML = "";
  if (!groups.length) {
    host.appendChild(createEmptyState("Select at least one model and one variant to render this chart."));
    renderLegend(legendId, []);
    return;
  }

  const metricDefs = Array.isArray(metrics) && metrics.length ? metrics : [];
  const allBars = groups.flatMap((group) => (
    metricDefs.map((metric) => ({
      groupLabel: group.label,
      metricLabel: metric.label,
      color: metric.color,
      value: Number(group[metric.key]),
    }))
  )).filter((bar) => Number.isFinite(bar.value));

  const bounds = computeFiniteBounds(allBars.map((bar) => bar.value));
  if (!bounds) {
    host.appendChild(createEmptyState("No numeric values available for this chart."));
    renderLegend(legendId, []);
    return;
  }

  const width = Math.max(980, host.clientWidth || 980, groups.length * 125);
  const height = 420;
  const margin = { top: 20, right: 24, bottom: 120, left: 64 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const yMin = Math.min(0, bounds.min);
  const yMax = Math.max(0, bounds.max);
  const yRange = (yMax - yMin) || 1;
  const yToPx = (value) => margin.top + plotHeight - ((value - yMin) / yRange) * plotHeight;
  const slotWidth = plotWidth / groups.length;
  const innerGap = 8;
  const barWidth = Math.max(12, Math.min(28, (slotWidth - innerGap * (metricDefs.length - 1)) / Math.max(1, metricDefs.length) * 0.84));

  const formatValue = typeof options.valueFormatter === "function"
    ? options.valueFormatter
    : (value) => `${(Number(value) * 100).toFixed(2)}%`;
  const formatAxisTick = typeof options.axisTickFormatter === "function"
    ? options.axisTickFormatter
    : (value) => Number(value).toFixed(2);
  const showHalfReference = options.showHalfReference !== false;

  const svg = createSvg(width, height);
  appendSvg(svg, "rect", {
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
    fill: "#0f1620",
    stroke: "#334155",
    rx: 14,
  });

  for (let i = 0; i <= 4; i += 1) {
    const ratio = i / 4;
    const yValue = yMin + (1 - ratio) * yRange;
    const y = margin.top + ratio * plotHeight;
    appendSvg(svg, "line", {
      x1: margin.left,
      y1: y,
      x2: margin.left + plotWidth,
      y2: y,
      class: "grid-line",
    });
    const label = appendSvg(svg, "text", {
      x: margin.left - 10,
      y: y + 4,
      "text-anchor": "end",
      class: "axis",
    });
    label.textContent = formatAxisTick(yValue);
  }

  if (showHalfReference && yMin <= 0.5 && yMax >= 0.5) {
    appendSvg(svg, "line", {
      x1: margin.left,
      y1: yToPx(0.5),
      x2: margin.left + plotWidth,
      y2: yToPx(0.5),
      stroke: "#ffd166",
      "stroke-width": 1.5,
      "stroke-dasharray": "6 4",
      opacity: 0.95,
    });
    const referenceLabel = appendSvg(svg, "text", {
      x: margin.left + plotWidth - 8,
      y: yToPx(0.5) - 6,
      "text-anchor": "end",
      class: "axis",
      fill: "#ffd166",
    });
    referenceLabel.textContent = "50%";
  }

  appendSvg(svg, "line", {
    x1: margin.left,
    y1: yToPx(0),
    x2: margin.left + plotWidth,
    y2: yToPx(0),
    class: "axis-line",
  });
  appendSvg(svg, "line", {
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + plotHeight,
    class: "axis-line",
  });

  groups.forEach((group, groupIndex) => {
    const groupCenter = margin.left + (groupIndex + 0.5) * slotWidth;
    const totalBarsWidth = metricDefs.length * barWidth + Math.max(0, metricDefs.length - 1) * innerGap;
    const groupStart = groupCenter - totalBarsWidth / 2;

    metricDefs.forEach((metric, metricIndex) => {
      const value = Number(group[metric.key]);
      if (!Number.isFinite(value)) return;
      const x = groupStart + metricIndex * (barWidth + innerGap);
      const y = Math.min(yToPx(value), yToPx(0));
      const h = Math.max(1, Math.abs(yToPx(value) - yToPx(0)));
      const rect = appendSvg(svg, "rect", {
        x,
        y,
        width: barWidth,
        height: h,
        fill: metric.color,
        rx: 6,
        class: "series-bar",
      });
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = `${group.label}\n${metric.label}: ${formatValue(value)}`;
      rect.appendChild(title);
    });

    const label = appendSvg(svg, "text", {
      x: groupCenter,
      y: margin.top + plotHeight + 18,
      "text-anchor": "end",
      class: "axis",
      transform: `rotate(-35 ${groupCenter} ${margin.top + plotHeight + 18})`,
    });
    const shortLabel = group.label.length > 24 ? `${group.label.slice(0, 24)}...` : group.label;
    label.textContent = shortLabel;
  });

  const axisLabel = appendSvg(svg, "text", {
    x: 16,
    y: margin.top - 4,
    class: "axis",
  });
  axisLabel.textContent = yLabel;

  host.appendChild(svg);
  renderLegend(legendId, metricDefs.map((metric) => ({ label: metric.label, color: metric.color })));
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload?.error) {
        detail = payload.error;
      }
    } catch (error) {
      // Keep the HTTP status text when the response body is not JSON.
    }
    throw new Error(`Request failed: ${detail}`);
  }
  return response.json();
}

async function loadExperimentIndex() {
  state.experimentIndex = await fetchJson("/api/experiments");
}

async function computeFixed5kRuns() {
  state.fixed5kRunsState = "loading";
  state.fixed5kRunsError = "";
  renderFixed5kRunsTable();
  try {
    state.fixed5kRuns = await fetchJson("/api/fixed-5k-runs");
    state.fixed5kRunsState = "ready";
  } catch (error) {
    state.fixed5kRunsState = "error";
    state.fixed5kRunsError = error instanceof Error ? error.message : String(error);
  }
  renderFixed5kRunsTable();
}

function formatSignedPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(2)}%`;
}

function formatSignedWinsPerDay(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(3)}`;
}

function formatAccuracyPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  return `${(numeric * 100).toFixed(2)}%`;
}

function formatIntegerCount(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  return `${Math.round(numeric)}`;
}

async function loadExperimentPayload(experimentId) {
  if (state.experimentCache.has(experimentId)) {
    return state.experimentCache.get(experimentId);
  }
  const payload = await fetchJson(`/api/experiments/${experimentId}`);
  payload.seriesCache = new Map();
  state.experimentCache.set(experimentId, payload);
  return payload;
}

function getActiveExperiment() {
  return state.experimentCache.get(state.activeExperimentId) || null;
}

function orderedAvailableVariants(experiment) {
  const preferred = ["full", "market_hours", "outside_market_hours", "derived_market_hours"];
  const available = Array.isArray(experiment?.available_variants) ? experiment.available_variants : [];
  const result = [];
  preferred.forEach((name) => {
    if (available.includes(name)) result.push(name);
  });
  available.forEach((name) => {
    if (!result.includes(name)) result.push(name);
  });
  return result;
}

function orderedAvailableModels(experiment) {
  const models = Array.isArray(experiment?.available_models) ? experiment.available_models : [];
  return [...models].sort((a, b) => {
    const aBase = isBaselineModelName(a);
    const bBase = isBaselineModelName(b);
    if (aBase !== bBase) return aBase ? -1 : 1;
    return a.localeCompare(b);
  });
}

function orderedAvailableModelsForVariant(experiment, variantName) {
  const variant = experiment?.variants?.[variantName];
  const models = Array.isArray(variant?.models) ? variant.models : [];
  return models
    .map((model) => String(model?.name || ""))
    .filter((name) => !!name)
    .sort((a, b) => {
      const aBase = isBaselineModelName(a);
      const bBase = isBaselineModelName(b);
      if (aBase !== bBase) return aBase ? -1 : 1;
      return a.localeCompare(b);
    });
}

function isBaselineModelName(modelName) {
  const name = String(modelName || "");
  if (BASELINE_MODELS.has(name)) return true;
  return [...BASELINE_MODELS].some((baselineName) => name === `${baselineName}_AGGREGATE` || name.startsWith(`${baselineName}_M`));
}

function ensureSelections(experiment) {
  if (!experiment) return;
  const variants = orderedAvailableVariants(experiment);
  const models = orderedAvailableModels(experiment);
  if (!state.selectedVariants.size) {
    const initialVariants = state.pendingInitialVariants.filter((variant) => variants.includes(variant));
    (initialVariants.length ? initialVariants : variants).forEach((variant) => state.selectedVariants.add(variant));
  }
  if (!state.selectedModels.size && !state.userClearedModels) {
    getDefaultSelectedModels(experiment).forEach((model) => state.selectedModels.add(model));
  }
  [...state.selectedVariants].forEach((variant) => {
    if (!variants.includes(variant)) state.selectedVariants.delete(variant);
  });
  [...state.selectedModels].forEach((model) => {
    if (!models.includes(model)) state.selectedModels.delete(model);
  });
}

function renderExperimentSelector() {
  const select = byId("experimentSelect");
  select.innerHTML = "";
  [...state.experimentIndex]
    .sort((a, b) => Number(a?.experiment_id || 0) - Number(b?.experiment_id || 0))
    .forEach((experiment) => {
    const option = document.createElement("option");
    option.value = String(experiment.experiment_id);
    option.textContent = `#${experiment.experiment_id} ${experiment.name}`;
    select.appendChild(option);
  });
  if (state.activeExperimentId) {
    select.value = String(state.activeExperimentId);
  }
}

function formatVariantLabel(variant) {
  const normalized = String(variant || "").toLowerCase();
  if (normalized === "full") return "Full";
  if (normalized === "market_hours" || normalized === "derived_market_hours") return "Market Hours";
  if (normalized === "outside_market_hours" || normalized === "derived_outside_market_hours") return "Outside Market Hours";
  return String(variant || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatSeriesLabel(variantName, modelName) {
  const modelText = String(modelName || "");
  if (/_((FULL)|(MARKET_HOURS)|(OUTSIDE_MARKET_HOURS))_/.test(modelText)) {
    return modelText;
  }
  return `${variantName} / ${modelText}`;
}

function normalizeTimeVariantLabel(value) {
  const normalized = String(value || "").toUpperCase();
  if (normalized === "FULL") return "Full";
  if (normalized === "MARKET_HOURS") return "Market Hours";
  if (normalized === "OUTSIDE_MARKET_HOURS") return "Outside Market Hours";
  return formatVariantLabel(String(value || "").toLowerCase());
}

function normalizeVariantKey(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) return null;
  if (normalized === "full" || normalized === "derived_full") return "full";
  if (normalized === "market_hours" || normalized === "derived_market_hours") return "market_hours";
  if (normalized === "outside_market_hours" || normalized === "derived_outside_market_hours") return "outside_market_hours";
  return normalized;
}

function renderVariantToggles(experiment) {
  const host = byId("variantToggles");
  host.innerHTML = "";
  const metadataByName = buildModelMetadataMap(experiment);
  const modelTimeVariants = [...new Set(
    [...metadataByName.values()]
      .map((metadata) => metadata.timeVariant)
      .filter((value) => !!value)
  )];
  if (modelTimeVariants.length > 1) {
    ["full", "market_hours", "outside_market_hours"].filter((value) => modelTimeVariants.includes(value)).forEach((timeVariant) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `toggle${state.activeModelFilters.timeVariants.has(timeVariant) ? " active" : ""}`;
      button.textContent = normalizeTimeVariantLabel(timeVariant);
      button.addEventListener("click", () => {
        if (state.activeModelFilters.timeVariants.has(timeVariant)) {
          state.activeModelFilters.timeVariants.delete(timeVariant);
        } else {
          state.activeModelFilters.timeVariants.add(timeVariant);
        }
        applyActiveModelFilters(experiment);
        render().catch(renderFailure);
      });
      host.appendChild(button);
    });
    return;
  }
  orderedAvailableVariants(experiment).forEach((variant) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.selectedVariants.has(variant) ? " active" : ""}`;
    button.textContent = formatVariantLabel(variant);
    button.addEventListener("click", () => {
      if (state.selectedVariants.has(variant)) {
        if (state.selectedVariants.size === 1) return;
        state.selectedVariants.delete(variant);
      } else {
        state.selectedVariants.add(variant);
      }
      render().catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderModelToggles(experiment) {
  const host = byId("modelToggles");
  host.innerHTML = "";
  const orderedModelNames = orderedAvailableModels(experiment).sort((left, right) => {
    const leftSelected = state.selectedModels.has(left);
    const rightSelected = state.selectedModels.has(right);
    if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;
    return left.localeCompare(right);
  });
  orderedModelNames.forEach((modelName) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.selectedModels.has(modelName) ? " active" : ""}`;
    button.textContent = modelName;
    button.addEventListener("click", () => {
      clearActiveModelFilters();
      if (state.selectedModels.has(modelName)) {
        state.selectedModels.delete(modelName);
      } else {
        state.selectedModels.add(modelName);
      }
      render().catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderPortfolioToggles() {
  document.querySelectorAll(".portfolio-toggle[data-portfolio-mode]").forEach((button) => {
    button.classList.remove("active");
    button.onclick = () => {
      state.portfolioSizingMode = button.dataset.portfolioMode;
      if ((state.portfolioSizingMode === "current_percent" || state.portfolioSizingMode === "peak_percent") && Number(state.portfolioIncrementValue) === 1) {
        state.portfolioIncrementValue = 0.05;
      }
      renderPortfolioToggles();
      renderPortfolioControls();
      rerenderPortfolioOnly(getActiveExperiment(), state.renderToken).catch(renderFailure);
    };
    if (button.dataset.portfolioMode === state.portfolioSizingMode) {
      button.classList.add("active");
    }
  });
}

function renderPortfolioControls() {
  const startingBalanceInput = byId("portfolioStartingBalanceInput");
  const incrementValueInput = byId("portfolioIncrementValueInput");
  if (startingBalanceInput) {
    startingBalanceInput.value = String(state.portfolioStartingBalance);
    startingBalanceInput.oninput = (event) => {
      const numeric = Number(event.target.value);
      state.portfolioStartingBalance = Number.isFinite(numeric) && numeric >= 0 ? numeric : 0;
      portfolioCurveCache.clear();
      rerenderPortfolioOnly(getActiveExperiment(), state.renderToken).catch(renderFailure);
    };
  }
  if (incrementValueInput) {
    incrementValueInput.value = String(state.portfolioIncrementValue);
    incrementValueInput.oninput = (event) => {
      const numeric = Number(event.target.value);
      state.portfolioIncrementValue = Number.isFinite(numeric) && numeric >= 0 ? numeric : 0;
      portfolioCurveCache.clear();
      rerenderPortfolioOnly(getActiveExperiment(), state.renderToken).catch(renderFailure);
    };
  }
}

function isWindowSeriesModel(modelName) {
  const name = String(modelName || "");
  if (/_M\d+$/.test(name)) return true;
  return /_(\d+K|\d+)_(\d+K|\d+)_(FULL|MARKET_HOURS|OUTSIDE_MARKET_HOURS)_/.test(name) && (
    name.includes("_WINDOW_") || name.includes("_PPO_WINDOW_")
  );
}

function isAggregateSeriesModel(modelName) {
  return /_AGGREGATE(?:_[A-Z0-9]+)?$/.test(String(modelName || ""));
}

function detectModelFamily(modelName) {
  const name = String(modelName || "").toUpperCase();
  const families = ["ALWAYS_UP", "ALWAYS_DOWN", "RANDOM", "TRANSFORMER", "MAMBA", "XGBOOST", "DAGGER", "LSTMPPO", "LSTM", "PPO", "NN", "RF", "BC"];
  const match = families.find((family) => name.startsWith(family));
  if (match) return match;
  return name.split("_", 1)[0] || "UNKNOWN";
}

function detectThresholdPct(modelName, explicitThreshold) {
  const numericExplicit = Number(explicitThreshold);
  if (Number.isFinite(numericExplicit)) return numericExplicit;
  const match = String(modelName || "").match(/_(50|90|95|99|995)$/);
  return match ? Number(match[1]) : null;
}

function detectModelVariation(modelName, explicitVariation) {
  if (explicitVariation) return String(explicitVariation);
  const name = String(modelName || "").toUpperCase();
  if (name.includes("_PPO_WINDOW_RETRAIN_ONLY_")) return "PPO_WINDOW_RETRAIN_ONLY";
  if (name.includes("_PPO_WINDOW_RETRAIN_")) return "PPO_WINDOW_RETRAIN";
  if (name.includes("_PPO_WINDOW_CONTINUE_ONLY_")) return "PPO_WINDOW_CONTINUE_ONLY";
  if (name.includes("_PPO_WINDOW_CONTINUE_")) return "PPO_WINDOW_CONTINUE";
  if (name.includes("_WINDOW_RETRAIN_ONLY_")) return "WINDOW_RETRAIN_ONLY";
  if (name.includes("_WINDOW_RETRAIN_")) return "WINDOW_RETRAIN";
  if (name.includes("_WINDOW_CONTINUE_ONLY_")) return "WINDOW_CONTINUE_ONLY";
  if (name.includes("_WINDOW_CONTINUE_")) return "WINDOW_CONTINUE";
  return "BASE";
}

function inferWindowLength(modelName, explicitWindowLength) {
  if (explicitWindowLength != null && explicitWindowLength !== "") return String(explicitWindowLength);
  const name = String(modelName || "").toUpperCase();
  const windowMatch = name.match(/_WINDOW(\d+K?|\d+)_/);
  if (windowMatch) return windowMatch[1];
  const trainWindowMatch = name.match(/_(\d+K|\d+)_(\d+K|\d+)_((FULL|MARKET_HOURS|OUTSIDE_MARKET_HOURS))_/);
  if (trainWindowMatch) return trainWindowMatch[2];
  return null;
}

function detectTimeVariant(modelName, explicitTimeVariant) {
  if (explicitTimeVariant) return normalizeVariantKey(explicitTimeVariant);
  const name = String(modelName || "").toUpperCase();
  if (name.includes("_OUTSIDE_MARKET_HOURS_")) return "outside_market_hours";
  if (name.includes("_MARKET_HOURS_")) return "market_hours";
  if (name.includes("_FULL_")) return "full";
  return null;
}

function buildModelMetadataMap(experiment) {
  const metadataByName = new Map();
  const variants = experiment?.variants || {};
  Object.values(variants).forEach((variant) => {
    const models = Array.isArray(variant?.models) ? variant.models : [];
    models.forEach((model) => {
      const name = String(model?.name || "");
      if (!name || metadataByName.has(name)) return;
      const modelVariation = detectModelVariation(name, model?.model_variation);
      const windowLength = inferWindowLength(name, model?.window_length);
      metadataByName.set(name, {
        name,
        family: String(model?.family || detectModelFamily(name)),
        modelVariation,
        timeVariant: detectTimeVariant(name, model?.time_variant),
        thresholdPct: detectThresholdPct(name, model?.threshold_pct),
        trainLength: model?.train_length == null ? null : String(model.train_length),
        windowLength,
        isAggregate: isAggregateSeriesModel(name),
        isBaseline: isBaselineModelName(name),
        isWindow: Boolean(windowLength) || modelVariation.includes("WINDOW"),
        isPpoContinue: modelVariation === "PPO_WINDOW_CONTINUE" || modelVariation === "PPO_WINDOW_CONTINUE_ONLY",
        isContinue: modelVariation.includes("CONTINUE"),
        isRetrain: modelVariation.includes("RETRAIN"),
      });
    });
  });
  orderedAvailableModels(experiment).forEach((name) => {
    if (!metadataByName.has(name)) {
      const modelVariation = detectModelVariation(name, null);
      const windowLength = inferWindowLength(name, null);
      metadataByName.set(name, {
        name,
        family: detectModelFamily(name),
        modelVariation,
        timeVariant: detectTimeVariant(name, null),
        thresholdPct: detectThresholdPct(name, null),
        trainLength: null,
        windowLength,
        isAggregate: isAggregateSeriesModel(name),
        isBaseline: isBaselineModelName(name),
        isWindow: Boolean(windowLength) || modelVariation.includes("WINDOW"),
        isPpoContinue: modelVariation === "PPO_WINDOW_CONTINUE" || modelVariation === "PPO_WINDOW_CONTINUE_ONLY",
        isContinue: modelVariation.includes("CONTINUE"),
        isRetrain: modelVariation.includes("RETRAIN"),
      });
    }
  });
  return metadataByName;
}

function clearActiveModelFilters() {
  state.activeModelFilters.timeVariants = new Set();
  state.activeModelFilters.thresholds = new Set();
  state.activeModelFilters.trainSizes = new Set();
  state.activeModelFilters.strategies = new Set();
  state.activeModelFilters.families = new Set();
}

function trainLengthSortValue(value) {
  const text = String(value || "").trim().toUpperCase();
  const match = text.match(/^(\\d+)(K)?$/);
  if (!match) return Number.POSITIVE_INFINITY;
  const numeric = Number(match[1]);
  if (!Number.isFinite(numeric)) return Number.POSITIVE_INFINITY;
  return match[2] ? numeric * 1000 : numeric;
}

function matchesStrategyFilter(metadata, strategyKey) {
  if (strategyKey === "baseline") return metadata.isBaseline;
  if (strategyKey === "learned") return !metadata.isBaseline;
  if (strategyKey === "aggregate") return metadata.isAggregate;
  if (strategyKey === "non_aggregate") return !metadata.isAggregate;
  if (strategyKey === "window") return metadata.isWindow;
  if (strategyKey === "non_window") return !metadata.isWindow;
  if (strategyKey === "ppo_continue") return metadata.isPpoContinue;
  if (strategyKey === "non_ppo_continue") return !metadata.isPpoContinue;
  if (strategyKey === "retrain") return metadata.isRetrain;
  if (strategyKey === "continue") return metadata.isContinue;
  return true;
}

function applyActiveModelFilters(experiment) {
  const metadataByName = buildModelMetadataMap(experiment);
  const selected = orderedAvailableModels(experiment).filter((name) => {
    const metadata = metadataByName.get(name);
    if (!metadata) return false;
    if (state.activeModelFilters.timeVariants.size && !state.activeModelFilters.timeVariants.has(String(metadata.timeVariant))) {
      return false;
    }
    if (state.activeModelFilters.thresholds.size && !state.activeModelFilters.thresholds.has(String(metadata.thresholdPct))) {
      return false;
    }
    if (state.activeModelFilters.trainSizes.size && !state.activeModelFilters.trainSizes.has(String(metadata.trainLength))) {
      return false;
    }
    if (state.activeModelFilters.families.size && !state.activeModelFilters.families.has(String(metadata.family))) {
      return false;
    }
    if (state.activeModelFilters.strategies.size) {
      for (const strategyKey of state.activeModelFilters.strategies) {
        if (!matchesStrategyFilter(metadata, strategyKey)) return false;
      }
    }
    return true;
  });
  state.userClearedModels = false;
  state.selectedModels = new Set(selected.length ? selected : []);
}

function renderThresholdQuickActions(experiment) {
  const host = byId("thresholdQuickActions");
  if (!host) return;
  host.innerHTML = "";
  [
    { label: "None (50)", threshold: "50" },
    { label: "90", threshold: "90" },
    { label: "95", threshold: "95" },
    { label: "99", threshold: "99" },
    { label: "995", threshold: "995" },
  ].forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.activeModelFilters.thresholds.has(option.threshold) ? " active" : ""}`;
    button.textContent = option.label;
    button.addEventListener("click", () => {
      if (state.activeModelFilters.thresholds.has(option.threshold)) {
        state.activeModelFilters.thresholds.delete(option.threshold);
      } else {
        state.activeModelFilters.thresholds.add(option.threshold);
      }
      applyActiveModelFilters(experiment);
      render().catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderTrainSizeQuickActions(experiment) {
  const host = byId("trainSizeQuickActions");
  if (!host) return;
  host.innerHTML = "";
  const trainSizes = [...new Set(
    [...buildModelMetadataMap(experiment).values()]
      .filter((metadata) => !!metadata?.trainLength && !metadata?.isBaseline)
      .map((metadata) => metadata.trainLength)
  )].sort((left, right) => (
    trainLengthSortValue(left) - trainLengthSortValue(right)
    || String(left).localeCompare(String(right))
  ));
  trainSizes.forEach((trainSize) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.activeModelFilters.trainSizes.has(String(trainSize)) ? " active" : ""}`;
    button.textContent = trainSize;
    button.addEventListener("click", () => {
      if (state.activeModelFilters.trainSizes.has(String(trainSize))) {
        state.activeModelFilters.trainSizes.delete(String(trainSize));
      } else {
        state.activeModelFilters.trainSizes.add(String(trainSize));
      }
      applyActiveModelFilters(experiment);
      render().catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderStrategyQuickActions(experiment) {
  const host = byId("strategyQuickActions");
  if (!host) return;
  host.innerHTML = "";
  [
    { key: "baseline", label: "Baselines" },
    { key: "learned", label: "Learned Models" },
    { key: "aggregate", label: "Aggregate" },
    { key: "non_aggregate", label: "Non-Aggregate" },
    { key: "window", label: "Window" },
    { key: "non_window", label: "Non-Window" },
    { key: "ppo_continue", label: "PPO Continue" },
    { key: "non_ppo_continue", label: "Non-PPO Continue" },
    { key: "retrain", label: "Retrain" },
    { key: "continue", label: "Continue" },
  ].forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.activeModelFilters.strategies.has(option.key) ? " active" : ""}`;
    button.textContent = option.label;
    button.addEventListener("click", () => {
      if (state.activeModelFilters.strategies.has(option.key)) {
        state.activeModelFilters.strategies.delete(option.key);
      } else {
        state.activeModelFilters.strategies.add(option.key);
      }
      applyActiveModelFilters(experiment);
      render().catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderFamilyQuickActions(experiment) {
  const host = byId("familyQuickActions");
  if (!host) return;
  host.innerHTML = "";
  const families = [...new Set(
    [...buildModelMetadataMap(experiment).values()]
      .map((metadata) => metadata.family)
      .filter((family) => !!family)
  )].sort((a, b) => a.localeCompare(b));
  families.forEach((family) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.activeModelFilters.families.has(String(family)) ? " active" : ""}`;
    button.textContent = family;
    button.addEventListener("click", () => {
      if (state.activeModelFilters.families.has(String(family))) {
        state.activeModelFilters.families.delete(String(family));
      } else {
        state.activeModelFilters.families.add(String(family));
      }
      applyActiveModelFilters(experiment);
      render().catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function getDefaultSelectedModels(experiment) {
  const models = orderedAvailableModels(experiment);
  if (models.length <= DEFAULT_LEARNED_SELECTION_LIMIT) return models;
  const aggregateModels = models.filter((name) => isAggregateSeriesModel(name));
  if (models.length >= LARGE_EXPERIMENT_MODEL_COUNT && aggregateModels.length) {
    const selected = [];
    aggregateModels.filter((name) => isBaselineModelName(name)).forEach((name) => {
      if (selected.length >= LARGE_EXPERIMENT_DEFAULT_SELECTION_LIMIT) return;
      selected.push(name);
    });
    aggregateModels.filter((name) => !isBaselineModelName(name)).forEach((name) => {
      if (selected.length >= LARGE_EXPERIMENT_DEFAULT_SELECTION_LIMIT) return;
      selected.push(name);
    });
    return selected;
  }
  const selected = [];
  const preferredBaselines = models.filter((name) => (
    BASELINE_MODELS.has(name) ||
    ([...BASELINE_MODELS].some((baselineName) => name === `${baselineName}_AGGREGATE`))
  ));
  preferredBaselines.forEach((name) => {
    if (!selected.includes(name)) selected.push(name);
  });
  models.filter((name) => isAggregateSeriesModel(name) && !isBaselineModelName(name)).forEach((name) => {
    if (selected.length >= DEFAULT_LEARNED_SELECTION_LIMIT) return;
    if (!selected.includes(name)) selected.push(name);
  });
  models.filter((name) => isBaselineModelName(name) && !selected.includes(name)).forEach((name) => {
    if (selected.length >= DEFAULT_LEARNED_SELECTION_LIMIT) return;
    selected.push(name);
  });
  models.forEach((name) => {
    if (selected.length >= DEFAULT_LEARNED_SELECTION_LIMIT) return;
    if (!selected.includes(name)) selected.push(name);
  });
  return selected;
}

function renderXAxisPercentControl() {
  const slider = byId("xAxisPercentRange");
  const valueLabel = byId("xAxisPercentValue");
  if (!slider || !valueLabel) return;
  slider.value = String(state.xAxisPercent);
  valueLabel.textContent = `${state.xAxisPercent}%`;
  slider.oninput = (event) => {
    state.xAxisPercent = clampXAxisPercent(event.target.value);
    valueLabel.textContent = `${state.xAxisPercent}%`;
    renderCharts(getActiveExperiment(), state.renderToken).catch(renderFailure);
  };
}

function renderTimelineModeControl() {
  const host = byId("timelineModeToggles");
  if (!host) return;
  host.innerHTML = "";
  [
    { key: "actual_dates", label: "Actual Dates" },
    { key: "aligned_start", label: "Align Starts" },
  ].forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.timelineMode === option.key ? " active" : ""}`;
    button.textContent = option.label;
    button.addEventListener("click", () => {
      if (state.timelineMode === option.key) return;
      state.timelineMode = option.key;
      renderTimelineModeControl();
      renderCharts(getActiveExperiment(), state.renderToken).catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderPortfolioZeroCrossingControl() {
  const host = byId("portfolioZeroCrossingToggles");
  if (!host) return;
  host.innerHTML = "";
  [
    { key: "stop_at_zero", label: "Stop At $0", full: false },
    { key: "show_full_curve", label: "Show Full Curve", full: true },
  ].forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.showFullPortfolioAfterZero === option.full ? " active" : ""}`;
    button.textContent = option.label;
    button.addEventListener("click", () => {
      if (state.showFullPortfolioAfterZero === option.full) return;
      state.showFullPortfolioAfterZero = option.full;
      renderPortfolioZeroCrossingControl();
      renderCharts(getActiveExperiment(), state.renderToken).catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderEnsembleControl() {
  const host = byId("ensembleToggles");
  if (!host) return;
  host.innerHTML = "";
  [
    { label: "Ensemble Off", enabled: false },
    { label: "Ensemble On", enabled: true },
  ].forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `toggle${state.showEnsemble === option.enabled ? " active" : ""}`;
    button.textContent = option.label;
    button.addEventListener("click", () => {
      if (state.showEnsemble === option.enabled) return;
      state.showEnsemble = option.enabled;
      renderEnsembleControl();
      renderCharts(getActiveExperiment(), state.renderToken).catch(renderFailure);
    });
    host.appendChild(button);
  });
}

function renderThresholdAccuracyRangeControls() {
  const startSlider = byId("thresholdAccuracyRangeStart");
  const endSlider = byId("thresholdAccuracyRangeEnd");
  const stepSlider = byId("thresholdAccuracyStep");
  const startValue = byId("thresholdAccuracyRangeStartValue");
  const endValue = byId("thresholdAccuracyRangeEndValue");
  const stepValue = byId("thresholdAccuracyStepValue");
  if (!startSlider || !endSlider || !stepSlider || !startValue || !endValue || !stepValue) return;

  startSlider.value = String(state.thresholdAccuracyRangeStart);
  endSlider.value = String(state.thresholdAccuracyRangeEnd);
  stepSlider.value = String(state.thresholdAccuracyStep);
  startValue.textContent = formatPercentLabel(state.thresholdAccuracyRangeStart);
  endValue.textContent = formatPercentLabel(state.thresholdAccuracyRangeEnd);
  stepValue.textContent = formatPercentLabel(state.thresholdAccuracyStep);

  if (startSlider.dataset.bound === "true") return;
  const rerender = () => renderCharts(getActiveExperiment(), state.renderToken).catch(renderFailure);

  startSlider.addEventListener("input", () => {
    const nextValue = Math.max(50, Math.min(Number(startSlider.value), state.thresholdAccuracyRangeEnd));
    state.thresholdAccuracyRangeStart = nextValue;
    if (Number(startSlider.value) !== nextValue) {
      startSlider.value = String(nextValue);
    }
    startValue.textContent = formatPercentLabel(nextValue);
    rerender();
  });

  endSlider.addEventListener("input", () => {
    const nextValue = Math.min(100, Math.max(Number(endSlider.value), state.thresholdAccuracyRangeStart));
    state.thresholdAccuracyRangeEnd = nextValue;
    if (Number(endSlider.value) !== nextValue) {
      endSlider.value = String(nextValue);
    }
    endValue.textContent = formatPercentLabel(nextValue);
    rerender();
  });

  stepSlider.addEventListener("input", () => {
    const nextValue = Math.max(0.25, Number(stepSlider.value));
    state.thresholdAccuracyStep = nextValue;
    if (Number(stepSlider.value) !== nextValue) {
      stepSlider.value = String(nextValue);
    }
    stepValue.textContent = formatPercentLabel(nextValue);
    rerender();
  });

  startSlider.dataset.bound = "true";
  endSlider.dataset.bound = "true";
  stepSlider.dataset.bound = "true";
}

function isSkippedAction(action) {
  const numeric = Number(action);
  return !Number.isFinite(numeric) || (numeric !== 0 && numeric !== 1);
}

function buildTestDiagnostics(seriesEntry) {
  if (!seriesEntry) return null;
  const actions = Array.isArray(seriesEntry.testActions) ? seriesEntry.testActions : [];
  const labels = Array.isArray(seriesEntry.testLabels) ? seriesEntry.testLabels : [];
  const timestamps = Array.isArray(seriesEntry.testTimestamps) ? seriesEntry.testTimestamps : [];
  const chosenProbs = Array.isArray(seriesEntry.testChosenActionProbabilities) ? seriesEntry.testChosenActionProbabilities : [];
  const limit = Math.min(actions.length, labels.length, timestamps.length);
  const points = [];
  for (let index = 0; index < limit; index += 1) {
    const action = Number(actions[index]);
    const label = Number(labels[index]);
    const timestamp = timestamps[index];
    const date = new Date(timestamp);
    const hasDate = !Number.isNaN(date.getTime());
    const skipped = isSkippedAction(action);
    const correct = !skipped && Number.isFinite(label) && action === label;
    const probabilityValueRaw = chosenProbs[index];
    const probabilityValue = probabilityValueRaw == null ? null : Number(probabilityValueRaw);
    points.push({
      index,
      action,
      label,
      timestamp,
      dayKey: hasDate ? date.toISOString().slice(0, 10) : `step-${index}`,
      hour: hasDate ? date.getUTCHours() : 0,
      skipped,
      correct,
      wrong: !skipped && !correct,
      direction: skipped ? null : (action === 1 ? "up" : "down"),
      chosenProbability: Number.isFinite(probabilityValue) ? probabilityValue : null,
    });
  }
  return {
    label: seriesEntry.label,
    points,
  };
}

function renderTestOutcomeMatrix(series) {
  const host = byId("testOutcomeMatrixChart");
  if (!host) return;
  host.innerHTML = "";
  if (series.length !== 1) {
    host.appendChild(createEmptyState("Select exactly one variant/model series to view the test outcome matrix."));
    return;
  }
  const diagnostics = buildTestDiagnostics(series[0]);
  if (!diagnostics || !diagnostics.points.length) {
    host.appendChild(createEmptyState("No test data available for the selected series."));
    return;
  }

  const dayKeys = [...new Set(diagnostics.points.map((point) => point.dayKey))];
  const pointByCell = new Map(diagnostics.points.map((point) => [`${point.dayKey}|${point.hour}`, point]));
  const wrapper = document.createElement("div");
  wrapper.style.display = "grid";
  wrapper.style.gridTemplateColumns = `70px repeat(${dayKeys.length}, minmax(22px, 1fr))`;
  wrapper.style.gap = "4px";
  wrapper.style.alignItems = "center";
  wrapper.style.minWidth = `${Math.max(320, 70 + (dayKeys.length * 26))}px`;

  const blank = document.createElement("div");
  wrapper.appendChild(blank);
  dayKeys.forEach((dayKey) => {
    const label = document.createElement("div");
    label.className = "axis";
    label.style.textAlign = "center";
    label.style.fontSize = "11px";
    label.textContent = dayKey.slice(5);
    wrapper.appendChild(label);
  });

  for (let hour = 0; hour < 24; hour += 1) {
    const hourLabel = document.createElement("div");
    hourLabel.className = "axis";
    hourLabel.style.textAlign = "right";
    hourLabel.style.paddingRight = "8px";
    hourLabel.textContent = String(hour).padStart(2, "0");
    wrapper.appendChild(hourLabel);
    dayKeys.forEach((dayKey) => {
      const point = pointByCell.get(`${dayKey}|${hour}`);
      const cell = document.createElement("div");
      cell.style.height = "22px";
      cell.style.borderRadius = "4px";
      cell.style.border = "1px solid rgba(255,255,255,0.08)";
      cell.style.background = point
        ? (point.skipped ? "#f6c85f" : (point.correct ? "#52d273" : "#ff7f7f"))
        : "rgba(255,255,255,0.04)";
      cell.title = point
        ? `${diagnostics.label}\n${point.timestamp}\nHour: ${point.hour}\nOutcome: ${point.skipped ? "Skipped" : (point.correct ? "Correct" : "Wrong")}\nAction: ${point.action}\nLabel: ${point.label}\nChosen probability: ${point.chosenProbability == null ? "N/A" : point.chosenProbability.toFixed(4)}`
        : `${dayKey} ${String(hour).padStart(2, "0")}:00\nNo scored test point`;
      wrapper.appendChild(cell);
    });
  }
  host.appendChild(wrapper);
}

function buildThresholdAccuracyPoints(seriesEntry) {
  const diagnostics = buildTestDiagnostics(seriesEntry);
  if (!diagnostics) return [];
  const start = Math.max(50, Math.min(100, Number(state.thresholdAccuracyRangeStart) || 50));
  const end = Math.max(start, Math.min(100, Number(state.thresholdAccuracyRangeEnd) || 100));
  const step = Math.max(0.25, Number(state.thresholdAccuracyStep) || 1);
  const points = [];
  for (let threshold = start; threshold <= end + 1e-9 && points.length < 500; threshold += step) {
    const normalizedThreshold = Number(threshold.toFixed(6));
    const probabilityThreshold = normalizedThreshold / 100;
    const eligiblePoints = diagnostics.points.filter((point) => (
      !point.skipped
      && point.chosenProbability != null
      && point.chosenProbability >= probabilityThreshold
    ));
    const taken = eligiblePoints.length;
    const correct = eligiblePoints.filter((point) => point.correct).length;
    const accuracy = taken > 0 ? (correct / taken) : null;
    points.push({
      x: normalizedThreshold,
      y: accuracy,
      stepIndex: normalizedThreshold,
      usesTimestamp: false,
      tooltip: `${seriesEntry.label}\nThreshold: ${normalizedThreshold.toFixed(2)}%\nAccuracy: ${accuracy == null ? "N/A" : `${(accuracy * 100).toFixed(2)}%`}\nActions above threshold: ${taken}\nCorrect above threshold: ${correct}`,
    });
  }
  return points;
}

function renderThresholdAccuracyChart(series) {
  const chartSeries = series
    .map((item) => ({
      label: item.label,
      color: item.color,
      points: buildThresholdAccuracyPoints(item),
    }))
    .filter((item) => item.points.some((point) => Number.isFinite(point.y)));
  const thresholdTicks = [];
  for (let threshold = 50; threshold <= 100; threshold += 5) {
    thresholdTicks.push({ x: threshold, label: `${threshold}%` });
  }
  drawLineChart(
    "thresholdAccuracyChart",
    "thresholdAccuracyLegend",
    chartSeries,
    "Accuracy",
    [],
    {
      emptyMessage: "Select at least one variant/model series with chosen-probability test diagnostics to render threshold accuracy.",
      xTicks: thresholdTicks,
      xLabel: "Threshold (%)",
      yScale: "linear",
    },
  );
}

function buildCumulativeAccuracyPoints(actions, labels, timestamps, mode) {
  const hasValidTimestamp = Array.isArray(timestamps) && timestamps.some((timestamp) => Number.isFinite(parseTimestampMs(timestamp)));
  const limit = Math.min(actions.length, labels.length);
  const points = [];
  let correct = 0;
  let scored = 0;
  let lastAccuracy = null;
  for (let index = 0; index < limit; index += 1) {
    const action = Number(actions[index]);
    const label = Number(labels[index]);
    const dateLabel = Array.isArray(timestamps) ? (timestamps[index] || null) : null;
    const timestampMs = parseTimestampMs(dateLabel);
    if (!Number.isFinite(action) || !Number.isFinite(label) || isSkippedAction(action)) {
      points.push({
        x: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs) ? timestampMs : index,
        y: lastAccuracy,
        dateLabel,
        stepIndex: index,
        usesTimestamp: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs),
      });
      continue;
    }
    scored += 1;
    if (action === label) correct += 1;
    lastAccuracy = correct / scored;
    points.push({
      x: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs) ? timestampMs : index,
      y: lastAccuracy,
      dateLabel,
      stepIndex: index,
      usesTimestamp: mode === "actual_dates" && hasValidTimestamp && Number.isFinite(timestampMs),
    });
  }
  return points;
}

function computeThresholdActionCounts(actions, labels) {
  const limit = Math.min(actions.length, labels.length);
  let taken = 0;
  let correct = 0;
  for (let index = 0; index < limit; index += 1) {
    const action = Number(actions[index]);
    const label = Number(labels[index]);
    if (!Number.isFinite(action) || !Number.isFinite(label) || isSkippedAction(action)) {
      continue;
    }
    taken += 1;
    if (action === label) {
      correct += 1;
    }
  }
  return { taken, correct };
}

function computeRolling24AccuracyExtremes(actions, labels) {
  const limit = Math.min(actions.length, labels.length);
  if (limit < 24) {
    return { best: 0, lowest: 0 };
  }

  let best = -Infinity;
  let lowest = Infinity;

  for (let endIndex = 23; endIndex < limit; endIndex += 1) {
    let correct = 0;
    let validCount = 0;
    for (let windowIndex = endIndex - 23; windowIndex <= endIndex; windowIndex += 1) {
      const action = Number(actions[windowIndex]);
      const label = Number(labels[windowIndex]);
      if (!Number.isFinite(action) || !Number.isFinite(label)) continue;
      validCount += 1;
      if (action === label) correct += 1;
    }
    const accuracy = validCount === 24 ? (correct / 24) : 0;
    if (accuracy > best) best = accuracy;
    if (accuracy < lowest) lowest = accuracy;
  }

  if (!Number.isFinite(best) || !Number.isFinite(lowest)) {
    return { best: 0, lowest: 0 };
  }

  return { best, lowest };
}

function computeSummaryStatsFromActions(actions, labels, chosenProbabilities = []) {
  const rewards = [0];
  let totalReward = 0;
  let scoredCount = 0;
  let correctCount = 0;
  const normalizedProbabilities = Array.isArray(chosenProbabilities) ? [...chosenProbabilities] : [];
  const limit = Math.min(actions.length, labels.length);
  while (normalizedProbabilities.length < limit) {
    normalizedProbabilities.push(null);
  }
  for (let index = 0; index < limit; index += 1) {
    const action = Number(actions[index]);
    const label = Number(labels[index]);
    let reward = 0;
    if (Number.isFinite(action) && Number.isFinite(label) && !isSkippedAction(action)) {
      reward = action === label ? 1 : -1;
      scoredCount += 1;
      if (action === label) correctCount += 1;
    }
    totalReward += reward;
    rewards.push(totalReward);
  }
  return {
    cumulativeRewards: rewards,
    totalReward,
    meanReward: limit ? (totalReward / limit) : 0,
    accuracy: scoredCount ? (correctCount / scoredCount) : 0,
    rolling24Accuracy: computeRolling24AccuracyExtremes(actions, labels),
  };
}

function computeZeroCrossingFromBalances(balances) {
  let previousValue = null;
  let previousIndex = null;
  for (let index = 0; index < balances.length; index += 1) {
    const numeric = Number(balances[index]);
    if (!Number.isFinite(numeric)) continue;
    if (numeric <= 0) {
      return {
        crossed: true,
        index,
        value: numeric,
        previous_index: previousIndex,
        previous_value: previousValue,
      };
    }
    previousValue = numeric;
    previousIndex = index;
  }
  return {
    crossed: false,
    index: null,
    value: null,
    previous_index: previousIndex,
    previous_value: previousValue,
  };
}

function simulatePortfolioBalances(actions, labels, portfolioSizingMode, startingBalance, incrementValue) {
  const normalizedStartingBalance = Number.isFinite(Number(startingBalance)) ? Number(startingBalance) : 10;
  const normalizedIncrementValue = Number.isFinite(Number(incrementValue)) ? Math.max(0, Number(incrementValue)) : 1;
  const balances = [normalizedStartingBalance];
  let peakBalance = normalizedStartingBalance;
  const limit = Math.min(actions.length, labels.length);

  for (let index = 0; index < limit; index += 1) {
    const action = Number(actions[index]);
    const label = Number(labels[index]);
    const balance = balances[balances.length - 1];
    let pnl = 0;
    if (Number.isFinite(action) && Number.isFinite(label) && !isSkippedAction(action)) {
      if (portfolioSizingMode === "current_percent") {
        pnl = Math.max(1, normalizedIncrementValue * Math.max(0, balance));
      } else if (portfolioSizingMode === "peak_percent") {
        pnl = Math.max(1, normalizedIncrementValue * Math.max(0, peakBalance));
      } else {
        pnl = normalizedIncrementValue;
      }
      if (action !== label) pnl *= -1;
    }
    const nextBalance = balance + pnl;
    balances.push(nextBalance);
    if (nextBalance > peakBalance) peakBalance = nextBalance;
  }

  return {
    balances,
    zeroCrossing: computeZeroCrossingFromBalances(balances),
  };
}

function getPortfolioCacheKey(item) {
  return [
    item.key,
    state.portfolioSizingMode,
    state.portfolioStartingBalance,
    state.portfolioIncrementValue,
    state.timelineMode,
    state.showFullPortfolioAfterZero ? "full" : "stop",
  ].join("||");
}

function buildDynamicPortfolioAnalysis(item) {
  const cacheKey = getPortfolioCacheKey(item);
  if (portfolioCurveCache.has(cacheKey)) {
    return portfolioCurveCache.get(cacheKey);
  }
  const simulation = simulatePortfolioBalances(
    item.testActions || [],
    item.testLabels || [],
    state.portfolioSizingMode,
    state.portfolioStartingBalance,
    state.portfolioIncrementValue,
  );
  const portfolioPoints = buildCurvePoints(simulation.balances, item.testTimestamps || [], state.timelineMode);
  const portfolioAnalysis = analyzePortfolioPoints(portfolioPoints);
  const result = {
    points: portfolioPoints,
    visiblePoints: portfolioAnalysis.visiblePoints,
    crossedZero: portfolioAnalysis.crossedZero,
    zeroCrossPoint: portfolioAnalysis.zeroCrossPoint,
    startingBalance: state.portfolioStartingBalance,
  };
  portfolioCurveCache.set(cacheKey, result);
  return result;
}

function averageNumericSeries(seriesList) {
  const normalized = (Array.isArray(seriesList) ? seriesList : []).filter((series) => Array.isArray(series) && series.length);
  if (!normalized.length) return [];
  const minLength = Math.min(...normalized.map((series) => series.length));
  const result = [];
  for (let index = 0; index < minLength; index += 1) {
    let total = 0;
    let count = 0;
    normalized.forEach((series) => {
      const value = Number(series[index]);
      if (!Number.isFinite(value)) return;
      total += value;
      count += 1;
    });
    result.push(count ? (total / count) : 0);
  }
  return result;
}

function averageNumbers(values, fallback = null) {
  const numericValues = (Array.isArray(values) ? values : [])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  if (!numericValues.length) return fallback;
  return numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length;
}

function buildEnsembleSeriesForVariant(variantName, models) {
  const contributors = (Array.isArray(models) ? models : []).filter(Boolean);
  if (!contributors.length) return null;
  const reference = contributors[0];
  const testActionLists = contributors.map((model) => Array.isArray(model.test?.actions) ? model.test.actions : []);
  const labelLists = contributors.map((model) => Array.isArray(model.test?.labels) ? model.test.labels : []);
  const timestampLists = contributors.map((model) => Array.isArray(model.test?.timestamps) ? model.test.timestamps : []);
  const probabilityLists = contributors.map((model) => Array.isArray(model.test?.chosen_action_probabilities) ? model.test.chosen_action_probabilities : []);
  const minLength = Math.min(
    ...testActionLists.map((values) => values.length),
    ...labelLists.map((values) => values.length),
    ...timestampLists.map((values) => values.length),
  );
  if (!Number.isFinite(minLength) || minLength <= 0) return null;

  const ensembleActions = [];
  const ensembleLabels = [];
  const ensembleTimestamps = [];
  const ensembleProbabilities = [];

  for (let index = 0; index < minLength; index += 1) {
    const voteCounts = new Map();
    const firstSeenOrder = [];
    contributors.forEach((model, modelIndex) => {
      const numericAction = Number(testActionLists[modelIndex][index]);
      if (!Number.isFinite(numericAction)) return;
      if (!voteCounts.has(numericAction)) {
        voteCounts.set(numericAction, 0);
        firstSeenOrder.push(numericAction);
      }
      voteCounts.set(numericAction, voteCounts.get(numericAction) + 1);
    });
    if (!voteCounts.size) continue;
    const ensembleAction = firstSeenOrder.reduce((bestAction, candidateAction) => {
      if (bestAction == null) return candidateAction;
      const bestCount = voteCounts.get(bestAction) || 0;
      const candidateCount = voteCounts.get(candidateAction) || 0;
      return candidateCount > bestCount ? candidateAction : bestAction;
    }, null);

    const chosenProbabilities = [];
    contributors.forEach((model, modelIndex) => {
      const numericAction = Number(testActionLists[modelIndex][index]);
      if (!Number.isFinite(numericAction) || numericAction !== ensembleAction) return;
      const probability = Number(probabilityLists[modelIndex][index]);
      if (Number.isFinite(probability)) chosenProbabilities.push(probability);
    });
    if (!chosenProbabilities.length) {
      contributors.forEach((model, modelIndex) => {
        const probability = Number(probabilityLists[modelIndex][index]);
        if (Number.isFinite(probability)) chosenProbabilities.push(probability);
      });
    }

    ensembleActions.push(ensembleAction);
    ensembleLabels.push(Number(labelLists[0][index]));
    ensembleTimestamps.push(timestampLists[0][index] || null);
    ensembleProbabilities.push(
      chosenProbabilities.length
        ? (chosenProbabilities.reduce((sum, value) => sum + value, 0) / chosenProbabilities.length)
        : null,
    );
  }

  if (!ensembleActions.length) return null;

  const testStats = computeSummaryStatsFromActions(ensembleActions, ensembleLabels, ensembleProbabilities);
  const trainRewardAverage = averageNumericSeries(
    contributors.map((model) => Array.isArray(model.train?.cumulative_rewards) ? model.train.cumulative_rewards : []),
  );

  const portfolioPayload = {};
  ["fixed_dollar", "peak_fraction", "current_fraction"].forEach((portfolioType) => {
    const fallbackStart = Number(reference.portfolio?.[portfolioType]?.starting_balance);
    const startingBalance = averageNumbers(
      contributors.map((model) => model.portfolio?.[portfolioType]?.starting_balance),
      fallbackStart,
    );
    const sizingMode = portfolioType === "fixed_dollar"
      ? "fixed_dollar"
      : (portfolioType === "current_fraction" ? "current_percent" : "peak_percent");
    const simulated = simulatePortfolioBalances(
      ensembleActions,
      ensembleLabels,
      sizingMode,
      startingBalance,
      portfolioType === "fixed_dollar" ? 1 : 0.05,
    );
    portfolioPayload[portfolioType] = {
      balances: simulated.balances,
      starting_balance: startingBalance,
      zero_crossing: simulated.zeroCrossing,
    };
  });

  return {
    name: "ENSEMBLE",
    is_baseline: false,
    window_size: null,
    series_window_scale: null,
    train: {
      cumulative_rewards: trainRewardAverage,
      timestamps: Array.isArray(reference.train?.timestamps) ? reference.train.timestamps : [],
    },
    test: {
      accuracy: testStats.accuracy,
      cumulative_rewards: testStats.cumulativeRewards,
      actions: ensembleActions,
      labels: ensembleLabels,
      chosen_action_probabilities: ensembleProbabilities,
      timestamps: ensembleTimestamps,
      cumulative_accuracy: [],
      rolling24_accuracy: testStats.rolling24Accuracy,
    },
    portfolio: portfolioPayload,
    _variantName: variantName,
    _plotting: reference._plotting || {},
  };
}

function analyzePortfolioPoints(points) {
  const visiblePoints = [];
  let crossedZero = false;
  let zeroCrossPoint = null;

  for (const point of points) {
    visiblePoints.push(point);
    if (point.y <= 0) {
      crossedZero = true;
      zeroCrossPoint = point;
      break;
    }
  }

  return {
    visiblePoints,
    crossedZero,
    zeroCrossPoint,
  };
}

function selectedSeriesKeys(experiment) {
  const keys = [];
  orderedAvailableVariants(experiment).forEach((variantName) => {
    if (!state.selectedVariants.has(variantName)) return;
    orderedAvailableModelsForVariant(experiment, variantName).forEach((modelName) => {
      if (!state.selectedModels.has(modelName)) return;
      keys.push(`${variantName}||${modelName}`);
    });
    if (state.showEnsemble && state.selectedModels.size) {
      keys.push(`${variantName}||ENSEMBLE`);
    }
  });
  return keys;
}

function maxRenderedSeriesForSelection(selectedKeys) {
  if (!Array.isArray(selectedKeys) || !selectedKeys.length) return MAX_RENDERED_SERIES;
  const allAggregate = selectedKeys.every((key) => {
    const parts = String(key || "").split("||");
    const modelName = parts.length > 1 ? parts[1] : "";
    return isAggregateSeriesModel(modelName);
  });
  return allAggregate ? MAX_RENDERED_AGGREGATE_SERIES : MAX_RENDERED_SERIES;
}

async function ensureSeriesDataLoaded(experiment) {
  if (!experiment) return;
  if (!experiment.seriesCache) {
    experiment.seriesCache = new Map();
  }
  const missingPairs = [];
  orderedAvailableVariants(experiment).forEach((variantName) => {
    if (!state.selectedVariants.has(variantName)) return;
    orderedAvailableModelsForVariant(experiment, variantName).forEach((modelName) => {
      if (!state.selectedModels.has(modelName)) return;
      const cacheKey = `${variantName}||${modelName}`;
      if (!experiment.seriesCache.has(cacheKey)) {
        missingPairs.push({ variantName, modelName });
      }
    });
  });
  if (!missingPairs.length) return;
  const params = new URLSearchParams();
  [...new Set(missingPairs.map((item) => item.variantName))].forEach((variantName) => params.append("variant", variantName));
  [...new Set(missingPairs.map((item) => item.modelName))].forEach((modelName) => params.append("model", modelName));
  const payload = await fetchJson(`/api/experiments/${experiment.experiment_id}/series?${params.toString()}`);
  Object.entries(payload?.variants || {}).forEach(([variantName, variant]) => {
    (variant.models || []).forEach((model) => {
      experiment.seriesCache.set(`${variantName}||${model.name}`, { ...model, _variantName: variantName, _plotting: variant.plotting || {} });
    });
  });
}

function buildSelectedSeries(experiment) {
  const series = [];
  orderedAvailableVariants(experiment).forEach((variantName) => {
    if (!state.selectedVariants.has(variantName)) return;
    const variant = experiment.variants?.[variantName];
    if (!variant) return;
    const selectedVariantModels = [];
    orderedAvailableModelsForVariant(experiment, variantName).forEach((modelName) => {
      if (!state.selectedModels.has(modelName)) return;
      const model = experiment.seriesCache?.get(`${variantName}||${modelName}`);
      if (!model) return;
      selectedVariantModels.push(model);
      const label = formatSeriesLabel(variantName, model.name);
      const color = getSeriesColor(label);
      const trainTimestamps = model.train?.timestamps?.length ? model.train.timestamps : (model._plotting?.train_timestamps || []);
      const testTimestamps = model.test?.timestamps?.length ? model.test.timestamps : (model._plotting?.test_timestamps || []);
      const thresholdCounts = computeThresholdActionCounts(
        model.test?.actions || [],
        model.test?.labels || [],
      );
      series.push({
        key: label,
        label,
        color,
        envVersion: model.env_version || variant.env_version || null,
        isBaseline: !!model.is_baseline,
        windowSize: Number(model.window_size),
        seriesWindowScale: model.series_window_scale || null,
        testActions: model.test?.actions || [],
        testLabels: model.test?.labels || [],
        testTimestamps: testTimestamps,
        testChosenActionProbabilities: model.test?.chosen_action_probabilities || [],
        trainRewards: buildCurvePoints(model.train?.cumulative_rewards || [], trainTimestamps, state.timelineMode),
        testRewards: buildCurvePoints(model.test?.cumulative_rewards || [], testTimestamps, state.timelineMode),
        cumulativeAccuracy: buildCumulativeAccuracyPoints(
          model.test?.actions || [],
          model.test?.labels || [],
          testTimestamps,
          state.timelineMode,
        ),
        rolling24Accuracy: model.test?.rolling24_accuracy || computeRolling24AccuracyExtremes(
          model.test?.actions || [],
          model.test?.labels || [],
        ),
        thresholdActionCount: Number(model.test?.threshold_action_count ?? model.test?.accuracy_scored_count ?? thresholdCounts.taken),
        thresholdCorrectCount: Number(model.test?.threshold_correct_count ?? thresholdCounts.correct),
        accuracy: Number(model.test?.accuracy),
      });
    });
    if (state.showEnsemble) {
      const ensembleModel = buildEnsembleSeriesForVariant(variantName, selectedVariantModels);
      if (ensembleModel) {
        const label = formatSeriesLabel(variantName, ensembleModel.name);
        const color = getSeriesColor(label);
        const trainTimestamps = ensembleModel.train?.timestamps?.length ? ensembleModel.train.timestamps : (ensembleModel._plotting?.train_timestamps || []);
        const testTimestamps = ensembleModel.test?.timestamps?.length ? ensembleModel.test.timestamps : (ensembleModel._plotting?.test_timestamps || []);
        const thresholdCounts = computeThresholdActionCounts(
          ensembleModel.test?.actions || [],
          ensembleModel.test?.labels || [],
        );
        series.push({
          key: label,
          label,
          color,
          envVersion: ensembleModel.env_version || variant.env_version || null,
          isBaseline: false,
          windowSize: null,
          seriesWindowScale: null,
          testActions: ensembleModel.test?.actions || [],
          testLabels: ensembleModel.test?.labels || [],
          testTimestamps: testTimestamps,
          testChosenActionProbabilities: ensembleModel.test?.chosen_action_probabilities || [],
          trainRewards: buildCurvePoints(ensembleModel.train?.cumulative_rewards || [], trainTimestamps, state.timelineMode),
          testRewards: buildCurvePoints(ensembleModel.test?.cumulative_rewards || [], testTimestamps, state.timelineMode),
          cumulativeAccuracy: buildCumulativeAccuracyPoints(
            ensembleModel.test?.actions || [],
            ensembleModel.test?.labels || [],
            testTimestamps,
            state.timelineMode,
          ),
          rolling24Accuracy: ensembleModel.test?.rolling24_accuracy || computeRolling24AccuracyExtremes(
            ensembleModel.test?.actions || [],
            ensembleModel.test?.labels || [],
          ),
          thresholdActionCount: Number(ensembleModel.test?.threshold_action_count ?? ensembleModel.test?.accuracy_scored_count ?? thresholdCounts.taken),
          thresholdCorrectCount: Number(ensembleModel.test?.threshold_correct_count ?? thresholdCounts.correct),
          accuracy: Number(ensembleModel.test?.accuracy),
        });
      }
    }
  });
  return series;
}

function collectSeriesXValues(series, pointKey) {
  return series.flatMap((item) => (
    Array.isArray(item?.[pointKey]) ? item[pointKey].map((point) => point.x) : []
  )).filter((value) => Number.isFinite(Number(value)));
}

function chooseReferenceTimestamps(experiment, splitName) {
  const ordered = orderedAvailableVariants(experiment);
  const preferred = ["full", "derived_market_hours", "market_hours", "outside_market_hours"];
  const merged = [...preferred.filter((name) => ordered.includes(name)), ...ordered.filter((name) => !preferred.includes(name))];
  for (const variantName of merged) {
    if (!state.selectedVariants.has(variantName)) continue;
    const variant = experiment.variants?.[variantName];
    const timestamps = splitName === "train"
      ? (variant?.plotting?.train_timestamps || [])
      : (variant?.plotting?.test_timestamps || []);
    if (timestamps.length) return timestamps;
    for (const modelName of orderedAvailableModelsForVariant(experiment, variantName)) {
      if (!state.selectedModels.has(modelName)) continue;
      const cachedModel = experiment.seriesCache?.get(`${variantName}||${modelName}`);
      const cachedTimestamps = splitName === "train"
        ? (cachedModel?._plotting?.train_timestamps || [])
        : (cachedModel?._plotting?.test_timestamps || []);
      if (cachedTimestamps.length) return cachedTimestamps;
    }
  }
  return [];
}

function renderFixed5kRunsTable() {
  const host = byId("fixed5kRunsTable");
  const computeButton = byId("fixed5kRunsComputeButton");
  if (!host) return;
  if (computeButton) {
    computeButton.disabled = state.fixed5kRunsState === "loading";
    computeButton.textContent = state.fixed5kRunsState === "loading" ? "Computing..." : "Compute";
  }
  host.innerHTML = "";
  if (state.fixed5kRunsState === "idle") {
    host.appendChild(createEmptyState("Press Compute to load comparable fixed 5K runs."));
    return;
  }
  if (state.fixed5kRunsState === "loading") {
    host.appendChild(createEmptyState("Computing comparable fixed 5K runs..."));
    return;
  }
  if (state.fixed5kRunsState === "error") {
    host.appendChild(createEmptyState(`Failed to compute comparable fixed 5K runs: ${state.fixed5kRunsError || "Unknown error"}`));
    return;
  }
  const payload = state.fixed5kRuns;
  const rows = Array.isArray(payload?.rows) ? payload.rows : [];
  const windowInfo = payload?.window;
  if (!rows.length) {
    host.appendChild(createEmptyState("No shared fixed 5K runs found in the current experiments root."));
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap";
  if (windowInfo && windowInfo.start && windowInfo.end) {
    const summary = document.createElement("div");
    summary.className = "subtle";
    summary.style.marginBottom = "10px";
    summary.textContent = `Shared test window: ${windowInfo.start} to ${windowInfo.end} (${windowInfo.count} rows)`;
    wrapper.appendChild(summary);
  }

  const table = document.createElement("table");
  table.className = "compare-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Experiment</th>
        <th>Model</th>
        <th>Variant</th>
        <th>Accuracy</th>
        <th>Net Wins / Day</th>
        <th>PnL</th>
      </tr>
    </thead>
  `;
  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const dailyNetWins = Number(row.daily_net_wins);
    const pnlPct = Number(row.pnl_pct);
    tr.innerHTML = `
      <td>#${escapeHtml(row.experiment_id)} ${escapeHtml(row.experiment_name)}</td>
      <td>${escapeHtml(row.model_name)}</td>
      <td>${escapeHtml(row.variant)}</td>
      <td>${escapeHtml(formatAccuracyPercent(row.accuracy))}</td>
      <td class="${dailyNetWins >= 0 ? "metric-good" : "metric-bad"}">${escapeHtml(formatSignedWinsPerDay(row.daily_net_wins))}</td>
      <td class="${pnlPct >= 0 ? "metric-good" : "metric-bad"}">${escapeHtml(formatSignedPercent(row.pnl_pct))}</td>
    `;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrapper.appendChild(table);
  host.appendChild(wrapper);
}

function renderSummaries(experiment, series) {
  byId("summaryExperiment").textContent = experiment ? `#${experiment.experiment_id} ${experiment.name}` : "-";
  byId("summaryVariants").textContent = String(state.selectedVariants.size);
  byId("summaryModels").textContent = String(state.selectedModels.size);
  byId("summarySeries").textContent = String(series.length);
}

function renderChartHostsMessage(message) {
  [
    "accuracyChart",
    "cumulativeAccuracyChart",
    "thresholdActionChart",
    "trainRewardChart",
    "testRewardChart",
    "testNetWinsPerDayChart",
    "portfolioChart",
    "testOutcomeMatrixChart",
    "thresholdAccuracyChart",
  ].forEach((hostId) => {
    const host = byId(hostId);
    if (!host) return;
    host.innerHTML = "";
    host.appendChild(createEmptyState(message));
  });
  ["accuracyLegend", "cumulativeAccuracyLegend", "thresholdActionLegend", "trainRewardLegend", "testRewardLegend", "testNetWinsPerDayLegend", "portfolioLegend", "thresholdAccuracyLegend"].forEach((hostId) => {
    const host = byId(hostId);
    if (host) host.innerHTML = "";
  });
}

function renderFailure(error) {
  renderChartHostsMessage(`Failed to load experiment dashboard: ${error.message}`);
}

function buildPortfolioChartSeries(series) {
  return series
    .map((item) => {
      const portfolioAnalysis = buildDynamicPortfolioAnalysis(item);
      return { item, portfolioAnalysis };
    })
    .filter(({ portfolioAnalysis }) => Array.isArray(portfolioAnalysis.points) && portfolioAnalysis.points.length)
    .map(({ item, portfolioAnalysis }) => {
      const portfolioBasePoints = state.showFullPortfolioAfterZero ? portfolioAnalysis.points : portfolioAnalysis.visiblePoints;
      const visiblePoints = trimPointsByPercent(portfolioBasePoints, state.xAxisPercent);
      const lastVisiblePoint = visiblePoints.length ? visiblePoints[visiblePoints.length - 1] : null;
      const stopMarker = (
        !state.showFullPortfolioAfterZero &&
        portfolioAnalysis.crossedZero &&
        portfolioAnalysis.zeroCrossPoint &&
        lastVisiblePoint &&
        portfolioAnalysis.zeroCrossPoint.x <= lastVisiblePoint.x
      ) ? portfolioAnalysis.zeroCrossPoint : null;
      return {
        label: item.label,
        legendLabel: portfolioAnalysis.crossedZero && !state.showFullPortfolioAfterZero ? `${item.label} X` : item.label,
        color: item.color,
        points: visiblePoints,
        stopMarker,
        startingBalance: portfolioAnalysis.startingBalance,
      };
    })
    .filter((item) => item.points.length);
}

function drawPortfolioChartFromSeries(experiment, series) {
  const portfolioSeries = buildPortfolioChartSeries(series);
  const testTickValues = state.timelineMode === "actual_dates"
    ? collectSeriesXValues(portfolioSeries, "points")
    : trimTimestampsByPercent(chooseReferenceTimestamps(experiment, "test"), state.xAxisPercent);
  const portfolioStartingBalances = [...new Set(
    portfolioSeries.map((item) => item.startingBalance).filter((value) => Number.isFinite(value)),
  )].sort((a, b) => a - b);
  const portfolioReferenceLines = [
    { key: "zero", value: 0, color: "#ff8a8a", dasharray: "4 4", label: "0" },
    ...portfolioStartingBalances.map((value, index) => ({
      key: `start-${index}`,
      value,
      color: "#ffd166",
      dasharray: "10 6",
      label: portfolioStartingBalances.length === 1 ? "Start" : `Start ${Number(value).toFixed(2)}`,
    })),
  ];
  const portfolioLabel = state.portfolioSizingMode === "current_percent"
    ? `Portfolio balance (Current percent, ${(Number(state.portfolioIncrementValue) * 100).toFixed(2)}%)`
    : (state.portfolioSizingMode === "peak_percent"
      ? `Portfolio balance (Peak percent, ${(Number(state.portfolioIncrementValue) * 100).toFixed(2)}%)`
      : `Portfolio balance (Fixed $${Number(state.portfolioIncrementValue).toFixed(2)})`);
  const lineRenderer = chooseRenderer(
    "line",
    portfolioSeries.length,
    portfolioSeries.reduce((sum, item) => sum + item.points.length, 0),
  );
  (lineRenderer === "canvas" ? drawLineChartCanvas : drawLineChart)(
    "portfolioChart",
    "portfolioLegend",
    portfolioSeries,
    portfolioLabel,
    testTickValues,
    { referenceLines: portfolioReferenceLines, tickMode: state.timelineMode, yScale: "piecewise_log_over_1000" },
  );
}

async function rerenderPortfolioOnly(experiment, renderToken) {
  if (!experiment) return;
  await ensureSeriesDataLoaded(experiment);
  if (renderToken !== state.renderToken) return;
  const series = buildSelectedSeries(experiment);
  drawPortfolioChartFromSeries(experiment, series);
}

async function renderCharts(experiment, renderToken) {
  const selectedKeys = selectedSeriesKeys(experiment);
  renderFixed5kRunsTable();
  renderChartHostsMessage("Loading selected series...");
  await ensureSeriesDataLoaded(experiment);
  if (renderToken !== state.renderToken) return;
  const series = buildSelectedSeries(experiment);
  renderSummaries(experiment, series);

  const accuracyGroups = series
    .filter((item) => Number.isFinite(item.accuracy))
    .map((item) => ({
      label: item.label,
      accuracy: item.accuracy,
      best24: Number(item.rolling24Accuracy?.best),
      lowest24: Number(item.rolling24Accuracy?.lowest),
    }))
    .filter((item) => Number.isFinite(item.best24) && Number.isFinite(item.lowest24))
    .sort((a, b) => a.accuracy - b.accuracy || a.label.localeCompare(b.label));
  const trainSeries = series
    .filter((item) => item.trainRewards.length)
    .map((item) => ({
      label: item.label,
      color: item.color,
      points: trimPointsByPercent(item.trainRewards, state.xAxisPercent),
    }))
    .filter((item) => item.points.length);
  const testSeries = series
    .filter((item) => item.testRewards.length)
    .map((item) => ({
      label: item.label,
      color: item.color,
      points: trimPointsByPercent(item.testRewards, state.xAxisPercent),
    }))
    .filter((item) => item.points.length);
  const cumulativeAccuracySeries = series
    .filter((item) => item.cumulativeAccuracy.length)
    .map((item) => ({
      label: item.label,
      color: item.color,
      points: trimPointsByPercent(item.cumulativeAccuracy, state.xAxisPercent),
    }))
    .filter((item) => item.points.length);
  const thresholdActionGroups = series
    .filter((item) => Number.isFinite(item.thresholdActionCount) || Number.isFinite(item.thresholdCorrectCount))
    .map((item) => ({
      label: item.label,
      thresholdActionsTaken: Number(item.thresholdActionCount),
      thresholdActionsCorrect: Number(item.thresholdCorrectCount),
    }))
    .filter((item) => Number.isFinite(item.thresholdActionsTaken) && Number.isFinite(item.thresholdActionsCorrect))
    .sort((a, b) => a.thresholdActionsTaken - b.thresholdActionsTaken || a.label.localeCompare(b.label));
  const testNetWinsPerDayBars = series
    .filter((item) => Array.isArray(item.testActions) && Array.isArray(item.testLabels) && item.testActions.length && item.testLabels.length)
    .map((item) => ({
      label: item.label,
      color: item.color,
      value: computeNetWinsPerDay(item.testActions, item.testLabels, item.testTimestamps, item.envVersion),
    }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => a.value - b.value || a.label.localeCompare(b.label));
  const portfolioSeries = buildPortfolioChartSeries(series);
  const trainTickValues = state.timelineMode === "actual_dates"
    ? collectSeriesXValues(trainSeries, "points")
    : trimTimestampsByPercent(chooseReferenceTimestamps(experiment, "train"), state.xAxisPercent);
  const testTickValues = state.timelineMode === "actual_dates"
    ? [
        ...collectSeriesXValues(testSeries, "points"),
        ...collectSeriesXValues(cumulativeAccuracySeries, "points"),
        ...collectSeriesXValues(portfolioSeries, "points"),
      ]
    : trimTimestampsByPercent(chooseReferenceTimestamps(experiment, "test"), state.xAxisPercent);
  const lineRenderer = chooseRenderer(
    "line",
    Math.max(trainSeries.length, testSeries.length, cumulativeAccuracySeries.length, portfolioSeries.length),
    trainSeries.reduce((sum, item) => sum + item.points.length, 0)
      + testSeries.reduce((sum, item) => sum + item.points.length, 0)
      + cumulativeAccuracySeries.reduce((sum, item) => sum + item.points.length, 0)
      + portfolioSeries.reduce((sum, item) => sum + item.points.length, 0),
  );
  const barRenderer = chooseRenderer(
    "bar",
    testNetWinsPerDayBars.length,
    testNetWinsPerDayBars.length,
  );

  drawGroupedBarChart(
    "accuracyChart",
    "accuracyLegend",
    accuracyGroups,
    "Test accuracy",
    [
      { key: "accuracy", label: "Overall test accuracy", color: "#5ab0ff" },
      { key: "best24", label: "Best 24-point accuracy", color: "#6fe3a4" },
      { key: "lowest24", label: "Lowest 24-point accuracy", color: "#ff9f68" },
    ],
  );
  (lineRenderer === "canvas" ? drawLineChartCanvas : drawLineChart)("cumulativeAccuracyChart", "cumulativeAccuracyLegend", cumulativeAccuracySeries, "Cumulative test accuracy", testTickValues, { tickMode: state.timelineMode });
  drawGroupedBarChart(
    "thresholdActionChart",
    "thresholdActionLegend",
    thresholdActionGroups,
    "Action count",
    [
      { key: "thresholdActionsTaken", label: "Actions above threshold", color: "#5ab0ff" },
      { key: "thresholdActionsCorrect", label: "Correct actions above threshold", color: "#6fe3a4" },
    ],
    {
      showHalfReference: false,
      valueFormatter: formatIntegerCount,
      axisTickFormatter: (value) => formatIntegerCount(value),
    },
  );
  (lineRenderer === "canvas" ? drawLineChartCanvas : drawLineChart)("trainRewardChart", "trainRewardLegend", trainSeries, "Cumulative train reward", trainTickValues, { tickMode: state.timelineMode });
  (lineRenderer === "canvas" ? drawLineChartCanvas : drawLineChart)("testRewardChart", "testRewardLegend", testSeries, "Cumulative test reward", testTickValues, { tickMode: state.timelineMode });
  (barRenderer === "canvas" ? drawBarChartCanvas : drawBarChart)("testNetWinsPerDayChart", "testNetWinsPerDayLegend", testNetWinsPerDayBars, "Net wins per day");
  drawPortfolioChartFromSeries(experiment, series);
  renderTestOutcomeMatrix(series);
  renderThresholdAccuracyChart(series);
}

function attachQuickActions(experiment) {
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.onclick = () => {
      const action = button.dataset.action;
      const models = orderedAvailableModels(experiment);
      if (action === "select-all-models") {
        clearActiveModelFilters();
        state.userClearedModels = false;
        state.selectedModels = new Set(models);
      } else if (action === "clear-models") {
        clearActiveModelFilters();
        state.userClearedModels = true;
        state.selectedModels = new Set();
      }
      render().catch(renderFailure);
    };
  });
}

async function render() {
  const renderToken = ++state.renderToken;
  renderFixed5kRunsTable();
  const experiment = getActiveExperiment();
  if (!experiment) {
    renderSummaries(null, []);
    renderChartHostsMessage("No experiment loaded.");
    return;
  }
  ensureSelections(experiment);
  renderVariantToggles(experiment);
  renderModelToggles(experiment);
  renderThresholdQuickActions(experiment);
  renderTrainSizeQuickActions(experiment);
  renderStrategyQuickActions(experiment);
  renderFamilyQuickActions(experiment);
  renderPortfolioToggles();
  renderPortfolioControls();
  renderXAxisPercentControl();
  renderTimelineModeControl();
  renderPortfolioZeroCrossingControl();
  renderEnsembleControl();
  renderThresholdAccuracyRangeControls();
  attachQuickActions(experiment);
  await renderCharts(experiment, renderToken);
}

async function setActiveExperiment(experimentId) {
  state.activeExperimentId = String(experimentId);
  if (!state.experimentCache.has(state.activeExperimentId)) {
    const payload = await loadExperimentPayload(state.activeExperimentId);
    state.experimentCache.set(state.activeExperimentId, payload);
  }
  state.selectedVariants = new Set();
  state.selectedModels = new Set();
  clearActiveModelFilters();
  state.userClearedModels = false;
  renderExperimentSelector();
  await render();
  if (state.pendingInitialExperimentId && String(state.pendingInitialExperimentId) === state.activeExperimentId) {
    state.pendingInitialExperimentId = null;
    state.pendingInitialVariants = [];
  }
}

async function init() {
  await loadExperimentIndex();
  renderExperimentSelector();
  renderFixed5kRunsTable();
  const select = byId("experimentSelect");
  select.addEventListener("change", async (event) => {
    await setActiveExperiment(event.target.value);
  });
  const fixed5kRunsComputeButton = byId("fixed5kRunsComputeButton");
  if (fixed5kRunsComputeButton) {
    fixed5kRunsComputeButton.addEventListener("click", () => {
      computeFixed5kRuns().catch(renderFailure);
    });
  }
  if (state.experimentIndex.length) {
    const initialExperimentId = state.pendingInitialExperimentId || state.experimentIndex[0].experiment_id;
    await setActiveExperiment(initialExperimentId);
  } else {
    await render();
  }
}

init().catch(renderFailure);
"""


class ExperimentDashboardHandler(BaseHTTPRequestHandler):
    store: ExperimentStore

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def _send_html(self, document: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(document.encode("utf-8"), "text/html; charset=utf-8", status=status)

    def _send_js(self, script: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(script.encode("utf-8"), "application/javascript; charset=utf-8", status=status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/experiments")
                self.end_headers()
                return
            if path == "/experiments":
                self._send_html(build_html())
                return
            if path == "/app.js":
                self._send_js(build_app_js())
                return
            if path == "/api/health":
                self._send_json({"ok": True})
                return
            if path == "/api/experiments":
                self.store.refresh_index()
                self._send_json(self.store.list_experiments())
                return
            if path == "/api/fixed-5k-runs":
                self.store.refresh_index()
                self._send_json(self.store.list_fixed_5k_runs())
                return
            if path.startswith("/experiments/") and path.endswith("/snapshot"):
                parts = path.split("/")
                if len(parts) >= 5:
                    experiment_id = parts[-3]
                    variant_name = parts[-2]
                    manifest_path = self.store._manifest_paths.get(str(experiment_id))
                    if manifest_path is None:
                        self._send_html("<html><body>Unknown experiment.</body></html>", status=HTTPStatus.NOT_FOUND)
                        return
                    manifest, _ = self.store._load_manifest(manifest_path)
                    variants = manifest.get("variants", {}) if isinstance(manifest, dict) else {}
                    variant_payload = variants.get(variant_name) if isinstance(variants, dict) else None
                    snapshot_path = self.store._snapshot_path_from_payload(variant_payload) if isinstance(variant_payload, dict) else None
                    if snapshot_path is None or not snapshot_path.exists():
                        self._send_html("<html><body>Snapshot not found.</body></html>", status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_html(snapshot_path.read_text(encoding="utf-8"))
                    return
            if path.startswith("/api/experiments/") and path.endswith("/series"):
                experiment_id = path.split("/")[-2]
                query = parse_qs(parsed.query)
                variant_names = [value for value in query.get("variant", []) if value]
                model_names = [value for value in query.get("model", []) if value]
                payload = self.store.get_experiment_series(experiment_id, variant_names, model_names)
                if payload is None:
                    self._send_json({"error": f"Unknown experiment: {experiment_id}"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(payload)
                return
            if path.startswith("/api/experiments/"):
                experiment_id = path.split("/")[-1]
                payload = self.store.get_experiment(experiment_id)
                if payload is None:
                    self._send_json({"error": f"Unknown experiment: {experiment_id}"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(payload)
                return
        except ArtifactNotBuiltError as err:
            self._send_json({"error": str(err)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._send_html(
            f"<html><body style='font-family:sans-serif;background:#111827;color:#e5e7eb;padding:24px;'><h1>404</h1><p>No route for <code>{html.escape(path)}</code>.</p></body></html>",
            status=HTTPStatus.NOT_FOUND,
        )


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    store = ExperimentStore(experiments_root)
    handler_class = type(
        "BoundExperimentDashboardHandler",
        (ExperimentDashboardHandler,),
        {"store": store},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "experiments_root": str(experiments_root),
                "url": f"http://{args.host}:{args.port}/experiments",
                "experiment_count": len(store.list_experiments()),
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
