from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_experiment_tables import detect_family
from scripts.serve_experiment_dashboard import build_html
from src.utils.experiment_support import read_summary_rows_parquet

DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "EXPERIMENTS"
ARTIFACT_VERSION = 1
CATALOG_FILE_NAME = "dashboard_catalog.json"
SERIES_DIR_NAME = "dashboard_series"
SNAPSHOT_FILE_NAME = "dashboard_snapshot.html"
BASELINE_MODELS = {"RANDOM", "ALWAYS_UP", "ALWAYS_DOWN"}


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


def maybe_int(value: Any) -> int | None:
    numeric = maybe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def format_daily(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.3f}"


def format_int(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+d}"


def compute_days_for_variant(test_timestamps: list[Any], fallback_count: int, time_variant: str) -> float:
    if time_variant in {"MARKET_HOURS", "OUTSIDE_MARKET_HOURS"} and test_timestamps:
        unique_days = {str(timestamp)[:10] for timestamp in test_timestamps if timestamp}
        if unique_days:
            return float(len(unique_days))
    if fallback_count > 0:
        return fallback_count / 24.0
    return 1.0


def build_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_rows = maybe_int(summary.get("dataset_metadata", {}).get("test_rows")) or 5000
    rows: list[dict[str, Any]] = []
    for model in summary.get("models", []):
        if not isinstance(model, dict):
            continue
        test_data = model.get("test", {})
        portfolio = model.get("portfolio", {})
        fixed_portfolio = portfolio.get("fixed_dollar", {}) if isinstance(portfolio, dict) else {}
        trade_pnls = fixed_portfolio.get("trade_pnls", [])
        wins = sum(1 for pnl in trade_pnls if maybe_float(pnl) is not None and float(pnl) > 0)
        losses = sum(1 for pnl in trade_pnls if maybe_float(pnl) is not None and float(pnl) < 0)
        net_wins = wins - losses
        timestamps = test_data.get("timestamps", []) if isinstance(test_data, dict) else []
        time_variant = str(model.get("time_variant") or "FULL").upper()
        active_days = compute_days_for_variant(timestamps if isinstance(timestamps, list) else [], dataset_rows, time_variant)
        pnl_pct = maybe_float(fixed_portfolio.get("pnl_pct"))
        daily_pnl_pct = (pnl_pct / active_days) if pnl_pct is not None and active_days > 0 else None
        daily_net_wins = (net_wins / active_days) if active_days > 0 else None
        family = str(model.get("family") or detect_family(str(model.get("name") or "")))
        rows.append(
            {
                "family": family,
                "train_length": str(model.get("train_length") or ""),
                "window_length": str(model.get("window_length") or ""),
                "window_name": str(model.get("normalized_window_name") or model.get("window_name") or ""),
                "model_variation": str(model.get("model_variation") or "BASE"),
                "time_variant": time_variant,
                "selection_split": str(model.get("selection_split") or ""),
                "threshold": maybe_int(model.get("threshold_pct")),
                "accuracy": maybe_float(test_data.get("accuracy")) if isinstance(test_data, dict) else None,
                "validation_accuracy": maybe_float((((model.get("validation") or {}).get("metrics") or {}).get("taken_accuracy"))),
                "validation_daily_net_wins": maybe_float((((model.get("validation") or {}).get("metrics") or {}).get("daily_net_wins"))),
                "coverage": maybe_float(model.get("coverage")),
                "pnl_pct": pnl_pct,
                "daily_pnl_pct": daily_pnl_pct,
                "net_wins": net_wins,
                "daily_net_wins": daily_net_wins,
                "status": str(model.get("status") or ""),
                "has_saved_model": bool(str(model.get("saved_model_path") or "").strip()),
            }
        )
    return rows


def render_html(rows: list[dict[str, Any]], *, sort_by: str, title: str) -> str:
    def sort_value(row: dict[str, Any]) -> float:
        value = row.get(sort_by)
        numeric = maybe_float(value)
        return numeric if numeric is not None else float("-inf")

    ordered_rows = sorted(rows, key=sort_value, reverse=True)
    headers = [
        ("family", "Family", "text"),
        ("train_length", "Train Length", "text"),
        ("window_length", "Window Length", "text"),
        ("window_name", "Window Name", "text"),
        ("model_variation", "Model Variation", "text"),
        ("time_variant", "Time Variant", "text"),
        ("selection_split", "Selection Split", "text"),
        ("threshold", "Threshold", "num"),
        ("accuracy", "Accuracy", "num"),
        ("validation_accuracy", "Validation Accuracy", "num"),
        ("validation_daily_net_wins", "Validation Daily Net Wins", "num"),
        ("coverage", "Coverage", "num"),
        ("pnl_pct", "%PnL", "num"),
        ("daily_pnl_pct", "Daily %PnL", "num"),
        ("net_wins", "Net Wins", "num"),
        ("daily_net_wins", "Daily Net Wins", "num"),
        ("status", "Status", "text"),
        ("has_saved_model", "Saved Model", "text"),
    ]

    body_rows = []
    for row in ordered_rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('family', '')))}</td>"
            f"<td>{html.escape(str(row.get('train_length', '')))}</td>"
            f"<td>{html.escape(str(row.get('window_length', '')))}</td>"
            f"<td>{html.escape(str(row.get('window_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('model_variation', '')))}</td>"
            f"<td>{html.escape(str(row.get('time_variant', '')))}</td>"
            f"<td>{html.escape(str(row.get('selection_split', '')))}</td>"
            f"<td data-sort='{'' if row.get('threshold') is None else row.get('threshold')}'>{html.escape(str(row.get('threshold') if row.get('threshold') is not None else ''))}</td>"
            f"<td data-sort='{'' if row.get('accuracy') is None else row.get('accuracy')}'>{html.escape(format_accuracy(row.get('accuracy')))}</td>"
            f"<td data-sort='{'' if row.get('validation_accuracy') is None else row.get('validation_accuracy')}'>{html.escape(format_accuracy(row.get('validation_accuracy')))}</td>"
            f"<td data-sort='{'' if row.get('validation_daily_net_wins') is None else row.get('validation_daily_net_wins')}'>{html.escape(format_daily(row.get('validation_daily_net_wins')))}</td>"
            f"<td data-sort='{'' if row.get('coverage') is None else row.get('coverage')}'>{html.escape(format_accuracy(row.get('coverage')))}</td>"
            f"<td data-sort='{'' if row.get('pnl_pct') is None else row.get('pnl_pct')}'>{html.escape(format_pct(row.get('pnl_pct')))}</td>"
            f"<td data-sort='{'' if row.get('daily_pnl_pct') is None else row.get('daily_pnl_pct')}'>{html.escape(format_pct(row.get('daily_pnl_pct')))}</td>"
            f"<td data-sort='{row.get('net_wins', '')}'>{html.escape(format_int(row.get('net_wins')))}</td>"
            f"<td data-sort='{'' if row.get('daily_net_wins') is None else row.get('daily_net_wins')}'>{html.escape(format_daily(row.get('daily_net_wins')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{'yes' if row.get('has_saved_model') else 'no'}</td>"
            "</tr>"
        )

    header_html = "".join(
        f"<th data-key='{key}' data-kind='{kind}'>{html.escape(label)}</th>"
        for key, label, kind in headers
    )
    table_rows_html = "\n".join(body_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #0b1220;
      color: #dbe7f3;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1 {{
      margin: 0 0 16px 0;
      font-size: 1.6rem;
    }}
    .subtle {{
      color: #9fb0c3;
      margin-bottom: 18px;
    }}
    .table-wrap {{
      border: 1px solid #334155;
      border-radius: 14px;
      overflow: auto;
      background: #0f1726;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1100px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #243142;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #142034;
      color: #9fb0c3;
      text-transform: uppercase;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      cursor: pointer;
    }}
    tr:hover td {{
      background: rgba(255,255,255,0.03);
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="subtle">Sortable direct-model comparison table for EXPERIMENTSV2/1. Click any column header to sort.</div>
  <div class="table-wrap">
    <table id="comparisonTable">
      <thead>
        <tr>{header_html}</tr>
      </thead>
      <tbody>
        {table_rows_html}
      </tbody>
    </table>
  </div>
  <script>
    const table = document.getElementById("comparisonTable");
    const tbody = table.querySelector("tbody");
    let currentSort = {{ index: 16, direction: "desc" }};
    function getCellValue(row, index, kind) {{
      const cell = row.children[index];
      const raw = cell.dataset.sort ?? cell.textContent.trim();
      if (kind === "num") {{
        const numeric = Number(raw);
        return Number.isFinite(numeric) ? numeric : Number.NEGATIVE_INFINITY;
      }}
      return String(raw).toLowerCase();
    }}
    function sortRows(index, kind) {{
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const direction = (currentSort.index === index && currentSort.direction === "desc") ? "asc" : "desc";
      rows.sort((left, right) => {{
        const a = getCellValue(left, index, kind);
        const b = getCellValue(right, index, kind);
        if (a < b) return direction === "asc" ? -1 : 1;
        if (a > b) return direction === "asc" ? 1 : -1;
        return 0;
      }});
      currentSort = {{ index, direction }};
      tbody.innerHTML = "";
      rows.forEach((row) => tbody.appendChild(row));
    }}
    table.querySelectorAll("th").forEach((header, index) => {{
      header.addEventListener("click", () => sortRows(index, header.dataset.kind || "text"));
    }});
  </script>
</body>
</html>
"""


def load_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    else:
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
    return json.loads(text)


def write_json_atomic(path: str | Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp.replace(path)


def load_variant_summary(variant_payload: dict[str, Any]) -> dict[str, Any]:
    summary_payload = variant_payload.get("summary", {})
    summary_path_raw = variant_payload.get("summary_path")
    if not summary_path_raw:
        if isinstance(summary_payload, dict) and isinstance(summary_payload.get("models"), list):
            return summary_payload
        return summary_payload if isinstance(summary_payload, dict) else {}
    summary = load_json(summary_path_raw)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact dashboard artifacts for experiment comparisons.")
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument("--experiments", nargs="*", help="Experiment ids to build. Omit with --all to build every experiment.")
    parser.add_argument("--all", action="store_true", help="Build artifacts for every experiment manifest under the root.")
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even when the existing catalog is fresh.")
    parser.add_argument("--snapshots", action="store_true", help="Emit per-variant dashboard snapshot HTML files.")
    return parser.parse_args()


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


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def slugify_text(value: str) -> str:
    chars: list[str] = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
            continue
        if not previous_dash:
            chars.append("-")
            previous_dash = True
    text = "".join(chars).strip("-")
    return text or "model"


def is_baseline_model_name(model_name: str) -> bool:
    if model_name in BASELINE_MODELS:
        return True
    for baseline_name in BASELINE_MODELS:
        if model_name == f"{baseline_name}_AGGREGATE" or model_name.startswith(f"{baseline_name}_M"):
            return True
    return False


def is_skipped_action(action: Any) -> bool:
    try:
        numeric = int(action)
    except (TypeError, ValueError):
        return True
    return numeric not in {0, 1}


def compute_cumulative_accuracy(actions: list[Any], labels: list[Any]) -> list[float | None]:
    limit = min(len(actions), len(labels))
    correct = 0
    scored = 0
    values: list[float | None] = []
    for index in range(limit):
        try:
            action = int(actions[index])
            label = int(labels[index])
        except (TypeError, ValueError):
            values.append(None)
            continue
        if is_skipped_action(action):
            values.append(None)
            continue
        scored += 1
        if action == label:
            correct += 1
        values.append(correct / scored)
    return values


def compute_threshold_action_counts(actions: list[Any], labels: list[Any]) -> dict[str, int]:
    scored_count = 0
    correct_count = 0
    for action, label in zip(actions, labels):
        try:
            action_int = int(action)
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        if is_skipped_action(action_int):
            continue
        scored_count += 1
        if action_int == label_int:
            correct_count += 1
    return {
        "threshold_action_count": scored_count,
        "threshold_correct_count": correct_count,
    }


def compute_rolling24_extremes(actions: list[Any], labels: list[Any]) -> dict[str, float | None]:
    scored: list[int] = []
    for action, label in zip(actions, labels):
        try:
            action_int = int(action)
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        if is_skipped_action(action_int):
            continue
        scored.append(1 if action_int == label_int else 0)
    if len(scored) < 24:
        return {"best": None, "lowest": None}
    window_total = sum(scored[:24])
    best = window_total / 24
    lowest = best
    for index in range(24, len(scored)):
        window_total += scored[index] - scored[index - 24]
        accuracy = window_total / 24
        if accuracy > best:
            best = accuracy
        if accuracy < lowest:
            lowest = accuracy
    return {"best": best, "lowest": lowest}


def compute_zero_crossing(balances: list[Any]) -> dict[str, Any]:
    previous_balance: float | None = None
    previous_index: int | None = None
    for index, balance in enumerate(balances):
        numeric_balance = maybe_float(balance)
        if numeric_balance is None:
            continue
        if numeric_balance <= 0:
            return {
                "crossed": True,
                "index": index,
                "value": numeric_balance,
                "previous_index": previous_index,
                "previous_value": previous_balance,
            }
        previous_balance = numeric_balance
        previous_index = index
    return {
        "crossed": False,
        "index": None,
        "value": None,
        "previous_index": previous_index,
        "previous_value": previous_balance,
    }


def parse_timestamp_ms(value: Any) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return __import__("datetime").datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def align_timestamps_to_length(timestamps: list[Any], target_length: int) -> list[Any]:
    raw_timestamps = list(timestamps or [])
    if target_length <= 0:
        return []
    if not raw_timestamps:
        return [None] * target_length
    if len(raw_timestamps) == target_length:
        return raw_timestamps
    if len(raw_timestamps) == target_length - 1:
        return [raw_timestamps[0], *raw_timestamps]
    if len(raw_timestamps) > target_length:
        return raw_timestamps[len(raw_timestamps) - target_length :]
    padded = list(raw_timestamps)
    while len(padded) < target_length:
        padded.insert(0, padded[0])
    return padded


def sort_series_by_timestamps(timestamps: list[Any], values: list[Any]) -> tuple[list[Any], list[Any]]:
    aligned_timestamps = align_timestamps_to_length(timestamps, len(values))
    rows = []
    for index, value in enumerate(values):
        timestamp = aligned_timestamps[index] if index < len(aligned_timestamps) else None
        sort_key = parse_timestamp_ms(timestamp)
        rows.append((sort_key is None, sort_key if sort_key is not None else float(index), index, timestamp, value))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    sorted_timestamps = [item[3] for item in rows]
    sorted_values = [item[4] for item in rows]
    return sorted_timestamps, sorted_values


def sort_split_rows(
    timestamps: list[Any],
    *,
    actions: list[Any] | None = None,
    labels: list[Any] | None = None,
    chosen_action_probabilities: list[Any] | None = None,
) -> dict[str, list[Any]]:
    actions_list = list(actions or [])
    labels_list = list(labels or [])
    probabilities_list = list(chosen_action_probabilities or [])
    limit = min(len(timestamps or []), len(actions_list), len(labels_list))
    if limit <= 0:
        return {
            "timestamps": list(timestamps or []),
            "actions": actions_list,
            "labels": labels_list,
            "chosen_action_probabilities": probabilities_list,
        }
    rows = []
    for index in range(limit):
        timestamp = list(timestamps)[index]
        sort_key = parse_timestamp_ms(timestamp)
        probability = probabilities_list[index] if index < len(probabilities_list) else None
        rows.append((sort_key is None, sort_key if sort_key is not None else float(index), index, timestamp, actions_list[index], labels_list[index], probability))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return {
        "timestamps": [item[3] for item in rows],
        "actions": [item[4] for item in rows],
        "labels": [item[5] for item in rows],
        "chosen_action_probabilities": [item[6] for item in rows],
    }


def normalize_curve(split_data: dict[str, Any], *, include_actions: bool) -> dict[str, Any]:
    raw_timestamps = list(split_data.get("timestamps", []))
    sorted_timestamps = raw_timestamps
    sorted_actions: list[Any] = []
    sorted_labels: list[Any] = []
    sorted_probabilities: list[Any] = []
    if include_actions:
        sorted_rows = sort_split_rows(
            raw_timestamps,
            actions=split_data.get("actions", []),
            labels=split_data.get("labels", []),
            chosen_action_probabilities=split_data.get("chosen_action_probabilities", []),
        )
        sorted_timestamps = sorted_rows["timestamps"]
        sorted_actions = sorted_rows["actions"]
        sorted_labels = sorted_rows["labels"]
        sorted_probabilities = sorted_rows["chosen_action_probabilities"]
    else:
        sorted_timestamps, _ = sort_series_by_timestamps(raw_timestamps, list(raw_timestamps))

    _, sorted_cumulative_rewards = sort_series_by_timestamps(sorted_timestamps, split_data.get("cumulative_rewards", []))
    payload = {
        "accuracy": maybe_float(split_data.get("accuracy")),
        "accuracy_scored_count": split_data.get("accuracy_scored_count"),
        "total_reward": maybe_float(split_data.get("total_reward")),
        "mean_reward": maybe_float(split_data.get("mean_reward")),
        "cumulative_rewards": sorted_cumulative_rewards,
        "timestamps": sorted_timestamps,
    }
    if include_actions:
        counts = compute_threshold_action_counts(sorted_actions, sorted_labels)
        scored_count = counts["threshold_action_count"]
        correct_count = counts["threshold_correct_count"]
        payload.update(
            {
                "accuracy": (correct_count / scored_count) if scored_count else 0.0,
                "accuracy_scored_count": scored_count,
                "threshold_action_count": scored_count,
                "threshold_correct_count": correct_count,
                "actions": sorted_actions,
                "labels": sorted_labels,
                "chosen_action_probabilities": sorted_probabilities,
                "cumulative_accuracy": compute_cumulative_accuracy(sorted_actions, sorted_labels),
                "rolling24_accuracy": compute_rolling24_extremes(sorted_actions, sorted_labels),
            }
        )
    return payload


def normalize_portfolio(portfolio_data: dict[str, Any], *, test_timestamps: list[Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for portfolio_name in ["fixed_dollar", "peak_fraction", "current_fraction"]:
        values = portfolio_data.get(portfolio_name, {})
        if not isinstance(values, dict):
            values = {}
        _, balances = sort_series_by_timestamps(test_timestamps, values.get("balances", []))
        normalized[portfolio_name] = {
            "balances": balances,
            "starting_balance": maybe_float(values.get("starting_balance")),
            "final_balance": maybe_float(values.get("final_balance")),
            "pnl": maybe_float(values.get("pnl")),
            "pnl_pct": maybe_float(values.get("pnl_pct")),
            "zero_crossing": compute_zero_crossing(balances),
        }
    return normalized


def build_model_artifact(model_data: dict[str, Any]) -> dict[str, Any]:
    train = model_data.get("train", {})
    test = model_data.get("test", {})
    if not isinstance(train, dict):
        train = {}
    if not isinstance(test, dict):
        test = {}
    model_name = str(model_data.get("name", "UNKNOWN"))
    normalized_train = normalize_curve(train, include_actions=False)
    normalized_test = normalize_curve(test, include_actions=True)
    return {
        "name": model_name,
        "is_baseline": is_baseline_model_name(model_name),
        "family": model_data.get("family") or model_data.get("model_family"),
        "train_length": model_data.get("train_length"),
        "time_variant": model_data.get("time_variant"),
        "threshold_pct": model_data.get("threshold_pct"),
        "model_variation": model_data.get("model_variation"),
        "window_length": model_data.get("window_length"),
        "window_name": model_data.get("normalized_window_name") or model_data.get("window_name"),
        "window_checkpoints": model_data.get("window_checkpoints", []),
        "window_size": model_data.get("window_size"),
        "series_window_scale": model_data.get("series_window_scale"),
        "model_family": model_data.get("model_family"),
        "source_variant": model_data.get("source_variant"),
        "data_variant": model_data.get("data_variant"),
        "env_version": model_data.get("env_version"),
        "requested_train_rows": model_data.get("requested_train_rows"),
        "actual_train_rows": model_data.get("actual_train_rows"),
        "is_forward_window_series": bool(model_data.get("is_forward_window_series")),
        "is_aggregate_series": bool(model_data.get("is_aggregate_series")),
        "window_index": model_data.get("window_index"),
        "window_train_start": model_data.get("window_train_start"),
        "window_train_end": model_data.get("window_train_end"),
        "window_test_start": model_data.get("window_test_start"),
        "window_test_end": model_data.get("window_test_end"),
        "dataset_metadata": model_data.get("dataset_metadata", {}),
        "train": normalized_train,
        "test": normalized_test,
        "portfolio": normalize_portfolio(model_data.get("portfolio", {}), test_timestamps=normalized_test.get("timestamps", [])),
    }


def build_catalog_entry(model_artifact: dict[str, Any], series_file: str) -> dict[str, Any]:
    portfolio_summary = {
        portfolio_name: {
            "starting_balance": values.get("starting_balance"),
            "final_balance": values.get("final_balance"),
            "pnl": values.get("pnl"),
            "pnl_pct": values.get("pnl_pct"),
            "crossed_zero": bool(values.get("zero_crossing", {}).get("crossed")),
            "zero_cross_index": values.get("zero_crossing", {}).get("index"),
        }
        for portfolio_name, values in model_artifact.get("portfolio", {}).items()
        if isinstance(values, dict)
    }
    return {
        "name": model_artifact.get("name"),
        "is_baseline": model_artifact.get("is_baseline"),
        "family": model_artifact.get("family") or model_artifact.get("model_family"),
        "train_length": model_artifact.get("train_length"),
        "time_variant": model_artifact.get("time_variant"),
        "threshold_pct": model_artifact.get("threshold_pct"),
        "model_variation": model_artifact.get("model_variation"),
        "window_length": model_artifact.get("window_length"),
        "window_name": model_artifact.get("window_name"),
        "window_size": model_artifact.get("window_size"),
        "series_window_scale": model_artifact.get("series_window_scale"),
        "model_family": model_artifact.get("model_family"),
        "source_variant": model_artifact.get("source_variant"),
        "data_variant": model_artifact.get("data_variant"),
        "env_version": model_artifact.get("env_version"),
        "is_forward_window_series": model_artifact.get("is_forward_window_series"),
        "is_aggregate_series": model_artifact.get("is_aggregate_series"),
        "window_index": model_artifact.get("window_index"),
        "train_accuracy": model_artifact.get("train", {}).get("accuracy"),
        "test_accuracy": model_artifact.get("test", {}).get("accuracy"),
        "point_counts": {
            "train_rewards": len(model_artifact.get("train", {}).get("cumulative_rewards", [])),
            "test_rewards": len(model_artifact.get("test", {}).get("cumulative_rewards", [])),
            "test_actions": len(model_artifact.get("test", {}).get("actions", [])),
        },
        "rolling24_accuracy": model_artifact.get("test", {}).get("rolling24_accuracy"),
        "portfolio": portfolio_summary,
        "series_file": series_file,
    }


def has_renderable_dashboard_series(model_data: dict[str, Any]) -> bool:
    if not isinstance(model_data, dict):
        return False
    test = model_data.get("test", {})
    if not isinstance(test, dict):
        return False
    timestamps = test.get("timestamps", [])
    actions = test.get("actions", [])
    cumulative_rewards = test.get("cumulative_rewards", [])
    return bool(timestamps) and (bool(actions) or bool(cumulative_rewards))


def build_variant_catalog(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    variant_name: str,
    summary: dict[str, Any],
    summary_path: Path,
    catalog_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    plotting = summary.get("plotting", {})
    if not isinstance(plotting, dict):
        plotting = {}
    return {
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": int(manifest.get("experiment_id")),
        "experiment_name": manifest.get("name", f"Experiment {manifest.get('experiment_id')}"),
        "root_dir": manifest.get("root_dir", str(manifest_path.parent)),
        "markdown_path": manifest.get("markdown_path"),
        "variant": variant_name,
        "available_variants": list((manifest.get("variants") or {}).keys()),
        "source_summary_path": str(summary_path.resolve()),
        "source_signature": file_signature(summary_path),
        "manifest_signature": file_signature(manifest_path),
        "generated_from_summary_variant": summary.get("variant", variant_name),
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
        "models": catalog_entries,
    }


def build_snapshot_html(experiment_id: int, variant_name: str) -> str:
    html_document = build_html()
    bootstrap_script = (
        f"<script>window.__DASHBOARD_INITIAL_EXPERIMENT_ID__ = {json.dumps(str(experiment_id))};"
        f"window.__DASHBOARD_INITIAL_VARIANTS__ = {json.dumps([variant_name])};</script>\n"
    )
    return html_document.replace('<script src="/app.js"></script>', f"{bootstrap_script}<script src=\"/app.js\"></script>")


def catalog_is_fresh(catalog_path: Path, manifest_path: Path, summary_path: Path) -> bool:
    if not catalog_path.exists():
        return False
    try:
        catalog = load_json(catalog_path)
    except Exception:
        return False
    if not isinstance(catalog, dict) or catalog.get("artifact_version") != ARTIFACT_VERSION:
        return False
    if catalog.get("manifest_signature") != file_signature(manifest_path):
        return False
    if catalog.get("source_signature") != file_signature(summary_path):
        return False
    for model in catalog.get("models", []):
        if not isinstance(model, dict):
            return False
        series_file = model.get("series_file")
        if not series_file or not (catalog_path.parent / series_file).exists():
            return False
    return True


def build_variant_artifacts(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    variant_name: str,
    variant_payload: dict[str, Any],
    force: bool,
    snapshots: bool,
) -> dict[str, Any]:
    summary_path = Path(str(variant_payload.get("summary_path", "")))
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found for experiment {manifest.get('experiment_id')} variant {variant_name}: {summary_path}")

    variant_dir = summary_path.parent
    catalog_path = variant_dir / CATALOG_FILE_NAME
    if not force and catalog_is_fresh(catalog_path, manifest_path, summary_path):
        result = {"variant": variant_name, "status": "skipped", "catalog_path": str(catalog_path.resolve())}
        if snapshots:
            snapshot_path = variant_dir / SNAPSHOT_FILE_NAME
            if not snapshot_path.exists():
                snapshot_path.write_text(build_snapshot_html(int(manifest.get("experiment_id")), variant_name), encoding="utf-8")
            result["snapshot_path"] = str(snapshot_path.resolve())
        return result

    summary = load_variant_summary(variant_payload)
    if not isinstance(summary, dict):
        raise ValueError(f"Summary is not a JSON object: {summary_path}")

    series_dir = variant_dir / SERIES_DIR_NAME
    if series_dir.exists():
        shutil.rmtree(series_dir)
    series_dir.mkdir(parents=True, exist_ok=True)

    ordered_names = summary.get("series_order", [])
    models = summary.get("models", [])
    if not isinstance(models, list):
        models = []
    if not isinstance(ordered_names, list):
        ordered_names = []
    model_map = {
        str(model.get("name")): model
        for model in models
        if isinstance(model, dict) and model.get("name")
    }
    if ordered_names:
        ordered_models = [model_map[name] for name in ordered_names if name in model_map]
        ordered_models.extend([model for model in models if isinstance(model, dict) and model.get("name") not in ordered_names])
    else:
        ordered_models = [model for model in models if isinstance(model, dict) and model.get("name")]

    catalog_entries: list[dict[str, Any]] = []
    for index, model in enumerate(ordered_models, start=1):
        if not has_renderable_dashboard_series(model):
            continue
        model_artifact = build_model_artifact(model)
        series_filename = f"{index:04d}_{slugify_text(str(model_artifact['name']))}.json"
        write_json_atomic(
            series_dir / series_filename,
            {
                "artifact_version": ARTIFACT_VERSION,
                "experiment_id": int(manifest.get("experiment_id")),
                "variant": variant_name,
                "model": model_artifact,
            },
        )
        catalog_entries.append(build_catalog_entry(model_artifact, f"{SERIES_DIR_NAME}/{series_filename}"))

    catalog_payload = build_variant_catalog(
        manifest=manifest,
        manifest_path=manifest_path,
        variant_name=variant_name,
        summary=summary,
        summary_path=summary_path,
        catalog_entries=catalog_entries,
    )
    write_json_atomic(catalog_path, catalog_payload)

    result = {
        "variant": variant_name,
        "status": "built",
        "models": len(catalog_entries),
        "catalog_path": str(catalog_path.resolve()),
        "series_dir": str(series_dir.resolve()),
    }
    if snapshots:
        snapshot_path = variant_dir / SNAPSHOT_FILE_NAME
        snapshot_path.write_text(build_snapshot_html(int(manifest.get("experiment_id")), variant_name), encoding="utf-8")
        result["snapshot_path"] = str(snapshot_path.resolve())
    return result


def iter_manifest_paths(experiments_root: Path, selected_ids: set[str] | None) -> list[Path]:
    manifest_paths = sorted(experiments_root.rglob("manifest.json"))
    if not selected_ids:
        return manifest_paths
    filtered: list[Path] = []
    for manifest_path in manifest_paths:
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue
        experiment_id = str(manifest.get("experiment_id"))
        if experiment_id in selected_ids:
            filtered.append(manifest_path)
    return filtered


def build_v2_dashboard_artifacts(
    *,
    experiments_root: str | Path,
    experiments: set[str] | None = None,
    force: bool = False,
    snapshots: bool = False,
) -> dict[str, Any]:
    experiments_root = Path(experiments_root).resolve()
    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments root not found: {experiments_root}")

    selected_ids = None if not experiments else {str(value) for value in experiments}
    manifest_paths = iter_manifest_paths(experiments_root, selected_ids)
    if not manifest_paths:
        raise FileNotFoundError("No matching experiment manifests found to build.")

    results: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        variants = manifest.get("variants", {})
        if not isinstance(variants, dict):
            continue
        variant_results = []
        for variant_name, variant_payload in variants.items():
            if not isinstance(variant_payload, dict):
                continue
            variant_results.append(
                build_variant_artifacts(
                    manifest=manifest,
                    manifest_path=manifest_path,
                    variant_name=variant_name,
                    variant_payload=variant_payload,
                    force=force,
                    snapshots=snapshots,
                )
            )
        results.append(
            {
                "experiment_id": manifest.get("experiment_id"),
                "name": manifest.get("name"),
                "variant_results": variant_results,
            }
        )

    return {"experiments_root": str(experiments_root), "results": results}


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments root not found: {experiments_root}")

    selected_ids = None if args.all or not args.experiments else {str(value) for value in args.experiments}
    manifest_paths = iter_manifest_paths(experiments_root, selected_ids)
    if not manifest_paths:
        raise FileNotFoundError("No matching experiment manifests found to build.")

    result = build_v2_dashboard_artifacts(
        experiments_root=experiments_root,
        experiments=selected_ids,
        force=bool(args.force),
        snapshots=bool(args.snapshots),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
