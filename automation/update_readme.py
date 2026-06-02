from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backtest import BacktestConfig, run_binary_backtest  # noqa: E402
from src.utils import QuantStreamPaths, read_yaml_file, utc_now_iso  # noqa: E402

PLACEHOLDER = "-"
METRIC_COLUMNS = [
    "model_id",
    "model_type",
    "variation_or_slot",
    "status",
    "latest_timestamp",
    "prediction_count",
    "signal_count",
    "mean_probability",
    "accuracy",
    "accuracy_q1",
    "accuracy_q2",
    "accuracy_q3",
    "win_rate",
    "net_wins",
    "net_wins_per_day",
    "net_pnl",
    "max_drawdown",
    "source",
]
BADGES = [
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)]"
    "(https://www.linkedin.com/in/devmuniz)",
    "[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)]"
    "(https://github.com/devMuniz02)",
    "[![Portfolio]"
    "(https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)]"
    "(https://devmuniz02.github.io/)",
    "[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)]"
    "(https://huggingface.co/manu02)",
    "![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)",
    "![MLOps](https://img.shields.io/badge/MLOps-Quant--Stream-2563EB)",
    "![Quant](https://img.shields.io/badge/Quant-BTC%201h-111827)",
    "![CI/CD](https://img.shields.io/badge/CI%2FCD-Local%20%2B%20GitHub-16A34A)",
    "![MLflow](https://img.shields.io/badge/MLflow-Sync%20Ready-0194E2)",
    "![Azure](https://img.shields.io/badge/Azure-Disabled%20Until%20Sync-0078D4?logo=microsoftazure&logoColor=white)",
    "![CUDA](https://img.shields.io/badge/CUDA-Backtest%20Kernel-76B900?logo=nvidia&logoColor=white)",
]


@dataclass(frozen=True)
class RunCounts:
    pending: int
    done: int
    rejected: int
    deleted: int

    @property
    def total(self) -> int:
        return self.pending + self.done + self.rejected + self.deleted


def yaml_count(directory: Path) -> int:
    return len([path for path in directory.glob("*.y*ml") if path.is_file()])


def collect_run_counts(paths: QuantStreamPaths) -> RunCounts:
    return RunCounts(
        pending=yaml_count(paths.run_requests_dir),
        done=yaml_count(paths.runs_done_dir),
        rejected=yaml_count(paths.rejected_runs_dir),
        deleted=yaml_count(paths.deleted_runs_dir),
    )


def empty_metrics(**overrides: str) -> dict[str, str]:
    row = {column: PLACEHOLDER for column in METRIC_COLUMNS}
    row.update(overrides)
    return row


def fmt_number(value: Any) -> str:
    if value is None:
        return PLACEHOLDER
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else PLACEHOLDER
    if pd.isna(number):
        return PLACEHOLDER
    return f"{number:.6g}"


def read_tracking_buffer(model_dir: Path) -> dict[str, Any]:
    buffer_path = model_dir / "tracking_buffer.json"
    if not buffer_path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in buffer_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            latest = payload
    return latest


def read_model_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "hyperparameters.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def dev_metrics_from_results(paths: QuantStreamPaths, variation: str, model_id: str) -> dict[str, str]:
    results_path = paths.global_results_path(variation.removeprefix("var_"))
    if not results_path.exists():
        return {}
    pred_column = f"{model_id}_pred"
    prob_column = f"{model_id}_prob"
    results = pd.read_parquet(results_path)
    metrics: dict[str, str] = {}
    if pred_column in results.columns and prob_column in results.columns:
        backtest = run_binary_backtest(
            results,
            model_id,
            config=BacktestConfig(confidence_threshold=0.0, bet_fraction=0.02, use_cuda="cpu"),
        )
        for key in (
            "prediction_count",
            "signal_count",
            "mean_probability",
            "accuracy",
            "accuracy_q1",
            "accuracy_q2",
            "accuracy_q3",
            "win_rate",
            "net_wins",
            "net_wins_per_day",
            "net_pnl",
            "max_drawdown",
            "latest_timestamp",
        ):
            if key in backtest.metrics:
                metrics[key] = fmt_number(backtest.metrics[key])
    if metrics:
        metrics["source"] = "dev_global_results"
    return metrics


def collect_dev_models(paths: QuantStreamPaths) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metadata_path in paths.models_dev_dir.glob("var_*/*/hyperparameters.json"):
        model_dir = metadata_path.parent
        variation = model_dir.parent.name
        model_id = model_dir.name
        metadata = read_model_metadata(model_dir)
        tracking = read_tracking_buffer(model_dir)
        row = empty_metrics(
            model_id=model_id,
            model_type=str(metadata.get("model_type") or PLACEHOLDER),
            variation_or_slot=variation,
            status=str(metadata.get("status") or "completed"),
            latest_timestamp=str(metadata.get("finalized_at") or tracking.get("timestamp") or PLACEHOLDER),
            source="tracking_buffer" if tracking else PLACEHOLDER,
        )
        if tracking:
            row["prediction_count"] = fmt_number(tracking.get("prediction_count"))
            row["mean_probability"] = fmt_number(tracking.get("mean_probability"))
        row.update(dev_metrics_from_results(paths, variation, model_id))
        rows.append(row)
    return rows


def top_dev_model(paths: QuantStreamPaths) -> dict[str, str]:
    rows = collect_dev_models(paths)
    if not rows:
        return empty_metrics(variation_or_slot="dev")

    def sort_key(row: dict[str, str]) -> tuple[float, str]:
        try:
            probability = float(row.get("mean_probability", PLACEHOLDER))
        except ValueError:
            probability = -1.0
        return probability, row.get("latest_timestamp", "")

    return sorted(rows, key=sort_key, reverse=True)[0]


def prod_metrics_from_ledger(paths: QuantStreamPaths, slot: str) -> dict[str, str]:
    ledger_path = paths.data_dir / "prod" / "production_trades.parquet"
    if not ledger_path.exists():
        return {}
    ledger = pd.read_parquet(ledger_path)
    if "slot" in ledger.columns:
        ledger = ledger.loc[ledger["slot"].astype(str) == slot]
    if ledger.empty:
        return {}
    metrics: dict[str, str] = {
        "source": "prod_ledger",
        "prediction_count": str(len(ledger)),
    }
    if "timestamp" in ledger.columns:
        metrics["latest_timestamp"] = str(pd.to_datetime(ledger["timestamp"], utc=True).max())
    if "prediction" in ledger.columns:
        metrics["signal_count"] = str(int(ledger["prediction"].notna().sum()))
    if "probability" in ledger.columns:
        metrics["mean_probability"] = fmt_number(ledger["probability"].mean())
    if "pnl" in ledger.columns:
        metrics["net_pnl"] = fmt_number(ledger["pnl"].sum())
        equity = ledger["pnl"].cumsum().astype(float)
        peak = equity.cummax()
        drawdown = (equity - peak) / peak.where(peak != 0)
        metrics["max_drawdown"] = fmt_number(drawdown.min())
    result_column = next((column for column in ("win", "is_win", "profitable") if column in ledger.columns), "")
    if result_column:
        wins = ledger[result_column].astype(bool)
        net_wins = int(wins.sum()) - int((~wins).sum())
        metrics["win_rate"] = fmt_number(wins.astype(float).mean())
        metrics["net_wins"] = str(net_wins)
        if "timestamp" in ledger.columns:
            trade_dates = pd.to_datetime(ledger["timestamp"], utc=True, errors="coerce").dt.date.dropna()
            metrics["net_wins_per_day"] = fmt_number(net_wins / max(1, int(trade_dates.nunique())))
    return metrics


def collect_prod_slot(paths: QuantStreamPaths, slot: str) -> dict[str, str]:
    slot_dir = paths.models_prod_dir / slot
    metadata = read_model_metadata(slot_dir)
    tracking = read_tracking_buffer(slot_dir)
    row = empty_metrics(
        model_id=str(metadata.get("model_id") or PLACEHOLDER),
        model_type=str(metadata.get("model_type") or PLACEHOLDER),
        variation_or_slot=slot,
        status=str(metadata.get("status") or PLACEHOLDER),
        latest_timestamp=str(metadata.get("finalized_at") or tracking.get("timestamp") or PLACEHOLDER),
        source="tracking_buffer" if tracking else PLACEHOLDER,
    )
    if tracking:
        row["prediction_count"] = fmt_number(tracking.get("prediction_count"))
        row["mean_probability"] = fmt_number(tracking.get("mean_probability"))
    row.update(prod_metrics_from_ledger(paths, slot))
    return row


def collect_prod_slots(paths: QuantStreamPaths) -> list[dict[str, str]]:
    return [collect_prod_slot(paths, f"model_slot_{index}") for index in range(1, 6)]


def load_config(paths: QuantStreamPaths) -> dict[str, Any]:
    config_path = paths.automation_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        return read_yaml_file(config_path)
    except Exception:
        return {}


def render_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(column, PLACEHOLDER)) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def render_readme(paths: QuantStreamPaths) -> str:
    counts = collect_run_counts(paths)
    dev_row = top_dev_model(paths)
    prod_rows = collect_prod_slots(paths)
    config = load_config(paths)
    mlflow_synced = bool(config.get("mlflow_synced", False))
    mlflow_public_access = bool(config.get("mlflow_public_access", False))
    mlflow_public_url = str(config.get("mlflow_public_url") or "").strip()
    show_mlflow_link = mlflow_synced and mlflow_public_access and bool(mlflow_public_url)
    mlflow_status = "Public synced" if show_mlflow_link else "Not synced"
    mlflow_url = f"[Open public MLflow]({mlflow_public_url})" if show_mlflow_link else PLACEHOLDER
    mlflow_note = (
        "This MLflow link is public/open for anyone with the URL."
        if show_mlflow_link
        else "MLflow is hidden until sync marks the URL as public."
    )
    run_rows = [
        {
            "total": str(counts.total),
            "pending": str(counts.pending),
            "done": str(counts.done),
            "rejected": str(counts.rejected),
            "deleted": str(counts.deleted),
        }
    ]
    mlflow_rows = [{"status": mlflow_status, "public_url": mlflow_url, "note": mlflow_note}]
    return (
        "\n\n".join(
            [
                "# Quant-Stream",
                "\n".join(BADGES),
                f"Last updated: `{utc_now_iso()}`",
                "Quant-Stream is a local, file-state BTC 1h research pipeline. YAML requests move through automation "
                "folders, models write local artifacts, and prediction columns accumulate in variation-level parquet "
                "result stores.",
                "## Workflow",
                "- Generate `var_1` with "
                "`powershell -ExecutionPolicy Bypass -File .\\automation\\generate_var_1_dataset.ps1`.\n"
                "- Add run YAML files to `automation/run_requests/`.\n"
                "- Add delete YAML files to `automation/delete_requests/`.\n"
                "- Run `python automation_runner.py --once` or use the local Windows watcher.\n"
                "- Run `python state_sync.py --check` to verify model folders, result columns, scalers, and sync "
                "buffers.\n\n"
                "Supported active model families: `lstm`, `transformer`, `mamba`, `nn`, `rf`, `xgboost`, `bc`, "
                "`dagger`, `ppo`, `ppo_continue`, `actor_critic`, `mamba_post_base`, and `ensemble`.\n\n"
                "Supported training modes: `static_baseline`, `sliding_window_current_only`, "
                "`sliding_window_continue`, `sliding_window_retrain`, `reinforcement_ppo`, and `post_base`.",
                "## Backtest Contract",
                "`prediction` uses `0 = sell` and `1 = buy`. A trade is correct when `prediction == target`. "
                "Executed correct trades add `+stake`; executed incorrect trades subtract `stake`. BTC price movement "
                "does not change backtest reward.",
                "## Run Counts",
                render_table(run_rows, ["total", "pending", "done", "rejected", "deleted"]),
                "## Top Dev Model",
                render_table([dev_row], METRIC_COLUMNS),
                "## Production Slots",
                render_table(prod_rows, METRIC_COLUMNS),
                "## MLflow Sync",
                render_table(mlflow_rows, ["status", "public_url", "note"]),
                "## Local Automation",
                "- `automation/*.ps1` scripts are local-only and ignored by Git.\n"
                "- GitHub Actions run code quality only.\n"
                "- Local request execution stays on this machine until MLflow sync is explicitly enabled.",
            ]
        )
        + "\n"
    )


def update_readme(root: Path | None = None) -> Path:
    paths = QuantStreamPaths(root=root or QuantStreamPaths().root)
    output_path = paths.root / "README.md"
    output_path.write_text(render_readme(paths), encoding="utf-8")
    return output_path


def main() -> int:
    output_path = update_readme()
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
