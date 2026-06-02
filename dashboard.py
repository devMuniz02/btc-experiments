from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from automation.update_readme import (
    METRIC_COLUMNS,
    collect_prod_slots,
    collect_run_counts,
    fmt_number,
    load_config,
    top_dev_model,
)
from src.models.backtest import BacktestConfig, model_ids_from_results, run_binary_backtest
from src.utils import QuantStreamPaths, read_yaml_file

METRIC_ORDER = [
    "model_id",
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
    "backend",
]
TRADE_COLUMNS = [
    "timestamp",
    "prediction",
    "action",
    "probability",
    "target",
    "target_action",
    "signal",
    "executed",
    "win",
    "stake",
    "step_return",
    "pnl_fraction",
    "pnl",
    "equity",
    "model_id",
]
BET_MODE_LABELS = {
    "Current Percentage": "current_percentage",
    "Top Percentage": "top_percentage",
    "Fixed Amount": "fixed_amount",
}
DAY_OPTIONS = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}


def list_variations(paths: QuantStreamPaths) -> list[str]:
    data_variations = {path.name for path in paths.data_dir.glob("var_*") if path.is_dir()}
    model_variations = {path.name for path in paths.models_dev_dir.glob("var_*") if path.is_dir()}
    return sorted(data_variations | model_variations) or ["var_1"]


def load_results(paths: QuantStreamPaths, variation: str) -> pd.DataFrame:
    path = paths.global_results_path(variation.removeprefix("var_"))
    if not path.exists():
        return pd.DataFrame()
    results = pd.read_parquet(path)
    if "actual" in results.columns and "target" not in results.columns:
        results = results.rename(columns={"actual": "target"})
    if "actual" in results.columns and "target" in results.columns:
        results = results.drop(columns=["actual"])
    return results


def request_rows(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.y*ml")):
        row: dict[str, Any] = {"file": path.name}
        try:
            payload = read_yaml_file(path)
        except Exception:
            payload = {}
        row["model_id"] = payload.get("model_id", "-")
        row["model_type"] = payload.get("model_type", "-")
        row["variation_id"] = payload.get("variation_id", "-")
        row["train_mode"] = payload.get("train_mode", "-")
        row["status"] = payload.get("status", "-")
        rows.append(row)
    return pd.DataFrame(rows, columns=["file", "model_id", "model_type", "variation_id", "train_mode", "status"])


def render_run_overview(paths: QuantStreamPaths) -> None:
    counts = collect_run_counts(paths)
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total Runs", counts.total)
    metric_columns[1].metric("Pending", counts.pending)
    metric_columns[2].metric("Done", counts.done)
    metric_columns[3].metric("Rejected", counts.rejected)
    metric_columns[4].metric("Deleted", counts.deleted)
    tab_pending, tab_done, tab_rejected, tab_deleted = st.tabs(["Pending", "Done", "Rejected", "Deleted"])
    with tab_pending:
        st.dataframe(request_rows(paths.run_requests_dir), use_container_width=True, hide_index=True)
    with tab_done:
        st.dataframe(request_rows(paths.runs_done_dir), use_container_width=True, hide_index=True)
    with tab_rejected:
        st.dataframe(request_rows(paths.rejected_runs_dir), use_container_width=True, hide_index=True)
    with tab_deleted:
        st.dataframe(request_rows(paths.deleted_runs_dir), use_container_width=True, hide_index=True)


def formatted_metrics(metrics: dict[str, Any]) -> dict[str, str]:
    return {key: fmt_number(metrics.get(key)) for key in METRIC_ORDER}


def model_metric_table(results: pd.DataFrame, model_ids: list[str], threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for model_id in model_ids:
        backtest = run_binary_backtest(
            results,
            model_id,
            config=BacktestConfig(confidence_threshold=threshold, use_cuda="cpu"),
        )
        rows.append(formatted_metrics(backtest.metrics))
    return pd.DataFrame(rows, columns=METRIC_ORDER)


def render_top_model(paths: QuantStreamPaths) -> None:
    st.subheader("Top Dev Model")
    row = top_dev_model(paths)
    st.dataframe(pd.DataFrame([row], columns=METRIC_COLUMNS), use_container_width=True, hide_index=True)


def render_model_detail(paths: QuantStreamPaths, variation: str, model_id: str) -> None:
    model_dir = paths.variation_models_dir(variation.removeprefix("var_")) / model_id
    left, right = st.columns(2)
    with left:
        st.caption("hyperparameters.json")
        metadata_path = model_dir / "hyperparameters.json"
        if metadata_path.exists():
            try:
                st.json(json.loads(metadata_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                st.code(metadata_path.read_text(encoding="utf-8"))
        else:
            st.info("No metadata file for this model.")
    with right:
        st.caption("tracking_buffer.json")
        tracking_path = model_dir / "tracking_buffer.json"
        if tracking_path.exists():
            st.code(tracking_path.read_text(encoding="utf-8") or "{}")
        else:
            st.info("No tracking buffer for this model.")


def model_chart_labels(paths: QuantStreamPaths, variation: str, model_ids: list[str]) -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    variation_id = variation.removeprefix("var_")
    for model_id in model_ids:
        model_dir = paths.variation_models_dir(variation_id) / model_id
        metadata_path = model_dir / "hyperparameters.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        timestamp = pd.to_datetime(
            metadata.get("finalized_at") or metadata.get("completed_at") or pd.Timestamp.max.tz_localize("UTC"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.max.tz_localize("UTC")
        rows.append(
            {
                "model_id": model_id,
                "model_type": str(metadata.get("model_type") or "model").strip().lower(),
                "timestamp": timestamp,
            }
        )
    labels: dict[str, str] = {}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return labels
    for model_type, group in frame.sort_values(["model_type", "timestamp", "model_id"]).groupby("model_type"):
        for index, row in enumerate(group.itertuples(index=False), start=1):
            labels[str(row.model_id)] = f"{model_type} {index}"
    return labels


def render_backtest_sandbox(
    paths: QuantStreamPaths,
    variation: str,
    results: pd.DataFrame,
    model_ids: list[str],
) -> None:
    if not model_ids:
        st.info("No model prediction columns found.")
        return
    selected = st.multiselect("Models", model_ids, default=model_ids[: min(5, len(model_ids))])
    label_map = model_chart_labels(paths, variation, model_ids)
    control_columns = st.columns(4)
    with control_columns[0]:
        balance = st.number_input("Starting Balance", min_value=1.0, value=1000.0, step=100.0)
    with control_columns[1]:
        bet_mode_label = st.selectbox("Bet Mode", list(BET_MODE_LABELS), index=0)
    with control_columns[2]:
        threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with control_columns[3]:
        use_cuda = st.selectbox("Backtest Backend", ["auto", "cpu", "cuda"], index=0)
    sizing_columns = st.columns(2)
    with sizing_columns[0]:
        bet_fraction = st.slider("Bet Percentage", min_value=0.001, max_value=0.2, value=0.02, step=0.001)
    with sizing_columns[1]:
        fixed_bet_amount = st.number_input("Fixed Bet Amount", min_value=1.0, value=1.0, step=1.0)
    hours = st.multiselect("Hours UTC", list(range(24)), default=list(range(24)))
    selected_day_labels = st.multiselect("Days UTC", list(DAY_OPTIONS), default=list(DAY_OPTIONS))
    selected_days = tuple(DAY_OPTIONS[label] for label in selected_day_labels)
    if "timestamp" in results.columns and not results.empty:
        timestamps = pd.to_datetime(results["timestamp"], utc=True, errors="coerce")
        min_date = timestamps.min().date()
        max_date = timestamps.max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start = pd.Timestamp(date_range[0], tz="UTC")
            end = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
            results = results.loc[(timestamps >= start) & (timestamps < end)].reset_index(drop=True)
        range_percentage = st.slider("Selected Range Percentage", min_value=1, max_value=100, value=100, step=1)
        if range_percentage < 100 and not results.empty:
            row_count = max(1, int(np.ceil(len(results) * (range_percentage / 100.0))))
            results = results.iloc[:row_count].reset_index(drop=True)
    curves: list[pd.DataFrame] = []
    metrics: list[dict[str, str]] = []
    latest_trades = pd.DataFrame()
    for model_id in selected:
        backtest = run_binary_backtest(
            results,
            model_id,
            config=BacktestConfig(
                starting_balance=float(balance),
                bet_fraction=float(bet_fraction),
                bet_mode=BET_MODE_LABELS[str(bet_mode_label)],
                fixed_bet_amount=float(fixed_bet_amount),
                minimum_bet_amount=1.0,
                confidence_threshold=float(threshold),
                hours_utc=tuple(int(hour) for hour in hours),
                days_utc=selected_days,
                use_cuda=str(use_cuda),
            ),
        )
        metrics.append(formatted_metrics(backtest.metrics))
        trades = backtest.trades.copy()
        if not trades.empty:
            trades["model_id"] = model_id
            trades["model_label"] = label_map.get(model_id, model_id)
            curves.append(trades[["timestamp", "equity", "model_label"]])
            latest_trades = pd.concat([latest_trades, trades.tail(50)], ignore_index=True)
    st.dataframe(pd.DataFrame(metrics, columns=METRIC_ORDER), use_container_width=True, hide_index=True)
    if curves:
        curve_frame = pd.concat(curves, ignore_index=True)
        st.plotly_chart(px.line(curve_frame, x="timestamp", y="equity", color="model_label"), use_container_width=True)
    if not latest_trades.empty:
        visible_columns = [column for column in TRADE_COLUMNS if column in latest_trades.columns]
        st.dataframe(latest_trades[visible_columns], use_container_width=True, hide_index=True)


def render_research(paths: QuantStreamPaths) -> None:
    st.header("Research Sandbox")
    render_run_overview(paths)
    render_top_model(paths)
    variation = st.sidebar.selectbox("Variation", list_variations(paths))
    results = load_results(paths, variation)
    if results.empty:
        st.info("No global_results.parquet found for this variation.")
        return
    model_ids = model_ids_from_results(results)
    tab_metrics, tab_backtest, tab_detail = st.tabs(["Model Comparison", "Backtest", "Model Detail"])
    with tab_metrics:
        threshold = st.slider("Metrics Signal Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        st.dataframe(model_metric_table(results, model_ids, threshold), use_container_width=True, hide_index=True)
    with tab_backtest:
        render_backtest_sandbox(paths, variation, results, model_ids)
    with tab_detail:
        if model_ids:
            selected_model = st.selectbox("Model", model_ids)
            render_model_detail(paths, variation, selected_model)
        else:
            st.info("No model prediction columns found.")


def slot_metadata_row(paths: QuantStreamPaths, slot: str) -> dict[str, str]:
    slot_dir = paths.models_prod_dir / slot
    metadata_path = slot_dir / "hyperparameters.json"
    if not metadata_path.exists():
        return {"slot": slot, "model_id": "-", "model_type": "-", "status": "-"}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    return {
        "slot": slot,
        "model_id": str(payload.get("model_id") or "-"),
        "model_type": str(payload.get("model_type") or "-"),
        "status": str(payload.get("status") or "-"),
    }


def render_mlflow_status(paths: QuantStreamPaths) -> None:
    config = load_config(paths)
    public_url = str(config.get("mlflow_public_url") or "").strip()
    is_public = bool(config.get("mlflow_synced")) and bool(config.get("mlflow_public_access")) and bool(public_url)
    if is_public:
        st.success("MLflow is synced and the public link is open to anyone with the URL.")
        st.link_button("Open Public MLflow", public_url)
    else:
        st.info("MLflow is not publicly synced. Azure and MLflow remain disabled for local-only operation.")


def render_production(paths: QuantStreamPaths) -> None:
    st.header("Production Live Tracking")
    render_mlflow_status(paths)
    slots = [f"model_slot_{index}" for index in range(1, 6)]
    slot_rows = collect_prod_slots(paths)
    st.subheader("Champion Slots")
    st.dataframe(pd.DataFrame(slot_rows, columns=METRIC_COLUMNS), use_container_width=True, hide_index=True)
    ledger_path = paths.data_dir / "prod" / "production_trades.parquet"
    if not ledger_path.exists():
        st.info("No production ledger found.")
        st.dataframe(
            pd.DataFrame([slot_metadata_row(paths, slot) for slot in slots]),
            use_container_width=True,
            hide_index=True,
        )
        return
    ledger = pd.read_parquet(ledger_path)
    if ledger.empty:
        st.info("Production ledger is empty.")
        return
    selected_slots = st.multiselect("Slots", slots, default=slots)
    if "slot" in ledger.columns:
        filtered = ledger.loc[ledger["slot"].astype(str).isin(selected_slots)].copy()
    else:
        filtered = ledger.copy()
        filtered["slot"] = "production"
    if filtered.empty:
        st.info("No trades for the selected slots.")
        return
    if "pnl" in filtered.columns:
        filtered["equity"] = filtered.groupby("slot")["pnl"].cumsum()
        st.plotly_chart(px.line(filtered, x=filtered.index, y="equity", color="slot"), use_container_width=True)
        pnl_by_slot = filtered.groupby("slot", as_index=False)["pnl"].sum().rename(columns={"pnl": "net_pnl"})
        st.plotly_chart(
            go.Figure(data=[go.Bar(x=pnl_by_slot["slot"], y=pnl_by_slot["net_pnl"])]),
            use_container_width=True,
        )
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Quant-Stream", layout="wide")
    paths = QuantStreamPaths()
    st.sidebar.title("Quant-Stream")
    mode = st.sidebar.radio("Domain", ["Research", "Production"], index=0)
    if mode == "Research":
        render_research(paths)
    else:
        render_production(paths)


if __name__ == "__main__":
    main()
