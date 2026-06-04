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
from src.models.ensemble import build_ensemble_predictions
from src.utils import QuantStreamPaths, read_yaml_file

OPTUNA_AVAILABLE = False
DEFAULT_CONFIG_PATH = Path("automation/optimization/search_config.yaml")
load_search_config: Any = None
load_study: Any = None
completed_run_metrics: Any = None
top_candidate_rows: Any = None
top5_rows: Any = None

try:
    from src.optimization.orchestrator import (
        DEFAULT_CONFIG_PATH,
        completed_run_metrics,
        load_config as load_search_config,
        load_study,
        top_candidate_rows,
        top5_rows,
    )

    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

METRIC_ORDER = [
    "model_label",
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
    "model_label",
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


def optimization_trial_rows(study: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        precision = trial.values[0] if trial.values else None
        density = trial.values[1] if trial.values else None
        family = trial.user_attrs.get("model_family") or trial.params.get("model_family")
        if family is None:
            family = next(
                (value for key, value in trial.params.items() if str(key).startswith("model_family_")),
                "-",
            )
        rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "model_label": trial.user_attrs.get("model_label", "-"),
                "model_id": trial.user_attrs.get("model_id", "-"),
                "family": family,
                "precision_at_threshold": precision,
                "trade_density": density,
                "efficiency_score": (float(precision) * float(density)) if precision is not None and density else None,
                "threshold": trial.user_attrs.get("threshold", trial.params.get("threshold", "-")),
                "executed_count": trial.user_attrs.get("executed_count", "-"),
                "trial_source": trial.user_attrs.get("trial_source", "-"),
            }
        )
    return pd.DataFrame(rows)


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


def formatted_metrics(metrics: dict[str, Any], label_map: dict[str, str] | None = None) -> dict[str, str]:
    row = {key: fmt_number(metrics.get(key)) for key in METRIC_ORDER}
    model_id = str(metrics.get("model_id") or "")
    row["model_label"] = (label_map or {}).get(model_id, str(metrics.get("model_label") or "-"))
    return row


def model_metric_table(
    results: pd.DataFrame,
    model_ids: list[str],
    threshold: float = 0.5,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    for model_id in model_ids:
        backtest = run_binary_backtest(
            results,
            model_id,
            config=BacktestConfig(confidence_threshold=threshold, use_cuda="cpu"),
        )
        rows.append(formatted_metrics(backtest.metrics, label_map))
    return pd.DataFrame(rows, columns=METRIC_ORDER)


def render_top_model(paths: QuantStreamPaths) -> None:
    st.subheader("Top Dev Model")
    row = top_dev_model(paths)
    model_id = str(row.get("model_id") or "")
    variation = str(row.get("variation_or_slot") or "var_1")
    label = "-"
    if model_id and model_id != "-":
        label = model_chart_labels(paths, variation, [model_id]).get(model_id, "-")
    row = {"model_label": label, **row}
    st.dataframe(
        pd.DataFrame([row], columns=["model_label", *METRIC_COLUMNS]),
        use_container_width=True,
        hide_index=True,
    )


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


def load_model_metadata(paths: QuantStreamPaths, variation: str, model_ids: list[str]) -> pd.DataFrame:
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
        train_mode = str(metadata.get("train_mode") or "-").strip().lower()
        rows.append(
            {
                "model_id": model_id,
                "model_type": str(metadata.get("model_type") or "unknown").strip().lower(),
                "train_mode": train_mode,
                "training_group": training_group(train_mode),
                "finalized_at": timestamp,
            }
        )
    return pd.DataFrame(rows, columns=["model_id", "model_type", "train_mode", "training_group", "finalized_at"])


def training_group(train_mode: str) -> str:
    mode = str(train_mode or "").strip().lower()
    if mode == "static_baseline":
        return "static"
    if mode.startswith("sliding_window") or mode.startswith("windowed"):
        return "window"
    if mode == "post_base":
        return "post_base"
    if mode == "reinforcement_ppo":
        return "reinforcement"
    return mode or "unknown"


def filter_model_ids(metadata: pd.DataFrame, *, key_prefix: str) -> list[str]:
    if metadata.empty:
        return []
    model_types = sorted(str(value) for value in metadata["model_type"].dropna().unique())
    training_groups = sorted(str(value) for value in metadata["training_group"].dropna().unique())
    train_modes = sorted(str(value) for value in metadata["train_mode"].dropna().unique())
    columns = st.columns(3)
    with columns[0]:
        selected_types = st.multiselect(
            "Model Family",
            model_types,
            default=model_types,
            key=f"{key_prefix}_model_family",
        )
    with columns[1]:
        selected_groups = st.multiselect(
            "Training Type",
            training_groups,
            default=training_groups,
            key=f"{key_prefix}_training_type",
        )
    with columns[2]:
        selected_modes = st.multiselect(
            "Train Mode",
            train_modes,
            default=train_modes,
            key=f"{key_prefix}_train_mode",
        )
    filtered = metadata[
        metadata["model_type"].isin(selected_types)
        & metadata["training_group"].isin(selected_groups)
        & metadata["train_mode"].isin(selected_modes)
    ]
    return [str(model_id) for model_id in filtered.sort_values(["model_type", "finalized_at", "model_id"])["model_id"]]


def model_chart_labels(paths: QuantStreamPaths, variation: str, model_ids: list[str]) -> dict[str, str]:
    frame = load_model_metadata(paths, variation, model_ids).rename(columns={"finalized_at": "timestamp"})
    labels: dict[str, str] = {}
    if frame.empty:
        return labels
    for model_type, group in frame.sort_values(["model_type", "timestamp", "model_id"]).groupby("model_type"):
        for index, row in enumerate(group.itertuples(index=False), start=1):
            labels[str(row.model_id)] = f"{model_type} {index}"
    return labels


def model_label_formatter(label_map: dict[str, str]):
    def format_model(model_id: str) -> str:
        return label_map.get(str(model_id), str(model_id))

    return format_model


def render_backtest_sandbox(
    paths: QuantStreamPaths,
    variation: str,
    results: pd.DataFrame,
    model_ids: list[str],
) -> None:
    if not model_ids:
        st.info("No model prediction columns found.")
        return
    label_map = model_chart_labels(paths, variation, model_ids)
    selected = st.multiselect(
        "Models",
        model_ids,
        default=model_ids[: min(5, len(model_ids))],
        format_func=model_label_formatter(label_map),
    )
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
    st.divider()
    use_temporary_ensemble = st.checkbox("Temporary Voting Ensemble", value=False)
    ensemble_members: list[str] = []
    ensemble_mechanism = "hard_majority"
    ensemble_min_member_probability = 0.0
    ensemble_require_unanimous_confidence = False
    if use_temporary_ensemble:
        ensemble_columns = st.columns(4)
        with ensemble_columns[0]:
            ensemble_members = st.multiselect(
                "Ensemble Members",
                model_ids,
                default=selected[: min(3, len(selected))],
                format_func=model_label_formatter(label_map),
            )
        with ensemble_columns[1]:
            ensemble_mechanism = st.selectbox(
                "Voting",
                ["hard_majority", "soft_probability", "unanimity"],
                index=0,
            )
        with ensemble_columns[2]:
            ensemble_min_member_probability = st.slider(
                "Member Min Probability",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )
        with ensemble_columns[3]:
            ensemble_require_unanimous_confidence = st.checkbox("Require All Members Confident", value=False)
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
        metrics.append(formatted_metrics(backtest.metrics, label_map))
        trades = backtest.trades.copy()
        if not trades.empty:
            trades["model_id"] = model_id
            trades["model_label"] = label_map.get(model_id, model_id)
            curves.append(trades[["timestamp", "equity", "model_label"]])
            latest_trades = pd.concat([latest_trades, trades.tail(50)], ignore_index=True)
    if use_temporary_ensemble:
        if len(ensemble_members) < 2:
            st.warning("Temporary voting ensemble needs at least two selected trained models.")
        else:
            temporary_model_id = "__temporary_voting_ensemble__"
            temporary_results = results.copy()
            try:
                predictions, probabilities = build_ensemble_predictions(
                    temporary_results,
                    {
                        "models_pool": ensemble_members,
                        "voting_mechanism": ensemble_mechanism,
                        "min_member_probability": float(ensemble_min_member_probability),
                        "require_unanimous_confidence": bool(ensemble_require_unanimous_confidence),
                    },
                )
            except ValueError as exc:
                st.warning(str(exc))
                predictions = np.asarray([], dtype=np.int8)
                probabilities = np.asarray([], dtype=np.float32)
            if len(predictions) > 0:
                temporary_results[f"{temporary_model_id}_pred"] = predictions
                temporary_results[f"{temporary_model_id}_prob"] = probabilities
                backtest = run_binary_backtest(
                    temporary_results,
                    temporary_model_id,
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
                row = formatted_metrics(backtest.metrics)
                row["model_id"] = "temporary voting ensemble"
                row["model_label"] = "temporary voting ensemble"
                metrics.append(row)
                trades = backtest.trades.copy()
                if not trades.empty:
                    trades["model_id"] = "temporary voting ensemble"
                    trades["model_label"] = "temporary voting ensemble"
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
    metadata = load_model_metadata(paths, variation, model_ids)
    label_map = model_chart_labels(paths, variation, model_ids)
    st.subheader("Model Filters")
    filtered_model_ids = filter_model_ids(metadata, key_prefix=f"research_{variation}")
    if not filtered_model_ids:
        st.info("No models match the selected filters.")
        return
    tab_metrics, tab_backtest, tab_detail = st.tabs(["Model Comparison", "Backtest", "Model Detail"])
    with tab_metrics:
        threshold = st.slider("Metrics Signal Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        st.dataframe(
            model_metric_table(results, filtered_model_ids, threshold, label_map),
            use_container_width=True,
            hide_index=True,
        )
    with tab_backtest:
        render_backtest_sandbox(paths, variation, results, filtered_model_ids)
    with tab_detail:
        if filtered_model_ids:
            selected_model = st.selectbox(
                "Model",
                filtered_model_ids,
                format_func=model_label_formatter(label_map),
            )
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


def render_optimization(paths: QuantStreamPaths) -> None:
    st.header("Optimization Search")
    if not OPTUNA_AVAILABLE:
        st.warning("Optuna is not installed in the active Conda environment.")
        st.code("powershell -ExecutionPolicy Bypass -File .\\setup_repo.ps1", language="powershell")
        return

    config_path_text = st.sidebar.text_input("Search Config", value=str(DEFAULT_CONFIG_PATH))
    config_path = Path(config_path_text)
    try:
        config = load_search_config(config_path)
        study = load_study(config)
    except Exception as exc:
        st.error(f"Could not load optimization study: {exc}")
        return

    st.caption(str(config.storage_path))
    optuna_frame = optimization_trial_rows(study)
    metric_rows = completed_run_metrics(paths, config) if completed_run_metrics is not None else []
    frame = pd.DataFrame(metric_rows) if metric_rows else optuna_frame
    counts = optuna_frame["state"].value_counts().to_dict() if not optuna_frame.empty else {}
    columns = st.columns(5)
    columns[0].metric("Trials", len(optuna_frame))
    columns[1].metric("Complete", counts.get("COMPLETE", 0))
    columns[2].metric("Failed", counts.get("FAIL", 0))
    columns[3].metric("Running", counts.get("RUNNING", 0) + counts.get("WAITING", 0))
    columns[4].metric("Target", config.target_trials)

    st.subheader("Top 5 Production Candidates")
    if metric_rows and top_candidate_rows is not None:
        top = pd.DataFrame(top_candidate_rows(metric_rows, config.top_k))
    else:
        top = pd.DataFrame(top5_rows(study, config.top_k))
    if not top.empty and "model_id" in top.columns:
        model_ids = [str(model_id) for model_id in top["model_id"] if str(model_id) not in {"", "-"}]
        label_map = model_chart_labels(paths, f"var_{config.variation_id}", model_ids)
        top.insert(1, "model_label", [label_map.get(str(model_id), "-") for model_id in top["model_id"]])
    st.dataframe(top, use_container_width=True, hide_index=True)

    if frame.empty:
        st.info("No optimization trials found yet.")
        return

    family_options = sorted(str(value) for value in frame["family"].dropna().unique())
    state_options = sorted(str(value) for value in frame["state"].dropna().unique())
    filter_columns = st.columns(2)
    with filter_columns[0]:
        selected_families = st.multiselect("Families", family_options, default=family_options)
    with filter_columns[1]:
        selected_states = st.multiselect("Trial States", state_options, default=state_options)
    filtered = frame.loc[frame["family"].isin(selected_families) & frame["state"].isin(selected_states)].copy()
    if "trial" in filtered.columns:
        filtered["trial_sort"] = pd.to_numeric(filtered["trial"], errors="coerce")
    else:
        filtered["trial_sort"] = np.nan

    complete = filtered.loc[filtered["state"] == "COMPLETE"].dropna(subset=["precision_at_threshold", "trade_density"])
    if complete.empty:
        st.info("No completed optimization trials match the selected filters.")
    else:
        st.subheader("Pareto Trial Scatter")
        st.plotly_chart(
            px.scatter(
                complete,
                x="trade_density",
                y="precision_at_threshold",
                color="family",
                hover_data=[
                    "trial",
                    "model_id",
                    "threshold",
                    "original_threshold",
                    "threshold_policy",
                    "effective_threshold",
                    "threshold_fallback_used",
                    "efficiency_score",
                    "executed_count",
                ],
            ),
            use_container_width=True,
        )
        st.subheader("Efficiency By Trial")
        st.plotly_chart(
            px.line(
                complete.sort_values(["trial_sort", "family", "model_id"], na_position="last"),
                x="trial_sort",
                y="efficiency_score",
                color="family",
                markers=True,
                hover_data=[
                    "trial",
                    "model_id",
                    "threshold",
                    "original_threshold",
                    "threshold_policy",
                    "effective_threshold",
                    "threshold_fallback_used",
                    "executed_count",
                ],
            ),
            use_container_width=True,
        )

    if metric_rows:
        st.caption("Metrics are recalculated from completed run YAMLs and global_results.parquet.")
    st.subheader("Trial Metrics")
    display_frame = filtered.sort_values(["trial_sort", "family", "model_id"], na_position="last").drop(
        columns=["trial_sort"],
        errors="ignore",
    )
    st.dataframe(display_frame, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Quant-Stream", layout="wide")
    paths = QuantStreamPaths()
    st.sidebar.title("Quant-Stream")
    mode = st.sidebar.radio("Domain", ["Research", "Optimization", "Production"], index=0)
    if mode == "Research":
        render_research(paths)
    elif mode == "Optimization":
        render_optimization(paths)
    else:
        render_production(paths)


if __name__ == "__main__":
    main()
