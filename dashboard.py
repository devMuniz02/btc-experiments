from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import QuantStreamPaths


def list_variations(paths: QuantStreamPaths) -> list[str]:
    data_variations = {path.name for path in paths.data_dir.glob("var_*") if path.is_dir()}
    model_variations = {path.name for path in paths.models_dev_dir.glob("var_*") if path.is_dir()}
    return sorted(data_variations | model_variations) or ["var_1"]


def load_results(paths: QuantStreamPaths, variation: str) -> pd.DataFrame:
    path = paths.global_results_path(variation.removeprefix("var_"))
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def model_ids_from_results(results: pd.DataFrame) -> list[str]:
    return sorted(column.removesuffix("_pred") for column in results.columns if column.endswith("_pred"))


def temporal_mask(results: pd.DataFrame, hours: list[int]) -> np.ndarray:
    if "timestamp" not in results.columns:
        return np.ones(len(results), dtype=bool)
    timestamps = pd.to_datetime(results["timestamp"], utc=True)
    return timestamps.dt.hour.isin(hours).to_numpy(dtype=bool)


def simulate_equity(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    balance: float,
    bet_fraction: float,
) -> np.ndarray:
    equity = [float(balance)]
    current = float(balance)
    for pred, probability in zip(predictions, probabilities, strict=False):
        signed_return = float(pred) * float(probability) * float(bet_fraction)
        current *= 1.0 + signed_return
        equity.append(current)
    return np.asarray(equity[1:], dtype=float)


def render_research(paths: QuantStreamPaths) -> None:
    variation = st.sidebar.selectbox("Variation", list_variations(paths))
    results = load_results(paths, variation)
    if results.empty:
        st.info("No global_results.parquet found for this variation.")
        return
    model_ids = model_ids_from_results(results)
    selected = st.multiselect("Models", model_ids, default=model_ids[: min(5, len(model_ids))])
    tab_grid, tab_sandbox = st.tabs(["Configuration & Metrics", "Temporal Filter Sandbox"])
    with tab_grid:
        rows = []
        for model_id in model_ids:
            pred = results[f"{model_id}_pred"].to_numpy()
            prob = results[f"{model_id}_prob"].to_numpy()
            rows.append(
                {
                    "model_id": model_id,
                    "signals": int(np.count_nonzero(pred)),
                    "mean_probability": float(np.mean(prob)) if len(prob) else 0.0,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with tab_sandbox:
        balance = st.slider("Starting Balance", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
        bet_fraction = st.slider("Bet Size", min_value=0.001, max_value=0.2, value=0.02, step=0.001)
        threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        hours = st.multiselect("Hours UTC", list(range(24)), default=list(range(24)))
        mask = temporal_mask(results, hours)
        curves = []
        for model_id in selected:
            pred = results[f"{model_id}_pred"].to_numpy(dtype=np.int8).copy()
            prob = results[f"{model_id}_prob"].to_numpy(dtype=np.float32).copy()
            pred[~mask] = 0
            prob[~mask] = 0.0
            pred[prob < threshold] = 0
            equity = simulate_equity(pred, prob, balance=balance, bet_fraction=bet_fraction)
            curves.extend({"index": index, "equity": value, "model_id": model_id} for index, value in enumerate(equity))
        if curves:
            figure = px.line(pd.DataFrame(curves), x="index", y="equity", color="model_id")
            st.plotly_chart(figure, use_container_width=True)


def render_production(paths: QuantStreamPaths) -> None:
    ledger_path = paths.data_dir / "prod" / "production_trades.parquet"
    if not ledger_path.exists():
        st.info("No production ledger found.")
        return
    ledger = pd.read_parquet(ledger_path)
    slots = sorted(str(value) for value in ledger.get("slot", pd.Series(dtype=str)).dropna().unique())
    selected = [slot for slot in slots if st.sidebar.checkbox(slot, value=True)]
    filtered = ledger.loc[ledger["slot"].astype(str).isin(selected)] if selected else ledger.iloc[0:0]
    split_lines = st.toggle("Split component tracker", value=False)
    if filtered.empty:
        st.info("No selected production trades.")
        return
    group_columns = ["slot"] if split_lines and "slot" in filtered.columns else []
    if "pnl" not in filtered.columns:
        st.dataframe(filtered, use_container_width=True)
        return
    if group_columns:
        filtered = filtered.copy()
        filtered["equity"] = filtered.groupby("slot")["pnl"].cumsum()
        st.plotly_chart(px.line(filtered, x=filtered.index, y="equity", color="slot"), use_container_width=True)
    else:
        equity = filtered["pnl"].cumsum()
        figure = px.line(x=np.arange(len(equity)), y=equity, labels={"x": "trade", "y": "equity"})
        st.plotly_chart(figure, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Quant-Stream", layout="wide")
    paths = QuantStreamPaths()
    mode = st.sidebar.selectbox("Domain", ["Research Sandbox Mode", "Production Live Tracking Mode"])
    if mode.startswith("Research"):
        render_research(paths)
    else:
        render_production(paths)


if __name__ == "__main__":
    main()
