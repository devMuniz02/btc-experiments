from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CUDA_KERNEL_PATH = Path(__file__).resolve().parents[1] / "cuda_backtest.cu"
SELL_ACTION = 0
BUY_ACTION = 1


@dataclass(frozen=True)
class BacktestConfig:
    starting_balance: float = 1000.0
    bet_fraction: float = 0.02
    bet_mode: str = "current_percentage"
    fixed_bet_amount: float = 1.0
    minimum_bet_amount: float = 1.0
    confidence_threshold: float = 0.5
    hours_utc: tuple[int, ...] = tuple(range(24))
    days_utc: tuple[int, ...] = tuple(range(7))
    use_cuda: str = "auto"


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, Any]
    trades: pd.DataFrame
    backend: str


def model_ids_from_results(results: pd.DataFrame) -> list[str]:
    return sorted(column.removesuffix("_pred") for column in results.columns if column.endswith("_pred"))


def _as_float_array(values: pd.Series | np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    array = pd.Series(values).to_numpy(dtype=np.float64, na_value=np.nan)
    return np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _as_int_array(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.Series(values).to_numpy(dtype=np.float64, na_value=np.nan)
    return np.nan_to_num(array, nan=0.0).astype(np.int8)


def _time_mask(frame: pd.DataFrame, hours_utc: tuple[int, ...]) -> np.ndarray:
    if "timestamp" not in frame.columns or not hours_utc:
        return np.ones(len(frame), dtype=bool)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return timestamps.dt.hour.isin(set(hours_utc)).to_numpy(dtype=bool)


def _day_mask(frame: pd.DataFrame, days_utc: tuple[int, ...]) -> np.ndarray:
    if "timestamp" not in frame.columns or not days_utc:
        return np.ones(len(frame), dtype=bool)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return timestamps.dt.dayofweek.isin(set(days_utc)).to_numpy(dtype=bool)


def _load_cuda_kernel() -> str:
    return CUDA_KERNEL_PATH.read_text(encoding="utf-8")


def _cuda_step_contributions(
    predictions: np.ndarray,
    target: np.ndarray,
    trade_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if predictions.size == 0:
        return None
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        kernel = cp.RawKernel(_load_cuda_kernel(), "quant_stream_backtest_steps")
        count = int(len(predictions))
        pred_gpu = cp.asarray(predictions.astype(np.int32))
        target_gpu = cp.asarray(target.astype(np.int32))
        mask_gpu = cp.asarray(trade_mask.astype(np.int32))
        step_gpu = cp.zeros(count, dtype=cp.float32)
        win_gpu = cp.zeros(count, dtype=cp.int32)
        signal_gpu = cp.zeros(count, dtype=cp.int32)
        block_size = 256
        grid_size = (count + block_size - 1) // block_size
        kernel(
            (grid_size,),
            (block_size,),
            (
                pred_gpu,
                target_gpu,
                mask_gpu,
                step_gpu,
                win_gpu,
                signal_gpu,
                np.int32(count),
            ),
        )
        return cp.asnumpy(step_gpu), cp.asnumpy(win_gpu).astype(bool), cp.asnumpy(signal_gpu).astype(np.int8)
    except Exception:
        return None


def _cpu_step_contributions(
    predictions: np.ndarray,
    target: np.ndarray,
    trade_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signed_signal = np.where(predictions == BUY_ACTION, 1.0, -1.0)
    wins = predictions == target
    step_return = np.where(wins, 1.0, -1.0)
    step_return = np.where(trade_mask, step_return, 0.0)
    wins = np.where(trade_mask, wins, False)
    signal = np.where(trade_mask, signed_signal, 0.0).astype(np.int8)
    return step_return.astype(np.float64), wins.astype(bool), signal


def _action_labels(predictions: np.ndarray) -> np.ndarray:
    return np.where(predictions == BUY_ACTION, "buy", "sell")


def _stake_for_trade(current_equity: float, peak_equity: float, config: BacktestConfig) -> float:
    mode = str(config.bet_mode).strip().lower()
    minimum = max(1.0, float(config.minimum_bet_amount))
    if mode == "fixed_amount":
        raw_stake = float(config.fixed_bet_amount)
    elif mode == "top_percentage":
        raw_stake = float(peak_equity) * float(config.bet_fraction)
    else:
        raw_stake = float(current_equity) * float(config.bet_fraction)
    stake = max(minimum, raw_stake)
    return max(0.0, min(float(current_equity), stake))


def _equity_from_step_returns(
    step_returns: np.ndarray,
    trade_mask: np.ndarray,
    *,
    config: BacktestConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    equity = np.zeros(len(step_returns), dtype=np.float64)
    pnl_amount = np.zeros(len(step_returns), dtype=np.float64)
    stake_amount = np.zeros(len(step_returns), dtype=np.float64)
    current = float(config.starting_balance)
    peak = current
    for index, step_return in enumerate(step_returns):
        stake = _stake_for_trade(current, peak, config) if bool(trade_mask[index]) else 0.0
        pnl = stake * float(step_return)
        current += pnl
        peak = max(peak, current)
        stake_amount[index] = stake
        pnl_amount[index] = pnl
        equity[index] = current
    return equity, pnl_amount, stake_amount


def run_binary_backtest(
    results: pd.DataFrame,
    model_id: str,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    resolved = config or BacktestConfig()
    pred_column = f"{model_id}_pred"
    prob_column = f"{model_id}_prob"
    if pred_column not in results.columns or prob_column not in results.columns:
        empty = pd.DataFrame()
        return BacktestResult(metrics=empty_metrics(model_id=model_id), trades=empty, backend="missing")
    predictions = _as_int_array(results[pred_column])
    probabilities = np.clip(_as_float_array(results[prob_column]), 0.0, 1.0)
    if "target" in results.columns:
        target = _as_int_array(results["target"])
    else:
        target = np.zeros(len(results), dtype=np.int8)
    valid_prediction = pd.Series(results[pred_column]).notna().to_numpy(dtype=bool)
    valid_probability = (pd.Series(results[prob_column]).notna().to_numpy(dtype=bool)) & (probabilities > 0.0)
    trade_mask = (
        valid_prediction
        & valid_probability
        & (probabilities >= float(resolved.confidence_threshold))
        & _time_mask(results, resolved.hours_utc)
        & _day_mask(results, resolved.days_utc)
    )
    cuda_steps = None
    cuda_eligible = str(resolved.bet_mode).strip().lower() == "current_percentage"
    if str(resolved.use_cuda).lower() in {"auto", "cuda", "cupy"} and cuda_eligible:
        cuda_steps = _cuda_step_contributions(
            predictions,
            target,
            trade_mask,
        )
    if cuda_steps is None:
        step_returns, wins, signal = _cpu_step_contributions(
            predictions,
            target,
            trade_mask,
        )
        backend = "numpy"
    else:
        step_returns, wins, signal = cuda_steps
        backend = "cupy_cuda"
    equity, pnl_amount, stake_amount = _equity_from_step_returns(
        step_returns,
        trade_mask,
        config=resolved,
    )
    trades = pd.DataFrame(
        {
            "timestamp": results["timestamp"] if "timestamp" in results.columns else np.arange(len(results)),
            "prediction": predictions,
            "action": _action_labels(predictions),
            "probability": probabilities,
            "target": target,
            "target_action": _action_labels(target),
            "signal": signal,
            "executed": trade_mask,
            "win": wins,
            "stake": stake_amount,
            "step_return": step_returns,
            "pnl_fraction": np.divide(
                pnl_amount,
                np.maximum(stake_amount, 1e-12),
                out=np.zeros_like(pnl_amount, dtype=np.float64),
                where=stake_amount > 0,
            ),
            "pnl": pnl_amount,
            "equity": equity,
        }
    )
    metrics = summarize_backtest(results, trades, model_id=model_id)
    metrics["backend"] = backend
    return BacktestResult(metrics=metrics, trades=trades, backend=backend)


def summarize_backtest(results: pd.DataFrame, trades: pd.DataFrame, *, model_id: str) -> dict[str, Any]:
    pred_column = f"{model_id}_pred"
    prob_column = f"{model_id}_prob"
    if pred_column in results.columns and prob_column in results.columns:
        valid_probability = pd.to_numeric(results[prob_column], errors="coerce")
        valid_rows = results[pred_column].notna() & valid_probability.notna() & (valid_probability > 0.0)
        prediction_count = int(valid_rows.sum())
        probability_series = valid_probability.loc[valid_rows]
    else:
        prediction_count = 0
        probability_series = pd.Series(dtype=float)
    signal_count = int(trades["executed"].sum()) if "executed" in trades.columns else 0
    metrics: dict[str, Any] = {
        "model_id": model_id,
        "prediction_count": prediction_count,
        "signal_count": signal_count,
        "mean_probability": float(probability_series.mean()) if not probability_series.empty else None,
        "accuracy": None,
        "accuracy_q1": None,
        "accuracy_q2": None,
        "accuracy_q3": None,
        "win_rate": None,
        "net_wins": None,
        "net_wins_per_day": None,
        "net_pnl": None,
        "max_drawdown": None,
        "latest_timestamp": None,
    }
    if "timestamp" in results.columns and len(results):
        metrics["latest_timestamp"] = str(pd.to_datetime(results["timestamp"], utc=True, errors="coerce").max())
    if signal_count and "win" in trades.columns:
        executed_trades = trades.loc[trades["executed"]].copy()
        win_values = executed_trades["win"].astype(bool)
        win_count = int(win_values.sum())
        loss_count = int((~win_values).sum())
        net_wins = win_count - loss_count
        accuracy = float(win_values.astype(float).mean())
        metrics["accuracy"] = accuracy
        metrics["win_rate"] = accuracy
        for index, fraction in enumerate((0.25, 0.50, 0.75), start=1):
            cutoff = max(1, int(np.ceil(len(executed_trades) * fraction)))
            cumulative_slice = executed_trades.iloc[:cutoff]
            metrics[f"accuracy_q{index}"] = float(cumulative_slice["win"].astype(float).mean())
        metrics["net_wins"] = net_wins
        if "timestamp" in executed_trades.columns:
            trade_dates = pd.to_datetime(executed_trades["timestamp"], utc=True, errors="coerce").dt.date.dropna()
            day_count = max(1, int(trade_dates.nunique()))
            metrics["net_wins_per_day"] = float(net_wins / day_count)
    if "pnl" in trades.columns and not trades.empty:
        metrics["net_pnl"] = float(trades["pnl"].sum())
    if "equity" in trades.columns and not trades.empty:
        equity = trades["equity"].to_numpy(dtype=np.float64)
        peak = np.maximum.accumulate(equity)
        drawdown = np.divide(equity - peak, peak, out=np.zeros_like(equity), where=peak > 0)
        metrics["max_drawdown"] = float(drawdown.min()) if len(drawdown) else None
    return metrics


def empty_metrics(model_id: str = "-") -> dict[str, Any]:
    return {
        "model_id": model_id,
        "prediction_count": 0,
        "signal_count": 0,
        "mean_probability": None,
        "accuracy": None,
        "accuracy_q1": None,
        "accuracy_q2": None,
        "accuracy_q3": None,
        "win_rate": None,
        "net_wins": None,
        "net_wins_per_day": None,
        "net_pnl": None,
        "max_drawdown": None,
        "latest_timestamp": None,
        "backend": "-",
    }
