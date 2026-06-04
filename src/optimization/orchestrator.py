from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Optuna is required. Run setup_repo.ps1 to install the updated Conda dependencies.") from exc

from automation_runner import validate_request
from src.utils import QuantStreamPaths, compute_model_id, read_yaml_file, write_yaml_file

DEFAULT_CONFIG_PATH = Path("automation/optimization/search_config.yaml")
DEFAULT_STUDY_NAME = "var1_static_family_search"
DEFAULT_FAMILIES = ["lstm", "mamba", "transformer", "nn", "xgboost", "rf"]
FAMILY_PROBE_PHASE = "family_probe_v1"
MID_PHASE = "mid_phase_interval_v1"
FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "seq": (6, 24, 2),
        "train": (25000, 50000, 5000),
        "lr_range": (1e-3, 3e-1),
        "max_depth": (3, 9, 2),
        "n_estimators": [50, 100, 200, 400, 600],
        "alpha_choices": [0.0, 1.0, 10.0],
    },
    "rf": {
        "seq": (6, 24, 2),
        "train": (25000, 50000, 5000),
        "max_depth": (3, 15, 3),
        "n_estimators": [50, 100, 200],
    },
    "nn": {
        "seq": (24, 72, 6),
        "train": (50000, 100000, 10000),
        "lr_range": (1e-4, 1e-2),
        "hidden_dim": [64, 128, 256],
        "dropout": [0.0, 0.1, 0.3],
    },
    "lstm": {
        "seq": (24, 72, 12),
        "train": (50000, 100000, 10000),
        "lr_range": (1e-4, 1e-2),
        "hidden_dim": [32, 64, 128],
        "dropout": [0.0, 0.1, 0.3],
    },
    "transformer": {
        "seq": (48, 168, 24),
        "train": (75000, 100000, 12500),
        "lr_range": (1e-4, 1e-3),
        "hidden_dim": [64, 128, 256],
        "num_heads": [2, 4, 8],
        "num_layers": (2, 6, 2),
    },
    "mamba": {
        "seq": (48, 168, 24),
        "train": (75000, 100000, 12500),
        "lr_range": (1e-4, 5e-3),
        "hidden_dim": [64, 128, 256],
        "num_layers": [2, 3, 4],
    },
}


@dataclass(frozen=True)
class SearchConfig:
    study_name: str
    variation_id: int
    target_trials: int
    storage_path: Path
    poll_seconds: float
    warm_start_manual_runs: bool
    families: list[str]
    threshold_range: tuple[float, float]
    top_k: int
    ranking_metric: str
    auto_promote_to_prod: bool
    search_phase: str
    probe_trials_per_family: int
    top_family_count: int
    family_probe_completed: bool


def default_config() -> dict[str, Any]:
    return {
        "study_name": DEFAULT_STUDY_NAME,
        "variation_id": 1,
        "target_trials": 200,
        "storage_path": f"automation/optimization/{DEFAULT_STUDY_NAME}.db",
        "poll_seconds": 5,
        "warm_start_manual_runs": True,
        "families": DEFAULT_FAMILIES,
        "threshold_range": [0.51, 0.85],
        "top_k": 5,
        "ranking_metric": "efficiency_score",
        "auto_promote_to_prod": False,
        "search_phase": FAMILY_PROBE_PHASE,
        "probe_trials_per_family": 3,
        "top_family_count": 3,
        "family_probe_completed": False,
    }


def ensure_config(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        return {**default_config(), **loaded}
    path.parent.mkdir(parents=True, exist_ok=True)
    config = default_config()
    write_yaml_file(path, config)
    return config


def load_config(path: Path) -> SearchConfig:
    raw = ensure_config(path)
    threshold_range = raw.get("threshold_range", [0.51, 0.85])
    return SearchConfig(
        study_name=str(raw.get("study_name") or DEFAULT_STUDY_NAME),
        variation_id=int(raw.get("variation_id", 1)),
        target_trials=int(raw.get("target_trials", 20)),
        storage_path=Path(str(raw.get("storage_path") or f"automation/optimization/{DEFAULT_STUDY_NAME}.db")),
        poll_seconds=float(raw.get("poll_seconds", 5)),
        warm_start_manual_runs=bool(raw.get("warm_start_manual_runs", True)),
        families=[str(family).lower() for family in raw.get("families", DEFAULT_FAMILIES)],
        threshold_range=(float(threshold_range[0]), float(threshold_range[1])),
        top_k=int(raw.get("top_k", 5)),
        ranking_metric=str(raw.get("ranking_metric", "efficiency_score")),
        auto_promote_to_prod=bool(raw.get("auto_promote_to_prod", False)),
        search_phase=str(raw.get("search_phase", FAMILY_PROBE_PHASE)),
        probe_trials_per_family=int(raw.get("probe_trials_per_family", 3)),
        top_family_count=int(raw.get("top_family_count", 3)),
        family_probe_completed=bool(raw.get("family_probe_completed", False)),
    )


def load_study(config: SearchConfig) -> optuna.Study:
    config.storage_path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=config.study_name,
        storage=f"sqlite:///{config.storage_path.as_posix()}",
        directions=["maximize", "maximize"],
        load_if_exists=True,
    )


def precision_density(
    results: pd.DataFrame,
    model_id: str,
    *,
    threshold: float,
    evaluated_rows: int | None = None,
) -> tuple[float, float, int, float, bool]:
    pred_col = f"{model_id}_pred"
    prob_col = f"{model_id}_prob"
    if pred_col not in results.columns or prob_col not in results.columns or "target" not in results.columns:
        raise ValueError(f"Missing result columns for model_id {model_id}.")
    frame = results.iloc[:evaluated_rows].copy() if evaluated_rows else results
    probabilities = pd.to_numeric(frame[prob_col], errors="coerce").fillna(0.0)
    predictions = pd.to_numeric(frame[pred_col], errors="coerce").fillna(0).astype(int)
    target = pd.to_numeric(frame["target"], errors="coerce").fillna(0).astype(int)
    executed = probabilities >= float(threshold)
    executed_count = int(executed.sum())
    effective_threshold = float(threshold)
    used_threshold_fallback = False
    if executed_count == 0 and len(probabilities) > 0:
        positive_probabilities = probabilities[probabilities > 0.0]
        if not positive_probabilities.empty:
            fallback_threshold = max(0.5, float(positive_probabilities.quantile(0.75)))
            executed = probabilities >= fallback_threshold
            executed_count = int(executed.sum())
            effective_threshold = fallback_threshold
            used_threshold_fallback = True
    correct = int(((predictions == target) & executed).sum())
    precision = correct / max(executed_count, 1)
    density = math.log(executed_count + 1)
    return float(precision), float(density), executed_count, effective_threshold, used_threshold_fallback


def filled_result_length(results: pd.DataFrame, model_id: str, max_rows: int = 5000) -> int | None:
    pred_col = f"{model_id}_pred"
    prob_col = f"{model_id}_prob"
    if pred_col not in results.columns or prob_col not in results.columns:
        return None
    frame = results.iloc[:max_rows]
    probabilities = pd.to_numeric(frame[prob_col], errors="coerce")
    valid = frame[pred_col].notna() & probabilities.notna() & (probabilities > 0.0)
    if not bool(valid.any()):
        return None
    invalid_positions = valid.to_numpy().nonzero()[0]
    if len(invalid_positions) == 0:
        return None
    contiguous = 0
    for index, is_valid in enumerate(valid.to_numpy()):
        if not is_valid:
            break
        contiguous = index + 1
    return contiguous or None


def result_frame(paths: QuantStreamPaths, variation_id: int) -> pd.DataFrame:
    path = paths.global_results_path(variation_id)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def trial_user_attrs(search_id: str, trial_number: int) -> dict[str, Any]:
    return {"search_id": search_id, "trial_number": int(trial_number), "trial_source": "optuna"}


def matching_payloads(
    paths: QuantStreamPaths,
    search_id: str,
    trial_number: int,
) -> list[tuple[str, Path, dict[str, Any]]]:
    matches: list[tuple[str, Path, dict[str, Any]]] = []
    folders = [
        ("pending", paths.run_requests_dir),
        ("done", paths.runs_done_dir),
        ("rejected", paths.rejected_runs_dir),
    ]
    for status, folder in folders:
        for path in sorted(folder.glob("*.y*ml")):
            try:
                payload = read_yaml_file(path)
            except Exception:
                continue
            if str(payload.get("search_id", "")) == search_id and int(payload.get("trial_number", -1)) == trial_number:
                matches.append((status, path, payload))
    return matches


def is_optuna_search_trial(trial: optuna.trial.FrozenTrial, config: SearchConfig) -> bool:
    return (
        str(trial.user_attrs.get("search_id", "")) == config.study_name
        or str(trial.user_attrs.get("trial_source", "")) == "optuna"
    )


def family_parameter_name(config: SearchConfig) -> str:
    family_key = f"{config.search_phase}|{','.join(config.families)}"
    digest = hashlib.sha256(family_key.encode("utf-8")).hexdigest()[:8]
    return f"model_family_{digest}"


def parameter_name(config: SearchConfig, family: str, name: str) -> str:
    key = f"{config.search_phase}|{family}|{name}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{name}_{digest}"


def mid_phase_learning_rate_choices(family: str) -> list[float] | None:
    choices = {
        "xgboost": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        "nn": [0.0001, 0.0003, 0.001, 0.003, 0.01],
        "lstm": [0.0001, 0.0003, 0.001, 0.003, 0.01],
        "transformer": [0.0001, 0.0003, 0.001],
        "mamba": [0.0001, 0.0003, 0.001, 0.003, 0.005],
    }
    return choices.get(family)


def suggest_learning_rate(trial: optuna.Trial, config: SearchConfig, family: str) -> float:
    choices = mid_phase_learning_rate_choices(family)
    if choices:
        return float(trial.suggest_categorical(parameter_name(config, family, "learning_rate_bucket_v1"), choices))
    spec = FAMILY_SPECS[family]
    lr_min, lr_max = spec["lr_range"]
    return float(
        trial.suggest_float(
            parameter_name(config, family, "learning_rate"),
            float(lr_min),
            float(lr_max),
            log=True,
        )
    )


def suggest_interval_trial(
    trial: optuna.Trial,
    config: SearchConfig,
    family: str,
) -> tuple[dict[str, Any], float]:
    if family not in FAMILY_SPECS:
        raise ValueError(f"Unsupported search family: {family}")
    spec = FAMILY_SPECS[family]
    seq_min, seq_max, seq_step = spec["seq"]
    train_min, train_max, train_step = spec["train"]
    threshold = trial.suggest_float(
        parameter_name(config, "shared", "threshold"),
        config.threshold_range[0],
        config.threshold_range[1],
        step=0.02,
    )
    train_length = trial.suggest_int(
        parameter_name(config, family, "train_length"),
        int(train_min),
        int(train_max),
        step=int(train_step),
    )
    sequence_length = trial.suggest_int(
        parameter_name(config, family, "sequence_length"),
        int(seq_min),
        int(seq_max),
        step=int(seq_step),
    )
    test_length = 5000
    base: dict[str, Any] = {
        "seed": 42 + int(trial.number),
        "train_length": int(train_length),
        "test_length": int(test_length),
        "sequence_length": int(sequence_length),
        "action_dim": 2,
        "eval_batch_size": 1024,
    }
    if family in {"lstm", "mamba", "transformer", "nn"}:
        num_layers_spec = spec.get("num_layers", [1, 2, 3])
        if isinstance(num_layers_spec, tuple):
            layer_min, layer_max, layer_step = num_layers_spec
            num_layers = trial.suggest_int(
                parameter_name(config, family, "num_layers"),
                int(layer_min),
                int(layer_max),
                step=int(layer_step),
            )
        else:
            num_layers = trial.suggest_categorical(parameter_name(config, family, "num_layers"), num_layers_spec)
        base.update(
            {
                "hidden_dim": trial.suggest_categorical(
                    parameter_name(config, family, "hidden_dim"),
                    spec["hidden_dim"],
                ),
                "num_layers": int(num_layers),
                "epochs": trial.suggest_categorical(parameter_name(config, family, "epochs"), [20, 40, 80]),
                "batch_size": trial.suggest_categorical(parameter_name(config, family, "batch_size"), [64, 128, 256]),
                "learning_rate": suggest_learning_rate(trial, config, family),
                "dropout": trial.suggest_categorical(
                    parameter_name(config, family, "dropout"),
                    spec.get("dropout", [0.0, 0.1, 0.3]),
                ),
                "weight_decay": trial.suggest_categorical(
                    parameter_name(config, family, "weight_decay"),
                    [0.0, 1e-6, 1e-4, 1e-3],
                ),
                "gradient_clip_norm": 0.0,
                "early_stopping_patience": 0,
                "validation_fraction": 0.0,
                "device": "auto",
                "preload_to_device": trial.suggest_categorical(
                    parameter_name(config, family, "preload_to_device"),
                    [True, False],
                ),
            }
        )
        if family == "mamba":
            base["preload_to_device"] = True
        if family == "transformer":
            base["num_heads"] = trial.suggest_categorical(
                parameter_name(config, family, "num_heads"),
                spec["num_heads"],
            )
            ff_multiplier = trial.suggest_categorical(parameter_name(config, family, "ff_multiplier"), [2, 4])
            base["feedforward_dim"] = int(base["hidden_dim"]) * int(ff_multiplier)
    elif family == "xgboost":
        depth_min, depth_max, depth_step = spec["max_depth"]
        base.update(
            {
                "n_estimators": trial.suggest_categorical(
                    parameter_name(config, family, "n_estimators"),
                    spec["n_estimators"],
                ),
                "max_depth": trial.suggest_int(
                    parameter_name(config, family, "max_depth"),
                    int(depth_min),
                    int(depth_max),
                    step=int(depth_step),
                ),
                "learning_rate": suggest_learning_rate(trial, config, family),
                "subsample": trial.suggest_categorical(parameter_name(config, family, "subsample"), [0.7, 0.85, 1.0]),
                "colsample_bytree": trial.suggest_categorical(
                    parameter_name(config, family, "colsample_bytree"),
                    [0.7, 0.85, 1.0],
                ),
                "min_child_weight": trial.suggest_categorical(
                    parameter_name(config, family, "min_child_weight"),
                    [1.0, 5.0, 10.0],
                ),
                "reg_alpha": trial.suggest_categorical(
                    parameter_name(config, family, "reg_alpha"),
                    spec["alpha_choices"],
                ),
                "reg_lambda": trial.suggest_categorical(
                    parameter_name(config, family, "reg_lambda"),
                    [0.1, 1.0, 10.0],
                ),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": -1,
                "device": "auto",
            }
        )
    elif family == "rf":
        depth_min, depth_max, depth_step = spec["max_depth"]
        base.update(
            {
                "n_estimators": trial.suggest_categorical(
                    parameter_name(config, family, "n_estimators"),
                    spec["n_estimators"],
                ),
                "max_depth": trial.suggest_int(
                    parameter_name(config, family, "max_depth"),
                    int(depth_min),
                    int(depth_max),
                    step=int(depth_step),
                ),
                "min_samples_leaf": trial.suggest_categorical(
                    parameter_name(config, family, "min_samples_leaf"),
                    [1, 5, 10, 20],
                ),
                "min_samples_split": trial.suggest_categorical(
                    parameter_name(config, family, "min_samples_split"),
                    [2, 5, 10, 20],
                ),
                "max_features": trial.suggest_categorical(
                    parameter_name(config, family, "max_features"),
                    ["sqrt", "log2"],
                ),
                "class_weight": None,
                "bootstrap": True,
                "n_jobs": -1,
                "device": "auto",
                "n_streams": 4,
            }
        )
    else:
        raise ValueError(f"Unsupported search family: {family}")
    payload = {
        "model_type": family,
        "variation_id": config.variation_id,
        "train_mode": "static_baseline",
        "force_run": False,
        "search_id": config.study_name,
        "trial_number": int(trial.number),
        "trial_source": "optuna",
        "search_phase": config.search_phase,
        "objective_threshold": float(threshold),
        "requested_test_length": int(test_length),
        "hyperparameters": base,
    }
    validate_request(payload)
    return payload, float(threshold)


def request_for_trial(
    trial: optuna.Trial,
    config: SearchConfig,
    *,
    forced_family: str | None = None,
) -> tuple[dict[str, Any], float]:
    family = str(forced_family or trial.suggest_categorical(family_parameter_name(config), config.families)).lower()
    payload, threshold = suggest_interval_trial(trial, config, family)
    if forced_family:
        payload["probe_family"] = family
    return payload, threshold


def ask_and_write_request(paths: QuantStreamPaths, payload: dict[str, Any], trial: optuna.Trial) -> str:
    model_id = compute_model_id(payload)
    trial.set_user_attr("search_id", payload["search_id"])
    trial.set_user_attr("trial_number", int(payload["trial_number"]))
    trial.set_user_attr("trial_source", "optuna")
    trial.set_user_attr("model_id", model_id)
    trial.set_user_attr("model_family", payload["model_type"])
    trial.set_user_attr("threshold", payload["objective_threshold"])
    trial.set_user_attr("requested_test_length", payload["requested_test_length"])
    trial.set_user_attr("search_phase", payload.get("search_phase", "-"))
    if payload.get("probe_family"):
        trial.set_user_attr("probe_family", payload["probe_family"])
    trial.set_user_attr("request_path", str(paths.run_requests_dir / f"trial_{trial.number}.yaml"))
    write_yaml_file(paths.run_requests_dir / f"trial_{trial.number}.yaml", payload)
    return model_id


def harvest_done(paths: QuantStreamPaths, payload: dict[str, Any]) -> tuple[float, float, int]:
    model_id = str(payload.get("model_id") or compute_model_id(payload))
    results = result_frame(paths, int(payload["variation_id"]))
    evaluated_rows = filled_result_length(results, model_id, 5000)
    if evaluated_rows is None:
        evaluated_rows = int(payload.get("requested_test_length") or payload.get("resolved_test_length") or 0) or None
    precision, density, executed, _effective_threshold, _used_fallback = precision_density(
        results,
        model_id,
        threshold=float(payload.get("objective_threshold", 0.5)),
        evaluated_rows=evaluated_rows,
    )
    return precision, density, executed


def warm_start_manual_runs(study: optuna.Study, paths: QuantStreamPaths, config: SearchConfig) -> None:
    imported_ids = {str(trial.user_attrs.get("model_id")) for trial in study.trials if trial.user_attrs.get("model_id")}
    results = result_frame(paths, config.variation_id)
    if results.empty:
        return
    for done_path in sorted(paths.runs_done_dir.glob("*.y*ml")):
        try:
            payload = read_yaml_file(done_path)
        except Exception:
            continue
        if str(payload.get("trial_source", "")) == "optuna":
            continue
        if str(payload.get("train_mode", "")).lower() != "static_baseline":
            continue
        model_id = str(payload.get("model_id") or "")
        if not model_id or model_id in imported_ids:
            continue
        threshold = 0.5
        try:
            precision, density, executed, _effective_threshold, _used_fallback = precision_density(
                results,
                model_id,
                threshold=threshold,
            )
        except Exception:
            continue
        trial = optuna.trial.create_trial(
            values=[precision, density],
            params={},
            distributions={},
            user_attrs={
                "trial_source": "manual_warm_start",
                "model_id": model_id,
                "model_family": str(payload.get("model_type", "")),
                "threshold": threshold,
                "executed_count": executed,
            },
        )
        study.add_trial(trial)
        imported_ids.add(model_id)


def finished_count(study: optuna.Study) -> int:
    finished = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED}
    return sum(1 for trial in study.trials if trial.state in finished)


def pending_search_trials(study: optuna.Study, config: SearchConfig) -> list[optuna.trial.FrozenTrial]:
    waiting = {optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING}
    return [trial for trial in study.trials if trial.state in waiting and is_optuna_search_trial(trial, config)]


def fail_orphan_open_trial(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    paths: QuantStreamPaths,
    config: SearchConfig,
) -> bool:
    trial_number = int(trial.number)
    if matching_payloads(paths, config.study_name, trial_number):
        return False
    request_path = str(trial.user_attrs.get("request_path") or "")
    if request_path and Path(request_path).exists():
        return False
    print(
        (
            f"Open trial {trial_number} has no pending/done/rejected YAML"
            "; marking it failed so the search can continue."
        ),
        flush=True,
    )
    study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
    write_top5(study, config)
    return True


def top5_rows(study: optuna.Study, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or not trial.values:
            continue
        precision, density = float(trial.values[0]), float(trial.values[1])
        score = precision * density
        rows.append(
            {
                "rank": 0,
                "model_id": str(trial.user_attrs.get("model_id") or "-"),
                "family": str(trial.user_attrs.get("model_family") or "-"),
                "precision_at_threshold": precision,
                "trade_density": density,
                "threshold": trial.user_attrs.get("threshold", "-"),
                "efficiency_score": score,
                "source_trial_number": int(trial.number),
                "trial_source": str(trial.user_attrs.get("trial_source") or "-"),
            }
        )
    rows.sort(key=lambda row: float(row["efficiency_score"]), reverse=True)
    selected = rows[:top_k]
    for index, row in enumerate(selected, start=1):
        row["rank"] = index
    while len(selected) < top_k:
        selected.append(
            {
                "rank": len(selected) + 1,
                "model_id": "-",
                "family": "-",
                "precision_at_threshold": "-",
                "trade_density": "-",
                "threshold": "-",
                "efficiency_score": "-",
                "source_trial_number": "-",
                "trial_source": "-",
            }
        )
    return selected


def completed_run_metrics(paths: QuantStreamPaths, config: SearchConfig) -> list[dict[str, Any]]:
    results = result_frame(paths, config.variation_id)
    if results.empty:
        return []
    rows: list[dict[str, Any]] = []
    done_paths = sorted(paths.runs_done_dir.glob("*.y*ml"))
    for index, done_path in enumerate(done_paths, start=1):
        if index == 1 or index % 25 == 0 or index == len(done_paths):
            print(f"Recalculating optimization metrics: {index}/{len(done_paths)} run files scanned...", flush=True)
        try:
            payload = read_yaml_file(done_path)
        except Exception:
            continue
        if int(payload.get("variation_id", -1)) != config.variation_id:
            continue
        if str(payload.get("train_mode", "")).lower() != "static_baseline":
            continue
        family = str(payload.get("model_type", "")).lower()
        if family not in DEFAULT_FAMILIES:
            continue
        model_id = str(payload.get("model_id") or compute_model_id(payload))
        original_threshold = float(payload.get("objective_threshold", 0.5))
        trial_source = str(payload.get("trial_source") or "manual")
        search_phase = str(payload.get("search_phase") or "-")
        if trial_source == "optuna" and search_phase == MID_PHASE:
            threshold = original_threshold
            threshold_policy = "mid_phase_optuna_objective_threshold"
        else:
            threshold = 0.5
            threshold_policy = "default_0.5"
        evaluated_rows = filled_result_length(results, model_id, 5000)
        if evaluated_rows is None:
            evaluated_rows = (
                int(payload.get("requested_test_length") or payload.get("resolved_test_length") or 0) or None
            )
        try:
            precision, density, executed, effective_threshold, used_fallback = precision_density(
                results,
                model_id,
                threshold=threshold,
                evaluated_rows=evaluated_rows,
            )
        except Exception:
            continue
        score = precision * density
        rows.append(
            {
                "trial": payload.get("trial_number", "-"),
                "state": "COMPLETE",
                "model_id": model_id,
                "family": family,
                "precision_at_threshold": precision,
                "trade_density": density,
                "efficiency_score": score,
                "threshold": threshold,
                "original_threshold": original_threshold,
                "effective_threshold": effective_threshold,
                "threshold_fallback_used": used_fallback,
                "evaluated_rows": evaluated_rows or "-",
                "executed_count": executed,
                "trial_source": trial_source,
                "search_id": str(payload.get("search_id") or "-"),
                "search_phase": search_phase,
                "probe_family": str(payload.get("probe_family") or "-"),
                "threshold_policy": threshold_policy,
                "source_path": str(done_path),
            }
        )
    return rows


def top_candidate_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: float(row.get("efficiency_score") or 0.0), reverse=True)[:top_k]
    output: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        output.append(
            {
                "rank": index,
                "model_id": row.get("model_id", "-"),
                "family": row.get("family", "-"),
                "precision_at_threshold": row.get("precision_at_threshold", "-"),
                "trade_density": row.get("trade_density", "-"),
                "threshold": row.get("threshold", "-"),
                "efficiency_score": row.get("efficiency_score", "-"),
                "source_trial_number": row.get("trial", "-"),
                "trial_source": row.get("trial_source", "-"),
            }
        )
    while len(output) < top_k:
        output.append(
            {
                "rank": len(output) + 1,
                "model_id": "-",
                "family": "-",
                "precision_at_threshold": "-",
                "trade_density": "-",
                "threshold": "-",
                "efficiency_score": "-",
                "source_trial_number": "-",
                "trial_source": "-",
            }
        )
    return output


def write_top5(study: optuna.Study, config: SearchConfig, paths: QuantStreamPaths | None = None) -> Path:
    path = config.storage_path.parent / f"{config.study_name}_top5.yaml"
    completed = sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)
    total_finished = finished_count(study)
    status = "complete" if total_finished >= config.target_trials else "in_progress"
    metric_rows = completed_run_metrics(paths, config) if paths is not None else []
    payload = {
        "study_name": config.study_name,
        "variation_id": config.variation_id,
        "status": status,
        "completed_trials": completed,
        "finished_trials": total_finished,
        "target_trials": config.target_trials,
        "ranking_metric": config.ranking_metric,
        "auto_promote_to_prod": config.auto_promote_to_prod,
        "top_k": config.top_k,
        "production_candidates": (
            top_candidate_rows(metric_rows, config.top_k) if metric_rows else top5_rows(study, config.top_k)
        ),
    }
    write_yaml_file(path, payload)
    return path


def completed_probe_counts(paths: QuantStreamPaths, config: SearchConfig) -> dict[str, int]:
    counts = {family: 0 for family in config.families}
    for done_path in sorted(paths.runs_done_dir.glob("*.y*ml")):
        try:
            payload = read_yaml_file(done_path)
        except Exception:
            continue
        if str(payload.get("search_id", "")) != config.study_name:
            continue
        if str(payload.get("search_phase", "")) != FAMILY_PROBE_PHASE:
            continue
        family = str(payload.get("probe_family") or payload.get("model_type") or "").lower()
        if family in counts:
            counts[family] += 1
    return counts


def completed_family_counts_for_probe(paths: QuantStreamPaths, config: SearchConfig) -> dict[str, int]:
    counts = {family: 0 for family in config.families}
    for row in completed_run_metrics(paths, config):
        family = str(row.get("family") or "").lower()
        if family in counts:
            counts[family] += 1
    return counts


def next_probe_family(paths: QuantStreamPaths, config: SearchConfig) -> str | None:
    counts = completed_family_counts_for_probe(paths, config)
    missing = [family for family in config.families if counts.get(family, 0) < config.probe_trials_per_family]
    if not missing:
        return None
    return sorted(missing, key=lambda family: (counts.get(family, 0), config.families.index(family)))[0]


def rank_families(rows: list[dict[str, Any]], families: list[str]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for family in families:
        scores = sorted(
            [float(row.get("efficiency_score") or 0.0) for row in rows if row.get("family") == family],
            reverse=True,
        )
        probe_count = sum(
            1 for row in rows if row.get("family") == family and str(row.get("search_phase")) == FAMILY_PROBE_PHASE
        )
        if scores:
            best_score = scores[0]
            top5_average = sum(scores[:5]) / len(scores[:5])
        else:
            best_score = 0.0
            top5_average = 0.0
        ranked.append(
            {
                "family": family,
                "completed_count": len(scores),
                "probe_count": probe_count,
                "best_score": best_score,
                "top5_average_score": top5_average,
                "selected": False,
            }
        )
    ranked.sort(
        key=lambda row: (
            float(row["best_score"]),
            float(row["top5_average_score"]),
            int(row["completed_count"]),
        ),
        reverse=True,
    )
    return ranked


def complete_family_probe(config_path: Path, paths: QuantStreamPaths, config: SearchConfig) -> Path:
    print("Computing final family rankings from completed/scorable runs...", flush=True)
    rows = completed_run_metrics(paths, config)
    ranked = rank_families(rows, config.families)
    selected = [row["family"] for row in ranked[: config.top_family_count]]
    print(f"Selected top families: {', '.join(selected) if selected else '-'}", flush=True)
    for row in ranked:
        row["selected"] = row["family"] in selected
    summary_path = config.storage_path.parent / f"{config.study_name}_family_selection.yaml"
    summary = {
        "study_name": config.study_name,
        "variation_id": config.variation_id,
        "previous_search_phase": config.search_phase,
        "next_search_phase": MID_PHASE,
        "probe_trials_per_family": config.probe_trials_per_family,
        "top_family_count": config.top_family_count,
        "selected_families": selected,
        "families": ranked,
    }
    write_yaml_file(summary_path, summary)
    raw = ensure_config(config_path)
    raw["families"] = selected
    raw["search_phase"] = MID_PHASE
    raw["family_probe_completed"] = True
    raw["top_family_count"] = config.top_family_count
    raw["probe_trials_per_family"] = config.probe_trials_per_family
    write_yaml_file(config_path, raw)
    return summary_path


def select_top_families_now(config_path: Path) -> int:
    config = load_config(config_path)
    paths = QuantStreamPaths()
    selection_config = replace(
        config,
        families=DEFAULT_FAMILIES,
        search_phase=FAMILY_PROBE_PHASE,
        family_probe_completed=False,
    )
    print(
        ("Recalculating metrics and selecting top families across: " f"{', '.join(selection_config.families)}"),
        flush=True,
    )
    top5_path = write_top5(load_study(config), selection_config, paths)
    summary_path = complete_family_probe(config_path, paths, selection_config)
    print(f"Updated top candidates: {top5_path}", flush=True)
    print(f"Updated family selection: {summary_path}", flush=True)
    return 0


def reconcile_trial_number(
    study: optuna.Study,
    trial_number: int,
    paths: QuantStreamPaths,
    config: SearchConfig,
    trial: optuna.Trial | None = None,
) -> bool:
    for status, _path, payload in matching_payloads(paths, config.study_name, trial_number):
        if status == "pending":
            return False
        if status == "rejected":
            study.tell(trial if trial is not None else trial_number, state=optuna.trial.TrialState.FAIL)
            write_top5(study, config)
            return True
        if status == "done":
            model_id = str(payload.get("model_id") or compute_model_id(payload))
            precision, density, executed = harvest_done(paths, payload)
            if trial is not None:
                trial.set_user_attr("model_id", model_id)
                trial.set_user_attr("model_family", str(payload.get("model_type", "")))
                trial.set_user_attr("executed_count", executed)
            study.tell(trial if trial is not None else trial_number, values=[precision, density])
            write_top5(study, config)
            return True
    return False


def wait_for_open_trial(study: optuna.Study, trial_number: int, paths: QuantStreamPaths, config: SearchConfig) -> None:
    print(f"Waiting for existing trial {trial_number} to finish...", flush=True)
    while True:
        if reconcile_trial_number(study, trial_number, paths, config):
            print(f"Trial {trial_number} finished and was reconciled.", flush=True)
            return
        time.sleep(config.poll_seconds)


def run_search(config_path: Path, *, family_probe_only: bool = False) -> int:
    config = load_config(config_path)
    paths = QuantStreamPaths()
    print(
        (
            f"Starting optimization: study={config.study_name}, phase={config.search_phase}, "
            f"target_trials={config.target_trials}, families={', '.join(config.families)}"
        ),
        flush=True,
    )
    study = load_study(config)
    if config.warm_start_manual_runs:
        print("Scanning completed manual runs for warm-start import...", flush=True)
        before_warm_start = len(study.trials)
        warm_start_manual_runs(study, paths, config)
        imported = len(study.trials) - before_warm_start
        print(f"Warm-start scan complete. Imported {imported} new manual trials.", flush=True)
    try:
        if config.search_phase == FAMILY_PROBE_PHASE and not config.family_probe_completed:
            print(
                (
                    f"Family probe mode active. Need at least {config.probe_trials_per_family} "
                    "completed/scorable runs per family before selecting top families."
                ),
                flush=True,
            )
            while True:
                open_trials = pending_search_trials(study, config)
                if open_trials:
                    if fail_orphan_open_trial(study, open_trials[0], paths, config):
                        continue
                    print(
                        f"Found existing open trial {open_trials[0].number}; no new YAML will be created yet.",
                        flush=True,
                    )
                    wait_for_open_trial(study, int(open_trials[0].number), paths, config)
                    continue
                family_counts = completed_family_counts_for_probe(paths, config)
                counts_text = ", ".join(f"{name}={count}" for name, count in family_counts.items())
                print(f"Current completed/scorable family counts: {counts_text}", flush=True)
                family = next_probe_family(paths, config)
                if family is None:
                    print(
                        "All families reached the required count. Recalculating metrics and selecting top families...",
                        flush=True,
                    )
                    top5_path = write_top5(study, config, paths)
                    summary_path = complete_family_probe(config_path, paths, config)
                    print(
                        json.dumps(
                            {
                                "event": "family_probe_complete",
                                "top5": str(top5_path),
                                "summary": str(summary_path),
                            }
                        ),
                        flush=True,
                    )
                    return 0
                completed_for_family = int(family_counts.get(family, 0))
                missing_for_family = max(config.probe_trials_per_family - completed_for_family, 0)
                counts_text = ", ".join(f"{name}={count}" for name, count in family_counts.items())
                print(
                    (
                        f"Family probe: {family} "
                        f"completed {completed_for_family}/{config.probe_trials_per_family}; "
                        f"missing {missing_for_family}. All families: {counts_text}"
                    ),
                    flush=True,
                )
                trial = study.ask()
                payload, _threshold = request_for_trial(trial, config, forced_family=family)
                model_id = ask_and_write_request(paths, payload, trial)
                print(f"Wrote run request YAML for trial {trial.number}: model_id={model_id}", flush=True)
                print(
                    json.dumps(
                        {
                            "event": "family_probe_requested",
                            "trial": trial.number,
                            "family": family,
                            "model_id": model_id,
                        }
                    ),
                    flush=True,
                )
                while True:
                    if reconcile_trial_number(study, int(trial.number), paths, config, trial):
                        break
                    time.sleep(config.poll_seconds)
        elif family_probe_only:
            print(
                (
                    "Family probe is already complete or inactive. "
                    f"Current phase is {config.search_phase}; use run_optimization_search.ps1 to continue search."
                ),
                flush=True,
            )
            return 0
        while finished_count(study) < config.target_trials:
            print(
                f"Search progress: finished {finished_count(study)}/{config.target_trials} trials.",
                flush=True,
            )
            open_trials = pending_search_trials(study, config)
            if open_trials:
                if fail_orphan_open_trial(study, open_trials[0], paths, config):
                    continue
                print(
                    f"Found existing open trial {open_trials[0].number}; no new YAML will be created yet.",
                    flush=True,
                )
                wait_for_open_trial(study, int(open_trials[0].number), paths, config)
                continue
            trial = study.ask()
            payload, _threshold = request_for_trial(trial, config)
            model_id = ask_and_write_request(paths, payload, trial)
            print(f"Wrote run request YAML for trial {trial.number}: model_id={model_id}", flush=True)
            print(json.dumps({"event": "trial_requested", "trial": trial.number, "model_id": model_id}), flush=True)
            while True:
                if reconcile_trial_number(study, int(trial.number), paths, config, trial):
                    break
                time.sleep(config.poll_seconds)
        top5_path = write_top5(study, config, paths)
        print(json.dumps({"event": "search_complete", "top5": str(top5_path)}), flush=True)
        return 0
    except KeyboardInterrupt:
        top5_path = write_top5(study, config, paths)
        print(json.dumps({"event": "search_interrupted", "top5": str(top5_path)}), flush=True)
        return 130


def dry_run_family_probe(config_path: Path) -> int:
    config = load_config(config_path)
    paths = QuantStreamPaths()
    probe_counts = completed_probe_counts(paths, config)
    family_completion_counts = completed_family_counts_for_probe(paths, config)
    rows = completed_run_metrics(paths, config)
    ranked = rank_families(rows, config.families)
    selected = [row["family"] for row in ranked[: config.top_family_count]]
    for row in ranked:
        row["selected_if_completed_now"] = row["family"] in selected
    print(
        yaml.safe_dump(
            {
                "study_name": config.study_name,
                "search_phase": config.search_phase,
                "family_probe_completed": config.family_probe_completed,
                "probe_trials_per_family": config.probe_trials_per_family,
                "probe_counts": probe_counts,
                "family_completion_counts": family_completion_counts,
                "next_probe_family": next_probe_family(paths, config),
                "ranking_preview": ranked,
            },
            sort_keys=False,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable Quant-Stream Optuna search.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--family-probe-dry-run", action="store_true")
    parser.add_argument("--family-probe-only", action="store_true")
    parser.add_argument("--select-top-families-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.family_probe_dry_run:
        return dry_run_family_probe(Path(args.config))
    if args.select_top_families_only:
        return select_top_families_now(Path(args.config))
    return run_search(Path(args.config), family_probe_only=bool(args.family_probe_only))


if __name__ == "__main__":
    raise SystemExit(main())
