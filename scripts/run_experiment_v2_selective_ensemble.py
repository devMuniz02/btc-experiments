from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._experiment_eval_reuse import (
    evaluate_prediction_stream,
    load_prediction_stream_from_file,
    serialize_eval_result,
    summarize_actions_against_labels,
    threshold_pct_to_probability,
)
from src.utils.experiment_support import (
    EXPERIMENTS_V2_ROOT,
    THRESHOLDS,
    VARIANT_NAME,
    dedupe_rows_by_name,
    load_summary,
    load_summary_rows,
    normalize_model_variation,
    normalize_summary_rows,
    read_prediction_stream_parquet,
    target_row_from_model,
    target_row_prediction_output_path,
    target_row_saved_model_dir,
    write_prediction_stream_parquet,
    write_combinations_markdown,
    write_manifest_and_summary,
)
from scripts._helpers import write_json_atomic
from src.btc_direction_learning.env import ENV_VERSION_TERNARY, NONE_ACTION
from src.btc_direction_learning.train import build_portfolio_scenarios


SELECTIVE_FAMILY = "SELECTIVEENSEMBLE"
SELECTION_SPLIT = "PRE_HOLDOUT_5K"
HOLDOUT_SPLIT = "FIXED_5K_HOLDOUT"
DEFAULT_FAMILIES = ("LSTM", "TRANSFORMER", "PPO")
DEFAULT_VARIATIONS = ("BASE", "WINDOW_RETRAIN", "PPO_WINDOW_CONTINUE")


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    train_length: str
    window_length: str
    model_variation: str
    source_saved_model_path: str
    holdout_predictions_path: str
    validation_predictions_path: str
    validation_stream: dict[str, Any]
    holdout_stream: dict[str, Any]
    member_threshold: int
    validation_eval: dict[str, Any]
    validation_metrics: dict[str, float]
    score: tuple[float, float, float]

    @property
    def source_identity(self) -> tuple[str, str, str, str]:
        return (self.family, self.train_length, self.window_length, self.model_variation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a selective-edge ensemble on top of EXPERIMENTSV2 direct/window model prediction streams."
    )
    parser.add_argument("--output-dir", default=str(EXPERIMENTS_V2_ROOT / "1"))
    parser.add_argument("--time-variant", default="FULL", choices=["FULL"])
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--model-variations", nargs="+", default=list(DEFAULT_VARIATIONS))
    parser.add_argument("--search-mode", choices=["rules", "stacker", "both"], default="both")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--correlation-threshold", type=float, default=0.985)
    return parser.parse_args()


def _report_paths(output_root: Path) -> tuple[Path, Path]:
    report_dir = output_root / VARIANT_NAME / "reports"
    return (
        report_dir / "selective_ensemble_report.json",
        report_dir / "selective_ensemble_report.md",
    )


def _selection_prediction_path(row: dict[str, Any]) -> str:
    for key in (
        "validation_predictions_path",
        "selection_predictions_path",
        "source_validation_predictions_path",
        "source_selection_predictions_path",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _holdout_prediction_path(row: dict[str, Any]) -> str:
    for key in ("predictions_path", "source_predictions_path"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("family") or "").upper(),
        str(row.get("train_length") or "").upper(),
        str(row.get("window_length") or "").upper(),
        normalize_model_variation(row.get("model_variation")),
    )


def _compute_daily_net_wins(eval_payload: dict[str, Any]) -> float:
    test_payload = eval_payload.get("test", {}) if isinstance(eval_payload, dict) else {}
    portfolio = eval_payload.get("portfolio", {}) if isinstance(eval_payload, dict) else {}
    fixed_dollar = portfolio.get("fixed_dollar", {}) if isinstance(portfolio, dict) else {}
    trade_pnls = fixed_dollar.get("trade_pnls", []) if isinstance(fixed_dollar, dict) else []
    timestamps = test_payload.get("timestamps", []) if isinstance(test_payload, dict) else []
    if not isinstance(trade_pnls, list) or not isinstance(timestamps, list):
        return 0.0
    if not trade_pnls or len(trade_pnls) != len(timestamps):
        total = int(test_payload.get("accuracy_scored_count") or 0)
        if total <= 0:
            return 0.0
        wins = float(test_payload.get("accuracy") or 0.0) * total
        losses = total - wins
        return float((wins - losses) * 24.0 / total)
    by_day: dict[str, float] = {}
    for timestamp, pnl in zip(timestamps, trade_pnls):
        if pnl is None:
            continue
        by_day.setdefault(str(timestamp)[:10], 0.0)
        by_day[str(timestamp)[:10]] += float(pnl)
    if not by_day:
        return 0.0
    positive = sum(1 for value in by_day.values() if value > 0)
    negative = sum(1 for value in by_day.values() if value < 0)
    return float(positive - negative)


def _eval_metrics(eval_payload: dict[str, Any]) -> dict[str, float]:
    test_payload = eval_payload["test"]
    total_rows = max(1, len(test_payload.get("timestamps", [])))
    coverage = float(test_payload.get("accuracy_scored_count") or 0) / float(total_rows)
    return {
        "daily_net_wins": _compute_daily_net_wins(eval_payload),
        "taken_accuracy": float(test_payload.get("accuracy") or 0.0),
        "coverage": coverage,
        "taken_count": float(test_payload.get("accuracy_scored_count") or 0),
        "total_rows": float(total_rows),
    }


def _score_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["daily_net_wins"]),
        float(metrics["taken_accuracy"]),
        float(metrics["coverage"]),
    )


def _load_candidate_stream(path: str) -> dict[str, Any]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".parquet":
        payload = read_prediction_stream_parquet(source_path)
    else:
        raw_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(raw_payload, dict) and {"timestamps", "labels", "probability_up", "probability_down", "probability_none"}.issubset(
            raw_payload.keys()
        ):
            payload = {
                "timestamps": [str(value) for value in raw_payload.get("timestamps", [])],
                "labels": [int(value) for value in raw_payload.get("labels", [])],
                "probability_up": [float(value) for value in raw_payload.get("probability_up", [])],
                "probability_down": [float(value) for value in raw_payload.get("probability_down", [])],
                "probability_none": [float(value) for value in raw_payload.get("probability_none", [])],
                "source_kind": str(raw_payload.get("source_kind") or "predictions_file"),
                "predictions_path": str(source_path.resolve()),
            }
        else:
            payload = load_prediction_stream_from_file(path)
    timestamps = payload.get("timestamps", [])
    labels = payload.get("labels", [])
    if not isinstance(timestamps, list) or not isinstance(labels, list) or not timestamps or len(timestamps) != len(labels):
        raise RuntimeError(f"Prediction stream {path} is missing aligned timestamps/labels.")
    return payload


def _choose_best_member_threshold(stream: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, float]]:
    best_threshold = 50
    best_eval = evaluate_prediction_stream(prediction_stream=stream, threshold_pct=50, env_version=ENV_VERSION_TERNARY)
    best_metrics = _eval_metrics(best_eval)
    best_score = _score_tuple(best_metrics)
    for threshold in THRESHOLDS[1:]:
        current_eval = evaluate_prediction_stream(
            prediction_stream=stream,
            threshold_pct=int(threshold),
            env_version=ENV_VERSION_TERNARY,
        )
        current_metrics = _eval_metrics(current_eval)
        current_score = _score_tuple(current_metrics)
        if current_score > best_score:
            best_threshold = int(threshold)
            best_eval = current_eval
            best_metrics = current_metrics
            best_score = current_score
    return best_threshold, best_eval, best_metrics


def _build_candidate_registry(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[Candidate]:
    allowed_families = {str(value).strip().upper() for value in args.families if str(value).strip()}
    allowed_variations = {normalize_model_variation(value) for value in args.model_variations if str(value).strip()}
    chosen_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in normalize_summary_rows(rows):
        if str(row.get("status") or "") != "completed":
            continue
        if str(row.get("time_variant") or "").upper() != "FULL":
            continue
        family = str(row.get("family") or "").upper()
        if family not in allowed_families:
            continue
        variation = normalize_model_variation(row.get("model_variation"))
        if variation not in allowed_variations:
            continue
        validation_path = _selection_prediction_path(row)
        holdout_path = _holdout_prediction_path(row)
        if not validation_path or not holdout_path:
            continue
        identity = _row_identity(row)
        current = chosen_rows.get(identity)
        if current is None:
            chosen_rows[identity] = row
            continue
        current_threshold = int(current.get("threshold_pct") or 9999)
        new_threshold = int(row.get("threshold_pct") or 9999)
        if new_threshold < current_threshold:
            chosen_rows[identity] = row

    candidates: list[Candidate] = []
    for row in chosen_rows.values():
        validation_stream = _load_candidate_stream(_selection_prediction_path(row))
        holdout_stream = _load_candidate_stream(_holdout_prediction_path(row))
        if validation_stream.get("timestamps") != holdout_stream.get("timestamps") and len(validation_stream.get("timestamps", [])) == len(
            holdout_stream.get("timestamps", [])
        ):
            pass
        threshold, validation_eval, metrics = _choose_best_member_threshold(validation_stream)
        candidates.append(
            Candidate(
                name=str(row.get("normalized_base_name") or row.get("name") or ""),
                family=str(row.get("family") or "").upper(),
                train_length=str(row.get("train_length") or "").upper(),
                window_length=str(row.get("window_length") or "").upper(),
                model_variation=normalize_model_variation(row.get("model_variation")),
                source_saved_model_path=str(row.get("saved_model_path") or row.get("source_saved_model_path") or ""),
                holdout_predictions_path=_holdout_prediction_path(row),
                validation_predictions_path=_selection_prediction_path(row),
                validation_stream=validation_stream,
                holdout_stream=holdout_stream,
                member_threshold=threshold,
                validation_eval=validation_eval,
                validation_metrics=metrics,
                score=_score_tuple(metrics),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _signal_series(stream: dict[str, Any]) -> np.ndarray:
    up = np.asarray(stream.get("probability_up", []), dtype=np.float64)
    down = np.asarray(stream.get("probability_down", []), dtype=np.float64)
    if up.size != down.size:
        raise RuntimeError("Prediction stream probability arrays are misaligned.")
    return up - down


def _prune_correlated_candidates(candidates: list[Candidate], max_candidates: int, correlation_threshold: float) -> list[Candidate]:
    kept: list[Candidate] = []
    kept_signals: list[np.ndarray] = []
    for candidate in candidates:
        signal = _signal_series(candidate.validation_stream)
        is_redundant = False
        for existing_signal in kept_signals:
            if signal.size != existing_signal.size or signal.size < 2:
                continue
            correlation = float(np.corrcoef(signal, existing_signal)[0, 1])
            if not math.isnan(correlation) and abs(correlation) >= correlation_threshold:
                is_redundant = True
                break
        if is_redundant:
            continue
        kept.append(candidate)
        kept_signals.append(signal)
        if len(kept) >= max_candidates:
            break
    return kept


def _average_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _candidate_weight(candidate: Candidate) -> float:
    return max(0.001, candidate.validation_metrics["taken_accuracy"] - 0.5) + max(0.0, candidate.validation_metrics["coverage"])


def _build_rules_action_stream(
    candidates: list[Candidate],
    streams: list[dict[str, Any]],
    *,
    minimum_votes: int,
    use_score_weights: bool,
    abstain_on_disagreement: bool,
) -> dict[str, Any]:
    timestamps = [str(value) for value in streams[0].get("timestamps", [])]
    labels = [int(value) for value in streams[0].get("labels", [])]
    actions: list[int] = []
    probabilities: list[float | None] = []
    detail_rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        up_weight = 0.0
        down_weight = 0.0
        eligible_votes = 0
        winning_confidences: list[float] = []
        per_member_votes: list[dict[str, Any]] = []
        for candidate, stream in zip(candidates, streams):
            up_prob = float(stream["probability_up"][index])
            down_prob = float(stream["probability_down"][index])
            activation_threshold = threshold_pct_to_probability(candidate.member_threshold)
            action: int | None = None
            confidence: float | None = None
            if up_prob > down_prob and up_prob >= activation_threshold:
                action = 1
                confidence = up_prob
            elif down_prob > up_prob and down_prob >= activation_threshold:
                action = 0
                confidence = down_prob
            weight = _candidate_weight(candidate) if use_score_weights else 1.0
            per_member_votes.append(
                {
                    "member_name": candidate.name,
                    "member_threshold": candidate.member_threshold,
                    "action": action,
                    "confidence": confidence,
                    "weight": weight,
                }
            )
            if action is None:
                continue
            eligible_votes += 1
            if action == 1:
                up_weight += weight
            else:
                down_weight += weight
            if confidence is not None:
                winning_confidences.append(float(confidence))
        if eligible_votes < minimum_votes:
            chosen_action = NONE_ACTION
            chosen_probability = None
        elif up_weight > down_weight:
            if abstain_on_disagreement and down_weight > 0:
                chosen_action = NONE_ACTION
                chosen_probability = None
            else:
                chosen_action = 1
                chosen_probability = _average_or_none(winning_confidences)
        elif down_weight > up_weight:
            if abstain_on_disagreement and up_weight > 0:
                chosen_action = NONE_ACTION
                chosen_probability = None
            else:
                chosen_action = 0
                chosen_probability = _average_or_none(winning_confidences)
        else:
            chosen_action = NONE_ACTION
            chosen_probability = None
        actions.append(int(chosen_action))
        probabilities.append(chosen_probability)
        detail_rows.append(
            {
                "timestamp": timestamp,
                "label": labels[index],
                "action": int(chosen_action),
                "prediction": chosen_probability,
                "eligible_votes": eligible_votes,
                "up_weight": up_weight,
                "down_weight": down_weight,
                "member_votes": per_member_votes,
            }
        )
    return {
        "timestamps": timestamps,
        "labels": labels,
        "actions": actions,
        "chosen_action_probabilities": probabilities,
        "rows": detail_rows,
        "source_kind": "selective_rules_ensemble",
    }


def _evaluate_action_stream(stream: dict[str, Any]) -> dict[str, Any]:
    payload = evaluate_prediction_stream(
        prediction_stream=stream,
        threshold_pct=50,
        env_version=ENV_VERSION_TERNARY,
    )
    return payload


def _build_rules_search_results(candidates: list[Candidate], min_coverage: float, max_members: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    results: list[dict[str, Any]] = []
    subset_limit = min(len(candidates), max_members)
    for subset_size in range(1, subset_limit + 1):
        for subset in itertools.combinations(candidates, subset_size):
            validation_streams = [item.validation_stream for item in subset]
            for minimum_votes in range(1, subset_size + 1):
                for use_score_weights in (False, True):
                    for abstain_on_disagreement in (False, True):
                        stream = _build_rules_action_stream(
                            list(subset),
                            validation_streams,
                            minimum_votes=minimum_votes,
                            use_score_weights=use_score_weights,
                            abstain_on_disagreement=abstain_on_disagreement,
                        )
                        eval_payload = _evaluate_action_stream(stream)
                        metrics = _eval_metrics(eval_payload)
                        if metrics["coverage"] < min_coverage:
                            continue
                        results.append(
                            {
                                "kind": "rules",
                                "members": list(subset),
                                "validation_stream": stream,
                                "validation_eval": eval_payload,
                                "validation_metrics": metrics,
                                "selection_metric": _score_tuple(metrics),
                                "config": {
                                    "minimum_votes": minimum_votes,
                                    "use_score_weights": use_score_weights,
                                    "abstain_on_disagreement": abstain_on_disagreement,
                                },
                            }
                        )
    results.sort(key=lambda item: item["selection_metric"], reverse=True)
    return results


def _stacker_features(streams: list[dict[str, Any]]) -> np.ndarray:
    matrix: list[list[float]] = []
    size = len(streams[0].get("timestamps", []))
    for index in range(size):
        row: list[float] = []
        eligible = 0.0
        for stream in streams:
            up = float(stream["probability_up"][index])
            down = float(stream["probability_down"][index])
            spread = up - down
            row.extend([up, down, spread, max(up, down)])
            if max(up, down) >= 0.5:
                eligible += 1.0
        row.append(eligible)
        matrix.append(row)
    return np.asarray(matrix, dtype=np.float64)


def _blocked_folds(size: int, blocks: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    if size < 3:
        return [(np.arange(size, dtype=np.int64), np.arange(size, dtype=np.int64))]
    blocks = max(2, min(blocks, size))
    indices = np.arange(size, dtype=np.int64)
    splits = np.array_split(indices, blocks)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for split in splits:
        if split.size == 0:
            continue
        train = np.setdiff1d(indices, split)
        if train.size == 0:
            train = indices
        folds.append((train, split))
    return folds


def _fit_stacker_result(candidates: list[Candidate], min_coverage: float) -> dict[str, Any] | None:
    streams = [item.validation_stream for item in candidates]
    features = _stacker_features(streams)
    labels = np.asarray(streams[0].get("labels", []), dtype=np.int64)
    timestamps = [str(value) for value in streams[0].get("timestamps", [])]
    if features.shape[0] != labels.shape[0] or features.shape[0] == 0:
        return None

    oof_probabilities = np.zeros((features.shape[0], 2), dtype=np.float64)
    for train_indices, test_indices in _blocked_folds(features.shape[0]):
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
        model.fit(features[train_indices], labels[train_indices])
        probabilities = model.predict_proba(features[test_indices])
        classes = list(model.classes_)
        aligned = np.zeros((len(test_indices), 2), dtype=np.float64)
        for position, klass in enumerate(classes):
            aligned[:, int(klass)] = probabilities[:, position]
        oof_probabilities[test_indices] = aligned

    best_threshold = 50
    best_metrics: dict[str, float] | None = None
    best_eval: dict[str, Any] | None = None
    best_stream: dict[str, Any] | None = None
    best_score = (float("-inf"), float("-inf"), float("-inf"))
    for threshold in THRESHOLDS:
        threshold_probability = threshold_pct_to_probability(int(threshold))
        actions: list[int] = []
        chosen_probabilities: list[float | None] = []
        for probabilities in oof_probabilities:
            down_prob = float(probabilities[0])
            up_prob = float(probabilities[1])
            winner = max(up_prob, down_prob)
            if winner < threshold_probability:
                actions.append(NONE_ACTION)
                chosen_probabilities.append(None)
            elif up_prob >= down_prob:
                actions.append(1)
                chosen_probabilities.append(up_prob)
            else:
                actions.append(0)
                chosen_probabilities.append(down_prob)
        stream = {
            "timestamps": timestamps,
            "labels": labels.tolist(),
            "actions": actions,
            "chosen_action_probabilities": chosen_probabilities,
            "source_kind": "selective_stacker_oof",
        }
        eval_payload = _evaluate_action_stream(stream)
        metrics = _eval_metrics(eval_payload)
        score = _score_tuple(metrics)
        if metrics["coverage"] >= min_coverage and score > best_score:
            best_threshold = int(threshold)
            best_metrics = metrics
            best_eval = eval_payload
            best_stream = stream
            best_score = score
    if best_metrics is None or best_eval is None or best_stream is None:
        return None

    final_model = LogisticRegression(max_iter=2000, solver="lbfgs")
    final_model.fit(features, labels)
    return {
        "kind": "stacker",
        "members": list(candidates),
        "model": final_model,
        "selected_threshold": best_threshold,
        "validation_stream": best_stream,
        "validation_eval": best_eval,
        "validation_metrics": best_metrics,
        "selection_metric": _score_tuple(best_metrics),
        "config": {
            "abstain_threshold_pct": best_threshold,
            "feature_columns_per_member": ["probability_up", "probability_down", "spread", "directional_max"],
        },
    }


def _predict_stacker_actions(model: LogisticRegression, streams: list[dict[str, Any]], threshold_pct: int) -> dict[str, Any]:
    features = _stacker_features(streams)
    labels = [int(value) for value in streams[0].get("labels", [])]
    timestamps = [str(value) for value in streams[0].get("timestamps", [])]
    probabilities_raw = model.predict_proba(features)
    classes = list(model.classes_)
    probabilities = np.zeros((features.shape[0], 2), dtype=np.float64)
    for index, klass in enumerate(classes):
        probabilities[:, int(klass)] = probabilities_raw[:, index]
    activation = threshold_pct_to_probability(int(threshold_pct))
    actions: list[int] = []
    chosen: list[float | None] = []
    rows: list[dict[str, Any]] = []
    for timestamp, label, probability_row in zip(timestamps, labels, probabilities):
        down_prob = float(probability_row[0])
        up_prob = float(probability_row[1])
        winner = max(up_prob, down_prob)
        if winner < activation:
            action = NONE_ACTION
            chosen_probability = None
        elif up_prob >= down_prob:
            action = 1
            chosen_probability = up_prob
        else:
            action = 0
            chosen_probability = down_prob
        actions.append(action)
        chosen.append(chosen_probability)
        rows.append(
            {
                "timestamp": timestamp,
                "label": label,
                "action": action,
                "prediction": chosen_probability,
                "probability_up": up_prob,
                "probability_down": down_prob,
            }
        )
    return {
        "timestamps": timestamps,
        "labels": labels,
        "actions": actions,
        "chosen_action_probabilities": chosen,
        "rows": rows,
        "source_kind": "selective_stacker",
    }


def _build_holdout_result(candidate_result: dict[str, Any]) -> dict[str, Any]:
    members: list[Candidate] = list(candidate_result["members"])
    holdout_streams = [item.holdout_stream for item in members]
    if candidate_result["kind"] == "rules":
        stream = _build_rules_action_stream(
            members,
            holdout_streams,
            minimum_votes=int(candidate_result["config"]["minimum_votes"]),
            use_score_weights=bool(candidate_result["config"]["use_score_weights"]),
            abstain_on_disagreement=bool(candidate_result["config"]["abstain_on_disagreement"]),
        )
    else:
        stream = _predict_stacker_actions(
            candidate_result["model"],
            holdout_streams,
            int(candidate_result["config"]["abstain_threshold_pct"]),
        )
    eval_payload = _evaluate_action_stream(stream)
    metrics = _eval_metrics(eval_payload)
    return {
        "stream": stream,
        "eval": eval_payload,
        "metrics": metrics,
    }


def _selection_payload(metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "daily_net_wins": float(metrics["daily_net_wins"]),
        "taken_accuracy": float(metrics["taken_accuracy"]),
        "coverage": float(metrics["coverage"]),
        "taken_count": int(metrics["taken_count"]),
        "total_rows": int(metrics["total_rows"]),
    }


def _serialize_ensemble_row(
    output_root: Path,
    *,
    model_variation: str,
    threshold_pct: int,
    selection_split: str,
    source_provenance: str,
    member_candidates: list[Candidate],
    validation_result: dict[str, Any],
    holdout_result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "name": f"{SELECTIVE_FAMILY}_ALL_FULL_{model_variation}_{int(threshold_pct)}",
        "family": SELECTIVE_FAMILY,
        "train_length": "ALL",
        "window_length": "",
        "model_variation": model_variation,
        "time_variant": "FULL",
        "threshold_pct": int(threshold_pct),
        "env_version": ENV_VERSION_TERNARY,
        "status": "completed",
        "selection_split": selection_split,
        "selection_metric": _selection_payload(validation_result["metrics"]),
        "coverage": float(holdout_result["metrics"]["coverage"]),
        "source_provenance": source_provenance,
        "source_member_names": [member.name for member in member_candidates],
        "source_member_paths": [member.source_saved_model_path for member in member_candidates],
        "member_thresholds": {member.name: int(member.member_threshold) for member in member_candidates},
        "member_weights": {member.name: float(_candidate_weight(member)) for member in member_candidates},
        "validation": {
            "split": selection_split,
            "test": validation_result["eval"]["test"],
            "portfolio": validation_result["eval"]["portfolio"],
            "metrics": _selection_payload(validation_result["metrics"]),
        },
        "holdout": {
            "split": HOLDOUT_SPLIT,
            "test": holdout_result["eval"]["test"],
            "portfolio": holdout_result["eval"]["portfolio"],
            "metrics": _selection_payload(holdout_result["metrics"]),
        },
        "test": holdout_result["eval"]["test"],
        "portfolio": holdout_result["eval"]["portfolio"],
        "stacker_config": config if "feature_columns_per_member" in config else {},
        "ensemble_config": config,
    }
    target = target_row_from_model(row)
    saved_model_dir = target_row_saved_model_dir(output_root, target)
    metadata_path = saved_model_dir / f"{row['name'].lower()}.metadata.json"
    predictions_path = target_row_prediction_output_path(
        output_root,
        target,
        threshold=int(threshold_pct),
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        "family": SELECTIVE_FAMILY,
        "model_variation": model_variation,
        "selection_split": selection_split,
        "holdout_split": HOLDOUT_SPLIT,
        "selection_metric": _selection_payload(validation_result["metrics"]),
        "holdout_metric": _selection_payload(holdout_result["metrics"]),
        "source_member_names": row["source_member_names"],
        "source_member_paths": row["source_member_paths"],
        "member_thresholds": row["member_thresholds"],
        "member_weights": row["member_weights"],
        "ensemble_config": config,
    }
    write_json_atomic(metadata_path, metadata_payload)
    write_prediction_stream_parquet(predictions_path, holdout_result["stream"])
    row["saved_model_path"] = str(metadata_path.resolve())
    row["source_saved_model_path"] = str(metadata_path.resolve())
    row["predictions_path"] = str(predictions_path.resolve())
    row["source_predictions_path"] = str(predictions_path.resolve())
    row["grouped_saved_model_dir"] = str(metadata_path.parent.resolve())
    row["grouped_saved_model_paths"] = [str(metadata_path.resolve())]
    return row


def _champion_row(output_root: Path, selected_row: dict[str, Any]) -> dict[str, Any]:
    champion = dict(selected_row)
    champion["model_variation"] = "CHAMPION"
    champion["name"] = f"{SELECTIVE_FAMILY}_ALL_FULL_CHAMPION_{int(selected_row.get('threshold_pct') or 50)}"
    champion["champion_source_name"] = str(selected_row.get("name") or "")
    champion["source_provenance"] = "SELECTIVE_CHAMPION_ALIAS"
    target = target_row_from_model(champion)
    saved_model_dir = target_row_saved_model_dir(output_root, target)
    metadata_path = saved_model_dir / f"{champion['name'].lower()}.metadata.json"
    predictions_path = target_row_prediction_output_path(
        output_root,
        target,
        threshold=int(champion.get("threshold_pct") or 50),
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    champion_metadata = {
        "family": SELECTIVE_FAMILY,
        "alias_for": selected_row["name"],
        "selection_metric": champion.get("selection_metric", {}),
        "holdout_metric": champion.get("holdout", {}).get("metrics", {}),
    }
    write_json_atomic(metadata_path, champion_metadata)
    if str(selected_row.get("predictions_path") or "").strip():
        source_predictions_path = Path(str(selected_row["predictions_path"]))
        if source_predictions_path.suffix.lower() == ".parquet":
            write_prediction_stream_parquet(predictions_path, read_prediction_stream_parquet(source_predictions_path))
        else:
            write_json_atomic(predictions_path, json.loads(source_predictions_path.read_text(encoding="utf-8")))
    champion["saved_model_path"] = str(metadata_path.resolve())
    champion["source_saved_model_path"] = str(metadata_path.resolve())
    champion["predictions_path"] = str(predictions_path.resolve())
    champion["source_predictions_path"] = str(predictions_path.resolve())
    champion["grouped_saved_model_dir"] = str(metadata_path.parent.resolve())
    champion["grouped_saved_model_paths"] = [str(metadata_path.resolve())]
    return champion


def _write_report(output_root: Path, *, candidate_pool: list[Candidate], built_rows: list[dict[str, Any]], champion_name: str) -> dict[str, str]:
    json_path, markdown_path = _report_paths(output_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = []
    for row in built_rows:
        validation_metrics = row.get("validation", {}).get("metrics", {})
        holdout_metrics = row.get("holdout", {}).get("metrics", {})
        leaderboard.append(
            {
                "name": row.get("name"),
                "model_variation": row.get("model_variation"),
                "validation_metrics": validation_metrics,
                "holdout_metrics": holdout_metrics,
                "source_member_names": row.get("source_member_names", []),
            }
        )
    payload = {
        "selection_split": SELECTION_SPLIT,
        "holdout_split": HOLDOUT_SPLIT,
        "candidate_pool": [
            {
                "name": item.name,
                "family": item.family,
                "train_length": item.train_length,
                "window_length": item.window_length,
                "model_variation": item.model_variation,
                "member_threshold": item.member_threshold,
                "validation_metrics": _selection_payload(item.validation_metrics),
            }
            for item in candidate_pool
        ],
        "leaderboard": leaderboard,
        "champion_name": champion_name,
    }
    write_json_atomic(json_path, payload)
    lines = [
        "# Selective Ensemble Report",
        "",
        f"- Selection split: `{SELECTION_SPLIT}`",
        f"- Holdout split: `{HOLDOUT_SPLIT}`",
        f"- Champion: `{champion_name}`",
        "",
        "## Candidate Pool",
        "",
    ]
    for candidate in candidate_pool:
        lines.append(
            f"- `{candidate.name}`: daily_net_wins={candidate.validation_metrics['daily_net_wins']:.3f}, "
            f"taken_accuracy={candidate.validation_metrics['taken_accuracy']:.4f}, coverage={candidate.validation_metrics['coverage']:.4f}"
        )
    lines.extend(["", "## Leaderboard", ""])
    for entry in leaderboard:
        validation_metrics = entry["validation_metrics"]
        holdout_metrics = entry["holdout_metrics"]
        lines.append(
            f"- `{entry['name']}`: "
            f"validation daily_net_wins={validation_metrics.get('daily_net_wins', 0.0):.3f}, "
            f"validation accuracy={validation_metrics.get('taken_accuracy', 0.0):.4f}, "
            f"validation coverage={validation_metrics.get('coverage', 0.0):.4f}; "
            f"holdout daily_net_wins={holdout_metrics.get('daily_net_wins', 0.0):.3f}, "
            f"holdout accuracy={holdout_metrics.get('taken_accuracy', 0.0):.4f}, "
            f"holdout coverage={holdout_metrics.get('coverage', 0.0):.4f}"
        )
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "report_json_path": str(json_path.resolve()),
        "report_markdown_path": str(markdown_path.resolve()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    existing_summary = load_summary(output_root)
    existing_rows = load_summary_rows(output_root)
    candidate_pool = _build_candidate_registry(args, existing_rows)
    shortlist = _prune_correlated_candidates(candidate_pool, args.max_candidates, float(args.correlation_threshold))
    if not shortlist:
        raise RuntimeError(
            "No selective-ensemble candidates were found. Rows must provide validation_predictions_path or selection_predictions_path."
        )

    built_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    if args.search_mode in {"rules", "both"}:
        rules_results = _build_rules_search_results(shortlist, float(args.min_coverage), int(args.max_members))
        if rules_results:
            best_rules = rules_results[0]
            holdout = _build_holdout_result(best_rules)
            row = _serialize_ensemble_row(
                output_root,
                model_variation="RULES",
                threshold_pct=50,
                selection_split=SELECTION_SPLIT,
                source_provenance="SELECTIVE_RULES_ENSEMBLE",
                member_candidates=list(best_rules["members"]),
                validation_result={
                    "stream": best_rules["validation_stream"],
                    "eval": best_rules["validation_eval"],
                    "metrics": best_rules["validation_metrics"],
                },
                holdout_result=holdout,
                config=best_rules["config"],
            )
            built_rows.append(row)
            leaderboard_rows.append((_score_tuple(holdout["metrics"]), row))

    if args.search_mode in {"stacker", "both"}:
        stacker_result = _fit_stacker_result(shortlist, float(args.min_coverage))
        if stacker_result is not None:
            holdout = _build_holdout_result(stacker_result)
            row = _serialize_ensemble_row(
                output_root,
                model_variation="STACKER",
                threshold_pct=int(stacker_result["config"]["abstain_threshold_pct"]),
                selection_split=SELECTION_SPLIT,
                source_provenance="SELECTIVE_STACKER_ENSEMBLE",
                member_candidates=list(stacker_result["members"]),
                validation_result={
                    "stream": stacker_result["validation_stream"],
                    "eval": stacker_result["validation_eval"],
                    "metrics": stacker_result["validation_metrics"],
                },
                holdout_result=holdout,
                config=stacker_result["config"],
            )
            built_rows.append(row)
            leaderboard_rows.append((_score_tuple(holdout["metrics"]), row))

    if not built_rows:
        raise RuntimeError("Selective ensemble search did not produce any valid rule or stacker rows.")

    champion_source = max(leaderboard_rows, key=lambda item: item[0])[1]
    champion = _champion_row(output_root, champion_source)
    built_rows.append(champion)

    merged_rows = dedupe_rows_by_name(existing_rows + built_rows)
    summary = dict(existing_summary)
    summary["models"] = merged_rows
    summary["series_order"] = [str(row.get("name") or "") for row in merged_rows]
    summary.setdefault("selection_reports", {})
    summary["selection_reports"]["selective_ensemble"] = {
        "selection_split": SELECTION_SPLIT,
        "holdout_split": HOLDOUT_SPLIT,
        "champion_name": champion["name"],
        "candidate_count": len(shortlist),
    }

    summary_path, manifest_path = write_manifest_and_summary(output_root, summary)
    combinations_path = write_combinations_markdown(output_root, summary, merged_rows)
    report_paths = _write_report(output_root, candidate_pool=shortlist, built_rows=built_rows, champion_name=champion["name"])
    return {
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combinations_path": str(combinations_path.resolve()),
        "champion_name": champion["name"],
        "candidate_count": len(shortlist),
        **report_paths,
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
