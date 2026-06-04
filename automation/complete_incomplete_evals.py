from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from automation.update_readme import update_readme
from src.models.evaluation import evaluate_saved_model
from src.utils import QuantStreamPaths, TEST_CANDLES, read_yaml_file


def load_clean_frame(paths: QuantStreamPaths, variation_id: int) -> pd.DataFrame:
    clean_path = paths.clean_data_path(variation_id)
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean dataset for var_{variation_id}: {clean_path}")
    return pd.read_parquet(clean_path)


def incomplete_reason(results: pd.DataFrame, model_id: str, evaluated_rows: int) -> str:
    pred_col = f"{model_id}_pred"
    prob_col = f"{model_id}_prob"
    if pred_col not in results.columns or prob_col not in results.columns:
        return "missing result columns"
    first_5k_probabilities = pd.to_numeric(results[prob_col].iloc[:TEST_CANDLES], errors="coerce")
    zero_probability_count = int((first_5k_probabilities.fillna(0.0) == 0.0).sum())
    if zero_probability_count:
        return f"{zero_probability_count} missing parquet probabilities encoded as 0.0"
    if evaluated_rows < TEST_CANDLES:
        return f"metadata evaluated_rows={evaluated_rows}"
    tail_probabilities = pd.to_numeric(results[prob_col].iloc[evaluated_rows:], errors="coerce")
    if evaluated_rows < len(results) and bool((tail_probabilities.fillna(0.0) == 0.0).all()):
        return f"padded tail from row {evaluated_rows}"
    return ""


def update_model_metadata(model_dir: Path, updates: dict[str, Any]) -> None:
    metadata_path = model_dir / "hyperparameters.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(updates)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def normalize_done_payload(done_path: Path, payload: dict[str, Any], model_dir: Path) -> None:
    payload["evaluated_rows"] = TEST_CANDLES
    payload["requested_test_length"] = TEST_CANDLES
    payload["resolved_test_length"] = TEST_CANDLES
    payload.setdefault("hyperparameters", {})["test_length"] = TEST_CANDLES
    done_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    update_model_metadata(
        model_dir,
        {
            "evaluated_rows": TEST_CANDLES,
            "requested_test_length": TEST_CANDLES,
            "resolved_test_length": TEST_CANDLES,
            "full_eval_completed": True,
        },
    )


def has_full_results(results: pd.DataFrame, model_id: str) -> bool:
    pred_col = f"{model_id}_pred"
    prob_col = f"{model_id}_prob"
    if pred_col not in results.columns or prob_col not in results.columns:
        return False
    frame = results.iloc[:TEST_CANDLES]
    probabilities = pd.to_numeric(frame[prob_col], errors="coerce")
    return bool((frame[pred_col].notna() & probabilities.notna() & (probabilities > 0.0)).all())


def validate_full_probabilities(model_id: str, probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=np.float32)[:TEST_CANDLES]
    nonfinite_count = int((~np.isfinite(values)).sum())
    zero_count = int((values == 0.0).sum())
    if nonfinite_count or zero_count:
        raise RuntimeError(
            f"evaluator returned invalid probabilities: nonfinite={nonfinite_count}, zero={zero_count}"
        )


def write_and_verify_results(paths: QuantStreamPaths, variation_id: int, results: pd.DataFrame, model_id: str) -> None:
    path = paths.global_results_path(variation_id)
    results.to_parquet(path, index=False)
    verified = pd.read_parquet(path, columns=[f"{model_id}_pred", f"{model_id}_prob"])
    if not has_full_results(verified, model_id):
        verified_probabilities = pd.to_numeric(verified[f"{model_id}_prob"], errors="coerce").fillna(0.0)
        zero_count = int((verified_probabilities == 0.0).sum())
        raise RuntimeError(
            f"{model_id}: parquet verification failed after write; {zero_count} zero probabilities remain."
        )


def complete_evaluations(root: Path | None = None, *, dry_run: bool = False) -> list[dict[str, Any]]:
    paths = QuantStreamPaths(root=root or QuantStreamPaths().root)
    summaries: list[dict[str, Any]] = []
    clean_cache: dict[int, pd.DataFrame] = {}
    results_cache: dict[int, pd.DataFrame] = {}
    changed_variations: set[int] = set()
    for done_path in sorted(paths.runs_done_dir.glob("*.y*ml")):
        try:
            payload = read_yaml_file(done_path)
        except Exception as exc:
            summaries.append({"file": str(done_path), "status": "skipped", "reason": str(exc)})
            continue
        model_id = str(payload.get("model_id") or "")
        model_type = str(payload.get("model_type") or "").strip().lower()
        if not model_id or model_type == "ensemble":
            continue
        if bool(payload.get("full_eval_unrecoverable", False)):
            summaries.append(
                {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": str(payload.get("full_eval_error") or "previous full-eval repair marked unrecoverable"),
                }
            )
            continue
        variation_id = int(payload.get("variation_id", 1))
        hyperparameters = dict(payload.get("hyperparameters") or {})
        evaluated_rows = int(
            payload.get("evaluated_rows")
            or payload.get("requested_test_length")
            or hyperparameters.get("test_length")
            or 0
        )
        if variation_id not in clean_cache:
            clean_cache[variation_id] = load_clean_frame(paths, variation_id)
        if variation_id not in results_cache:
            results_path = paths.global_results_path(variation_id)
            if not results_path.exists():
                summaries.append(
                    {"model_id": model_id, "status": "skipped", "reason": "missing global_results.parquet"}
                )
                continue
            results_cache[variation_id] = pd.read_parquet(results_path)
        results = results_cache[variation_id]
        model_dir = paths.variation_models_dir(variation_id) / model_id
        metadata_needs_normalization = (
            evaluated_rows != TEST_CANDLES
            or int(payload.get("requested_test_length") or 0) != TEST_CANDLES
            or int(payload.get("resolved_test_length") or 0) != TEST_CANDLES
            or int(hyperparameters.get("test_length") or 0) != TEST_CANDLES
        )
        if has_full_results(results, model_id):
            if metadata_needs_normalization:
                summary = {
                    "model_id": model_id,
                    "variation_id": variation_id,
                    "status": "dry_run" if dry_run else "normalized",
                    "reason": "full 5k results already present; normalized run metadata",
                }
                if not dry_run:
                    normalize_done_payload(done_path, payload, model_dir)
                summaries.append(summary)
            continue
        reason = incomplete_reason(results, model_id, evaluated_rows)
        if not reason:
            continue
        summary = {
            "model_id": model_id,
            "variation_id": variation_id,
            "status": "dry_run" if dry_run else "completed",
            "reason": reason,
        }
        if not dry_run:
            full_hyperparameters = {**hyperparameters, "test_length": TEST_CANDLES}
            try:
                predictions, probabilities = evaluate_saved_model(
                    model_type=model_type,
                    train_mode=str(payload.get("train_mode") or ""),
                    clean_frame=clean_cache[variation_id],
                    hyperparameters=full_hyperparameters,
                    model_dir=model_dir,
                    test_length=TEST_CANDLES,
                )
                if len(predictions) != TEST_CANDLES or len(probabilities) != TEST_CANDLES:
                    raise RuntimeError(
                        f"expected {TEST_CANDLES} predictions/probabilities, "
                        f"got {len(predictions)}/{len(probabilities)}"
                    )
                validate_full_probabilities(model_id, probabilities)
            except Exception as exc:
                payload["full_eval_unrecoverable"] = True
                payload["full_eval_error"] = str(exc)
                done_path.write_text(
                    yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
                    encoding="utf-8",
                )
                summary["status"] = "skipped"
                summary["reason"] = f"unrecoverable full-eval repair failure: {exc}"
                summaries.append(summary)
                continue
            results[f"{model_id}_pred"] = np.asarray(predictions, dtype=np.int8)
            results[f"{model_id}_prob"] = np.asarray(probabilities, dtype=np.float32)
            changed_variations.add(variation_id)
            write_and_verify_results(paths, variation_id, results, model_id)
            normalize_done_payload(done_path, payload, model_dir)
        summaries.append(summary)
    if not dry_run:
        for variation_id in sorted(changed_variations):
            results_cache[variation_id].to_parquet(paths.global_results_path(variation_id), index=False)
        update_readme(paths.root)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete short Quant-Stream model evaluations to the full 5k test rows."
    )
    parser.add_argument("--root", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else None
    summaries = complete_evaluations(root=root, dry_run=bool(args.dry_run))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
