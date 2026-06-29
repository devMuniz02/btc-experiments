from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.public.config.schema import validate_config
from src.public.config.loader import stable_hash
from src.public.evaluation.ranking import deterministic_discovery_rank, validation_quality_key
from src.workflows.hf_state import (
    HfSettings,
    HfStateClient,
    LIVE_DATA_ROOT,
    REQUEST_ROOT,
    anonymized_dataset_id,
    dump_yaml,
    explicit_git_push,
    load_yaml,
    market_id_from_config,
    phase_folder,
    phase_value,
    pull_hf_folder,
    restore_artifact_branch_paths,
    workflow_lock,
)

LEGACY_FINAL_PHASE_INDEX = 2
EXHAUSTIVE_FINAL_PHASE_INDEX = 16
EXHAUSTIVE_PROFILE = "exhaustive_v1"
EXPERIMENT_STATE_ROOT = "experiment_state"


def _require_private_training() -> Any:
    try:
        from src.private.training import discovery_catalog, discovery_state, pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Private training source is required for this workflow path. "
            "Run scripts/pull_private_src.py with HF_TOKEN/HF_REPO_ID before executing private experiments."
        ) from exc
    return discovery_catalog, discovery_state, pipeline


def _catalog_hash() -> str:
    discovery_catalog, _, _ = _require_private_training()
    return str(discovery_catalog.catalog_hash())


def _phase_spec(phase: str) -> Any:
    discovery_catalog, _, _ = _require_private_training()
    return discovery_catalog.phase_spec(phase)


def _stable_id(prefix: str, payload: Any) -> str:
    _, discovery_state, _ = _require_private_training()
    return str(discovery_state.stable_id(prefix, payload))


def _rank_key_from_metrics(
    metrics: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None = None,
    ranking_metric: str = "direction_accuracy",
) -> tuple[Any, ...]:
    return validation_quality_key(
        {"validation": metrics or {}, "overfit_diagnostics": diagnostics or {}},
        ranking_metric,
    )


def _rank_key_from_result(result: dict[str, Any], ranking_metric: str = "direction_accuracy") -> tuple[Any, ...]:
    return _rank_key_from_metrics(
        result.get("validation") or {}, result.get("overfit_diagnostics") or {}, ranking_metric
    )


def _rank_key_from_recipe(recipe: dict[str, Any], ranking_metric: str = "direction_accuracy") -> tuple[Any, ...]:
    metrics = recipe.get("selection_metrics") or {}
    return _rank_key_from_metrics(
        metrics.get("validation") or {}, metrics.get("overfit_diagnostics") or {}, ranking_metric
    )


def _recipe_from_result(result: dict[str, Any]) -> dict[str, Any]:
    recipe = dict(result.get("recipe") or {"recipe_hash": result.get("recipe_hash")})
    recipe["selection_metrics"] = {
        "train": result.get("train"),
        "validation": result.get("validation"),
        "test": result.get("test"),
        "overfit_diagnostics": result.get("overfit_diagnostics") or {},
    }
    return recipe


def _adopt_if_improved(
    result: dict[str, Any],
    baseline: dict[str, Any] | None,
    ranking_metric: str = "direction_accuracy",
) -> tuple[dict[str, Any], bool]:
    candidate = _recipe_from_result(result)
    if not baseline or not baseline.get("selection_metrics"):
        return candidate, True
    return (
        (candidate, True)
        if _rank_key_from_result(result, ranking_metric) > _rank_key_from_recipe(baseline, ranking_metric)
        else (dict(baseline), False)
    )


def _inherit_model_family_result(result: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    candidate = _recipe_from_result(result)
    if not baseline:
        return candidate
    carried = copy.deepcopy(baseline)
    candidate_decisions = dict(candidate.get("decisions") or {})
    carried_decisions = dict(carried.get("decisions") or {})
    candidate_model_family = dict(candidate_decisions.get("model_family") or {})
    carried_decisions["model_family"] = {
        "variation_id": candidate_model_family.get("variation_id") or result.get("variation_id") or result.get("candidate_id"),
        "parameters": {},
    }
    for key in (
        "phase_id",
        "axis",
        "variation_id",
        "candidate_id",
        "selection_metrics",
    ):
        if key in candidate:
            carried[key] = copy.deepcopy(candidate[key])
    carried["parent_recipe_hash"] = baseline.get("recipe_hash") or result.get("parent_recipe_hash") or ""
    carried["decisions"] = carried_decisions
    carried["recipe_hash"] = _stable_id("recipe", {key: value for key, value in carried.items() if key != "recipe_hash"})
    return carried


def _discovery_state_class() -> Any:
    _, discovery_state, _ = _require_private_training()
    return discovery_state.DiscoveryState


def _run_pipeline(options: Any) -> dict[str, Any]:
    _, _, pipeline = _require_private_training()
    return dict(pipeline.run_pipeline(options))


def _pipeline_options(**kwargs: Any) -> Any:
    _, _, pipeline = _require_private_training()
    return pipeline.PipelineOptions(**kwargs)


def _workflow_profile(config: dict[str, Any]) -> str:
    return str(config.get("experiments", {}).get("workflow_profile") or "legacy_v1")


def _final_phase_index(config: dict[str, Any]) -> int:
    return EXHAUSTIVE_FINAL_PHASE_INDEX if _workflow_profile(config) == EXHAUSTIVE_PROFILE else LEGACY_FINAL_PHASE_INDEX


def _phase_name(index: int, config: dict[str, Any]) -> str:
    return f"phase{index:02d}" if _workflow_profile(config) == EXHAUSTIVE_PROFILE else f"phase{index}"


def _experiment_config_hash(config: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(config))
    for key in ("current_phase", "discovery_state", "workflow", "data_variation"):
        payload.pop(key, None)
    return stable_hash(payload)


def _config_files(cache_root: Path) -> list[Path]:
    configs_dir = cache_root / REQUEST_ROOT / "configs"
    candidates = list(configs_dir.glob("*.yaml"))
    root_config = cache_root / REQUEST_ROOT / "config.yaml"
    if root_config.exists() and not candidates:
        candidates.append(root_config)
    return sorted(candidates, key=lambda path: path.name.lower())


def _first_incomplete_config(cache_root: Path) -> Path | None:
    for path in _config_files(cache_root):
        config = load_yaml(path)
        validate_config(config)
        if phase_value(config) != "done":
            return path
    return None


def _request_files(cache_root: Path, phase: str, market_id: str) -> list[Path]:
    folder = cache_root / REQUEST_ROOT / phase_folder(phase)
    return sorted(path for path in folder.glob("*.yaml") if market_id in path.name)


def _phase_is_complete(cache_root: Path, phase: str, market_id: str, *, exhaustive: bool = False) -> bool:
    if _request_files(cache_root, phase, market_id):
        return False
    if exhaustive:
        return bool(_load_phase_results(cache_root, market_id, phase))
    return True


def _phase_index(phase: str) -> int:
    text = phase_folder(phase).replace("phase", "", 1)
    return int(text)


def _phase_row_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    phase_id = str(row.get("phase_id") or "")
    digits = "".join(char for char in phase_id if char.isdigit())
    return (
        int(digits) if digits else 9999,
        phase_id,
        str(row.get("experiment_id") or ""),
    )


def _phase_remote_path(cache_root: Path, config_path: Path) -> str:
    return config_path.relative_to(cache_root).as_posix()


def _live_cache_files(cache_root: Path, market_id: str) -> list[tuple[Path, str]]:
    live_dir = cache_root / LIVE_DATA_ROOT / market_id
    if not live_dir.exists():
        return []
    return sorted(
        (path, f"{LIVE_DATA_ROOT}/{market_id}/{path.relative_to(live_dir).as_posix()}")
        for path in live_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".parquet", ".json"}
    )


def _push_live_cache(client: HfStateClient, *, cache_root: Path, market_id: str) -> dict[str, Any]:
    files = _live_cache_files(cache_root, market_id)
    if not files:
        return {"status": "skipped", "reason": "no live cache files", "paths": []}
    client.commit_files(files, commit_message=f"Update live data cache for {market_id}")
    return {"status": "pushed", "paths": [remote for _, remote in files]}


def _commit_hf_changes(
    client: Any,
    *,
    add_files: list[tuple[Path, str]],
    delete_paths: list[str],
    commit_message: str,
) -> dict[str, Any]:
    if not add_files and not delete_paths:
        return {"status": "skipped", "reason": "no hf changes", "paths": [], "deleted": []}
    if hasattr(client, "commit_changes"):
        client.commit_changes(add_files=add_files, delete_paths=delete_paths, commit_message=commit_message)
    else:
        for local_path, hf_path in add_files:
            client.upload_file(local_path, hf_path, commit_message=commit_message)
        for hf_path in delete_paths:
            client.delete_file(hf_path, commit_message=commit_message)
    return {"status": "pushed", "paths": [remote for _, remote in add_files], "deleted": list(delete_paths)}


def _result_remote_path(market_id: str, phase: str, request_id: str) -> str:
    return f"{EXPERIMENT_STATE_ROOT}/{market_id}/phases/{phase}/results/{request_id}.json"


def _state_local_path(cache_root: Path, market_id: str) -> Path:
    return cache_root / EXPERIMENT_STATE_ROOT / market_id / "workflow_state.json"


def _record_request_failure(
    client: HfStateClient,
    *,
    cache_root: Path,
    market_id: str,
    request_id: str,
    error: Exception,
) -> int:
    path = _state_local_path(cache_root, market_id)
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    failures = dict(payload.get("failure_counts") or {})
    signature = _stable_id("failure", {"request_id": request_id, "type": type(error).__name__, "message": str(error)})
    failures[signature] = int(failures.get(signature, 0)) + 1
    payload.update(
        {
            "market_id": market_id,
            "status": "blocked" if failures[signature] >= 3 else "retrying",
            "failure_counts": failures,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    client.upload_file(
        path,
        f"{EXPERIMENT_STATE_ROOT}/{market_id}/workflow_state.json",
        commit_message=f"Record failure for {request_id}",
    )
    return failures[signature]


def _load_phase_results(cache_root: Path, market_id: str, phase: str) -> list[dict[str, Any]]:
    result_dir = cache_root / EXPERIMENT_STATE_ROOT / market_id / "phases" / phase / "results"
    results: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            results.append(payload)
    return results


def _use_worked_candidate_result(result: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        candidate
        for candidate in result.get("candidate_results") or []
        if isinstance(candidate, dict) and candidate.get("status") == "worked"
    ]
    if not candidates:
        return result
    selected_candidate_id = str(result.get("candidate_id") or "")
    promoted = dict(
        next(
            (
                candidate
                for candidate in candidates
                if selected_candidate_id and str(candidate.get("candidate_id") or "") == selected_candidate_id
            ),
            candidates[0],
        )
    )
    promoted["request_id"] = result.get("request_id")
    promoted["workflow_id"] = result.get("workflow_id")
    promoted["parent_recipe_hash"] = result.get("parent_recipe_hash") or promoted.get("parent_recipe_hash")
    promoted["selected_for_next_phase"] = result.get("selected_for_next_phase", False)
    return promoted


def _collect_recipe_hyperparameters(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    parameters: dict[str, dict[str, Any]] = {}
    for record in records:
        recipe = record.get("recipe") or {}
        recipes = [recipe, *(recipe.get("family_recipes") or [])]
        for item in recipes:
            candidate_id = str(item.get("candidate_id") or "")
            hyperparameters = item.get("resolved_hyperparameters")
            if candidate_id and isinstance(hyperparameters, dict):
                parameters[candidate_id] = copy.deepcopy(hyperparameters)
    return parameters


def _run_final_post_lock_sliding_validation(
    *,
    root: Path,
    client: HfStateClient,
    config: dict[str, Any],
    market_id: str,
    selected: list[dict[str, Any]],
    ranking_metric: str,
    upload_private: bool = True,
) -> dict[str, Any] | None:
    if not selected:
        return None
    _, _, pipeline = _require_private_training()
    recipe = selected[0].get("recipe") or {}
    contract = recipe.get("resolved_data_contract") or {}
    feature_columns = list(contract.get("feature_columns") or recipe.get("feature_columns") or [])
    if not feature_columns:
        return None
    hyperparameters_by_model = _collect_recipe_hyperparameters(selected)
    request_hyperparameters = copy.deepcopy(
        recipe.get("resolved_hyperparameters") or next(iter(hyperparameters_by_model.values()), {})
    )
    recipe_decisions = copy.deepcopy(recipe.get("decisions") or {})
    data_variation = str((recipe_decisions.get("denoising") or {}).get("variation_id") or "none")
    validation = pipeline._evaluate_post_lock_sliding_windows(
        config=config,
        root=root,
        market_id=market_id,
        decisions=recipe_decisions,
        exhaustive=True,
        data_variation=data_variation,
        end_utc=str(config["experiments"]["test_end_utc"]),
        top_candidate_ids=[str(record.get("candidate_id") or "") for record in selected],
        feature_columns=feature_columns,
        scaling_method=str(contract.get("scaling_transform") or recipe.get("scaling_transform") or "standard"),
        ranking_metric=ranking_metric,
        sequence_length=int(contract.get("sequence_length") or recipe.get("sequence_length") or 1),
        hyperparameters_by_model=hyperparameters_by_model,
        request_hyperparameters=request_hyperparameters,
        candidate_records=selected,
        top_candidate_records=selected,
    )
    private_path = root / "privateexperiments" / market_id / "results" / "post_lock_sliding_validation.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    if upload_private:
        client.upload_file(
            private_path,
            f"privateexperiments/{market_id}/results/post_lock_sliding_validation.json",
            commit_message=f"Record {market_id} phase16 Top K post-lock validation",
        )
    summaries_by_public_id: dict[str, dict[str, Any]] = {}
    incomplete_public_ids: list[str] = []
    for model_public_id, model in (validation.get("models") or {}).items():
        record_public_id = str(model.get("record_public_id") or "")
        windows_completed = int(model.get("windows_completed") or 0)
        metrics_summary = model.get("window_metrics_summary") or {}
        if not record_public_id:
            continue
        if windows_completed <= 0 or not metrics_summary:
            incomplete_public_ids.append(record_public_id)
            continue
        summaries_by_public_id[record_public_id] = {
            "model_public_id": str(model_public_id),
            "windows_completed": model.get("windows_completed"),
            "window_metrics_summary": metrics_summary,
            "long_run_metrics": model.get("long_run_metrics"),
        }
    selected_public_ids = [str(record.get("public_id") or "") for record in selected]
    missing_public_ids = [
        public_id for public_id in selected_public_ids if public_id not in summaries_by_public_id
    ]
    if missing_public_ids:
        missing_text = ", ".join(missing_public_ids)
        incomplete_text = ", ".join(incomplete_public_ids)
        detail = f"; incomplete: {incomplete_text}" if incomplete_text else ""
        raise RuntimeError(f"Post-lock sliding validation did not complete for Top K models: {missing_text}{detail}")
    phase_path = root / "experiments" / market_id / "results" / "phase_results_public.json"
    if phase_path.exists():
        rows = json.loads(phase_path.read_text(encoding="utf-8"))
        selected_ids = set(selected_public_ids)
        for row in rows:
            if row.get("phase_id") != "phase16":
                continue
            row.pop("post_lock_sliding_summary", None)
            experiment_id = str(row.get("experiment_id") or "")
            if experiment_id in selected_ids and experiment_id in summaries_by_public_id:
                row["post_lock_sliding_summary"] = summaries_by_public_id[experiment_id]
        phase_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return validation


def _finalize_exhaustive_phase(
    client: HfStateClient,
    *,
    root: Path,
    cache_root: Path,
    config_path: Path,
    config: dict[str, Any],
    phase: str,
    market_id: str,
) -> dict[str, Any]:
    spec = _phase_spec(phase)
    results = _load_phase_results(cache_root, market_id, phase)
    expected_ids = list(config.get("discovery_state", {}).get("expected_requests") or [])
    expected = len(expected_ids) or len(spec.variations)
    if len(results) != expected:
        return {"status": "blocked", "reason": f"phase result count {len(results)} does not match expected {expected}"}
    infrastructure_failures = [
        result
        for result in results
        if result.get("status") == "failed"
        and result.get("failure_class") in {"dependency", "external_data", "infrastructure"}
    ]
    if infrastructure_failures:
        return {"status": "blocked", "reason": "required variation failures", "failures": len(infrastructure_failures)}
    results = [_use_worked_candidate_result(result) for result in results]
    minimum_trades = int(config.get("experiments", {}).get("min_trade_count", 1))
    for result in results:
        validation = result.get("validation") or {}
        if float(validation.get("trade_count", 0.0)) < minimum_trades:
            result["decision_eligible"] = False
    ranking_metric = str(config.get("experiments", {}).get("ranking_metric") or "direction_accuracy")
    ranked = deterministic_discovery_rank(results, ranking_metric)
    if not ranked:
        return {"status": "blocked", "reason": "phase has no eligible worked result"}
    prior_recipes_for_baseline = list(config.get("discovery_state", {}).get("selected_recipes") or [])
    adopted_selected_ids: set[str] = set()
    if spec.axis == "robustness":
        selected = []
        recipes_for_state = []
        baseline_by_hash = {
            str(recipe.get("recipe_hash") or ""): recipe for recipe in prior_recipes_for_baseline
        }
        grouped: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            parent_hash = str(result.get("parent_recipe_hash") or "")
            if not parent_hash and len(baseline_by_hash) == 1:
                parent_hash = next(iter(baseline_by_hash))
            grouped.setdefault(parent_hash, []).append(result)
        for parent_hash, baseline in baseline_by_hash.items():
            gates = grouped.get(parent_hash, [])
            if len(gates) != len(spec.variations) or any(gate.get("status") != "worked" for gate in gates):
                continue
            carried = copy.deepcopy(baseline)
            carried_decisions = dict(carried.get("decisions") or {})
            carried_decisions["robustness"] = {
                "variation_id": "all_mandatory_stresses_passed",
                "parameters": {"passed": [str(gate.get("variation_id")) for gate in gates]},
            }
            carried.update(
                {
                    "phase_id": spec.phase_id,
                    "axis": spec.axis,
                    "variation_id": "all_mandatory_stresses_passed",
                    "parent_recipe_hash": parent_hash,
                    "decisions": carried_decisions,
                }
            )
            carried["recipe_hash"] = _stable_id(
                "recipe", {key: value for key, value in carried.items() if key != "recipe_hash"}
            )
            recipes_for_state.append(carried)
            selected.extend(gates)
            adopted_selected_ids.update(str(gate["experiment_id"]) for gate in gates)
        if not recipes_for_state:
            return {"status": "blocked", "reason": "no finalist passed every mandatory robustness gate"}
    elif spec.axis == "model_family":
        sequence_models = {
            "temporal_cnn", "tcn", "lstm", "gru", "transformer", "patchtst", "mamba",
            "state_space", "cnn_lstm", "transformer_lstm",
        }
        neural_tabular = {"mlp", "residual_mlp"}
        grouped_models = {"classical": [], "neural_tabular": [], "sequence": []}
        for result in ranked:
            candidate_id = str(result.get("candidate_id"))
            category = "sequence" if candidate_id in sequence_models else ("neural_tabular" if candidate_id in neural_tabular else "classical")
            grouped_models[category].append(result)
        if any(not values for values in grouped_models.values()):
            return {"status": "blocked", "reason": "model-family phase requires a worked classical, neural-tabular, and sequence candidate"}
        selected = [values[0] for values in grouped_models.values()]
        adopted_selected_ids = {str(result["experiment_id"]) for result in selected}
    elif spec.carry == "top_3":
        selected = ranked[:3]
        adopted_selected_ids = {str(result["experiment_id"]) for result in selected}
    elif spec.carry == "top_1_per_family":
        selected = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for result in ranked:
            grouped.setdefault(str(result.get("parent_recipe_hash") or result.get("candidate_id")), []).append(result)
        for family_results in grouped.values():
            selected.append(family_results[0])
        baseline_by_hash = {str(recipe.get("recipe_hash") or ""): recipe for recipe in prior_recipes_for_baseline}
        recipes_for_state = []
        for result in selected:
            baseline = baseline_by_hash.get(str(result.get("parent_recipe_hash") or ""))
            recipe, adopted = _adopt_if_improved(result, baseline, ranking_metric)
            recipes_for_state.append(recipe)
            if adopted:
                adopted_selected_ids.add(str(result["experiment_id"]))
    else:
        selected = ranked[:1]
        baseline = prior_recipes_for_baseline[0] if prior_recipes_for_baseline else None
        recipe, adopted = _adopt_if_improved(selected[0], baseline, ranking_metric)
        recipes_for_state = [recipe]
        if adopted:
            adopted_selected_ids.add(str(selected[0]["experiment_id"]))
    if spec.axis == "model_family" or spec.carry == "top_3":
        recipes_for_state = [_recipe_from_result(result) for result in selected]
    if 8 <= spec.index <= 13:
        prior_recipes = list(config.get("discovery_state", {}).get("selected_recipes") or [])
        winning_recipe = dict(selected[0].get("recipe") or {})
        winning_decision = dict((winning_recipe.get("decisions") or {}).get(spec.axis) or {})
        baseline = prior_recipes[0] if prior_recipes else None
        if baseline and not (
            _rank_key_from_result(selected[0], ranking_metric) > _rank_key_from_recipe(baseline, ranking_metric)
        ):
            recipes_for_state = [dict(recipe) for recipe in prior_recipes]
            adopted_selected_ids = set()
        else:
            adopted_selected_ids = {str(selected[0]["experiment_id"])}
        synchronized: list[dict[str, Any]] = []
        if adopted_selected_ids:
            for prior in prior_recipes:
                carried = copy.deepcopy(prior)
                carried_decisions = dict(carried.get("decisions") or {})
                carried_decisions[spec.axis] = copy.deepcopy(winning_decision)
                carried.update(
                    {
                        "phase_id": spec.phase_id,
                        "axis": spec.axis,
                        "variation_id": winning_recipe.get("variation_id"),
                        "parent_recipe_hash": prior.get("recipe_hash", ""),
                        "decisions": carried_decisions,
                        "selection_metrics": {
                            "train": selected[0].get("train"),
                            "validation": selected[0].get("validation"),
                            "test": selected[0].get("test"),
                            "overfit_diagnostics": selected[0].get("overfit_diagnostics") or {},
                        },
                    }
                )
                carried["recipe_hash"] = _stable_id("recipe", {key: value for key, value in carried.items() if key != "recipe_hash"})
                synchronized.append(carried)
            if synchronized:
                recipes_for_state = synchronized
    fixed_inheritance = str(config.get("experiments", {}).get("phase5_to_phase6_inheritance") or "unfixed") == "fixed"
    if spec.axis == "model_family" and fixed_inheritance:
        baseline_by_hash = {str(recipe.get("recipe_hash") or ""): recipe for recipe in prior_recipes_for_baseline}
        recipes_for_state = []
        for result in selected:
            parent_hash = str(result.get("parent_recipe_hash") or "")
            baseline = baseline_by_hash.get(parent_hash)
            if baseline is None and len(prior_recipes_for_baseline) == 1:
                baseline = prior_recipes_for_baseline[0]
            recipes_for_state.append(_inherit_model_family_result(result, baseline))
    if spec.axis == "final_validation_lock":
        selected = ranked[: int(config.get("production", {}).get("top_k", 3))]
        adopted_selected_ids = {str(result["experiment_id"]) for result in selected}
        recipes_for_state = [_recipe_from_result(result) for result in selected]
        _run_final_post_lock_sliding_validation(
            root=root,
            client=client,
            config=config,
            market_id=market_id,
            selected=selected,
            ranking_metric=ranking_metric,
        )
    updated_result_files: list[tuple[Path, str]] = []
    for result in results:
        result["selected_for_next_phase"] = str(result.get("experiment_id")) in adopted_selected_ids
        local = (
            cache_root
            / EXPERIMENT_STATE_ROOT
            / market_id
            / "phases"
            / phase
            / "results"
            / f"{result['request_id']}.json"
        )
        local.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        updated_result_files.append((local, _result_remote_path(market_id, phase, str(result["request_id"]))))
    client.commit_files(updated_result_files, commit_message=f"Select {market_id} {phase} winners")
    config.setdefault("discovery_state", {})
    config["discovery_state"].update(
        {
            "catalog_hash": _catalog_hash(),
            "completed_phase": phase,
            "selected_recipes": recipes_for_state,
        }
    )
    state_path = _state_local_path(cache_root, market_id)
    existing_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    discovery_phases = dict(existing_state.get("discovery_phases") or {})
    discovery_phases[phase] = "done"
    state = _discovery_state_class()(
        workflow_profile=EXHAUSTIVE_PROFILE,
        workflow_id=str(selected[0].get("workflow_id") or ""),
        market_id=market_id,
        catalog_hash=_catalog_hash(),
        config_hash=str(selected[0].get("config_hash") or ""),
        current_phase=phase,
        expected_requests=[str(result["request_id"]) for result in results],
        completed_requests=[str(result["request_id"]) for result in results],
        selected_recipes=list(config["discovery_state"]["selected_recipes"]),
        discovery_phases=discovery_phases,
        failure_counts=dict(existing_state.get("failure_counts") or {}),
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    client.upload_file(
        state_path,
        f"{EXPERIMENT_STATE_ROOT}/{market_id}/workflow_state.json",
        commit_message=f"Finalize {market_id} {phase}",
    )
    config_path.write_text(dump_yaml(config), encoding="utf-8")
    return {"status": "completed", "selected": selected}


def _mark_public_phase_winners(root: Path, market_id: str, selected: list[dict[str, Any]]) -> None:
    path = root / "experiments" / market_id / "results" / "phase_results_public.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected_public_ids = {str(result.get("public_id")) for result in selected}
    for row in rows:
        if row.get("phase_id") == selected[0].get("phase_id"):
            row["selected_for_next_phase"] = str(row.get("experiment_id")) in selected_public_ids
    rows = sorted(rows, key=_phase_row_sort_key)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def _post_lock_selected_records(
    phase16_results: list[dict[str, Any]],
    *,
    ranking_metric: str,
    top_k: int,
) -> list[dict[str, Any]]:
    selected = [record for record in phase16_results if record.get("selected_for_next_phase")]
    if len(selected) >= top_k:
        return selected[:top_k]
    return deterministic_discovery_rank(phase16_results, ranking_metric)[:top_k]


def _write_request(
    path: Path,
    config: dict[str, Any],
    *,
    phase: str,
    variation: str,
    workflow_payload: dict[str, Any] | None = None,
) -> None:
    payload = json.loads(json.dumps(config))
    market_id = market_id_from_config(payload)
    payload["current_phase"] = phase.replace("phase", "")
    workflow_payload = workflow_payload or {}
    is_exhaustive = workflow_payload.get("workflow_profile") == EXHAUSTIVE_PROFILE
    if not is_exhaustive or workflow_payload.get("axis") == "denoising":
        payload.setdefault("data_variation", {})
        payload["data_variation"].update(
            {
                "anonymous_dataset_id": anonymized_dataset_id(market_id, phase, variation),
                "variation": variation,
                "single_dataset_variation": variation,
            }
        )
    payload.setdefault("workflow", {})
    payload["workflow"].update({"phase": phase, "request_id": path.stem, **workflow_payload})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(payload), encoding="utf-8")


def _generate_phase_requests(cache_root: Path, config_path: Path, config: dict[str, Any], phase: str) -> list[Path]:
    market_id = market_id_from_config(config)
    exhaustive = _workflow_profile(config) == EXHAUSTIVE_PROFILE
    spec = _phase_spec(phase) if exhaustive else None
    variations = (
        [variation.variation_id for variation in spec.variations]
        if spec
        else list(config.get("experiments", {}).get("denoising") or ["none"])
    )
    if exhaustive and spec and spec.index <= 14 and "default" not in variations:
        variations = ["default", *variations]
    frozen_catalog_hash = _catalog_hash() if exhaustive else ""
    previous_catalog_hash = str(config.get("discovery_state", {}).get("catalog_hash") or "")
    if exhaustive and previous_catalog_hash and previous_catalog_hash != frozen_catalog_hash:
        raise RuntimeError("The exhaustive catalog changed after the workflow started; reset with a new workflow id.")
    config_hash = _experiment_config_hash(config)
    parent_recipes = list(config.get("discovery_state", {}).get("selected_recipes") or [])
    request_parents = (
        parent_recipes if exhaustive and spec and spec.index > 1 and parent_recipes else [None]
    )
    generated: list[Path] = []
    for parent in request_parents:
        parent_recipe_hash = ""
        if exhaustive:
            parent_recipe_hash = str(
                (parent or {}).get("recipe_hash")
                or _stable_id("recipe", parent or {"market_id": market_id, "base": True})
            )
        for variation in variations:
            index = len(generated) + 1
            request_name = f"{market_id}_{phase}_request_{index:03d}.yaml"
            request_path = cache_root / REQUEST_ROOT / phase / request_name
            workflow_payload = {}
            if exhaustive and spec is not None:
                workflow_payload = {
                    "workflow_profile": EXHAUSTIVE_PROFILE,
                    "workflow_id": _stable_id(
                        "workflow",
                        {"market_id": market_id, "config_hash": config_hash, "catalog_hash": frozen_catalog_hash},
                    ),
                    "phase_id": spec.phase_id,
                    "axis": spec.axis,
                    "variation_id": str(variation),
                    "parent_recipe_hash": parent_recipe_hash,
                    "parent_recipe": parent or {},
                    "parent_recipes": parent_recipes,
                    "catalog_hash": frozen_catalog_hash,
                    "config_hash": config_hash,
                    "seed": int(config.get("project", {}).get("default_seed", 42)),
                }
            _write_request(
                request_path, config, phase=phase, variation=str(variation), workflow_payload=workflow_payload
            )
            generated.append(request_path)
    if exhaustive:
        config.setdefault("discovery_state", {})
        config["discovery_state"].update(
            {"catalog_hash": frozen_catalog_hash, "expected_requests": [path.stem for path in generated]}
        )
    config["current_phase"] = phase.replace("phase", "")
    config_path.write_text(dump_yaml(config), encoding="utf-8")
    return generated


def _push_phase_state(
    client: HfStateClient,
    *,
    cache_root: Path,
    config_path: Path,
    generated: list[Path],
    phase: str,
    market_id: str,
) -> None:
    remote_config = _phase_remote_path(cache_root, config_path)
    files: list[tuple[Path, str]] = [(config_path, remote_config)]
    files.extend((path, f"{REQUEST_ROOT}/{phase_folder(phase)}/{path.name}") for path in generated)
    client.commit_files(files, commit_message=f"Advance {market_id} to {phase_folder(phase)}")


def _advance_completed_phase(
    client: HfStateClient,
    *,
    cache_root: Path,
    config_path: Path,
    config: dict[str, Any],
    current: str,
    market_id: str,
) -> tuple[str, list[Path], bool]:
    next_index = _phase_index(current) + 1
    if next_index > _final_phase_index(config):
        config["current_phase"] = "done"
        config_path.write_text(dump_yaml(config), encoding="utf-8")
        state_path = _state_local_path(cache_root, market_id)
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["current_phase"] = "done"
            state["status"] = "completed"
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
            client.upload_file(
                state_path,
                f"{EXPERIMENT_STATE_ROOT}/{market_id}/workflow_state.json",
                commit_message=f"Mark {market_id} experiments done",
            )
        client.upload_file(
            config_path,
            _phase_remote_path(cache_root, config_path),
            commit_message=f"Mark {market_id} experiments done",
        )
        client.upload_file(
            config_path, f"markets/{market_id}.yaml", commit_message=f"Promote {market_id} production config"
        )
        return "done", [], True

    next_phase = _phase_name(next_index, config)
    generated = _generate_phase_requests(cache_root, config_path, config, next_phase)
    _push_phase_state(
        client,
        cache_root=cache_root,
        config_path=config_path,
        generated=generated,
        phase=next_phase,
        market_id=market_id,
    )
    return next_phase, generated, False


def run_experiment_workflow(
    *,
    root: Path,
    cache_root: Path,
    repo_id: str,
    token: str,
    repo_type: str,
    push_github: bool,
    github_branch: str | None,
    push_hf: bool,
    max_requests: int,
) -> dict[str, Any]:
    cache_root = cache_root if cache_root.is_absolute() else root / cache_root
    with workflow_lock(root, "experiments"):
        client = HfStateClient(HfSettings.from_env_or_args(repo_id=repo_id, token=token, repo_type=repo_type))
        if cache_root.exists():
            shutil.rmtree(cache_root)
        pulled_requests = pull_hf_folder(client, prefix=REQUEST_ROOT, local_dir=cache_root)
        pulled_live_cache = pull_hf_folder(client, prefix=LIVE_DATA_ROOT, local_dir=cache_root)
        pulled_experiment_state = pull_hf_folder(client, prefix=EXPERIMENT_STATE_ROOT, local_dir=cache_root)
        selected = _first_incomplete_config(cache_root)
        if selected is None:
            return {
                "status": "skipped",
                "reason": "all experiment configs are done",
                "pulled": len(pulled_requests),
                "pulled_live_cache": len(pulled_live_cache),
                "pulled_experiment_state": len(pulled_experiment_state),
            }

        config = load_yaml(selected)
        validate_config(config)
        market_id = market_id_from_config(config)
        if not market_id:
            raise RuntimeError(f"Selected config has no market.market_id: {selected}")
        restored_artifacts = restore_artifact_branch_paths(
            root,
            paths=[f"experiments/{market_id}"],
            target_branch=github_branch if push_github else None,
        )
        current = phase_value(config)
        results: list[dict[str, Any]] = []

        if current == "none":
            current = _phase_name(1 if _workflow_profile(config) == EXHAUSTIVE_PROFILE else 0, config)
            generated = _generate_phase_requests(cache_root, selected, config, current)
            _push_phase_state(
                client,
                cache_root=cache_root,
                config_path=selected,
                generated=generated,
                phase=current,
                market_id=market_id,
            )
        else:
            current = phase_folder(current)

        if _phase_is_complete(
            cache_root,
            current,
            market_id,
            exhaustive=_workflow_profile(config) == EXHAUSTIVE_PROFILE,
        ):
            if _workflow_profile(config) == EXHAUSTIVE_PROFILE:
                finalized = _finalize_exhaustive_phase(
                    client,
                    root=root,
                    cache_root=cache_root,
                    config_path=selected,
                    config=config,
                    phase=current,
                    market_id=market_id,
                )
                if finalized["status"] != "completed":
                    return {
                        "status": "blocked",
                        "market_id": market_id,
                        "current_phase": current,
                        "reason": finalized["reason"],
                        "remaining_requests": [],
                        "results": results,
                    }
            completed_phase = current
            current, _, done = _advance_completed_phase(
                client,
                cache_root=cache_root,
                config_path=selected,
                config=config,
                current=current,
                market_id=market_id,
            )
            if done:
                return {
                    "status": "completed",
                    "market_id": market_id,
                    "current_phase": "done",
                    "completed_phase": completed_phase,
                    "pulled": len(pulled_requests),
                    "pulled_live_cache": len(pulled_live_cache),
                    "remaining_requests": [],
                    "results": results,
                    "restored_artifacts": restored_artifacts,
                }

        batch_results: list[dict[str, Any]] = []
        batch_hf_add_files: list[tuple[Path, str]] = []
        batch_hf_delete_paths: list[str] = []
        batch_request_paths: list[Path] = []
        for request_path in _request_files(cache_root, current, market_id)[:max_requests]:
            local_request = root / "config" / f"_active_{request_path.name}"
            local_request.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(request_path, local_request)
            try:
                manifest = _run_pipeline(
                    _pipeline_options(
                        request=local_request,
                        root=root,
                        market=market_id,
                        phase=current,
                        push_hf=push_hf,
                        push_public=push_github,
                        skip_post_lock_sliding=True,
                    )
                )
            except Exception as exc:
                if _workflow_profile(config) != EXHAUSTIVE_PROFILE:
                    raise
                failure_count = _record_request_failure(
                    client,
                    cache_root=cache_root,
                    market_id=market_id,
                    request_id=request_path.stem,
                    error=exc,
                )
                if failure_count < 3:
                    raise
                return {
                    "status": "blocked",
                    "market_id": market_id,
                    "current_phase": current,
                    "reason": "same infrastructure failure reached retry limit",
                    "request": request_path.name,
                    "failure_count": failure_count,
                    "remaining_requests": [path.name for path in _request_files(cache_root, current, market_id)],
                    "results": results,
                }
            experiment_result = manifest.get("experiment_result")
            hf_add_files: list[tuple[Path, str]] = []
            if _workflow_profile(config) == EXHAUSTIVE_PROFILE:
                if not isinstance(experiment_result, dict):
                    raise RuntimeError("exhaustive_v1 pipeline did not return experiment_result")
                experiment_result["request_id"] = request_path.stem
                result_path = cache_root / _result_remote_path(market_id, current, request_path.stem)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(experiment_result, indent=2, sort_keys=True), encoding="utf-8")
                if push_hf:
                    hf_add_files.append((result_path, _result_remote_path(market_id, current, request_path.stem)))
            live_cache_files = _live_cache_files(cache_root, market_id) if push_hf else []
            hf_add_files.extend(live_cache_files)
            live_cache_push = {
                "status": "queued" if live_cache_files else "skipped",
                "reason": "" if push_hf else "push_hf flag not set",
                "paths": [remote for _, remote in live_cache_files],
            }
            batch_hf_add_files.extend(hf_add_files)
            if push_hf:
                batch_hf_delete_paths.append(f"{REQUEST_ROOT}/{current}/{request_path.name}")
            batch_request_paths.append(request_path)
            batch_results.append(
                {
                    "request": request_path.name,
                    "status": "completed_waiting_for_batch_push",
                    "manifest_status": manifest.get("status"),
                    "live_cache_push": live_cache_push,
                }
            )

        if batch_results:
            github_push = explicit_git_push(
                root=root,
                paths=[f"experiments/{market_id}"],
                message=f"Update anonymized experiment artifacts for {market_id} {current} batch",
                enabled=push_github,
                target_branch=github_branch,
            )
            github_artifacts_saved = github_push["status"] == "pushed" or (
                push_github and github_push["status"] == "skipped" and github_push.get("reason") == "no staged changes"
            )
            deduped_hf_add_files = [
                (local, remote)
                for remote, local in dict((remote, local) for local, remote in batch_hf_add_files).items()
            ]
            if github_artifacts_saved:
                hf_push = _commit_hf_changes(
                    client,
                    add_files=deduped_hf_add_files if push_hf else [],
                    delete_paths=batch_hf_delete_paths if push_hf else [],
                    commit_message=f"Complete {len(batch_results)} {market_id} {current} requests",
                )
                for request_path in batch_request_paths:
                    request_path.unlink(missing_ok=True)
                request_status = "completed_deleted_from_hf" if push_hf else "completed_not_deleted_push_hf_disabled"
            else:
                hf_push = _commit_hf_changes(
                    client,
                    add_files=deduped_hf_add_files if push_hf else [],
                    delete_paths=[],
                    commit_message=f"Record {len(batch_results)} {market_id} {current} requests",
                )
                request_status = "completed_not_deleted_waiting_for_github_push"
            for result in batch_results:
                result.update({"status": request_status, "github_push": github_push, "hf_push": hf_push})
            results.extend(batch_results)

        remaining = _request_files(cache_root, current, market_id)
        phase_advanced = False
        if not remaining:
            previous = current
            if _workflow_profile(config) == EXHAUSTIVE_PROFILE:
                finalized = _finalize_exhaustive_phase(
                    client,
                    root=root,
                    cache_root=cache_root,
                    config_path=selected,
                    config=config,
                    phase=current,
                    market_id=market_id,
                )
                if finalized["status"] != "completed":
                    return {
                        "status": "blocked",
                        "market_id": market_id,
                        "current_phase": current,
                        "reason": finalized["reason"],
                        "remaining_requests": [],
                        "results": results,
                    }
                _mark_public_phase_winners(root, market_id, list(finalized["selected"]))
                explicit_git_push(
                    root=root,
                    paths=[f"experiments/{market_id}/results/phase_results_public.json"],
                    message=f"Select anonymized {market_id} {current} results",
                    enabled=push_github,
                    target_branch=github_branch,
                )
            current, generated, done = _advance_completed_phase(
                client,
                cache_root=cache_root,
                config_path=selected,
                config=config,
                current=current,
                market_id=market_id,
            )
            phase_advanced = True
            if done:
                return {
                    "status": "completed",
                    "market_id": market_id,
                    "current_phase": "done",
                    "completed_phase": previous,
                    "pulled": len(pulled_requests),
                    "pulled_live_cache": len(pulled_live_cache),
                    "remaining_requests": [],
                    "results": results,
                }
            remaining = _request_files(cache_root, current, market_id)
        return {
            "status": "completed" if not remaining else "partial",
            "market_id": market_id,
            "current_phase": current,
            "pulled": len(pulled_requests),
            "pulled_live_cache": len(pulled_live_cache),
            "phase_advanced": phase_advanced,
            "remaining_requests": [path.name for path in remaining],
            "results": results,
            "restored_artifacts": restored_artifacts,
        }


def rerun_post_lock_validation(
    *,
    root: Path,
    cache_root: Path,
    repo_id: str,
    token: str,
    repo_type: str,
    push_github: bool,
    github_branch: str,
    push_hf: bool,
    market: str = "",
) -> dict[str, Any]:
    settings = HfSettings.from_env_or_args(repo_id=repo_id, token=token, repo_type=repo_type)
    client = HfStateClient(settings)
    with workflow_lock(cache_root, "experiments"):
        pulled_requests = pull_hf_folder(client, prefix=REQUEST_ROOT, local_dir=cache_root)
        pulled_live_cache = pull_hf_folder(client, prefix=LIVE_DATA_ROOT, local_dir=cache_root)
        pulled_experiment_state = pull_hf_folder(client, prefix=EXPERIMENT_STATE_ROOT, local_dir=cache_root)
        config_paths = _config_files(cache_root)
        if market:
            config_paths = [
                path for path in config_paths if market_id_from_config(load_yaml(path)) == market
            ]
        if not config_paths:
            raise RuntimeError("No experiment config found for post-lock rerun")

        results: list[dict[str, Any]] = []
        for config_path in config_paths:
            config = load_yaml(config_path)
            validate_config(config)
            market_id = market_id_from_config(config)
            if _workflow_profile(config) != EXHAUSTIVE_PROFILE:
                continue
            restored_artifacts = restore_artifact_branch_paths(
                root,
                paths=[f"experiments/{market_id}"],
                target_branch=github_branch,
            ) if push_github and github_branch else {"status": "skipped", "reason": "github push disabled"}
            phase16_results = [_use_worked_candidate_result(result) for result in _load_phase_results(cache_root, market_id, "phase16")]
            if not phase16_results:
                raise RuntimeError(f"No phase16 results found for {market_id}")
            ranking_metric = str(config.get("experiments", {}).get("ranking_metric") or "direction_accuracy")
            top_k = int(config.get("production", {}).get("top_k", 3))
            selected = _post_lock_selected_records(phase16_results, ranking_metric=ranking_metric, top_k=top_k)
            if len(selected) < top_k:
                raise RuntimeError(f"Post-lock rerun requires {top_k} Top K models for {market_id}; found {len(selected)}")
            try:
                validation = _run_final_post_lock_sliding_validation(
                    root=root,
                    client=client,
                    config=config,
                    market_id=market_id,
                    selected=selected,
                    ranking_metric=ranking_metric,
                    upload_private=push_hf,
                )
            except RuntimeError as exc:
                if "Post-lock sliding validation did not complete" not in str(exc):
                    raise
                results.append(
                    {
                        "market_id": market_id,
                        "status": "blocked_incomplete_post_lock",
                        "reason": str(exc),
                        "top_k": [str(record.get("public_id") or "") for record in selected],
                        "restored_artifacts": restored_artifacts,
                        "pulled": len(pulled_requests),
                        "pulled_live_cache": len(pulled_live_cache),
                        "pulled_experiment_state": len(pulled_experiment_state),
                    }
                )
                continue
            _mark_public_phase_winners(root, market_id, selected)
            github_push = explicit_git_push(
                root=root,
                paths=[f"experiments/{market_id}"],
                message=f"Rerun {market_id} phase16 Top K post-lock validation",
                enabled=push_github,
                target_branch=github_branch,
            )
            results.append(
                {
                    "market_id": market_id,
                    "status": "completed",
                    "restored_artifacts": restored_artifacts,
                    "github_push": github_push,
                    "top_k": [str(record.get("public_id") or "") for record in selected],
                    "post_lock_models": sorted((validation or {}).get("models") or {}),
                    "pulled": len(pulled_requests),
                    "pulled_live_cache": len(pulled_live_cache),
                    "pulled_experiment_state": len(pulled_experiment_state),
                }
            )
        if not results:
            raise RuntimeError("No exhaustive experiment configs found for post-lock rerun")
        return {"status": "completed", "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HF-backed experiment workflow with local/Actions parity.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--cache-root", default="private_hf_cache")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--repo-type", default=os.environ.get("HF_REPO_TYPE", "model"), choices=["model", "dataset", "space"]
    )
    parser.add_argument("--push-github", action="store_true")
    parser.add_argument("--github-branch", default=os.environ.get("GITHUB_ARTIFACT_BRANCH", ""))
    parser.add_argument("--push-hf", action="store_true")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--post-lock-only", action="store_true", help="Rerun only phase16 Top K post-lock validation.")
    parser.add_argument("--market", default="", help="Optional market id for --post-lock-only.")
    parser.add_argument("--result-file", default="", help="Optional path for machine-readable workflow result JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common = {
        "root": Path(args.root).resolve(),
        "cache_root": Path(args.cache_root),
        "repo_id": str(args.repo_id),
        "token": str(args.token),
        "repo_type": str(args.repo_type),
        "push_github": bool(args.push_github),
        "github_branch": str(args.github_branch or ""),
        "push_hf": bool(args.push_hf),
    }
    if args.post_lock_only:
        result = rerun_post_lock_validation(**common, market=str(args.market or ""))
    else:
        result = run_experiment_workflow(**common, max_requests=int(args.max_requests))
    result_json = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.result_file:
        result_path = Path(args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(result_json + "\n", encoding="utf-8")
    print(result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
