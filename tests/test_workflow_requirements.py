from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.pull_private_src import pull_private_src
import pytest

try:
    from src.private.training.discovery_catalog import PHASES as DISCOVERY_PHASES, catalog_hash
    PRIVATE_TRAINING_AVAILABLE = True
except ModuleNotFoundError:
    DISCOVERY_PHASES = []
    catalog_hash = None
    PRIVATE_TRAINING_AVAILABLE = False
from src.public.config.schema import EXHAUSTIVE_PHASES, validate_config
from src.public.evaluation.ranking import deterministic_discovery_rank, rank_validation_candidates
from src.public.evaluation.production import build_public_prediction_result, select_top_candidate_ids
from src.public.evaluation.versioning import update_model_version
from src.public.evaluation.sanitizer import assert_public_safe_text
from src.workflows.hf_state import dump_yaml, load_yaml, phase_value, workflow_lock
import src.workflows.hf_state as hf_state
from src.workflows.orchestrate_experiments import (
    _config_files,
    _advance_completed_phase,
    _finalize_exhaustive_phase,
    _first_incomplete_config,
    _generate_phase_requests,
    _inherit_model_family_result,
    _phase_is_complete,
    _push_live_cache,
    _push_phase_state,
)
import src.workflows.orchestrate_experiments as orchestrate_experiments
import src.workflows.sync_hf_artifacts as sync_hf_artifacts
from src.workflows.orchestrate_prod import _ensure_anonymized_dataset_marker, _prediction_is_current
from src.workflows.update_pages import _plot, update_pages
from src.workflows.update_readme import END_MARKER, START_MARKER, update_readme
from scripts.reset_market import delete_hf_paths, hf_reset_paths, local_reset_paths, reset_market
from src.workflows.reset_all_outputs import delete_hf_outputs, hf_output_paths, local_output_paths, reset_all_outputs


def _require_private_training() -> None:
    if not PRIVATE_TRAINING_AVAILABLE:
        pytest.skip("private training source not hydrated from HF")


def _base_config(market_id: str = "btc_1h", current_phase: str = "none") -> dict:
    return {
        "project": {"name": "quant_ml_platform", "public_delay_hours": 24, "default_seed": 42},
        "current_phase": current_phase,
        "market": {
            "market_id": market_id,
            "market_type": "crypto",
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "timeframe": "1h",
            "market_hours_mode": "full_time",
        },
        "fetch": {"enabled": True},
        "schema": {},
        "split": {"train_length": 100, "validation_length": 40, "test_length": 40},
        "features": {},
        "experiments": {
            "denoising": ["none", "ema"],
            "sequence_lengths": [24],
            "test_end_utc": "2026-05-31T23:59:59+00:00",
        },
        "production": {"enabled": True, "start_utc": "2026-06-01T00:00:00+00:00"},
        "reporting": {},
    }


def _exhaustive_config() -> dict:
    config = _base_config()
    config["split"] = {"train_length": 500, "validation_length": 300, "test_length": 300}
    config["experiments"].update(
        {
            "workflow_profile": "exhaustive_v1",
            "phases": EXHAUSTIVE_PHASES,
            "ranking_metric": "direction_accuracy",
            "phase5_to_phase6_inheritance": "unfixed",
        }
    )
    return config


def test_exhaustive_catalog_has_all_16_ordered_phases() -> None:
    _require_private_training()
    assert [phase.phase_id for phase in DISCOVERY_PHASES] == EXHAUSTIVE_PHASES
    assert all(phase.variations for phase in DISCOVERY_PHASES)
    assert len(catalog_hash()) == 64
    validate_config(_exhaustive_config())


def test_direction_accuracy_is_primary_and_test_metrics_do_not_rank() -> None:
    rows = [
        {
            "candidate_id": "higher_ba",
            "status": "worked",
            "validation": {"direction_accuracy": 0.61, "balanced_accuracy": 0.90},
            "test": {"direction_accuracy": 1.0},
        },
        {
            "candidate_id": "higher_direction",
            "status": "worked",
            "validation": {"direction_accuracy": 0.62, "balanced_accuracy": 0.55},
            "test": {"direction_accuracy": 0.0},
        },
    ]
    assert deterministic_discovery_rank(rows, "direction_accuracy")[0]["candidate_id"] == "higher_direction"
    rows[0]["test"]["direction_accuracy"] = 0.0
    rows[1]["test"]["direction_accuracy"] = 1.0
    assert deterministic_discovery_rank(rows, "direction_accuracy")[0]["candidate_id"] == "higher_direction"


def test_phase_inheritance_mode_is_validated() -> None:
    config = _exhaustive_config()
    config["experiments"]["phase5_to_phase6_inheritance"] = "unknown"
    with pytest.raises(ValueError, match="fixed or unfixed"):
        validate_config(config)


def test_fixed_phase5_requests_include_phase4_parent_recipe(tmp_path: Path) -> None:
    _require_private_training()
    config = _exhaustive_config()
    config["experiments"]["phase5_to_phase6_inheritance"] = "fixed"
    config["discovery_state"] = {
        "catalog_hash": catalog_hash(),
        "selected_recipes": [
            {
                "candidate_id": "logistic_regression",
                "recipe_hash": "phase4-winner",
                "decisions": {"lookback_window": {"variation_id": "lookback_48", "parameters": {}}},
                "resolved_hyperparameters": {"seed": 42, "hidden_dim": 37, "epochs": 23},
            }
        ],
    }
    config_path = tmp_path / "markets" / "btc_1h.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(dump_yaml(config), encoding="utf-8")

    generated = _generate_phase_requests(tmp_path, config_path, config, "phase05")
    payload = load_yaml(generated[0])

    assert len(generated) == len(DISCOVERY_PHASES[4].variations)
    assert {load_yaml(path)["workflow"]["parent_recipe_hash"] for path in generated} == {"phase4-winner"}
    assert payload["workflow"]["parent_recipe"]["recipe_hash"] == "phase4-winner"
    assert payload["workflow"]["parent_recipe_hash"] == "phase4-winner"
    assert payload["workflow"]["parent_recipe"]["resolved_hyperparameters"]["hidden_dim"] == 37


def test_phase6_requests_receive_phase5_winning_recipe(tmp_path: Path) -> None:
    _require_private_training()
    config = _exhaustive_config()
    phase5_winner = {
        "candidate_id": "random_forest",
        "recipe_hash": "phase5-winner",
        "parent_recipe_hash": "phase4-winner",
        "decisions": {
            "lookback_window": {"variation_id": "lookback_48", "parameters": {}},
            "model_family": {"variation_id": "random_forest", "parameters": {}},
        },
        "resolved_hyperparameters": {"seed": 42, "hidden_dim": 37, "epochs": 23},
        "resolved_data_contract": {"feature_columns": ["return", "volume"], "sequence_length": 48},
    }
    config["discovery_state"] = {
        "catalog_hash": catalog_hash(),
        "selected_recipes": [phase5_winner],
    }
    config_path = tmp_path / "markets" / "btc_1h.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(dump_yaml(config), encoding="utf-8")

    generated = _generate_phase_requests(tmp_path, config_path, config, "phase06")
    payload = load_yaml(generated[0])

    assert len(generated) == len(DISCOVERY_PHASES[5].variations)
    assert payload["workflow"]["parent_recipe_hash"] == "phase5-winner"
    assert payload["workflow"]["parent_recipe"] == phase5_winner
    assert payload["workflow"]["parent_recipe"]["decisions"]["model_family"] == {
        "variation_id": "random_forest",
        "parameters": {},
    }


def test_fixed_phase5_selected_recipe_carries_full_phase4_recipe() -> None:
    baseline = {
        "recipe_hash": "phase4-winner",
        "candidate_id": "lstm",
        "decisions": {
            "target_horizon": {"variation_id": "next_6", "parameters": {}},
            "lookback_window": {"variation_id": "lookback_48", "parameters": {}},
        },
        "resolved_hyperparameters": {"seed": 42, "hidden_dim": 37, "epochs": 23},
        "resolved_data_contract": {"feature_columns": ["return", "volume"], "sequence_length": 48},
        "custom_private_marker": "keep-me",
    }
    phase5_result = {
        "candidate_id": "random_forest",
        "variation_id": "random_forest",
        "parent_recipe_hash": "phase4-winner",
        "recipe": {
            "candidate_id": "random_forest",
            "decisions": {"model_family": {"variation_id": "random_forest", "parameters": {"hidden_dim": 999}}},
            "resolved_hyperparameters": {"seed": 99, "hidden_dim": 999, "epochs": 99},
            "resolved_data_contract": {"feature_columns": ["wrong"], "sequence_length": 1},
        },
        "train": {"direction_accuracy": 0.7},
        "validation": {"direction_accuracy": 0.65},
        "test": {"direction_accuracy": 0.6},
        "overfit_diagnostics": {"train_valid_gap": 0.05},
    }

    carried = _inherit_model_family_result(phase5_result, baseline)

    assert carried["custom_private_marker"] == "keep-me"
    assert carried["parent_recipe_hash"] == "phase4-winner"
    assert carried["candidate_id"] == "random_forest"
    assert carried["decisions"]["target_horizon"]["variation_id"] == "next_6"
    assert carried["decisions"]["lookback_window"]["variation_id"] == "lookback_48"
    assert carried["decisions"]["model_family"] == {"variation_id": "random_forest", "parameters": {}}
    assert carried["resolved_hyperparameters"] == baseline["resolved_hyperparameters"]
    assert carried["resolved_data_contract"] == baseline["resolved_data_contract"]
    assert carried["recipe_hash"] != "phase4-winner"


def test_production_versions_are_append_only_and_independent() -> None:
    first = update_model_version(
        previous=None,
        public_id="V001",
        rank=1,
        train={"direction_accuracy": 0.7},
        validation={"direction_accuracy": 0.65},
        result=None,
        generated_at_utc="2026-06-24T00:00:00+00:00",
        trained_through_utc="2026-06-23T23:00:00+00:00",
        activate_new_version=True,
        refresh_active_metrics=False,
    )
    second = update_model_version(
        previous=first,
        public_id="V001",
        rank=1,
        train={"direction_accuracy": 0.72},
        validation={"direction_accuracy": 0.66},
        result={"metrics": {"direction_accuracy": 0.8}, "prediction_series": []},
        generated_at_utc="2026-06-25T00:00:00+00:00",
        trained_through_utc="2026-06-24T23:00:00+00:00",
        activate_new_version=True,
        refresh_active_metrics=False,
    )
    assert second["current_version"] == 2
    assert len(second["versions"]) == 2
    assert second["versions"][0]["deactivated_at_utc"] == "2026-06-25T00:00:00+00:00"
    assert second["versions"][1]["production_metrics"]["direction_accuracy"] == 0.8


def test_initial_production_version_can_activate_at_public_start() -> None:
    first = update_model_version(
        previous=None,
        public_id="V001",
        rank=1,
        train={"direction_accuracy": 0.7},
        validation={"direction_accuracy": 0.65},
        result={"metrics": {"direction_accuracy": 0.8}, "prediction_series": []},
        generated_at_utc="2026-06-25T00:00:00+00:00",
        activated_at_utc="2026-06-01T00:00:00+00:00",
        trained_through_utc="2026-05-31T23:00:00+00:00",
        activate_new_version=True,
        refresh_active_metrics=False,
    )
    assert first["versions"][0]["activated_at_utc"] == "2026-06-01T00:00:00+00:00"
    assert first["delayed_metrics"]["direction_accuracy"] == 0.8


def test_empty_active_version_can_be_backfilled_from_public_start() -> None:
    previous = update_model_version(
        previous=None,
        public_id="V001",
        rank=1,
        train={"direction_accuracy": 0.7},
        validation={"direction_accuracy": 0.65},
        result=None,
        generated_at_utc="2026-06-25T00:00:00+00:00",
        trained_through_utc="2026-05-31T23:00:00+00:00",
        activate_new_version=True,
        refresh_active_metrics=False,
    )
    refreshed = update_model_version(
        previous=previous,
        public_id="V001",
        rank=1,
        train={"direction_accuracy": 0.7},
        validation={"direction_accuracy": 0.65},
        result={"metrics": {"direction_accuracy": 0.8}, "prediction_series": [{"timestamp": "2026-06-01T00:00:00+00:00", "performance": 0.0}]},
        generated_at_utc="2026-06-25T01:00:00+00:00",
        activated_at_utc="2026-06-01T00:00:00+00:00",
        trained_through_utc="2026-05-31T23:00:00+00:00",
        activate_new_version=False,
        refresh_active_metrics=True,
    )
    assert len(refreshed["versions"]) == 1
    assert refreshed["versions"][0]["activated_at_utc"] == "2026-06-01T00:00:00+00:00"
    assert refreshed["delayed_metrics"]["direction_accuracy"] == 0.8


def test_production_cannot_start_before_june_2026() -> None:
    config = _exhaustive_config()
    config["production"]["start_utc"] = "2026-05-31T23:59:59+00:00"
    config["experiments"]["test_end_utc"] = "2026-05-30T23:59:59+00:00"

    with pytest.raises(ValueError, match="2026-06-01"):
        validate_config(config)


def test_phase_completion_triggers_reporting_and_final_production() -> None:
    experiments = Path(".github/workflows/experiments.yml").read_text(encoding="utf-8")
    pages = Path(".github/workflows/pages-update.yml").read_text(encoding="utf-8")
    readme = Path(".github/workflows/readme-update.yml").read_text(encoding="utf-8")
    prod = Path(".github/workflows/prod.yml").read_text(encoding="utf-8")

    assert "phase_completed" in experiments
    assert "gh workflow run pages-update.yml" in experiments
    assert "gh workflow run readme-update.yml" in experiments
    assert "workflow_done" in experiments
    assert 'completed_phase in {"phase16", "16"}' in experiments
    assert "gh workflow run prod.yml" in experiments
    assert 'workflows: ["prod"]' in pages
    assert 'workflows: ["prod"]' in readme
    assert 'cron: "17 6 * * *"' in prod


def test_exhaustive_phase_requests_are_deterministic_and_axis_specific(tmp_path: Path) -> None:
    _require_private_training()
    config = _exhaustive_config()
    path = tmp_path / "requests" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(dump_yaml(config), encoding="utf-8")

    generated = _generate_phase_requests(tmp_path, path, config, "phase01")

    assert len(generated) == len(DISCOVERY_PHASES[0].variations)
    payload = load_yaml(generated[0])
    assert payload["workflow"]["phase_id"] == "phase01"
    assert payload["workflow"]["axis"] == "target_horizon"
    assert payload["workflow"]["catalog_hash"] == catalog_hash()
    assert payload["workflow"]["parent_recipe_hash"].startswith("recipe_")


def test_exhaustive_phase15_advances_to_phase16_then_done(tmp_path: Path) -> None:
    _require_private_training()
    class FakeClient:
        def __init__(self) -> None:
            self.commits = []
            self.uploads = []

        def commit_files(self, files, *, commit_message):
            self.commits.append((files, commit_message))

        def upload_file(self, local_path, hf_path, *, commit_message):
            self.uploads.append((local_path, hf_path, commit_message))

    client = FakeClient()
    config = _exhaustive_config()
    config["current_phase"] = "15"
    config_path = tmp_path / "requests" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(dump_yaml(config), encoding="utf-8")
    state_path = tmp_path / "experiment_state" / "btc_1h" / "workflow_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"current_phase": "phase16", "status": "running"}), encoding="utf-8")

    current, generated, done = _advance_completed_phase(
        client,
        cache_root=tmp_path,
        config_path=config_path,
        config=config,
        current="phase15",
        market_id="btc_1h",
    )
    assert current == "phase16"
    assert len(generated) == 1
    assert not done

    current, generated, done = _advance_completed_phase(
        client,
        cache_root=tmp_path,
        config_path=config_path,
        config=config,
        current="phase16",
        market_id="btc_1h",
    )
    assert current == "done"
    assert generated == []
    assert done
    assert phase_value(load_yaml(config_path)) == "done"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["current_phase"] == "done"
    assert state["status"] == "completed"


def test_robustness_finalizer_uses_worked_candidate_result_when_wrapper_failed(tmp_path: Path) -> None:
    _require_private_training()

    class FakeClient:
        def __init__(self) -> None:
            self.commits = []
            self.uploads = []

        def commit_files(self, files, *, commit_message):
            self.commits.append((files, commit_message))

        def upload_file(self, local_path, hf_path, *, commit_message):
            self.uploads.append((local_path, hf_path, commit_message))

    config = _exhaustive_config()
    variations = [item.variation_id for item in DISCOVERY_PHASES[14].variations]
    config["discovery_state"] = {
        "expected_requests": [f"btc_1h_phase15_request_{index:03d}" for index, _ in enumerate(variations, 1)],
        "selected_recipes": [{"recipe_hash": "recipe_locked", "decisions": {}}],
    }
    config_path = tmp_path / "requests" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(dump_yaml(config), encoding="utf-8")
    result_dir = tmp_path / "experiment_state" / "btc_1h" / "phases" / "phase15" / "results"
    result_dir.mkdir(parents=True)

    for index, variation in enumerate(variations, 1):
        request_id = f"btc_1h_phase15_request_{index:03d}"
        payload = {
            "request_id": request_id,
            "experiment_id": f"experiment_{index}",
            "variation_id": variation,
            "status": "failed",
            "failure_class": "candidate",
            "candidate_results": [
                {
                    "experiment_id": f"candidate_{index}",
                    "candidate_id": "rank_weighted",
                    "variation_id": variation,
                    "status": "worked",
                    "decision_eligible": True,
                    "passed_leakage_checks": True,
                    "passed_privacy_checks": True,
                    "recipe": {"recipe_hash": f"recipe_{index}", "candidate_id": "rank_weighted"},
                    "recipe_hash": f"recipe_{index}",
                    "validation": {"balanced_accuracy": 0.7, "mcc": 0.2, "f1": 0.6, "trade_count": 10},
                    "overfit_diagnostics": {},
                }
            ],
        }
        (result_dir / f"{request_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _finalize_exhaustive_phase(
        FakeClient(),
        root=tmp_path,
        cache_root=tmp_path,
        config_path=config_path,
        config=config,
        phase="phase15",
        market_id="btc_1h",
    )

    assert result["status"] == "completed"


def test_final_lock_finalizer_uses_worked_candidate_result_when_wrapper_failed(tmp_path: Path) -> None:
    _require_private_training()

    class FakeClient:
        def commit_files(self, files, *, commit_message):
            self.files = files

        def upload_file(self, local_path, hf_path, *, commit_message):
            self.upload = (local_path, hf_path)

    config = _exhaustive_config()
    config["discovery_state"] = {
        "expected_requests": ["btc_1h_phase16_request_001"],
        "selected_recipes": [{"recipe_hash": "recipe_locked", "decisions": {}}],
    }
    config_path = tmp_path / "requests" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(dump_yaml(config), encoding="utf-8")
    result_dir = tmp_path / "experiment_state" / "btc_1h" / "phases" / "phase16" / "results"
    result_dir.mkdir(parents=True)
    (result_dir / "btc_1h_phase16_request_001.json").write_text(
        json.dumps(
            {
                "request_id": "btc_1h_phase16_request_001",
                "experiment_id": "experiment_wrapper",
                "variation_id": "lock_winner",
                "status": "failed",
                "failure_class": "candidate",
                "candidate_results": [
                    {
                        "experiment_id": "candidate_1",
                        "candidate_id": "rank_weighted",
                        "variation_id": "lock_winner",
                        "status": "worked",
                        "decision_eligible": True,
                        "passed_leakage_checks": True,
                        "passed_privacy_checks": True,
                        "recipe": {"recipe_hash": "recipe_final", "candidate_id": "rank_weighted"},
                        "recipe_hash": "recipe_final",
                        "validation": {"balanced_accuracy": 0.7, "mcc": 0.2, "f1": 0.6, "trade_count": 10},
                        "overfit_diagnostics": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _finalize_exhaustive_phase(
        FakeClient(),
        root=tmp_path,
        cache_root=tmp_path,
        config_path=config_path,
        config=config,
        phase="phase16",
        market_id="btc_1h",
    )

    assert result["status"] == "completed"


def test_final_post_lock_uses_full_selected_recipe_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakePipeline:
        @staticmethod
        def _evaluate_post_lock_sliding_windows(**kwargs):
            captured.update(kwargs)
            return {"status": "worked", "models": {}}

    class FakeClient:
        def upload_file(self, local_path, hf_path, *, commit_message):
            self.upload = (local_path, hf_path, commit_message)

    monkeypatch.setattr(
        orchestrate_experiments,
        "_require_private_training",
        lambda: (None, None, FakePipeline),
    )
    config = _exhaustive_config()
    selected = [
        {
            "public_id": "E_1",
            "candidate_id": "lstm",
            "recipe": {
                "candidate_id": "lstm",
                "feature_columns": ["return", "return_context_24", "return_context_168"],
                "sequence_length": 168,
                "scaling_transform": "robust",
                "resolved_hyperparameters": {"sequence_length": 168},
                "decisions": {
                    "denoising": {"variation_id": "ema"},
                    "lookback_window": {"variation_id": "multi_resolution_24_168"},
                    "feature_set": {"variation_id": "momentum"},
                },
            },
        }
    ]

    orchestrate_experiments._run_final_post_lock_sliding_validation(
        root=tmp_path,
        client=FakeClient(),
        config=config,
        market_id="btc_1h",
        selected=selected,
        ranking_metric="direction_accuracy",
    )

    assert captured["data_variation"] == "ema"
    assert captured["decisions"] == selected[0]["recipe"]["decisions"]
    assert captured["feature_columns"] == selected[0]["recipe"]["feature_columns"]
    assert captured["sequence_length"] == 168
    assert captured["scaling_method"] == "robust"


def test_discovery_ranking_never_reads_test_metrics() -> None:
    candidates = [
        {
            "candidate_id": "a",
            "status": "worked",
            "validation": {"balanced_accuracy": 0.7, "mcc": 0.2, "f1": 0.6},
            "test": {"balanced_accuracy": 0.0},
        },
        {
            "candidate_id": "b",
            "status": "worked",
            "validation": {"balanced_accuracy": 0.6, "mcc": 0.9, "f1": 0.9},
            "test": {"balanced_accuracy": 1.0},
        },
    ]
    assert deterministic_discovery_rank(candidates)[0]["candidate_id"] == "a"
    candidates[0]["test"]["balanced_accuracy"] = 1.0
    candidates[1]["test"]["balanced_accuracy"] = 0.0
    assert deterministic_discovery_rank(candidates)[0]["candidate_id"] == "a"


def test_discovery_ranking_rejects_ineligible_or_failed_safety_checks() -> None:
    base = {
        "status": "worked",
        "validation": {"balanced_accuracy": 0.6, "mcc": 0.2, "f1": 0.5},
    }
    candidates = [
        {**base, "candidate_id": "eligible"},
        {**base, "candidate_id": "leaky", "validation": {"balanced_accuracy": 0.99}, "passed_leakage_checks": False},
        {**base, "candidate_id": "private", "validation": {"balanced_accuracy": 0.98}, "passed_privacy_checks": False},
        {**base, "candidate_id": "thin", "validation": {"balanced_accuracy": 0.97}, "decision_eligible": False},
    ]
    assert [row["candidate_id"] for row in deterministic_discovery_rank(candidates)] == ["eligible"]


def test_phase_candidate_must_improve_baseline_recipe_to_be_adopted() -> None:
    baseline = {
        "recipe_hash": "baseline",
        "selection_metrics": {
            "validation": {"balanced_accuracy": 0.70, "mcc": 0.30, "f1": 0.40},
            "overfit_diagnostics": {"train_valid_gap": 0.05},
        },
    }
    worse = {
        "recipe": {"recipe_hash": "candidate"},
        "validation": {"balanced_accuracy": 0.69, "mcc": 0.50, "f1": 0.50},
        "overfit_diagnostics": {"train_valid_gap": 0.01},
    }
    better = {
        "recipe": {"recipe_hash": "candidate"},
        "validation": {"balanced_accuracy": 0.71, "mcc": 0.10, "f1": 0.10},
        "overfit_diagnostics": {"train_valid_gap": 0.20},
    }

    recipe, adopted = orchestrate_experiments._adopt_if_improved(worse, baseline)
    assert adopted is False
    assert recipe["recipe_hash"] == "baseline"

    recipe, adopted = orchestrate_experiments._adopt_if_improved(better, baseline)
    assert adopted is True
    assert recipe["recipe_hash"] == "candidate"
    assert recipe["selection_metrics"]["validation"]["balanced_accuracy"] == 0.71


def test_experiment_configs_are_sorted_and_first_not_done_selected(tmp_path: Path) -> None:
    configs = tmp_path / "requests" / "configs"
    configs.mkdir(parents=True)
    (configs / "b.yaml").write_text(dump_yaml(_base_config("b_1h", "none")), encoding="utf-8")
    (configs / "a.yaml").write_text(dump_yaml(_base_config("a_1h", "done")), encoding="utf-8")
    (configs / "c.yaml").write_text(dump_yaml(_base_config("c_1h", "1")), encoding="utf-8")

    assert [path.name for path in _config_files(tmp_path)] == ["a.yaml", "b.yaml", "c.yaml"]
    assert _first_incomplete_config(tmp_path).name == "b.yaml"


def test_experiment_config_requires_current_phase(tmp_path: Path) -> None:
    configs = tmp_path / "requests" / "configs"
    configs.mkdir(parents=True)
    config = _base_config("a_1h", "none")
    config.pop("current_phase")
    (configs / "a.yaml").write_text(dump_yaml(config), encoding="utf-8")

    try:
        _first_incomplete_config(tmp_path)
        raise AssertionError("missing current_phase should fail validation")
    except ValueError as exc:
        assert "current_phase" in str(exc)


def test_phase_requests_are_anonymized_and_committed_with_config(tmp_path: Path) -> None:
    config_path = tmp_path / "requests" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config = _base_config("btc_1h", "none")
    config_path.write_text(dump_yaml(config), encoding="utf-8")

    generated = _generate_phase_requests(tmp_path, config_path, config, "phase0")
    updated_config = load_yaml(config_path)

    assert phase_value(updated_config) == "0"
    assert len(generated) == 2
    for request in generated:
        payload = load_yaml(request)
        assert payload["data_variation"]["anonymous_dataset_id"].startswith("ds_")
        assert payload["market"]["market_id"] == "btc_1h"
        assert payload["workflow"]["phase"] == "phase0"

    class FakeClient:
        def __init__(self) -> None:
            self.commits: list[tuple[list[tuple[Path, str]], str]] = []

        def commit_files(self, files: list[tuple[Path, str]], *, commit_message: str) -> None:
            self.commits.append((files, commit_message))

    fake = FakeClient()
    _push_phase_state(
        fake, cache_root=tmp_path, config_path=config_path, generated=generated, phase="phase0", market_id="btc_1h"
    )
    assert len(fake.commits) == 1
    committed_paths = [remote for _, remote in fake.commits[0][0]]
    assert "requests/config.yaml" in committed_paths
    assert all(path.startswith("requests/phase0/") or path == "requests/config.yaml" for path in committed_paths)


def test_live_cache_push_uploads_only_market_private_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "private_hf_cache"
    live_dir = cache_root / "live_data" / "btc_1h"
    live_dir.mkdir(parents=True)
    (live_dir / "1h.parquet").write_text("binary-placeholder", encoding="utf-8")
    (live_dir / "1h.parquet.json").write_text("{}", encoding="utf-8")
    (cache_root / "live_data" / "eth_1h").mkdir(parents=True)
    (cache_root / "live_data" / "eth_1h" / "1h.parquet").write_text("other", encoding="utf-8")

    class FakeClient:
        def __init__(self) -> None:
            self.commits: list[tuple[list[tuple[Path, str]], str]] = []

        def commit_files(self, files: list[tuple[Path, str]], *, commit_message: str) -> None:
            self.commits.append((files, commit_message))

    fake = FakeClient()
    result = _push_live_cache(fake, cache_root=cache_root, market_id="btc_1h")

    assert result["status"] == "pushed"
    assert result["paths"] == ["live_data/btc_1h/1h.parquet", "live_data/btc_1h/1h.parquet.json"]
    assert fake.commits[0][1] == "Update live data cache for btc_1h"


def test_phase_completion_uses_absence_of_market_requests(tmp_path: Path) -> None:
    phase_dir = tmp_path / "requests" / "phase1"
    phase_dir.mkdir(parents=True)
    (phase_dir / "other_1h_phase1_request_01.yaml").write_text("current_phase: 1\n", encoding="utf-8")
    assert _phase_is_complete(tmp_path, "phase1", "btc_1h")
    (phase_dir / "btc_1h_phase1_request_01.yaml").write_text("current_phase: 1\n", encoding="utf-8")
    assert not _phase_is_complete(tmp_path, "phase1", "btc_1h")


def test_completed_phase_advances_state_immediately(tmp_path: Path) -> None:
    config_path = tmp_path / "requests" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config = _base_config("btc_1h", "0")
    config_path.write_text(dump_yaml(config), encoding="utf-8")

    class FakeClient:
        def __init__(self) -> None:
            self.commits: list[tuple[list[tuple[Path, str]], str]] = []
            self.uploads: list[tuple[Path, str, str]] = []

        def commit_files(self, files: list[tuple[Path, str]], *, commit_message: str) -> None:
            self.commits.append((files, commit_message))

        def upload_file(self, local_path: Path, hf_path: str, *, commit_message: str) -> None:
            self.uploads.append((local_path, hf_path, commit_message))

    fake = FakeClient()
    current, generated, done = _advance_completed_phase(
        fake,
        cache_root=tmp_path,
        config_path=config_path,
        config=config,
        current="phase0",
        market_id="btc_1h",
    )

    assert current == "phase1"
    assert not done
    assert len(generated) == 2
    assert phase_value(load_yaml(config_path)) == "1"
    committed_paths = [remote for _, remote in fake.commits[0][0]]
    assert "requests/config.yaml" in committed_paths
    assert all(path.startswith("requests/phase1/") or path == "requests/config.yaml" for path in committed_paths)


def test_production_dataset_marker_is_single_anonymized_variation_and_resumes(tmp_path: Path) -> None:
    config = _base_config("btc_1h", "done")
    first = _ensure_anonymized_dataset_marker(tmp_path, "btc_1h", config)
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    assert first_payload["single_dataset_variation"] is True
    assert first_payload["anonymized"] is True
    assert first_payload["raw_symbol_saved"] is False
    assert first_payload["append_policy"] == "initial_full_history_fetch"

    second = _ensure_anonymized_dataset_marker(tmp_path, "btc_1h", config)
    assert second == first
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert second_payload["append_policy"] == "append_only_new_rows_since_last_saved_point"
    assert second_payload["resume_from"] == first_payload["last_saved_point"]


def test_prediction_currency_gate_uses_24_hour_window(tmp_path: Path) -> None:
    path = tmp_path / "prod" / "btc_1h" / "results" / "production_public.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"prediction_series": [{"timestamp": datetime.now(timezone.utc).isoformat()}]}),
        encoding="utf-8",
    )
    assert _prediction_is_current(path, hours=24)

    old = datetime.now(timezone.utc) - timedelta(hours=25)
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_series": [{"timestamp": old.isoformat()}],
            }
        ),
        encoding="utf-8",
    )
    assert not _prediction_is_current(path, hours=24)


def test_workflow_lock_blocks_simultaneous_local_runs(tmp_path: Path) -> None:
    with workflow_lock(tmp_path, "experiments"):
        try:
            with workflow_lock(tmp_path, "experiments"):
                raise AssertionError("second lock should not be acquired")
        except RuntimeError as exc:
            assert "already held" in str(exc)
    with workflow_lock(tmp_path, "experiments"):
        assert (tmp_path / "private_hf_cache" / "locks" / "experiments.lock").exists()


def test_private_source_pull_requires_hf_credentials(tmp_path: Path) -> None:
    skipped = pull_private_src(
        root=tmp_path,
        repo_id="",
        token="",
        repo_type="model",
        prefix="src/private",
        required=False,
        clean=False,
    )
    assert skipped["status"] == "skipped"

    try:
        pull_private_src(
            root=tmp_path,
            repo_id="",
            token="",
            repo_type="model",
            prefix="src/private",
            required=True,
            clean=False,
        )
        raise AssertionError("required private source pull should fail without HF credentials")
    except RuntimeError as exc:
        assert "HF_TOKEN" in str(exc)


def test_readme_is_public_showcase_without_setup_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "```" not in readme
    assert "Quickstart" not in readme
    assert "python " not in readme
    assert "conda " not in readme
    assert "HF_TOKEN" not in readme
    assert "src/private" not in readme
    assert "data_variation" not in readme
    assert "Results Dashboard" in readme
    assert "Latest Public Results" in readme
    assert_public_safe_text(readme)


def test_update_readme_refreshes_public_results_snapshot(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Results\n", encoding="utf-8")
    experiment_results = tmp_path / "experiments" / "btc_1h" / "results"
    production_results = tmp_path / "prod" / "btc_1h" / "results"
    experiment_results.mkdir(parents=True)
    production_results.mkdir(parents=True)
    (experiment_results / "run_manifest_public.json").write_text(
        json.dumps({"market_id": "btc_1h", "workflow_profile": "exhaustive_v1"}), encoding="utf-8"
    )
    (experiment_results / "phase_results_public.json").write_text(
        json.dumps(
            [
                {
                    "phase_id": "phase01",
                    "status": "worked",
                    "selected_for_next_phase": True,
                    "validation": {"balanced_accuracy": 0.63},
                }
            ]
        ),
        encoding="utf-8",
    )
    (production_results / "production_public.json").write_text(
        json.dumps(
            {
                "market_id": "btc_1h",
                "winner": "model-a",
                "delayed_metrics": {"weighted_score": 1.25},
            }
        ),
        encoding="utf-8",
    )

    update_readme(tmp_path)
    first = (tmp_path / "README.md").read_text(encoding="utf-8")
    update_readme(tmp_path)
    second = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert first == second
    assert first.count(START_MARKER) == 1
    assert first.count(END_MARKER) == 1
    assert "btc_1h | 1/16 | 0.6300" in first
    assert "model-a | 1.2500" in first
    assert_public_safe_text(first)


def test_update_pages_generates_docs_from_public_artifacts(tmp_path: Path) -> None:
    exp_results = tmp_path / "experiments" / "btc_1h" / "results"
    prod_results = tmp_path / "prod" / "btc_1h" / "results"
    exp_results.mkdir(parents=True)
    prod_results.mkdir(parents=True)
    (exp_results / "run_manifest_public.json").write_text(
        json.dumps(
            {
                "market_id": "btc_1h",
                "workflow_profile": "exhaustive_v1",
                "status": "completed",
                "selection_policy": "validation_only",
                "test_policy": "frozen_winner_only",
                "data_variation": "rolling_mean",
                "provider_used": "binance",
            }
        ),
        encoding="utf-8",
    )
    (exp_results / "candidates_public.json").write_text(
        json.dumps(
            [
                {"public_id": "V001", "status": "worked"},
                {"public_id": "V002", "status": "worked"},
            ]
        ),
        encoding="utf-8",
    )
    phase_rows = [
        {
            "experiment_id": "E_abc",
            "phase_id": "phase01",
            "axis": "target_horizon",
            "status": "worked",
            "selected_for_next_phase": True,
            "train": {"balanced_accuracy": 0.71},
            "validation": {
                "balanced_accuracy": 0.68,
                "direction_accuracy": 0.67,
                "mcc": 0.31,
                "weighted_score": 0.72,
            },
            "test": {"balanced_accuracy": 0.66},
            "runtime_seconds": 12.5,
            "overfit_diagnostics": {"train_valid_gap": 0.03},
        }
    ]
    for index, direction_accuracy in enumerate((0.74, 0.73, 0.72, 0.71, 0.70), start=1):
        phase_rows.append(
            {
                "experiment_id": f"E_lock_{index}",
                "phase_id": "phase16",
                "status": "worked",
                "selected_for_next_phase": True,
                "train": {"balanced_accuracy": 0.70},
                "validation": {"balanced_accuracy": 0.69, "direction_accuracy": direction_accuracy},
                "test": {"balanced_accuracy": 0.68, "direction_accuracy": 0.67},
                "post_lock_sliding_summary": {
                    "window_count": 10,
                    "balanced_accuracy": {"min": 0.60, "avg": 0.65, "max": 0.70},
                    "direction_accuracy": {"min": 0.61, "avg": direction_accuracy, "max": 0.75},
                    "mcc": {"min": 0.10, "avg": 0.20, "max": 0.30},
                    "weighted_score": {"min": 0.50, "avg": 0.60, "max": 0.70},
                    "long_run": {"balanced_accuracy": 0.66, "direction_accuracy": 0.67},
                },
            }
        )
    (exp_results / "phase_results_public.json").write_text(json.dumps(phase_rows), encoding="utf-8")
    (prod_results / "production_public.json").write_text(
        json.dumps(
            {
                "market_id": "btc_1h",
                "status": "simulated",
                "public_delay_hours": 24,
                "latest_public_window": "2026-06-17T00:00:00+00:00",
                "generated_at_utc": "2026-06-18T00:00:00+00:00",
                "winner": "V001",
                "delayed_metrics": {
                    "weighted_score": 0.73,
                    "direction_accuracy": 0.66,
                    "balanced_accuracy": 0.65,
                    "mcc": 0.29,
                    "max_drawdown": -0.02,
                },
                "prediction_series": [
                    {"timestamp": "2026-06-15T00:00:00+00:00", "performance": 0.0},
                    {"timestamp": "2026-06-16T00:00:00+00:00", "performance": 0.02},
                ],
                "top_k_requested": 3,
                "top_k_available": 1,
                "top_models": [
                        {
                            "rank": 1,
                            "public_id": "V001",
                            "current_version": 1,
                            "train": {"balanced_accuracy": 0.71},
                            "validation": {"balanced_accuracy": 0.68},
                            "delayed_metrics": {"weighted_score": 0.73, "direction_accuracy": 0.66},
                        "prediction_series": [
                            {"timestamp": "2026-06-15T00:00:00+00:00", "performance": 0.0},
                                {"timestamp": "2026-06-16T00:00:00+00:00", "performance": 0.02},
                            ],
                            "versions": [
                                {
                                    "version": 1,
                                    "activated_at_utc": "2026-06-15T00:00:00+00:00",
                                    "deactivated_at_utc": None,
                                    "train": {"balanced_accuracy": 0.71},
                                    "validation": {"balanced_accuracy": 0.68},
                                    "production_metrics": {
                                        "balanced_accuracy": 0.65,
                                        "direction_accuracy": 0.66,
                                    },
                                    "prediction_series": [
                                        {"timestamp": "2026-06-15T00:00:00+00:00", "performance": 0.0},
                                        {"timestamp": "2026-06-16T00:00:00+00:00", "performance": 0.02},
                                    ],
                                }
                            ],
                        }
                ],
            }
        ),
        encoding="utf-8",
    )

    paths = update_pages(tmp_path)

    assert {path.name for path in paths} == {
        "index.html",
        "index.md",
        "experiments.html",
        "experiments.md",
        "prod.html",
        "prod.md",
    }
    experiments = (tmp_path / "docs" / "experiments.md").read_text(encoding="utf-8")
    experiments_html = (tmp_path / "docs" / "experiments.html").read_text(encoding="utf-8")
    prod = (tmp_path / "docs" / "prod.md").read_text(encoding="utf-8")
    index = (tmp_path / "docs" / "index.md").read_text(encoding="utf-8")
    index_html = (tmp_path / "docs" / "index.html").read_text(encoding="utf-8")
    assert "Experiment markets: 1" in index
    assert "<!doctype html>" in index_html
    assert 'href="experiments.html"' in index_html
    assert 'class="kpis"' in index_html
    assert 'id="theme-toggle"' in index_html
    assert 'class="phase-toggle"' in experiments_html
    assert "Target horizon" not in experiments_html
    assert "target_horizon" not in experiments_html
    assert "v1" in (tmp_path / "docs" / "prod.html").read_text(encoding="utf-8")
    assert "Active From" in (tmp_path / "docs" / "prod.html").read_text(encoding="utf-8")
    assert "Valid Direction" in experiments_html
    assert "Test Direction" in experiments_html
    assert ">16<" in experiments_html
    assert ">A<" in experiments_html
    post_lock_section = experiments_html.split("Post-Lock Sliding Validation", 1)[1]
    assert ">A<" in post_lock_section and ">B<" in post_lock_section and ">C<" in post_lock_section
    assert ">D<" not in post_lock_section
    assert "E_abc" not in experiments_html
    assert "rolling_mean" not in experiments
    assert "rolling_mean" not in experiments_html
    assert "<table" in experiments_html
    assert "BTC 1H: 2/16" in experiments
    assert "BTC 1H: V001" in prod
    prod_html = (tmp_path / "docs" / "prod.html").read_text(encoding="utf-8")
    assert "Top K Models" in prod_html
    assert "Train BA" in prod_html and "Valid BA" in prod_html and "Production BA" in prod_html
    assert "0.7100" in prod_html and "0.6800" in prod_html
    assert "UTC date" in prod_html and "Performance" in prod_html
    assert "2026-05" not in prod_html
    assert "Delay Hours" not in prod_html and "simulated" not in prod_html
    assert "Provider" not in index_html and "binance" not in experiments_html
    for text in (index, index_html, experiments, experiments_html, prod):
        assert_public_safe_text(text)


def test_production_plot_ticks_follow_the_series_timestamp_range() -> None:
    html = _plot(
        [
            (
                "V001",
                [
                    {"timestamp": "2026-06-01T00:00:00+00:00", "performance": 0.0},
                    {"timestamp": "2026-06-23T02:00:00+00:00", "performance": 8.0},
                ],
            )
        ]
    )

    assert "2026-06-01" in html
    assert "2026-06-23" in html
    assert "2026-05" not in html


def test_update_pages_empty_artifacts_write_public_safe_placeholders(tmp_path: Path) -> None:
    update_pages(tmp_path)

    for relative in (
        "docs/index.html",
        "docs/index.md",
        "docs/experiments.html",
        "docs/experiments.md",
        "docs/prod.html",
        "docs/prod.md",
    ):
        text = (tmp_path / relative).read_text(encoding="utf-8")
        assert_public_safe_text(text)
    experiments_html = (tmp_path / "docs" / "experiments.html").read_text(encoding="utf-8")
    index_html = (tmp_path / "docs" / "index.html").read_text(encoding="utf-8")
    assert "No public runs yet" in (tmp_path / "docs" / "experiments.md").read_text(encoding="utf-8")
    assert "No public runs yet" in (tmp_path / "docs" / "prod.md").read_text(encoding="utf-8")
    assert 'class="kpis"' in index_html
    assert 'class="empty"' in experiments_html


def test_public_prediction_result_respects_cutoff_and_starts_at_zero() -> None:
    result = build_public_prediction_result(
        timestamps=[
            "2026-06-01T00:00:00+00:00",
            "2026-06-01T01:00:00+00:00",
            "2026-06-01T02:00:00+00:00",
        ],
        y_true=[1, 0, 1],
        predictions=[1, 1, 0],
        probabilities=[0.8, 0.7, 0.2],
        returns=[0.02, -0.01, 0.03],
        latest_public_window="2026-06-01T01:00:00+00:00",
    )

    assert result is not None
    assert result["prediction_series"][0]["performance"] == 0.0
    assert [row["performance"] for row in result["prediction_series"]] == [0.0, 1.0, 0.0]
    assert len(result["prediction_series"]) == 3
    assert max(row["timestamp"] for row in result["prediction_series"]) <= "2026-06-01T01:00:00+00:00"
    assert result["metrics"]["trade_count"] == 2.0
    assert result["metrics"]["cumulative_return"] == 0.0


def test_public_prediction_result_respects_start_window() -> None:
    result = build_public_prediction_result(
        timestamps=[
            "2026-05-31T23:00:00+00:00",
            "2026-06-01T00:00:00+00:00",
            "2026-06-01T01:00:00+00:00",
        ],
        y_true=[1, 1, 0],
        predictions=[0, 1, 0],
        probabilities=[0.2, 0.8, 0.7],
        returns=[0.0, 0.02, -0.01],
        earliest_public_window="2026-06-01T00:00:00+00:00",
        latest_public_window="2026-06-01T01:00:00+00:00",
    )

    assert result is not None
    assert result["prediction_series"][0]["timestamp"] == "2026-06-01T00:00:00+00:00"
    assert min(row["timestamp"] for row in result["prediction_series"]) >= "2026-06-01T00:00:00+00:00"
    assert result["metrics"]["trade_count"] == 2.0


def test_public_prediction_baseline_is_exact_production_start_before_sequence_output() -> None:
    result = build_public_prediction_result(
        timestamps=["2026-06-01T23:00:00+00:00", "2026-06-02T00:00:00+00:00"],
        y_true=[1, 0],
        predictions=[1, 1],
        probabilities=[0.8, 0.7],
        returns=[0.01, -0.01],
        earliest_public_window="2026-06-01T00:00:00+00:00",
        latest_public_window="2026-06-02T00:00:00+00:00",
    )

    assert result is not None
    assert result["prediction_series"][0] == {
        "timestamp": "2026-06-01T00:00:00+00:00",
        "performance": 0.0,
    }
    assert result["prediction_series"][1]["timestamp"] == "2026-06-01T23:00:00+00:00"


def test_top_k_ids_use_only_validation_ranking_order() -> None:
    candidates = [
        {
            "candidate_id": "model_b",
            "status": "worked",
            "validation_metrics": {"balanced_accuracy": 0.72},
            "test": {"balanced_accuracy": 0.1},
        },
        {
            "candidate_id": "model_a",
            "status": "worked",
            "validation_metrics": {"balanced_accuracy": 0.70},
            "test": {"balanced_accuracy": 0.99},
        },
    ]
    changed_test = [
        {**row, "test": {"balanced_accuracy": 1.0 - row["test"]["balanced_accuracy"]}}
        for row in candidates
    ]

    ranked = rank_validation_candidates(candidates, "balanced_accuracy")
    reranked = rank_validation_candidates(changed_test, "balanced_accuracy")
    assert select_top_candidate_ids(ranked, 3) == ["model_b", "model_a"]
    assert select_top_candidate_ids(reranked, 3) == ["model_b", "model_a"]


def test_hf_sync_discovers_top_k_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = {
        "models": [
            {"rank": 1, "public_id": "V001", "directory": "top_k/01-V001"},
        ]
    }
    bundle = {"model": "model.joblib", "scaler": "scaler.joblib"}

    def fake_download(**kwargs):
        local_path = kwargs["local_path"]
        candidate = kwargs["candidates"][0]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if candidate.endswith("top_models_private.json"):
            local_path.write_text(json.dumps(index), encoding="utf-8")
        elif candidate.endswith("production_bundle.json"):
            local_path.write_text(json.dumps(bundle), encoding="utf-8")
        else:
            local_path.write_bytes(b"artifact")
        return {"status": "downloaded", "local_path": str(local_path)}

    monkeypatch.setattr(sync_hf_artifacts, "_download_first_available", fake_download)
    results = sync_hf_artifacts._pull_prod_bundle(
        repo_id="owner/repo",
        repo_type="model",
        token="token",
        market_id="btc_1h",
        output_dir=tmp_path,
    )

    top_dir = tmp_path / "btc_1h" / "top_k" / "01-V001"
    assert results
    assert (top_dir / "production_bundle.json").exists()
    assert (top_dir / "model.joblib").exists()
    assert (top_dir / "scaler.joblib").exists()


def test_reset_market_dry_run_deletes_nothing(tmp_path: Path) -> None:
    for path in local_reset_paths(tmp_path, market="btc_1h", cache_root=tmp_path / "private_hf_cache"):
        path.mkdir(parents=True, exist_ok=True)

    result = reset_market(
        root=tmp_path,
        cache_root=tmp_path / "private_hf_cache",
        market="btc_1h",
        client=None,
        dry_run=True,
        force=False,
        local_only=True,
    )

    assert result["deleted_local_paths"] == []
    assert all(Path(path).exists() for path in result["local_paths"])


def test_reset_market_force_deletes_only_matching_local_market(tmp_path: Path) -> None:
    for market in ("btc_1h", "eth_1h"):
        for path in local_reset_paths(tmp_path, market=market, cache_root=tmp_path / "private_hf_cache"):
            path.mkdir(parents=True, exist_ok=True)
            (path / "state.txt").write_text(market, encoding="utf-8")

    result = reset_market(
        root=tmp_path,
        cache_root=tmp_path / "private_hf_cache",
        market="btc_1h",
        client=None,
        dry_run=False,
        force=True,
        local_only=True,
    )

    assert result["deleted_local_paths"]
    assert not (tmp_path / "experiments" / "btc_1h").exists()
    assert (tmp_path / "experiments" / "eth_1h").exists()
    assert (tmp_path / "docs" / "index.html").exists()


def test_hf_reset_paths_only_targets_matching_market(tmp_path: Path) -> None:
    class FakeClient:
        def list_files(self, prefix: str = "") -> list[str]:
            return [
                "requests/configs/shared.yaml",
                "requests/configs/other.yaml",
                "requests/phase0/btc_1h_phase0_request_01.yaml",
                "requests/phase0/eth_1h_phase0_request_01.yaml",
                "markets/btc_1h.yaml",
                "markets/btc_1h/results/production_public.json",
                "markets/eth_1h.yaml",
                "models/btc_1h/model.json",
                "models/eth_1h/model.json",
                "live_data/btc_1h/1h.parquet",
                "live_data/eth_1h/1h.parquet",
            ]

        def download_file(self, hf_path: str, local_path: Path) -> Path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if hf_path == "requests/configs/shared.yaml":
                local_path.write_text(dump_yaml(_base_config("btc_1h", "0")), encoding="utf-8")
            else:
                local_path.write_text(dump_yaml(_base_config("eth_1h", "0")), encoding="utf-8")
            return local_path

    paths = hf_reset_paths(FakeClient(), market="btc_1h", scratch_root=tmp_path / "scan")

    assert paths == [
        "live_data/btc_1h/1h.parquet",
        "markets/btc_1h.yaml",
        "markets/btc_1h/results/production_public.json",
        "models/btc_1h/model.json",
        "requests/configs/shared.yaml",
        "requests/phase0/btc_1h_phase0_request_01.yaml",
    ]


def test_reset_all_outputs_dry_run_deletes_nothing(tmp_path: Path) -> None:
    for path in local_output_paths(tmp_path, cache_root=tmp_path / "private_hf_cache"):
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.txt").write_text("generated", encoding="utf-8")

    result = reset_all_outputs(
        root=tmp_path,
        cache_root=tmp_path / "private_hf_cache",
        client=None,
        dry_run=True,
        force=False,
        local_only=True,
    )

    assert result["market"] == "all"
    assert result["deleted_local_paths"] == []
    assert all(Path(path).exists() for path in result["local_paths"])


def test_reset_all_outputs_force_deletes_generated_roots_and_regenerates_pages(tmp_path: Path) -> None:
    for path in local_output_paths(tmp_path, cache_root=tmp_path / "private_hf_cache"):
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.txt").write_text("generated", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.example.yaml").write_text("project:\n  name: keep\n", encoding="utf-8")

    result = reset_all_outputs(
        root=tmp_path,
        cache_root=tmp_path / "private_hf_cache",
        client=None,
        dry_run=False,
        force=True,
        local_only=True,
    )

    assert result["deleted_local_paths"]
    assert not (tmp_path / "experiments").exists()
    assert not (tmp_path / "prod").exists()
    assert (tmp_path / "config" / "config.example.yaml").exists()
    assert (tmp_path / "docs" / "index.html").exists()


def test_reset_all_outputs_batches_hf_deletes() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.deleted_batches = []

        def delete_files(self, paths, *, commit_message):
            self.deleted_batches.append((list(paths), commit_message))

    client = FakeClient()

    deleted = delete_hf_outputs(client, paths=["requests/config.yaml", "models/btc_1h/model.json"], market="btc_1h")

    assert deleted == ["requests/config.yaml", "models/btc_1h/model.json"]
    assert client.deleted_batches == [
        (["requests/config.yaml", "models/btc_1h/model.json"], "Reset btc_1h: delete generated outputs")
    ]


def test_hf_output_paths_targets_all_generated_state_or_one_market() -> None:
    class FakeClient:
        def list_files(self, prefix: str = "") -> list[str]:
            return [
                "src/private/training/pipeline.py",
                "requests/configs/btc.yaml",
                "requests/phase0/btc_1h_phase0_request_01.yaml",
                "requests/phase0/eth_1h_phase0_request_01.yaml",
                "markets/btc_1h.yaml",
                "markets/eth_1h.yaml",
                "models/btc_1h/model.json",
                "models/eth_1h/model.json",
                "live_data/btc_1h/1h.parquet",
                "live_data/eth_1h/1h.parquet",
            ]

    assert hf_output_paths(FakeClient()) == [
        "live_data/btc_1h/1h.parquet",
        "live_data/eth_1h/1h.parquet",
        "markets/btc_1h.yaml",
        "markets/eth_1h.yaml",
        "models/btc_1h/model.json",
        "models/eth_1h/model.json",
        "requests/configs/btc.yaml",
        "requests/phase0/btc_1h_phase0_request_01.yaml",
        "requests/phase0/eth_1h_phase0_request_01.yaml",
    ]
    assert hf_output_paths(FakeClient(), market="btc_1h") == [
        "live_data/btc_1h/1h.parquet",
        "markets/btc_1h.yaml",
        "models/btc_1h/model.json",
        "requests/configs/btc.yaml",
        "requests/phase0/btc_1h_phase0_request_01.yaml",
    ]


def test_reset_market_batches_hf_deletes() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.deleted_batches = []

        def delete_files(self, paths, *, commit_message):
            self.deleted_batches.append((list(paths), commit_message))

    client = FakeClient()

    deleted = delete_hf_paths(client, market="btc_1h", paths=["models/btc_1h/model.json", "markets/btc_1h.yaml"])

    assert deleted == ["models/btc_1h/model.json", "markets/btc_1h.yaml"]
    assert client.deleted_batches == [
        (["models/btc_1h/model.json", "markets/btc_1h.yaml"], "Reset btc_1h: delete generated outputs")
    ]


def test_experiment_workflow_writes_result_file_for_action_chaining(tmp_path: Path, monkeypatch) -> None:
    result_file = tmp_path / "result.json"

    def fake_run_experiment_workflow(**kwargs):
        return {"status": "partial", "remaining_requests": ["btc_1h_phase0_request_02.yaml"]}

    monkeypatch.setattr(orchestrate_experiments, "run_experiment_workflow", fake_run_experiment_workflow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments_workflow.py",
            "--result-file",
            str(result_file),
            "--repo-id",
            "repo",
            "--token",
            "token",
        ],
    )

    assert orchestrate_experiments.main() == 0
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["remaining_requests"] == ["btc_1h_phase0_request_02.yaml"]


def test_git_push_rebases_and_retries_after_remote_update(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    class FakeCompleted:
        def __init__(self, returncode: int = 0, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command == ["git", "push"] and calls.count(["git", "push"]) == 1:
            return FakeCompleted(returncode=1)
        if command == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return FakeCompleted(stdout="main\n")
        return FakeCompleted()

    monkeypatch.setattr(hf_state.subprocess, "run", fake_run)

    result = hf_state._push_with_rebase_retry(tmp_path)

    assert result == {"push_attempts": 2, "rebased": True, "branch": "main"}
    assert calls == [
        ["git", "push"],
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        ["git", "fetch", "origin", "main"],
        ["git", "rebase", "-X", "theirs", "origin/main"],
        ["git", "push"],
    ]


def test_git_push_aborts_failed_rebase_retry(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    class FakeCompleted:
        def __init__(self, returncode: int = 0, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command == ["git", "push"]:
            return FakeCompleted(returncode=1)
        if command == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return FakeCompleted(stdout="main\n")
        if command == ["git", "rebase", "-X", "theirs", "origin/main"]:
            raise hf_state.subprocess.CalledProcessError(1, command)
        return FakeCompleted()

    monkeypatch.setattr(hf_state.subprocess, "run", fake_run)

    try:
        hf_state._push_with_rebase_retry(tmp_path)
        raise AssertionError("failed rebase should raise")
    except hf_state.subprocess.CalledProcessError:
        pass

    assert ["git", "rebase", "--abort"] in calls
