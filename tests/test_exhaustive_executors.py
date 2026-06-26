from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

pytest.importorskip("src.private.training.discovery_catalog", reason="private training source not hydrated from HF")

from src.private.training.discovery_catalog import PHASES
from src.private.training.discovery_executors import (
    add_regime_features,
    apply_robustness_stress,
    apply_sampling,
    apply_threshold,
    executor_signature,
    fit_calibration,
    objective_parameters,
    threshold_policy,
)
from src.private.training.sequence_models import SequenceDirectionModel, TorchSequenceDirectionModel, build_sequence_model
from src.private.training.trainer import prediction_arrays
from src.private.training.trainer import train_candidates
from src.private.training.ensembles import MixedSequenceEnsemble
from src.private.training.hf_artifacts import load_hf_artifacts, stage_hf_artifacts
from src.private.training.pipeline import (
    _apply_target_variation,
    _effective_decisions,
    _effective_hyperparameters,
    _feature_subset,
    _fixed_model_family_parent_contract,
    _ensemble_mechanisms_for_request,
    _hyperparameters_by_model_for_request,
    _locked_production_candidate_ids,
    _materialize_recipe_feature_columns,
    _model_ids_for_request,
    _parent_recipe_for_result,
    _production_public_id,
    _require_exact_top_k,
    _request_hyperparameters,
    _trainable_ids_for_recipe,
)
from src.private.training.sklearn_models import NumpyLogisticRegression, TorchResidualMLP
from src.public.data.scaling import fit_train_only_transform


def _frame(rows: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "return": returns,
            "rolling_return_std_24": pd.Series(returns).rolling(24, min_periods=2).std().fillna(0.0),
            "feature_a": rng.normal(size=rows),
            "feature_b": rng.normal(size=rows),
            "target": (returns > 0).astype(int),
        }
    )


def test_every_catalog_variation_has_a_concrete_executor_route() -> None:
    signatures = []
    for phase in PHASES:
        for variation in phase.variations:
            signatures.append(executor_signature(phase.axis, variation.variation_id))
    assert len(signatures) == sum(len(phase.variations) for phase in PHASES)
    assert len(set(signatures)) == len(signatures)


def test_torch_sequence_training_is_reproducible_for_same_seed() -> None:
    frame = _frame(96)
    params = {
        "sequence_length": 24,
        "seed": 42,
        "hidden_dim": 16,
        "batch_size": 16,
        "epochs": 4,
        "strict_backend": True,
    }

    first = build_sequence_model("lstm", dict(params)).fit_sequence(frame, ["feature_a"])
    second = build_sequence_model("lstm", dict(params)).fit_sequence(frame, ["feature_a"])

    first_probability, first_indices = first.predict_proba_sequence(frame)
    second_probability, second_indices = second.predict_proba_sequence(frame)
    assert np.array_equal(first_indices, second_indices)
    assert np.allclose(first_probability, second_probability, atol=0.0)
    assert np.array_equal(first.predict_sequence(frame)[0], second.predict_sequence(frame)[0])


def test_materialize_recipe_feature_columns_recreates_context_features() -> None:
    frame = pd.DataFrame({"return": [1.0, 2.0, 3.0], "target": [0, 1, 1]})
    output, columns = _materialize_recipe_feature_columns(frame, ["return"], ["return", "return_context_2"])

    assert "return_context_2" in columns
    assert output["return_context_2"].tolist() == [1.0, 1.5, 2.5]


def test_ensemble_recipe_trainable_ids_use_family_members_before_registry() -> None:
    recipe = {
        "candidate_id": "stacked_logistic",
        "family_recipes": [
            {"candidate_id": "xgboost"},
            {"candidate_id": "transformer_lstm"},
            {"candidate_id": "residual_mlp"},
        ],
    }
    registry = {
        "stacked_logistic": object(),
        "xgboost": object(),
        "transformer_lstm": object(),
        "residual_mlp": object(),
    }

    assert _trainable_ids_for_recipe(recipe, registry) == ["xgboost", "transformer_lstm", "residual_mlp"]


def test_sequence_validation_uses_context_without_shifting_target_rows() -> None:
    frame = _frame(120)
    warmup = frame.iloc[:24].reset_index(drop=True)
    train = frame.iloc[24:84].reset_index(drop=True)
    validation = frame.iloc[84:120].reset_index(drop=True)

    candidates = train_candidates(
        model_ids=["lstm"],
        train=train,
        validation=validation,
        feature_columns=["feature_a"],
        ranking_metric="direction_accuracy",
        sequence_length=24,
        hyperparameters={"seed": 42, "epochs": 3, "batch_size": 16, "hidden_dim": 16, "strict_backend": True},
        sequence_context=warmup,
        validation_sequence_context=train,
    )

    assert candidates[0].status == "worked"
    assert candidates[0].row_offset == 0
    assert candidates[0].validation_metrics["trade_count"] == len(validation)


def test_production_public_ids_are_letters_and_top_k_is_exact() -> None:
    assert [_production_public_id(rank) for rank in (1, 2, 3, 27)] == ["A", "B", "C", "AA"]
    _require_exact_top_k(phase="prod", available=3, requested=3)
    with pytest.raises(RuntimeError, match="exactly 3 Top K models"):
        _require_exact_top_k(phase="prod", available=1, requested=3)
    _require_exact_top_k(phase="phase16", available=1, requested=3)


def test_prod_uses_all_locked_ensemble_finalists() -> None:
    config = {
        "workflow": {"workflow_profile": "exhaustive_v1"},
        "discovery_state": {
            "selected_recipes": [
                {"candidate_id": "confidence_weighted"},
                {"candidate_id": "rank_weighted"},
                {"candidate_id": "stacked_logistic"},
            ]
        },
    }
    assert _locked_production_candidate_ids(config) == [
        "confidence_weighted",
        "rank_weighted",
        "stacked_logistic",
    ]
    assert _ensemble_mechanisms_for_request(config, {}, exhaustive=True, phase="prod") == [
        "confidence_weighted",
        "rank_weighted",
        "stacked_logistic",
    ]


def test_fixed_phase6_default_reuses_phase5_resolved_hyperparameters() -> None:
    parent_parameters = {
        "seed": 42,
        "hidden_dim": 37,
        "batch_size": 16,
        "epochs": 23,
        "strict_backend": True,
        "dropout": 0.17,
    }
    config = {
        "project": {"default_seed": 42},
        "experiments": {"phase5_to_phase6_inheritance": "fixed"},
        "workflow": {
            "workflow_profile": "exhaustive_v1",
            "axis": "architecture",
            "variation_id": "default",
            "parent_recipe": {
                "candidate_id": "lstm",
                "recipe_hash": "phase5-lstm",
                "decisions": {},
                "resolved_hyperparameters": parent_parameters,
            },
        },
    }

    decisions = _effective_decisions(config)
    assert decisions["architecture"]["parameters"] == {}
    assert _effective_hyperparameters(config, decisions) == parent_parameters

    config["workflow"]["variation_id"] = "hidden_size"
    changed = _effective_hyperparameters(config, _effective_decisions(config))
    assert changed["hidden_dim"] == 128
    assert {key: value for key, value in changed.items() if key != "hidden_dim"} == {
        key: value for key, value in parent_parameters.items() if key != "hidden_dim"
    }


def test_fixed_phase5_model_family_reuses_phase4_resolved_hyperparameters() -> None:
    parent_parameters = {
        "seed": 42,
        "hidden_dim": 37,
        "batch_size": 16,
        "epochs": 23,
        "strict_backend": True,
        "dropout": 0.17,
    }
    config = {
        "project": {"default_seed": 42},
        "experiments": {"phase5_to_phase6_inheritance": "fixed"},
        "workflow": {
            "workflow_profile": "exhaustive_v1",
            "axis": "model_family",
            "variation_id": "lstm",
            "parent_recipe": {
                "candidate_id": "logistic_regression",
                "recipe_hash": "phase4-winner",
                "decisions": {},
                "resolved_hyperparameters": parent_parameters,
            },
        },
    }

    decisions = _effective_decisions(config)
    assert decisions["model_family"] == {"variation_id": "lstm", "parameters": {}}
    assert _effective_hyperparameters(config, decisions) == parent_parameters


def test_fixed_phase5_uses_phase4_recipe_for_every_family_variation() -> None:
    parent_parameters = {
        "seed": 42,
        "hidden_dim": 37,
        "batch_size": 16,
        "epochs": 23,
        "strict_backend": True,
        "dropout": 0.17,
    }
    config = {
        "project": {"default_seed": 42},
        "experiments": {"phase5_to_phase6_inheritance": "fixed"},
        "workflow": {
            "workflow_profile": "exhaustive_v1",
            "axis": "model_family",
            "variation_id": "lstm",
            "parent_recipe": {
                "candidate_id": "lstm",
                "recipe_hash": "phase4-winner",
                "decisions": {},
                "resolved_hyperparameters": parent_parameters,
                "resolved_data_contract": {
                    "feature_columns": ["log_return"],
                    "scaling_transform": "robust",
                    "sequence_length": 48,
                },
            },
        },
    }
    request_params = _effective_hyperparameters(config, _effective_decisions(config))
    by_model = _hyperparameters_by_model_for_request(
        config,
        ["lstm", "random_forest"],
        request_params,
        _effective_decisions(config),
    )

    assert by_model["lstm"] == parent_parameters
    assert by_model["random_forest"] == parent_parameters
    assert _fixed_model_family_parent_contract(config) == {
        "feature_columns": ["log_return"],
        "scaling_transform": "robust",
        "sequence_length": 48,
    }


def test_sampling_regime_calibration_threshold_and_robustness_execute() -> None:
    frame = _frame()
    sampling_variations = [variation.variation_id for variation in PHASES[8].variations]
    sampled_fingerprints = set()
    for variation in sampling_variations:
        sampled, metadata = apply_sampling(frame, variation, seed=42)
        assert len(sampled) == len(frame)
        sampled_fingerprints.add((variation, float(sampled["feature_a"].sum()), bool(metadata)))
    assert len(sampled_fingerprints) == len(sampling_variations)

    for variation in [item.variation_id for item in PHASES[10].variations]:
        transformed, columns = add_regime_features(frame, variation)
        assert len(transformed) == len(frame)
        assert variation == "none" or columns

    y = frame["target"].to_numpy(dtype=int)
    probability = np.clip(0.5 + frame["feature_a"].to_numpy() * 0.1, 0.01, 0.99)
    for variation in [item.variation_id for item in PHASES[11].variations]:
        calibration = fit_calibration(variation, y, probability, add_regime_features(frame, "regime_feature")[0])
        calibrated = calibration.apply(probability, add_regime_features(frame, "regime_feature")[0])
        assert calibrated.shape == probability.shape
        assert np.isfinite(calibrated).all()

    for variation in [item.variation_id for item in PHASES[12].variations]:
        policy = threshold_policy(variation, y, probability, frame["return"].to_numpy())
        prediction = apply_threshold(policy, probability, add_regime_features(frame, "regime_feature")[0])
        assert prediction.shape == y.shape
        assert set(np.unique(prediction)).issubset({0, 1})

    for variation in [item.variation_id for item in PHASES[14].variations]:
        stressed = apply_robustness_stress(frame, variation, seed=42)
        assert not stressed.empty


def test_every_objective_has_distinct_parameters() -> None:
    variations = [item.variation_id for item in PHASES[7].variations]
    payloads = [objective_parameters(variation) for variation in variations]
    assert len({repr(sorted(payload.items())) for payload in payloads}) == len(payloads)


def test_robustness_reuses_only_registry_model_ids() -> None:
    config = {
        "workflow": {
            "workflow_profile": "exhaustive_v1",
            "axis": "robustness",
            "parent_recipe": {
                "candidate_id": "weighted_vote",
                "family_recipes": [
                    {"candidate_id": "weighted_vote"},
                    {"candidate_id": "lstm"},
                    {"candidate_id": "regime_gated"},
                ],
            },
        },
        "experiments": {},
    }

    assert _model_ids_for_request(config) == ["lstm"]


def test_all_target_feature_transform_and_search_presets_are_executable_and_distinct() -> None:
    frame = _frame(320)
    frame["close"] = 100.0 * (1.0 + frame["return"]).cumprod()
    target_outputs = []
    for item in PHASES[0].variations:
        transformed = _apply_target_variation(frame, item.variation_id)
        assert not transformed.empty
        assert set(transformed["target"].unique()).issubset({0, 1})
        target_outputs.append((item.variation_id, len(transformed), int(transformed["target"].sum())))
    assert len({output[0] for output in target_outputs}) == len(PHASES[0].variations)

    available = [
        "open", "high", "low", "close", "volume", "return", "log_return", "hl_range", "oc_body",
        "upper_wick", "lower_wick", "rolling_return_mean_6", "rolling_return_std_6",
        "rolling_return_mean_24", "rolling_return_std_24", "volume_change", "volume_z_24", "hour",
        "dayofweek", "eth_return", "sol_return",
    ]
    feature_sets = {}
    for item in PHASES[1].variations:
        selected = _feature_subset(available, item.variation_id)
        assert selected
        assert len(selected) == len(set(selected))
        feature_sets[item.variation_id] = tuple(selected)
    assert len(set(feature_sets.values())) == len(PHASES[1].variations)

    transform_frame = frame.assign(feature_a=np.linspace(-2, 3, len(frame)), feature_b=np.sin(np.arange(len(frame))))
    for item in PHASES[2].variations:
        transform = fit_train_only_transform(transform_frame, ["feature_a", "feature_b"], item.variation_id)
        output = transform.transform(transform_frame)
        assert np.isfinite(output[["feature_a", "feature_b"]].to_numpy()).all()
        assert transform.metadata()["method"] == item.variation_id

    for phase in (PHASES[5], PHASES[6]):
        parameter_sets = []
        assert phase.variations[0].variation_id == "default"
        for item in phase.variations:
            config = {
                "project": {"default_seed": 42},
                "workflow": {
                    "workflow_profile": "exhaustive_v1",
                    "axis": phase.axis,
                    "variation_id": item.variation_id,
                    "seed": 42,
                },
            }
            parameter_sets.append(repr(sorted(_request_hyperparameters(config).items())))
        assert len(set(parameter_sets)) == len(parameter_sets)


def test_every_sequence_family_uses_a_real_torch_fit_predict_cycle() -> None:
    frame = _frame(28)
    feature_columns = ["feature_a", "feature_b"]
    families = ["lstm", "gru", "transformer", "tcn", "temporal_cnn", "patchtst", "mamba", "state_space", "cnn_lstm", "transformer_lstm"]
    module_types = set()
    for family in families:
        model = build_sequence_model(
            family,
            {
                "strict_backend": True,
                "sequence_length": 4,
                "hidden_dim": 8,
                "attention_heads": 2,
                "epochs": 1,
                "seed": 42,
            },
        )
        assert isinstance(model, TorchSequenceDirectionModel)
        model.fit_sequence(frame, feature_columns)
        probability, indices = model.predict_proba_sequence(frame)
        assert len(probability) == len(indices) == len(frame) - 3
        assert np.isfinite(probability).all()
        module_types.add(type(model.module.base).__qualname__)
    assert len(module_types) >= 6


def test_torch_sequence_short_evaluation_returns_no_windows_cleanly() -> None:
    frame = _frame(12)
    model = build_sequence_model(
        "lstm",
        {"strict_backend": True, "sequence_length": 4, "hidden_dim": 8, "epochs": 1, "seed": 42},
    )
    model.fit_sequence(frame, ["feature_a", "feature_b"])

    probability, indices = model.predict_proba_sequence(frame.iloc[:3])

    assert probability.shape == (0,)
    assert indices.shape == (0,)


def test_sequence_regime_slice_can_reuse_full_validation_context() -> None:
    frame = _frame(36)
    model = build_sequence_model(
        "lstm",
        {"strict_backend": True, "sequence_length": 12, "hidden_dim": 8, "epochs": 1, "seed": 42},
    )
    model.fit_sequence(frame, ["feature_a", "feature_b"])
    stressed = apply_robustness_stress(frame, "bull_slice", seed=42)

    probability, indices = model.predict_proba_sequence(frame)
    aligned = frame.iloc[indices].reset_index(drop=True)
    selected = aligned["timestamp"].isin(stressed["timestamp"]).to_numpy()

    assert selected.any()
    assert len(probability[selected]) > 0


def test_sequence_warmup_provides_context_without_fitting_warmup_labels() -> None:
    frame = _frame(12)
    model = SequenceDirectionModel("sequence", sequence_length=4).fit_sequence(
        frame,
        ["feature_a", "feature_b"],
        fit_start_index=4,
    )

    assert model.row_indices_ is not None
    assert model.row_indices_[0] == 4


def test_production_sequence_context_predicts_the_first_production_row() -> None:
    frame = _frame(40)
    model = SequenceDirectionModel("sequence", sequence_length=24).fit_sequence(
        frame.iloc[:32].reset_index(drop=True),
        ["feature_a", "feature_b"],
    )
    context = frame.iloc[8:32].reset_index(drop=True)
    production = frame.iloc[32:].reset_index(drop=True)

    y_true, probability, returns, row_offset = prediction_arrays(
        model,
        production,
        ["feature_a", "feature_b"],
        sequence_context=context,
    )

    assert row_offset == 0
    assert len(probability) == len(production)
    assert y_true[0] == production["target"].iloc[0]
    assert returns[0] == production["return"].iloc[0]
    assert production["timestamp"].iloc[0] == frame["timestamp"].iloc[32]


def test_carried_recipe_decisions_are_replayed_with_family_specific_training() -> None:
    config = {
        "project": {"default_seed": 42},
        "workflow": {
            "workflow_profile": "exhaustive_v1",
            "axis": "objective_loss",
            "variation_id": "focal",
            "parent_recipes": [
                {
                    "candidate_id": "lstm",
                    "decisions": {
                        "target_horizon": {"variation_id": "next_6", "parameters": {}},
                        "architecture": {"variation_id": "dropout", "parameters": {"architecture_search": True, "dropout": 0.35}},
                        "training_hyperparams": {"variation_id": "seed_43", "parameters": {"training_search": True, "seed": 43}},
                    },
                }
            ],
        },
    }
    decisions = _effective_decisions(config)
    params = _effective_hyperparameters(config, decisions)
    assert decisions["target_horizon"]["variation_id"] == "next_6"
    assert decisions["objective_loss"]["variation_id"] == "focal"
    assert params["dropout"] == 0.35
    assert params["seed"] == 43
    assert params["loss"] == "focal"


def test_result_recipe_does_not_inherit_mismatched_model_family_parent() -> None:
    inherited = _parent_recipe_for_result(
        {"axis": "model_family", "parent_recipes": [{"candidate_id": "logistic_regression"}]},
        winner_id="catboost",
    )

    assert inherited == {}


def test_explicit_lock_recipe_is_not_replaced_by_matching_family_recipe() -> None:
    locked = {
        "candidate_id": "weighted_vote",
        "family_recipes": [{"candidate_id": "catboost"}],
        "decisions": {"ensemble": {"variation_id": "weighted_vote", "parameters": {}}},
    }
    family = {"candidate_id": "catboost", "decisions": {"model_family": {"variation_id": "catboost"}}}

    inherited = _parent_recipe_for_result(
        {"axis": "final_validation_lock", "parent_recipe": locked, "parent_recipes": [family]},
        winner_id="catboost",
    )

    assert inherited == locked


def test_mixed_ensemble_keeps_sequence_and_tabular_members() -> None:
    frame = _frame(40)
    columns = ["feature_a", "feature_b"]
    tabular = NumpyLogisticRegression(epochs=5).fit(frame[columns], frame["target"])
    sequence = build_sequence_model(
        "lstm",
        {"strict_backend": True, "sequence_length": 4, "hidden_dim": 8, "epochs": 1, "seed": 42},
    ).fit_sequence(frame, columns)
    ensemble = MixedSequenceEnsemble("model_family", [tabular, sequence], "model_family", columns)
    ensemble.fit_sequence(frame, columns)
    probability, indices = ensemble.predict_proba_sequence(frame)
    assert len(probability) == len(indices) == len(frame) - 3
    assert len(ensemble.metadata()["members"]) == 2
    assert np.isfinite(probability).all()


def test_mixed_ensemble_with_torch_member_round_trips_joblib(tmp_path: Path) -> None:
    import joblib

    frame = _frame(40)
    columns = ["feature_a", "feature_b"]
    tabular = NumpyLogisticRegression(epochs=5).fit(frame[columns], frame["target"])
    sequence = build_sequence_model(
        "lstm",
        {"strict_backend": True, "sequence_length": 4, "hidden_dim": 8, "epochs": 1, "seed": 42},
    ).fit_sequence(frame, columns)
    ensemble = MixedSequenceEnsemble("model_family", [tabular, sequence], "model_family", columns)
    ensemble.fit_sequence(frame, columns)
    before, before_indices = ensemble.predict_proba_sequence(frame)
    path = tmp_path / "mixed_ensemble.joblib"
    joblib.dump(ensemble, path)
    loaded = joblib.load(path)
    after, after_indices = loaded.predict_proba_sequence(frame)
    assert np.array_equal(after_indices, before_indices)
    assert np.allclose(after, before)


def test_torch_residual_delegate_round_trips_joblib(tmp_path: Path) -> None:
    import joblib

    frame = _frame(32)
    columns = ["feature_a", "feature_b"]
    model = TorchResidualMLP(
        {"architecture_search": True, "hidden_dim": 8, "epochs": 1, "seed": 42},
        model_id="architecture_search_delegate",
    ).fit(frame[columns], frame["target"])
    before = model.predict_proba(frame[columns])
    path = tmp_path / "torch_residual_delegate.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)

    assert np.allclose(loaded.predict_proba(frame[columns]), before)


def test_torch_production_bundle_round_trips_real_weights(tmp_path: Path) -> None:
    frame = _frame(32)
    columns = ["feature_a", "feature_b"]
    model = build_sequence_model(
        "lstm",
        {"strict_backend": True, "sequence_length": 4, "hidden_dim": 8, "epochs": 1, "seed": 42},
    ).fit_sequence(frame, columns)
    scaler = fit_train_only_transform(frame, columns, "standard")
    before, before_indices = model.predict_proba_sequence(frame)
    stage_hf_artifacts(
        tmp_path,
        market_id="btc_1h",
        metadata={"recipe_hash": "recipe_test"},
        scaler_metadata=scaler.metadata(),
        model_object=model,
        scaler_object=scaler,
    )
    loaded, loaded_scaler, bundle = load_hf_artifacts(tmp_path)
    after, after_indices = loaded.predict_proba_sequence(frame)
    assert bundle["model_format"] == "torch_state_dict"
    assert loaded_scaler.metadata() == scaler.metadata()
    assert np.array_equal(after_indices, before_indices)
    assert np.allclose(after, before)


def test_residual_mlp_production_bundle_round_trips_real_weights(tmp_path: Path) -> None:
    frame = _frame(32)
    columns = ["feature_a", "feature_b"]
    model = TorchResidualMLP({"hidden_dim": 8, "epochs": 1, "seed": 42}).fit(frame[columns], frame["target"])
    scaler = fit_train_only_transform(frame, columns, "standard")
    before = model.predict_proba(frame[columns])
    stage_hf_artifacts(
        tmp_path,
        market_id="btc_1h",
        metadata={"recipe_hash": "recipe_test"},
        scaler_metadata=scaler.metadata(),
        model_object=model,
        scaler_object=scaler,
    )
    loaded, loaded_scaler, bundle = load_hf_artifacts(tmp_path)
    after = loaded.predict_proba(frame[columns])
    assert bundle["model_format"] == "torch_state_dict"
    assert loaded_scaler.metadata() == scaler.metadata()
    assert np.allclose(after, before)
