from __future__ import annotations

import json
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

try:
    from src.private.training.model_registry import default_model_registry
    from src.private.training.discovery_catalog import catalog_hash
    from src.private.training.pipeline import PipelineOptions, _top_k_public_result, _workflow_for_run, run_pipeline
    PRIVATE_TRAINING_AVAILABLE = True
except ModuleNotFoundError:
    default_model_registry = None
    catalog_hash = None
    PipelineOptions = None
    run_pipeline = None
    _workflow_for_run = None
    _top_k_public_result = None
    PRIVATE_TRAINING_AVAILABLE = False
from src.public.config.loader import load_config
from src.public.data.denoising import SUPPORTED_DENOISING_METHODS, causal_denoise_frame
from src.public.data.features import build_basic_features
from src.public.data.fetch import fetch_market_data, historical_context_rows, live_cache_path, max_target_horizon
from src.public.data.scaling import TrainOnlyStandardScaler
from src.public.data.split import FEATURE_WARMUP_ROWS, chronological_split, production_split
from src.public.data.validation import validate_ohlcv_schema
from src.public.evaluation.ranking import rank_validation_candidates
from src.public.evaluation.sanitizer import assert_public_safe_text, public_safety_violations
from src.public.reporting.public_audit import scan_public_files
from src.public.security.allowlist import is_public_git_allowed
from src.public.security.repo_hygiene import find_tracked_hygiene_violations

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config.example.yaml"


def _require_private_training() -> None:
    if not PRIVATE_TRAINING_AVAILABLE:
        pytest.skip("private training source not hydrated from HF")


def _ohlcv_frame(rows: int = 700, *, start: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    close = [100.0 + index * 0.1 for index in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000.0 + index for index in range(rows)],
        }
    )


def _write_live_cache(
    root: Path,
    config: dict,
    *,
    provider: str = "binance",
    rows: int | None = None,
    start: str = "2024-01-01",
) -> Path:
    if rows is None:
        split = config["split"]
        rows = (
            int(split["train_length"])
            + int(split["validation_length"])
            + int(split["test_length"])
            + historical_context_rows(config)
            + max_target_horizon(config)
            + 1
        )
    path = live_cache_path(
        root,
        market_id=str(config["market"]["market_id"]),
        timeframe=str(config["market"]["timeframe"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _ohlcv_frame(rows, start=start).to_parquet(path, index=False)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({"provider_used": provider, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return path


def _ccxt_rows(rows: int = 700, *, start: str = "2024-01-01") -> list[list[float]]:
    frame = _ohlcv_frame(rows, start=start)
    return [
        [
            int(row.timestamp.timestamp() * 1000),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        ]
        for row in frame.itertuples(index=False)
    ]


def test_config_example_loads_and_validates() -> None:
    config, config_hash = load_config(CONFIG)
    assert config["market"]["market_type"] == "crypto"
    assert config["project"]["public_delay_hours"] >= 24
    assert len(config_hash) == 64


def test_schema_split_and_train_only_scaler(tmp_path: Path) -> None:
    config, _ = load_config(CONFIG)
    _write_live_cache(tmp_path, config)
    fetch = fetch_market_data(config, root=tmp_path)
    validate_ohlcv_schema(fetch.frame)
    featured, columns = build_basic_features(fetch.frame)
    splits = chronological_split(featured, train_length=100, validation_length=40, test_length=40)
    assert splits.train["timestamp"].max() < splits.validation["timestamp"].min() < splits.test["timestamp"].min()
    assert splits.test["timestamp"].iloc[-1] == featured["timestamp"].iloc[-1]
    scaler = TrainOnlyStandardScaler.fit(splits.train, columns)
    assert scaler.metadata()["fit_split"] == "train"
    transformed_validation = scaler.transform(splits.validation)
    assert set(columns).issubset(transformed_validation.columns)


def test_experiment_split_is_chronological_and_uses_fetched_feature_warmup(tmp_path: Path) -> None:
    config, _ = load_config(CONFIG)
    _write_live_cache(tmp_path, config, rows=5000, start="2026-04-01")
    fetched = fetch_market_data(config, root=tmp_path).frame
    featured, _ = build_basic_features(fetched)

    split = config["split"]
    total = sum(int(split[key]) for key in ("train_length", "validation_length", "test_length"))
    expected_train_start = len(featured) - total
    expected_first_train_mean = featured["return"].iloc[expected_train_start - 5 : expected_train_start + 1].mean()
    splits = chronological_split(
        featured,
        train_length=int(split["train_length"]),
        validation_length=int(split["validation_length"]),
        test_length=int(split["test_length"]),
    )

    assert len(fetched) == total + historical_context_rows(config) + max_target_horizon(config) + 1
    assert splits.train["rolling_return_mean_6"].iloc[0] == pytest.approx(expected_first_train_mean)


def test_historical_context_uses_max_sequence_and_feature_windows() -> None:
    config, _ = load_config(CONFIG)
    config["experiments"]["workflow_profile"] = "legacy_v1"
    config["experiments"]["sequence_lengths"] = [12, 48]
    assert historical_context_rows(config) >= 48
    config["experiments"]["workflow_profile"] = "exhaustive_v1"
    assert historical_context_rows(config) >= 240


def test_production_split_uses_fixed_start_and_preproduction_history() -> None:
    frame = _ohlcv_frame(1000, start="2026-04-25")
    splits = production_split(
        frame,
        production_start="2026-06-01T00:00:00+00:00",
        train_length=500,
        validation_length=300,
    )

    assert splits.warmup is not None
    assert len(splits.warmup) == 64
    assert len(splits.train) == 500
    assert len(splits.validation) == 300
    assert splits.warmup["timestamp"].max() < splits.train["timestamp"].min()
    assert splits.test["timestamp"].iloc[0] == pd.Timestamp("2026-06-01T00:00:00+00:00")
    assert splits.train["timestamp"].max() < splits.validation["timestamp"].min()
    assert splits.validation["timestamp"].max() < splits.test["timestamp"].min()


def test_single_top_k_public_result_uses_candidate_predictions() -> None:
    _require_private_training()
    test = pd.DataFrame({"timestamp": pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC")})

    result = _top_k_public_result(
        test=test,
        top_candidate_ids=["model"],
        candidate_audits={
            "model": {
                "test_arrays": {
                    "y_true": [0, 0],
                    "probabilities": [0.8, 0.9],
                    "predictions": [0, 0],
                    "returns": [0.0, 0.0],
                    "row_offset": 0,
                }
            }
        },
        latest_public_window="2026-06-01T01:00:00+00:00",
    )

    assert result is not None
    assert result["metrics"]["direction_accuracy"] == 1.0


def test_causal_denoising_variations_do_not_use_future_rows(tmp_path: Path) -> None:
    config, _ = load_config(CONFIG)
    _write_live_cache(tmp_path, config)
    frame = fetch_market_data(config, root=tmp_path).frame.head(24)
    future_changed = frame.copy(deep=True)
    future_changed.loc[10, "close"] = 999999.0

    for method in SUPPORTED_DENOISING_METHODS:
        baseline = causal_denoise_frame(frame, method=method)
        changed = causal_denoise_frame(future_changed, method=method)
        assert baseline["close"].iloc[:10].tolist() == changed["close"].iloc[:10].tolist()
        assert baseline["timestamp"].iloc[10] == frame["timestamp"].iloc[10]


def test_input_transforms_do_not_change_raw_close_targets() -> None:
    raw = _ohlcv_frame(4)
    raw["close"] = [10.0, 11.0, 10.0, 12.0]
    transformed_inputs = raw.copy()
    transformed_inputs["close"] = [10.0, 9.0, 8.0, 7.0]

    featured, _ = build_basic_features(transformed_inputs, target_frame=raw)

    assert featured["target"].tolist() == [1, 0, 1]


def test_fetch_uses_cached_live_parquet_before_network(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)
    cache_path = _write_live_cache(tmp_path, config, provider="kraken")
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace())

    result = fetch_market_data(config, root=tmp_path)

    assert result.provider_used == "kraken"
    assert result.cache_status == "hit"
    assert result.cache_path == cache_path


def test_experiment_fetch_stops_at_fixed_end(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)
    cache_path = live_cache_path(tmp_path, market_id="btc_1h", timeframe="1h")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _ohlcv_frame(2000, start="2026-03-20").to_parquet(cache_path, index=False)
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace())

    result = fetch_market_data(
        config,
        root=tmp_path,
        end_utc="2026-05-31T23:59:59+00:00",
    )

    assert len(result.frame) == (
        sum(int(config["split"][key]) for key in ("train_length", "validation_length", "test_length"))
        + historical_context_rows(config)
        + max_target_horizon(config)
        + 1
    )
    assert result.frame["timestamp"].max() == pd.Timestamp("2026-05-31T23:00:00+00:00")


def test_production_fetch_appends_only_rows_after_cache(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)
    step = pd.Timedelta(hours=1)
    production_start = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=20)
    history_rows = historical_context_rows(config) + int(config["split"]["train_length"]) + int(config["split"]["validation_length"])
    required_start = production_start - step * history_rows
    cached = _ohlcv_frame(history_rows + 20, start=required_start.isoformat())
    cache_path = live_cache_path(tmp_path, market_id="btc_1h", timeframe="1h")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path, index=False)
    requested_since: list[int] = []

    class Binance:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params, since):
            requested_since.append(since)
            start = pd.Timestamp(since, unit="ms", tz="UTC")
            return _ccxt_rows(limit, start=start.isoformat())

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=Binance))

    result = fetch_market_data(
        config,
        root=tmp_path,
        production_start_utc=production_start.isoformat(),
    )

    assert requested_since == [int(cached["timestamp"].iloc[-1].timestamp() * 1000)]
    assert result.cache_status == "appended"
    assert result.frame["timestamp"].is_unique


def test_fetch_binance_spot_success_writes_private_cache(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)

    class Binance:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params):
            assert symbol == "BTC/USDT"
            return _ccxt_rows(limit)

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=Binance))

    result = fetch_market_data(config, root=tmp_path)

    assert result.provider_used == "binance"
    assert result.cache_status == "written"
    assert live_cache_path(tmp_path, market_id="btc_1h", timeframe="1h").exists()


def test_fetch_paginates_large_ccxt_history(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)
    calls: list[tuple[int, int]] = []

    class Binance:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params, since):
            calls.append((limit, since))
            start = pd.Timestamp(since, unit="ms", tz="UTC")
            return _ccxt_rows(min(limit, 1000), start=start.isoformat())

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=Binance))

    result = fetch_market_data(
        config,
        root=tmp_path,
        end_utc="2025-05-31T23:59:59+00:00",
        post_end_rows=3000,
    )

    assert len(calls) >= 5
    assert len(result.frame) == (
        int(config["split"]["train_length"])
        + int(config["split"]["validation_length"])
        + int(config["split"]["test_length"])
        + historical_context_rows(config)
        + max_target_horizon(config)
        + 1
        + 3000
    )
    assert result.frame["timestamp"].is_monotonic_increasing


def test_fetch_caps_post_end_rows_at_completed_live_window(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)
    calls: list[tuple[int, int]] = []

    class Binance:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params, since):
            calls.append((limit, since))
            start = pd.Timestamp(since, unit="ms", tz="UTC")
            return _ccxt_rows(min(limit, 1000), start=start.isoformat())

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=Binance))
    now = pd.Timestamp.now(tz="UTC").floor("h")
    end = now - pd.Timedelta(hours=48)

    result = fetch_market_data(
        config,
        root=tmp_path,
        end_utc=end.isoformat(),
        post_end_rows=3000,
    )

    assert len(result.frame) == (
        int(config["split"]["train_length"])
        + int(config["split"]["validation_length"])
        + int(config["split"]["test_length"])
        + historical_context_rows(config)
        + max_target_horizon(config)
        + 1
        + 48
    )
    assert result.frame["timestamp"].max() == now
    assert calls


def test_fetch_uses_pre_end_rows_before_fixed_anchor(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)
    calls: list[tuple[int, int]] = []

    class Binance:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params, since):
            calls.append((limit, since))
            start = pd.Timestamp(since, unit="ms", tz="UTC")
            return _ccxt_rows(min(limit, 1000), start=start.isoformat())

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=Binance))

    result = fetch_market_data(
        config,
        root=tmp_path,
        end_utc="2026-05-31T23:59:59+00:00",
        pre_end_rows=3000,
    )

    assert len(calls) >= 5
    assert len(result.frame) == (
        int(config["split"]["train_length"])
        + int(config["split"]["validation_length"])
        + int(config["split"]["test_length"])
        + historical_context_rows(config)
        + max_target_horizon(config)
        + 1
        + 3000
    )
    assert result.frame["timestamp"].max() == pd.Timestamp("2026-05-31T23:00:00+00:00")
    assert result.frame["timestamp"].is_monotonic_increasing


def test_fetch_binance_futures_after_spot_failure(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)
    calls: list[str] = []

    class Binance:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params):
            calls.append(symbol)
            if symbol == "BTC/USDT":
                raise RuntimeError("spot down")
            assert symbol == "BTC/USDT:USDT"
            return _ccxt_rows(limit)

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=Binance))

    result = fetch_market_data(config, root=tmp_path)

    assert result.provider_used == "binance_futures"
    assert calls == ["BTC/USDT", "BTC/USDT:USDT"]


def test_fetch_fallback_exchange_after_binance_variants_fail(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)

    class FailingExchange:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params):
            raise RuntimeError("down")

    class Kraken:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params):
            return _ccxt_rows(limit)

    monkeypatch.setitem(
        sys.modules,
        "ccxt",
        SimpleNamespace(binance=FailingExchange, binanceus=FailingExchange, kraken=Kraken),
    )

    result = fetch_market_data(config, root=tmp_path)

    assert result.provider_used == "kraken"


def test_fetch_all_live_providers_fail_without_cache(tmp_path: Path, monkeypatch) -> None:
    config, _ = load_config(CONFIG)

    class FailingExchange:
        def __init__(self, settings):
            self.settings = settings

        def fetch_ohlcv(self, symbol, timeframe, limit, params):
            raise RuntimeError("down")

    monkeypatch.setitem(
        sys.modules,
        "ccxt",
        SimpleNamespace(
            binance=FailingExchange,
            binanceus=FailingExchange,
            kraken=FailingExchange,
            okx=FailingExchange,
            kucoin=FailingExchange,
            bitfinex=FailingExchange,
        ),
    )

    try:
        fetch_market_data(config, root=tmp_path)
        raise AssertionError("fetch should fail when all live providers fail and no cache exists")
    except RuntimeError as exc:
        assert "All live OHLCV providers failed" in str(exc)


def test_model_registry_and_validation_ranking_exclude_test_metrics() -> None:
    _require_private_training()
    registry = default_model_registry()
    assert any(entry["model_id"] == "logistic_regression" for entry in registry)
    registered = {entry["model_id"] for entry in registry}
    assert {
        "naive_direction",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "mlp",
        "nn",
        "centroid",
        "lstm",
        "gru",
        "transformer",
        "tcn",
        "mamba",
        "state_space",
        "bc",
        "dagger",
        "ppo",
        "actor_critic",
        "hard_vote",
        "soft_vote",
        "stacked_logistic",
    }.issubset(registered)
    ranked = rank_validation_candidates(
        [
            {
                "status": "worked",
                "validation_metrics": {"weighted_score": 0.1},
                "test_metrics_if_frozen": {"weighted_score": 9},
            },
            {
                "status": "worked",
                "validation_metrics": {"weighted_score": 0.2},
                "test_metrics_if_frozen": {"weighted_score": 0},
            },
        ],
        "weighted_score",
    )
    assert ranked[0]["validation_metrics"]["weighted_score"] == 0.2


def test_public_sanitizer_and_allowlist() -> None:
    assert_public_safe_text("Model A delayed production summary")
    assert public_safety_violations("privateexperiments/btc_1h/model.pkl")
    assert is_public_git_allowed("src/public/data/fetch.py")
    assert is_public_git_allowed("experiments/btc_1h/reports/experiment-report.md")
    assert not is_public_git_allowed("privateexperiments/btc_1h/models/scaler.pkl")


def test_public_repo_hygiene_and_safety_audit_rules(tmp_path: Path) -> None:
    findings = find_tracked_hygiene_violations(
        [
            "README.md",
            "src/public/data/fetch.py",
            "src/private/training/pipeline.py",
            "models/prod/model_slot_1/.gitkeep",
            "models/prod/model.safetensors",
            "raw_data/btc.parquet",
        ]
    )
    assert [finding.path for finding in findings] == [
        "src/private/training/pipeline.py",
        "models/prod/model.safetensors",
        "raw_data/btc.parquet",
    ]

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text("Delayed public metrics only\n", encoding="utf-8")
    (tmp_path / "docs" / "index.html").write_text("<h1>Delayed public metrics only</h1>\n", encoding="utf-8")
    safe = scan_public_files(tmp_path)
    assert safe["status"] == "ok"
    assert "docs/index.html" in safe["scanned_files"]
    (tmp_path / "docs" / "unsafe.md").write_text("HF_TOKEN should never appear here\n", encoding="utf-8")
    unsafe = scan_public_files(tmp_path)
    assert unsafe["status"] == "failed"
    assert unsafe["findings"][0]["path"] == "docs/unsafe.md"


def test_production_refresh_reuses_frozen_exhaustive_recipe() -> None:
    _require_private_training()
    family_recipes = [
        {"candidate_id": "logistic_regression", "recipe_hash": "family-a", "decisions": {}},
        {"candidate_id": "random_forest", "recipe_hash": "family-b", "decisions": {}},
    ]
    locked_recipe = {
        "candidate_id": "logistic_regression",
        "recipe_hash": "locked-recipe",
        "decisions": {"final_validation_lock": {"variation_id": "lock_winner"}},
        "family_recipes": family_recipes,
    }
    config = {
        "project": {"default_seed": 42},
        "experiments": {"workflow_profile": "exhaustive_v1"},
        "discovery_state": {
            "workflow_id": "workflow-1",
            "catalog_hash": "catalog-1",
            "selected_recipes": [locked_recipe],
        },
    }

    workflow = _workflow_for_run(config, "prod")

    assert workflow["axis"] == "final_validation_lock"
    assert workflow["variation_id"] == "lock_winner"
    assert workflow["parent_recipe"]["recipe_hash"] == "locked-recipe"
    assert workflow["parent_recipes"] == family_recipes


def test_production_refresh_trains_all_locked_top_k_recipes() -> None:
    _require_private_training()
    from src.private.training.pipeline import _model_ids_for_request

    selected_recipes = [
        {"candidate_id": "logistic_regression", "recipe_hash": "recipe-a", "decisions": {}},
        {
            "candidate_id": "regime_gated",
            "recipe_hash": "recipe-b",
            "decisions": {},
            "family_recipes": [
                {"candidate_id": "catboost"},
                {"candidate_id": "transformer_lstm"},
                {"candidate_id": "mlp"},
            ],
        },
        {"candidate_id": "random_forest", "recipe_hash": "recipe-c", "decisions": {}},
    ]
    config = {
        "project": {"default_seed": 42},
        "experiments": {"workflow_profile": "exhaustive_v1"},
        "discovery_state": {
            "workflow_id": "workflow-1",
            "catalog_hash": "catalog-1",
            "selected_recipes": selected_recipes,
        },
    }

    config["workflow"] = _workflow_for_run(config, "prod")

    assert config["workflow"]["production_refresh"] is True
    assert _model_ids_for_request(config) == [
        "logistic_regression",
        "catboost",
        "transformer_lstm",
        "mlp",
        "random_forest",
    ]


def test_full_pipeline_smoke(tmp_path: Path) -> None:
    _require_private_training()
    request = tmp_path / "request.yaml"
    shutil.copy(CONFIG, request)
    config, _ = load_config(request)
    _write_live_cache(tmp_path, config)
    result = run_pipeline(PipelineOptions(request=request, root=tmp_path))
    assert result["status"] == "completed"
    assert result["selection_policy"] == "validation_only"
    assert (tmp_path / "privateexperiments" / "btc_1h" / "reports" / "experiment-report.md").exists()
    public_report = tmp_path / "experiments" / "btc_1h" / "reports" / "experiment-report.md"
    assert public_report.exists()
    report_text = public_report.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in report_text
    assert "## Phase 0" in report_text
    assert "Passed variation:" in report_text
    assert "Reason:" in report_text
    assert "Candidates evaluated:" in report_text
    public_manifest = json.loads(
        (tmp_path / "experiments" / "btc_1h" / "results" / "run_manifest_public.json").read_text()
    )
    assert public_manifest["test_policy"] == "frozen_winner_only"
    assert "data_variation" not in public_manifest
    assert "private_report" not in public_manifest["outputs"]
    production_public = json.loads((tmp_path / "prod" / "btc_1h" / "results" / "production_public.json").read_text())
    assert production_public["public_delay_hours"] >= 24
    assert production_public["generated_at_utc"]
    assert production_public["top_k_requested"] == 3
    assert production_public["top_k_available"] == len(production_public["top_models"])
    assert all(model["train"] and model["validation"] for model in production_public["top_models"])
    assert 1 <= production_public["top_k_available"] <= 3
    if production_public["top_k_available"] > 1:
        assert production_public["winner"] == "top_k"
        assert production_public["production_model_set"] == [
            model["public_id"] for model in production_public["top_models"]
        ]
    assert production_public["prediction_series"] == []
    assert all(model["current_version"] == 1 for model in production_public["top_models"])
    assert all(model["versions"][0]["prediction_series"] == [] for model in production_public["top_models"])
    assert all(
        row["timestamp"] <= production_public["latest_public_window"]
        for model in production_public["top_models"]
        for row in model["prediction_series"]
    )
    private_candidates = json.loads(
        (tmp_path / "privateexperiments" / "btc_1h" / "results" / "candidates.json").read_text()
    )
    assert len(private_candidates) == 22
    assert {row["backend"] for row in private_candidates} >= {"numpy", "numpy_sequence", "ensemble"}
    assert (tmp_path / "privateexperiments" / "btc_1h" / "results" / "drift" / "cache_manifest.json").exists()
    assert (tmp_path / "privateexperiments" / "btc_1h" / "results" / "hardware" / "profile.json").exists()
    assert (tmp_path / "privateexperiments" / "btc_1h" / "results" / "deployment" / "export_plan.json").exists()
    stage = tmp_path / "privateexperiments" / "btc_1h" / "models" / "hf_stage"
    assert (stage / "production_bundle.json").exists()
    assert (stage / "scaler.joblib").stat().st_size > 0
    assert any(path.name in {"model.joblib", "model_state.pt"} and path.stat().st_size > 0 for path in stage.iterdir())
    top_k_index = json.loads((stage / "top_models_private.json").read_text(encoding="utf-8"))
    assert len(top_k_index["models"]) == production_public["top_k_available"]
    top_k_policy = json.loads((stage / "top_k_policy_private.json").read_text(encoding="utf-8"))
    assert top_k_policy["production_policy"] == "top_k_mean_probability"
    from src.private.training.hf_artifacts import load_hf_artifacts

    for model in top_k_index["models"]:
        loaded_model, loaded_scaler, bundle = load_hf_artifacts(stage / model["directory"])
        assert loaded_model is not None and loaded_scaler is not None
        assert bundle["market_id"] == "btc_1h"
        recipe = json.loads((stage / model["recipe_path"]).read_text(encoding="utf-8"))
        assert recipe["recipe_hash"] == model["recipe_hash"]
        assert model["recipe"]["recipe_hash"] == model["recipe_hash"]
    export = json.loads(
        (tmp_path / "privateexperiments" / "btc_1h" / "results" / "deployment" / "export_plan.json").read_text()
    )
    assert export["export_executed"] is True
    assert (tmp_path / "privateexperiments" / "btc_1h" / "results" / "phases" / "P23.done.json").exists()


def test_pipeline_applies_request_data_variation(tmp_path: Path) -> None:
    _require_private_training()
    request = tmp_path / "request.yaml"
    text = CONFIG.read_text(encoding="utf-8") + "\ndata_variation:\n  variation: rolling_mean\n"
    request.write_text(text, encoding="utf-8")
    config, _ = load_config(request)
    _write_live_cache(tmp_path, config)
    result = run_pipeline(PipelineOptions(request=request, root=tmp_path))

    assert result["data_variation"] == "rolling_mean"
    public_manifest = json.loads(
        (tmp_path / "experiments" / "btc_1h" / "results" / "run_manifest_public.json").read_text()
    )
    assert "data_variation" not in public_manifest
    assert "rolling_mean" not in json.dumps(public_manifest)
    private_fingerprint = json.loads(
        (tmp_path / "privateexperiments" / "btc_1h" / "data" / "dataset_fingerprint.json").read_text()
    )
    assert private_fingerprint["data_variation"] == "rolling_mean"


def test_exhaustive_request_writes_all_split_result_without_early_production(tmp_path: Path) -> None:
    _require_private_training()
    request = tmp_path / "request.yaml"
    config, _ = load_config(CONFIG)
    config["workflow"] = {
        "workflow_profile": "exhaustive_v1",
        "workflow_id": "workflow_test",
        "request_id": "request_test",
        "phase_id": "phase01",
        "phase": "phase01",
        "axis": "target_horizon",
        "variation_id": "next_1",
        "parent_recipe_hash": "recipe_base",
        "catalog_hash": catalog_hash(),
        "config_hash": "config_test",
        "seed": 42,
    }
    import yaml

    request.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_live_cache(tmp_path, config)

    result = run_pipeline(PipelineOptions(request=request, root=tmp_path, phase="phase01"))

    row = result["experiment_result"]
    assert row["phase_id"] == "phase01"
    assert row["selection_basis"] == "validation_only"
    assert row["train"] and row["validation"] and row["test"]
    assert "balanced_accuracy" in row["validation"]
    assert row["candidate_results"]
    assert all(candidate["train"] and candidate["validation"] and candidate["test"] for candidate in row["candidate_results"])
    public_rows = json.loads(
        (tmp_path / "experiments" / "btc_1h" / "results" / "phase_results_public.json").read_text()
    )
    assert len(public_rows) == len(row["candidate_results"])
    assert all(public_row["train"] and public_row["validation"] and public_row["test"] for public_row in public_rows)
    assert not (tmp_path / "prod" / "btc_1h" / "results" / "production_public.json").exists()


def test_exhaustive_phase16_stages_models_without_writing_production_public(tmp_path: Path) -> None:
    _require_private_training()
    request = tmp_path / "request.yaml"
    config, _ = load_config(CONFIG)
    config["workflow"] = {
        "workflow_profile": "exhaustive_v1",
        "workflow_id": "workflow_test",
        "request_id": "request_test",
        "phase_id": "phase16",
        "phase": "phase16",
        "axis": "final_validation_lock",
        "variation_id": "lock_winner",
        "parent_recipe_hash": "recipe_base",
        "catalog_hash": catalog_hash(),
        "config_hash": "config_test",
        "seed": 42,
    }
    import yaml

    request.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_live_cache(tmp_path, config, rows=5000, start="2025-12-01")

    result = run_pipeline(PipelineOptions(request=request, root=tmp_path, phase="phase16"))

    assert result["deployment_export"]["export_executed"] is True
    sliding = result["experiment_result"]["post_lock_sliding_validation"]
    assert sliding["status"] == "worked"
    assert sliding["top_k_requested"] == config["production"]["top_k"]
    assert sliding["top_k_evaluated"] <= config["production"]["top_k"]
    assert len(sliding["models"]) <= config["production"]["top_k"]
    assert any(model["windows_completed"] > 0 for model in sliding["models"].values())
    public_rows = json.loads(
        (tmp_path / "experiments" / "btc_1h" / "results" / "phase_results_public.json").read_text(
            encoding="utf-8"
        )
    )
    post_lock_rows = [row for row in public_rows if row.get("post_lock_sliding_summary")]
    assert len(post_lock_rows) <= config["production"]["top_k"]
    assert (tmp_path / "privateexperiments" / "btc_1h" / "models" / "hf_stage" / "production_bundle.json").exists()
    assert not (tmp_path / "prod" / "btc_1h" / "results" / "production_public.json").exists()
    assert not (tmp_path / "prod" / "btc_1h" / "data" / "delayed_public_curve.json").exists()
