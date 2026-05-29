from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from argparse import Namespace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _install_train_test_stubs() -> None:
    if "src.utils.experiment_runners" not in sys.modules:
        unified_v2 = types.ModuleType("src.utils.experiment_runners")
        unified_v2.run_train = lambda args: {"summary_path": "", "manifest_path": ""}
        unified_v2.run_reeval = lambda args: {"summary_path": "", "manifest_path": ""}
        unified_v2.run_report = lambda args: {"summary_path": "", "manifest_path": ""}
        unified_v2.run_migrate = lambda args: {"summary_path": "", "manifest_path": ""}
        sys.modules["src.utils.experiment_runners"] = unified_v2

    if "scripts._helpers" not in sys.modules:
        helpers = types.ModuleType("scripts._helpers")

        def load_json(path: str | Path) -> dict[str, object]:
            return json.loads(Path(path).read_text(encoding="utf-8"))

        def write_json_atomic(path: str | Path, payload: dict[str, object]) -> None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        helpers.load_json = load_json
        helpers.write_json_atomic = write_json_atomic
        sys.modules["scripts._helpers"] = helpers

    if "src.utils.experiment_support" not in sys.modules:
        experiments_v2 = types.ModuleType("src.utils.experiment_support")
        experiments_v2.EXPERIMENTS_V2_ROOT = ROOT / "EXPERIMENTSV2"
        experiments_v2.load_summary_rows = lambda output_root: []
        experiments_v2.summary_rows_path = lambda output_root: Path(output_root) / "summary_rows.parquet"
        sys.modules["src.utils.experiment_support"] = experiments_v2


_install_train_test_stubs()

from scripts import train as train_cli


class TrainRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = ROOT / "temp" / f"train-request-tests-{self._testMethodName}"
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)

    def _request_payload(self) -> dict[str, object]:
        return {
            "request_id": "req-lstm-35k-base",
            "status": "pending",
            "scope": "fixed_5k",
            "experiment": 1,
            "family": "LSTM",
            "train_length": "35K",
            "window_length": "",
            "model_variation": "BASE",
            "time_variants": ["FULL"],
            "env": "ternary",
            "overwrite": False,
            "output_dir": str(self.temp_root / "EXPERIMENTSV2" / "1"),
            "mlflow": {"enabled": False},
            "result": {},
        }

    def _create_result_artifacts(self, artifact_root: Path) -> dict[str, Path]:
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths = {
            "summary_path": artifact_root / "summary.json",
            "manifest_path": artifact_root / "manifest.json",
            "models_path": artifact_root / "summary_rows.parquet",
            "saved_model_path": artifact_root / "model.pt",
            "predictions_path": artifact_root / "predictions.parquet",
        }
        for path in paths.values():
            path.write_text("x", encoding="utf-8")
        return paths

    def test_normalize_request_payload_applies_defaults(self) -> None:
        payload = self._request_payload()
        normalized = train_cli._normalize_request_payload(payload)

        self.assertEqual(normalized["scope"], "fixed_5k")
        self.assertEqual(normalized["status"], "pending")
        self.assertEqual(normalized["family"], "LSTM")
        self.assertEqual(normalized["train_length"], "35K")
        self.assertEqual(normalized["model_variation"], "BASE")
        self.assertEqual(normalized["time_variants"], ["FULL"])
        self.assertEqual(normalized["env"], "ternary")
        self.assertEqual(normalized["runner"], "local")
        self.assertIn("requested_at", normalized)
        self.assertIn("requested_by", normalized)

    def test_normalize_request_payload_rejects_non_fixed_scope(self) -> None:
        payload = self._request_payload()
        payload["scope"] = "rolling"

        with self.assertRaises(ValueError):
            train_cli._normalize_request_payload(payload)

    def test_parse_args_train_and_request_run_defaults(self) -> None:
        with patch.object(sys, "argv", ["train.py", "train"]):
            train_args = train_cli.parse_args()
        self.assertEqual(train_args.command, "train")
        self.assertEqual(train_args.runner, "local")
        self.assertEqual(train_args.time_variants, ["FULL"])
        self.assertEqual(train_args.envs, ["ternary"])

        with patch.object(sys, "argv", ["train.py", "request-run", "--request-file", "request.json"]):
            request_args = train_cli.parse_args()
        self.assertEqual(request_args.command, "request-run")
        self.assertEqual(request_args.request_file, "request.json")
        self.assertEqual(request_args.requests_dir, str(train_cli.DEFAULT_REQUEST_FOLDER))
        self.assertFalse(request_args.publish_azure)

    def test_request_artifacts_exist_requires_summary_manifest_models_and_saved_outputs(
        self,
    ) -> None:
        artifact_root = self.temp_root / "EXPERIMENTSV2" / "1"
        paths = self._create_result_artifacts(artifact_root)

        request = {
            "result": {
                "summary_path": str(paths["summary_path"]),
                "manifest_path": str(paths["manifest_path"]),
                "models_path": str(paths["models_path"]),
                "matched_rows": [
                    {
                        "saved_model_path": str(paths["saved_model_path"]),
                        "predictions_path": str(paths["predictions_path"]),
                    }
                ],
            }
        }

        self.assertTrue(train_cli._request_artifacts_exist(request))
        paths["predictions_path"].unlink()
        self.assertFalse(train_cli._request_artifacts_exist(request))

    def test_result_payload_from_command_filters_and_sorts_matching_rows(self) -> None:
        output_root = self.temp_root / "EXPERIMENTSV2" / "1"
        paths = self._create_result_artifacts(output_root)
        request = train_cli._normalize_request_payload(
            {**self._request_payload(), "time_variants": ["FULL", "REGULAR"]}
        )
        rows = [
            {
                "name": "ignore-me",
                "status": "completed",
                "family": "MLP",
                "train_length": "35K",
                "window_length": "",
                "model_variation": "BASE",
                "time_variant": "FULL",
                "env_version": "ternary",
                "threshold_pct": 99,
            },
            {
                "name": "target-b",
                "status": "completed",
                "family": "LSTM",
                "train_length": "35K",
                "window_length": "",
                "model_variation": "BASE",
                "time_variant": "REGULAR",
                "env_version": "ternary",
                "threshold_pct": 20,
                "saved_model_path": str(paths["saved_model_path"]),
                "predictions_path": str(paths["predictions_path"]),
                "test": {"accuracy": 0.71},
                "daily_net_wins": 4,
            },
            {
                "name": "target-a",
                "status": "completed",
                "family": "LSTM",
                "train_length": "35K",
                "window_length": "",
                "model_variation": "BASE",
                "time_variant": "FULL",
                "env_version": "ternary",
                "threshold_pct": 10,
                "saved_model_path": str(paths["saved_model_path"]),
                "predictions_path": str(paths["predictions_path"]),
                "test": {"accuracy": 0.83},
                "daily_net_wins": 7,
            },
        ]

        with (
            patch.object(train_cli, "load_summary_rows", return_value=rows),
            patch.object(train_cli, "summary_rows_path", return_value=paths["models_path"]),
            patch.object(train_cli, "_utc_now", return_value="2026-05-28T12:00:00+00:00"),
        ):
            result_payload = train_cli._result_payload_from_command(
                output_root,
                {"summary_path": str(paths["summary_path"]), "manifest_path": str(paths["manifest_path"])},
                request,
            )

        self.assertEqual(result_payload["artifact_root"], str(output_root.resolve()))
        self.assertEqual(result_payload["models_path"], str(paths["models_path"].resolve()))
        self.assertEqual(
            [row["name"] for row in result_payload["matched_rows"]],
            ["target-a", "target-b"],
        )
        self.assertEqual(result_payload["matched_rows"][0]["test_accuracy"], 0.83)
        self.assertEqual(result_payload["completed_at"], "2026-05-28T12:00:00+00:00")

    def test_training_namespace_from_request_keeps_runtime_options(self) -> None:
        request = train_cli._normalize_request_payload(
            {
                **self._request_payload(),
                "window_length": "48H",
                "selection_mode": "subset",
                "device": "cpu",
                "overwrite": True,
                "max_workers": 2,
                "verbose": 3,
                "experiments_root": str(self.temp_root / "EXPERIMENTS"),
                "mamba_epochs": 10,
                "mamba_batch_size": 64,
                "mamba_disable_early_stopping": True,
                "overfit": True,
                "overfit_max_epochs": 88,
                "visualize_test_acc": True,
                "bandit_strategy": "ucb",
                "post_base_total_updates": 11,
                "post_base_learning_rate": 0.05,
                "post_base_clip_epsilon": 0.2,
                "post_base_entropy_coef": 0.03,
                "post_base_value_loss_coef": 0.7,
                "post_base_ppo_epochs": 5,
                "post_base_minibatch_size": 16,
                "retrain_epochs": 7,
                "mlflow": {
                    "enabled": True,
                    "experiment_name": "req-exp",
                    "tracking_uri": "file:///mlruns",
                },
            }
        )

        args = train_cli._training_namespace_from_request(request, runner="azure")

        self.assertEqual(args.command, "train")
        self.assertEqual(args.runner, "azure")
        self.assertEqual(args.window_lengths, ["48H"])
        self.assertEqual(args.selection_mode, "subset")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.overwrite)
        self.assertEqual(args.bandit_strategy, "ucb")
        self.assertTrue(args.mlflow_enabled)
        self.assertEqual(args.mlflow_experiment, "req-exp")
        self.assertEqual(args.mlflow_tracking_uri, "file:///mlruns")
        self.assertEqual(args.retrain_epochs, 7)

    def test_run_train_command_wraps_runner_output(self) -> None:
        output_root = self.temp_root / "EXPERIMENTSV2" / "1"
        models_path = output_root / "summary_rows.parquet"
        rows = [{"status": "completed", "test": {"accuracy": 0.9}, "daily_net_wins": 3}]
        args = Namespace(
            experiment=1,
            output_dir=str(output_root),
            runner="local",
            mlflow_enabled=False,
            mlflow_experiment="exp",
            mlflow_tracking_uri="",
            selection_mode="all",
            families=["LSTM"],
            train_lengths=["35K"],
            window_lengths=[],
            model_variations=["BASE"],
            time_variants=["FULL"],
            envs=["ternary"],
            device="cpu",
            overwrite=False,
            max_workers=4,
            experiments_root=str(self.temp_root / "EXPERIMENTS"),
            verbose=1,
            mamba_epochs=0,
            mamba_batch_size=0,
            mamba_disable_early_stopping=False,
            overfit=False,
            overfit_max_epochs=5000,
            visualize_test_acc=False,
            bandit_strategy="ts",
            post_base_total_updates=0,
            post_base_learning_rate=0.0,
            post_base_clip_epsilon=0.0,
            post_base_entropy_coef=0.0,
            post_base_value_loss_coef=0.0,
            post_base_ppo_epochs=0,
            post_base_minibatch_size=0,
            retrain_epochs=0,
        )

        with (
            patch.object(
                train_cli.unified_v2, "run_train", return_value={"summary_path": "s.json", "manifest_path": "m.json"}
            ) as patched_run,
            patch.object(train_cli, "load_summary_rows", return_value=rows),
            patch.object(train_cli, "summary_rows_path", return_value=models_path),
            patch.object(train_cli, "_log_mlflow_result", return_value="run-123"),
        ):
            result = train_cli.run_train_command(args)

        self.assertEqual(result["command"], "train")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output_dir"], str(output_root.resolve()))
        self.assertEqual(result["models_path"], str(models_path.resolve()))
        self.assertEqual(result["mlflow_run_id"], "run-123")
        forwarded_args = patched_run.call_args.args[0]
        self.assertEqual(forwarded_args.families, ["LSTM"])
        self.assertEqual(forwarded_args.output_dir, str(output_root.resolve()))

    def test_run_report_reeval_and_migrate_commands_wrap_outputs(self) -> None:
        output_root = self.temp_root / "EXPERIMENTSV2" / "1"
        rows = [{"status": "completed", "test": {"accuracy": 0.91}, "daily_net_wins": 5}]
        cases = [
            (
                "report",
                train_cli.run_report_command,
                "run_report",
                Namespace(
                    experiment=1,
                    output_dir=str(output_root),
                    runner="local",
                    mlflow_enabled=False,
                    mlflow_experiment="exp",
                    mlflow_tracking_uri="",
                    rebuild_artifacts=True,
                ),
            ),
            (
                "reeval",
                train_cli.run_reeval_command,
                "run_reeval",
                Namespace(
                    experiment=1,
                    output_dir=str(output_root),
                    runner="local",
                    mlflow_enabled=False,
                    mlflow_experiment="exp",
                    mlflow_tracking_uri="",
                    families=["LSTM"],
                    train_lengths=["35K"],
                    target_scope="all",
                    missing_only=False,
                    max_workers=2,
                    envs=["ternary"],
                ),
            ),
            (
                "migrate",
                train_cli.run_migrate_command,
                "run_migrate",
                Namespace(
                    experiment=1,
                    output_dir=str(output_root),
                    runner="local",
                    mlflow_enabled=False,
                    mlflow_experiment="exp",
                    mlflow_tracking_uri="",
                    rewrite_paths=True,
                    rebuild_artifacts=True,
                ),
            ),
        ]

        for command_name, command_fn, runner_attr, args in cases:
            with self.subTest(command=command_name):
                with (
                    patch.object(
                        train_cli.unified_v2,
                        runner_attr,
                        return_value={"summary_path": "s.json", "manifest_path": "m.json"},
                    ) as patched_runner,
                    patch.object(train_cli, "load_summary_rows", return_value=rows),
                    patch.object(train_cli, "_log_mlflow_result", return_value="run-xyz"),
                ):
                    result = command_fn(args)

                self.assertEqual(result["command"], command_name)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["output_dir"], str(output_root.resolve()))
                self.assertEqual(result["mlflow_run_id"], "run-xyz")
                patched_runner.assert_called_once()

    def test_request_reconcile_marks_healthy_completed_request_without_rerun(
        self,
    ) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = self.temp_root / "EXPERIMENTSV2" / "1"
        paths = self._create_result_artifacts(artifact_root)

        request_path = requests_dir / "healthy.json"
        payload = self._request_payload()
        payload["status"] = "completed"
        payload["result"] = {
            "summary_path": str(paths["summary_path"]),
            "manifest_path": str(paths["manifest_path"]),
            "models_path": str(paths["models_path"]),
            "matched_rows": [
                {
                    "saved_model_path": str(paths["saved_model_path"]),
                    "predictions_path": str(paths["predictions_path"]),
                }
            ],
        }
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        result = train_cli.run_request_reconcile_command(
            Namespace(
                requests_dir=str(requests_dir),
                runner="local",
                publish_azure=False,
                mlflow_enabled=False,
                mlflow_experiment="fixed-5k-train-requests",
                mlflow_tracking_uri="",
                retry_failed=False,
            )
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["healthy"], 1)
        self.assertEqual(result["retrained"], 0)
        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "completed")

    def test_request_reconcile_reruns_pending_request(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "pending.json"
        request_path.write_text(json.dumps(self._request_payload(), indent=2), encoding="utf-8")

        with patch.object(
            train_cli,
            "_run_request_file",
            return_value={
                "command": "request-run",
                "status": "completed",
                "request_id": "req-lstm-35k-base",
            },
        ) as patched:
            result = train_cli.run_request_reconcile_command(
                Namespace(
                    requests_dir=str(requests_dir),
                    runner="local",
                    publish_azure=False,
                    mlflow_enabled=False,
                    mlflow_experiment="fixed-5k-train-requests",
                    mlflow_tracking_uri="",
                    retry_failed=False,
                )
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["retrained"], 1)
        self.assertEqual(result["failed"], 0)
        patched.assert_called_once()

    def test_run_request_file_marks_failed_request_and_persists_traceback(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "pending.json"
        request_path.write_text(json.dumps(self._request_payload(), indent=2), encoding="utf-8")

        with patch.object(train_cli, "run_train_command", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                train_cli._run_request_file(
                    request_path,
                    runner="local",
                    publish_azure=False,
                    mlflow_enabled_override=None,
                    mlflow_experiment_override="",
                    mlflow_tracking_uri_override="",
                    retry_failed=False,
                )

        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["result"]["error"], "boom")
        self.assertIn("RuntimeError: boom", updated["result"]["traceback"])
        self.assertIn("failed_at", updated["result"])

    def test_run_request_file_skips_previous_failure_without_retry(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "failed.json"
        payload = self._request_payload()
        payload["status"] = "failed"
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        result = train_cli._run_request_file(
            request_path,
            runner="local",
            publish_azure=False,
            mlflow_enabled_override=None,
            mlflow_experiment_override="",
            mlflow_tracking_uri_override="",
            retry_failed=False,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "request already failed")

    def test_validate_request_file_formats_healthy_json(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "healthy.json"
        request_path.write_text('{"family":"LSTM","train_length":"35K","request_id":"req-1"}', encoding="utf-8")

        request, validation_error = train_cli._validate_request_file_for_execution(request_path)

        self.assertEqual(validation_error, "")
        self.assertIsNotNone(request)
        formatted = request_path.read_text(encoding="utf-8")
        self.assertIn('\n  "request_id": "req-1"', formatted)
        self.assertIn('"scope": "fixed_5k"', formatted)

    def test_run_request_file_skips_invalid_json_request(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "broken.json"
        request_path.write_text('{"request_id": "req-1", "family": ', encoding="utf-8")

        with patch.object(train_cli, "run_train_command") as patched_train:
            result = train_cli._run_request_file(
                request_path,
                runner="local",
                publish_azure=False,
                mlflow_enabled_override=None,
                mlflow_experiment_override="",
                mlflow_tracking_uri_override="",
                retry_failed=False,
            )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["request_id"], "broken")
        self.assertIn("invalid request json", result["reason"])
        patched_train.assert_not_called()

    def test_run_request_file_submits_azure_job_and_updates_request(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "azure.json"
        payload = self._request_payload()
        payload["runner"] = "azure"
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        submission = {
            "workspace": "aml-ws",
            "compute": "cpu-cluster",
            "job_name": "job-123",
            "job_id": "job-123",
            "portal_url": "https://ml.azure.com/runs/job-123",
            "experiment_name": "fixed-5k",
            "result_blob_uri": "https://storage/container/result.json",
            "artifact_root_uri": "https://storage/container/artifacts",
            "dashboard_uri": "https://dashboard/req-1",
            "model_artifact_uri": "https://storage/container/best-model",
            "request_snapshot_blob_uri": "https://storage/container/request.json",
        }

        with patch.object(train_cli.azure_ml, "submit_train_request", return_value=submission) as patched_submit:
            result = train_cli._run_request_file(
                request_path,
                runner="azure",
                publish_azure=False,
                mlflow_enabled_override=True,
                mlflow_experiment_override="azure-exp",
                mlflow_tracking_uri_override="http://mlflow",
                retry_failed=False,
            )

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["azure"]["job_name"], "job-123")
        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "submitted")
        self.assertEqual(updated["runner"], "azure")
        self.assertEqual(updated["azure"]["job_name"], "job-123")
        self.assertEqual(updated["result"]["artifact_root"], "https://storage/container/artifacts")
        self.assertEqual(updated["mlflow"]["experiment_name"], "azure-exp")
        self.assertEqual(updated["mlflow"]["tracking_uri"], "http://mlflow")
        patched_submit.assert_called_once()

    def test_run_request_file_publishes_local_results_to_azure(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = self.temp_root / "EXPERIMENTSV2" / "1"
        paths = self._create_result_artifacts(artifact_root)
        request_path = requests_dir / "local-publish.json"
        request_path.write_text(json.dumps(self._request_payload(), indent=2), encoding="utf-8")

        command_result = {
            "command": "train",
            "status": "completed",
            "summary_path": str(paths["summary_path"]),
            "manifest_path": str(paths["manifest_path"]),
            "models_path": str(paths["models_path"]),
            "output_dir": str(artifact_root),
            "mlflow_run_id": "run-123",
        }
        publication = {
            "storage_container": "artifacts",
            "storage_prefix": "fixed-5k",
            "artifact_root_uri": "https://storage/container/artifacts",
            "dashboard_uri": "https://dashboard/req-1",
            "model_artifact_uri": "https://storage/container/artifacts/model.pt",
            "request_snapshot_blob_uri": "https://storage/container/request.json",
            "result_blob_uri": "https://storage/container/result.json",
        }
        rows = [
            {
                "name": "target-a",
                "status": "completed",
                "family": "LSTM",
                "train_length": "35K",
                "window_length": "",
                "model_variation": "BASE",
                "time_variant": "FULL",
                "env_version": "ternary",
                "threshold_pct": 10,
                "saved_model_path": str(paths["saved_model_path"]),
                "predictions_path": str(paths["predictions_path"]),
            }
        ]

        with (
            patch.object(train_cli, "run_train_command", return_value=command_result),
            patch.object(train_cli, "load_summary_rows", return_value=rows),
            patch.object(train_cli, "summary_rows_path", return_value=paths["models_path"]),
            patch.object(train_cli.azure_ml, "publish_local_run", return_value=publication) as patched_publish,
        ):
            result = train_cli._run_request_file(
                request_path,
                runner="local",
                publish_azure=True,
                mlflow_enabled_override=None,
                mlflow_experiment_override="",
                mlflow_tracking_uri_override="",
                retry_failed=False,
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["result"]["published_to_azure"])
        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["runner"], "local")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["azure"]["artifact_root_uri"], "https://storage/container/artifacts")
        self.assertEqual(updated["azure"]["dashboard_uri"], "https://dashboard/req-1")
        self.assertEqual(updated["result"]["artifact_root"], "https://storage/container/artifacts")
        patched_publish.assert_called_once()

    def test_run_request_file_sets_reconcile_required_when_publish_fails(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = self.temp_root / "EXPERIMENTSV2" / "1"
        paths = self._create_result_artifacts(artifact_root)
        request_path = requests_dir / "local-publish-fail.json"
        request_path.write_text(json.dumps(self._request_payload(), indent=2), encoding="utf-8")

        command_result = {
            "command": "train",
            "status": "completed",
            "summary_path": str(paths["summary_path"]),
            "manifest_path": str(paths["manifest_path"]),
            "models_path": str(paths["models_path"]),
            "output_dir": str(artifact_root),
            "mlflow_run_id": "run-123",
        }
        rows = [
            {
                "name": "target-a",
                "status": "completed",
                "family": "LSTM",
                "train_length": "35K",
                "window_length": "",
                "model_variation": "BASE",
                "time_variant": "FULL",
                "env_version": "ternary",
                "threshold_pct": 10,
                "saved_model_path": str(paths["saved_model_path"]),
                "predictions_path": str(paths["predictions_path"]),
            }
        ]

        with (
            patch.object(train_cli, "run_train_command", return_value=command_result),
            patch.object(train_cli, "load_summary_rows", return_value=rows),
            patch.object(train_cli, "summary_rows_path", return_value=paths["models_path"]),
            patch.object(train_cli.azure_ml, "publish_local_run", side_effect=RuntimeError("publish boom")),
        ):
            result = train_cli._run_request_file(
                request_path,
                runner="local",
                publish_azure=True,
                mlflow_enabled_override=None,
                mlflow_experiment_override="",
                mlflow_tracking_uri_override="",
                retry_failed=False,
            )

        self.assertEqual(result["status"], "reconcile_required")
        self.assertIn("Azure publish failed after local training", result["result"]["error"])
        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "reconcile_required")
        self.assertFalse(updated["result"]["published_to_azure"])

    def test_run_request_file_marks_failed_when_azure_submission_errors(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "azure-fail.json"
        payload = self._request_payload()
        payload["runner"] = "azure"
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with patch.object(train_cli.azure_ml, "submit_train_request", side_effect=RuntimeError("azure boom")):
            result = train_cli._run_request_file(
                request_path,
                runner="azure",
                publish_azure=False,
                mlflow_enabled_override=None,
                mlflow_experiment_override="",
                mlflow_tracking_uri_override="",
                retry_failed=False,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["result"]["error"], "azure boom")
        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["result"]["error"], "azure boom")

    def test_request_reconcile_syncs_completed_azure_job(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "azure-running.json"
        payload = self._request_payload()
        payload["runner"] = "azure"
        payload["status"] = "running"
        payload["azure"] = {
            "job_name": "job-123",
            "job_id": "job-123",
            "portal_url": "https://ml.azure.com/runs/job-123",
            "result_blob_uri": "https://storage/container/result.json",
        }
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with (
            patch.object(
                train_cli.azure_ml,
                "fetch_job_snapshot",
                return_value={
                    "raw_status": "Completed",
                    "request_status": "completed",
                    "job_name": "job-123",
                    "job_id": "job-123",
                    "portal_url": "https://ml.azure.com/runs/job-123",
                    "artifact_root_uri": "https://storage/container/artifacts",
                    "dashboard_uri": "https://dashboard/req-1",
                    "model_artifact_uri": "https://storage/container/best-model",
                    "result_blob_uri": "https://storage/container/result.json",
                },
            ),
            patch.object(
                train_cli.azure_ml,
                "download_json_blob",
                return_value={
                    "status": "completed",
                    "mlflow_run_id": "run-123",
                    "artifact_root": "https://storage/container/artifacts",
                    "model_artifact_uri": "https://storage/container/best-model",
                    "dashboard_uri": "https://dashboard/req-1",
                },
            ),
        ):
            result = train_cli.run_request_reconcile_command(
                Namespace(
                    requests_dir=str(requests_dir),
                    runner="azure",
                    publish_azure=False,
                    mlflow_enabled=False,
                    mlflow_experiment="fixed-5k-train-requests",
                    mlflow_tracking_uri="",
                    retry_failed=False,
                )
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["healthy"], 1)
        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["result"]["mlflow_run_id"], "run-123")
        self.assertEqual(updated["result"]["dashboard_uri"], "https://dashboard/req-1")

    def test_request_reconcile_reports_failed_when_azure_metadata_is_missing(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        request_path = requests_dir / "azure-bad.json"
        payload = self._request_payload()
        payload["runner"] = "azure"
        payload["status"] = "running"
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with patch.object(train_cli.azure_ml, "fetch_job_snapshot", side_effect=ValueError("Missing Azure job_name")):
            result = train_cli.run_request_reconcile_command(
                Namespace(
                    requests_dir=str(requests_dir),
                    runner="azure",
                    publish_azure=False,
                    mlflow_enabled=False,
                    mlflow_experiment="fixed-5k-train-requests",
                    mlflow_tracking_uri="",
                    retry_failed=False,
                )
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertIn("Missing Azure job_name", result["results"][0]["error"])

    def test_request_reconcile_repairs_local_publish_metadata(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = self.temp_root / "EXPERIMENTSV2" / "1"
        paths = self._create_result_artifacts(artifact_root)
        request_path = requests_dir / "repair-publish.json"
        payload = self._request_payload()
        payload["status"] = "completed"
        payload["result"] = {
            "summary_path": str(paths["summary_path"]),
            "manifest_path": str(paths["manifest_path"]),
            "models_path": str(paths["models_path"]),
            "artifact_root": str(artifact_root.resolve()),
            "matched_rows": [
                {
                    "saved_model_path": str(paths["saved_model_path"]),
                    "predictions_path": str(paths["predictions_path"]),
                }
            ],
        }
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        publication = {
            "storage_container": "artifacts",
            "storage_prefix": "fixed-5k",
            "artifact_root_uri": "https://storage/container/artifacts",
            "dashboard_uri": "https://dashboard/req-1",
            "model_artifact_uri": "https://storage/container/artifacts/model.pt",
            "request_snapshot_blob_uri": "https://storage/container/request.json",
            "result_blob_uri": "https://storage/container/result.json",
        }

        with patch.object(train_cli.azure_ml, "publish_local_run", return_value=publication):
            result = train_cli.run_request_reconcile_command(
                Namespace(
                    requests_dir=str(requests_dir),
                    runner="local",
                    publish_azure=True,
                    mlflow_enabled=False,
                    mlflow_experiment="fixed-5k-train-requests",
                    mlflow_tracking_uri="",
                    retry_failed=False,
                )
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["healthy"], 1)
        updated = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["azure"]["artifact_root_uri"], "https://storage/container/artifacts")
        self.assertTrue(updated["result"]["published_to_azure"])

    def test_request_reconcile_skips_invalid_json_request(self) -> None:
        requests_dir = self.temp_root / "train_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        (requests_dir / "broken.json").write_text('{"request_id": "req-1", "family": ', encoding="utf-8")

        result = train_cli.run_request_reconcile_command(
            Namespace(
                requests_dir=str(requests_dir),
                runner="local",
                publish_azure=False,
                mlflow_enabled=False,
                mlflow_experiment="fixed-5k-train-requests",
                mlflow_tracking_uri="",
                retry_failed=False,
            )
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["results"][0]["status"], "skipped")
        self.assertIn("invalid request json", result["results"][0]["reason"])

    def test_main_dispatches_report_command_and_prints_json(self) -> None:
        output = StringIO()
        args = Namespace(command="report")
        payload = {"command": "report", "status": "completed", "output_dir": "x"}

        with (
            patch.object(train_cli, "parse_args", return_value=args),
            patch.object(train_cli, "run_report_command", return_value=payload) as patched_run,
            patch("sys.stdout", new=output),
        ):
            exit_code = train_cli.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), payload)
        patched_run.assert_called_once_with(args)
