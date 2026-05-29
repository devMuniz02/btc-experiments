from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils import azure_ml


class AzureMlAdapterTests(unittest.TestCase):
    def test_build_submission_payload_includes_request_uris_and_wrapper_command(self) -> None:
        request_path = Path("train_requests/requests/example.json")
        request = {
            "request_id": "req-lstm-35k-base",
            "scope": "fixed_5k",
            "experiment": 1,
            "family": "LSTM",
            "train_length": "35K",
            "commit_sha": "abc123",
            "runner": "azure",
        }
        train_args = {
            "experiment": 1,
            "output_dir": "EXPERIMENTSV2/1",
            "runner": "azure",
            "mlflow_enabled": True,
            "mlflow_experiment": "fixed-5k",
            "mlflow_tracking_uri": "http://mlflow",
            "selection_mode": "all",
            "families": ["LSTM"],
            "train_lengths": ["35K"],
            "window_lengths": [],
            "model_variations": ["BASE"],
            "time_variants": ["FULL"],
            "envs": ["ternary"],
            "device": "cpu",
            "overwrite": False,
            "max_workers": 4,
            "verbose": 1,
            "experiments_root": "EXPERIMENTS",
            "mamba_epochs": 0,
            "mamba_batch_size": 0,
            "mamba_disable_early_stopping": False,
            "overfit": False,
            "overfit_max_epochs": 5000,
            "visualize_test_acc": False,
            "bandit_strategy": "ts",
            "post_base_total_updates": 0,
            "post_base_learning_rate": 0.0,
            "post_base_clip_epsilon": 0.0,
            "post_base_entropy_coef": 0.0,
            "post_base_value_loss_coef": 0.0,
            "post_base_ppo_epochs": 0,
            "post_base_minibatch_size": 0,
            "retrain_epochs": 0,
        }
        config = {
            "subscription_id": "sub",
            "resource_group": "rg",
            "workspace": "ws",
            "compute": "cpu-cluster",
            "experiment_name": "fixed-5k",
            "environment_image": "image:latest",
            "storage_account_url": "https://account.blob.core.windows.net",
            "storage_container": "artifacts",
            "storage_prefix": "fixed-5k",
            "storage_connection_string": "",
            "dashboard_base_url": "https://dashboard.example.com",
            "mlflow_tracking_uri": "http://mlflow",
            "mlflow_tracking_username": "",
            "mlflow_tracking_password": "",
        }

        payload = azure_ml.build_submission_payload(request_path, request, train_args, config)

        self.assertEqual(payload["job_name"], "fixed-5k-exp-1-req-lstm-35k-base")
        self.assertIn("scripts/azure_train_request_job.py", payload["command"])
        self.assertIn("train_requests/requests/example.json", payload["command"])
        self.assertEqual(
            payload["result_blob_uri"],
            "https://account.blob.core.windows.net/artifacts/fixed-5k/experiments/1/requests/req-lstm-35k-base/request_result.json",
        )
        self.assertEqual(
            payload["dashboard_uri"],
            "https://dashboard.example.com/fixed-5k/experiments/1/requests/req-lstm-35k-base/index.html",
        )

    def test_map_azure_job_status_handles_core_states(self) -> None:
        self.assertEqual(azure_ml.map_azure_job_status("Queued"), "submitted")
        self.assertEqual(azure_ml.map_azure_job_status("Running"), "running")
        self.assertEqual(azure_ml.map_azure_job_status("Completed"), "completed")
        self.assertEqual(azure_ml.map_azure_job_status("Canceled"), "failed")
        self.assertEqual(azure_ml.map_azure_job_status("SomethingElse"), "reconcile_required")

    def test_map_local_path_to_blob_uri_preserves_relative_structure(self) -> None:
        local_root = Path("C:/repo/EXPERIMENTSV2/1")
        local_path = local_root / "models" / "best.pt"
        blob_uri = azure_ml.map_local_path_to_blob_uri(
            str(local_path),
            str(local_root),
            "https://account.blob.core.windows.net/container/fixed-5k/experiments/1/requests/req-1/artifacts",
        )

        self.assertEqual(
            blob_uri,
            "https://account.blob.core.windows.net/container/fixed-5k/experiments/1/requests/req-1/artifacts/models/best.pt",
        )

    def test_publish_local_run_returns_expected_uris(self) -> None:
        request_path = Path("train_requests/requests/example.json")
        request = {
            "request_id": "req-lstm-35k-base",
            "scope": "fixed_5k",
            "experiment": 1,
        }
        artifact_root = Path("C:/repo/EXPERIMENTSV2/1")
        result_payload = {
            "artifact_root": str(artifact_root),
            "matched_rows": [
                {
                    "saved_model_path": str(artifact_root / "models" / "best.pt"),
                    "predictions_path": str(artifact_root / "predictions" / "best.parquet"),
                }
            ],
        }
        config = {
            "storage_account_url": "https://account.blob.core.windows.net",
            "storage_container": "artifacts",
            "storage_prefix": "fixed-5k",
            "storage_connection_string": "UseDevelopmentStorage=true",
            "dashboard_base_url": "https://dashboard.example.com",
        }

        with (
            patch.object(azure_ml, "upload_directory_to_blob", return_value=["blob://one"]) as patched_upload_dir,
            patch.object(azure_ml, "upload_json_blob") as patched_upload_json,
            patch.object(azure_ml, "upload_text_blob") as patched_upload_text,
            patch.object(Path, "exists", return_value=True),
        ):
            publication = azure_ml.publish_local_run(request_path, request, result_payload, config=config)

        self.assertEqual(publication["storage_container"], "artifacts")
        self.assertEqual(
            publication["artifact_root_uri"],
            "https://account.blob.core.windows.net/artifacts/fixed-5k/experiments/1/requests/req-lstm-35k-base/artifacts",
        )
        self.assertEqual(
            publication["model_artifact_uri"],
            "https://account.blob.core.windows.net/artifacts/fixed-5k/experiments/1/requests/req-lstm-35k-base/artifacts/models/best.pt",
        )
        self.assertEqual(
            publication["dashboard_uri"],
            "https://dashboard.example.com/fixed-5k/experiments/1/requests/req-lstm-35k-base/index.html",
        )
        patched_upload_dir.assert_called_once()
        self.assertEqual(patched_upload_json.call_count, 2)
        patched_upload_text.assert_called_once()

    def test_missing_blob_publish_fields_are_reported(self) -> None:
        config = {
            "storage_account_url": "",
            "storage_container": "",
            "storage_connection_string": "",
            "dashboard_base_url": "",
        }

        missing = azure_ml.missing_blob_publish_config_fields(config)

        self.assertEqual(missing, ["dashboard_base_url", "storage_account_url", "storage_container"])
