# Quant-Stream

[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Quant--Stream-2563EB)
![Quant](https://img.shields.io/badge/Quant-BTC%201h-111827)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Local%20%2B%20GitHub-16A34A)
![MLflow](https://img.shields.io/badge/MLflow-Sync%20Ready-0194E2)
![Azure](https://img.shields.io/badge/Azure-Disabled%20Until%20Sync-0078D4?logo=microsoftazure&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Backtest%20Kernel-76B900?logo=nvidia&logoColor=white)

Last updated: `2026-06-02T05:18:22.899310+00:00`

Quant-Stream is a local, file-state BTC 1h research pipeline. YAML requests move through automation folders, models write local artifacts, and prediction columns accumulate in variation-level parquet result stores.

## Workflow

- Generate `var_1` with `powershell -ExecutionPolicy Bypass -File .\automation\generate_var_1_dataset.ps1`.
- Add run YAML files to `automation/run_requests/`.
- Add delete YAML files to `automation/delete_requests/`.
- Run `python automation_runner.py --once` or use the local Windows watcher.
- Run `python state_sync.py --check` to verify model folders, result columns, scalers, and sync buffers.

Supported active model families: `lstm`, `transformer`, `mamba`, `nn`, `rf`, `xgboost`, `bc`, `dagger`, `ppo`, `ppo_continue`, `actor_critic`, `mamba_post_base`, and `ensemble`.

Supported training modes: `static_baseline`, `sliding_window_current_only`, `sliding_window_continue`, `sliding_window_retrain`, `reinforcement_ppo`, and `post_base`.

## Run Counts

| total | pending | done | rejected | deleted |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |

## Top Dev Model

| model_id | model_type | variation_or_slot | status | latest_timestamp | prediction_count | signal_count | mean_probability | accuracy | win_rate | net_pnl | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | dev | - | - | - | - | - | - | - | - | - |

## Production Slots

| model_id | model_type | variation_or_slot | status | latest_timestamp | prediction_count | signal_count | mean_probability | accuracy | win_rate | net_pnl | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | model_slot_1 | - | - | - | - | - | - | - | - | - |
| - | - | model_slot_2 | - | - | - | - | - | - | - | - | - |
| - | - | model_slot_3 | - | - | - | - | - | - | - | - | - |
| - | - | model_slot_4 | - | - | - | - | - | - | - | - | - |
| - | - | model_slot_5 | - | - | - | - | - | - | - | - | - |

## MLflow Sync

| status | public_url | note |
| --- | --- | --- |
| Not synced | - | MLflow is hidden until sync marks the URL as public. |

## Local Automation

- `automation/*.ps1` scripts are local-only and ignored by Git.
- GitHub Actions run code quality only.
- Local request execution stays on this machine until MLflow sync is explicitly enabled.
