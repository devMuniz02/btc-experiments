from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def candidate_conda_executables() -> list[str]:
    found: list[str] = []
    on_path = shutil.which("conda")
    if on_path:
        found.append(on_path)
    home = Path.home()
    for candidate in (
        home / "miniforge3" / "Scripts" / "conda.exe",
        home / "miniconda3" / "Scripts" / "conda.exe",
        home / "anaconda3" / "Scripts" / "conda.exe",
        Path("C:/ProgramData/miniforge3/Scripts/conda.exe"),
        Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
        Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
    ):
        if candidate.exists():
            resolved = str(candidate)
            if resolved not in found:
                found.append(resolved)
    return found


def conda_environment_preflight(env_name: str = "btc-quant-stream", *, timeout_seconds: int = 30) -> dict[str, Any]:
    conda_executables = candidate_conda_executables()
    conda_executable = conda_executables[0] if conda_executables else ""
    report: dict[str, Any] = {
        "phase": "conda_environment_preflight",
        "env_name": env_name,
        "conda_on_path": bool(shutil.which("conda")),
        "conda_available": bool(conda_executable),
        "candidate_conda_executables": conda_executables,
        "conda_executable": conda_executable,
        "env_available": False,
        "pytest_available": False,
        "python_executable": "",
        "status": "blocked_conda_not_found",
        "blockers": [],
        "commands": {
            "activate": f"conda activate {env_name}",
            "pytest": f"conda run -n {env_name} python -m pytest",
        },
    }
    if not conda_executable:
        report["blockers"] = ["conda executable was not found on PATH or common Windows install paths"]
        return report

    try:
        env_result = subprocess.run(
            [conda_executable, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        report["status"] = "blocked_conda_env_list_failed"
        report["blockers"] = [str(exc)]
        return report
    if env_result.returncode != 0:
        report["status"] = "blocked_conda_env_list_failed"
        report["blockers"] = [env_result.stderr.strip() or env_result.stdout.strip() or "conda env list failed"]
        return report

    try:
        env_payload = json.loads(env_result.stdout or "{}")
    except json.JSONDecodeError as exc:
        report["status"] = "blocked_conda_env_list_unparseable"
        report["blockers"] = [str(exc)]
        return report

    env_paths = [str(path) for path in env_payload.get("envs", [])]
    report["env_paths"] = env_paths
    report["env_available"] = any(path.replace("\\", "/").rstrip("/").endswith(f"/{env_name}") for path in env_paths)
    if not report["env_available"]:
        report["status"] = "blocked_env_not_found"
        report["blockers"] = [f"conda environment is not listed: {env_name}"]
        return report

    probe = "import sys; print(sys.executable); import pytest; print(pytest.__version__)"
    try:
        pytest_result = subprocess.run(
            [conda_executable, "run", "-n", env_name, "python", "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        report["status"] = "blocked_pytest_probe_failed"
        report["blockers"] = [str(exc)]
        return report

    output_lines = [line.strip() for line in pytest_result.stdout.splitlines() if line.strip()]
    if output_lines:
        report["python_executable"] = output_lines[0]
    report["pytest_available"] = pytest_result.returncode == 0
    if not report["pytest_available"]:
        report["status"] = "blocked_pytest_unavailable"
        report["blockers"] = [pytest_result.stderr.strip() or pytest_result.stdout.strip() or "pytest import failed"]
        return report

    report["status"] = "ready"
    report["blockers"] = []
    return report
