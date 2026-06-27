from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


REQUEST_ROOT = "requests"
MARKETS_ROOT = "markets"
LIVE_DATA_ROOT = "live_data"


def env_value(name: str) -> str:
    return os.environ.get(name, "").strip()


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        from src.public.config.loader import _simple_yaml_load

        return _simple_yaml_load(text)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def dump_yaml(payload: dict[str, Any]) -> str:
    if yaml is None:
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for child_key, child_value in value.items():
                    lines.append(f"  {child_key}: {_format_simple_yaml_scalar(child_value)}")
            else:
                lines.append(f"{key}: {_format_simple_yaml_scalar(value)}")
        return "\n".join(lines) + "\n"
    return yaml.safe_dump(payload, sort_keys=False)


def _format_simple_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_format_simple_yaml_scalar(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#{}[]&,"):
        return json.dumps(text)
    return text


def market_id_from_config(config: dict[str, Any]) -> str:
    return str(config.get("market", {}).get("market_id", "")).strip()


def phase_value(config: dict[str, Any]) -> str:
    value = config.get("current_phase", "none")
    return "none" if value is None else str(value).strip()


def phase_folder(phase: str | int) -> str:
    text = str(phase).strip()
    return text if text.startswith("phase") else f"phase{text}"


def anonymized_dataset_id(market_id: str, phase: str | int, variation: str) -> str:
    import hashlib

    raw = f"{market_id}:{phase_folder(phase)}:{variation}".encode("utf-8")
    return "ds_" + hashlib.sha256(raw).hexdigest()[:16]


@contextmanager
def workflow_lock(root: Path, name: str):
    lock_dir = root / "private_hf_cache" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Workflow lock is already held: {lock_path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"lock": name, "pid": os.getpid()}))
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class HfSettings:
    repo_id: str
    token: str
    repo_type: str = "model"

    @classmethod
    def from_env_or_args(cls, *, repo_id: str = "", token: str = "", repo_type: str = "model") -> "HfSettings":
        resolved_repo = (repo_id or env_value("HF_REPO_ID")).strip()
        resolved_token = (token or env_value("HF_TOKEN")).strip()
        if not resolved_repo or not resolved_token:
            raise RuntimeError("HF_TOKEN and HF_REPO_ID are required for workflow parity.")
        return cls(repo_id=resolved_repo, token=resolved_token, repo_type=repo_type)


class HfStateClient:
    def __init__(self, settings: HfSettings) -> None:
        from huggingface_hub import HfApi

        self.settings = settings
        self.api = HfApi(token=settings.token)

    def list_files(self, prefix: str = "") -> list[str]:
        files = self.api.list_repo_files(repo_id=self.settings.repo_id, repo_type=self.settings.repo_type)
        normalized = prefix.strip("/")
        if not normalized:
            return sorted(files)
        return sorted(path for path in files if path.startswith(f"{normalized}/") or path == normalized)

    def download_file(self, hf_path: str, local_path: Path) -> Path:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=self.settings.repo_id,
            repo_type=self.settings.repo_type,
            token=self.settings.token,
            filename=hf_path.strip("/"),
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(Path(downloaded).read_bytes())
        return local_path

    def upload_file(self, local_path: Path, hf_path: str, *, commit_message: str) -> None:
        self.api.upload_file(
            repo_id=self.settings.repo_id,
            repo_type=self.settings.repo_type,
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path.strip("/"),
            commit_message=commit_message,
        )

    def commit_files(self, files: list[tuple[Path, str]], *, commit_message: str) -> None:
        self.commit_changes(
            add_files=files,
            delete_paths=[],
            commit_message=commit_message,
        )

    def upload_text(self, text: str, hf_path: str, *, tmp_root: Path, commit_message: str) -> None:
        tmp_path = tmp_root / "_hf_upload" / hf_path.strip("/")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(text, encoding="utf-8")
        self.upload_file(tmp_path, hf_path, commit_message=commit_message)

    def delete_file(self, hf_path: str, *, commit_message: str) -> None:
        self.api.delete_file(
            repo_id=self.settings.repo_id,
            repo_type=self.settings.repo_type,
            path_in_repo=hf_path.strip("/"),
            commit_message=commit_message,
        )

    def delete_files(self, hf_paths: list[str], *, commit_message: str) -> None:
        self.commit_changes(add_files=[], delete_paths=hf_paths, commit_message=commit_message)

    def commit_changes(
        self,
        *,
        add_files: list[tuple[Path, str]],
        delete_paths: list[str],
        commit_message: str,
    ) -> None:
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub import CommitOperationDelete

        operations = [
            CommitOperationAdd(path_in_repo=hf_path.strip("/"), path_or_fileobj=str(local_path))
            for local_path, hf_path in add_files
        ]
        operations.extend(CommitOperationDelete(path_in_repo=path.strip("/")) for path in delete_paths)
        if not operations:
            return
        self.api.create_commit(
            repo_id=self.settings.repo_id,
            repo_type=self.settings.repo_type,
            operations=operations,
            commit_message=commit_message,
        )


def pull_hf_folder(client: HfStateClient, *, prefix: str, local_dir: Path) -> list[Path]:
    downloaded: list[Path] = []
    for hf_path in client.list_files(prefix):
        if hf_path.endswith("/"):
            continue
        local_path = local_dir / hf_path
        downloaded.append(client.download_file(hf_path, local_path))
    return downloaded


def _current_branch(root: Path) -> str:
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not branch or branch == "HEAD":
        return "main"
    return branch


def _push_command(root: Path, target_branch: str | None) -> list[str]:
    if target_branch:
        return ["git", "push", "origin", f"HEAD:{target_branch}"]
    return ["git", "push"]


def _push_with_rebase_retry(root: Path, target_branch: str | None = None) -> dict[str, Any]:
    first_push = subprocess.run(_push_command(root, target_branch), cwd=root)
    if first_push.returncode == 0:
        return {"push_attempts": 1, "rebased": False, "branch": target_branch or _current_branch(root)}

    branch = target_branch or _current_branch(root)
    subprocess.run(["git", "fetch", "origin", branch], cwd=root, check=True)
    try:
        # During rebase, "theirs" is the commit being replayed. Generated public
        # artifacts should use the freshly produced local snapshot on conflict.
        subprocess.run(["git", "rebase", "-X", "theirs", f"origin/{branch}"], cwd=root, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "rebase", "--abort"], cwd=root)
        raise
    subprocess.run(_push_command(root, target_branch), cwd=root, check=True)
    return {"push_attempts": 2, "rebased": True, "branch": branch}


def _safe_relative_path(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe artifact path: {path}")
    return relative


def _copy_artifact_path(root: Path, worktree: Path, path: str) -> None:
    relative = _safe_relative_path(path)
    source = root / relative
    target = worktree / relative
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def _remove_non_artifact_roots(worktree: Path, allowed_roots: set[str]) -> None:
    for child in worktree.iterdir():
        if child.name == ".git" or child.name in allowed_roots:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _git_stdout(root: Path, command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(command, cwd=root, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def _copy_git_identity(source_root: Path, target_root: Path) -> None:
    name = _git_stdout(source_root, ["git", "config", "--get", "user.name"], check=False)
    email = _git_stdout(source_root, ["git", "config", "--get", "user.email"], check=False)
    subprocess.run(["git", "config", "user.name", name or "github-actions[bot]"], cwd=target_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", email or "41898282+github-actions[bot]@users.noreply.github.com"],
        cwd=target_root,
        check=True,
    )


def _copy_git_auth_config(source_root: Path, target_root: Path) -> None:
    patterns = [
        r"http\..*\.extraheader",
        r"url\..*\.insteadOf",
        r"credential\..*",
        r"credential\.helper",
    ]
    for pattern in patterns:
        output = _git_stdout(source_root, ["git", "config", "--local", "--get-regexp", pattern], check=False)
        for line in output.splitlines():
            if not line.strip() or " " not in line:
                continue
            key, value = line.split(" ", 1)
            result = subprocess.run(
                ["git", "config", "--local", key, value],
                cwd=target_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to copy git auth config key {key!r} into artifact repo.")


def _push_artifact_branch(root: Path, *, paths: list[str], message: str, target_branch: str) -> dict[str, Any]:
    allowed_roots = {_safe_relative_path(path).parts[0] for path in paths}
    tmp_parent = root / "tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"artifact-{target_branch.replace('/', '-')}-", dir=tmp_parent) as tmp:
        artifact_repo = Path(tmp) / "repo"
        remote_url = _git_stdout(root, ["git", "remote", "get-url", "origin"])
        subprocess.run(["git", "init", str(artifact_repo)], cwd=root, check=True)
        subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=artifact_repo, check=True)
        _copy_git_identity(root, artifact_repo)
        _copy_git_auth_config(root, artifact_repo)
        fetch = subprocess.run(["git", "fetch", "origin", target_branch], cwd=artifact_repo)
        if fetch.returncode == 0:
            subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=artifact_repo, check=True)
        else:
            subprocess.run(["git", "checkout", "--orphan", target_branch], cwd=artifact_repo, check=True)
            subprocess.run(["git", "rm", "-r", "--cached", "."], cwd=artifact_repo, check=False)
        _remove_non_artifact_roots(artifact_repo, allowed_roots)
        for path in paths:
            _copy_artifact_path(root, artifact_repo, path)
        subprocess.run(["git", "add", "-A", "-f"], cwd=artifact_repo, check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=artifact_repo)
        if diff.returncode == 0:
            return {"status": "skipped", "reason": "no staged changes", "branch": target_branch}
        subprocess.run(["git", "commit", "-m", message], cwd=artifact_repo, check=True)
        push = ["git", "push", "origin", f"HEAD:{target_branch}"]
        if fetch.returncode == 0:
            push = ["git", "push", "--force-with-lease", "origin", f"HEAD:{target_branch}"]
        subprocess.run(push, cwd=artifact_repo, check=True)
    return {"status": "pushed", "paths": paths, "message": message, "branch": target_branch}


def explicit_git_push(
    *,
    root: Path,
    paths: list[str],
    message: str,
    enabled: bool,
    target_branch: str | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "skipped", "reason": "push_github flag not set"}
    existing = [path for path in paths if (root / path).exists()]
    if not existing:
        return {"status": "skipped", "reason": "no public paths exist to push"}
    if target_branch:
        return _push_artifact_branch(root, paths=existing, message=message, target_branch=target_branch)
    preexisting = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if preexisting:
        raise RuntimeError("Refusing to commit because the git index already has staged files.")
    subprocess.run(["git", "add", "-f", "--", *existing], cwd=root, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root)
    if diff.returncode == 0:
        return {"status": "skipped", "reason": "no staged changes"}
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True)
    push_result = _push_with_rebase_retry(root, target_branch=target_branch)
    return {"status": "pushed", "paths": existing, "message": message, **push_result}
