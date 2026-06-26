from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from src.private.training.pipeline import PipelineOptions, run_pipeline


def _handle_request(path: Path, root: Path) -> None:
    running = path.parent.parent / "running" / path.name
    completed = path.parent.parent / "completed" / path.name
    failed = path.parent.parent / "failed" / path.name
    running.parent.mkdir(parents=True, exist_ok=True)
    completed.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(running))
    try:
        run_pipeline(PipelineOptions(request=running, root=root))
    except Exception as exc:
        failed_reason = failed.with_suffix(failed.suffix + ".error.txt")
        shutil.move(str(running), str(failed))
        failed_reason.write_text(str(exc), encoding="utf-8")
        raise
    else:
        shutil.move(str(running), str(completed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch request YAML files and run the shared pipeline.")
    parser.add_argument("--requests-dir", default="requests/inbox")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requests_dir = Path(args.requests_dir)
    requests_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root).resolve()
    while True:
        for request in sorted([*requests_dir.glob("*.yaml"), *requests_dir.glob("*.yml")]):
            _handle_request(request, root)
        if args.once:
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
