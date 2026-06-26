from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_pipeline():
    try:
        from src.private.training import pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Private training source is required. Run scripts/pull_private_src.py with HF credentials first."
        ) from exc
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full fresh-start Quant/ML workflow.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--market", default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push-public", action="store_true")
    parser.add_argument("--push-hf", action="store_true")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = _require_pipeline()
    result = pipeline.run_pipeline(
        pipeline.PipelineOptions(
            request=Path(args.request),
            root=Path(args.root).resolve(),
            market=args.market,
            phase=args.phase,
            fresh=args.fresh,
            resume=args.resume,
            dry_run=args.dry_run,
            push_public=args.push_public,
            push_hf=args.push_hf,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
