from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.public.reporting.environment_preflight import conda_environment_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local Conda environment readiness.")
    parser.add_argument("--env-name", default="btc-quant-stream")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = conda_environment_preflight(str(args.env_name))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
