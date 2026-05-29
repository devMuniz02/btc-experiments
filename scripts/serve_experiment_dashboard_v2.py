from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.serve_experiment_dashboard as legacy


DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "EXPERIMENTSV2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a localhost dashboard for EXPERIMENTSV2.")
    parser.add_argument("--host", default=legacy.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=legacy.DEFAULT_PORT)
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    store = legacy.ExperimentStore(experiments_root)
    handler_class = type(
        "BoundExperimentDashboardV2Handler",
        (legacy.ExperimentDashboardHandler,),
        {"store": store},
    )
    server = legacy.ThreadingHTTPServer((args.host, args.port), handler_class)
    print(
        legacy.json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "experiments_root": str(experiments_root),
                "url": f"http://{args.host}:{args.port}/experiments",
                "experiment_count": len(store.list_experiments()),
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
