from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.public.runtime import configure_native_runtime

configure_native_runtime()

from src.workflows.orchestrate_prod import main


if __name__ == "__main__":
    raise SystemExit(main())
