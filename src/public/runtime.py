from __future__ import annotations

import os
import warnings


def configure_native_runtime() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    warnings.filterwarnings(
        "ignore",
        message=r"Found Intel OpenMP .* LLVM OpenMP",
        category=RuntimeWarning,
        module=r"threadpoolctl",
    )
