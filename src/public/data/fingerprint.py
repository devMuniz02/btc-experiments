from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def dataset_fingerprint(frame: pd.DataFrame, config_hash: str) -> dict[str, Any]:
    sample = frame.to_json(date_format="iso", orient="split").encode("utf-8")
    digest = hashlib.sha256(config_hash.encode("utf-8") + sample).hexdigest()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    return {
        "fingerprint": digest,
        "row_count": int(len(frame)),
        "start_timestamp": timestamps.iloc[0].isoformat(),
        "end_timestamp": timestamps.iloc[-1].isoformat(),
        "columns": list(map(str, frame.columns)),
    }
