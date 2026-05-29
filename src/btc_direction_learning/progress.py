from __future__ import annotations

import time


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressLogger:
    def __init__(self, label: str, total_steps: int | None = None) -> None:
        self.label = label
        self.total_steps = max(1, total_steps) if total_steps is not None else None
        self.start_time = time.perf_counter()

    def log(self, current_step: int, message: str) -> None:
        elapsed = time.perf_counter() - self.start_time
        if self.total_steps is None:
            prefix = f"[{self.label}] {current_step} finished"
        else:
            prefix = f"[{self.label}] {current_step}/{self.total_steps} finished"
        print(f"{prefix} | elapsed={_format_duration(elapsed)} | {message}", flush=True)
