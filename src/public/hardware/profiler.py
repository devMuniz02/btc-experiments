from __future__ import annotations

import os
import platform


def build_static_hardware_profile(sequence_lengths: list[int] | None = None, batch_sizes: list[int] | None = None) -> dict[str, object]:
    torch_available = False
    torch_version = ""
    cuda_available = False
    try:
        import torch

        torch_available = True
        torch_version = str(getattr(torch, "__version__", ""))
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "profile_kind": "static_safe_default",
        "profile_mode": "public_safe_no_allocation",
        "cpu_available": True,
        "cpu_count": int(os.cpu_count() or 1),
        "torch_available": torch_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "selected_device": "cuda" if cuda_available else "cpu",
        "safe_sequence_length": min(sequence_lengths or [24]),
        "safe_batch_size": min(batch_sizes or [32]),
        "oom_fallback_enabled": True,
        "streaming_dataloader_fallback": True,
        "training_allowed": True,
        "training_gate": "hardware profile completed before training",
        "platform": platform.platform(),
    }
