from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Precision:
    compute: torch.dtype
    logits: torch.dtype


def resolve_precision(kind: str | None, device: torch.device) -> Precision:
    normalized = (kind or "auto").lower()
    if normalized == "auto":
        normalized = "bfloat16" if device.type == "cuda" else "float32"
    if normalized == "bfloat16":
        compute = torch.bfloat16 if device.type == "cuda" else torch.float32
        return Precision(compute=compute, logits=torch.float32)
    if normalized == "float32":
        return Precision(compute=torch.float32, logits=torch.float32)
    raise ValueError(f"Unsupported dtype '{kind}'")
