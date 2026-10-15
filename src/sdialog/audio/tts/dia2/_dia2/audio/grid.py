from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def delay_frames(aligned: torch.Tensor, delays: Sequence[int], pad_id: int) -> torch.Tensor:
    channels, total = aligned.shape
    max_delay = max(delays) if delays else 0
    out = aligned.new_full((channels, total + max_delay), pad_id)
    for idx, delay in enumerate(delays):
        out[idx, delay : delay + total] = aligned[idx]
    return out


def undelay_frames(delayed: torch.Tensor, delays: Sequence[int], pad_id: int) -> torch.Tensor:
    channels, total = delayed.shape
    max_delay = max(delays) if delays else 0
    target = max(0, total - max_delay)
    out = delayed.new_full((channels, target), pad_id)
    for idx, delay in enumerate(delays):
        out[idx] = delayed[idx, delay : delay + target]
    return out


def mask_audio_logits(logits: torch.Tensor, pad_idx: int, bos_idx: int) -> torch.Tensor:
    if logits.shape[-1] == 0:
        return logits
    max_idx = logits.shape[-1] - 1
    targets = [idx for idx in (pad_idx, bos_idx) if 0 <= idx <= max_idx]
    if not targets:
        return logits
    masked = logits.clone()
    neg_inf = torch.finfo(masked.dtype).min
    for idx in targets:
        masked[..., idx] = neg_inf
    return masked


def fill_audio_channels(
    delays: Sequence[int],
    constants,
    step: int,
    step_tokens: torch.Tensor,
    audio_buf: torch.Tensor,
) -> None:
    for cb, delay in enumerate(delays):
        idx = step - delay
        in_bounds = idx >= 0 and step < audio_buf.shape[-1]
        if in_bounds:
            step_tokens[:, 2 + cb, 0] = audio_buf[:, cb, step]
        else:
            step_tokens[:, 2 + cb, 0] = constants.audio_bos


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


__all__ = [
    "delay_frames",
    "undelay_frames",
    "mask_audio_logits",
    "fill_audio_channels",
    "write_wav",
]
