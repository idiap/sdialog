from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import sphn
import torch
import torch.nn.functional as F

from ..audio import MimiCodec

PathLike = Union[str, Path]


def load_mono_audio(path, target_sr: int) -> np.ndarray:
    """Read an audio file (or an ``(array, sample_rate)`` tuple), convert to mono float32,
    and resample to target_sr."""
    if isinstance(path, (tuple, list)):  # [sdialog] in-memory audio
        audio, sr = path
        audio = np.asarray(audio, dtype=np.float32)
    else:
        path = str(path)
        try:
            audio, sr = sphn.read_wav(path)
        except Exception:
            import soundfile as sf  # Local fallback

            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # sphn returns [channels, samples]; soundfile returns [samples, channels]
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[1] else 1)
    if sr != target_sr:
        if hasattr(sphn, "resample_audio"):
            audio = sphn.resample_audio(audio, sr, target_sr).astype(np.float32)
        else:
            audio = _resample_linear(audio, sr, target_sr)
    return audio


def audio_to_tensor(audio: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert mono PCM samples into shape [1, 1, T] tensor."""
    tensor = torch.from_numpy(audio).to(device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def encode_audio_tokens(mimi: MimiCodec, audio: np.ndarray) -> torch.Tensor:
    """Encode PCM audio into Mimi codebook tokens [C, T]."""
    waveform = audio_to_tensor(audio, mimi.device)
    with torch.inference_mode():
        codes, *_ = mimi.encode(waveform, return_dict=False)
    if isinstance(codes, (tuple, list)):
        codes = codes[0]
    # Mimi.encode returns [B, num_codebooks, T]; select batch 0.
    codes = codes[0].to(torch.long)
    return codes


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    length = audio.shape[0]
    new_length = max(1, int(round(length * dst_sr / src_sr)))
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        resampled = F.interpolate(tensor, size=new_length, mode="linear", align_corners=False)
    return resampled.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


__all__ = ["load_mono_audio", "audio_to_tensor", "encode_audio_tokens"]
