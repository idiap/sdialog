"""
Signal-level audio effects used by :func:`sdialog.audio.synthesis.to_audio`.

Every function here works on in-memory mono ``np.ndarray`` signals and is free of side effects
(no files are written), so effects can be freely combined and run in parallel.
Randomized effects take an explicit ``np.random.Generator`` for reproducibility.

Example:

    .. code-block:: python

        import numpy as np
        from sdialog.audio import effects

        rng = np.random.default_rng(0)
        x = effects.set_active_level(speech, sr=16000, target_db=-26)
        rirs, room = effects.simulate_rirs(rng, sr=16000, size="medium", n_sources=2)
        x = effects.apply_rir(x, rirs[0], sr=16000)
        x, snr = effects.add_noise(x, sr=16000, rng=rng, snr_db=20, kind="pink")
        x = effects.telephone(x, sr=16000, codec="g711")
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import numpy as np

from math import gcd
from typing import List, Optional, Tuple, Union

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")

# Simulated room presets: dimensions ranges (m), RT60 range (s) and source-to-mic distance range (m)
ROOM_SIZES = {
    "small": {"dims": ((3.0, 4.5), (2.5, 4.0), (2.4, 2.8)), "rt60": (0.2, 0.4), "distance": (0.5, 1.5)},
    "medium": {"dims": ((5.0, 8.0), (4.0, 6.0), (2.7, 3.2)), "rt60": (0.4, 0.7), "distance": (1.0, 3.0)},
    "large": {"dims": ((10.0, 20.0), (8.0, 15.0), (3.5, 6.0)), "rt60": (0.7, 1.2), "distance": (2.0, 6.0)},
}

# Telephone codec simulation parameters: band-pass (Hz), narrowband sample rate and quantization levels
CODECS = {
    "g711": {"band": (300.0, 3400.0), "sr": 8000, "levels": 128},
    "g729": {"band": (300.0, 3400.0), "sr": 8000, "levels": 8},
    "amr": {"band": (200.0, 3400.0), "sr": 8000, "levels": 16},
    "amr-wb": {"band": (50.0, 7000.0), "sr": 16000, "levels": 32},
}
CODEC_ALIASES = {"telephone": "g711", "phone": "g711", "g.711": "g711", "g.729": "g729", "amrwb": "amr-wb"}


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resamples a signal with a polyphase filter (``scipy.signal.resample_poly``)."""
    if sr_in == sr_out or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)


def to_mono(x: np.ndarray) -> np.ndarray:
    """Converts a ``(samples,)`` or ``(samples, channels)`` signal to mono float32."""
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x.reshape(-1)


def load_audio(path: str, sr: int) -> np.ndarray:
    """Loads an audio file as mono float32 at sample rate ``sr``."""
    import soundfile as sf
    x, file_sr = sf.read(path, dtype="float32", always_2d=False)
    return resample(to_mono(x), file_sr, sr)


def list_audio_files(path: str) -> List[str]:
    """Returns ``[path]`` if it is a file, or the sorted list of audio files found (recursively) under it."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Audio file or folder not found: '{path}'")
    files = sorted(os.path.join(root, f)
                   for root, _, names in os.walk(path)
                   for f in names if f.lower().endswith(AUDIO_EXTENSIONS))
    if not files:
        raise FileNotFoundError(f"No audio files ({', '.join(AUDIO_EXTENSIONS)}) found under '{path}'")
    return files


def _frame_energy_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> Tuple[np.ndarray, int]:
    n = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(x) // n)
    frames = x[:n_frames * n].reshape(n_frames, n) if len(x) >= n else x.reshape(1, -1)
    return 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12), n


def active_level_db(x: np.ndarray, sr: int, rel_db: float = 35.0) -> float:
    """
    Simplified active speech level (dBFS): mean power of the frames within ``rel_db`` of the loudest one.

    Unlike plain RMS, it is not affected by the amount of silence in the signal.
    """
    if len(x) == 0:
        return -120.0
    e, _ = _frame_energy_db(x, sr)
    active = e[e > e.max() - rel_db]
    return float(10 * np.log10(np.mean(10 ** (active / 10)) + 1e-12))


def set_active_level(x: np.ndarray, sr: int, target_db: float = -26.0) -> np.ndarray:
    """Scales ``x`` so that its active speech level is ``target_db`` dBFS."""
    if len(x) == 0 or not np.any(x):
        return x
    return (x * 10 ** ((target_db - active_level_db(x, sr)) / 20)).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, top_db: float = 40.0, pad: float = 0.05) -> Tuple[np.ndarray, int]:
    """
    Trims leading and trailing frames more than ``top_db`` below the loudest frame.

    :return: The trimmed signal and the number of samples removed at the beginning.
    """
    if len(x) == 0 or not np.any(x):
        return x, 0
    e, n = _frame_energy_db(x, sr)
    idx = np.where(e > e.max() - top_db)[0]
    pad = int(pad * sr)
    start = max(0, idx[0] * n - pad)
    end = min(len(x), (idx[-1] + 1) * n + pad)
    return x[start:end], start


def soft_limit(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Sample-wise soft clipper: samples above ``threshold`` are smoothly compressed below 1.0."""
    x = np.array(x, dtype=np.float32, copy=True)
    a = np.abs(x)
    over = a > threshold
    x[over] = np.sign(x[over]) * (threshold + (1 - threshold) * np.tanh((a[over] - threshold) / (1 - threshold)))
    return x


# ---------------------------------------------------------------------------------------------------------------- RIR
def simulate_rirs(rng: np.random.Generator,
                  sr: int,
                  size: str = "medium",
                  n_sources: int = 2) -> Tuple[List[np.ndarray], dict]:
    """
    Simulates the impulse responses from ``n_sources`` speaker positions to one microphone in a random
    shoebox room, using ``pyroomacoustics`` (image source method).

    :param rng: Random generator used to draw the room, RT60 and positions.
    :param sr: Sample rate of the returned impulse responses.
    :param size: Room size preset: "small", "medium" or "large".
    :param n_sources: Number of speaker positions (one impulse response each).
    :return: The list of impulse responses and a dictionary describing the simulated room.
    """
    try:
        import pyroomacoustics as pra
    except ImportError as e:
        raise ImportError("Room simulation requires pyroomacoustics: `pip install pyroomacoustics`.") from e

    if size not in ROOM_SIZES:
        raise ValueError(f"Unknown room size '{size}'. Valid values: {list(ROOM_SIZES)}")
    preset = ROOM_SIZES[size]
    dims = np.array([rng.uniform(*r) for r in preset["dims"]])
    rt60 = float(rng.uniform(*preset["rt60"]))
    try:
        absorption, max_order = pra.inverse_sabine(rt60, dims)
        materials = pra.Material(absorption)
    except ValueError:  # RT60 not reachable for this room: fall back to a fixed absorption
        materials, max_order = pra.Material(0.2), 17
    room = pra.ShoeBox(dims, fs=sr, materials=materials, max_order=min(max_order, 30))

    margin = 0.5
    mic = np.array([rng.uniform(margin, dims[0] - margin),
                    rng.uniform(margin, dims[1] - margin),
                    rng.uniform(0.8, 1.5)])
    sources = []
    for _ in range(n_sources):
        distance = rng.uniform(*preset["distance"])
        angle = rng.uniform(0, 2 * np.pi)
        pos = mic + np.array([distance * np.cos(angle), distance * np.sin(angle), 0.0])
        pos[0] = np.clip(pos[0], margin, dims[0] - margin)
        pos[1] = np.clip(pos[1], margin, dims[1] - margin)
        pos[2] = rng.uniform(1.2, min(1.8, dims[2] - 0.3))
        sources.append(pos)
        room.add_source(pos.tolist())
    room.add_microphone(mic.tolist())
    room.compute_rir()

    rirs = [np.asarray(room.rir[0][i], dtype=np.float32) for i in range(n_sources)]
    info = {"type": "simulated", "size": size, "dimensions": dims.round(2).tolist(), "rt60": round(rt60, 3),
            "microphone": mic.round(2).tolist(), "sources": [s.round(2).tolist() for s in sources]}
    return rirs, info


def load_rir(path: str, rng: np.random.Generator, sr: int) -> Tuple[np.ndarray, str]:
    """Loads a random impulse response from ``path`` (an audio file or a folder of them)."""
    files = list_audio_files(path)
    file = files[int(rng.integers(len(files)))]
    return load_audio(file, sr), file


def apply_rir(x: np.ndarray, rir: np.ndarray, sr: int) -> np.ndarray:
    """
    Convolves ``x`` with the impulse response ``rir``.

    The impulse response is first aligned on its direct path (the delay before it is removed) so that
    speech onsets do not move and time annotations stay valid. The output keeps the reverberation tail
    (it is longer than ``x``) and is re-leveled to the RMS of the input.
    """
    from scipy.signal import fftconvolve
    if len(x) == 0 or len(rir) == 0:
        return x
    direct = int(np.argmax(np.abs(rir)))
    rir = rir[max(0, direct - int(0.001 * sr)):]
    y = fftconvolve(x, rir).astype(np.float32)
    x_rms, y_rms = np.sqrt(np.mean(x.astype(np.float64) ** 2)), np.sqrt(np.mean(y.astype(np.float64) ** 2))
    return y * (x_rms / y_rms) if y_rms > 0 else y


# -------------------------------------------------------------------------------------------------------------- noise
def colored_noise(n: int, rng: np.random.Generator, kind: str = "white") -> np.ndarray:
    """Generates ``n`` samples of unit-RMS "white", "pink" (1/f) or "brown" (1/f^2) noise."""
    exponents = {"white": 0.0, "pink": 1.0, "brown": 2.0}
    if kind not in exponents:
        raise ValueError(f"Unknown noise type '{kind}'. Valid values: {list(exponents)} or an audio path.")
    white = rng.standard_normal(n)
    if exponents[kind] > 0:
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1] if n > 1 else 1.0
        white = np.fft.irfft(spectrum / freqs ** (exponents[kind] / 2), n)
    white = white - white.mean()
    return (white / (np.sqrt(np.mean(white ** 2)) + 1e-12)).astype(np.float32)


def noise_from_files(path: str, n: int, sr: int, rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    """Builds ``n`` samples of unit-RMS noise from a random file under ``path``, looped from a random offset."""
    files = list_audio_files(path)
    file = files[int(rng.integers(len(files)))]
    noise = load_audio(file, sr)
    if len(noise) == 0 or not np.any(noise):
        raise ValueError(f"Noise file '{file}' is empty or silent")
    offset = int(rng.integers(len(noise)))
    noise = np.tile(np.roll(noise, -offset), int(np.ceil(n / len(noise))))[:n]
    return noise / (np.sqrt(np.mean(noise.astype(np.float64) ** 2)) + 1e-12), file


def add_noise(x: np.ndarray,
              sr: int,
              rng: np.random.Generator,
              snr_db: Union[float, Tuple[float, float]] = 20.0,
              kind: str = "white",
              speech_level_db: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    """
    Adds background noise to ``x`` at the given signal-to-noise ratio.

    :param snr_db: SNR in dB, or a ``(min, max)`` range to draw it from. The signal level is the active
                   speech level of ``x`` (so pauses do not lower it).
    :param kind: "white", "pink", "brown", or a path to a noise audio file/folder.
    :param speech_level_db: Reference speech level; defaults to the active level of ``x``.
    :return: The noisy signal and a dictionary describing the noise that was added.
    """
    if isinstance(snr_db, (tuple, list)):
        snr_db = float(rng.uniform(*snr_db))
    info = {"type": kind, "snr_db": round(float(snr_db), 2)}
    if kind in ("white", "pink", "brown"):
        noise = colored_noise(len(x), rng, kind)
    else:
        noise, info["file"] = noise_from_files(kind, len(x), sr, rng)
    level = active_level_db(x, sr) if speech_level_db is None else speech_level_db
    return (x + noise * 10 ** ((level - snr_db) / 20)).astype(np.float32), info


# ---------------------------------------------------------------------------------------------------------- telephone
def telephone(x: np.ndarray, sr: int, codec: str = "g711") -> np.ndarray:
    """
    Simulates a telephone channel: band-pass filtering, narrowband resampling and mu-law companding with
    the codec's quantization (a coarse approximation of the codec's distortion).

    :param codec: "g711" (alias "telephone"), "g729", "amr" or "amr-wb".
    :return: The processed signal at the original sample rate ``sr``.
    """
    from scipy.signal import butter, sosfilt
    name = CODEC_ALIASES.get(codec.lower(), codec.lower())
    if name not in CODECS:
        raise ValueError(f"Unknown codec '{codec}'. Valid values: {list(CODECS) + list(CODEC_ALIASES)}")
    params = CODECS[name]
    low, high = params["band"]
    high = min(high, 0.45 * sr)
    y = sosfilt(butter(5, [low, high], btype="bandpass", fs=sr, output="sos"), x)
    y = resample(y, sr, params["sr"])
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > 0:
        mu, levels = 255.0, params["levels"]
        c = np.sign(y) * np.log1p(mu * np.abs(y / peak)) / np.log1p(mu)
        c = np.round(c * levels) / levels
        y = np.sign(c) * np.expm1(np.abs(c) * np.log1p(mu)) / mu * peak
    y = resample(y, params["sr"], sr)[:len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)
