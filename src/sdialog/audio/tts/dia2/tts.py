"""
Dia2 dialogue text-to-speech engine (https://github.com/nari-labs/dia2), the default engine of
``dialog.to_audio()``.

Dia2 generates a whole two-speaker conversation in a single pass (natural turn-taking and prosody),
optionally conditioned on a reference audio ("prefix") per speaker, and returns word timestamps.

Example:

    .. code-block:: python

        from sdialog.audio.tts import Dia2TTS

        tts = Dia2TTS("nari-labs/Dia2-2B", device="cuda")
        audio = dialog.to_audio(tts=tts, voices="random", seed=0)
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import sys
import logging
import numpy as np

from typing import Any, List, Optional, Tuple

from ..dialog_tts import BaseDialogTTS, TurnAudio

logger = logging.getLogger(__name__)

MIN_PYTHON = (3, 10)
SPEAKER_TAG_RE = re.compile(r"\[S\d+\]")


def check_python_version():
    """Raises a ``RuntimeError`` with a helpful message if the Python version cannot run Dia2."""
    if sys.version_info[:2] < MIN_PYTHON:
        raise RuntimeError(
            f"Dia2 (the default audio engine) requires Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} "
            f"(found {sys.version_info[0]}.{sys.version_info[1]}). Upgrade Python, or use another engine, "
            "e.g. `dialog.to_audio(tts=PerTurnTTS(KokoroTTS()))`."
        )


def clean_text(text: str) -> str:
    """Prepares a turn text for Dia2: removes speaker tags and newlines."""
    return " ".join(SPEAKER_TAG_RE.sub(" ", text).split())


def count_words(text: str) -> int:
    """Number of words Dia2 timestamps for ``text`` (mirrors the tokenization of its script parser)."""
    return len(text.replace("’", "'").replace(":", " ").split())


class Dia2TTS(BaseDialogTTS):
    """
    Dia2 dialogue TTS engine.

    Consecutive turns are generated together in windows of up to ``max_window_words`` words (Dia2 can
    generate up to ~2 minutes per call) and each window is cut into turns using Dia2's word timestamps.
    If the timestamps do not match the script, the window is regenerated one turn at a time.

    Voices: each speaker can be conditioned on a reference audio (a path or an ``(array, sample_rate)``
    tuple). Speakers without a reference get a random voice, which is then reused as reference for the
    following windows so that it stays the same across the whole dialogue.

    :param model: Hugging Face repository of the Dia2 model ("nari-labs/Dia2-2B" or "nari-labs/Dia2-1B").
    :param device: Torch device; defaults to "cuda" if available, otherwise "cpu" (very slow).
    :param dtype: "auto" (bfloat16 on GPU, float32 on CPU), "bfloat16" or "float32".
    :param cfg_scale: Classifier-free guidance scale (1.0 disables guidance).
    :param text_temperature: Sampling temperature of the text stream.
    :param audio_temperature: Sampling temperature of the audio stream.
    :param top_k: Top-k sampling for both streams.
    :param max_window_words: Maximum number of words generated in a single Dia2 call.
    :param reference_seconds: Maximum length of reference voice audio (longer references are cut).
    :param use_cuda_graph: Use CUDA graphs to speed up generation (GPU only).
    :param whisper_model: Whisper model used by Dia2 to align the reference voices.
    """
    supports_random_voices = True
    supports_reference_audio = True

    def __init__(self,
                 model: str = "nari-labs/Dia2-2B",
                 device: Optional[str] = None,
                 dtype: str = "auto",
                 cfg_scale: float = 2.0,
                 text_temperature: float = 0.6,
                 audio_temperature: float = 0.8,
                 top_k: int = 50,
                 max_window_words: int = 170,
                 reference_seconds: float = 8.0,
                 use_cuda_graph: bool = True,
                 whisper_model: str = "openai/whisper-large-v3"):
        check_python_version()
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.cfg_scale = cfg_scale
        self.text_temperature = text_temperature
        self.audio_temperature = audio_temperature
        self.top_k = top_k
        self.max_window_words = max_window_words
        self.reference_seconds = reference_seconds
        self.use_cuda_graph = use_cuda_graph
        self.whisper_model = whisper_model
        self._model = None
        self._config = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from ._dia2 import Dia2, GenerationConfig, SamplingConfig
            from ._dia2.runtime import voice_clone
        except ImportError as e:
            raise ImportError(f"Dia2 dependencies are missing ({e}). Install them with `pip install sdialog[audio]` "
                              "(requires Python >= 3.10).") from e
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                logger.warning("No GPU found: running Dia2 on CPU will be very slow.")
        voice_clone.WHISPER_MODEL_NAME = self.whisper_model
        self._model = Dia2.from_repo(self.model_name, device=self.device, dtype=self.dtype)
        self._config = GenerationConfig(
            cfg_scale=self.cfg_scale,
            text=SamplingConfig(temperature=self.text_temperature, top_k=self.top_k),
            audio=SamplingConfig(temperature=self.audio_temperature, top_k=self.top_k),
            use_cuda_graph=self.use_cuda_graph,
        )

    def _generate(self, script: str, prefix_1: Any, prefix_2: Any,
                  seed: Optional[int]) -> Tuple[np.ndarray, int, List[Tuple[str, float]]]:
        """Runs one Dia2 call; returns the waveform, its sample rate and the word timestamps."""
        if prefix_1 is None:  # Dia2 requires speaker 1's prefix to use speaker 2's
            prefix_2 = None
        result = self._model.generate(script, config=self._config,
                                      prefix_speaker_1=self._reference(prefix_1),
                                      prefix_speaker_2=self._reference(prefix_2),
                                      include_prefix=False, verbose=False, seed=seed)
        wav = result.waveform.detach().float().cpu().numpy().reshape(-1)
        return wav, int(result.sample_rate), list(result.timestamps)

    def _reference(self, voice: Any) -> Any:
        """Normalizes a reference voice to a path or a (possibly shortened) ``(array, sr)`` tuple."""
        if voice is None or isinstance(voice, str):
            return voice
        audio, sr = voice
        audio = np.asarray(audio, dtype=np.float32)
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio.reshape(-1)
        return audio[:int(self.reference_seconds * sr)], int(sr)

    def _windows(self, turns: List[Tuple[int, str]]) -> List[List[int]]:
        """Splits turn indexes into runs of consecutive turns within the word budget."""
        windows, current, n_words = [], [], 0
        for ix, (_, text) in enumerate(turns):
            n = count_words(text)
            if current and n_words + n > self.max_window_words:
                windows.append(current)
                current, n_words = [], 0
            current.append(ix)
            n_words += n
        return windows + ([current] if current else [])

    def synthesize(self, turns, voices, seed=None):
        if len(voices) > 2 or any(speaker > 1 for speaker, _ in turns):
            raise ValueError("Dia2 supports dialogues with at most two speakers.")
        self._load()
        texts = [clean_text(text) for _, text in turns]
        voices = list(voices) + [None] * (2 - len(voices))
        results: List[Optional[TurnAudio]] = [None] * len(turns)
        n_windows = 0
        generated = {0: [], 1: []}  # audio generated so far for speakers with a random voice

        for w_ix, window in enumerate(self._windows(turns)):
            w_seed = None if seed is None else seed + w_ix
            # Dia2's [S1] is always the first speaker of the script, so map speakers per window
            first = turns[window[0]][0]
            tag = {first: "[S1]", 1 - first: "[S2]"}
            script = " ".join(f"{tag[turns[ix][0]]} {texts[ix]}" for ix in window)
            wav, sr, timestamps = self._generate(script, voices[first], voices[1 - first], w_seed)

            counts = [count_words(texts[ix]) for ix in window]
            if len(window) > 1 and len(timestamps) != sum(counts):
                logger.info(f"Dia2 timestamps do not match the script ({len(timestamps)} != {sum(counts)} words), "
                            "generating the turns of this window one by one.")
                for ix in window:
                    speaker = turns[ix][0]
                    t_wav, t_sr, t_ts = self._generate(f"[S1] {texts[ix]}", voices[speaker], None, w_seed)
                    a = self._start_sample(t_ts, t_sr)
                    t_wav, t_ts = t_wav[a:], [(word, t - a / t_sr) for word, t in t_ts]
                    t_wav = self._trim_tail(t_wav, t_ts, t_sr)
                    results[ix] = TurnAudio(t_wav, t_sr, window=n_windows, words=t_ts)
                    n_windows += 1
                    self._keep_random_voice(voices, generated, speaker, t_wav, t_sr)
            else:
                # each turn starts at its first word; for the first turn this also drops any audio generated
                # before it (e.g. the end of the reference voice prompt leaking into the output)
                word_starts = np.cumsum([0] + counts[:-1])
                bounds = [self._start_sample(timestamps[k:k + 1], sr) for k in word_starts] + [len(wav)]
                word_ix = 0
                for k, ix in enumerate(window):
                    a, b = bounds[k], bounds[k + 1]
                    words = [(word, t - a / sr) for word, t in timestamps[word_ix:word_ix + counts[k]]]
                    if k == len(window) - 1:  # Dia2 may keep talking after the end of the script
                        b = a + len(self._trim_tail(wav[a:b], words, sr))
                    word_ix += counts[k]
                    results[ix] = TurnAudio(wav[a:b], sr, window=n_windows, offset=a / sr, words=words)
                    self._keep_random_voice(voices, generated, turns[ix][0], wav[a:b], sr)
                n_windows += 1
        return results

    @staticmethod
    def _start_sample(timestamps: List[Tuple[str, float]], sr: int, lead: float = 0.08) -> int:
        """Sample where a turn starts: slightly before its first word timestamp (0 if there is none)."""
        return max(0, int((timestamps[0][1] - lead) * sr)) if timestamps else 0

    @staticmethod
    def _trim_tail(audio: np.ndarray, words: List[Tuple[str, float]], sr: int, min_pause: float = 0.3,
                   after_last_word: float = 0.25, rel_db: float = 35.0) -> np.ndarray:
        """
        Cuts speech that Dia2 sometimes adds after the end of the script (e.g. an extra "Okay."): the audio is cut
        at the first pause of at least ``min_pause`` seconds after the start of the last scripted word.
        """
        if not words or len(audio) == 0:
            return audio
        frame = int(0.02 * sr)
        n_frames = len(audio) // frame
        if n_frames == 0:
            return audio
        energy = 10 * np.log10(np.mean(audio[:n_frames * frame].reshape(n_frames, frame) ** 2, axis=1) + 1e-12)
        silent = energy < energy.max() - rel_db
        first = int((words[-1][1] + after_last_word) / 0.02)
        needed, run = int(min_pause / 0.02), 0
        for ix in range(max(first, 0), n_frames):
            run = run + 1 if silent[ix] else 0
            if run >= needed:
                return audio[:(ix - run + 1) * frame + int(0.1 * sr)]
        return audio

    @staticmethod
    def _keep_random_voice(voices: list, generated: dict, speaker: int, audio: np.ndarray, sr: int,
                           min_seconds: float = 2.0):
        """
        Speakers with a random voice keep it in the next windows: once enough of their generated audio has been
        collected (across windows), it becomes their reference voice.
        """
        if voices[speaker] is not None:
            return
        generated[speaker].append(audio)
        if sum(len(clip) for clip in generated[speaker]) >= min_seconds * sr:
            voices[speaker] = (np.concatenate(generated[speaker]), sr)

    def json(self):
        return {"engine": "Dia2TTS", "model": self.model_name, "cfg_scale": self.cfg_scale,
                "text_temperature": self.text_temperature, "audio_temperature": self.audio_temperature,
                "top_k": self.top_k, "max_window_words": self.max_window_words}
