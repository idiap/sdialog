"""
Dialogue-level text-to-speech interface used by :func:`sdialog.audio.synthesis.to_audio`.

A dialogue TTS engine receives all the turns of a dialogue at once (so engines designed for
dialogues, like Dia2, can model turn-taking and prosody across turns) and returns one audio
clip per turn.

Example:

    .. code-block:: python

        from sdialog.audio.tts import KokoroTTS, PerTurnTTS

        # Use any classic (one utterance at a time) TTS engine with dialog.to_audio()
        audio = dialog.to_audio(tts=PerTurnTTS(KokoroTTS()))
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .base import BaseTTS, BaseVoiceCloneTTS


@dataclass
class TurnAudio:
    """
    Audio generated for one dialogue turn.

    :ivar audio: Mono float32 waveform.
    :ivar sample_rate: Sample rate of ``audio``.
    :ivar window: Index of the generation call that produced this turn; turns generated together
                  share the same window and keep the engine's own timing between them.
    :ivar offset: Start time (in seconds) of this clip within its window.
    :ivar words: Optional word timestamps ``(word, start_seconds)`` relative to the start of ``audio``.
    """
    audio: np.ndarray
    sample_rate: int
    window: int = 0
    offset: float = 0.0
    words: List[Tuple[str, float]] = field(default_factory=list)


class BaseDialogTTS(ABC):
    """
    Abstract base class for dialogue-level text-to-speech engines.

    Subclasses implement :meth:`synthesize`, which receives the whole dialogue and returns one
    :class:`TurnAudio` per turn.

    :cvar supports_random_voices: True if the engine can make up voices when no reference voice is given.
    :cvar supports_reference_audio: True if voices can be given as reference audio (path or ``(array, sr)``).
    :cvar default_voice_database: Hugging Face voice database used by ``voices="db"`` when none is given.
    """
    supports_random_voices: bool = False
    supports_reference_audio: bool = False
    default_voice_database: Optional[str] = "sdialog/voices-libritts"

    @abstractmethod
    def synthesize(self,
                   turns: List[Tuple[int, str]],
                   voices: List[Any],
                   seed: Optional[int] = None) -> List[TurnAudio]:
        """
        Generates the audio of every turn of a dialogue.

        :param turns: List of ``(speaker_index, text)`` pairs, in dialogue order.
        :param voices: Voice of each speaker (indexed by ``speaker_index``); ``None`` lets the engine
                       choose (random voice), otherwise an engine-specific voice (e.g. a reference audio
                       path, an ``(array, sample_rate)`` tuple, or a voice name).
        :param seed: Random seed for reproducible generation.
        :return: One :class:`TurnAudio` per turn, in the same order as ``turns``.
        """
        raise NotImplementedError

    def json(self) -> dict:
        """Returns a serializable description of the engine configuration."""
        return {"engine": type(self).__name__}


class PerTurnTTS(BaseDialogTTS):
    """
    Adapter to use a classic, one-utterance-at-a-time TTS engine (:class:`BaseTTS` or
    :class:`BaseVoiceCloneTTS`, e.g. ``KokoroTTS`` or ``Qwen3TTSVoiceClone``) with ``dialog.to_audio()``.

    Each turn is generated independently, so there is no prosodic continuity between turns.

    :param engine: The TTS engine to wrap.
    :param tts_kwargs: Extra keyword arguments passed to the engine's ``generate`` method.
    """
    supports_random_voices = False

    def __init__(self, engine, tts_kwargs: Optional[dict] = None):
        if not isinstance(engine, (BaseTTS, BaseVoiceCloneTTS)):
            raise TypeError("PerTurnTTS expects a BaseTTS or BaseVoiceCloneTTS instance, "
                            f"got {type(engine).__name__}")
        self.engine = engine
        self.tts_kwargs = tts_kwargs or {}
        self.supports_reference_audio = isinstance(engine, BaseVoiceCloneTTS)
        if isinstance(engine, BaseTTS):  # engines with predefined voice names
            qwen3 = type(engine).__name__ == "Qwen3TTS"
            self.default_voice_database = "sdialog/voices-qwen3-tts" if qwen3 else "sdialog/voices-kokoro"

    def synthesize(self, turns, voices, seed=None):
        results = []
        for ix, (speaker, text) in enumerate(turns):
            audio, sr = self.engine.generate(text, speaker_voice=voices[speaker], tts_pipeline_kwargs=self.tts_kwargs)
            audio = np.asarray(audio, dtype=np.float32)
            results.append(TurnAudio(audio=audio.mean(axis=1) if audio.ndim == 2 else audio.reshape(-1),
                                     sample_rate=int(sr), window=ix))
        return results

    def json(self):
        return {"engine": "PerTurnTTS", "wrapped": type(self.engine).__name__}
