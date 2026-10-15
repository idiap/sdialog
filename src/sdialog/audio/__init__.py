"""
Audio generation for dialogues: turn any two-speaker :class:`~sdialog.Dialog` into realistic audio with
ground-truth annotations (who speaks when, what is said), at different levels of difficulty.

The main entry point is ``dialog.to_audio()`` (:func:`sdialog.audio.synthesis.to_audio`), which uses the
Dia2 dialogue TTS model by default and lets you control:

  - Voices: matched to the personas from a voice database, designed from the persona description, random,
    or given explicitly.
  - Timing: natural pauses, sampled pauses and overlapping speech.
  - Acoustics: room reverberation, background noise, telephone channel, loudness differences.
  - Output: mono or stereo (one channel per speaker), sample rate.
  - Presets: "clean", "conversational", "room", "call_center", "far_field", "hard" (or your own).

Example:

    .. code-block:: python

        audio = dialog.to_audio()                                      # clean audio, persona-matched voices
        audio = dialog.to_audio(preset="call_center", stereo=True)     # call center recording, one speaker per channel
        audio = dialog.to_audio(voices="random", reverb="medium", noise=(5, 15), overlap=0.2, seed=0)

        audio.audio, audio.sample_rate      # waveform
        audio.segments                      # turn-level annotations (speaker, text, start, end, word timestamps)
        audio.save("output/dialog")         # output/dialog.wav, .json and .rttm

        # Many dialogues in parallel (one model per GPU)
        from sdialog.audio import to_audio_batch
        to_audio_batch(dialogs, "output/", devices=["cuda:0", "cuda:1"], preset="room", seed=0)
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import importlib

from .presets import register_preset, list_presets
from .result import DialogAudio, Segment
from .synthesis import to_audio, to_audio_batch
from .voice_database import (BaseVoiceDatabase, HuggingfaceVoiceDatabase, LocalVoiceDatabase,
                             Voice, VoiceDatabase)

# Names kept for backward compatibility, loaded lazily on first access (they require heavy dependencies)
_LAZY = {
    "generate_utterances_audios": ".legacy",
    "generate_utterance": ".legacy",
    "generate_audio_room_accoustic": ".legacy",
    "device": ".legacy",
    "AudioDialog": ".dialog",
    "RoomAcousticsConfig": ".dialog",
    "Room": ".room",
    "RoomPosition": ".room",
    "AcousticsSimulator": ".acoustics_simulator",
    "SourceVolume": ".utils",
    "Role": ".utils",
    "logger": ".utils",
    "BaseTTS": ".tts",
    "BaseDialogTTS": ".tts",
    "PerTurnTTS": ".tts",
    "Dia2TTS": ".tts",
    "KokoroTTS": ".tts",
    "Qwen3TTS": ".tts",
    "Qwen3TTSVoiceClone": ".tts",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "to_audio",
    "to_audio_batch",
    "DialogAudio",
    "Segment",
    "register_preset",
    "list_presets",
    "Voice",
    "BaseVoiceDatabase",
    "HuggingfaceVoiceDatabase",
    "LocalVoiceDatabase",
    "VoiceDatabase",
    "BaseDialogTTS",
    "PerTurnTTS",
    "Dia2TTS",
    "KokoroTTS",
]
