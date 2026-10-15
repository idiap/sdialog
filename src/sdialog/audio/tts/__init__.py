import importlib

from .base import BaseTTS, BaseVoiceCloneTTS
from .dialog_tts import BaseDialogTTS, PerTurnTTS, TurnAudio

# Engines are imported lazily, on first access, to avoid loading heavy dependencies (e.g. torch) on import
_ENGINES = {
    "Dia2TTS": ".dia2",
    "HuggingFaceTTS": ".huggingface",
    "IndexTTS": ".index",
    "KokoroTTS": ".kokoro",
    "Qwen3TTS": ".qwen3",
    "Qwen3TTSVoiceClone": ".qwen3",
}


def __getattr__(name):
    if name in _ENGINES:
        return getattr(importlib.import_module(_ENGINES[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTTS",
    "BaseVoiceCloneTTS",
    "BaseDialogTTS",
    "PerTurnTTS",
    "TurnAudio",
    "Dia2TTS",
    "KokoroTTS",
    "IndexTTS",
    "HuggingFaceTTS",
    "Qwen3TTS",
    "Qwen3TTSVoiceClone"
]
