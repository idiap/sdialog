"""Tests for the per-turn TTS engine wrappers, with their backends replaced by stubs (no model is loaded)."""
import sys
import types

import numpy as np
import pytest


class FakeQwen3TTSModel:
    calls = []

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        obj = cls()
        obj.model_name, obj.kwargs = model, kwargs
        return obj

    def generate_custom_voice(self, text, speaker=None, **kwargs):
        FakeQwen3TTSModel.calls.append(("custom", text, speaker, kwargs))
        return [np.zeros(240, dtype=np.float32)], 24000

    def generate_voice_clone(self, text, **kwargs):
        FakeQwen3TTSModel.calls.append(("clone", text, None, kwargs))
        return [np.ones(240, dtype=np.float32)], 24000


@pytest.fixture
def qwen_tts(monkeypatch):
    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = FakeQwen3TTSModel
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    FakeQwen3TTSModel.calls = []
    return FakeQwen3TTSModel


def test_qwen3_tts_does_not_modify_the_given_kwargs(qwen_tts):
    from sdialog.audio.tts import Qwen3TTS, Qwen3TTSVoiceClone
    kwargs = {}
    Qwen3TTS(device_map="cpu", deterministic=True).generate("Hi", speaker_voice="Ryan", tts_pipeline_kwargs=kwargs)
    Qwen3TTSVoiceClone(device_map="cpu").generate("Hi", speaker_voice="ref.wav", tts_pipeline_kwargs=kwargs)
    assert kwargs == {}
    assert qwen_tts.calls[0][3] == {"language": "English", "do_sample": False}
    assert qwen_tts.calls[1][3]["ref_audio"] == "ref.wav"
