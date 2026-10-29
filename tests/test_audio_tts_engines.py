"""Tests for the per-turn TTS engine wrappers, with their backends replaced by stubs (no model is loaded)."""
import sys
import types

import numpy as np
import pytest

from sdialog.audio.normalizers import LowercaseNormalizer


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


def test_qwen3_tts(qwen_tts):
    from sdialog.audio.tts import Qwen3TTS
    tts = Qwen3TTS(device_map="cpu", text_normalizers=[LowercaseNormalizer()])
    audio, sr = tts.generate("Hello “World”", speaker_voice="Ryan", tts_pipeline_kwargs={"language": "French"})
    assert sr == 24000 and audio.shape == (240,)
    assert qwen_tts.calls[-1] == ("custom", 'hello "world"', "Ryan", {"language": "French"})
    assert Qwen3TTS(unicode_to_ascii=False).text_normalizers == []


def test_qwen3_voice_clone(qwen_tts):
    from sdialog.audio.tts import Qwen3TTSVoiceClone
    tts = Qwen3TTSVoiceClone(device_map="cpu", deterministic=True)
    audio, sr = tts.generate("Hi", speaker_voice=(np.array([0, 32767, -32768], dtype=np.int16), 16000))
    kwargs = qwen_tts.calls[-1][3]
    ref, ref_sr = kwargs["ref_audio"]
    assert ref_sr == 16000 and ref.dtype == np.float32 and np.abs(ref).max() <= 1
    assert kwargs["x_vector_only_mode"] and kwargs["do_sample"] is False
    with pytest.raises(ValueError):
        tts.generate("Hi")


def test_qwen3_normalize_audio():
    from sdialog.audio.tts.qwen3.tts import _normalize_audio
    assert _normalize_audio(np.array([0, 255], dtype=np.uint8)).tolist() == pytest.approx([-1, 255 / 128 - 1])
    assert np.abs(_normalize_audio(np.array([0.5, 4.0]))).max() == pytest.approx(1)
    assert _normalize_audio(np.ones((10, 2), dtype=np.float32)).shape == (10,)  # stereo to mono
    with pytest.raises(TypeError):
        _normalize_audio(np.array(["a"]))


def test_qwen3_missing_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_tts", None)
    from sdialog.audio.tts import Qwen3TTS, Qwen3TTSVoiceClone
    with pytest.raises(ImportError, match="qwen-tts"):
        Qwen3TTS()
    with pytest.raises(ImportError, match="qwen-tts"):
        Qwen3TTSVoiceClone()


class FakeKPipeline:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.calls = []

    def __call__(self, text, voice=None, speed=1.0):
        self.calls.append((text, voice, speed))
        yield "graphemes", "phonemes", np.zeros(100, dtype=np.float32)


def test_kokoro_tts(monkeypatch):
    module = types.ModuleType("kokoro")
    module.KPipeline = FakeKPipeline
    monkeypatch.setitem(sys.modules, "kokoro", module)
    from sdialog.audio.tts import KokoroTTS
    tts = KokoroTTS(lang_code="e", speed=1.2)
    audio, sr = tts.generate("Hola — ¿qué tal?", speaker_voice="ef_dora")
    assert sr == 24000 and len(audio) == 100
    assert tts.pipeline.calls == [("Hola - ¿qué tal?", "ef_dora", 1.2)]
    with pytest.raises(ValueError):
        KokoroTTS(lang_code="xx")
    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(ImportError, match="kokoro"):
        KokoroTTS()


def test_huggingface_tts(monkeypatch):
    import transformers
    calls = {}

    def fake_pipeline(task, model=None, device=None, **kwargs):
        calls["init"] = (task, model, device)
        return lambda text, **kw: calls.setdefault("generate", (text, kw)) and {
            "audio": [np.zeros(50, dtype=np.float32)], "sampling_rate": 16000}

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    from sdialog.audio.tts import HuggingFaceTTS
    tts = HuggingFaceTTS("facebook/mms-tts-fra", device="cpu", text_normalizers=[LowercaseNormalizer()])
    audio, sr = tts.generate("Bonjour", speaker_voice=None, tts_pipeline_kwargs={"forward_params": {}})
    assert calls["init"] == ("text-to-speech", "facebook/mms-tts-fra", "cpu")
    assert calls["generate"] == ("bonjour", {"forward_params": {}})
    assert sr == 16000 and len(audio) == 50


def test_index_tts(monkeypatch):
    class FakeIndexTTS:
        def __init__(self, model_dir, cfg_path, device):
            self.args = (model_dir, cfg_path, device)

        def infer(self, voice, text, output_path=None):
            self.last = (voice, text)
            return 22050, np.zeros(30, dtype=np.float32)

    for name, attr in (("indextts.infer_v2", "IndexTTS2"), ("indextts.infer", "IndexTTS")):
        module = types.ModuleType(name)
        setattr(module, attr, FakeIndexTTS)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "indextts", types.ModuleType("indextts"))
    from sdialog.audio.tts import IndexTTS
    for version in ("2", "1"):
        tts = IndexTTS(model_dir="m", cfg_path="m/config.yaml", device="cpu", version=version)
        audio, sr = tts.generate("It’s fine", speaker_voice="ref.wav")
        assert sr == 22050 and tts.pipeline.last == ("ref.wav", "It's fine")
