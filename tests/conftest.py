import sys
import types
import importlib.machinery
from types import SimpleNamespace

import numpy as np


def _install_qwen_tts_stub() -> None:
    try:
        __import__("qwen_tts")
        return
    except ImportError:
        pass

    class _FakeQwen3TTSModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def generate_custom_voice(self, text, speaker=None, **kwargs):
            return [np.zeros(24_000, dtype=np.float32)], 24_000

        def generate_voice_clone(self, text, **kwargs):
            return [np.zeros(24_000, dtype=np.float32)], 24_000

        def generate_voice_design(self, text, language=None, instruct=None, **kwargs):
            return [np.zeros(24_000, dtype=np.float32)], 24_000

        def create_voice_clone_prompt(self, ref_audio=None, ref_text=None, **kwargs):
            return {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
            }

    qwen_tts_module = types.ModuleType("qwen_tts")
    qwen_tts_module.Qwen3TTSModel = _FakeQwen3TTSModel
    qwen_tts_module.__spec__ = importlib.machinery.ModuleSpec("qwen_tts", loader=None)
    sys.modules["qwen_tts"] = qwen_tts_module


_install_qwen_tts_stub()


def _install_torchcodec_stub() -> None:
    try:
        __import__("torchcodec")
        return
    except ImportError:
        pass

    class _FakeTensor:
        def __init__(self, array):
            self._array = array

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class _FakeAudioSamples:
        def __init__(self, data=None, sample_rate: int = 16_000):
            _arr = np.zeros((1, sample_rate), dtype=np.float32) if data is None else data
            self.data = _FakeTensor(_arr)
            self.sample_rate = sample_rate

    class _FakeAudioDecoder:
        def __init__(self, source=None, *args, **kwargs):
            self.source = source
            self.args = args
            self.kwargs = kwargs
            _path = None
            if isinstance(source, dict):
                _path = source.get("path")
            else:
                _path = getattr(source, "path", None)

            self.metadata = SimpleNamespace(
                sample_rate=16_000,
                path=_path,
            )

        def __getitem__(self, key: str):
            if key == "path":
                return self.metadata.path
            if key == "sampling_rate":
                return self.metadata.sample_rate
            if key == "array":
                y = self.get_all_samples().data.cpu().numpy()
                return np.mean(y, axis=tuple(range(y.ndim - 1))) if y.ndim > 1 else y
            raise KeyError(key)

        def get_all_samples(self):
            return _FakeAudioSamples()

        def get_samples_played_in_range(self, *_args, **_kwargs):
            return SimpleNamespace(sample_rate=self.metadata.sample_rate)

    torchcodec_module = types.ModuleType("torchcodec")
    decoders_module = types.ModuleType("torchcodec.decoders")
    decoders_module.AudioDecoder = _FakeAudioDecoder
    torchcodec_module.decoders = decoders_module
    torchcodec_module.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
    decoders_module.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)

    sys.modules["torchcodec"] = torchcodec_module
    sys.modules["torchcodec.decoders"] = decoders_module


_install_torchcodec_stub()
