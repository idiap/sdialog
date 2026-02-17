# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import torch
import numpy as np

from ..base import BaseTTS, BaseVoiceCloneTTS
from sdialog.audio.normalizers import TextNormalizer, normalize_text


def _seed_torch(seed: int) -> None:
    """Seed all torch RNGs for reproducible generation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


class Qwen3TTS(BaseTTS):
    def __init__(
            self,
            model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map: str = None,
            dtype: torch.dtype = torch.bfloat16,
            text_normalizers: list[TextNormalizer] = None,
            deterministic: bool = False,
            seed: int = 42,
            **model_kwargs):
        """
        Initializes the Qwen3-TTS engine.

        :param model: The model identifier from the Hugging Face Hub.
        :type model: str
        :param deterministic: If True, disables sampling (greedy decoding) and seeds
                              the torch RNG before every generation call, removing all randomness.
        :type deterministic: bool
        :param seed: Random seed used when ``deterministic=True``. Ignored otherwise.
        :type seed: int
        """

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError("qwen_tts is not installed. Please install it with `pip install qwen-tts`.")

        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.dtype = dtype
        self.device_map = device_map
        self.deterministic = deterministic
        self.seed = seed
        self.model = Qwen3TTSModel.from_pretrained(
            model,
            device_map=device_map,
            dtype=dtype,
            **model_kwargs
            # attn_implementation="flash_attention_2",
        )
        self.text_normalizers = text_normalizers

    def generate(self, text: str, speaker_voice: str = None, tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:

        if self.text_normalizers is not None and len(self.text_normalizers) > 0:
            text = normalize_text(text, self.text_normalizers)

        if "language" not in tts_pipeline_kwargs:
            tts_pipeline_kwargs["language"] = "English"

        if self.deterministic:
            if "do_sample" not in tts_pipeline_kwargs:
                tts_pipeline_kwargs["do_sample"] = False
            _seed_torch(self.seed)

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            speaker=speaker_voice,
            **tts_pipeline_kwargs
        )

        audio = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else np.asarray(wavs[0])
        return (audio, sr)


class Qwen3TTSVoiceClone(BaseVoiceCloneTTS):
    def __init__(
            self,
            model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map: str = None,
            dtype: torch.dtype = torch.bfloat16,
            text_normalizers: list[TextNormalizer] = None,
            deterministic: bool = False,
            seed: int = 42,
            **model_kwargs):
        """
        Initializes the Qwen3-TTS voice clone engine.

        :param model: The model identifier from the Hugging Face Hub.
        :type model: str
        :param deterministic: If True, disables sampling (greedy decoding) and seeds
                              the torch RNG before every generation call, removing all randomness.
        :type deterministic: bool
        :param seed: Random seed used when ``deterministic=True``. Ignored otherwise.
        :type seed: int
        """

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError("qwen_tts is not installed. Please install it with `pip install qwen-tts`.")

        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.dtype = dtype
        self.device_map = device_map
        self.deterministic = deterministic
        self.seed = seed
        self.model = Qwen3TTSModel.from_pretrained(
            model,
            device_map=device_map,
            dtype=dtype,
            **model_kwargs
            # attn_implementation="flash_attention_2",
        )
        self.text_normalizers = text_normalizers

    def generate(self,
                 text: str,
                 speaker_voice: str | object = None,
                 tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
        """
        Generates audio from text using voice cloning.

        This method passes any additional keyword arguments directly to the
        pipeline, allowing for model-specific parameters like speaker embeddings.

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: Either a string path to a reference audio file for voice cloning,
                              or a voice clone prompt object created by create_voice_clone_prompt(),
                              or None to use default voice.
        :type speaker_voice: str | object | None
        :param tts_pipeline_kwargs: Additional keyword arguments to be passed to the TTS pipeline.
                                    Should contain 'voice_clone_prompt' key if voice cloning is desired.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as a numpy array and the sampling rate.
        :rtype: tuple[np.ndarray, int]
        """

        # Normalize the text if text normalizers are provided.
        if self.text_normalizers is not None and len(self.text_normalizers) > 0:
            text = normalize_text(text, self.text_normalizers)

        if "language" not in tts_pipeline_kwargs:
            tts_pipeline_kwargs["language"] = "English"

        if self.deterministic:
            if "do_sample" not in tts_pipeline_kwargs:
                tts_pipeline_kwargs["do_sample"] = False
            _seed_torch(self.seed)

        if speaker_voice is not None:
            # Normalize the reference audio input if it's a (wav, sr) tuple
            if isinstance(speaker_voice, tuple) and len(speaker_voice) == 2:
                wav, ref_sr = speaker_voice
                wav = _normalize_audio(wav)
                speaker_voice = (wav, ref_sr)
            tts_pipeline_kwargs["ref_audio"] = speaker_voice
            # tts_pipeline_kwargs["ref_text"] = ref_text  # TODO: should be the transcription of ref_audio
            tts_pipeline_kwargs["x_vector_only_mode"] = True
        else:
            raise ValueError("speaker_voice must be provided for voice cloning in Qwen3TTSVoiceClone")

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            **tts_pipeline_kwargs
        )
        audio = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else np.asarray(wavs[0])

        return (audio, sr)
