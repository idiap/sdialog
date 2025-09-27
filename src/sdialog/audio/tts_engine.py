"""
This module provides a base class for TTS engines and all the derivated models supported by the sdialog library.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from abc import abstractmethod


class BaseTTS:
    """
    Base class for TTS engines.
    """

    def __init__(self):
        self.pipeline = None

    @abstractmethod
    def generate(self, text: str, voice: str) -> np.ndarray:
        return None


class KokoroTTS(BaseTTS):
    """
    Kokoro is a TTS engine that uses the Kokoro pipeline.
    Available languages :
        American English (a), British English (b), Spanish (e), French (f), Hindi (h),
        Italian (i), Japanese (j), Brazilian Portuguese (p), Mandarin Chinese (z)
    In order to use mandarin chinese and japanese, you need to install the corresponding package:
        pip install misaki[zh] or pip install misaki[ja]
    More details on: https://github.com/hexgrad/kokoro
    Supported voices: https://github.com/nazdridoy/kokoro-tts?tab=readme-ov-file#supported-voices
    """

    def __init__(
            self,
            lang_code="a"):
        """
        Initializes the Kokoro model.
        """
        from kokoro import KPipeline

        self.available_languages = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]

        if lang_code not in self.available_languages:
            raise ValueError(f"Invalid language code: {lang_code}")

        self.lang_code = lang_code

        self.pipeline = KPipeline(lang_code=self.lang_code)

    def generate(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """
        Generate audio from text using the Kokoro model.
        """

        generator = self.pipeline(text, voice=voice, speed=speed)

        gs, ps, audio = next(iter(generator))

        return (audio, 24000)


class IndexTTS(BaseTTS):
    """
    IndexTTS is a TTS engine that uses the IndexTTS model.
    Available languages: Chinese and English (the language is detected automatically by the text instruction)
    More details on: https://github.com/index-tts/index-tts
    """

    def __init__(
            self,
            model_dir="model",
            cfg_path="model/config.yaml",
            device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the IndexTTS model.
        """
        from indextts.infer import IndexTTS

        self.pipeline = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, device=device)

    def generate(self, text: str, voice: str) -> np.ndarray:
        """
        Generate audio from text using the IndexTTS model.
        """

        sampling_rate, wav_data = self.pipeline.infer(voice, text, output_path=None)

        return (wav_data, sampling_rate)


# TODO: Test this model
class ChatterboxTTS(BaseTTS):
    """
    Chatterbox is a TTS engine that uses the Chatterbox model.
    Available languages:
        Arabic (ar), Danish (da), German (de), Greek (el), English (en), Spanish (es),
        Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Italian (it), Japanese (ja),
        Korean (ko), Malay (ms), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt),
        Russian (ru), Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)
    More details on: https://github.com/resemble-ai/chatterbox
    """

    def __init__(
            self,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lang_code="en"):
        """
        Initializes the Chatterbox model.
        """

        self.available_languages = [
            "ar", "da", "de", "el", "en", "es", "fi", "fr",
            "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
            "pl", "pt", "ru", "sv", "sw", "tr", "zh"
        ]

        if lang_code not in self.available_languages:
            raise ValueError(f"Invalid language code: {lang_code}")

        self.lang_code = lang_code

        if self.lang_code == "en":
            from chatterbox.tts import ChatterboxTTS as ChatterboxTTSModel
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxTTSModel

        self.pipeline = ChatterboxTTSModel.from_pretrained(device=device)

    def generate(self, text: str, voice: str) -> np.ndarray:
        """
        Generate audio from text using the Chatterbox model.
        """

        if self.lang_code == "en":
            wav = self.pipeline.generate(text, audio_prompt_path=voice)
        else:
            wav = self.pipeline.generate(text, audio_prompt_path=voice, language_id=self.lang_code)

        return (wav.cpu().numpy().squeeze(), -1)


class XttsTTS(BaseTTS):
    """
    XTTS is a TTS engine that uses the XTTSv2 model from Coqui-TTS.
    Available languages of the XTTSv2 model (https://huggingface.co/coqui/XTTS-v2):
        English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt),
        Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn),
        Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi)
    More details on: https://github.com/coqui-ai/TTS
    """

    DEFAULT_MODEL_NAME_XTTSV2 = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
            self,
            model_name=DEFAULT_MODEL_NAME_XTTSV2,
            device=torch.cuda.is_available(),
            progress_bar=True,
            lang_code="en"):
        """
        Initializes the XTTS model.
        """
        from TTS.api import TTS as XTTSModel

        self.available_languages = [
            "en", "es", "fr", "de", "it", "pt",
            "pl", "tr", "ru", "nl", "cs", "ar",
            "zh-cn", "ja", "hu", "ko", "hi"
        ]

        self.lang_code = lang_code

        if model_name == XttsTTS.DEFAULT_MODEL_NAME_XTTSV2 and self.lang_code not in self.available_languages:
            raise ValueError(f"Invalid language code: {self.lang_code}")

        self.pipeline = XTTSModel(
            model_name=model_name,
            progress_bar=progress_bar,
            gpu=device,
        )

    def generate(self, text: str, voice: str) -> np.ndarray:
        """
        Generate audio from text using the XTTSv2 model.
        """

        audio = self.pipeline.tts(text=text, speaker_wav=voice, language=self.lang_code)

        return (np.array(audio), -1)
