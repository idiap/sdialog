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
