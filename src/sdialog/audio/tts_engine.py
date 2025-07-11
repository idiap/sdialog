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
    """

    def __init__(self):
        """
        Initializes the Kokoro model.
        """
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code='a')

    def generate(self, text: str, voice: str) -> np.ndarray:
        """
        Generate audio from text using the Kokoro model.
        """

        generator = self.pipeline(text, voice=voice)

        gs, ps, audio = next(iter(generator))

        return audio


class ChatterboxTTS(BaseTTS):
    """
    Chatterbox is a TTS engine that uses the Chatterbox model.
    """

    def __init__(self):
        """
        Initializes the Chatterbox model.
        """
        from chatterbox.tts import ChatterboxTTS as ChatterboxTTSModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = ChatterboxTTSModel.from_pretrained(device=device)

    def generate(self, text: str, voice: str) -> np.ndarray:
        """
        Generate audio from text using the Chatterbox model.
        """
        
        wav = self.pipeline.generate(text, audio_prompt_path=voice)

        return wav.cpu().numpy().squeeze()


class XttsTTS(BaseTTS):
    """
    XTTS is a TTS engine that uses the XTTSv2 model from Coqui-TTS.
    """

    def __init__(self):
        """
        Initializes the XTTS model.
        """
        from TTS.api import TTS as XTTSModel

        self.pipeline = XTTSModel(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True,
            gpu=torch.cuda.is_available(),
        )

    def generate(self, text: str, voice: str) -> np.ndarray:
        """
        Generate audio from text using the XTTSv2 model.
        """

        audio = self.pipeline.tts(text=text, speaker_wav=voice, language="en")

        return np.array(audio)
