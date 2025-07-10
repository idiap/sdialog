"""
This module provides a base class for TTS engines and all the derivated models supported by the sdialog library.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import numpy as np
from kokoro import KPipeline
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
        self.pipeline = KPipeline(lang_code='a')

    def generate(self, text: str, voice: str) -> np.ndarray:
        """
        Generate audio from text using the Kokoro model.
        """

        generator = self.pipeline(text, voice=voice)

        gs, ps, audio = next(iter(generator))

        return audio
