"""
This module provides a comprehensive text-to-speech (TTS) engine framework for the sdialog library.

The module includes a base abstract class for TTS engines and concrete implementations
for various TTS models, enabling flexible audio generation from text with support
for multiple languages and voice characteristics.

Key Components:

  - BaseTTS: Abstract base class defining the TTS interface

Example:

    .. code-block:: python

        from sdialog.audio.tts import KokoroTTS, IndexTTS, HuggingFaceTTS

        # Initialize Kokoro TTS for American English
        tts = KokoroTTS(lang_code="a")
        audio, sample_rate = tts.generate("Hello world", voice="am_echo")

        # Initialize IndexTTS for bilingual support
        tts = IndexTTS(model_dir="model", cfg_path="model/config.yaml")
        audio, sample_rate = tts.generate("你好世界", voice="chinese_voice")

        # Initialize HuggingFaceTTS for facebook/mms-tts-eng model
        tts = HuggingFaceTTS(model_id="facebook/mms-tts-eng")
        audio, sample_rate = tts.generate("[clears throat] This is a test ...")
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import numpy as np
from abc import abstractmethod, ABC


class BaseTTS(ABC):
    """
    Abstract base class for text-to-speech (TTS) engines.

    This class defines the interface that all TTS engine implementations must follow.
    It provides a common structure for initializing TTS pipelines and generating
    audio from text input with specified voice characteristics.

    Subclasses must implement the generate() method to provide the actual
    TTS functionality. The pipeline attribute should be initialized in the
    subclass constructor with the appropriate TTS model or pipeline.

    Key Features:

      - Abstract interface for TTS engine implementations
      - Common initialization pattern for TTS pipelines
      - Standardized audio generation interface
      - Support for voice-specific audio generation

    :ivar pipeline: The TTS pipeline or model instance (initialized by subclasses).
    :vartype pipeline: Any
    """

    def __init__(self):
        """
        Initializes the base TTS engine.

        Subclasses should call this method and then initialize their specific
        TTS pipeline in the pipeline attribute.
        """
        self.pipeline = None

    @abstractmethod
    def generate(self, text: str, speaker_voice: str, tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
        """
        Generates audio from text using the specified voice.

        This abstract method must be implemented by all TTS engine subclasses.
        It should convert the input text to audio using the specified voice
        and return both the audio data and sampling rate.

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: The voice identifier to use for speech generation.
        :type speaker_voice: str
        :param tts_pipeline_kwargs: Additional keyword arguments to be passed to the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as a numpy array and the sampling rate.
        :rtype: tuple[np.ndarray, int]
        :raises NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the generate method")
