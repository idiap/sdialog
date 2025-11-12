"""
This module provides a comprehensive text-to-speech (TTS) engine framework for the sdialog library.

The module includes a base abstract class for TTS engines and concrete implementations
for various TTS models, enabling flexible audio generation from text with support
for multiple languages and voice characteristics.

Key Components:

  - BaseTTS: Abstract base class defining the TTS interface
  - KokoroTTS: Implementation using the Kokoro TTS pipeline
  - IndexTTS: Implementation using the IndexTTS model
  - HuggingFaceTTS: Generic implementation for models from the Hugging Face Hub

Supported TTS Engines:

  - Kokoro: Multi-language TTS with support for 9 languages including English,
    Spanish, French, Hindi, Italian, Japanese, Portuguese, and Mandarin Chinese
  - IndexTTS: Bilingual TTS supporting Chinese and English with automatic
    language detection

Example:

    .. code-block:: python

        from sdialog.audio import KokoroTTS, IndexTTS

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

import torch
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


class KokoroTTS(BaseTTS):
    """
    Kokoro TTS engine implementation using the Kokoro pipeline.

    Kokoro is a high-quality multi-language TTS engine that supports 9 different
    languages with various voice options. It provides natural-sounding speech
    synthesis with good prosody and pronunciation.

    Supported Languages:
        - American English (a)
        - British English (b)
        - Spanish (e)
        - French (f)
        - Hindi (h)
        - Italian (i)
        - Japanese (j)
        - Brazilian Portuguese (p)
        - Mandarin Chinese (z)

    Installation Requirements:
        For Mandarin Chinese and Japanese support, install additional packages:
        - pip install misaki[zh]  # For Mandarin Chinese
        - pip install misaki[ja]  # For Japanese

    References:
        - Kokoro GitHub: https://github.com/hexgrad/kokoro
        - Supported voices: https://github.com/nazdridoy/kokoro-tts?tab=readme-ov-file#supported-voices

    :ivar available_languages: List of supported language codes.
    :vartype available_languages: List[str]
    :ivar lang_code: The language code for this TTS instance.
    :vartype lang_code: str
    :ivar pipeline: The Kokoro KPipeline instance.
    :vartype pipeline: KPipeline
    """

    def __init__(
            self,
            lang_code: str = "a",
            speed: float = 1.0):
        """
        Initializes the Kokoro TTS engine with the specified language.

        This constructor sets up the Kokoro TTS pipeline for the specified language.
        It validates the language code and initializes the underlying KPipeline
        for audio generation.

        :param lang_code: Language code for TTS generation (default: "a" for American English).
        :type lang_code: str
        :param speed: Speech speed multiplier (default: 1.0 for normal speed).
        :type speed: float
        :raises ValueError: If the provided language code is not supported.
        :raises ImportError: If the kokoro package is not installed.
        """

        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "The 'kokoro' library is required to use KokoroTTS. "
                "Please install following the instructions here: https://github.com/hexgrad/kokoro"
            )

        self.available_languages = ["a", "b", "e", "f", "h", "i", "j", "p", "z"]

        if lang_code not in self.available_languages:
            raise ValueError(
                f"Invalid language code: {lang_code}. "
                f"Supported languages: {self.available_languages}"
            )

        self.lang_code = lang_code
        self.speed = speed

        # Initialize the Kokoro pipeline
        self.pipeline = KPipeline(lang_code=self.lang_code)

    def generate(self, text: str, speaker_voice: str, tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
        """
        Generates audio from text using the Kokoro TTS engine.

        This method converts the input text to speech using the specified voice
        and speed parameters. The Kokoro pipeline generates high-quality audio
        with natural prosody and pronunciation.

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: The voice identifier to use for speech generation.
                     Must be compatible with the selected language.
        :type speaker_voice: str
        :param speed: Speech speed multiplier (default: 1.0 for normal speed).
        :type speed: float
        :param tts_pipeline_kwargs: Additional keyword arguments to be passed to the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as a numpy array and the sampling rate (24000 Hz).
        :rtype: tuple[np.ndarray, int]
        :raises ValueError: If the voice is not compatible with the selected language.
        :raises RuntimeError: If audio generation fails.
        """

        # Generate audio using the Kokoro pipeline
        generator = self.pipeline(text, voice=speaker_voice, speed=self.speed)

        # Extract audio data from the generator
        gs, ps, audio = next(iter(generator))

        # Return audio data with Kokoro's standard sampling rate
        return (audio, 24000)


class IndexTTS(BaseTTS):
    """
    IndexTTS engine implementation using the IndexTTS model.

    IndexTTS is a bilingual text-to-speech engine that supports both Chinese
    and English languages with automatic language detection. It provides
    high-quality speech synthesis with natural prosody and pronunciation
    for both languages.

    Key Features:
        - Bilingual support (Chinese and English)
        - Automatic language detection from text input
        - High-quality speech synthesis
        - GPU acceleration support
        - Flexible model configuration

    References:
        - IndexTTS GitHub: https://github.com/index-tts/index-tts

    :ivar pipeline: The IndexTTS model instance.
    :vartype pipeline: IndexTTS
    """

    def __init__(
            self,
            model_dir="model",
            cfg_path="model/config.yaml",
            device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the IndexTTS engine with the specified model configuration.

        This constructor sets up the IndexTTS model for bilingual speech synthesis.
        It loads the model from the specified directory and configuration file,
        and configures the device for inference (GPU or CPU).

        :param model_dir: Directory path containing the IndexTTS model files (default: "model").
        :type model_dir: str
        :param cfg_path: Path to the model configuration file (default: "model/config.yaml").
        :type cfg_path: str
        :param device: Device for model inference - "cuda" for GPU or "cpu" for CPU
                      (default: automatically detects CUDA availability).
        :type device: str
        :raises ImportError: If the indextts package is not installed.
        :raises FileNotFoundError: If the model directory or config file is not found.
        :raises RuntimeError: If model initialization fails.
        :raises ImportError: If the indextts package is not installed.
        """

        try:
            from indextts.infer import IndexTTS
        except ImportError:
            raise ImportError(
                "The 'indextts' library is required to use IndexTTS. "
                "Please install following the instructions here: https://github.com/index-tts/index-tts"
            )

        # Initialize the IndexTTS model
        self.pipeline = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, device=device)

    def generate(self, text: str, speaker_voice: str, tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
        """
        Generates audio from text using the IndexTTS engine.

        This method converts the input text to speech using the specified voice.
        The IndexTTS engine automatically detects the language of the input text
        and generates appropriate speech synthesis.

        :param text: The text to be converted to speech (Chinese or English).
        :type text: str
        :param speaker_voice: The voice identifier to use for speech generation.
        :type speaker_voice: str
        :param tts_pipeline_kwargs: Additional keyword arguments to be passed to the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as a numpy array and the sampling rate.
        :rtype: tuple[np.ndarray, int]
        :raises ValueError: If the voice is not compatible with the detected language.
        :raises RuntimeError: If audio generation fails.
        """

        # Generate audio using the IndexTTS model
        sampling_rate, wav_data = self.pipeline.infer(speaker_voice, text, output_path=None)

        return (wav_data, sampling_rate)


class HuggingFaceTTS(BaseTTS):
    """
    Hugging Face TTS engine implementation using the transformers pipeline.

    This class provides a generic interface for various text-to-speech models
    available on the Hugging Face Hub that are supported by the `text-to-speech`
    pipeline.

    Key Features:
        - Support for any `text-to-speech` compatible model from Hugging Face.
        - GPU acceleration support.
        - Flexible voice/speaker selection through a keyword argument.

    :ivar pipeline: The Hugging Face pipeline instance.
    :vartype pipeline: transformers.Pipeline
    """

    def __init__(
            self,
            model_id: str = "facebook/mms-tts-eng",
            device: str = None,
            **kwargs):
        """
        Initializes the Hugging Face TTS engine.

        :param model_id: The model identifier from the Hugging Face Hub.
        :type model_id: str
        :param device: Device for model inference ("cuda" or "cpu"). If None,
                       it will auto-detect CUDA availability.
        :type device: str
        :raises ImportError: If the `transformers` package is not installed.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "The 'transformers' library is required to use HuggingFaceTTS. "
                "Please install it with 'pip install transformers'."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = pipeline("text-to-speech", model=model_id, device=device, **kwargs)

    def generate(self, text: str, speaker_voice: str, tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
        """
        Generates audio from text using the Hugging Face TTS pipeline.

        This method passes any additional keyword arguments directly to the
        pipeline, allowing for model-specific parameters like speaker embeddings.

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: The voice identifier to use for speech generation.
        :type speaker_voice: str
        :param tts_pipeline_kwargs: Additional keyword arguments to be passed to the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as a numpy array and the sampling rate.
        :rtype: tuple[np.ndarray, int]
        """
        output = self.pipeline(text, **tts_pipeline_kwargs)

        return (output["audio"][0], output["sampling_rate"])
