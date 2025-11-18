# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import numpy as np

from ..base import BaseTTS


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
