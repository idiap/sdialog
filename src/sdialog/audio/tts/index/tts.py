# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import torch
import numpy as np

from ..base import BaseTTS


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
