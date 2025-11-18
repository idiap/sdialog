# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import torch
import numpy as np

from ..base import BaseTTS


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
