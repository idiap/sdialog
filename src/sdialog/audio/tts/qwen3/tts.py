# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import torch
import numpy as np

from qwen_tts import Qwen3TTSModel
from ..base import BaseTTS, BaseVoiceCloneTTS


class Qwen3TTS(BaseTTS):
    def __init__(
            self,
            model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map: str = None,
            dtype: torch.dtype = torch.bfloat16,
            **model_kwargs):
        """
        Initializes the Qwen3-TTS engine.

        :param model: The model identifier from the Hugging Face Hub.
        :type model: str
        """
        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = Qwen3TTSModel.from_pretrained(
            model,
            device_map=device_map,
            dtype=dtype,
            **model_kwargs
            # attn_implementation="flash_attention_2",
        )

    def generate(self, text: str, speaker_voice: str = None, tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
        if "language" not in tts_pipeline_kwargs:
            tts_pipeline_kwargs["language"] = "English"
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            speaker=speaker_voice,
            # instruct="Very happy", # Omit if not needed.
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
            **model_kwargs):
        """
        Initializes the Qwen3-TTS engine.

        :param model: The model identifier from the Hugging Face Hub.
        :type model: str
        """
        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = Qwen3TTSModel.from_pretrained(
            model,
            device_map=device_map,
            dtype=dtype,
            **model_kwargs
            # attn_implementation="flash_attention_2",
        )

    def generate(self,
                 text: str,
                 speaker_voice: str | object = None,
                 tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
        """
        Generates audio from text using the Hugging Face TTS pipeline.

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
        if "language" not in tts_pipeline_kwargs:
            tts_pipeline_kwargs["language"] = "English"

        if type(speaker_voice) is str:
            tts_pipeline_kwargs["ref_audio"] = speaker_voice  # Path to reference audio
            tts_pipeline_kwargs["ref_text"] = text  # TODO: should be the transcription of ref_audio
        elif speaker_voice is not None:
            tts_pipeline_kwargs["voice_clone_prompt"] = speaker_voice

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            **tts_pipeline_kwargs
        )
        audio = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else np.asarray(wavs[0])

        return (audio, sr)
