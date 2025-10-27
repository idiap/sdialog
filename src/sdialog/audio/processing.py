"""
This module provides a class for processing audio signals.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, David Grünert <david.gruenert@zhaw.ch>
# SPDX-License-Identifier: MIT

import os
import logging
from typing import Union, Optional

import librosa
import soundfile as sf
from scipy.signal import fftconvolve

from sdialog.audio.impulse_response_database import ImpulseResponseDatabase, RecordingDevice


class AudioProcessor:
    """
    A class for processing audio signals. It provides functionalities to
    apply various audio effects, such as microphone simulation through
    convolution with impulse responses.

    This class is designed to work with audio files and relies on an
    impulse response database to apply realistic microphone and environmental
    effects.

    Example:
        .. code-block:: python

            from sdialog.audio.processing import AudioProcessor
            from sdialog.audio.impulse_response_database import LocalImpulseResponseDatabase, RecordingDevice
            impulse_response_database = LocalImpulseResponseDatabase(
                "path/to/your/impulse_response_database",
                metadata_file="path/to/your/metadata.csv"
            )
            AudioProcessor.apply_microphone_effect(
                input_audio_path="path/to/your/input.wav",
                output_audio_path="path/to/your/output.wav",
                device=RecordingDevice.SHURE_SM57,
                impulse_response_database=impulse_response_database
            )

    Note:
        This class uses static methods, so you don't need to instantiate it.
    """

    DEFAULT_GAIN_REDUCTION_DB = -10.0

    @staticmethod
    def apply_microphone_effect(
        input_audio_path: str,
        output_audio_path: str,
        device: Union[RecordingDevice, str],
        impulse_response_database: ImpulseResponseDatabase,
        gain_db: Optional[float] = None
    ):
        """
        Applies a microphone effect to an audio signal by convolving it with an
        impulse response from the database.

        The function loads an audio file, retrieves a specified impulse
        response, and applies it to the audio. The sample rates of the audio
        and impulse response are matched by resampling the impulse response if
        necessary. The resulting audio is then saved to a specified output
        path. A gain reduction can be applied to prevent clipping.

        :param input_audio_path: Path to the input audio file.
        :type input_audio_path: str
        :param output_audio_path: Path to save the processed audio file.
        :type output_audio_path: str
        :param device: The recording device or its identifier to select the
                       impulse response.
        :type device: Union[RecordingDevice, str]
        :param impulse_response_database: The database containing impulse
                                          responses.
        :type impulse_response_database: ImpulseResponseDatabase
        :param gain_db: The gain in decibels to apply to the processed audio.
                        If None, a default gain reduction is used.
        :type gain_db: Optional[float]
        """

        # Load the input audio of step 3
        audio, sample_rate = sf.read(input_audio_path)

        # Get the impulse response from the database
        impulse_response_path = impulse_response_database.get_ir(device)

        if not os.path.exists(impulse_response_path):
            raise ValueError(f"Impulse response path not found: {impulse_response_path}")

        impulse_response, ir_sr = sf.read(impulse_response_path)

        # Ensure the impulse response is mono
        if impulse_response.ndim > 1:
            impulse_response = impulse_response.mean(axis=1)

        # Resample impulse response if sample rates don't match
        if ir_sr != sample_rate:
            logging.info(
                f"[Post-Processing] Impulse response sample rate ({ir_sr}Hz) does not match "
                f"audio sample rate ({sample_rate}Hz). Resampling impulse response..."
            )
            impulse_response = librosa.resample(
                y=impulse_response,
                orig_sr=ir_sr,
                target_sr=sample_rate
            )

        # check if the audio is mono otherwise convert it to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Apply convolution to the audio of step 3
        processed_audio = fftconvolve(audio, impulse_response, mode="full")

        # # Normalize the output audio to prevent clipping
        # max_val = np.max(np.abs(processed_audio))
        # if max_val > 0:
        #     processed_audio /= max_val

        # Reduce gain by the user defined or default gain reduction in decibels
        if gain_db is not None:
            processed_audio *= (10 ** (gain_db / 20))
        else:
            processed_audio *= (10 ** (AudioProcessor.DEFAULT_GAIN_REDUCTION_DB / 20))

        # Save the processed audio
        sf.write(output_audio_path, processed_audio, sample_rate)
