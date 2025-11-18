"""
This module provides comprehensive functionality for generating audio from text utterances in dialogues.

The audio module extends the core sdialog functionality by adding:

  - Audio generation from text using various TTS engines (Kokoro, IndexTTS)
  - Voice databases with speaker characteristics (gender, age, language)
  - Room acoustics simulation for realistic audio environments
  - Audio dialogue processing with turn-based audio generation

Key Components:

  - AudioDialog: Extended dialogue class with audio turn support
  - AudioTurn: Individual dialogue turns with associated audio data
  - BaseTTS: Abstract base class for text-to-speech engines
  - BaseVoiceDatabase: Voice database management with speaker characteristics
  - Room: 3D room specification for acoustics simulation
  - AcousticsSimulator: Acoustics simulation engine

Example:

    .. code-block:: python

        from sdialog.audio import AudioDialog, KokoroTTS, HuggingfaceVoiceDatabase
        from sdialog.audio.room import Room

        # Create TTS engine and voice database
        tts = KokoroTTS(lang_code="a")  # American English
        voice_db = HuggingfaceVoiceDatabase("sdialog/voices-libritts")

        # Convert regular dialog to audio dialog
        audio_dialog = AudioDialog.from_dialog(dialog)

        # Generate audio for all utterances
        audio_dialog = generate_utterances_audios(
            dialog=audio_dialog,
            voice_database=voice_db,
            tts_pipeline=tts,
            seed=42
        )

        # Simulate room acoustics
        room = Room(dimensions=(5.0, 4.0, 3.0))
        audio_dialog = generate_audio_room_accoustic(
            dialog=audio_dialog,
            room=room,
            dialog_directory="output",
            room_name="living_room"
        )
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from typing import Union

from sdialog.audio.tts import BaseTTS
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.room import Room, RoomPosition
from sdialog.audio.utils import AudioUtils, SourceVolume, Role, logger
from sdialog.audio.acoustics_simulator import AcousticsSimulator
from sdialog.audio.voice_database import BaseVoiceDatabase, Voice

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_utterances_audios(
    dialog: AudioDialog,
    voice_database: BaseVoiceDatabase,
    tts_pipeline: BaseTTS,
    voices: dict[Role, Union[Voice, tuple[str, str]]] = None,
    keep_duplicate: bool = True,
    seed: int = None,
    sampling_rate: int = 24_000,
    tts_pipeline_kwargs: dict = {}
) -> AudioDialog:
    """
    Generates audio for each utterance in an AudioDialog object using the specified TTS engine.

    This function processes each turn in the dialogue, assigns appropriate voices to speakers
    based on their persona characteristics (gender, age, language), and generates audio
    using the provided TTS pipeline. The generated audio is stored in each AudioTurn object.

    The voice assignment process:
    1. Extracts speaker persona information (gender, age, language)
    2. Assigns voices from the voice database based on persona characteristics
    3. Generates audio for each utterance using the TTS engine
    4. Stores the generated audio in the corresponding AudioTurn

    :param dialog: The AudioDialog object containing the conversation turns.
    :type dialog: AudioDialog
    :param voice_database: Database containing available voices with speaker characteristics.
    :type voice_database: BaseVoiceDatabase
    :param tts_pipeline: Text-to-speech engine for audio generation.
    :type tts_pipeline: BaseTTS
    :param voices: Optional dictionary mapping speaker roles to specific voices.
                  If None, voices are automatically selected based on persona characteristics.
    :type voices: Optional[dict[Role, Union[Voice, tuple[str, str]]]]
    :param keep_duplicate: If True, allows the same voice to be used multiple times.
                          If False, ensures each voice is used only once.
    :type keep_duplicate: bool
    :param seed: Seed for random number generator.
    :type seed: int
    :param sampling_rate: Sampling rate for the audio generation.
    :type sampling_rate: int
    :return: The AudioDialog object with generated audio for each turn.
    :rtype: AudioDialog
    """

    # Attribute the voice to the persona of the dialog
    dialog.persona_to_voice(
        voice_database=voice_database,
        voices=voices,
        keep_duplicate=keep_duplicate,
        seed=seed
    )

    for turn in tqdm(dialog.turns, desc="Generating utterances audios"):

        # Get the voice of the turn
        turn.voice = dialog.personas[turn.speaker]["voice"].voice

        # Generate the utterance audio
        utterance_audio, utterance_sampling_rate = generate_utterance(
            text=AudioUtils.remove_audio_tags(turn.text),
            voice=turn.voice,
            tts_pipeline=tts_pipeline,
            tts_pipeline_kwargs=tts_pipeline_kwargs
        )

        # If the sampling rate of the audio is not the same as the sampling rate of the project, resample the audio
        if utterance_sampling_rate != sampling_rate:

            logger.info(
                f"[Step 1] Resampling the audio ({utterance_sampling_rate} Hz) to the sampling "
                f"rate of the project ({sampling_rate} Hz)..."
            )

            utterance_audio = librosa.resample(
                y=utterance_audio.astype(np.float32),
                orig_sr=utterance_sampling_rate,
                target_sr=sampling_rate,
            )

        # Set the utterance audio to the turn
        turn.set_audio(utterance_audio, sampling_rate)

    return dialog


def generate_utterance(
        text: str,
        voice: str,
        tts_pipeline: BaseTTS,
        tts_pipeline_kwargs: dict = {}) -> tuple[np.ndarray, int]:
    """
    Generates an audio recording of a text utterance using the specified TTS engine.

    This function takes a text string and converts it to audio using the provided
    TTS pipeline and voice identifier. The audio is returned as a numpy array
    along with the sampling rate.

    The function handles text preprocessing by removing audio-specific tags and
    formatting that might interfere with TTS generation.

    :param text: The text to be converted to audio. Audio tags are automatically removed.
    :type text: str
    :param voice: The voice identifier to use for the audio generation.
    :type voice: str
    :param tts_pipeline: The TTS engine to use for audio generation.
    :type tts_pipeline: BaseTTS
    :param tts_pipeline_kwargs: Additional keyword arguments to be passed to the TTS pipeline.
    :type tts_pipeline_kwargs: dict
    :return: A tuple containing the audio data as a numpy array and the sampling rate.
    :rtype: tuple[np.ndarray, int]
    """
    return tts_pipeline.generate(text, speaker_voice=voice, tts_pipeline_kwargs=tts_pipeline_kwargs)


def generate_audio_room_accoustic(
    dialog: AudioDialog,
    room: Room,
    dialog_directory: str,
    room_name: str,
    kwargs_pyroom: dict = {},
    source_volumes: dict[str, SourceVolume] = {},
    audio_file_format: str = "wav",
    background_effect: str = "white_noise",
    foreground_effect: str = "ac_noise_minimal",
    foreground_effect_position: RoomPosition = RoomPosition.TOP_RIGHT
) -> AudioDialog:
    """
    Generates room acoustics simulation for the dialogue audio.

    This function simulates how the dialogue would sound in a specific room environment
    by applying room acoustics effects such as reverberation, echo, and spatial positioning.
    The simulation uses the pyroomacoustics library to model realistic acoustic conditions.

    The process:
    1. Creates an AcousticsSimulator with the specified room configuration
    2. Extracts audio sources from the dialogue turns
    3. Applies room acoustics simulation with specified source volumes
    4. Saves the resulting audio with room effects applied
    5. Updates the dialog with room acoustics file paths and metadata

    :param dialog: The AudioDialog object containing turns with generated audio.
    :type dialog: AudioDialog
    :param room: Room configuration specifying dimensions, materials, and microphone position.
    :type room: Room
    :param dialog_directory: Directory path for organizing the dialog's audio files.
    :type dialog_directory: str
    :param room_name: Name identifier for this room configuration.
    :type room_name: str
    :param kwargs_pyroom: Additional parameters for pyroomacoustics simulation.
    :type kwargs_pyroom: dict
    :param source_volumes: Dictionary mapping source identifiers to volume levels.
    :type source_volumes: dict[str, SourceVolume]
    :param audio_file_format: Output audio file format (default: "wav").
    :type audio_file_format: str
    :param background_effect: Background audio effect type.
    :type background_effect: str
    :param foreground_effect: Foreground audio effect type.
    :type foreground_effect: str
    :param foreground_effect_position: Position for foreground effects.
    :type foreground_effect_position: RoomPosition
    :return: The AudioDialog with room acoustics simulation results and file paths.
    :rtype: AudioDialog
    """

    # Create the room acoustics simulator
    room_acoustics = AcousticsSimulator(room=room, kwargs_pyroom=kwargs_pyroom)

    _audio_accoustic = room_acoustics.simulate(
        sources=dialog.get_audio_sources(),
        source_volumes=source_volumes
    )

    # Save the audio file
    current_room_audio_path = os.path.join(
        dialog.audio_dir_path,
        dialog_directory,
        "exported_audios",
        "rooms",
        f"audio_pipeline_step3-{room_name}.{audio_file_format}"
    )
    sf.write(
        current_room_audio_path,
        _audio_accoustic,
        44_100
    )

    # Save the audio path and configuration into the dialog
    if room_name in dialog.audio_step_3_filepaths:
        logger.warning(f"Room '{room_name}' already exists in the dialog")

    # If the audio paths post processing are already in the dialog, use them, otherwise create a new dictionary
    if (
        room_name in dialog.audio_step_3_filepaths
        and "audio_paths_post_processing" in dialog.audio_step_3_filepaths[room_name]
        and dialog.audio_step_3_filepaths[room_name]["audio_paths_post_processing"] != {}
    ):
        audio_paths_post_processing = dialog.audio_step_3_filepaths[room_name]["audio_paths_post_processing"]
        logger.info(
            f"Existing audio paths for the post processing stage "
            f"already exist for room name: '{room_name}' and are kept unchanged"
        )
    else:
        audio_paths_post_processing = {}

    dialog.audio_step_3_filepaths[room_name] = {
        "audio_path": current_room_audio_path,
        "microphone_position": room.mic_position,
        "room_name": room_name,
        "room": room,
        "source_volumes": source_volumes,
        "kwargs_pyroom": kwargs_pyroom,
        "background_effect": background_effect,
        "foreground_effect": foreground_effect,
        "foreground_effect_position": foreground_effect_position,
        "audio_paths_post_processing": audio_paths_post_processing
    }

    return dialog
