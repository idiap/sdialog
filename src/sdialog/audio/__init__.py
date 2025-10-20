"""
This module provides comprehensive functionality for generating audio from text utterances in dialogues.

The audio module extends the core sdialog functionality by adding:

  - Audio generation from text using various TTS engines (Kokoro, IndexTTS)
  - Voice databases with speaker characteristics (gender, age, language)
  - Room acoustics simulation for realistic audio environments
  - Audio dialogue processing with turn-based audio generation
  - Integration with external audio processing libraries (pyroomacoustics, scaper)

Key Components:

  - AudioDialog: Extended dialogue class with audio turn support
  - AudioTurn: Individual dialogue turns with associated audio data
  - BaseTTS: Abstract base class for text-to-speech engines
  - BaseVoiceDatabase: Voice database management with speaker characteristics
  - Room: 3D room specification for acoustics simulation
  - AcousticsSimulator: Room acoustics simulation engine

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
            tts_pipeline=tts
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
import random
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf
from typing import Union
from sdialog.audio.room import Room
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.utils import AudioUtils, SourceVolume, Role
from sdialog.audio.acoustics_simulator import AcousticsSimulator
from sdialog.audio.voice_database import BaseVoiceDatabase, Voice

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_utterances_audios(
    dialog: AudioDialog,
    voice_database: BaseVoiceDatabase,
    tts_pipeline: BaseTTS,
    voices: dict[Role, Union[Voice, tuple[str, str]]] = None,
    keep_duplicate: bool = True
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
    :return: The AudioDialog object with generated audio for each turn.
    :rtype: AudioDialog
    """

    # Attribute the voice to the persona of the dialog
    dialog = attribute_voice_to_persona(
        dialog,
        voice_database=voice_database,
        voices=voices,
        keep_duplicate=keep_duplicate
    )

    for turn in tqdm(dialog.turns, desc="Generating utterances audios"):

        # Get the voice of the turn
        turn.voice = dialog.personas[turn.speaker]["voice"].voice

        # Generate the utterance audio
        utterance_audio, sampling_rate = generate_utterance(
            text=AudioUtils.remove_audio_tags(turn.text),
            voice=turn.voice,
            tts_pipeline=tts_pipeline
        )

        # Set the utterance audio to the turn
        turn.set_audio(utterance_audio, sampling_rate)

    return dialog


def attribute_voice_to_persona(
    dialog: AudioDialog,
    voice_database: BaseVoiceDatabase,
    voices: dict[Role, Union[Voice, tuple[str, str]]] = None,
    keep_duplicate: bool = True
) -> AudioDialog:
    """
    Assigns appropriate voices to speakers based on their persona characteristics.

    This function analyzes each speaker's persona information (gender, age, language)
    and assigns a suitable voice from the voice database. If persona information is
    missing, default values are assigned with appropriate warnings.

    Voice assignment logic:
    1. If explicit voices are provided, use them for the specified roles
    2. If no explicit voices, select from database based on persona characteristics
    3. Handle missing persona information by assigning random/default values
    4. Support both Voice objects and voice identifier tuples

    :param dialog: The AudioDialog object containing speaker personas.
    :type dialog: AudioDialog
    :param voice_database: Database containing available voices with metadata.
    :type voice_database: BaseVoiceDatabase
    :param voices: Optional dictionary mapping speaker roles to specific voices.
                  Keys are Role enums, values can be Voice objects or (identifier, language) tuples.
    :type voices: Optional[dict[Role, Union[Voice, tuple[str, str]]]]
    :param keep_duplicate: If True, allows voice reuse across speakers.
    :type keep_duplicate: bool
    :return: The AudioDialog with voice assignments added to each persona.
    :rtype: AudioDialog
    """
    for speaker, persona in dialog.personas.items():

        # Check if the information about the voice is already in the persona, else add a random information
        if "gender" not in persona or persona["gender"] is None:
            persona["gender"] = random.choice(["male", "female"])
            logging.warning(f"Gender not found in the persona {speaker}, a random gender has been added")

        if "age" not in persona or persona["age"] is None:
            persona["age"] = random.randint(18, 65)
            logging.warning(f"Age not found in the persona {speaker}, a random age has been added")

        if "language" not in persona or persona["language"] is None:
            persona["language"] = "english"
            logging.warning(f"Language not found in the persona {speaker}, english has been considered by default")

        # Get the role of the speaker (speaker_1 or speaker_2)
        role: Role = dialog.speakers_roles[speaker]

        if voices is not None and voices != {} and role not in voices:
            raise ValueError(f"Voice for role {role} not found in the voices dictionary")

        # If no voices are provided, get a voice from the voice database based on the gender, age and language
        if voices is None or voices == {}:
            persona["voice"] = voice_database.get_voice(
                gender=persona["gender"],
                age=persona["age"],
                lang=persona["language"],
                keep_duplicate=keep_duplicate
            )

        # If the voice of the speaker is provided as a Voice object
        elif isinstance(voices[role], Voice):
            persona["voice"] = voices[role]

        # If the voice of the speaker is provided as an identifier (like "am_echo")
        elif isinstance(voices[role], tuple):
            _identifier, _language = voices[role]
            persona["voice"] = voice_database.get_voice_by_identifier(
                _identifier,
                _language,
                keep_duplicate=keep_duplicate
            )

    return dialog


def generate_utterance(
        text: str,
        voice: str,
        tts_pipeline: BaseTTS) -> tuple[np.ndarray, int]:
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
    :return: A tuple containing the audio data as a numpy array and the sampling rate.
    :rtype: tuple[np.ndarray, int]
    """
    return tts_pipeline.generate(text, voice=voice)


def save_utterances_audios(
    dialog: AudioDialog,
    dir_audio: str,
    project_path: str,
    sampling_rate: int = 24_000
) -> AudioDialog:
    """
    Saves individual utterance audio files to the specified directory structure.

    This function creates the necessary directory structure and saves each turn's
    audio as a separate WAV file. It also calculates timing information for each
    utterance and updates the AudioTurn objects with file paths and timing data.

    Directory structure created:
    - {project_path}/utterances/ - Individual utterance audio files
    - {project_path}/exported_audios/ - Combined audio files
    - {project_path}/exported_audios/rooms/ - Room acoustics simulation results

    :param dialog: The AudioDialog object containing turns with generated audio.
    :type dialog: AudioDialog
    :param dir_audio: Base directory path for audio storage.
    :type dir_audio: str
    :param project_path: Project-specific path for organizing audio files.
    :type project_path: str
    :param sampling_rate: Audio sampling rate for saving files (default: 24000 Hz).
    :type sampling_rate: int
    :return: The AudioDialog with updated file paths and timing information.
    :rtype: AudioDialog
    """

    dialog.audio_dir_path = dir_audio.rstrip("/")
    os.makedirs(f"{project_path}/utterances", exist_ok=True)
    os.makedirs(f"{project_path}/exported_audios", exist_ok=True)
    os.makedirs(f"{project_path}/exported_audios/rooms", exist_ok=True)

    current_time = 0.0

    for idx, turn in enumerate(dialog.turns):
        turn.audio_path = f"{project_path}/utterances/{idx}_{turn.speaker}.wav"
        turn.audio_duration = turn.get_audio().shape[0] / sampling_rate
        turn.audio_start_time = current_time
        current_time += turn.audio_duration

        sf.write(turn.audio_path, turn.get_audio(), sampling_rate)

    return dialog


def generate_audio_room_accoustic(
    dialog: AudioDialog,
    room: Room,
    dialog_directory: str,
    room_name: str,
    kwargs_pyroom: dict = {},
    source_volumes: dict[str, SourceVolume] = {},
    audio_file_format: str = "wav"
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
        logging.warning(f"Room '{room_name}' already exists in the dialog")

    dialog.audio_step_3_filepaths[room_name] = {
        "audio_path": current_room_audio_path,
        "microphone_position": room.mic_position,
        "room_name": room_name,
        "room": room
    }

    return dialog
