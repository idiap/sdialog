"""
This module provides functionality to generate audio from text utterances in a dialog.
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
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.voice_database import BaseVoiceDatabase, Voice
from sdialog.audio.audio_utils import AudioUtils, SourceVolume, Role
from sdialog.audio.room_acoustics_simulator import RoomAcousticsSimulator

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_utterances_audios(
    dialog: AudioDialog,
    voice_database: BaseVoiceDatabase,
    tts_pipeline: BaseTTS,
    voices: dict[Role, Union[Voice, tuple[str, str]]] = None,
    keep_duplicate: bool = True
) -> AudioDialog:
    """
    Generates audio for each utterance in a Dialog object.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A Dialog object with audio turns.
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
    Attributes a voice to a persona.
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
        tts_pipeline: BaseTTS) -> np.ndarray:
    """
    Generates an audio recording of a text utterance based on the speaker persona.

    :param text: The text to be converted to audio.
    :type text: str
    :param voice: The voice identifier to use for the audio generation.
    :type voice: str
    :return: A numpy array representing the audio of the utterance.
    :rtype: np.ndarray
    """
    return tts_pipeline.generate(text, voice=voice)


def save_utterances_audios(
        dialog: AudioDialog,
        dir_audio: str,
        project_path: str,
        sampling_rate: int = 24_000) -> AudioDialog:
    """
    Save the utterances audios to the given path.
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
        source_volumes: dict[str, SourceVolume] = {}) -> AudioDialog:
    """
    Generates the audio room accoustic.
    """

    # Create the room acoustics simulator
    room_acoustics = RoomAcousticsSimulator(room=room, kwargs_pyroom=kwargs_pyroom)

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
        f"audio_pipeline_step3-{room_name}.wav"
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
