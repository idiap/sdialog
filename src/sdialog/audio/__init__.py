"""
This module provides functionality to generate audio from text utterances in a dialog.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import torch
import numpy as np
import soundfile as sf
from sdialog import Dialog, Turn
from sdialog.personas import BasePersona
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.audio_utils import AudioUtils
from sdialog.audio.room import MicrophonePosition
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.voice_database import BaseVoiceDatabase
from sdialog.audio.room_acoustics_simulator import RoomAcousticsSimulator

device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_persona_voice(
        dialog: Dialog,
        turn: Turn) -> BasePersona:
    """
    Gets a persona from a dialog.
    """
    persona = dialog.personas[turn.speaker]
    return persona["_metadata"]["voice"]


def generate_utterances_audios(
        dialog: AudioDialog,
        voice_database: BaseVoiceDatabase,
        tts_pipeline: BaseTTS) -> AudioDialog:
    """
    Generates audio for each utterance in a Dialog object.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A Dialog object with audio turns.
    :rtype: AudioDialog
    """

    # Match the voice to the persona of the dialog
    dialog = match_voice_to_persona(dialog, voice_database=voice_database)

    for turn in dialog.turns:

        # Get the voice of the turn
        turn_voice = _get_persona_voice(dialog, turn)["voice"]

        # Generate the utterance audio
        utterance_audio, sampling_rate = generate_utterance(
            text=AudioUtils.remove_audio_tags(turn.text),
            voice=turn_voice,
            tts_pipeline=tts_pipeline
        )

        # Set the utterance audio and voice to the turn
        turn.set_audio(utterance_audio, sampling_rate)
        turn.voice = turn_voice

    return dialog


def match_voice_to_persona(
        dialog: AudioDialog,
        voice_database: BaseVoiceDatabase) -> AudioDialog:
    """
    Matches a voice to a persona.
    """
    for speaker, persona in dialog.personas.items():
        persona["_metadata"]["voice"] = voice_database.get_voice(genre=persona["gender"], age=persona["age"])
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


def generate_word_alignments(
        dialog: AudioDialog,
        whisper_model_name: str = "large-v3") -> AudioDialog:
    """
    Generates word alignments for each utterance in a Dialog object.
    """
    from sdialog.audio.audio_utils import AudioUtils

    whisper_model = AudioUtils.get_whisper_model(model_name=whisper_model_name)

    for turn in dialog.turns:
        result = whisper_model.transcribe(turn.get_audio(), word_timestamps=True, fp16=False, language="en")
        turn.alignment = result["segments"][0]["words"]
        turn.transcript = result["text"]

    return dialog


def save_utterances_audios(
        dialog: AudioDialog,
        dir_audio: str,
        sampling_rate: int = 24_000) -> AudioDialog:
    """
    Save the utterances audios to the given path.
    """

    dialog.audio_dir_path = dir_audio.rstrip("/")
    os.makedirs(f"{dialog.audio_dir_path}/dialog_{dialog.id}/utterances", exist_ok=True)
    os.makedirs(f"{dialog.audio_dir_path}/dialog_{dialog.id}/exported_audios", exist_ok=True)

    current_time = 0.0

    for idx, turn in enumerate(dialog.turns):
        turn.audio_path = f"{dialog.audio_dir_path}/dialog_{dialog.id}/utterances/{idx}_{turn.speaker}.wav"
        turn.audio_duration = turn.get_audio().shape[0] / sampling_rate
        turn.audio_start_time = current_time
        current_time += turn.audio_duration

        sf.write(turn.audio_path, turn.get_audio(), sampling_rate)

    return dialog


def generate_audio_room_accoustic(
        dialog: AudioDialog,
        microphone_position: MicrophonePosition) -> AudioDialog:
    """
    Generates the audio room accoustic.
    """

    # Create the room acoustics simulator
    room_acoustics = RoomAcousticsSimulator(
        dialog.get_room()
    )

    # Add the microphone to the room acoustics simulator
    room_acoustics.set_microphone_position(microphone_position)

    # Simulate the audio
    _audio_accoustic = room_acoustics.simulate(
        dialog.get_audio_sources()
    )

    # Save the audio
    dialog.audio_step_3_filepath = os.path.join(
        dialog.audio_dir_path,
        f"dialog_{dialog.id}",
        "exported_audios",
        "audio_pipeline_step3.wav"
    )
    sf.write(
        dialog.audio_step_3_filepath,
        _audio_accoustic,
        44_100
    )

    return dialog
