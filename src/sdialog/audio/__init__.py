"""
This module provides functionality to generate audio from text utterances in a dialog.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import numpy as np
import soundfile as sf
from typing import List, Tuple
from sdialog import Dialog, Turn
from sdialog.personas import BasePersona
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.voice_database import BaseVoiceDatabase


def _master_audio(dialogue_audios: List[Tuple[np.ndarray, str]]) -> np.ndarray:
    """
    Combines multiple audio segments into a single master audio track.
    """
    return np.concatenate([da[0] for da in dialogue_audios])


def _get_persona_voice(dialog: Dialog, turn: Turn) -> BasePersona:
    """
    Gets a persona from a dialog.
    """
    persona = dialog.personas[turn.speaker]
    return persona["_metadata"]["voice"]


def generate_utterances_audios(dialog: Dialog, voice_database: BaseVoiceDatabase, tts_pipeline: BaseTTS) -> List[Tuple[np.ndarray, str]]:
    """
    Generates audio for each utterance in a Dialog object.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A list of tuples consisting of a numpy arrays and a string, representing the audio of an utterance and the speaker identity.
    :rtype: list
    """

    dialog = match_voice_to_persona(dialog, voice_database=voice_database)

    dialogue_audios = []

    for turn in dialog.turns:
        turn_voice = _get_persona_voice(dialog, turn)["identifier"]
        utterance_audio = generate_utterance(turn.text, turn_voice, tts_pipeline=tts_pipeline)
        dialogue_audios.append((utterance_audio, turn.speaker))

    return dialogue_audios


def match_voice_to_persona(dialog: Dialog, voice_database: BaseVoiceDatabase) -> Dialog:
    """
    Matches a voice to a persona.
    """
    for speaker, persona in dialog.personas.items():
        persona["_metadata"]["voice"] = voice_database.get_voice(
            genre=persona["gender"], age=persona["age"]
        )
    return dialog


def generate_utterance(text: str, voice: str, tts_pipeline: BaseTTS) -> np.ndarray:
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


def to_wav(audio, output_file, sampling_rate=24_000) -> None:
    """
    Combines multiple audio segments into a single master audio track.
    """
    sf.write(output_file, audio, sampling_rate)


def dialog_to_audio(dialog: Dialog, voice_database: BaseVoiceDatabase, tts_pipeline: BaseTTS) -> np.ndarray:
    """
    Converts a Dialog object into a single audio track by generating audio for each utterance.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A numpy array representing the combined audio of the dialog.
    :rtype: np.ndarray
    """

    dialogue_audios = generate_utterances_audios(dialog, voice_database=voice_database, tts_pipeline=tts_pipeline)

    combined_audio = _master_audio(dialogue_audios)

    return combined_audio
