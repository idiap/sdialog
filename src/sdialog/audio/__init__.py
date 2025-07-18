"""
This module provides functionality to generate audio from text utterances in a dialog.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import numpy as np
from typing import List, Tuple
from sdialog import Dialog, Turn
from sdialog.personas import BasePersona
from sdialog.util import remove_audio_tags
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.voice_database import BaseVoiceDatabase


def _master_audio(dialog: AudioDialog) -> np.ndarray:
    """
    Combines multiple audio segments into a single master audio track.
    """
    return np.concatenate([turn.audio for turn in dialog.turns])


def _get_persona_voice(dialog: Dialog, turn: Turn) -> BasePersona:
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

    dialog = match_voice_to_persona(dialog, voice_database=voice_database)

    for turn in dialog.turns:
        turn_voice = _get_persona_voice(dialog, turn)["identifier"]
        utterance_audio = generate_utterance(
            remove_audio_tags(turn.text),
            turn_voice,
            tts_pipeline=tts_pipeline
        )
        turn.audio = utterance_audio

    return dialog


def match_voice_to_persona(dialog: AudioDialog, voice_database: BaseVoiceDatabase) -> AudioDialog:
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


@staticmethod
def audio_pipeline(dialog: AudioDialog, voice_database: BaseVoiceDatabase, tts_pipeline: BaseTTS) -> np.ndarray:
    """
    Converts a Dialog object into a single audio track by generating audio for each utterance.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A numpy array representing the combined audio of the dialog.
    :rtype: np.ndarray
    """

    dialog = generate_utterances_audios(dialog, voice_database=voice_database, tts_pipeline=tts_pipeline)

    dialog.set_combined_audio(
        _master_audio(dialog)
    )

    return dialog
