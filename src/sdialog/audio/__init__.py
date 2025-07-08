"""
This module provides functionality to generate audio from text utterances in a dialog.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import numpy as np
from typing import List, Tuple

# Audio processing
import soundfile as sf
from kokoro import KPipeline

from sdialog import Dialog, Turn
from sdialog.personas import BasePersona


voice_database = {
    ("male", 45): {"identifier": "af_heart", "path": "af_heart.wav"},
    ("female", 23): {"identifier": "af_heart", "path": "af_heart.wav"}
}


pipeline = KPipeline(lang_code='a')


def _master_audio(dialogue_audios: List[Tuple[np.ndarray, str]]) -> np.ndarray:
    """
    Combines multiple audio segments into a single master audio track.
    """
    return np.concatenate([da[0] for da in dialogue_audios])


def _get_persona_voice(dialog: Dialog, turn: Turn) -> BasePersona:
    """
    Gets a persona from a dialog.
    """
    return dialog.personas[turn.speaker]._metadata["voice"]


def generate_utterances_audios(dialog: Dialog) -> List[Tuple[np.ndarray, str]]:
    """
    Generates audio for each utterance in a Dialog object.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A list of tuples consisting of a numpy arrays and a string, representing the audio of an utterance and the speaker identity.
    :rtype: list
    """

    dialog = match_voice_to_persona(dialog)

    dialogue_audios = []

    for turn in dialog.turns:
        turn_voice = _get_persona_voice(dialog, turn)["identifier"]
        utterance_audio = generate_utterance(turn.text, turn_voice)
        dialogue_audios.append((utterance_audio, turn.speaker))

    return dialogue_audios


def match_voice_to_persona(dialog: Dialog) -> Dialog:
    """
    Matches a voice to a persona.
    """
    for speaker, persona in dialog.personas.items():
        print(persona)
        persona["_metadata"]["voice"] = voice_database[(persona["gender"], persona["age"])]
    return dialog


def generate_utterance(text: str, persona: dict, voice: str) -> np.ndarray:
    """
    Generates an audio recording of a text utterance based on the speaker persona.

    :param text: The text to be converted to audio.
    :type text: str
    :param persona: The speaker persona containing voice characteristics.
    :type persona: dict
    :param voice: The voice identifier to use for the audio generation.
    :type voice: str
    :return: A numpy array representing the audio of the utterance.
    :rtype: np.ndarray
    """

    generator = pipeline(text, voice=voice)

    gs, ps, audio = next(iter(generator))

    return audio


def to_wav(audio, output_file, sampling_rate=24_000) -> None:
    """
    Combines multiple audio segments into a single master audio track.
    """
    sf.write(output_file, audio, sampling_rate)


def dialog_to_audio(dialog: Dialog) -> np.ndarray:
    """
    Converts a Dialog object into a single audio track by generating audio for each utterance.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A numpy array representing the combined audio of the dialog.
    :rtype: np.ndarray
    """

    dialogue_audios = generate_utterances_audios(dialog)

    combined_audio = _master_audio(dialogue_audios)

    return combined_audio
