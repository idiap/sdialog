"""
This module provides functionality to generate audio from text utterances in a dialog.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import json
import numpy as np
from typing import List, Tuple

# Audio processing
import soundfile as sf
from kokoro import KPipeline

from sdialog import Dialog


pipeline = KPipeline(lang_code='a')


def _master_audio(dialogue_audios: list) -> np.ndarray:
    """
    Combines multiple audio segments into a single master audio track.
    """
    return np.concatenate(dialogue_audios)


def generate_utterances_audios(dialog: Dialog) -> List[Tuple[np.ndarray, str]]:
    """
    Generates audio for each utterance in a Dialog object.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A list of numpy arrays, each representing the audio of an utterance.
    :rtype: list
    """

    dialogue_audios = []

    for turn in dialog.turns:

        utterance_audio = generate_utterance(turn.text, dialog.personas[turn.speaker])
        dialogue_audios.append((utterance_audio, turn.speaker))

    return dialogue_audios


def generate_utterance(text: str, persona: dict, voice: str = "af_heart") -> np.ndarray:
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


def to_wav(audio, output_file, sampling_rate=16_000) -> None:
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


def make_serializable(data: dict) -> dict:
    """
    Converts non-serializable values in a dictionary to strings so the dictionary can be safely serialized to JSON.

    :param data: The dictionary to process.
    :type data: dict
    :return: The dictionary with all values JSON-serializable.
    :rtype: dict
    """

    if type(data) is not dict:
        raise TypeError("Input must be a dictionary")

    for key, value in data.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            data[key] = str(value)

    return data
