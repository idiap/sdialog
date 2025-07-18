"""
This module provides functionality to generate audio from text utterances in a dialog.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import torch
import scaper
import logging
import whisper
import numpy as np
import soundfile as sf
from typing import List, Tuple
from sdialog import Dialog, Turn
from sdialog.personas import BasePersona
from sdialog.util import remove_audio_tags
from sdialog.audio.tts_engine import BaseTTS
from scaper.dscaper_datatypes import DscaperAudio, DscaperTimeline, DscaperEvent, DscaperGenerate
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.voice_database import BaseVoiceDatabase

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3", device=device)

def _master_audio(dialog: AudioDialog) -> np.ndarray:
    """
    Combines multiple audio segments into a single master audio track.
    """
    return np.concatenate([turn.get_audio() for turn in dialog.turns])


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
        turn.set_audio(utterance_audio)
        turn.voice = turn_voice

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


def generate_word_alignments(dialog: AudioDialog) -> AudioDialog:
    """
    Generates word alignments for each utterance in a Dialog object.
    """
    for turn in dialog.turns:
        result = whisper_model.transcribe(
            turn.get_audio(),
            word_timestamps=True,
            fp16=False
        )
        turn.alignment = result['segments'][0]['words']
        turn.transcript = result['text']

    return dialog


def save_utterances_audios(dialog: AudioDialog, dir_audio: str) -> AudioDialog:
    """
    Save the utterances audios to the given path.
    """

    dialog.audio_dir_path = dir_audio.rstrip("/")
    os.makedirs(f"{dialog.audio_dir_path}/dialog_{dialog.id}/utterances", exist_ok=True)
    os.makedirs(f"{dialog.audio_dir_path}/dialog_{dialog.id}/exported_audios", exist_ok=True)

    for idx, turn in enumerate(dialog.turns):
        
        turn.audio_path = f"{dialog.audio_dir_path}/dialog_{dialog.id}/utterances/{idx}_{turn.speaker}.wav"        
        turn.audio_duration = turn.get_audio().shape[0] / 24_000

        sf.write(
            turn.audio_path,
            turn.get_audio(),
            24_000
        )
    
    return dialog


def send_utterances_to_dscaper(dialog: AudioDialog, dscaper: scaper.Dscaper) -> AudioDialog:
    """
    Sends the utterances to DSCAPER.
    """

    for turn in dialog.turns:

        metadata = DscaperAudio(
            library=f"dialog_{dialog.id}",
            label=turn.speaker,
            filename=os.path.basename(turn.audio_path)
        )
        
        resp = dscaper.store_audio(turn.audio_path, metadata)
        
        if resp.status != "success":
            logging.error(f"Problem storing audio for turn {turn.audio_path}: {resp.message}")

    return dialog


def generate_dscaper_timeline(dialog: AudioDialog, dscaper: scaper.Dscaper) -> AudioDialog:
    """
    Generates a DSCAPER format for a Dialog object.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: AudioDialog
    :param dscaper: The dscaper object.
    :type dscaper: scaper.Dscaper
    :return: A Dialog object with DSCAPER format.
    :rtype: AudioDialog
    """
    timeline_name = f"dialog_{dialog.id}"
    total_duration = dialog.get_combined_audio().shape[0] / 24_000

    timeline_metadata = DscaperTimeline(
        name=timeline_name,
        duration=total_duration,
        description=f"Timeline for dialog {dialog.id}"
    )
    dscaper.create_timeline(timeline_metadata)

    current_time = 0.0
    for i, turn in enumerate(dialog.turns):
        position = "chair_1" if i % 2 == 0 else "chair_2"
        event_metadata = DscaperEvent(
            library=timeline_name,
            label=["const", turn.speaker],
            source_file=["const", os.path.basename(turn.audio_path)],
            event_time=["const", str(current_time)],
            event_duration=turn.audio_duration,
            speaker=turn.speaker,
            text=turn.text,
            position=position,
        )
        dscaper.add_event(timeline_name, event_metadata)
        current_time += turn.audio_duration

    generate_metadata = DscaperGenerate(seed=0, save_isolated_positions=True)
    resp = dscaper.generate_timeline(timeline_name, generate_metadata)

    if resp.status == "success":
        content = DscaperGenerate(**resp.content)
        logging.info(f"Successfully generated dscaper timeline with ID: {content.id}")
    else:
        logging.error(f"Failed to generate dscaper timeline for {timeline_name}: {resp.message}")

    print(f"Timeline path: {resp.content.timeline_path}")

    return dialog


# TODO: Implement this function
def generate_audio_room_accoustic(dialog: AudioDialog) -> AudioDialog:
    """
    Generates the audio room accoustic.
    """
    return dialog


def audio_pipeline(
    dialog: AudioDialog,
    voice_database: BaseVoiceDatabase,
    tts_pipeline: BaseTTS,
    dir_audio: str,
    dscaper: scaper.Dscaper) -> AudioDialog:
    """
    Converts a Dialog object into a single audio track by generating audio for each utterance.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: Dialog
    :return: A Dialog object with the audio pipeline applied.
    :rtype: AudioDialog
    :param voice_database: The voice database to use for the audio generation.
    :type voice_database: BaseVoiceDatabase
    :param tts_pipeline: The TTS pipeline to use for the audio generation.
    :type tts_pipeline: BaseTTS
    :param dir_audio: The directory to save the audio files.
    :type dir_audio: str
    """

    dialog = generate_utterances_audios(
        dialog,
        voice_database=voice_database,
        tts_pipeline=tts_pipeline
    )

    dialog = save_utterances_audios(dialog, dir_audio)

    dialog.set_combined_audio(
        _master_audio(dialog)
    )
    # save the combined audio to exported_audios folder
    sf.write(
        f"{dialog.audio_dir_path}/dialog_{dialog.id}/exported_audios/combined_audio.wav",
        dialog.get_combined_audio(),
        24_000
    )

    # dialog = generate_word_alignments(dialog)

    # TODO: Generate SNR and position of the speaker in the room

    dialog = send_utterances_to_dscaper(dialog, dscaper)

    dialog = generate_dscaper_timeline(dialog, dscaper)
    
    dialog = generate_audio_room_accoustic(dialog)

    return dialog
