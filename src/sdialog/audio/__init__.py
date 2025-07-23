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
from sdialog import Dialog, Turn
from sdialog.personas import BasePersona
from sdialog.util import remove_audio_tags
from sdialog.audio.room import AudioSource
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.voice_database import BaseVoiceDatabase
from sdialog.audio.room_acoustics_simulator import RoomAcousticsSimulator
from scaper.dscaper_datatypes import DscaperAudio, DscaperTimeline, DscaperEvent, DscaperGenerate, DscaperBackground

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3", device=device)


def _get_persona_voice(dialog: Dialog, turn: Turn) -> BasePersona:
    """
    Gets a persona from a dialog.
    """
    persona = dialog.personas[turn.speaker]
    return persona["_metadata"]["voice"]


def generate_utterances_audios(
    dialog: AudioDialog, voice_database: BaseVoiceDatabase, tts_pipeline: BaseTTS
) -> AudioDialog:
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
        utterance_audio = generate_utterance(remove_audio_tags(turn.text), turn_voice, tts_pipeline=tts_pipeline)
        turn.set_audio(utterance_audio)
        turn.voice = turn_voice

    return dialog


def match_voice_to_persona(dialog: AudioDialog, voice_database: BaseVoiceDatabase) -> AudioDialog:
    """
    Matches a voice to a persona.
    """
    for speaker, persona in dialog.personas.items():
        persona["_metadata"]["voice"] = voice_database.get_voice(genre=persona["gender"], age=persona["age"])
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
        result = whisper_model.transcribe(turn.get_audio(), word_timestamps=True, fp16=False)
        turn.alignment = result["segments"][0]["words"]
        turn.transcript = result["text"]

    return dialog


def save_utterances_audios(dialog: AudioDialog, dir_audio: str, sampling_rate: int = 24_000) -> AudioDialog:
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


def send_utterances_to_dscaper(dialog: AudioDialog, _dscaper: scaper.Dscaper) -> AudioDialog:
    """
    Sends the utterances audio files to dSCAPER database.
    """

    for turn in dialog.turns:
        metadata = DscaperAudio(
            library=f"dialog_{dialog.id}", label=turn.speaker, filename=os.path.basename(turn.audio_path)
        )

        resp = _dscaper.store_audio(turn.audio_path, metadata)

        if resp.status != "success":
            logging.error(f"Problem storing audio for turn {turn.audio_path}")
        else:
            turn.is_stored_in_dscaper = True

    return dialog


def generate_dscaper_timeline(
    dialog: AudioDialog, _dscaper: scaper.Dscaper, sampling_rate: int = 24_000
) -> AudioDialog:
    """
    Generates a dSCAPER timeline for a Dialog object.

    :param dialog: The Dialog object containing the conversation.
    :type dialog: AudioDialog
    :param _dscaper: The _dscaper object.
    :type _dscaper: scaper.Dscaper
    :return: A Dialog object with dSCAPER timeline.
    :rtype: AudioDialog
    """
    timeline_name = f"dialog_{dialog.id}"
    total_duration = dialog.get_combined_audio().shape[0] / sampling_rate
    dialog.total_duration = total_duration
    dialog.timeline_name = timeline_name

    # Create the timeline
    timeline_metadata = DscaperTimeline(
        name=timeline_name, duration=total_duration, description=f"Timeline for dialog {dialog.id}"
    )
    _dscaper.create_timeline(timeline_metadata)

    # Add the background to the timeline
    background_metadata = DscaperBackground(
        library="background", label=["const", "ac_noise"], source_file=["choose", "[]"]
    )
    _dscaper.add_background(timeline_name, background_metadata)

    # Add the foreground to the timeline
    # TODO: Add the foreground to the timeline dynamically
    foreground_metadata = DscaperEvent(
        library="foreground",
        label=["const", "white_noise"],
        source_file=["choose", "[]"],
        event_time=["const", "0"],
        event_duration=["const", "0.1"],
        position="at_desk_sitting",
        speaker="foreground",
        text="foreground",
    )
    _dscaper.add_event(timeline_name, foreground_metadata)

    # Add the events and utterances to the timeline
    current_time = 0.0
    for i, turn in enumerate(dialog.turns):
        # TODO: Remove this hardcoded default position
        default_position = "at_desk_sitting" if turn.speaker == "DOCTOR" else "next_to_desk_sitting"
        _event_metadata = DscaperEvent(
            library=timeline_name,
            label=["const", turn.speaker],
            source_file=["const", os.path.basename(turn.audio_path)],
            event_time=["const", str(f"{turn.audio_start_time:.1f}")],
            event_duration=["const", str(f"{turn.audio_duration:.1f}")],
            speaker=turn.speaker,
            text=turn.text,
            position=turn.position if turn.position else default_position,
            # TODO: Add the microphone position
        )
        _dscaper.add_event(timeline_name, _event_metadata)
        current_time += turn.audio_duration

    # Generate the timeline
    resp = _dscaper.generate_timeline(
        timeline_name,
        DscaperGenerate(seed=0, save_isolated_positions=True, ref_db=-20, reverb=0, save_isolated_events=False),
    )

    # Buil the generate directory path
    soundscape_positions_path = os.path.join(
        _dscaper.get_dscaper_base_path(),
        "timelines",
        timeline_name,
        "generate",
        resp.content["id"],
        "soundscape_positions"
    )

    # Get the sounds files
    sounds_files = [_ for _ in os.listdir(soundscape_positions_path) if _.endswith(".wav")]

    # Build the audio sources for the room simulation
    for file_name in sounds_files:

        file_path = os.path.join(soundscape_positions_path, file_name)

        dialog.add_audio_source(
            AudioSource(
                name=file_name.split(".")[0],
                position=file_name.split(".")[0],
                snr=0.0,
                source_file=file_path
            )
        )

    # Check if the timeline was generated successfully
    if resp.status == "success":
        logging.info("Successfully generated dscaper timeline.")
    else:
        logging.error(f"Failed to generate dscaper timeline for {timeline_name}: {resp.message}")

    return dialog


# TODO: Implement this function
def generate_audio_room_accoustic(dialog: AudioDialog) -> AudioDialog:
    """
    Generates the audio room accoustic.
    """

    room_acoustics = RoomAcousticsSimulator(dialog.get_room())

    room_acoustics.add_sources(
        dialog.get_audio_sources()
    )
    # room_acoustics.add_microphone()
    _audio_accoustic = room_acoustics.simulate()

    dialog._audio_step_3_filepath = os.join(
        dialog.audio_dir_path,
        f"dialog_{dialog.id}",
        "audio_pipeline_step3.wav"
    )
    sf.wavwrite(dialog._audio_step_3_filepath, _audio_accoustic, 16_000)

    return dialog
