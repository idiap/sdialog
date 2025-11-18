"""
This module provides utility functions for dSCAPER integration in the sdialog library.

The module includes functions for integrating with the dSCAPER framework for
realistic audio environment simulation. It provides utilities for sending
audio utterances to dSCAPER, generating timelines, and managing audio sources
for room acoustics simulation.

Key Features:

  - dSCAPER integration for audio environment simulation
  - Timeline generation with background and foreground effects
  - Audio source management for room acoustics
  - Support for multiple audio file formats
  - Comprehensive logging and error handling

Example:

    .. code-block:: python

        from sdialog.audio.dscaper_utils import send_utterances_to_dscaper, generate_dscaper_timeline
        from sdialog.audio.room import RoomPosition

        # Send utterances to dSCAPER
        dialog = send_utterances_to_dscaper(
            dialog=audio_dialog,
            _dscaper=dscaper_instance,
            dialog_directory="my_dialog"
        )

        # Generate dSCAPER timeline
        dialog = generate_dscaper_timeline(
            dialog=audio_dialog,
            _dscaper=dscaper_instance,
            dialog_directory="my_dialog",
            background_effect="white_noise",
            foreground_effect="ac_noise_minimal"
        )
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import scaper
import shutil
import logging

from sdialog.audio.utils import logger
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.room import AudioSource, RoomPosition
from scaper.dscaper_datatypes import (
    DscaperAudio,
    DscaperTimeline,
    DscaperEvent,
    DscaperGenerate,
    DscaperBackground
)


def send_utterances_to_dscaper(
        dialog: AudioDialog,
        _dscaper: scaper.Dscaper,
        dialog_directory: str) -> AudioDialog:
    """
    Send audio utterances to dSCAPER database for timeline generation.

    Processes all audio utterances from the dialogue and stores them in the
    dSCAPER database with appropriate metadata. This function handles the
    integration between the audio dialogue and dSCAPER for realistic audio
    environment simulation.

    :param dialog: Audio dialogue containing turns with audio data.
    :type dialog: AudioDialog
    :param _dscaper: dSCAPER instance for audio database management.
    :type _dscaper: scaper.Dscaper
    :param dialog_directory: Directory name for organizing audio files in dSCAPER.
    :type dialog_directory: str
    :return: Audio dialogue with updated dSCAPER storage status.
    :rtype: AudioDialog
    """

    count_audio_added = 0
    count_audio_present = 0
    count_audio_error = 0

    for turn in dialog.turns:

        metadata = DscaperAudio(
            library=dialog_directory, label=turn.speaker, filename=os.path.basename(turn.audio_path)
        )

        resp = _dscaper.store_audio(turn.audio_path, metadata)

        if resp.status != "success":
            if "File already exists. Use PUT to update it." in resp.content["description"]:
                count_audio_present += 1
                turn.is_stored_in_dscaper = True
            else:
                logger.error(f"Problem storing audio for turn {turn.audio_path}")
                logger.error(f"Error: {resp.content['description']}")
                count_audio_error += 1
        else:
            count_audio_added += 1
            turn.is_stored_in_dscaper = True

    logger.info("[dSCAPER] " + "=" * 30)
    logger.info("[dSCAPER] " + "# Audio sent to dSCAPER")
    logger.info("[dSCAPER] " + "=" * 30)
    logger.info("[dSCAPER] " + f"Already present: {count_audio_present}")
    logger.info("[dSCAPER] " + f"Correctly added: {count_audio_added}")
    logger.info("[dSCAPER] " + f"Errors: {count_audio_error}")
    logger.info("[dSCAPER] " + "=" * 30)

    return dialog


def generate_dscaper_timeline(
        dialog: AudioDialog,
        dscaper: scaper.Dscaper,
        dialog_directory: str,
        sampling_rate: int = 24_000,
        background_effect: str = "white_noise",
        foreground_effect: str = "ac_noise_minimal",
        foreground_effect_position: RoomPosition = RoomPosition.TOP_RIGHT,
        audio_file_format: str = "wav",
        seed: int = 0,
        referent_db: int = -40,
        reverberation: int = 0
) -> AudioDialog:
    """
    Generate a dSCAPER timeline for realistic audio environment simulation.

    Creates a comprehensive timeline in dSCAPER with background and foreground
    effects, along with all dialogue utterances positioned according to their
    timing and speaker roles. The timeline is then generated to produce a
    realistic audio environment with spatial positioning and acoustic effects.

    :param dialog: Audio dialogue containing turns with audio data.
    :type dialog: AudioDialog
    :param dscaper: dSCAPER instance for timeline generation.
    :type dscaper: scaper.Dscaper
    :param dialog_directory: Directory name for organizing timeline in dSCAPER.
    :type dialog_directory: str
    :param sampling_rate: Audio sampling rate in Hz.
    :type sampling_rate: int
    :param background_effect: Background audio effect type.
    :type background_effect: str
    :param foreground_effect: Foreground audio effect type.
    :type foreground_effect: str
    :param foreground_effect_position: Position for foreground effects in the room.
    :type foreground_effect_position: RoomPosition
    :param audio_file_format: Audio file format for output (wav, mp3, flac).
    :type audio_file_format: str
    :param seed: Seed for random number generator.
    :type seed: int
    :param referent_db: Referent dB for audio level normalization.
    :type referent_db: int
    :param reverberation: Reverberation time in seconds.
    :type reverberation: int
    :return: Audio dialogue with generated timeline and audio sources.
    :rtype: AudioDialog
    """

    if audio_file_format not in ["mp3", "wav", "flac"]:
        raise ValueError((
            "The audio file format must be either mp3, wav or flac."
            f"You provided: {audio_file_format}"
        ))

    timeline_name = dialog_directory
    total_duration = dialog.get_combined_audio().shape[0] / sampling_rate
    dialog.total_duration = total_duration
    dialog.timeline_name = timeline_name

    sox_logger = logging.getLogger('sox')
    original_level = sox_logger.level
    sox_logger.setLevel(logging.ERROR)

    try:
        # Create the timeline
        timeline_metadata = DscaperTimeline(
            name=timeline_name,
            duration=total_duration,
            description=f"Timeline for dialog {dialog.id}"
        )
        dscaper.create_timeline(timeline_metadata)

        # Add the background to the timeline
        background_metadata = DscaperBackground(
            library="background",
            label=[
                "const",
                background_effect if background_effect is not None and background_effect != "" else "white_noise"
            ],
            source_file=["choose", "[]"]
        )
        dscaper.add_background(timeline_name, background_metadata)

        # Add the foreground to the timeline
        if foreground_effect is not None and foreground_effect != "":
            foreground_metadata = DscaperEvent(
                library="foreground",
                speaker="foreground",
                text="foreground",
                label=["const", foreground_effect],
                source_file=["choose", "[]"],
                event_time=["const", "0"],
                event_duration=["const", str(f"{total_duration:.1f}")],  # Force infinite loop
                position=(
                    foreground_effect_position
                    if foreground_effect_position is not None
                    else RoomPosition.TOP_RIGHT
                ),
            )
            dscaper.add_event(timeline_name, foreground_metadata)

        # Add the events and utterances to the timeline
        current_time = 0.0
        for i, turn in enumerate(dialog.turns):

            # The role is used here to identify the source of emission of the audio
            # We consider that it is immutable and will not change over the dialog timeline
            _speaker_role = dialog.speakers_roles[turn.speaker]

            _event_metadata = DscaperEvent(
                library=timeline_name,
                label=["const", turn.speaker],
                source_file=["const", os.path.basename(turn.audio_path)],
                event_time=["const", str(f"{turn.audio_start_time:.1f}")],
                event_duration=["const", str(f"{turn.audio_duration:.1f}")],
                speaker=turn.speaker,
                text=turn.text,
                position=_speaker_role
            )
            dscaper.add_event(timeline_name, _event_metadata)
            current_time += turn.audio_duration

        # Generate the timeline
        resp = dscaper.generate_timeline(
            timeline_name,
            DscaperGenerate(
                seed=seed if seed is not None else 0,
                save_isolated_positions=True,
                ref_db=referent_db,
                reverb=reverberation,
                save_isolated_events=False
            ),
        )
    finally:
        sox_logger.setLevel(original_level)

    # Build the generate directory path
    soundscape_positions_path = os.path.join(
        dscaper.get_dscaper_base_path(),
        "timelines",
        timeline_name,
        "generate",
        resp.content["id"],
        "soundscape_positions"
    )

    # Build the path to the audio output
    audio_output_path = os.path.join(
        dscaper.get_dscaper_base_path(),
        "timelines",
        timeline_name,
        "generate",
        resp.content["id"],
        "soundscape.wav"
    )
    # Copy the audio output to the dialog audio directory
    dialog.audio_step_2_filepath = os.path.join(
        dialog.audio_dir_path,
        dialog_directory,
        "exported_audios",
        f"audio_pipeline_step2.{audio_file_format}"
    )
    shutil.copy(audio_output_path, dialog.audio_step_2_filepath)

    # Get the sounds files
    sounds_files = [_ for _ in os.listdir(soundscape_positions_path) if _.endswith(".wav")]

    # Build the audio sources for the room simulation
    for file_name in sounds_files:

        file_path = os.path.join(soundscape_positions_path, file_name)

        position_name = file_name.split(".")[0]

        dialog.add_audio_source(
            AudioSource(
                name=position_name,
                position=position_name,
                snr=-15.0 if position_name == "no_type" else 0.0,
                source_file=file_path
            )
        )

    # Check if the timeline was generated successfully
    if resp.status == "success":
        logger.info("Successfully generated dscaper timeline.")
    else:
        logger.error(f"Failed to generate dscaper timeline for {timeline_name}: {resp.message}")

    return dialog
