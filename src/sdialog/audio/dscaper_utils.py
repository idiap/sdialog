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
import json
import shutil
import dscaper
import logging
import soundfile as sf
from typing import Optional

from sdialog.audio.utils import logger, Role
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.room import AudioSource, RoomPosition, Room
from dscaper.dscaper_datatypes import (
    DscaperAudio,
    DscaperTimeline,
    DscaperEvent,
    DscaperGenerate,
    DscaperBackground
)


def send_utterances_to_dscaper(
        dialog: AudioDialog,
        _dscaper: dscaper.Dscaper,
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
    :type _dscaper: dscaper.Dscaper
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
            library=dialog_directory,
            label=turn.speaker,
            filename=os.path.basename(turn.audio_path)
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


def get_sound_effects_db(
    dscaper_manager: dscaper.Dscaper
) -> dict[str, dict]:
    """
    Returns the sound effects database.

    :param dscaper_manager: dSCAPER instance for audio database management.
    :type dscaper_manager: dscaper.Dscaper
    :return: Dictionary containing the sound effect metadata.
    :rtype: dict[str, dict]: Dictionary containing the sound effect metadata.
    :returns:
    {
        "my_label": {
            "library": "sfx|my_dataset_name",
            "label": "my_label",
            "description": "my_description",
            "duration": "1.0",
        }
    }
    :rtype: dict[str, dict]
    """

    # List all libraries in dscaper_manager that start with "sfx|"
    _libraries = [
        _l
        for _l in dscaper_manager.get_libraries().content
        if _l.startswith("sfx|")
    ]

    # Extract the items from the libraries that start with "sfx|"
    _libraries_labels = [
        (_library, dscaper_manager.get_labels(_library).content)
        for _library in _libraries
    ]

    metadata = {}

    for _lib, _labels in _libraries_labels:

        for _label in _labels:

            _metadata = dscaper_manager.get_label_metadata(
                library=_lib,
                label=_label,
                include_audios=True
            ).content

            _first_audio_description = json.loads(
                _metadata["audios"][0]["sandbox"]
            )["description"]

            metadata[_label] = {
                "library": _lib,
                "label": _label,
                "description": _first_audio_description,
            }

    return metadata


def _resolve_sound_effect_position(
        sfx_position: str,
        sfx_tag: str,
        default_role: str,
        room: Room) -> str:
    """
    Resolve the position of a sound effect.

    :param sfx_position: The position string to resolve.
    :type sfx_position: str
    :param sfx_tag: The tag of the sound effect.
    :type sfx_tag: str
    :param default_role: The default role to use if the position is invalid.
    :type default_role: str
    :param room: The room configuration for checking validity.
    :type room: Room
    :return: The resolved position.
    :rtype: str
    """

    sfx_position = sfx_position.replace("sfx|", "")

    if sfx_position == "human":
        return default_role

    is_valid = False

    # Check if it's a speaker role
    if sfx_position in [Role.SPEAKER_1.value, Role.SPEAKER_2.value]:
        is_valid = True

    # Check if it's a room position
    elif sfx_position.startswith("room-") and isinstance(RoomPosition(sfx_position), RoomPosition):
        is_valid = True

    # Check if it's a furniture
    elif sfx_position in room.furnitures:
        is_valid = True

    if not is_valid:
        logger.warning(
            f"Position '{sfx_position}' for sound effect '{sfx_tag}' "
            f"is not valid in the current room configuration. "
            f"Defaulting to '{default_role}'."
        )
        return default_role

    return sfx_position


def generate_dscaper_timeline(
        dialog: AudioDialog,
        dscaper: dscaper.Dscaper,
        dialog_directory: str,
        sampling_rate: int = 24_000,
        background_effect: str = "white_noise",
        foreground_effect: str = "ac_noise_minimal",
        foreground_effect_position: RoomPosition = RoomPosition.TOP_RIGHT,
        audio_file_format: str = "wav",
        seed: int = 0,
        referent_db: int = -40,
        reverberation: int = 0,
        room: Room = None
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
    :type dscaper: dscaper.Dscaper
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
    :param room: Room configuration for checking the validity of the positions.
    :type room: Room
    :return: Audio dialogue with generated timeline and audio sources.
    :rtype: AudioDialog
    """
    dialog.audio_sources = []

    if audio_file_format not in ["mp3", "wav", "flac"]:
        raise ValueError((
            "The audio file format must be either mp3, wav or flac."
            f"You provided: {audio_file_format}"
        ))

    # Compute the duration of the whole dialogue by considering for each turn:
    # the duration of the turn, the gap and the duration of the sound effects
    # dialog.total_duration = dialog.get_timeline_duration()

    timeline_name = dialog_directory
    dialog.timeline_name = timeline_name

    timeline_path = os.path.join(
        dscaper.get_dscaper_base_path(),
        "timelines",
        timeline_name,
    )
    if os.path.exists(timeline_path):
        logger.info(f"Timeline '{timeline_name}' already exists. Deleting it to avoid audio overlap.")
        shutil.rmtree(timeline_path)

    sox_logger = logging.getLogger('sox')
    original_level = sox_logger.level
    sox_logger.setLevel(logging.ERROR)

    try:
        # Create the timeline
        timeline_metadata = DscaperTimeline(
            name=timeline_name,
            # duration=dialog.total_duration, # BEFORE
            description=f"Timeline for dialog {dialog.id}"
        )
        dscaper.create_timeline(timeline_metadata)

        # Add the background to the timeline
        background_metadata = DscaperBackground(
            library="background",
            label=[
                "const",
                background_effect
                if background_effect is not None and background_effect != ""
                else "white_noise"
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
                # event_duration=["const", str(f"{dialog.total_duration:.1f}")],  # Force infinite loop
                position=(
                    foreground_effect_position
                    if foreground_effect_position is not None
                    else RoomPosition.TOP_RIGHT
                ),
            )
            dscaper.add_event(timeline_name, foreground_metadata)

        # Add the events and utterances to the timeline
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

            # Add sound effects events if any
            if hasattr(turn, 'sound_effects') and turn.sound_effects:

                for sfx in turn.sound_effects:

                    sfx_start_time = turn.audio_start_time + sfx['start_time']
                    sfx_position = _resolve_sound_effect_position(
                        sfx_position=sfx['position'],
                        sfx_tag=sfx['tag'],
                        default_role=_speaker_role,
                        room=room
                    )

                    _sfx_event_metadata = DscaperEvent(
                        library="sfx|sound_events",
                        label=["const", sfx['tag']],
                        source_file=["choose", "[]"],
                        event_time=["const", str(f"{sfx_start_time:.1f}")],
                        # event_duration=["const", str(f"{sfx['duration']:.1f}")],
                        position=sfx_position,
                        snr=["const", "-10.0"]
                    )
                    dscaper.add_event(timeline_name, _sfx_event_metadata)

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

    # Create the dry audio by taking the isolated soundscape positions
    # for speakers 1/2 and for SFX and stack them together (they are aligned)
    _dir_soundscape_positions = os.path.join(
        dscaper.get_dscaper_base_path(),
        "timelines",
        timeline_name,
        "generate",
        resp.content["id"],
        "soundscape_positions"
    )

    _audio_2_audio = None
    _audio_2_sr = None

    for _file in os.listdir(_dir_soundscape_positions):

        if not _file.endswith(".wav") or not (_file.startswith("speaker_") or _file.startswith("sfx_")):
            continue

        _path = os.path.join(_dir_soundscape_positions, _file)
        _audio, _sr = sf.read(_path)

        if _audio_2_audio is None:
            _audio_2_audio = _audio
            _audio_2_sr = _sr
        else:
            _audio_2_audio += _audio

    # Copy the audio output to the dialog audio directory
    dialog.audio_step_2_filepath = os.path.join(
        dialog.audio_dir_path,
        dialog_directory,
        "exported_audios",
        f"audio_pipeline_step2.{audio_file_format}"
    )

    # If the user want to re-sample the output audio to a different sampling rate
    if sampling_rate != _audio_2_sr:

        import librosa

        _audio_2_audio = librosa.resample(
            y=_audio_2_audio.T,
            orig_sr=_audio_2_sr,
            target_sr=sampling_rate
        ).T

    # Overwrite the audio file with the new sampling rate
    sf.write(dialog.audio_step_2_filepath, _audio_2_audio, sampling_rate)

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
