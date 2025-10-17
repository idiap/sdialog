import os
import shutil
import logging

import scaper  # noqa: F401
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.room import AudioSource, RoomPosition
from scaper.dscaper_datatypes import (
    DscaperAudio,
    DscaperTimeline,
    DscaperEvent,
    DscaperGenerate,
    DscaperBackground
)  # noqa: F401


def send_utterances_to_dscaper(
        dialog: AudioDialog,
        _dscaper: scaper.Dscaper,
        dialog_directory: str) -> AudioDialog:
    """
    Sends the utterances audio files to dSCAPER database.
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
                logging.error(f"Problem storing audio for turn {turn.audio_path}")
                count_audio_error += 1
        else:
            count_audio_added += 1
            turn.is_stored_in_dscaper = True

    logging.info("[dSCAPER] " + "="*30)
    logging.info("[dSCAPER] " + "# Audio sent to dSCAPER")
    logging.info("[dSCAPER] " + "="*30)
    logging.info("[dSCAPER] " + f"Already present: {count_audio_present}")
    logging.info("[dSCAPER] " + f"Correctly added: {count_audio_added}")
    logging.info("[dSCAPER] " + f"Errors: {count_audio_error}")
    logging.info("[dSCAPER] " + "="*30)

    return dialog


def generate_dscaper_timeline(
        dialog: AudioDialog,
        _dscaper: scaper.Dscaper,
        dialog_directory: str,
        sampling_rate: int = 24_000,
        background_effect: str = "white_noise",
        foreground_effect: str = "ac_noise_minimal",
        foreground_effect_position: RoomPosition = RoomPosition.TOP_RIGHT,
        audio_file_format: str = "wav"
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

    if audio_file_format not in ["mp3", "wav", "flac"]:
        raise ValueError((
            "The audio file format must be either mp3, wav or flac."
            f"You provided: {audio_file_format}"
        ))

    timeline_name = dialog_directory
    total_duration = dialog.get_combined_audio().shape[0] / sampling_rate
    dialog.total_duration = total_duration
    dialog.timeline_name = timeline_name

    # Create the timeline
    timeline_metadata = DscaperTimeline(
        name=timeline_name,
        duration=total_duration,
        description=f"Timeline for dialog {dialog.id}"
    )
    _dscaper.create_timeline(timeline_metadata)

    # Add the background to the timeline
    background_metadata = DscaperBackground(
        library="background",
        label=["const", background_effect],
        source_file=["choose", "[]"]
    )
    _dscaper.add_background(timeline_name, background_metadata)

    # Add the foreground to the timeline
    foreground_metadata = DscaperEvent(
        library="foreground",
        speaker="foreground",
        text="foreground",
        label=["const", foreground_effect],
        source_file=["choose", "[]"],
        event_time=["const", "0"],
        event_duration=["const", str(f"{total_duration:.1f}")],
        position=foreground_effect_position,
    )
    _dscaper.add_event(timeline_name, foreground_metadata)

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
        _dscaper.add_event(timeline_name, _event_metadata)
        current_time += turn.audio_duration

    # Generate the timeline
    resp = _dscaper.generate_timeline(
        timeline_name,
        DscaperGenerate(
            seed=0,
            save_isolated_positions=True,
            ref_db=-40,
            reverb=0,
            save_isolated_events=False
        ),
    )

    # Build the generate directory path
    soundscape_positions_path = os.path.join(
        _dscaper.get_dscaper_base_path(),
        "timelines",
        timeline_name,
        "generate",
        resp.content["id"],
        "soundscape_positions"
    )

    # Build the path to the audio output
    audio_output_path = os.path.join(
        _dscaper.get_dscaper_base_path(),
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
        logging.info("Successfully generated dscaper timeline.")
    else:
        logging.error(f"Failed to generate dscaper timeline for {timeline_name}: {resp.message}")

    return dialog
