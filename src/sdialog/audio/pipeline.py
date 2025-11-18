"""
This module provides a comprehensive audio pipeline for generating audio from dialogues.

The module includes the main audio processing pipeline that orchestrates the complete
audio generation workflow, from text-to-speech conversion to room acoustics simulation.
It provides a high-level interface for generating realistic audio dialogues with
support for multiple TTS engines, voice databases, and room acoustics simulation.

Key Features:

  - Complete audio generation pipeline from dialogue to audio
  - Multi-step audio processing workflow
  - Integration with TTS engines and voice databases
  - Room acoustics simulation support
  - Background and foreground audio mixing
  - Flexible configuration and customization

Audio Processing Pipeline:

  1. Step 1: Text-to-speech conversion and voice assignment
  2. Step 2: Audio combination and processing
  3. Step 3: Room acoustics simulation
  4. Optional: Background/foreground audio mixing with dscaper

Example:

    .. code-block:: python

        from sdialog.audio import to_audio, KokoroTTS, HuggingfaceVoiceDatabase
        from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole

        # Generate audio from dialogue
        audio_dialog = to_audio(
            dialog=dialog,
            dir_audio="./outputs",
            perform_room_acoustics=True,
            tts_engine=KokoroTTS(),
            voice_database=HuggingfaceVoiceDatabase("sdialog/voices-kokoro"),
            room=MedicalRoomGenerator().generate(args={"room_type": RoomRole.EXAMINATION})
        )
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import scaper
import librosa
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset
from typing import List, Optional, Union

from sdialog import Dialog
from sdialog.audio.utils import logger
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.processing import AudioProcessor
from sdialog.audio.tts import BaseTTS, HuggingFaceTTS
from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole
from sdialog.audio.room import Room, RoomPosition, DirectivityType
from sdialog.audio.voice_database import BaseVoiceDatabase, HuggingfaceVoiceDatabase, Voice
from sdialog.audio import generate_utterances_audios, generate_audio_room_accoustic
from sdialog.audio.impulse_response_database import ImpulseResponseDatabase, RecordingDevice
from sdialog.audio.utils import Role, SourceType, SourceVolume, SpeakerSide, default_dscaper_datasets


def to_audio(
    dialog: Dialog,
    dir_audio: Optional[str] = "./outputs_to_audio",
    dialog_dir_name: Optional[str] = None,
    dscaper_data_path: Optional[str] = "./dscaper_data",
    room_name: Optional[str] = None,
    perform_room_acoustics: Optional[bool] = False,
    tts_engine: Optional[BaseTTS] = None,
    voice_database: Optional[BaseVoiceDatabase] = None,
    dscaper_datasets: Optional[List[str]] = None,
    room: Optional[Room] = None,
    speaker_positions: Optional[dict[Role, dict]] = None,
    background_effect: Optional[str] = None,
    foreground_effect: Optional[str] = None,
    foreground_effect_position: Optional[RoomPosition] = None,
    kwargs_pyroom: Optional[dict] = None,
    source_volumes: Optional[dict[SourceType, SourceVolume]] = None,
    audio_file_format: Optional[str] = "wav",
    seed: Optional[int] = None,
    re_sampling_rate: Optional[int] = None,
    recording_devices: Optional[List[Union[RecordingDevice, str]]] = None,
    impulse_response_database: Optional[ImpulseResponseDatabase] = None,
    override_tts_audio: Optional[bool] = False,
    verbose: Optional[bool] = False
) -> AudioDialog:
    """
    Convert a dialogue into an audio dialogue with comprehensive audio processing.

    This function provides a high-level interface for converting text dialogues
    into realistic audio dialogues with support for multiple processing steps:
    text-to-speech conversion, audio combination, and room acoustics simulation.

    The function orchestrates the complete audio generation pipeline, including
    voice assignment, audio processing, and room acoustics simulation using
    the dSCAPER framework for realistic audio environments.

    :param dialog: The input dialogue to convert to audio.
    :type dialog: Dialog
    :param dir_audio: Directory path for storing audio outputs.
    :type dir_audio: str
    :param dialog_dir_name: Custom name for the dialogue directory.
    :type dialog_dir_name: str
    :param dscaper_data_path: Path to dSCAPER data directory.
    :type dscaper_data_path: Optional[str]
    :param room_name: Custom name for the room configuration.
    :type room_name: Optional[str]
    :param perform_room_acoustics: Enable dSCAPER timeline generation and room acoustics simulation.
    :type perform_room_acoustics: bool
    :param tts_engine: Text-to-speech engine for audio generation.
    :type tts_engine: BaseTTS
    :param voice_database: Voice database for speaker selection.
    :type voice_database: BaseVoiceDatabase
    :param dscaper_datasets: List of Hugging Face datasets for dSCAPER.
    :type dscaper_datasets: List[str]
    :param room: Room configuration for acoustics simulation.
    :type room: Room
    :param speaker_positions: Speaker positioning configuration.
    :type speaker_positions: dict[Role, dict]
    :param background_effect: Background audio effect type.
    :type background_effect: str
    :param foreground_effect: Foreground audio effect type.
    :type foreground_effect: str
    :param foreground_effect_position: Position for foreground effects.
    :type foreground_effect_position: RoomPosition
    :param kwargs_pyroom: PyRoomAcoustics configuration parameters.
    :type kwargs_pyroom: dict
    :param source_volumes: Volume levels for different audio sources.
    :type source_volumes: dict[SourceType, SourceVolume]
    :param audio_file_format: Audio file format (wav, mp3, flac).
    :type audio_file_format: str
    :param seed: Seed for random number generator.
    :type seed: int
    :param re_sampling_rate: Re-sampling rate for the output audio.
    :type re_sampling_rate: Optional[int]
    :param recording_devices: The identifiers of the recording devices to simulate.
    :type recording_devices: Optional[List[Union[RecordingDevice, str]]]
    :param impulse_response_database: The database for impulse responses.
    :type impulse_response_database: Optional[ImpulseResponseDatabase]
    :param override_tts_audio: Override the TTS audio if it already exists.
    :type override_tts_audio: Optional[bool]
    :param verbose: Verbose mode for logging.
    :type verbose: Optional[bool]
    :return: Audio dialogue with processed audio data.
    :rtype: AudioDialog
    """

    # Save the original logger level
    original_level = logger.level
    if not verbose:
        logger.setLevel(logging.ERROR)

    # Reset the logger level to the original level after the function is executed
    try:

        if room is None:
            room = MedicalRoomGenerator().generate(args={"room_type": RoomRole.EXAMINATION})

        if speaker_positions is None:
            speaker_positions = {
                Role.SPEAKER_1: {
                    "furniture_name": "center",
                    "max_distance": 1.0,
                    "side": SpeakerSide.FRONT
                },
                Role.SPEAKER_2: {
                    "furniture_name": "center",
                    "max_distance": 1.0,
                    "side": SpeakerSide.BACK
                }
            }

        if audio_file_format not in ["mp3", "wav", "flac"]:
            raise ValueError(
                f"The audio file format must be either mp3, wav or flac. You provided: {audio_file_format}"
            )

        if room_name is not None and not perform_room_acoustics:
            raise ValueError("The room name is only used if the step 3 is done")

        # Build the path to save the audio dialog
        if dialog_dir_name is not None and dir_audio is not None:
            audio_dialog_save_path = os.path.join(
                dir_audio,
                dialog_dir_name,
                "exported_audios",
                "audio_dialog.json"
            )
        else:
            audio_dialog_save_path = None

        # Load the audio dialog from the existing file if it exists
        if audio_dialog_save_path is not None and os.path.exists(audio_dialog_save_path):
            _dialog: AudioDialog = AudioDialog.from_file(audio_dialog_save_path)
        else:
            _dialog: AudioDialog = AudioDialog.from_dialog(dialog)

        os.makedirs(dir_audio, exist_ok=True)

        # Initialize the audio pipeline
        _audio_pipeline = AudioPipeline(
            voice_database=(
                voice_database
                if voice_database is not None
                else HuggingfaceVoiceDatabase("sdialog/voices-kokoro")
            ),
            tts_engine=(
                tts_engine
                if tts_engine is not None
                else HuggingFaceTTS("facebook/mms-tts-eng")
            ),
            dscaper_data_path=dscaper_data_path,
            dir_audio=dir_audio,
            impulse_response_database=impulse_response_database
        )

        if (
            perform_room_acoustics
            and dscaper_datasets is not None
            and dscaper_datasets != default_dscaper_datasets()
        ):
            _audio_pipeline.populate_dscaper(dscaper_datasets)

        if perform_room_acoustics:

            # Place the speakers around the furnitures in the room
            for _role, _kwargs in speaker_positions.items():

                if _role in room.speakers_positions:
                    continue

                room.place_speaker_around_furniture(
                    speaker_name=_role,
                    furniture_name=_kwargs["furniture_name"],
                    max_distance=_kwargs["max_distance"],
                    side=_kwargs["side"]
                )

            _environment = {
                "room": room,
                "background_effect": background_effect,
                "foreground_effect": foreground_effect,
                "foreground_effect_position": foreground_effect_position,
                "source_volumes": source_volumes,
                "kwargs_pyroom": kwargs_pyroom
            }

        else:
            _environment = {}

        _dialog: AudioDialog = _audio_pipeline.inference(
            _dialog,
            environment=_environment,
            perform_room_acoustics=perform_room_acoustics,
            dialog_dir_name=dialog_dir_name,
            room_name=room_name,
            audio_file_format=audio_file_format,
            seed=seed,
            re_sampling_rate=re_sampling_rate,
            recording_devices=recording_devices,
            override_tts_audio=override_tts_audio,
            verbose=verbose
        )

    finally:
        # Reset the logger level to the original level
        logger.setLevel(original_level)

    return _dialog


class AudioPipeline:
    """
    Comprehensive audio generation pipeline for dialogue processing.

    AudioPipeline orchestrates the complete audio generation workflow from text
    dialogues to realistic audio dialogues with room acoustics simulation. It
    provides a flexible framework for multi-step audio processing including
    text-to-speech conversion, audio combination, and room acoustics simulation.

    Key Features:

      - Multi-step audio processing pipeline (TTS, combination, acoustics)
      - Integration with TTS engines and voice databases
      - Room acoustics simulation using pyroomacoustics
      - dSCAPER integration for realistic audio environments
      - Flexible configuration and customization options
      - Support for multiple audio file formats

    Pipeline Steps:

      1. perform_tts: Text-to-speech conversion and voice assignment and audio combination
      2. perform_room_acoustics: dSCAPER timeline generation and room acoustics simulation

    :ivar dir_audio: Base directory for audio file storage.
    :vartype dir_audio: str
    :ivar tts_engine: Text-to-speech engine for audio generation.
    :vartype tts_engine: BaseTTS
    :ivar voice_database: Voice database for speaker selection.
    :vartype voice_database: BaseVoiceDatabase
    :ivar _dscaper: dSCAPER instance for audio environment simulation.
    :vartype _dscaper: Optional[Dscaper]
    :ivar sampling_rate: Audio sampling rate in Hz.
    :vartype sampling_rate: int
    :ivar impulse_response_database: The database for impulse responses.
    :vartype impulse_response_database: Optional[ImpulseResponseDatabase]
    """

    def __init__(
        self,
        dir_audio: Optional[str] = "./outputs",
        tts_engine: Optional[BaseTTS] = None,
        voice_database: Optional[BaseVoiceDatabase] = None,
        sampling_rate: Optional[int] = 24_000,
        dscaper_data_path: Optional[str] = "./dscaper_data",
        impulse_response_database: Optional[ImpulseResponseDatabase] = None,
    ):
        """
        Initialize the audio generation pipeline with configuration.

        Creates a new AudioPipeline instance with the specified configuration
        for audio processing, TTS engine, voice database, and dSCAPER integration.

        :param dir_audio: Base directory for audio file storage.
        :type dir_audio: Optional[str]
        :param tts_engine: Text-to-speech engine for audio generation.
        :type tts_engine: Optional[BaseTTS]
        :param voice_database: Voice database for speaker selection.
        :type voice_database: Optional[BaseVoiceDatabase]
        :param sampling_rate: Audio sampling rate in Hz.
        :type sampling_rate: Optional[int]
        :param dscaper_data_path: Path to dSCAPER data directory.
        :type dscaper_data_path: Optional[str]
        :param impulse_response_database: The database for impulse responses.
        :type impulse_response_database: Optional[ImpulseResponseDatabase]
        """

        self.dir_audio = dir_audio

        self.tts_engine = tts_engine
        if self.tts_engine is None:

            logger.warning((
                "No TTS provided, using the default Hugging Face TTS model: "
                "HuggingFaceTTS(\"facebook/mms-tts-eng\")"
            ))
            self.tts_engine = HuggingFaceTTS("facebook/mms-tts-eng")

        self.voice_database = voice_database
        if self.voice_database is None:
            self.voice_database = HuggingfaceVoiceDatabase("sdialog/voices-kokoro")

        self.dscaper_data_path = (
            dscaper_data_path
            if dscaper_data_path is not None or dscaper_data_path != ""
            else "./dscaper_data"
        )
        os.makedirs(self.dscaper_data_path, exist_ok=True)

        self._dscaper = scaper.Dscaper(dscaper_base_path=self.dscaper_data_path)

        self.sampling_rate = sampling_rate

        self.impulse_response_database = impulse_response_database

        # Populate the dSCAPER with the default datasets
        self.populate_dscaper(default_dscaper_datasets())

    def populate_dscaper(
            self,
            datasets: List[str],
            split: str = "train") -> dict:
        """
        Populate the dSCAPER with audio recordings from Hugging Face datasets.

        Downloads and stores audio recordings from specified Hugging Face datasets
        into the dSCAPER library for use in audio environment simulation. This
        method processes each dataset and stores the audio files with appropriate
        metadata for later use in timeline generation.

        :param datasets: List of Hugging Face dataset names to populate.
        :type datasets: List[str]
        :param split: Dataset split to use (train, validation, test).
        :type split: str
        :return: Dictionary with statistics about the population process.
        :rtype: dict
        """
        from scaper.dscaper_datatypes import DscaperAudio  # noqa: F401

        count_existing_audio_files = 0
        count_error_audio_files = 0
        count_success_audio_files = 0

        # For each huggingface dataset, save the audio recordings to the dSCAPER
        for dataset_name in datasets:

            # Load huggingface dataset
            dataset = load_dataset(dataset_name, split=split)

            for data in tqdm(
                dataset,
                desc=f"Populating dSCAPER with {dataset_name} dataset...",
                leave=False
            ):

                filename = data["audio"]["path"].split("/")[-1]
                label_str = dataset.features["label"].names[data["label"]]

                # WARNING: Create a name for the "library" based
                # on the dataset name minus the organization name
                metadata = DscaperAudio(
                    library=dataset_name.split("/")[-1],
                    label=label_str,
                    filename=filename
                )

                # Try to store the audio using the dSCAPER API
                resp = self._dscaper.store_audio(data["audio"]["path"], metadata)

                # If an error occurs
                if resp.status != "success":

                    # Check if the audio is already stored in the library
                    if resp.content["description"] == "File already exists. Use PUT to update it.":
                        count_existing_audio_files += 1
                    else:
                        logger.error(
                            f"Problem storing audio {data['audio']['path']}: {resp.content['description']}"
                        )
                        count_error_audio_files += 1
                else:
                    count_success_audio_files += 1

        return {
            "count_existing_audio_files": count_existing_audio_files,
            "count_error_audio_files": count_error_audio_files,
            "count_success_audio_files": count_success_audio_files
        }

    def master_audio(
            self,
            dialog: AudioDialog) -> np.ndarray:
        """
        Combine multiple audio segments into a single master audio track.

        Concatenates all audio segments from the dialogue turns into a single
        continuous audio track. This creates a baseline audio representation
        of the entire dialogue for further processing and analysis.

        :param dialog: Audio dialogue containing turns with audio data.
        :type dialog: AudioDialog
        :return: Combined audio data as numpy array.
        :rtype: np.ndarray
        """
        return np.concatenate([turn.get_audio() for turn in dialog.turns])

    def inference(
        self,
        dialog: Dialog,
        environment: dict = {},
        perform_tts: Optional[bool] = True,
        perform_room_acoustics: Optional[bool] = False,
        dialog_dir_name: Optional[str] = None,
        room_name: Optional[str] = None,
        voices: dict[Role, Union[Voice, tuple[str, str]]] = None,
        keep_duplicate: bool = True,
        audio_file_format: str = "wav",
        seed: int = None,
        re_sampling_rate: Optional[int] = None,
        recording_devices: Optional[List[Union[RecordingDevice, str]]] = None,
        tts_pipeline_kwargs: Optional[dict] = {},
        override_tts_audio: Optional[bool] = False,
        verbose: Optional[bool] = False
    ) -> AudioDialog:
        """
        Execute the complete audio generation pipeline.

        Runs the multi-step audio generation pipeline with configurable steps:
        text-to-speech conversion, audio combination, and room acoustics simulation.
        The method handles the complete workflow from text dialogue to realistic
        audio dialogue with room acoustics simulation.

        :param dialog: Input dialogue to process.
        :type dialog: Dialog
        :param environment: Environment configuration for room acoustics.
        :type environment: dict
        :param perform_tts: Convert the dialog into audio using the text-to-speech engine.
        :type perform_tts: Optional[bool]
        :param perform_room_acoustics: Enable dSCAPER timeline generation and room acoustics simulation.
        :type perform_room_acoustics: Optional[bool]
        :param dialog_dir_name: Custom name for the dialogue directory.
        :type dialog_dir_name: Optional[str]
        :param room_name: Custom name for the room configuration.
        :type room_name: Optional[str]
        :param voices: Voice assignments for different speaker roles.
        :type voices: dict[Role, Union[Voice, tuple[str, str]]]
        :param keep_duplicate: Allow duplicate voice assignments.
        :type keep_duplicate: bool
        :param audio_file_format: Audio file format (wav, mp3, flac).
        :type audio_file_format: str
        :param seed: Seed for random number generator.
        :type seed: int
        :param re_sampling_rate: Re-sampling rate for the output audio.
        :type re_sampling_rate: Optional[int]
        :param recording_devices: The identifiers of the recording devices to simulate.
        :type recording_devices: Optional[List[Union[RecordingDevice, str]]]
        :param tts_pipeline_kwargs: Additional keyword arguments to be passed to the TTS pipeline.
        :type tts_pipeline_kwargs: Optional[dict]
        :param override_tts_audio: Override the TTS audio if it already exists.
        :type override_tts_audio: Optional[bool]
        :param verbose: Verbose mode for logging.
        :type verbose: Optional[bool]
        :return: Processed audio dialogue with all audio data.
        :rtype: AudioDialog

        .. note::
            Microphone simulation via `recording_devices` requires the `impulse_response_database`
            to be set on the `AudioPipeline` instance.
        """

        # Save the original logger level
        original_level = logger.level
        if not verbose:
            logger.setLevel(logging.ERROR)

        # Reset the logger level to the original level after the function is executed
        try:

            if self.impulse_response_database is None and recording_devices is not None:
                logger.warning(
                    "[Initialization] The impulse response database is not set, "
                    "using the default Hugging Face database for microphone simulation..."
                )
                from sdialog.audio.impulse_response_database import HuggingFaceImpulseResponseDatabase
                self.impulse_response_database = HuggingFaceImpulseResponseDatabase("sdialog/impulse-responses")

            if audio_file_format not in ["mp3", "wav", "flac"]:
                raise ValueError((
                    "The audio file format must be either mp3, wav or flac."
                    f"You provided: {audio_file_format}"
                ))
            else:
                logger.info(f"[Initialization] Audio file format for generation is set to {audio_file_format}")

            # Create variables from room from the environment
            room: Room = (
                environment["room"]
                if environment is not None
                and "room" in environment
                else MedicalRoomGenerator().generate(args={"room_type": RoomRole.CONSULTATION})
            )

            # Check if the ray tracing is enabled and the directivity is set to something else than omnidirectional
            if (
                environment is not None
                and "kwargs_pyroom" in environment
                and environment["kwargs_pyroom"] is not None
                and "ray_tracing" in environment["kwargs_pyroom"]
                and environment["kwargs_pyroom"]["ray_tracing"]
                and room.directivity_type is not None
                and room.directivity_type != DirectivityType.OMNIDIRECTIONAL
            ):
                raise ValueError((
                    "The ray tracing is enabled with a non-omnidirectional directivity, "
                    "which make the generation of the room accoustic audio impossible.\n"
                    "The microphone directivity must be set to omnidirectional "
                    "(pyroomacoustics only supports omnidirectional directivity for ray tracing)."
                ))

            # Override the dialog directory name if provided otherwise use the dialog id as the directory name
            dialog_directory = dialog_dir_name if dialog_dir_name is not None else f"dialog_{dialog.id}"
            dialog.audio_dir_path = self.dir_audio

            dialog.audio_step_1_filepath = os.path.join(
                dialog.audio_dir_path,
                dialog_directory,
                "exported_audios",
                f"audio_pipeline_step1.{audio_file_format}"
            )

            # Path to save the audio dialog
            audio_dialog_save_path = os.path.join(
                dialog.audio_dir_path,
                dialog_directory,
                "exported_audios",
                "audio_dialog.json"
            )

            # Load the audio dialog from the existing file
            if os.path.exists(audio_dialog_save_path):
                dialog = AudioDialog.from_file(audio_dialog_save_path)
                logger.info(
                    f"[Initialization] Dialogue ({dialog.id}) has been loaded successfully from "
                    f"the existing file: {audio_dialog_save_path} !"
                )
            else:
                logger.info(
                    f"[Initialization] No existing file found for the dialogue ({dialog.id}), "
                    "starting from scratch..."
                )

            if (not os.path.exists(dialog.audio_step_1_filepath) or override_tts_audio) and perform_tts:

                logger.info(f"[Step 1] Generating audio recordings from the utterances of the dialogue: {dialog.id}")

                dialog: AudioDialog = generate_utterances_audios(
                    dialog,
                    voice_database=self.voice_database,
                    tts_pipeline=self.tts_engine,
                    voices=voices,
                    keep_duplicate=keep_duplicate,
                    seed=seed,
                    sampling_rate=self.sampling_rate,
                    tts_pipeline_kwargs=tts_pipeline_kwargs,
                )

                # Save the utterances audios to the project path
                dialog.save_utterances_audios(
                    dir_audio=self.dir_audio,
                    project_path=os.path.join(dialog.audio_dir_path, dialog_directory)
                )

                # Combine the audio segments into a single master audio track as a baseline
                dialog.set_combined_audio(
                    self.master_audio(dialog)
                )

                # Save the combined audio to exported_audios folder
                sf.write(
                    dialog.audio_step_1_filepath,
                    dialog.get_combined_audio(),
                    self.sampling_rate
                )
                logger.info(f"[Step 1] Audio files have been saved here: {dialog.audio_step_1_filepath}")

                # If the user want to re-sample the output audio to a different sampling rate
                if re_sampling_rate is not None and os.path.exists(dialog.audio_step_1_filepath):

                    logger.info(f"[Step 1] Re-sampling audio to {re_sampling_rate} Hz...")

                    y_resampled = librosa.resample(
                        y=dialog.get_combined_audio().T,
                        orig_sr=self.sampling_rate,
                        target_sr=re_sampling_rate
                    )

                    # Overwrite the audio file with the new sampling rate
                    sf.write(
                        dialog.audio_step_1_filepath,
                        y_resampled,
                        re_sampling_rate
                    )

                    logger.info(f"[Step 1] Audio has been re-sampled successfully to {re_sampling_rate} Hz!")

            if perform_room_acoustics and not perform_tts and not os.path.exists(dialog.audio_step_1_filepath):
                raise ValueError(
                    "Room acoustics cannot be performed without TTS unless the audio from step 1 is already available. "
                    "Please run with perform_tts=True first, or provide a dialog with existing audio paths."
                )

            # If the user want to generate the timeline from dSCAPER
            if perform_room_acoustics:

                from sdialog.audio.dscaper_utils import (
                    send_utterances_to_dscaper,
                    generate_dscaper_timeline
                )

                logger.info("[Step 2] Sending utterances to dSCAPER...")

                # Send the utterances to dSCAPER
                dialog: AudioDialog = send_utterances_to_dscaper(
                    dialog,
                    self._dscaper,
                    dialog_directory=dialog_directory
                )

                # Generate the timeline from dSCAPER
                logger.info("[Step 2] Generating timeline from dSCAPER...")
                dialog: AudioDialog = generate_dscaper_timeline(
                    dialog=dialog,
                    dscaper=self._dscaper,
                    dialog_directory=dialog_directory,
                    foreground_effect=environment.get("foreground_effect"),
                    foreground_effect_position=environment.get("foreground_effect_position"),
                    background_effect=environment.get("background_effect") or "white_noise",
                    audio_file_format=audio_file_format
                )
                logger.info("[Step 2] Has been completed!")

                # If the user want to re-sample the output audio to a different sampling rate
                if re_sampling_rate is not None and os.path.exists(dialog.audio_step_2_filepath):

                    logger.info(f"[Step 2] Re-sampling audio to {re_sampling_rate} Hz...")

                    y, sr = librosa.load(dialog.audio_step_2_filepath, sr=None)

                    y_resampled = librosa.resample(
                        y=y,
                        orig_sr=sr,
                        target_sr=re_sampling_rate
                    )

                    # Overwrite the audio file with the new sampling rate
                    sf.write(
                        dialog.audio_step_2_filepath,
                        y_resampled,
                        re_sampling_rate
                    )

                    logger.info(f"[Step 2] Audio has been re-sampled successfully to {re_sampling_rate} Hz!")

            # Generate the audio room accoustic
            if (
                perform_room_acoustics
                and room is not None
            ):

                logger.info("[Step 3] Starting...")

                if not isinstance(environment["room"], Room):
                    raise ValueError("The room must be a Room object")

                # Check if the step 2 is not done
                if len(dialog.audio_step_2_filepath) < 1:

                    logger.warning((
                        "[Step 3] The timeline from dSCAPER is not generated, which"
                        "makes the generation of the room accoustic impossible"
                    ))

                    # Save the audio dialog to a json file
                    dialog.to_file(audio_dialog_save_path)
                    logger.info(f"[Step 3] Audio dialog saved to the existing file ({dialog.id}) successfully!")

                    return dialog

                logger.info(f"[Step 3] Generating room accoustic for dialogue {dialog.id}")

                # Override the room name if provided otherwise use the hash of the room
                room_name = room_name if room_name is not None else room.name

                # Generate the audio room accoustic from the dialog and room object
                dialog: AudioDialog = generate_audio_room_accoustic(
                    dialog=dialog,
                    room=room,
                    dialog_directory=dialog_directory,
                    room_name=room_name,
                    kwargs_pyroom=environment["kwargs_pyroom"] if "kwargs_pyroom" in environment else {},
                    source_volumes=environment["source_volumes"] if "source_volumes" in environment else {},
                    audio_file_format=audio_file_format,
                    background_effect=(
                        environment["background_effect"]
                        if "background_effect" in environment
                        else "white_noise"
                    ),
                    foreground_effect=(
                        environment["foreground_effect"]
                        if "foreground_effect" in environment
                        else "ac_noise_minimal"
                    ),
                    foreground_effect_position=(
                        environment["foreground_effect_position"]
                        if "foreground_effect_position" in environment
                        else RoomPosition.TOP_RIGHT
                    )
                )

                logger.info(f"[Step 3] Room accoustic has been generated successfully for dialogue {dialog.id}!")

                # If the user want to re-sample the output audio to a different sampling rate
                if re_sampling_rate is not None:

                    for config_name, config_data in dialog.audio_step_3_filepaths.items():
                        audio_path = config_data["audio_path"]
                        if os.path.exists(audio_path):
                            logger.info(f"[Step 3] Re-sampling audio for '{config_name}' to {re_sampling_rate} Hz...")

                            y, sr = librosa.load(audio_path, sr=None)

                            y_resampled = librosa.resample(
                                y=y,
                                orig_sr=sr,
                                target_sr=re_sampling_rate
                            )

                            # Overwrite the audio file with the new sampling rate
                            sf.write(
                                audio_path,
                                y_resampled,
                                re_sampling_rate
                            )

                            logger.info(
                                f"[Step 3] Audio for '{config_name}' has been "
                                f"re-sampled successfully to {re_sampling_rate} Hz!"
                            )

            elif perform_room_acoustics and room is None:

                raise ValueError(
                    "The room is not set, which makes the generation of the room accoustic audios impossible"
                )

            # Apply microphone effect if a recording device is specified
            if recording_devices is not None and perform_room_acoustics:

                if self.impulse_response_database is None:
                    raise ValueError(
                        "The impulse response database is not set, simulation of the microphone is impossible"
                    )

                logger.info(f"[Post-Processing] Applying microphone effect for devices: {recording_devices}")

                if not dialog.audio_step_3_filepaths or len(dialog.audio_step_3_filepaths) == 0:
                    raise ValueError("[Post-Processing] No room acoustics audio found to apply post-processing on.")

                for _room_name, room_data in list(dialog.audio_step_3_filepaths.items()):

                    # Process only the room with the same name as the one specified
                    if room_name is not None and room_name != _room_name:
                        continue

                    input_audio_path = room_data["audio_path"]

                    # Check if the input audio (step 3) path exists
                    if not os.path.exists(input_audio_path):
                        raise ValueError(f"[Post-Processing] Input audio path not found: {input_audio_path}")

                    # If the audio paths post processing are not in the room data, create a new dictionary
                    if "audio_paths_post_processing" not in room_data:
                        room_data["audio_paths_post_processing"] = {}

                    # For each recording device, apply the microphone effect
                    for recording_device in recording_devices:

                        if str(recording_device) in room_data["audio_paths_post_processing"]:
                            logger.warning(
                                f"[Post-Processing] Microphone effect already applied for device: {recording_device} "
                                f" and room configuration: {_room_name}. Skipping..."
                            )
                            continue

                        output_audio_name = (
                            f"audio_post_processing-{_room_name}-"
                            f"{str(recording_device)}"
                            f".{audio_file_format}"
                        )

                        # Build the path to save the output audio
                        output_audio_path = os.path.join(
                            dialog.audio_dir_path,
                            dialog_directory,
                            "exported_audios",
                            "post_processing",
                            output_audio_name
                        )

                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

                        AudioProcessor.apply_microphone_effect(
                            input_audio_path=input_audio_path,
                            output_audio_path=output_audio_path,
                            device=recording_device,
                            impulse_response_database=self.impulse_response_database
                        )

                        room_data["audio_paths_post_processing"][str(recording_device)] = output_audio_path

                        logger.info(
                            f"[Post-Processing] Microphone effect applied for device: {recording_device}. "
                            f"Output saved to: {output_audio_path}"
                        )

            # Save the audio dialog to a json file
            dialog.to_file(audio_dialog_save_path)

        finally:
            # Reset the logger level to the original level
            logger.setLevel(original_level)

        return dialog
