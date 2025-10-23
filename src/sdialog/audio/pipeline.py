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
            do_step_1=True,
            do_step_2=True,
            do_step_3=True,
            tts_engine=KokoroTTS(),
            voice_database=HuggingfaceVoiceDatabase("sdialog/voices-kokoro"),
            room=MedicalRoomGenerator().generate(args={"room_type": RoomRole.EXAMINATION})
        )
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import librosa
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf

from datasets import load_dataset
from typing import List, Optional, Union

from sdialog import Dialog
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.tts_engine import KokoroTTS
from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole
from sdialog.audio.room import Room, RoomPosition, DirectivityType
from sdialog.audio.utils import Role, SourceType, SourceVolume, SpeakerSide
from sdialog.audio.voice_database import BaseVoiceDatabase, HuggingfaceVoiceDatabase, Voice
from sdialog.audio import (
    generate_utterances_audios,
    save_utterances_audios,
    generate_audio_room_accoustic
)


@staticmethod
def to_audio(
    dialog: Dialog,
    dir_audio: str = "./outputs_to_audio",
    dialog_dir_name: str = None,
    dscaper_data_path: Optional[str] = "./dscaper_data",
    room_name: Optional[str] = None,
    do_step_1: bool = True,
    do_step_2: bool = False,
    do_step_3: bool = False,
    tts_engine: BaseTTS = KokoroTTS(),
    voice_database: BaseVoiceDatabase = HuggingfaceVoiceDatabase("sdialog/voices-kokoro"),
    dscaper_datasets: List[str] = ["sdialog/background", "sdialog/foreground"],
    room: Room = MedicalRoomGenerator().generate(args={"room_type": RoomRole.EXAMINATION}),
    speaker_positions: dict[Role, dict] = {
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
    },
    background_effect: str = "white_noise",
    foreground_effect: str = "ac_noise_minimal",
    foreground_effect_position: RoomPosition = RoomPosition.TOP_RIGHT,
    kwargs_pyroom: dict = {
        "ray_tracing": True,
        "air_absorption": True
    },
    source_volumes: dict[SourceType, SourceVolume] = {
        SourceType.ROOM: SourceVolume.HIGH,
        SourceType.BACKGROUND: SourceVolume.VERY_LOW
    },
    audio_file_format: str = "wav",
    seed: int = None
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
    :param do_step_1: Enable text-to-speech conversion and voice assignment.
    :type do_step_1: bool
    :param do_step_2: Enable audio combination and dSCAPER timeline generation.
    :type do_step_2: bool
    :param do_step_3: Enable room acoustics simulation.
    :type do_step_3: bool
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
    :return: Audio dialogue with processed audio data.
    :rtype: AudioDialog
    """

    if audio_file_format not in ["mp3", "wav", "flac"]:
        raise ValueError(f"The audio file format must be either mp3, wav or flac. You provided: {audio_file_format}")

    if do_step_3 and not do_step_2:
        raise ValueError("The step 3 requires the step 2 to be done")
    if do_step_2 and not do_step_1:
        raise ValueError("The step 2 requires the step 1 to be done")

    if room_name is not None and not do_step_3:
        raise ValueError("The room name is only used if the step 3 is done")

    _dialog: AudioDialog = AudioDialog.from_dialog(dialog)

    os.makedirs(dir_audio, exist_ok=True)

    if do_step_2 or do_step_3:

        import scaper  # noqa: F401

        if not dscaper_data_path:
            raise ValueError("The dSCAPER data path is not provided")

        os.makedirs(dscaper_data_path, exist_ok=True)
        _dsc = scaper.Dscaper(dscaper_base_path=dscaper_data_path)

    else:
        _dsc = None

    # Initialize the audio pipeline
    _audio_pipeline = AudioPipeline(
        voice_database=voice_database,
        tts_pipeline=tts_engine,
        dscaper=_dsc,
        dir_audio=dir_audio,
    )

    if do_step_2 or do_step_3:
        _audio_pipeline.populate_dscaper(dscaper_datasets)

    if do_step_3:

        # Place the speakers around the furnitures in the room
        for _role, _kwargs in speaker_positions.items():

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
            "foreround_effect_position": foreground_effect_position,
            "source_volumes": source_volumes,
            "kwargs_pyroom": kwargs_pyroom
        }

    else:
        _environment = {}

    _dialog: AudioDialog = _audio_pipeline.inference(
        _dialog,
        environment=_environment,
        do_step_1=do_step_1,
        do_step_2=do_step_2,
        do_step_3=do_step_3,
        dialog_dir_name=dialog_dir_name,
        room_name=room_name,
        audio_file_format=audio_file_format,
        seed=seed
    )

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

      1. Step 1: Text-to-speech conversion and voice assignment
      2. Step 2: Audio combination and dSCAPER timeline generation
      3. Step 3: Room acoustics simulation and final audio processing

    :ivar dir_audio: Base directory for audio file storage.
    :vartype dir_audio: str
    :ivar tts_pipeline: Text-to-speech engine for audio generation.
    :vartype tts_pipeline: BaseTTS
    :ivar voice_database: Voice database for speaker selection.
    :vartype voice_database: BaseVoiceDatabase
    :ivar _dscaper: dSCAPER instance for audio environment simulation.
    :vartype _dscaper: Optional[Dscaper]
    :ivar sampling_rate: Audio sampling rate in Hz.
    :vartype sampling_rate: int
    """

    def __init__(
            self,
            dir_audio: Optional[str] = "./outputs",
            tts_pipeline: Optional[BaseTTS] = None,
            voice_database: Optional[BaseVoiceDatabase] = None,
            sampling_rate: Optional[int] = 24_000,
            dscaper=None):
        """
        Initialize the audio generation pipeline with configuration.

        Creates a new AudioPipeline instance with the specified configuration
        for audio processing, TTS engine, voice database, and dSCAPER integration.

        :param dir_audio: Base directory for audio file storage.
        :type dir_audio: Optional[str]
        :param tts_pipeline: Text-to-speech engine for audio generation.
        :type tts_pipeline: Optional[BaseTTS]
        :param voice_database: Voice database for speaker selection.
        :type voice_database: Optional[BaseVoiceDatabase]
        :param sampling_rate: Audio sampling rate in Hz.
        :type sampling_rate: Optional[int]
        :param dscaper: dSCAPER instance for audio environment simulation.
        :type dscaper: Optional[Dscaper]
        """

        self.dir_audio = dir_audio

        self.tts_pipeline = tts_pipeline
        if self.tts_pipeline is None:
            self.tts_pipeline = KokoroTTS()

        self.voice_database = voice_database
        if self.voice_database is None:
            self.voice_database = HuggingfaceVoiceDatabase("sdialog/voices-kokoro")

        self._dscaper = dscaper

        self.sampling_rate = sampling_rate

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

        if self._dscaper is None:
            raise ValueError("The dSCAPER is not provided to the audio pipeline")
        else:
            from scaper import Dscaper  # noqa: F401
            from scaper.dscaper_datatypes import DscaperAudio  # noqa: F401
            if not isinstance(self._dscaper, Dscaper):
                raise ValueError("The dSCAPER is not a Dscaper instance")

        count_existing_audio_files = 0
        count_error_audio_files = 0
        count_success_audio_files = 0

        # For each huggingface dataset, save the audio recordings to the dSCAPER
        for dataset_name in datasets:

            # Load huggingface dataset
            dataset = load_dataset(dataset_name, split=split)

            for data in tqdm(dataset, desc=f"Populating dSCAPER with {dataset_name} dataset..."):

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
                        logging.error(
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
        do_step_1: Optional[bool] = True,
        do_step_2: Optional[bool] = False,
        do_step_3: Optional[bool] = False,
        dialog_dir_name: Optional[str] = None,
        room_name: Optional[str] = None,
        voices: dict[Role, Union[Voice, tuple[str, str]]] = None,
        keep_duplicate: bool = True,
        audio_file_format: str = "wav",
        seed: int = None,
        re_sampling_rate: Optional[int] = None
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
        :param do_step_1: Enable text-to-speech conversion and voice assignment.
        :type do_step_1: Optional[bool]
        :param do_step_2: Enable audio combination and dSCAPER timeline generation.
        :type do_step_2: Optional[bool]
        :param do_step_3: Enable room acoustics simulation.
        :type do_step_3: Optional[bool]
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
        :return: Processed audio dialogue with all audio data.
        :rtype: AudioDialog
        """

        if audio_file_format not in ["mp3", "wav", "flac"]:
            raise ValueError((
                "The audio file format must be either mp3, wav or flac."
                f"You provided: {audio_file_format}"
            ))
        else:
            logging.info(f"[Initialization] Audio file format for generation is set to {audio_file_format}")

        # Create variables from the environment
        room: Room = environment["room"] if "room" in environment else None

        # Check if the ray tracing is enabled and the directivity is set to something else than omnidirectional
        if (
            "kwargs_pyroom" in environment
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
            logging.info(
                f"[Initialization] Dialogue ({dialog.id}) has been loaded successfully from "
                f"the existing file: {audio_dialog_save_path} !"
            )
        else:
            logging.info(
                f"[Initialization] No existing file found for the dialogue ({dialog.id}), "
                "starting from scratch..."
            )

        if not os.path.exists(dialog.audio_step_1_filepath) and do_step_1:

            logging.info(f"[Step 1] Generating audio recordings from the utterances of the dialogue: {dialog.id}")

            dialog: AudioDialog = generate_utterances_audios(
                dialog,
                voice_database=self.voice_database,
                tts_pipeline=self.tts_pipeline,
                voices=voices,
                keep_duplicate=keep_duplicate,
                seed=seed
            )

            # Save the utterances audios to the project path
            dialog: AudioDialog = save_utterances_audios(
                dialog,
                self.dir_audio,
                project_path=f"{dialog.audio_dir_path}/{dialog_directory}"
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
            logging.info(f"[Step 1] Audio files have been saved here: {dialog.audio_step_1_filepath}")

            # If the user want to re-sample the output audio to a different sampling rate
            if re_sampling_rate is not None and os.path.exists(dialog.audio_step_1_filepath):

                logging.info(f"[Step 1] Re-sampling audio to {re_sampling_rate} Hz...")

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

                logging.info(f"[Step 1] Audio has been re-sampled successfully to {re_sampling_rate} Hz!")

        # If the user want to generate the timeline from dSCAPER (whatever if the timeline is already generated or not)
        if self._dscaper is not None and do_step_2:

            from scaper import Dscaper  # noqa: F401

            if not isinstance(self._dscaper, Dscaper):
                raise ValueError("The dSCAPER is not a Dscaper instance")

            from sdialog.audio.dscaper_utils import (
                send_utterances_to_dscaper,
                generate_dscaper_timeline
            )

            logging.info("[Step 2] Sending utterances to dSCAPER...")

            # Send the utterances to dSCAPER
            dialog: AudioDialog = send_utterances_to_dscaper(dialog, self._dscaper, dialog_directory=dialog_directory)

            # Generate the timeline from dSCAPER
            logging.info("[Step 2] Generating timeline from dSCAPER...")
            dialog: AudioDialog = generate_dscaper_timeline(
                dialog=dialog,
                _dscaper=self._dscaper,
                dialog_directory=dialog_directory,
                foreground_effect=(
                    environment["foreground_effect"]
                    if "foreground_effect" in environment
                    else "ac_noise_low"
                ),
                foreground_effect_position=(
                    environment["foreground_effect_position"]
                    if "foreground_effect_position" in environment
                    else RoomPosition.TOP_RIGHT
                ),
                background_effect=(
                    environment["background_effect"]
                    if "background_effect" in environment
                    else "white_noise"
                ),
                audio_file_format=audio_file_format
            )
            logging.info("[Step 2] Has been completed!")

            # If the user want to re-sample the output audio to a different sampling rate
            if re_sampling_rate is not None and os.path.exists(dialog.audio_step_2_filepath):

                logging.info(f"[Step 2] Re-sampling audio to {re_sampling_rate} Hz...")

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

                logging.info(f"[Step 2] Audio has been re-sampled successfully to {re_sampling_rate} Hz!")

        elif do_step_2 and self._dscaper is None:

            raise ValueError(
                "The dSCAPER is not set, which makes the generation of the timeline impossible"
            )

        # Generate the audio room accoustic
        if (
            do_step_3
            and room is not None
            and self._dscaper is not None
        ):

            logging.info("[Step 3] Starting...")

            if not isinstance(environment["room"], Room):
                raise ValueError("The room must be a Room object")

            # Check if the step 2 is not done
            if not do_step_2 and len(dialog.audio_step_2_filepath) < 1:

                logging.warning((
                    "[Step 3] The timeline from dSCAPER is not generated, which"
                    "makes the generation of the room accoustic impossible"
                ))

                # Save the audio dialog to a json file
                dialog.to_file(audio_dialog_save_path)
                logging.info(f"[Step 3] Audio dialog saved to the existing file ({dialog.id}) successfully!")

                return dialog

            logging.info(f"[Step 3] Generating room accoustic for dialogue {dialog.id}")

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
                audio_file_format=audio_file_format
            )

            logging.info(f"[Step 3] Room accoustic generated for dialogue {dialog.id}!")
            logging.info("[Step 3] Done!")

            # If the user want to re-sample the output audio to a different sampling rate
            if re_sampling_rate is not None:

                for config_name, config_data in dialog.audio_step_3_filepaths.items():
                    audio_path = config_data["audio_path"]
                    if os.path.exists(audio_path):
                        logging.info(f"[Step 3] Re-sampling audio for '{config_name}' to {re_sampling_rate} Hz...")

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

                        logging.info(
                            f"[Step 3] Audio for '{config_name}' has been "
                            f"re-sampled successfully to {re_sampling_rate} Hz!"
                        )

        elif do_step_3 and (room is None or self._dscaper is None):

            raise ValueError(
                "The room or the dSCAPER is not set, which makes the generation of the room accoustic audios impossible"
            )

        # Save the audio dialog to a json file
        dialog.to_file(audio_dialog_save_path)

        return dialog
