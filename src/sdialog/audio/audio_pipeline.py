"""
This module provides the audio pipeline for generating audio from a dialog.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf

from datasets import load_dataset
from typing import List, Optional, Union

from sdialog import Dialog
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.tts_engine import KokoroTTS
from sdialog.audio.room import Room, RoomPosition
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole
from sdialog.audio.audio_utils import Role, SourceType, SourceVolume, SpeakerSide
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
    audio_file_format: str = "wav"
) -> AudioDialog:
    """
    Convert a dialog into an audio dialog.
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

        import scaper

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
        audio_file_format=audio_file_format
    )

    return _dialog


class AudioPipeline:
    """
    Audio pipeline.
    """

    def __init__(
            self,
            dir_audio: Optional[str] = "./outputs",
            tts_pipeline: Optional[BaseTTS] = None,
            voice_database: Optional[BaseVoiceDatabase] = None,
            sampling_rate: Optional[int] = 24_000,
            dscaper=None):
        """
        Initialize the audio pipeline.
        """

        self.dir_audio = dir_audio

        self.tts_pipeline = tts_pipeline
        if self.tts_pipeline is None:
            self.tts_pipeline = KokoroTTS()

        self.voice_database = voice_database
        if self.voice_database is None:
            self.voice_database = HuggingfaceVoiceDatabase("sdialog/voices-kokoro")

        self._dscaper = dscaper

        self.sampling_rate = sampling_rate  # TODO: Need to be set to the same as the TTS model

    def populate_dscaper(
            self,
            datasets: List[str],
            split: str = "train") -> int:
        """
        Populate the dSCAPER with the audio recordings.
        """

        if self._dscaper is None:
            raise ValueError("The dSCAPER is not provided to the audio pipeline")
        else:
            from scaper import Dscaper
            from scaper.dscaper_datatypes import DscaperAudio
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
        Combines multiple audio segments into a single master audio track.
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
        audio_file_format: str = "wav"
    ) -> AudioDialog:
        """
        Run the audio pipeline.
        """

        if audio_file_format not in ["mp3", "wav", "flac"]:
            raise ValueError((
                "The audio file format must be either mp3, wav or flac."
                f"You provided: {audio_file_format}"
            ))

        # Create variables from the environment
        room = environment["room"] if "room" in environment else None

        # Override the dialog directory name if provided otherwise use the dialog id as the directory name
        dialog_directory = dialog_dir_name if dialog_dir_name is not None else f"dialog_{dialog.id}"

        dialog.audio_dir_path = self.dir_audio
        logging.info(f"Dialog audio dir path: {dialog.audio_dir_path}")

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

        if os.path.exists(audio_dialog_save_path):
            # Load the audio dialog from the existing file
            dialog = AudioDialog.from_file(audio_dialog_save_path)
            logging.info(f"Audio dialog loaded from the existing file ({dialog.id}) successfully!")

        if not os.path.exists(dialog.audio_step_1_filepath) and do_step_1:

            logging.info(f"Generating utterances audios from dialogue {dialog.id}")

            dialog: AudioDialog = generate_utterances_audios(
                dialog,
                voice_database=self.voice_database,
                tts_pipeline=self.tts_pipeline,
                voices=voices,
                keep_duplicate=keep_duplicate
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
            logging.info(f"Step 1 audio saved to {dialog.audio_step_1_filepath}")

        # If the user want to generate the timeline from dSCAPER (whatever if the timeline is already generated or not)
        if self._dscaper is not None and do_step_2:

            logging.info("Starting step 2...")

            from scaper import Dscaper

            if not isinstance(self._dscaper, Dscaper):
                raise ValueError("The dSCAPER is not a Dscaper instance")

            from sdialog.audio.audio_scaper_utils import (
                send_utterances_to_dscaper,
                generate_dscaper_timeline
            )

            # Send the utterances to dSCAPER
            dialog: AudioDialog = send_utterances_to_dscaper(dialog, self._dscaper, dialog_directory=dialog_directory)

            # Generate the timeline from dSCAPER
            logging.info(f"Generating timeline from dSCAPER for dialogue {dialog.id}")
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
            logging.info(f"Timeline generated from dSCAPER for dialogue {dialog.id}")
            logging.info("Step 2 done!")

        elif do_step_2 and self._dscaper is None:

            logging.warning(
                "The dSCAPER is not set, which make the generation of the timeline impossible"
            )

        # Generate the audio room accoustic
        if room is not None and self._dscaper is not None and do_step_3:

            logging.info("Starting step 3...")

            if room is None:
                raise ValueError("The room is not set, which make the generation of the room accoustic impossible")

            if not isinstance(environment["room"], Room):
                raise ValueError("The room must be a Room object")

            # Check if the step 2 is not done
            if not do_step_2 and len(dialog.audio_step_2_filepath) < 1:

                logging.warning((
                    "The timeline from dSCAPER is not generated, which"
                    "make the generation of the room accoustic impossible"
                ))

                # Save the audio dialog to a json file
                dialog.to_file(audio_dialog_save_path)
                logging.info(f"Audio dialog saved to the existing file ({dialog.id}) successfully!")

                return dialog

            logging.info(f"Generating room accoustic for dialogue {dialog.id}")

            # Override the room name if provided otherwise use the hash of the room
            room_name = room_name if room_name is not None else room.name

            # TODO: Remove this after testing
            logging.info("Internal audio sources length:")
            logging.info(str(len(dialog.get_audio_sources())))
            logging.info("-"*25)

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

            logging.info(f"Room accoustic generated for dialogue {dialog.id}!")
            logging.info("Step 3 done!")

        elif do_step_3:
            logging.warning(
                "The room or the dSCAPER is not set, which make the generation of the room accoustic audio impossible"
            )

        # Save the audio dialog to a json file
        dialog.to_file(audio_dialog_save_path)
        logging.info(f"Audio dialog saved to the existing file ({dialog.id}) successfully at the end of the pipeline!")

        return dialog
