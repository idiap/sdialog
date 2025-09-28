"""
This module provides the audio pipeline for generating audio from a dialog.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import json
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf

from typing import List, Optional
from datasets import load_dataset

from sdialog import Dialog
from sdialog.audio.room import Room
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.tts_engine import KokoroTTS
from sdialog.audio.room import MicrophonePosition
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.audio_events_enricher import AudioEventsEnricher
from sdialog.audio.voice_database import BaseVoiceDatabase, DummyKokoroVoiceDatabase
from sdialog.audio import (
    generate_utterances_audios,
    save_utterances_audios,
    generate_audio_room_accoustic,
    generate_word_alignments
)


class AudioPipeline:
    """
    Audio pipeline.
    """

    def __init__(
            self,
            dir_audio: Optional[str] = "./outputs",
            tts_pipeline: Optional[BaseTTS] = None,
            voice_database: Optional[BaseVoiceDatabase] = None,
            enricher: Optional[AudioEventsEnricher] = None,
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
            self.voice_database = DummyKokoroVoiceDatabase()

        self.enricher = enricher
        self._dscaper = dscaper

        self.sampling_rate = sampling_rate  # Need to be set to the same as the TTS model

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

    def enrich(
            self,
            dialog: AudioDialog) -> AudioDialog:
        """
        Enrich with audio events, SNR and room position.
        """
        if self.enricher is None:
            raise ValueError("Enricher is not set")

        dialog = self.enricher.extract_events(dialog)

        return dialog

    def inference(
            self,
            dialog: Dialog,
            room: Optional[Room] = None,
            do_word_alignments: Optional[bool] = False,
            do_snr: Optional[bool] = False,
            do_room_position: Optional[bool] = False,
            microphone_position: Optional[MicrophonePosition] = MicrophonePosition.CEILING_CENTERED,
            do_step_1: Optional[bool] = True,
            do_step_2: Optional[bool] = True,
            do_step_3: Optional[bool] = True,
            dialog_dir_name: Optional[str] = None,
            room_name: Optional[str] = None) -> AudioDialog:
        """
        Run the audio pipeline.
        Args:
            dialog: The text dialog object.
            room: The room object.
            do_word_alignments: Whether to do word alignments between the text and the audio.
            do_snr: Whether to do dynamic SNR prediction.
            do_room_position: Whether to do room position.
            microphone_position: The microphone position in the room.
            do_step_1: Whether to do step 1 (generate the utterances audios).
            do_step_2: Whether to do step 2 (generate the timeline from the utterances audios).
            do_step_3: Whether to do step 3 (generate the room accoustic).
            dialog_dir_name: Override the name of the directory containing the dialog audios.
            room_name: Override the name of the room (only for the 3rd step).
        Returns:
            The audio enriched dialog.
        """

        # Override the dialog directory name if provided otherwise use the dialog id as the directory name
        dialog_directory = dialog_dir_name if dialog_dir_name is not None else f"dialog_{dialog.id}"

        dialog.audio_dir_path = self.dir_audio
        logging.info(f"Dialog audio dir path: {dialog.audio_dir_path}")

        dialog.audio_step_1_filepath = os.path.join(
            dialog.audio_dir_path,
            dialog_directory,
            "exported_audios",
            "audio_pipeline_step1.wav"
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
                tts_pipeline=self.tts_pipeline
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

        # else:

            # ###################
            # # TODO: Change completely this part after the refactoring of the room object serialization.
            # ###################

            # # Load combined audio from the exported_audios folder
            # dialog.set_combined_audio(
            #     sf.read(dialog.audio_step_1_filepath)[0]  # WARNING: watchout for the sampling rate
            # )

            # # Load utterances to the dialog turns
            # path_utterances = os.path.join(dialog.audio_dir_path, dialog_directory, "utterances")

            # audio_start_time = 0

            # for utterance_audio in os.listdir(path_utterances):

            #     utterance_id = utterance_audio.split("_")[0]

            #     # WARNING: watchout for the sampling rate
            #     audio_utterance_filepath = os.path.join(
            #         path_utterances,
            #         utterance_audio
            #     )

            #     # Populate the turn audio fields from the audio path
            #     _turn = dialog.turns[int(utterance_id)]
            #     _turn.audio_path = audio_utterance_filepath
            #     _audio, _sampling_rate = sf.read(audio_utterance_filepath)
            #     _turn.set_audio(_audio, _sampling_rate)
            #     _turn.audio_duration = _audio.shape[0] / _sampling_rate
            #     _turn.audio_start_time = audio_start_time
            #     audio_start_time += _turn.audio_duration

            # logging.info(f"Audio data from step 1 loaded into the dialog ({dialog.id}) successfully!")

        # TODO: Test this computation of word alignments
        if do_word_alignments:
            dialog: AudioDialog = generate_word_alignments(dialog)

        # TODO: Test this generation of SNR
        if do_snr:
            dialog: AudioDialog = self.enricher.generate_snr(dialog)

        # Generate the position of the speakers in the room
        if do_room_position:
            dialog: AudioDialog = self.enricher.generate_room_position(dialog)

        # # Randomly sample a static microphone position for the whole dialogue
        # dialog: AudioDialog = self.enricher.generate_microphone_position(dialog)

        if self._dscaper is not None and do_step_2 and len(dialog.audio_step_2_filepath) > 0:
            logging.info(f"Audio sources from dSCAPER loaded in the dialog ({dialog.id}) successfully!")

        elif self._dscaper is not None and do_step_2:

            from scaper import Dscaper

            if not isinstance(self._dscaper, Dscaper):
                raise ValueError("The dSCAPER is not a Dscaper instance")

            from sdialog.audio.audio_scaper_utils import (
                send_utterances_to_dscaper,
                generate_dscaper_timeline
            )

            # TODO: Remove the files previously generated by dSCAPER
            # if do_step_2 and file_exists(dialog.audio_step_2_filepath):
            # utterances
            # timeline
            # audio file

            # Send the utterances to dSCAPER
            dialog: AudioDialog = send_utterances_to_dscaper(dialog, self._dscaper, dialog_directory=dialog_directory)

            # Generate the timeline from dSCAPER
            logging.info(f"Generating timeline from dSCAPER for dialogue {dialog.id}")
            dialog: AudioDialog = generate_dscaper_timeline(dialog, self._dscaper, dialog_directory=dialog_directory)
            logging.info(f"Timeline generated from dSCAPER for dialogue {dialog.id}")

        elif do_step_2 and self._dscaper is None:

            logging.warning(
                "The dSCAPER is not set, which make the generation of the timeline impossible"
            )

        # Generate the audio room accoustic
        if room is not None and self._dscaper is not None and do_step_3:

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

            # Generate the audio room accoustic from the dialog and room object
            dialog: AudioDialog = generate_audio_room_accoustic(
                dialog=dialog,
                room=room,
                microphone_position=microphone_position,
                dialog_directory=dialog_directory,
                room_name=room_name
            )

            logging.info(f"Room accoustic generated for dialogue {dialog.id}!")

        elif do_step_3:
            logging.warning(
                "The room or the dSCAPER is not set, which make the generation of the room accoustic audio impossible"
            )

        # Save the audio dialog to a json file
        dialog.to_file(audio_dialog_save_path)
        logging.info(f"Audio dialog saved to the existing file ({dialog.id}) successfully at the end of the pipeline!")

        return dialog
