import os
import json
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf

from typing import List, Optional
from datasets import load_dataset, load_from_disk

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

        n_audio_files = 0

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

                resp = self._dscaper.store_audio(data["audio"]["path"], metadata)

                if resp.status != "success":
                    logging.error(
                        f"Problem storing audio {data['audio']['path']} (the audio can also be already stored)"
                    )
                else:
                    n_audio_files += 1

        return n_audio_files

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
            microphone_position: Optional[MicrophonePosition] = MicrophonePosition.CEILING_CENTERED) -> AudioDialog:
        """
        Run the audio pipeline.
        """

        if room is not None:
            dialog.set_room(room)

        dialog.audio_dir_path = self.dir_audio
        dialog.audio_step_1_filepath = os.path.join(
            dialog.audio_dir_path,
            f"dialog_{dialog.id}",
            "exported_audios",
            "audio_pipeline_step1.wav"
        )
        print(dialog.audio_step_1_filepath)

        if not os.path.exists(dialog.audio_step_1_filepath):

            print(f"Generating utterances audios from dialogue {dialog.id}")

            dialog: AudioDialog = generate_utterances_audios(
                dialog,
                voice_database=self.voice_database,
                tts_pipeline=self.tts_pipeline
            )

            dialog: AudioDialog = save_utterances_audios(dialog, self.dir_audio)

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

        else:

            print(f"Loading utterances and combined audio from dialogue {dialog.id}")

            # Load combined audio from the exported_audios folder
            dialog.set_combined_audio(
                sf.read(dialog.audio_step_1_filepath)[0]  # WARNING: watchout for the sampling rate
            )

            # Load utterances to the dialog turns
            path_utterances = os.path.join(dialog.audio_dir_path, f"dialog_{dialog.id}", "utterances")

            audio_start_time = 0

            for utterance_audio in os.listdir(path_utterances):

                utterance_id = utterance_audio.split("_")[0]

                # WARNING: watchout for the sampling rate
                audio_utterance_filepath = os.path.join(
                    path_utterances,
                    utterance_audio
                )

                # Populate the turn audio fields from the audio path
                _turn = dialog.turns[int(utterance_id)]
                _turn.audio_path = audio_utterance_filepath
                _audio, _sampling_rate = sf.read(audio_utterance_filepath)
                _turn.set_audio(_audio, _sampling_rate)
                _turn.audio_duration = _audio.shape[0] / _sampling_rate
                _turn.audio_start_time = audio_start_time
                audio_start_time += _turn.audio_duration

            print(f"Audio data from step 1 loaded into the dialog ({dialog.id}) successfully!")

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

        if self._dscaper is not None:

            from scaper import Dscaper

            if not isinstance(self._dscaper, Dscaper):
                raise ValueError("The dSCAPER is not a Dscaper instance")

            from sdialog.audio.audio_scaper_utils import (
                send_utterances_to_dscaper,
                generate_dscaper_timeline
            )

            # Send the utterances to dSCAPER
            print(f"Sending utterances to dSCAPER for dialogue {dialog.id}")
            dialog: AudioDialog = send_utterances_to_dscaper(dialog, self._dscaper)

            # Generate the timeline from dSCAPER
            print(f"Generating timeline from dSCAPER for dialogue {dialog.id}")
            dialog: AudioDialog = generate_dscaper_timeline(dialog, self._dscaper)
            print(f"Timeline generated from dSCAPER for dialogue {dialog.id}")

        else:
            logging.warning(
                "The dSCAPER is not set, which make the generation of the timeline impossible"
            )

        # Generate the audio room accoustic
        if room is not None and self._dscaper is not None:
            print(f"Generating room accoustic for dialogue {dialog.id}")
            dialog: AudioDialog = generate_audio_room_accoustic(dialog, microphone_position)
            print(f"Room accoustic generated for dialogue {dialog.id}!")
        else:
            logging.warning(
                "The room or the dSCAPER is not set, which make the generation of the room accoustic audio impossible"
            )

        # Save information about the audio pipeline
        audio_pipeline_info = {
            "dialog_id": dialog.id,
            "tts_pipeline": self.tts_pipeline.__class__.__name__,
            "voice_database": self.voice_database.__class__.__name__,
            "enricher": self.enricher.__class__.__name__,
            "dscaper": self._dscaper.__class__.__name__ if self._dscaper is not None else None,
            "room": room.get_info() if room is not None else None,
            "microphone_position": microphone_position.value if microphone_position is not None else None,
            "do_word_alignments": do_word_alignments,
            "do_snr": do_snr,
            "do_room_position": do_room_position,
            "sampling_rate": self.sampling_rate,
            "dir_audio": self.dir_audio,
            "personas": dialog.personas,
        }
        # Save the audio pipeline info to a json file
        with open(os.path.join(
            dialog.audio_dir_path,
            f"dialog_{dialog.id}",
            "exported_audios",
            "audio_pipeline_info.json"
        ), "w") as f:
            json.dump(audio_pipeline_info, f, indent=4)

        return dialog
