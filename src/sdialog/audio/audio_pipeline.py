import scaper
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf
from typing import List
from sdialog import Dialog
from datasets import load_dataset
from sdialog.audio.tts_engine import BaseTTS
from scaper.dscaper_datatypes import DscaperAudio
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.voice_database import BaseVoiceDatabase
from sdialog.audio.audio_events_enricher import AudioEventsEnricher
from sdialog.audio import generate_utterances_audios, save_utterances_audios, send_utterances_to_dscaper
from sdialog.audio import generate_dscaper_timeline, generate_audio_room_accoustic, generate_word_alignments


class AudioPipeline:
    """
    Audio pipeline.
    """

    def __init__(
            self,
            dir_audio: str,
            tts_pipeline: BaseTTS,
            voice_database: BaseVoiceDatabase,
            dscaper: scaper.Dscaper,
            enricher: AudioEventsEnricher,
            sampling_rate: int = 24_000):
        """
        Initialize the audio pipeline.
        """

        self.dir_audio = dir_audio
        self.tts_pipeline = tts_pipeline
        self.voice_database = voice_database
        self._dscaper = dscaper
        self.enricher = enricher
        self.sampling_rate = sampling_rate

    def populate_dscaper(self, datasets: List[str]) -> int:
        """
        Populate the dSCAPER with the audio recordings.
        """

        n_audio_files = 0

        # For each huggingface dataset, save the audio recordings to the dSCAPER
        for dataset_name in datasets:

            # Load huggingface dataset
            dataset = load_dataset(dataset_name, split="train")

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

    def master_audio(self, dialog: AudioDialog) -> np.ndarray:
        """
        Combines multiple audio segments into a single master audio track.
        """
        return np.concatenate([turn.get_audio() for turn in dialog.turns])

    def enrich(self, dialog: AudioDialog) -> AudioDialog:
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
            do_word_alignments: bool = False,
            do_snr: bool = False,
            do_room_position: bool = False) -> AudioDialog:
        """
        Run the audio pipeline.
        """

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
        # save the combined audio to exported_audios folder
        sf.write(
            f"{dialog.audio_dir_path}/dialog_{dialog.id}/exported_audios/combined_audio.wav",
            dialog.get_combined_audio(),
            self.sampling_rate
        )

        # TODO: Test this computation of word alignments
        if do_word_alignments:
            dialog: AudioDialog = generate_word_alignments(dialog)

        # TODO: Test this generation of SNR
        if do_snr:
            dialog: AudioDialog = self.enricher.generate_snr(dialog)

        if do_room_position:
            dialog: AudioDialog = self.enricher.generate_room_position(dialog)

        # Randomly sample a static microphone position for the whole dialogue
        dialog: AudioDialog = self.enricher.generate_microphone_position(dialog)

        # Send the utterances to dSCAPER
        dialog: AudioDialog = send_utterances_to_dscaper(dialog, self._dscaper)

        # Generate the timeline from dSCAPER
        dialog: AudioDialog = generate_dscaper_timeline(dialog, self._dscaper)

        # Generate the audio room accoustic
        dialog: AudioDialog = generate_audio_room_accoustic(dialog)

        return dialog
