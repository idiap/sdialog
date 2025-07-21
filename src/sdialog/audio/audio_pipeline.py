import scaper
import numpy as np
import soundfile as sf
from sdialog import Dialog
from sdialog.audio.tts_engine import BaseTTS
from sdialog.audio.audio_events import Timeline
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.voice_database import BaseVoiceDatabase
from sdialog.audio.audio_events_enricher import AudioEventsEnricher
from sdialog.audio import generate_utterances_audios, save_utterances_audios, send_utterances_to_dscaper, generate_dscaper_timeline, generate_audio_room_accoustic, generate_word_alignments

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
        self.dscaper = dscaper
        self.sampling_rate = sampling_rate
        self.enricher = enricher
    

    def _master_audio(dialog: AudioDialog) -> np.ndarray:
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

        dialog = generate_utterances_audios(
            dialog,
            voice_database=self.voice_database,
            tts_pipeline=self.tts_pipeline
        )

        dialog = save_utterances_audios(dialog, self.dir_audio)

        # Combine the audio segments into a single master audio track as a baseline
        dialog.set_combined_audio(
            self._master_audio(dialog)
        )
        # save the combined audio to exported_audios folder
        sf.write(
            f"{dialog.audio_dir_path}/dialog_{dialog.id}/exported_audios/combined_audio.wav",
            dialog.get_combined_audio(),
            self.sampling_rate
        )

        # TODO: Test this computation of word alignments
        if do_word_alignments:
            dialog = generate_word_alignments(dialog)

        # TODO: Test this generation of SNR
        if do_snr:
            dialog = self.enricher.generate_snr(dialog)

        # TODO: Test this generation of room position
        if do_room_position:
            dialog = self.enricher.generate_room_position(dialog)

        # Send the utterances to dSCAPER
        dialog = send_utterances_to_dscaper(dialog, self.dscaper)

        # Generate the timeline from dSCAPER
        dialog = generate_dscaper_timeline(dialog, self.dscaper)
        
        # Generate the audio room accoustic
        dialog = generate_audio_room_accoustic(dialog)

        return dialog
