import numpy as np
from typing import List
from sdialog import Dialog
from sdialog.audio.room import AudioSource
from sdialog.audio.audio_turn import AudioTurn
from sdialog.audio.audio_events import Timeline


class AudioDialog(Dialog):
    """
    Represents a dialogue with audio turns.
    """

    turns: List[AudioTurn] = []
    audio_dir_path: str = None
    timeline: Timeline = None
    total_duration: float = None
    timeline_name: str = None

    _combined_audio: np.ndarray = None
    _audio_sources: List[AudioSource] = []

    audio_step_1_filepath: str = None
    audio_step_2_filepath: str = None
    audio_step_3_filepaths: dict[str, dict[str, str]] = {}

    # Room hash or user input name

    def __init__(self):
        super().__init__()

    def set_audio_sources(self, audio_sources: List[AudioSource]):
        """
        Set the audio sources of the dialog.
        """
        self._audio_sources = audio_sources

    def add_audio_source(self, audio_source: AudioSource):
        """
        Add an audio source to the dialog.
        """
        self._audio_sources.append(audio_source)

    def get_audio_sources(self) -> List[AudioSource]:
        """
        Get the audio sources of the dialog.
        """
        return self._audio_sources

    def set_combined_audio(self, audio: np.ndarray):
        """
        Set the combined audio of the dialog.
        """
        self._combined_audio = audio

    def get_combined_audio(self) -> np.ndarray:
        """
        Get the combined audio of the dialog.
        """
        return self._combined_audio

    @staticmethod
    def from_dialog(dialog: Dialog):
        audio_dialog = AudioDialog()

        for attr in dialog.__dict__:
            setattr(audio_dialog, attr, getattr(dialog, attr))

        audio_dialog.turns = [AudioTurn.from_turn(turn) for turn in dialog.turns]
        return audio_dialog
