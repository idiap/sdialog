import numpy as np
from typing import List
from sdialog import Dialog
from sdialog.audio.audio_turn import AudioTurn
from sdialog.audio.audio_events import Timeline
from sdialog.audio.room import Room, AudioSource


class AudioDialog(Dialog):
    """
    Represents a dialogue with audio turns.
    """

    turns: List[AudioTurn] = []
    audio_dir_path: str = None
    timeline: Timeline = None
    total_duration: float = None
    timeline_name: str = None

    _room: Room = None
    _combined_audio: np.ndarray = None
    _audio_sources: List[AudioSource] = []

    _audio_step_1_filepath: str = None
    _audio_step_2_filepath: str = None
    _audio_step_3_filepath: str = None

    def __init__(self):
        super().__init__()

    def set_room(self, room: Room):
        """
        Set the room of the dialog.
        """
        self._room = room

    def get_room(self) -> Room:
        """
        Get the room of the dialog.
        """
        return self._room

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
