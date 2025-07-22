import numpy as np
from typing import List
from sdialog import Dialog
from sdialog.audio.audio_turn import AudioTurn
from sdialog.audio.audio_events import Timeline


class AudioDialog(Dialog):
    """
    Represents a dialogue with audio turns.
    """
    turns: List[AudioTurn] = []
    _combined_audio: np.ndarray = None
    audio_dir_path: str = None
    timeline: Timeline = None
    total_duration: float = None
    timeline_name: str = None

    def __init__(self):
        super().__init__()

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
