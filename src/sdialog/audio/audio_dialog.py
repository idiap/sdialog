import os
import numpy as np
import soundfile as sf
from typing import List
from sdialog import Dialog
from sdialog.audio.audio_turn import AudioTurn

class AudioDialog(Dialog):
    """
    Represents a dialogue with audio turns.
    """
    turns: List[AudioTurn] = []
    _combined_audio: np.ndarray = None
    audio_dir_path: str = None

    def __init__(self):
        super().__init__()

    def set_combined_audio(self, audio: np.ndarray):
        """
        Set the combined audio of the dialog.
        """
        self._combined_audio = audio
    
    @staticmethod
    def from_dialog(dialog: Dialog):
        audio_dialog = AudioDialog()

        for attr in dialog.__dict__:
            setattr(audio_dialog, attr, getattr(dialog, attr))
        
        audio_dialog.turns = [AudioTurn.from_turn(turn) for turn in dialog.turns]
        return audio_dialog
    
    def set_audio_dir_path(self, path: str):
        """
        Set the audio directory path for the dialog.
        """
        self.audio_dir_path = path.rstrip("/")
        os.makedirs(f"{self.audio_dir_path}/dialog_{self.id}/utterances", exist_ok=True)
        os.makedirs(f"{self.audio_dir_path}/dialog_{self.id}/exported_audios", exist_ok=True)

    def save_utterances_audios(self):
        """
        Save the utterances audios to the given path.
        """
        if self.audio_dir_path is None:
            self.set_audio_dir_path("./outputs")

        for idx, turn in enumerate(self.turns):
            sf.write(
                f"{self.audio_dir_path}/dialog_{self.id}/utterances/{idx}_{turn.speaker}.wav",
                turn.get_audio(),
                24_000
            )
