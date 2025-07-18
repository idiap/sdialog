import numpy as np
from sdialog import Turn
from typing import List, Tuple, Optional

class AudioTurn(Turn):
    """
    Represents a single turn in a dialogue, with associated audio data.

    :ivar audio_path: The path to the audio file for this turn.
    :vartype audio_path: Optional[str]
    :ivar audio_duration: The duration of the audio in seconds.
    :vartype audio_duration: Optional[float]
    :ivar audio_start_time: The start time of the audio in seconds.
    :vartype audio_start_time: Optional[float]
    :ivar snr: The signal-to-noise ratio of the audio.
    :vartype snr: Optional[float]
    :ivar alignment: The alignment of the audio with the text.
    :vartype alignment: Optional[List[Tuple[float, float, str]]]
    """
    
    _audio: np.ndarray = None
    audio_path: str = None
    audio_duration: float = None
    audio_start_time: float = None
    snr: float = None
    alignment: List[Tuple[float, float, str]] = None
    transcript: str = None

    def get_audio(self) -> np.ndarray:
        """
        Get the audio of the turn.
        """
        return self._audio

    def set_audio(self, audio: np.ndarray):
        """
        Set the audio of the turn.
        """
        self._audio = audio

    @staticmethod
    def from_turn(
        turn: Turn,
        audio: np.ndarray = None,
        audio_path: str = None,
        audio_duration: float = None,
        audio_start_time: float = None,
        snr: float = None,
        alignment: List[Tuple[float, float, str]] = None,
        transcript: str = None):

        audio_turn = AudioTurn(text=turn.text, speaker=turn.speaker)

        audio_turn._audio = audio
        audio_turn.audio_path = audio_path
        audio_turn.audio_duration = audio_duration
        audio_turn.audio_start_time = audio_start_time
        audio_turn.snr = snr
        audio_turn.alignment = alignment
        audio_turn.transcript = transcript

        return audio_turn
