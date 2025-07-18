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
    
    def __init__(
        self,
        turn: Turn,
        audio: np.ndarray = None,
        audio_path: str = None,
        audio_duration: float = None,
        audio_start_time: float = None,
        snr: float = None,
        alignment: List[Tuple[float, float, str]] = None,
        transcript: str = None):

        super().__init__(turn.speaker, turn.text)
        
        self._audio = audio
        self.audio_path = audio_path
        self.audio_duration = audio_duration
        self.audio_start_time = audio_start_time
        self.snr = snr
        self.alignment = alignment
        self.transcript = transcript
