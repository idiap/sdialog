"""
Base room acoustics backend contract.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from sdialog.audio.dialog import AudioDialog


class BaseRoomAcousticsBackend(ABC):
    """
    Abstract base class for room acoustics backends.
    """

    requires_room: bool = True
    name: str = "base"

    @abstractmethod
    def simulate(
        self,
        dialog: AudioDialog,
        room: Optional[Any],
        dialog_directory: str,
        room_name: str,
        audio_file_format: str = "wav",
        environment: Optional[dict] = None,
        callback_mix_fn: Optional[Callable] = None,
        callback_mix_kwargs: Optional[dict] = None,
        sampling_rate: int = 44_100,
    ) -> AudioDialog:
        """
        Run room acoustics simulation and update the dialog outputs.

        :param dialog: Audio dialog object to update.
        :type dialog: AudioDialog
        :param room: Room configuration used for simulation.
        :type room: Optional[Any]
        :param dialog_directory: Relative output directory for generated files.
        :type dialog_directory: str
        :param room_name: Name of the room profile to generate.
        :type room_name: str
        :param audio_file_format: Audio format for exported files (default: "wav").
        :type audio_file_format: str
        :param environment: Backend-specific environment parameters.
        :type environment: Optional[dict]
        :param callback_mix_fn: Optional callback used during audio mixing.
        :type callback_mix_fn: Optional[Callable]
        :param callback_mix_kwargs: Optional keyword arguments for the mix callback.
        :type callback_mix_kwargs: Optional[dict]
        :param sampling_rate: Sampling rate used for generated audio (default: 44100).
        :type sampling_rate: int
        :return: Updated dialog with room acoustics outputs.
        :rtype: AudioDialog
        """
