"""
Pyroomacoustics backend implementation.
"""

from typing import Any, Callable, Optional

from sdialog.audio.dialog import AudioDialog
from sdialog.audio.room import Room, RoomPosition

from .base import BaseRoomAcousticsBackend


class PyroomAcousticsBackend(BaseRoomAcousticsBackend):
    """
    Room acoustics backend using the existing pyroomacoustics flow.
    """

    requires_room = True
    name = "pyroom"

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
        Generate room acoustics audio with pyroomacoustics.

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
        :param environment: Optional environment overrides for pyroom settings.
        :type environment: Optional[dict]
        :param callback_mix_fn: Optional callback used during audio mixing.
        :type callback_mix_fn: Optional[Callable]
        :param callback_mix_kwargs: Optional keyword arguments for the mix callback.
        :type callback_mix_kwargs: Optional[dict]
        :param sampling_rate: Unused argument kept for API compatibility.
        :type sampling_rate: int
        :return: Updated dialog with room acoustics outputs.
        :rtype: AudioDialog
        :raises ValueError: If ``room`` is not an instance of ``Room``.
        """
        del sampling_rate
        if not isinstance(room, Room):
            raise ValueError("PyroomAcousticsBackend expects `room` to be an instance of `Room`.")

        from sdialog.audio import generate_audio_room_accoustic

        env = environment or {}
        return generate_audio_room_accoustic(
            dialog=dialog,
            room=room,
            dialog_directory=dialog_directory,
            room_name=room_name,
            kwargs_pyroom=env.get("kwargs_pyroom", {}),
            source_volumes=env.get("source_volumes", {}),
            audio_file_format=audio_file_format,
            background_effect=env.get("background_effect", "white_noise"),
            foreground_effect=env.get("foreground_effect", "ac_noise_minimal"),
            foreground_effect_position=env.get("foreground_effect_position", RoomPosition.TOP_RIGHT),
            callback_mix_fn=callback_mix_fn,
            callback_mix_kwargs=callback_mix_kwargs or {},
        )
