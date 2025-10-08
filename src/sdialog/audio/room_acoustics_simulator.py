"""
This module provides a class for simulating room acoustics.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import logging
import numpy as np
import soundfile as sf
from typing import List, Union
from sdialog.audio.audio_utils import BodyPosture
from sdialog.audio.room import Room, AudioSource, Position3D
from sdialog.audio.room import (
    DoctorPosition,
    PatientPosition,
    SoundEventPosition
)


class RoomAcousticsSimulator:
    """
    Simulates sound based on room acoustics based on room definition,
    sound sources provided and microphone(s) setup.
    """

    def __init__(
        self,
        room: Room = None,
        sampling_rate=44_100,
        kwargs_pyroom: dict = {}
    ):
        """
        Initialize the room acoustics simulator.
        """
        import pyroomacoustics as pra

        self.sampling_rate = sampling_rate
        self.ref_db = -65  # - 45 dB
        self.audiosources: List[AudioSource] = []
        self.room: Room = room
        self.kwargs_pyroom: dict = kwargs_pyroom

        if room is None:
            raise ValueError("Room is required")

        self._pyroom = self._create_pyroom(self.room, self.sampling_rate, self.kwargs_pyroom)

        # Remove existing microphone and add new one
        if hasattr(self._pyroom, "mic_array") and self._pyroom.mic_array is not None:
            self._pyroom.mic_array = None

        # Add microphone at new position
        self._pyroom.add_microphone_array(
            pra.MicrophoneArray(
                np.array([self.room.mic_position_3d.to_list()]).T, self._pyroom.fs
            )
        )

    def _create_pyroom(
        self,
        room: Room,
        sampling_rate=44_100,
        kwargs_pyroom: dict = {}
    ):
        """
        Create a pyroomacoustics room based on the room definition.
        """
        import pyroomacoustics as pra

        # If reverberation time ratio is provided, use it to create the materials
        if room.reverberation_time_ratio is not None:
            logging.info(f"Reverberation time ratio: {room.reverberation_time_ratio}")
            e_absorption, max_order = pra.inverse_sabine(room.reverberation_time_ratio, room.dimensions)
            _m = pra.Material(e_absorption)
        else:
            logging.info("Reverberation time ratio is not provided, using room materials")
            max_order = 17  # Number of reflections
            _m = pra.make_materials(
                ceiling=room.materials.ceiling,
                floor=room.materials.floor,
                east=room.materials.walls,
                west=room.materials.walls,
                north=room.materials.walls,
                south=room.materials.walls
            )

        _accoustic_room = pra.ShoeBox(
            room.dimensions,
            fs=sampling_rate,
            materials=_m,
            max_order=max_order,
            **kwargs_pyroom
        )

        if "ray_tracing" in kwargs_pyroom:
            _accoustic_room.set_ray_tracing()

        return _accoustic_room

    def _add_sources(
        self,
        audiosources: List[AudioSource]
    ):
        """
        Add audio sources to the room acoustics simulator.
        """

        for i, audio_source in enumerate(audiosources):

            self.audiosources.append(audio_source)

            # Parse the position of the audio source
            position = self.parse_position(audio_source.position)

            if position is not SoundEventPosition:

                audio_source._position3d = self.position_to_room_position(
                    self.room,
                    position
                )

            else:
                room_center = [dim / 2 for dim in self._pyroom.dimensions]
                audio_source._position3d = Position3D(room_center)

            # Load the audio file from the file system for the audio source
            if audio_source.source_file and os.path.exists(audio_source.source_file):

                # Read the audio file
                audio, original_fs = sf.read(audio_source.source_file)

                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)

                # Add the audio source to the room acoustics simulator at the position
                self._pyroom.add_source(
                    audio_source._position3d.to_list(),
                    signal=audio
                )

                logging.info((
                    f"✓ Loaded audio file '{audio_source.source_file}' for "
                    f"'{audio_source.name}' with {len(audio)} samples"
                ))

            else:
                logging.warning(f"Warning: No audio data found for '{audio_source.name}'")

    def simulate(
        self,
        sources: List[AudioSource] = [],
        reset: bool = False
    ):
        """
        Simulate the audio sources in the room.
        """

        if reset:
            # see https://github.com/LCAV/pyroomacoustics/issues/311
            self.reset()
            self._pyroom = self._create_pyroom(self.room, self.sampling_rate, self.kwargs_pyroom)

        self._add_sources(sources)
        self._pyroom.simulate()
        mixed_signal = self._pyroom.mic_array.signals[0, :]

        mixed_signal = self.apply_snr(mixed_signal, -0.03)  # scale audio to max 1dB
        return mixed_signal

    def reset(self):
        """
        Reset the room acoustics simulator.
        """

        del self._pyroom
        self._pyroom = None

    @staticmethod
    def apply_snr(x, snr):
        """Scale an audio signal to a given maximum SNR."""
        dbfs = 10 ** (snr / 20)
        x *= dbfs / np.abs(x).max(initial=1e-15)
        return x

    @staticmethod
    def parse_position(
        position: str,
    ) -> Union[DoctorPosition, PatientPosition, SoundEventPosition]:
        """
        Convert a position string to the appropriate position enum.
        """
        if position.startswith("doctor-"):
            try:
                return DoctorPosition(position)
            except ValueError:
                raise ValueError(f"Invalid doctor position: {position}")
        elif position.startswith("patient-"):
            try:
                return PatientPosition(position)
            except ValueError:
                raise ValueError(f"Invalid patient position: {position}")
        elif position.startswith("soundevent-"):
            return SoundEventPosition(position)
        elif position.startswith(
            SoundEventPosition.BACKGROUND.value
        ):  # no_type - background
            return SoundEventPosition(position)
        else:
            raise ValueError(
                f"Position must start with 'doctor-' or 'patient-', got: {position}"
            )

    @staticmethod
    def position_to_room_position(
        room: Room,
        pos: Union[DoctorPosition, PatientPosition]
    ) -> Position3D:
        """
        Convert semantic position enums to actual 3D coordinates within the room.

        This function maps abstract position descriptions (like "doctor:at_desk_sitting")
        to concrete 3D coordinates that can be used for acoustic simulation.

        Args:
            room: Room object containing dimensions and layout information
            pos: Position enum (DoctorPosition or PatientPosition)

        Returns:
            Position3D: 3D coordinates (x, y, z) in meters within the room
        """
        width, length, height = (
            room.dimensions.width,
            room.dimensions.length,
            room.dimensions.height,
        )

        def clamp_position(x, y, z):
            """Ensure position is within room bounds with safety margin"""
            margin = 0.1  # 10cm safety margin from walls
            x = max(margin, min(x, width - margin))
            y = max(margin, min(y, length - margin))
            z = max(0.1, min(z, height - 0.1))
            return Position3D.from_list([x, y, z])

        # Map doctor positions
        if isinstance(pos, DoctorPosition):
            if pos == DoctorPosition.AT_DESK_SITTING:
                return clamp_position(
                    room.furnitures["desk"].x,
                    room.furnitures["desk"].y,
                    BodyPosture.SITTING.value
                )
            elif pos == DoctorPosition.AT_DESK_SIDE_STANDING:
                return clamp_position(
                    room.furnitures["desk"].x + 0.5,
                    room.furnitures["desk"].y,
                    BodyPosture.STANDING.value
                )
            elif pos == DoctorPosition.NEXT_TO_BENCH_STANDING:
                return clamp_position(
                    room.furnitures["bench"].x - 0.8,
                    room.furnitures["bench"].y,
                    BodyPosture.STANDING.value
                )
            elif pos == DoctorPosition.NEXT_TO_SINK_FRONT:
                return clamp_position(
                    room.furnitures["sink"].x + 0.3,
                    room.furnitures["sink"].y - 0.5,
                    BodyPosture.STANDING.value
                )
            elif pos == DoctorPosition.NEXT_TO_SINK_BACK:
                return clamp_position(
                    room.furnitures["sink"].x - 0.3,
                    room.furnitures["sink"].y + 0.3,
                    BodyPosture.STANDING.value
                )
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_FRONT:
                return clamp_position(
                    room.furnitures["cupboard"].x - 0.3,
                    room.furnitures["cupboard"].y - 0.5,
                    BodyPosture.STANDING.value
                )
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_BACK:
                return clamp_position(
                    room.furnitures["cupboard"].x + 0.3,
                    room.furnitures["cupboard"].y + 0.3,
                    BodyPosture.STANDING.value
                )
            elif pos == DoctorPosition.NEXT_TO_DOOR_STANDING:
                return clamp_position(
                    room.furnitures["door"].x + 0.5,
                    room.furnitures["door"].y + 0.3,
                    BodyPosture.STANDING.value
                )

        # Map patient positions
        elif isinstance(pos, PatientPosition):
            if pos == PatientPosition.AT_DOOR_STANDING:
                return clamp_position(
                    room.furnitures["door"].x + 0.3,
                    room.furnitures["door"].y + 0.2,
                    BodyPosture.STANDING.value
                )
            elif pos == PatientPosition.NEXT_TO_DESK_SITTING:
                return clamp_position(
                    room.furnitures["desk"].x + 0.8,
                    room.furnitures["desk"].y + 0.3,
                    BodyPosture.SITTING.value
                )
            elif pos == PatientPosition.NEXT_TO_DESK_STANDING:
                return clamp_position(
                    room.furnitures["desk"].x + 0.8,
                    room.furnitures["desk"].y + 0.3,
                    BodyPosture.STANDING.value
                )
            elif pos == PatientPosition.SITTING_ON_BENCH:
                return clamp_position(
                    room.furnitures["bench"].x,
                    room.furnitures["bench"].y,
                    BodyPosture.SITTING.value
                )
            elif pos == PatientPosition.CENTER_ROOM_STANDING:
                return clamp_position(
                    room.furnitures["center"].x,
                    room.furnitures["center"].y,
                    BodyPosture.STANDING.value
                )

        # Fallback to center of room if position not recognized
        return clamp_position(
            room.furnitures["center"].x,
            room.furnitures["center"].y,
            BodyPosture.STANDING.value
        )
