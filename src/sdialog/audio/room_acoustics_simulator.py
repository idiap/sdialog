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
from sdialog.audio.room import Room, AudioSource, Position3D
from sdialog.audio.audio_scaper_utils import microphone_position_to_room_position
from sdialog.audio.room import (
    DoctorPosition,
    PatientPosition,
    SoundEventPosition,
    MicrophonePosition,
)


class RoomAcousticsSimulator:
    """
    Simulates sound based on room acoustics based on room definition,
    sound sources provided and microphone(s) setup.

     Example:
        from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole
        from sdialog.audio.room import MicrophonePosition

        # Create room with specific microphone position
        room = MedicalRoomGenerator().generate(RoomRole.CONSULTATION)
        room_acoustics = RoomAcousticsSimulator(room)

        # Change microphone position using enum
        room_acoustics.set_microphone_position(MicrophonePosition.CEILING_CENTERED)
        # Or use explicit coordinates
        room_acoustics.set_microphone_position([2.0, 1.5, 1.8])

        # Add audio sources and simulate
        audio = room_acoustics.simulate(audio_sources)
    """

    def __init__(self, room: Room = None, sampling_rate=44_100):
        """
        Initialize the room acoustics simulator.
        """

        self.sampling_rate = sampling_rate
        self.ref_db = -65  # - 45 dB
        self.audiosources: List[AudioSource] = []
        self.room: Room = room

        if room is None:
            raise ValueError("Room is required")

        self._pyroom = self._create_pyroom(self.room, self.sampling_rate)

        # Set microphone position based on room's mic_position setting
        self.mic_position = microphone_position_to_room_position(
            self.room,
            self.room.mic_position
        )

        self.set_microphone_position(self.mic_position.to_list())

    def _create_pyroom(self, room: Room, sampling_rate=44_100):
        """
        Create a pyroomacoustics room based on the room definition.
        """

        import pyroomacoustics as pra
        e_absorption, max_order = pra.inverse_sabine(room.reverberation_time_ratio, room.dimensions)
        # max_order = 17  # Number of reflections
        return pra.ShoeBox(
            room.dimensions,
            fs=sampling_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

    def set_microphone_position(
        self,
        mic_pos: Union[MicrophonePosition, List[float], Position3D, str]
    ):
        """
        Set microphone position using MicrophonePosition enum or explicit coordinates.

        Args:
            mic_pos: Can be MicrophonePosition enum, list [x,y,z], or Position3D object
        """
        import pyroomacoustics as pra

        if isinstance(mic_pos, MicrophonePosition):
            position_3d = microphone_position_to_room_position(self.room, mic_pos)
        elif isinstance(mic_pos, list):
            position_3d = Position3D.from_list(mic_pos)
        elif isinstance(mic_pos, Position3D):
            position_3d = mic_pos
        elif isinstance(mic_pos, str):
            mic_pos = MicrophonePosition(mic_pos)
            position_3d = microphone_position_to_room_position(self.room, mic_pos)
        else:
            raise ValueError(
                "mic_pos must be MicrophonePosition enum, list [x,y,z], or Position3D object"
            )

        self.mic_position = position_3d

        # Remove existing microphone and add new one
        if hasattr(self._pyroom, "mic_array") and self._pyroom.mic_array is not None:
            # Clear existing microphone array
            self._pyroom.mic_array = None

        # Add microphone at new position
        mic_array = pra.MicrophoneArray(
            np.array([self.mic_position.to_list()]).T, self._pyroom.fs
        )
        self._pyroom.add_microphone_array(mic_array)
        logging.info(f"  Microphone set to position {self.mic_position.to_list()}")

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
                    self.room, position
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
            self._pyroom = self._create_pyroom(self.room, self.sampling_rate)

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
        room: Room, pos: Union[DoctorPosition, PatientPosition]
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

        Standard room layout assumptions as define in enums DoctorPosition or PatientPosition :
        - Door at (0, 0) corner
        - Desk along the width wall at 1/4 from door
        - Examination bench in center area
        - Sink and cupboards along length walls
        - Standard sitting height: 0.5m, standing height: 1.7m
        """
        width, length, height = (
            room.dimensions.width,
            room.dimensions.length,
            room.dimensions.height,
        )

        # Define standard furniture positions as fractions of room dimensions
        desk_pos = (width * 0.25, length * 0.15)  # Near corner, away from door
        bench_pos = (width * 0.6, length * 0.5)  # Center-right area
        door_pos = (0.1, 0.1)  # Near corner
        sink_pos = (width * 0.05, length * 0.8)  # Back wall, near corner
        cupboard_pos = (width * 0.95, length * 0.8)  # Back wall, opposite corner
        center_pos = (width * 0.5, length * 0.5)  # Room center

        # Heights for different postures
        sitting_height = 0.5  # Chair/bench sitting height
        standing_height = 1.7  # Average person standing height

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
                return clamp_position(desk_pos[0], desk_pos[1], sitting_height)
            elif pos == DoctorPosition.AT_DESK_SIDE_STANDING:
                return clamp_position(desk_pos[0] + 0.5, desk_pos[1], standing_height)
            elif pos == DoctorPosition.NEXT_TO_BENCH_STANDING:
                return clamp_position(bench_pos[0] - 0.8, bench_pos[1], standing_height)
            elif pos == DoctorPosition.NEXT_TO_SINK_FRONT:
                return clamp_position(
                    sink_pos[0] + 0.3, sink_pos[1] - 0.5, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_SINK_BACK:
                return clamp_position(
                    sink_pos[0] - 0.3, sink_pos[1] + 0.3, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_FRONT:
                return clamp_position(
                    cupboard_pos[0] - 0.3, cupboard_pos[1] - 0.5, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_CUPBOARD_BACK:
                return clamp_position(
                    cupboard_pos[0] + 0.3, cupboard_pos[1] + 0.3, standing_height
                )
            elif pos == DoctorPosition.NEXT_TO_DOOR_STANDING:
                return clamp_position(
                    door_pos[0] + 0.5, door_pos[1] + 0.3, standing_height
                )

        # Map patient positions
        elif isinstance(pos, PatientPosition):
            if pos == PatientPosition.AT_DOOR_STANDING:
                return clamp_position(
                    door_pos[0] + 0.3, door_pos[1] + 0.2, standing_height
                )
            elif pos == PatientPosition.NEXT_TO_DESK_SITTING:
                return clamp_position(
                    desk_pos[0] + 0.8, desk_pos[1] + 0.3, sitting_height
                )
            elif pos == PatientPosition.NEXT_TO_DESK_STANDING:
                return clamp_position(
                    desk_pos[0] + 0.8, desk_pos[1] + 0.3, standing_height
                )
            elif pos == PatientPosition.SITTING_ON_BENCH:
                return clamp_position(bench_pos[0], bench_pos[1], sitting_height)
            elif pos == PatientPosition.CENTER_ROOM_STANDING:
                return clamp_position(center_pos[0], center_pos[1], standing_height)

        # Fallback to center of room if position not recognized
        return clamp_position(center_pos[0], center_pos[1], standing_height)
