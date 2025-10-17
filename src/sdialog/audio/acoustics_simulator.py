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
from typing import List
from sdialog.audio.utils import SourceVolume
from sdialog.audio.room import Room, AudioSource
from sdialog.audio.room import (
    RoomPosition,
    DirectivityType
)


class AcousticsSimulator:
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
        if (
            self.room.directivity_type is None or
            self.room.directivity_type == DirectivityType.OMNIDIRECTIONAL
        ):
            self._pyroom.add_microphone_array(
                pra.MicrophoneArray(
                    np.array([self.room.mic_position_3d.to_list()]).T, self._pyroom.fs
                )
            )
        else:
            _directivity: pra.directivities.Cardioid = self.room.microphone_directivity.to_pyroomacoustics()
            self._pyroom.add_microphone(
                self.room.mic_position_3d.to_list(),
                directivity=_directivity
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

        if "ray_tracing" in kwargs_pyroom and kwargs_pyroom["ray_tracing"]:
            _accoustic_room.set_ray_tracing()

        if "air_absorption" in kwargs_pyroom and kwargs_pyroom["air_absorption"]:
            _accoustic_room.set_air_absorption()

        return _accoustic_room

    def _add_sources(
        self,
        audiosources: List[AudioSource],
        source_volumes: dict[str, SourceVolume] = {}
    ):
        """
        Add audio sources to the room acoustics simulator.
        """

        for i, audio_source in enumerate(audiosources):

            self.audiosources.append(audio_source)

            # Get the position of the audio source
            if audio_source.position.startswith("no_type"):
                _position3d = self.room.room_position_to_position3d(RoomPosition.CENTER)
            elif audio_source.position.startswith("room-"):
                _position3d = self.room.room_position_to_position3d(audio_source.position)
            elif audio_source.position.startswith("speaker_"):
                _position3d = self.room.speakers_positions[audio_source.position]

            # Load the audio file from the file system for the audio source
            if audio_source.source_file and os.path.exists(audio_source.source_file):

                # Read the audio file
                audio, original_fs = sf.read(audio_source.source_file)

                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)

                # Reduce the volume of those audio sources
                if audio_source.position.startswith("room-"):
                    audio = (
                        audio * source_volumes["room-"].value
                        if "room-" in source_volumes
                        else SourceVolume.HIGH.value
                    )
                elif audio_source.position.startswith("no_type"):
                    audio = (
                        audio * source_volumes["no_type"].value
                        if "no_type" in source_volumes
                        else SourceVolume.VERY_LOW.value
                    )

                # Add the audio source to the room acoustics simulator at the position
                self._pyroom.add_source(
                    _position3d.to_list(),
                    signal=audio
                )

            else:
                logging.warning(f"Warning: No audio data found for '{audio_source.name}'")

    def simulate(
        self,
        sources: List[AudioSource] = [],
        source_volumes: dict[str, SourceVolume] = {},
        reset: bool = False
    ):
        """
        Simulate the audio sources in the room.
        """

        if reset:
            # see https://github.com/LCAV/pyroomacoustics/issues/311
            self.reset()
            self._pyroom = self._create_pyroom(self.room, self.sampling_rate, self.kwargs_pyroom)

        self._add_sources(sources, source_volumes)
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
