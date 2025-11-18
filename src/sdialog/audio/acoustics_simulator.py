"""
This module provides comprehensive room acoustics simulation capabilities.

The module includes the AcousticsSimulator class that enables realistic
room acoustics simulation using the pyroomacoustics library. It supports
complex room geometries, acoustic materials, microphone positioning,
and audio source management for high-quality acoustic environment modeling.

Key Features:

  - Room acoustics simulation using pyroomacoustics
  - Support for complex room geometries and materials
  - Microphone positioning with directivity patterns
  - Audio source management and positioning
  - Reverberation and acoustic effect modeling
  - High-quality audio processing and simulation

Acoustics Simulation Process:

  1. Room geometry and material setup
  2. Microphone positioning and directivity configuration
  3. Audio source placement and characteristics
  4. Acoustic simulation with pyroomacoustics
  5. Audio processing and output generation

Example:

    .. code-block:: python

        from sdialog.audio import AcousticsSimulator, Room
        from sdialog.audio.utils import SourceVolume

        # Create room configuration
        room = Room(dimensions=(5.0, 4.0, 3.0))

        # Initialize acoustics simulator
        simulator = AcousticsSimulator(room=room)

        # Simulate room acoustics
        audio_output = simulator.simulate(
            sources=audio_sources,
            source_volumes={"speaker_1": SourceVolume.MEDIUM}
        )
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Pawel Cyrta <pawel@cyrta.com>
# SPDX-License-Identifier: MIT
import os
import numpy as np
import soundfile as sf
from typing import List

from sdialog.audio.utils import logger, SourceVolume
from sdialog.audio.room import Room, AudioSource, RoomPosition, DirectivityType


class AcousticsSimulator:
    """
    Comprehensive room acoustics simulator using pyroomacoustics.

    This class provides realistic room acoustics simulation by modeling
    sound propagation, reflection, and absorption in 3D room environments.
    It integrates with the pyroomacoustics library to provide high-quality
    acoustic modeling with support for complex room geometries, materials,
    and audio source positioning.

    Key Features:

      - Room acoustics simulation using pyroomacoustics
      - Support for complex room geometries and materials
      - Microphone positioning with directivity patterns
      - Audio source management and positioning
      - Reverberation and acoustic effect modeling
      - High-quality audio processing and simulation

    Simulation Process:

      1. Room geometry and material setup
      2. Microphone positioning and directivity configuration
      3. Audio source placement and characteristics
      4. Acoustic simulation with pyroomacoustics
      5. Audio processing and output generation

    :ivar sampling_rate: Audio sampling rate in Hz (default: 44100).
    :vartype sampling_rate: int
    :ivar ref_db: Reference decibel level for audio processing.
    :vartype ref_db: int
    :ivar audiosources: List of audio sources for simulation.
    :vartype audiosources: List[AudioSource]
    :ivar room: Room configuration for acoustics simulation.
    :vartype room: Room
    :ivar kwargs_pyroom: Additional parameters for pyroomacoustics.
    :vartype kwargs_pyroom: dict
    :ivar _pyroom: Internal pyroomacoustics room object.
    :vartype _pyroom: Any
    """

    def __init__(
        self,
        room: Room = None,
        sampling_rate=44_100,
        kwargs_pyroom: dict = {}
    ):
        """
        Initializes the room acoustics simulator.

        This constructor sets up the acoustics simulator with the specified
        room configuration, sampling rate, and pyroomacoustics parameters.
        It creates the internal pyroomacoustics room object and configures
        the microphone positioning and directivity.

        :param room: Room configuration for acoustics simulation.
        :type room: Room
        :param sampling_rate: Audio sampling rate in Hz (default: 44100).
        :type sampling_rate: int
        :param kwargs_pyroom: Additional parameters for pyroomacoustics.
        :type kwargs_pyroom: dict
        :raises ValueError: If room is not provided.
        :raises ImportError: If pyroomacoustics is not installed.
        """
        import pyroomacoustics as pra

        self.sampling_rate = sampling_rate
        self.ref_db = -65  # - 45 dB
        self.audiosources: List[AudioSource] = []
        self.room: Room = room
        self.kwargs_pyroom: dict = kwargs_pyroom if kwargs_pyroom is not None else {}

        if room is None:
            raise ValueError("Room is required")

        self._pyroom = self._create_pyroom(self.room, self.sampling_rate, self.kwargs_pyroom)

        # Remove existing microphone and add new one
        if hasattr(self._pyroom, "mic_array") and self._pyroom.mic_array is not None:
            self._pyroom.mic_array = None

        # Add microphone at new position
        if (
            self.room.directivity_type is None
            or self.room.directivity_type == DirectivityType.OMNIDIRECTIONAL
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
        Creates a pyroomacoustics room object based on the room definition.

        This method constructs the internal pyroomacoustics room object
        using the provided room configuration, including dimensions,
        materials, and acoustic properties. It handles both material-based
        and reverberation time-based room setup.

        Room setup process:
        1. Determine acoustic materials (from room materials or reverberation time)
        2. Create pyroomacoustics ShoeBox room with dimensions
        3. Configure materials and acoustic properties
        4. Set up room acoustics simulation parameters

        :param room: Room configuration for acoustics simulation.
        :type room: Room
        :param sampling_rate: Audio sampling rate in Hz (default: 44100).
        :type sampling_rate: int
        :param kwargs_pyroom: Additional parameters for pyroomacoustics.
        :type kwargs_pyroom: dict
        :return: Configured pyroomacoustics room object.
        :rtype: Any
        :raises ImportError: If pyroomacoustics is not installed.
        :raises ValueError: If room configuration is invalid.
        """
        import pyroomacoustics as pra

        # If reverberation time ratio is provided, use it to create the materials
        if room.reverberation_time_ratio is not None:
            logger.info(f"Reverberation time ratio: {room.reverberation_time_ratio}")
            e_absorption, max_order = pra.inverse_sabine(room.reverberation_time_ratio, room.dimensions)
            _m = pra.Material(e_absorption)
        else:
            logger.info("Reverberation time ratio is not provided, using room materials")
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
            room.dimensions.to_list(),
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
            if audio_source.position.startswith("no_type"):  # no_type is the background sound
                _position3d = self.room.room_position_to_position3d(RoomPosition.CENTER)
            elif audio_source.position.startswith("room-"):  # room- is the foreground sound
                _position3d = self.room.room_position_to_position3d(
                    RoomPosition(audio_source.position)
                )
            elif audio_source.position.startswith("speaker_"):  # speaker_ is the speaker sound
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
                        if source_volumes is not None and "room-" in source_volumes
                        else audio * SourceVolume.HIGH.value
                    )
                elif audio_source.position.startswith("no_type"):
                    audio = (
                        audio * source_volumes["no_type"].value
                        if source_volumes is not None and "no_type" in source_volumes
                        else audio * SourceVolume.VERY_LOW.value
                    )

                # Add the audio source to the room acoustics simulator at the position
                self._pyroom.add_source(
                    _position3d.to_list(),
                    signal=audio
                )

            else:
                logger.warning(f"Warning: No audio data found for '{audio_source.name}'")

    def simulate(
        self,
        sources: List[AudioSource] = [],
        source_volumes: dict[str, SourceVolume] = {},
        reset: bool = False
    ):
        """
        Simulates room acoustics for the given audio sources.

        This method performs the complete room acoustics simulation process,
        including audio source placement, volume adjustment, and acoustic
        processing using pyroomacoustics. It returns the processed audio
        with room acoustics effects applied.

        Simulation process:
        1. Optionally reset the room acoustics simulator
        2. Add audio sources with specified volumes
        3. Perform room acoustics simulation
        4. Process and return the resulting audio

        :param sources: List of audio sources to simulate in the room.
        :type sources: List[AudioSource]
        :param source_volumes: Dictionary mapping source identifiers to volume levels.
        :type source_volumes: dict[str, SourceVolume]
        :param reset: If True, resets the room acoustics simulator before simulation.
        :type reset: bool
        :return: Processed audio with room acoustics effects applied.
        :rtype: np.ndarray
        :raises ValueError: If audio sources are invalid or empty.
        :raises RuntimeError: If simulation fails.
        """

        if reset:
            # see https://github.com/LCAV/pyroomacoustics/issues/311
            self.reset()
            self._pyroom = self._create_pyroom(self.room, self.sampling_rate, self.kwargs_pyroom)

        try:
            self._add_sources(sources, source_volumes)

            logger.info("[Step 3] Simulating room acoustics...")
            self._pyroom.simulate()

        except ValueError as e:

            if "zero-size array to reduction operation maximum" in str(e):
                raise ValueError(
                    "[Step 3] Simulation failed: The distance between the sources (speakers, background or foreground) "
                    "and the microphone is too large for the current room dimensions. "
                    "Please place sources closer to the microphone or increase the room size."
                ) from e

            elif "The source must be added inside the room" in str(e):
                raise ValueError(
                    "[Step 3] Simulation failed: One or more audio sources (speakers, background or foreground) "
                    "are positioned outside the room boundaries. Please check that all speakers, "
                    "foreground and background sound positions are within the room dimensions. "
                    "You can use the `room.to_image()` method to visualize the room and its components."
                ) from e

            else:
                raise e

        mixed_signal = self._pyroom.mic_array.signals[0, :]
        mixed_signal = self.apply_snr(mixed_signal, -0.03)  # scale audio to max 1dB

        return mixed_signal

    def reset(self):
        """
        Resets the room acoustics simulator to its initial state.

        This method clears the internal pyroomacoustics room object and
        resets the simulator to its initial state. It's useful for
        starting a new simulation or clearing previous simulation data.

        Reset process:
        1. Delete the existing pyroomacoustics room object
        2. Clear the internal room reference
        3. Prepare for new simulation setup

        :raises RuntimeError: If reset fails due to internal state issues.
        """

        del self._pyroom
        self._pyroom = None

    @staticmethod
    def apply_snr(x, snr):
        """Scale an audio signal to a given maximum SNR."""
        dbfs = 10 ** (snr / 20)
        x *= dbfs / np.abs(x).max(initial=1e-15)
        return x
