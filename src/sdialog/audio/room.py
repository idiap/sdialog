"""
Room specification for the reverberation of :func:`sdialog.audio.synthesis.to_audio`.

A :class:`Room` describes a shoebox room (dimensions, absorption, microphone and speaker positions, microphone
directivity) and computes the impulse response from each speaker to the microphone with ``pyroomacoustics``
(image source method), installed with ``sdialog[audio]``, so no other dependency is needed. Pass it as ``reverb``
to ``dialog.to_audio()``; anything not given (e.g. the speaker positions) is drawn at random from the ``to_audio``
seed.

Example:

    .. code-block:: python

        from sdialog.audio import Room

        room = Room(dimensions=(6, 4, 3),                     # width (x), length (y), height (z) in meters
                    rt60=0.5,                                 # or materials={"walls": "brickwork", ...}
                    mic="between_speakers",
                    mic_pattern="cardioid", mic_direction="Alice",
                    speakers={"Alice": (2, 1.5, 1.2), "Bob": (4, 2.5, 1.6)})
        audio = dialog.to_audio(reverb=room)

        # Each speaker in their own room (e.g. both ends of a phone call)
        audio = dialog.to_audio(reverb={"Alice": Room(dimensions=(3, 3, 2.5)),
                                        "Bob": Room(dimensions=(10, 8, 4), rt60=1.0)},
                                channel="g711")
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Pawel Cyrta <pawel@cyrta.com>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import numpy as np

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Tuple, Union

Position = Tuple[float, float, float]

# Named microphone positions (besides 3D coordinates and speaker names, see Room.mic)
MIC_POSITIONS = ("ceiling", "wall", "between_speakers")
# Microphone directivity patterns: name -> pyroomacoustics directivity class (None = omnidirectional)
MIC_PATTERNS = {"omni": None, "cardioid": "Cardioid", "subcardioid": "SubCardioid",
                "hypercardioid": "HyperCardioid", "figure8": "FigureEight"}
# Room surfaces accepted in Room.materials ("walls" sets the four walls)
SURFACES = ("walls", "floor", "ceiling", "east", "west", "north", "south")
DEFAULT_MATERIALS = {"walls": "wooden_lining", "floor": "carpet_hairy", "ceiling": "ceiling_fibre_absorber"}
SPEAKER_HEIGHT = (1.2, 1.7)  # mouth height range (m), from sitting to standing
MIN_DISTANCE = 0.5  # minimum distance (m) between randomly placed speakers and the microphone


class Room(BaseModel):
    """
    A shoebox room used to simulate the reverberation of :func:`sdialog.audio.synthesis.to_audio`
    (``dialog.to_audio(reverb=Room(...))``).

    The absorption is given either as a target ``rt60`` or as ``materials`` (default: wooden walls, carpet
    and an absorbing ceiling). Speakers without a position are placed at random (mouth height, at least
    0.5 m from the microphone and from each other).

    :ivar dimensions: Width (x), length (y) and height (z) of the room, in meters.
    :vartype dimensions: Tuple[float, float, float]
    :ivar rt60: Target reverberation time in seconds (absorption computed with Sabine's formula).
    :vartype rt60: Optional[float]
    :ivar materials: Surface materials: a ``pyroomacoustics`` material name (e.g. "brickwork", "plasterboard")
                     or energy absorption coefficient for all surfaces, or a dictionary by surface ("walls",
                     "floor", "ceiling", or a single wall: "east", "west", "north", "south").
    :vartype materials: Optional[Union[str, float, Dict[str, Union[str, float]]]]
    :ivar mic: Microphone position: (x, y, z) coordinates, "ceiling" (center of the ceiling), "wall" (middle of
               the west wall, at standing height), "between_speakers" (midpoint between the speakers), or a
               speaker name or index (a lapel microphone on that speaker's chest).
    :vartype mic: Union[str, int, Tuple[float, float, float]]
    :ivar mic_pattern: Microphone directivity: "omni", "cardioid", "subcardioid", "hypercardioid" or "figure8".
    :vartype mic_pattern: str
    :ivar mic_direction: Where a directional microphone points: a speaker name or index, "speakers" (the
                         midpoint between the speakers, the default; up for the "between_speakers" microphone,
                         right below it), or (azimuth, colatitude) in degrees (azimuth from the x axis;
                         colatitude from the z axis, 90 = horizontal).
    :vartype mic_direction: Optional[Union[str, int, Tuple[float, float]]]
    :ivar speakers: Speaker positions (x, y, z), by speaker name or index (0 = first speaker of the dialogue).
    :vartype speakers: Dict[Union[int, str], Tuple[float, float, float]]
    :ivar max_order: Maximum reflection order of the image source method (default: from the RT60, up to 30,
                     or 17 with materials).
    :vartype max_order: Optional[int]
    :ivar air_absorption: Whether to simulate air absorption.
    :vartype air_absorption: bool
    :ivar ray_tracing: Whether to add ray tracing to the image source method (more realistic late
                       reverberation, slower).
    :vartype ray_tracing: bool

    Example:

        .. code-block:: python

            from sdialog.audio import Room

            room = Room(dimensions=(5, 4, 2.8), materials={"walls": "plasterboard", "floor": "linoleum_on_concrete"},
                        mic=(2.5, 2, 0.8), speakers={0: (1.5, 2, 1.2), 1: (3.5, 2, 1.2)})
            audio = dialog.to_audio(reverb=room, seed=0)
            audio.config["room"]  # the simulated room (positions included)

            rirs, info = room.compute_rirs(["Alice", "Bob"], sample_rate=16000)  # the impulse responses only
    """
    dimensions: Position
    rt60: Optional[float] = None
    materials: Optional[Union[str, float, Dict[str, Union[str, float]]]] = None
    mic: Union[Position, int, str] = "ceiling"
    mic_pattern: str = "omni"
    mic_direction: Optional[Union[Tuple[float, float], int, str]] = None
    speakers: Dict[Union[int, str], Position] = Field(default_factory=dict)
    max_order: Optional[int] = None
    air_absorption: bool = False
    ray_tracing: bool = False

    @model_validator(mode="after")
    def _check(self) -> "Room":
        if min(self.dimensions) <= 0:
            raise ValueError(f"Room dimensions must be positive, got {self.dimensions}")
        if self.rt60 is not None and self.materials is not None:
            raise ValueError("Give either `rt60` or `materials`, not both")
        if self.rt60 is not None and self.rt60 <= 0:
            raise ValueError(f"`rt60` must be positive, got {self.rt60}")
        if isinstance(self.materials, dict):
            invalid = [k for k in self.materials if k not in SURFACES]
            if invalid:
                raise ValueError(f"Invalid surface(s) {invalid} in `materials`. Valid surfaces: {list(SURFACES)}")
        if self.mic_pattern not in MIC_PATTERNS:
            raise ValueError(f"Unknown mic_pattern '{self.mic_pattern}'. Valid values: {list(MIC_PATTERNS)}")
        if self.mic_direction is not None and self.mic_pattern == "omni":
            raise ValueError("`mic_direction` needs a directional `mic_pattern` (e.g. 'cardioid')")
        if isinstance(self.mic, tuple):
            self._check_inside(self.mic, "Microphone")
        for name, position in self.speakers.items():
            self._check_inside(position, f"Speaker {name!r}")
        return self

    def _check_inside(self, position: Position, what: str):
        if not all(0 < p < d for p, d in zip(position, self.dimensions)):
            raise ValueError(f"{what} position {position} is outside the room {self.dimensions}")

    def _given_position(self, speakers: List[str], ix: int) -> Optional[np.ndarray]:
        position = self.speakers.get(speakers[ix], self.speakers.get(ix))
        return None if position is None else np.array(position, dtype=float)

    def _random_position(self, rng: np.random.Generator, avoid: List[np.ndarray]) -> np.ndarray:
        dims = np.array(self.dimensions, dtype=float)
        margin = min(0.5, dims[0] / 4, dims[1] / 4)
        z_max = max(min(SPEAKER_HEIGHT[1], dims[2] - 0.3), min(SPEAKER_HEIGHT[0], dims[2] / 2))
        z_min = min(SPEAKER_HEIGHT[0], z_max)
        for _ in range(100):
            position = np.array([rng.uniform(margin, dims[0] - margin),
                                 rng.uniform(margin, dims[1] - margin),
                                 rng.uniform(z_min, z_max)])
            if all(np.linalg.norm(position - p) >= MIN_DISTANCE for p in avoid):
                break
        return position

    def _mic_position(self, speakers: List[str], positions: List[np.ndarray]) -> np.ndarray:
        dims = np.array(self.dimensions, dtype=float)
        if isinstance(self.mic, tuple):
            return np.array(self.mic, dtype=float)
        if self.mic == "ceiling":
            return np.array([dims[0] / 2, dims[1] / 2, dims[2] - 0.1])
        if self.mic == "wall":
            return np.array([min(0.1, dims[0] / 2), dims[1] / 2, min(SPEAKER_HEIGHT[1], dims[2] - 0.1)])
        if self.mic == "between_speakers":  # e.g. a table microphone, a bit lower than the speakers' mouths
            middle = np.mean(positions, axis=0)
            return middle - np.array([0, 0, min(0.3, middle[2] / 2)])
        speaker = self._speaker_index(self.mic, speakers, "mic")
        return positions[speaker] - np.array([0, 0, min(0.3, positions[speaker][2] / 2)])

    def _mic_direction(self, mic: np.ndarray, speakers: List[str], positions: List[np.ndarray]) -> Tuple[float, float]:
        """Returns the (azimuth, colatitude) in degrees the microphone points to."""
        if isinstance(self.mic_direction, tuple):
            return float(self.mic_direction[0]), float(self.mic_direction[1])
        if self.mic_direction is None or self.mic_direction == "speakers":
            target = np.mean(positions, axis=0)
        else:
            target = positions[self._speaker_index(self.mic_direction, speakers, "mic_direction")]
        v = target - mic
        if np.linalg.norm(v) < 1e-3:  # target on the microphone: point up
            return 0.0, 0.0
        azimuth = np.degrees(np.arctan2(v[1], v[0]))
        colatitude = np.degrees(np.arccos(np.clip(v[2] / np.linalg.norm(v), -1, 1)))
        return float(azimuth), float(colatitude)

    @staticmethod
    def _speaker_index(value: Union[int, str], speakers: List[str], param: str) -> int:
        if isinstance(value, int) and 0 <= value < len(speakers):
            return value
        if value in speakers:
            return speakers.index(value)
        raise ValueError(f"Invalid `{param}` value {value!r}: not a speaker of the dialogue ({speakers})"
                         + (f" nor one of {list(MIC_POSITIONS)}" if param == "mic" else ""))

    def _absorption(self, pra) -> Tuple[object, int]:
        """Returns the pyroomacoustics materials and the maximum reflection order."""
        if self.rt60 is not None:
            try:
                absorption, max_order = pra.inverse_sabine(self.rt60, list(self.dimensions))
            except ValueError as e:
                raise ValueError(f"An RT60 of {self.rt60} s cannot be reached in a room of {self.dimensions} m "
                                 "(the room is too large for such a short reverberation time)") from e
            return pra.Material(absorption), min(max_order, 30)
        if self.materials is None or isinstance(self.materials, dict):
            surfaces = {**DEFAULT_MATERIALS, **(self.materials or {})}
            faces = {wall: surfaces.get(wall, surfaces["walls"]) for wall in ("east", "west", "north", "south")}
            return pra.make_materials(floor=surfaces["floor"], ceiling=surfaces["ceiling"], **faces), 17
        return pra.Material(self.materials), 17

    def compute_rirs(self,
                     speakers: List[str],
                     sample_rate: int,
                     rng: Optional[np.random.Generator] = None) -> Tuple[List[np.ndarray], dict]:
        """
        Simulates the room and computes the impulse response from each speaker to the microphone.

        :param speakers: Names of the speakers to place, in order (positions given by index refer to this list).
        :type speakers: List[str]
        :param sample_rate: Sample rate of the returned impulse responses.
        :type sample_rate: int
        :param rng: Random generator used to place the speakers without a given position.
        :type rng: Optional[np.random.Generator]
        :return: The impulse response of each speaker, and a dictionary describing the simulated room
                 (dimensions, absorption, microphone and speaker positions).
        :rtype: Tuple[List[np.ndarray], dict]
        :raises ValueError: If the RT60 cannot be reached in the room or a speaker name is not valid.

        Example:

            .. code-block:: python

                rirs, info = Room(dimensions=(6, 4, 3), rt60=0.6).compute_rirs(["Alice", "Bob"], 16000)
        """
        try:
            import pyroomacoustics as pra
        except ImportError as e:
            raise ImportError("Room simulation requires pyroomacoustics: `pip install pyroomacoustics`.") from e
        rng = rng if rng is not None else np.random.default_rng()

        # Speakers with a given position first, so random ones avoid them (and a fixed microphone)
        positions = [self._given_position(speakers, ix) for ix in range(len(speakers))]
        fixed_mic = isinstance(self.mic, tuple) or self.mic in ("ceiling", "wall")
        avoid = [p for p in positions if p is not None]
        if fixed_mic:
            avoid.append(self._mic_position(speakers, positions))
        for ix, position in enumerate(positions):
            if position is None:
                positions[ix] = self._random_position(rng, avoid)
                avoid.append(positions[ix])
        mic = self._mic_position(speakers, positions)

        materials, max_order = self._absorption(pra)
        room = pra.ShoeBox(list(self.dimensions), fs=sample_rate, materials=materials,
                           max_order=self.max_order if self.max_order is not None else max_order,
                           air_absorption=self.air_absorption, ray_tracing=self.ray_tracing)
        for position in positions:
            room.add_source(position.tolist())
        direction = None
        if MIC_PATTERNS[self.mic_pattern] is None:
            room.add_microphone(mic.tolist())
        else:
            direction = self._mic_direction(mic, speakers, positions)
            pattern = getattr(pra.directivities, MIC_PATTERNS[self.mic_pattern])
            orientation = pra.directivities.DirectionVector(azimuth=direction[0], colatitude=direction[1],
                                                            degrees=True)
            room.add_microphone(mic.tolist(), directivity=pattern(orientation))
        room.compute_rir()

        rirs = [np.asarray(room.rir[0][ix], dtype=np.float32) for ix in range(len(speakers))]
        materials = DEFAULT_MATERIALS if self.rt60 is None and self.materials is None else self.materials
        info = {"type": "room", "dimensions": [round(float(d), 2) for d in self.dimensions], "rt60": self.rt60,
                "materials": materials,
                "microphone": mic.round(2).tolist(), "mic_pattern": self.mic_pattern,
                "mic_direction": None if direction is None else [round(d, 1) for d in direction],
                "speakers": list(speakers), "sources": [p.round(2).tolist() for p in positions]}
        return rirs, info
