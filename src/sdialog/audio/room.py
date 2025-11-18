"""
This module provides comprehensive room specification classes for acoustics simulation.

The module includes classes for defining 3D room environments, spatial positioning,
acoustic materials, and audio source management. These classes enable realistic
room acoustics simulation with support for complex room geometries, furniture
placement, and acoustic material modeling.

Key Components:

  - Position3D: 3D coordinate positioning system
  - Dimensions3D: 3D room dimensions and volume calculations
  - Room: Main room class with acoustics simulation support
  - AudioSource: Audio source positioning and characteristics
  - Furniture: 3D furniture models for room simulation
  - Directivity: Microphone and speaker directivity patterns
  - RoomPosition: Room positioning and orientation system

Room Acoustics Features:

  - 3D room geometry with customizable dimensions
  - Acoustic material modeling for walls, floor, and ceiling
  - Furniture placement and acoustic obstacle modeling
  - Microphone and speaker positioning with directivity
  - Room acoustics simulation integration
  - Spatial audio source management

Example:

    .. code-block:: python

        from sdialog.audio.room import Room, Position3D, Dimensions3D
        from sdialog.audio.utils import RoomMaterials, WallMaterial

        # Create room dimensions
        dimensions = Dimensions3D(width=5.0, length=4.0, height=3.0)

        # Create room materials
        materials = RoomMaterials(
            walls=WallMaterial.WOODEN_LINING,
            floor=FloorMaterial.CARPET_HAIRY,
            ceiling=CeilingMaterial.FIBRE_ABSORBER
        )

        # Create room with microphone position
        room = Room(
            dimensions=dimensions,
            materials=materials,
            mic_position=Position3D(2.5, 2.0, 1.5)
        )
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Pawel Cyrta <pawel@cyrta.com>
# SPDX-License-Identifier: MIT
import time
import math
import hashlib
import numpy as np
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field, PrivateAttr
from typing import Dict, Optional, Tuple, List, Any

from sdialog.audio.utils import logger, BodyPosture, Furniture, RoomMaterials, SpeakerSide, Role


@dataclass
class Position3D:
    """
    3D position coordinates for spatial positioning in room acoustics simulation.

    This class represents a 3D position in meters within a room environment,
    providing coordinate-based positioning for speakers, microphones, furniture,
    and other objects in room acoustics simulation. It includes utility methods
    for distance calculations, coordinate transformations, and data conversion.

    Key Features:

      - 3D coordinate system (x, y, z) in meters
      - Distance calculations in 2D or 3D space
      - Coordinate validation and error handling
      - Data conversion to various formats (array, list)
      - Spatial positioning utilities

    Coordinate System:
        - X-axis: Horizontal position (width)
        - Y-axis: Depth position (length)
        - Z-axis: Vertical position (height)

    :ivar x: X-coordinate in meters (horizontal position).
    :vartype x: float
    :ivar y: Y-coordinate in meters (depth position).
    :vartype y: float
    :ivar z: Z-coordinate in meters (height position).
    :vartype z: float
    """

    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Initializes a 3D position with the specified coordinates.

        :param x: X-coordinate in meters (default: 0.0).
        :type x: float
        :param y: Y-coordinate in meters (default: 0.0).
        :type y: float
        :param z: Z-coordinate in meters (default: 0.0).
        :type z: float
        """

        # Check if the coordinates are valid
        if x < 0 or y < 0 or z < 0:
            raise ValueError("Coordinates must be non-negative")

        self.x = x
        self.y = y
        self.z = z

    def __post_init__(self):
        """
        Validates coordinates after initialization.

        :raises ValueError: If any coordinate is negative.
        """
        if any(coord < 0 for coord in [self.x, self.y, self.z]):
            raise ValueError("Coordinates must be non-negative")

    def __str__(self):
        """
        Returns a string representation of the position.

        :return: String representation in format "pos: [x, y, z]".
        :rtype: str
        """
        return f"pos: [{self.x}, {self.y}, {self.z}]"

    def to_array(self) -> np.ndarray:
        """
        Converts the position to a numpy array.

        :return: Numpy array containing [x, y, z] coordinates.
        :rtype: np.ndarray
        """
        return np.array([self.x, self.y, self.z])

    def to_list(self):
        """
        Converts the position to a Python list.

        :return: List containing [x, y, z] coordinates.
        :rtype: List[float]
        """
        return [self.x, self.y, self.z]

    def distance_to(
        self,
        other_position: "Position3D",
        dimensions: int = 3
    ) -> float:
        """
        Calculates the Euclidean distance to another position.

        This method computes the straight-line distance between this position
        and another position, supporting both 2D and 3D distance calculations.

        :param other_position: The other position to calculate distance to.
        :type other_position: Position3D
        :param dimensions: Number of dimensions for distance calculation (2 or 3, default: 3).
        :type dimensions: int
        :return: The Euclidean distance in meters.
        :rtype: float
        :raises ValueError: If dimensions is not 2 or 3.
        """
        if dimensions == 2:
            return (
                (self.x - other_position.x) ** 2
                + (self.y - other_position.y) ** 2
            ) ** 0.5
        elif dimensions == 3:
            return (
                (self.x - other_position.x) ** 2
                + (self.y - other_position.y) ** 2
                + (self.z - other_position.z) ** 2
            ) ** 0.5
        else:
            raise ValueError(f"Invalid dimensions: {dimensions}")

    @classmethod
    def from_list(cls, position_list: List[float]) -> "Position3D":
        """
        Creates a Position3D from a list of coordinates.

        :param position_list: List containing [x, y, z] coordinates.
        :type position_list: List[float]
        :return: A new Position3D object.
        :rtype: Position3D
        :raises ValueError: If the list doesn't contain exactly 3 coordinates.
        """
        if len(position_list) != 3:
            raise ValueError("Position must have exactly 3 coordinates [x, y, z]")
        return cls(x=position_list[0], y=position_list[1], z=position_list[2])


@dataclass
class Dimensions3D:
    """
    3D dimensions for room geometry in room acoustics simulation.

    This class represents the 3D dimensions of a room in meters, providing
    width, length, and height measurements for room geometry definition.
    It includes validation, volume calculations, and data conversion utilities.

    Key Features:

      - 3D dimension system (width, length, height) in meters
      - Dimension validation and error handling
      - Volume calculation for room acoustics
      - Data conversion to various formats (list)
      - Room geometry utilities

    Dimension System:
        - Width: X-axis dimension (horizontal)
        - Length: Y-axis dimension (depth)
        - Height: Z-axis dimension (vertical)

    :ivar width: Width in meters (x-axis dimension).
    :vartype width: float
    :ivar length: Length in meters (y-axis dimension).
    :vartype length: float
    :ivar height: Height in meters (z-axis dimension).
    :vartype height: float
    """

    width: float  # x-axis
    length: float  # y-axis
    height: float  # z-axis

    def __post_init__(self):
        """
        Validates dimensions after initialization.

        :raises ValueError: If any dimension is not positive.
        """
        if any(dim <= 0 for dim in [self.width, self.length, self.height]):
            raise ValueError("All dimensions must be positive")

    def __str__(self):
        """
        Returns a string representation of the dimensions.

        :return: String representation in format "dim: [width, length, height]".
        :rtype: str
        """
        return f"dim: [{self.width}, {self.length}, {self.height}]"

    def to_list(self):
        """
        Converts the dimensions to a Python list.

        :return: List containing [width, length, height] dimensions.
        :rtype: List[float]
        """
        return [self.width, self.length, self.height]

    @property
    def volume(self) -> float:
        """
        Calculates the volume of the room.

        This property computes the total volume of the room by multiplying
        width, length, and height. The volume is used in room acoustics
        calculations for reverberation time and acoustic modeling.

        :return: The room volume in cubic meters.
        :rtype: float
        """
        return self.width * self.length * self.height

    @property
    def floor_area(self) -> float:
        return self.width * self.length

    def __len__(self):
        return 3

    def __iter__(self):
        return iter([self.length, self.width, self.height])

    def __getitem__(self, index):
        return [self.length, self.width, self.height][index]


class SoundEventPosition(str, Enum):
    BACKGROUND = "no_type"  # background -
    NOT_DEFINED = "soundevent-not_defined"
    DEFINED = "soundevent-defined"


class RoomPosition(str, Enum):
    """
    Room placement locations in the world
    """
    CENTER = "room-center"
    TOP_LEFT = "room-top_left"
    TOP_RIGHT = "room-top_right"
    BOTTOM_LEFT = "room-bottom_left"
    BOTTOM_RIGHT = "room-bottom_right"


class MicrophonePosition(str, Enum):
    """
    Different microphone placement options
    """

    DESK_SMARTPHONE = "desk_smartphone"
    MONITOR = "monitor"
    WALL_MOUNTED = "wall_mounted"
    CEILING_CENTERED = "ceiling_centered"
    CHEST_POCKET_SPEAKER_1 = "chest_pocket_speaker_1"
    CHEST_POCKET_SPEAKER_2 = "chest_pocket_speaker_2"
    MIDDLE_SPEAKERS = "middle_speakers"
    CUSTOM = "custom"


class DirectivityType(str, Enum):
    """
    Type of the directivity for a speaker microphone
    """
    CUSTOM = "custom"

    OMNIDIRECTIONAL = "omnidirectional"

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    NORTH_EAST = "north_east"
    NORTH_WEST = "north_west"
    SOUTH_EAST = "south_east"
    SOUTH_WEST = "south_west"

    SPEAKER_1 = Role.SPEAKER_1.value
    SPEAKER_2 = Role.SPEAKER_2.value
    MIDDLE_SPEAKERS = "middle_speakers"


class MicrophoneDirectivity(BaseModel):
    """
    Represents a directivity of an audio source
    """
    azimuth: int = 0
    colatitude: int = 0
    gain: float = 1.0

    def to_pyroomacoustics(self):
        """
        Convert the microphone directivity to a pyroomacoustics directivity.
        """
        import pyroomacoustics as pra
        from pyroomacoustics import DirectionVector
        return pra.directivities.Cardioid(
            DirectionVector(
                azimuth=self.azimuth,
                colatitude=self.colatitude
            ),
            gain=self.gain
        )


class AudioSource(BaseModel):
    """
    Represents an object, speaker that makes sounds in the room
    """

    name: str = ""
    position: str = "no_type"
    snr: float = 0.0  # dB SPL
    source_file: Optional[str] = "no_file"  # audio file e.g wav
    directivity: Optional[str] = "omnidirectional"
    _position3d: Optional[Position3D] = PrivateAttr(default=None)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @property
    def x(self) -> float:
        if self._position3d is None:
            raise ValueError("_position3d is not set")
        return self._position3d.x

    @property
    def y(self) -> float:
        if self._position3d is None:
            raise ValueError("_position3d is not set")
        return self._position3d.y

    @property
    def z(self) -> float:
        if self._position3d is None:
            raise ValueError("_position3d is not set")
        return self._position3d.z

    def distance_to(self, other_position: Tuple[float, float, float]) -> float:
        return (
            (self.x - other_position[0]) ** 2
            + (self.y - other_position[1]) ** 2
            + (self.z - other_position[2]) ** 2
        ) ** 0.5

# related to https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py


def get_room_id():
    """
    Get the room id based on the time in nanoseconds.
    """
    return str(int(time.time_ns()))


class Room(BaseModel):
    """
    Main room class for comprehensive room acoustics simulation.

    This class represents a complete room environment for acoustics simulation,
    including 3D geometry, acoustic materials, furniture placement, microphone
    positioning, and audio source management. It provides the foundation for
    realistic room acoustics modeling and simulation.

    Key Features:

      - 3D room geometry with customizable dimensions
      - Acoustic material modeling for all surfaces
      - Furniture placement and acoustic obstacle modeling
      - Microphone positioning with directivity patterns
      - Audio source management and positioning
      - Room acoustics simulation integration
      - Spatial audio processing support

    Room Components:

      - Dimensions: 3D room geometry (width, length, height)
      - Materials: Acoustic properties of walls, floor, and ceiling
      - Furniture: 3D objects that affect acoustics
      - Microphone: Recording position and directivity
      - Audio Sources: Speaker positions and characteristics
      - Acoustics: Reverberation and acoustic modeling

    :ivar id: Unique identifier for the room.
    :vartype id: str
    :ivar name: Human-readable name for the room.
    :vartype name: str
    :ivar description: Description of the room and its purpose.
    :vartype description: str
    :ivar dimensions: 3D room dimensions in meters.
    :vartype dimensions: Dimensions3D
    :ivar mic_position: Microphone position type (ceiling, floor, etc.).
    :vartype mic_position: MicrophonePosition
    :ivar mic_position_3d: 3D microphone position for acoustics simulation.
    :vartype mic_position_3d: Position3D
    :ivar directivity_type: Microphone directivity pattern type.
    :vartype directivity_type: Optional[DirectivityType]
    :ivar microphone_directivity: Microphone directivity configuration.
    :vartype microphone_directivity: Optional[MicrophoneDirectivity]
    :ivar furnitures: Dictionary of furniture objects in the room.
    :vartype furnitures: dict[str, Furniture]
    :ivar materials: Acoustic materials for room surfaces.
    :vartype materials: RoomMaterials
    :ivar reverberation_time_ratio: Reverberation time ratio for acoustics.
    :vartype reverberation_time_ratio: Optional[float]
    :ivar speakers_positions: Dictionary mapping speaker names to positions.
    :vartype speakers_positions: dict[str, Position3D]
    """
    id: str = Field(default_factory=get_room_id)
    name: str = "Room"
    description: str = ""

    dimensions: Dimensions3D = Field(default_factory=lambda: Dimensions3D(2, 2.5, 3))

    mic_position: MicrophonePosition = MicrophonePosition.CEILING_CENTERED
    mic_position_3d: Position3D = None
    directivity_type: Optional[DirectivityType] = DirectivityType.OMNIDIRECTIONAL
    microphone_directivity: Optional[MicrophoneDirectivity] = None

    # Furniture available in the room
    furnitures: dict[str, Furniture] = {}

    materials: RoomMaterials = RoomMaterials()
    reverberation_time_ratio: Optional[float] = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    speakers_positions: dict[str, Position3D] = {}  # dict[speaker_name, speaker_position]

    def directivity_type_to_azimuth_colatitude(self, type: DirectivityType) -> Tuple[int, int]:
        """
        Converts a directivity type to azimuth and colatitude coordinates.

        This method maps directivity types to their corresponding azimuth
        and colatitude values for microphone directivity configuration
        in room acoustics simulation.

        :param type: The directivity type to convert.
        :type type: DirectivityType
        :return: A tuple containing (azimuth, colatitude) in degrees.
        :rtype: Tuple[int, int]
        """

        if type == DirectivityType.OMNIDIRECTIONAL:
            return 0, 0

        elif type == DirectivityType.NORTH:
            return 0, 90
        elif type == DirectivityType.SOUTH:
            return 180, 90
        elif type == DirectivityType.EAST:
            return 90, 90
        elif type == DirectivityType.WEST:
            return -90, 90

        elif type == DirectivityType.NORTH_EAST:
            return 45, 90
        elif type == DirectivityType.NORTH_WEST:
            return -45, 90
        elif type == DirectivityType.SOUTH_EAST:
            return 135, 90
        elif type == DirectivityType.SOUTH_WEST:
            return -135, 90

        elif type in [DirectivityType.SPEAKER_1, DirectivityType.SPEAKER_2]:
            """
            The microphone will aim at the speaker.
            """

            if type.value not in self.speakers_positions:
                raise ValueError((
                    f"Speaker {type.value} is not set, the microphone directivity can't be computed."
                    f"Available speakers: {', '.join(self.speakers_positions.keys())}"
                ))

            speaker_position = self.speakers_positions[type.value]

            azimuth = math.atan2(
                speaker_position.y - self.mic_position_3d.y,
                speaker_position.x - self.mic_position_3d.x
            )

            colatitude = math.atan2(
                speaker_position.z - self.mic_position_3d.z,
                math.sqrt(
                    (speaker_position.x - self.mic_position_3d.x)**2
                    + (speaker_position.y - self.mic_position_3d.y)**2
                )
            )

            # Ensure colatitude is in range [0, π] as required by pyroomacoustics
            if colatitude < 0:
                colatitude += math.pi

            return int(azimuth * 180 / math.pi), int(colatitude * 180 / math.pi)

        elif type == DirectivityType.MIDDLE_SPEAKERS:
            """
            The microphone will aim at the position between the two speakers.
            """

            if Role.SPEAKER_1 not in self.speakers_positions or Role.SPEAKER_2 not in self.speakers_positions:
                raise ValueError("Speakers positions are not set, the microphone directivity can't be computed")

            speaker_1_position = self.speakers_positions[Role.SPEAKER_1]
            speaker_2_position = self.speakers_positions[Role.SPEAKER_2]

            # Calculer le point milieu entre les deux speakers
            middle_x = (speaker_1_position.x + speaker_2_position.x) / 2
            middle_y = (speaker_1_position.y + speaker_2_position.y) / 2
            middle_z = (speaker_1_position.z + speaker_2_position.z) / 2

            # Calculer l'angle depuis le microphone vers le point milieu
            azimuth = math.atan2(
                middle_y - self.mic_position_3d.y,
                middle_x - self.mic_position_3d.x
            )

            colatitude = math.atan2(
                middle_z - self.mic_position_3d.z,
                math.sqrt(
                    (middle_x - self.mic_position_3d.x)**2
                    + (middle_y - self.mic_position_3d.y)**2
                )
            )

            # Ensure colatitude is in range [0, π] as required by pyroomacoustics
            if colatitude < 0:
                colatitude += math.pi

            return int(azimuth * 180 / math.pi), int(colatitude * 180 / math.pi)

        raise ValueError(f"Directivity type {type} is not supported")

    def room_position_to_position3d(
        self,
        position: RoomPosition
    ) -> Position3D:
        if position == RoomPosition.CENTER:
            return self.get_roof_center()
        elif position == RoomPosition.TOP_LEFT:
            return self.get_top_left_corner()
        elif position == RoomPosition.TOP_RIGHT:
            return self.get_top_right_corner()
        elif position == RoomPosition.BOTTOM_LEFT:
            return self.get_bottom_left_corner()
        elif position == RoomPosition.BOTTOM_RIGHT:
            return self.get_bottom_right_corner()

    def place_speaker(self, speaker_name: str, position: Position3D):
        """
        Place a speaker in the room.
        """

        if speaker_name not in [Role.SPEAKER_1, Role.SPEAKER_2]:
            raise ValueError(f"Speaker name {speaker_name} is not valid, the speaker wasn't placed")

        # Check is the coordinates are valid
        if not self._is_position_valid(position.x, position.y):
            raise ValueError(f"Position {position} is not valid, the speaker wasn't placed")

        self.speakers_positions[speaker_name] = position

        if (
            self.mic_position == MicrophonePosition.MIDDLE_SPEAKERS
            and Role.SPEAKER_1 in self.speakers_positions
            and Role.SPEAKER_2 in self.speakers_positions
        ):
            self.mic_position_3d = Position3D(
                x=(self.speakers_positions[Role.SPEAKER_1].x + self.speakers_positions[Role.SPEAKER_2].x) / 2,
                y=(self.speakers_positions[Role.SPEAKER_1].y + self.speakers_positions[Role.SPEAKER_2].y) / 2,
                z=BodyPosture.STANDING.value - 0.3
            )

    def place_speaker_around_furniture(
        self,
        speaker_name: str,
        furniture_name: str = "center",
        max_distance: float = 0.3,
        side: Optional[str] = None
    ):
        """
        Place a speaker position around a furniture.

        Args:
            speaker_name: Name of the speaker to place
            furniture_name: Name of the furniture to place around
            max_distance: Maximum distance from the furniture edge (in meters)
            side: Specific side to place the speaker ("front", "back", "left", "right")
        """

        if furniture_name not in self.furnitures:
            raise ValueError(f"Furniture {furniture_name} not found in the room")

        if side is not None and side not in [SpeakerSide.FRONT, SpeakerSide.BACK, SpeakerSide.LEFT, SpeakerSide.RIGHT]:
            raise ValueError(f"Side {side} is not valid, the speaker wasn't placed")

        # Get the furniture
        furniture = self.furnitures[furniture_name]

        # Get position based on whether a specific side is requested
        if side is not None:
            position = self._get_position_on_furniture_side(furniture, side, max_distance)
        else:
            # Get a random position around the furniture (considering the furniture 2D dimensions)
            # Position validation is already handled within _get_random_position_around_furniture
            position = self._get_random_position_around_furniture(furniture, max_distance)

        # Add the speaker to the room
        self.speakers_positions[speaker_name] = position

        if (
            self.mic_position == MicrophonePosition.MIDDLE_SPEAKERS
            and Role.SPEAKER_1 in self.speakers_positions
            and Role.SPEAKER_2 in self.speakers_positions
        ):
            self.mic_position_3d = Position3D(
                x=(self.speakers_positions[Role.SPEAKER_1].x + self.speakers_positions[Role.SPEAKER_2].x) / 2,
                y=(self.speakers_positions[Role.SPEAKER_1].y + self.speakers_positions[Role.SPEAKER_2].y) / 2,
                z=BodyPosture.STANDING.value - 0.3
            )

    def _clamp_position_to_room_bounds(self, x: float, y: float, z: float) -> Position3D:
        """
        Ensure position is within room bounds with safety margin.

        Args:
            x, y, z: Position coordinates

        Returns:
            Position3D: Position clamped to room bounds
        """
        # Use adaptive margin based on room size
        margin = min(0.2, min(self.dimensions.width, self.dimensions.length) * 0.1)  # 20cm or 10% of smallest dimension
        clamped_x = max(margin, min(x, self.dimensions.width - margin))
        clamped_y = max(margin, min(y, self.dimensions.length - margin))
        clamped_z = max(0.1, min(z, self.dimensions.height - 0.05))  # Smaller top margin
        return Position3D(clamped_x, clamped_y, clamped_z)

    def _is_position_valid(self, x: float, y: float) -> bool:
        """
        Check if a position is valid (no collision with furniture and within room bounds).

        Args:
            x, y: Position coordinates

        Returns:
            bool: True if position is valid, False otherwise
        """
        # Use adaptive margin based on room size
        margin = min(0.2, min(self.dimensions.width, self.dimensions.length) * 0.1)  # 20cm or 10% of smallest dimension

        # Check if position is within room bounds
        if (
            x < margin or x > self.dimensions.width - margin
            or y < margin or y > self.dimensions.length - margin
        ):
            return False

        # Check for collision with any furniture
        for furniture_name, furniture in self.furnitures.items():
            if self._is_position_colliding_with_furniture(x, y, furniture):
                return False

        return True

    def _is_position_colliding_with_furniture(self, x: float, y: float, furniture: Furniture) -> bool:
        """
        Check if a position collides with a specific furniture.

        Args:
            x, y: Position coordinates
            furniture: The furniture to check collision with

        Returns:
            bool: True if position collides with furniture, False otherwise
        """
        # Check if position is within furniture bounds
        return (
            furniture.x <= x <= furniture.x + furniture.width
            and furniture.y <= y <= furniture.y + furniture.depth
        )

    def _get_random_position_around_furniture(
        self,
        furniture: Furniture,
        max_distance: float = 0.3
    ) -> Position3D:
        """
        Get a random position around a furniture.

        Args:
            furniture: The furniture object to position around
            max_distance: Maximum distance from the furniture edge (in meters)

        Returns:
            Position3D: A random position around the furniture
        """
        import random

        # Calculate the area around the furniture where we can place the position
        # We need to consider the furniture dimensions plus the max_distance
        min_x = furniture.x - max_distance
        max_x = furniture.x + furniture.width + max_distance
        min_y = furniture.y - max_distance
        max_y = furniture.y + furniture.depth + max_distance

        # Use adaptive margin based on room size
        margin = min(0.2, min(self.dimensions.width, self.dimensions.length) * 0.1)  # 20cm or 10% of smallest dimension

        # Ensure the position is within room bounds
        min_x = max(margin, min_x)
        max_x = min(self.dimensions.width - margin, max_x)
        min_y = max(margin, min_y)
        max_y = min(self.dimensions.length - margin, max_y)

        # Generate random position
        attempts = 0
        max_attempts = 9999

        while attempts < max_attempts:
            # Generate random coordinates
            random_x = random.uniform(min_x, max_x)
            random_y = random.uniform(min_y, max_y)

            # Clamp position to room bounds first
            clamped_position = self._clamp_position_to_room_bounds(random_x, random_y, 0.0)
            clamped_x, clamped_y = clamped_position.x, clamped_position.y

            # Check if the position is outside the furniture (not overlapping)
            # Position is outside furniture if it's not within furniture bounds
            is_outside_furniture = (
                clamped_x < furniture.x
                or clamped_x > furniture.x + furniture.width
                or clamped_y < furniture.y
                or clamped_y > furniture.y + furniture.depth
            )

            if is_outside_furniture:
                # Check if position is within max_distance from furniture edge
                # Calculate distance to furniture edge
                distance_to_furniture = self._calculate_distance_to_furniture_edge(
                    clamped_x, clamped_y, furniture
                )

                if distance_to_furniture <= max_distance:
                    # Check if position is valid (no collision with other furniture and within room bounds)
                    if self._is_position_valid(clamped_x, clamped_y):
                        # Use human standing height instead of furniture height for more realistic positioning
                        z_position = min(BodyPosture.STANDING.value, self.dimensions.height - 0.3)  # Standing height
                        return Position3D(clamped_x, clamped_y, z_position)

            attempts += 1

        # Fallback: if we can't find a valid position, place it at a corner of the furniture
        # with some offset
        fallback_x = furniture.x + furniture.width + 0.1
        fallback_y = furniture.y + furniture.depth + 0.1
        fallback_z = min(BodyPosture.STANDING.value, self.dimensions.height - 0.3)  # Standing height with margin

        # Ensure fallback is within room bounds using the clamp method
        return self._clamp_position_to_room_bounds(fallback_x, fallback_y, fallback_z)

    def _get_position_on_furniture_side(
        self,
        furniture: Furniture,
        side: str,
        max_distance: float = 0.3
    ) -> Position3D:
        """
        Get a position on a specific side of a furniture.

        Args:
            furniture: The furniture object to position around
            side: The side to place the speaker ("front", "back", "left", "right")
            max_distance: Maximum distance from the furniture edge (in meters)

        Returns:
            Position3D: A position on the specified side of the furniture
        """
        import random

        # Define the sides based on furniture orientation
        # Assuming furniture is oriented with front facing positive Y direction
        furniture_center_x = furniture.x + furniture.width / 2
        furniture_center_y = furniture.y + furniture.depth / 2

        # Calculate position ranges for each side - staying in "corridors"
        if side == SpeakerSide.BACK:
            # back side (positive Y direction) - X can vary, Y is fixed corridor
            x_min = furniture.x
            x_max = furniture.x + furniture.width
            y_min = furniture.y + furniture.depth
            y_max = furniture.y + furniture.depth + max_distance

        elif side == SpeakerSide.FRONT:
            # front side (negative Y direction) - X can vary, Y is fixed corridor
            x_min = furniture.x
            x_max = furniture.x + furniture.width
            y_min = furniture.y - max_distance
            y_max = furniture.y

        elif side == SpeakerSide.LEFT:
            # Left side (negative X direction) - Y can vary, X is fixed corridor
            x_min = furniture.x - max_distance
            x_max = furniture.x
            y_min = furniture.y
            y_max = furniture.y + furniture.depth

        elif side == SpeakerSide.RIGHT:
            # Right side (positive X direction) - Y can vary, X is fixed corridor
            x_min = furniture.x + furniture.width
            x_max = furniture.x + furniture.width + max_distance
            y_min = furniture.y
            y_max = furniture.y + furniture.depth

        else:
            raise ValueError(f"Invalid side: {side}")

        # Use adaptive margin based on room size
        margin = min(0.2, min(self.dimensions.width, self.dimensions.length) * 0.1)  # 20cm or 10% of smallest dimension

        # Ensure the position is within room bounds
        x_min = max(margin, x_min)
        x_max = min(self.dimensions.width - margin, x_max)
        y_min = max(margin, y_min)
        y_max = min(self.dimensions.length - margin, y_max)

        # Generate random position within the specified side corridor
        attempts = 0
        max_attempts = 9999

        while attempts < max_attempts:
            # Generate random coordinates within the side corridor
            random_x = random.uniform(x_min, x_max)
            random_y = random.uniform(y_min, y_max)

            # Clamp position to room bounds
            clamped_position = self._clamp_position_to_room_bounds(random_x, random_y, 0.0)
            clamped_x, clamped_y = clamped_position.x, clamped_position.y

            # Check if position is valid (no collision with other furniture and within room bounds)
            if self._is_position_valid(clamped_x, clamped_y):
                # Use furniture height for z coordinate (standing height)
                z_position = furniture.get_top_z() + 0.1  # Slightly above furniture
                return Position3D(clamped_x, clamped_y, z_position)

            attempts += 1

        # Fallback: place at the center of the side with minimum distance
        if side == SpeakerSide.BACK:
            fallback_x = furniture_center_x
            fallback_y = furniture.y + furniture.depth + 0.1
        elif side == SpeakerSide.FRONT:
            fallback_x = furniture_center_x
            fallback_y = furniture.y - 0.1
        elif side == SpeakerSide.LEFT:
            fallback_x = furniture.x - 0.1
            fallback_y = furniture_center_y
        elif side == SpeakerSide.RIGHT:
            fallback_x = furniture.x + furniture.width + 0.1
            fallback_y = furniture_center_y

        fallback_z = min(BodyPosture.STANDING.value, self.dimensions.height - 0.3)  # Standing height with margin

        # Ensure fallback is within room bounds
        return self._clamp_position_to_room_bounds(fallback_x, fallback_y, fallback_z)

    def _calculate_distance_to_furniture_edge(self, x: float, y: float, furniture: Furniture) -> float:
        """
        Calculate the minimum distance from a point to the edge of a furniture.

        Args:
            x, y: Point coordinates
            furniture: The furniture object

        Returns:
            float: Minimum distance to furniture edge
        """
        # Calculate distance to each edge of the furniture rectangle
        distance_to_left = abs(x - furniture.x)
        distance_to_right = abs(x - (furniture.x + furniture.width))
        distance_to_top = abs(y - furniture.y)
        distance_to_bottom = abs(y - (furniture.y + furniture.depth))

        # If point is inside furniture, calculate distance to nearest edge
        if (
            furniture.x <= x <= furniture.x + furniture.width
            and furniture.y <= y <= furniture.y + furniture.depth
        ):
            # Point is inside furniture, return distance to nearest edge
            return min(distance_to_left, distance_to_right, distance_to_top, distance_to_bottom)
        else:
            # Point is outside furniture, calculate distance to nearest corner/edge
            # Distance to nearest point on furniture rectangle
            dx = max(0, max(furniture.x - x, x - (furniture.x + furniture.width)))
            dy = max(0, max(furniture.y - y, y - (furniture.y + furniture.depth)))
            return (dx**2 + dy**2)**0.5

    def add_speaker(self, speaker_name: str, position: Position3D):
        """
        Add a speaker to the room.
        """
        pass

    def get_top_left_corner(self) -> Position3D:
        return Position3D(
            x=self.dimensions.width * 0.01,   # Top-left: x=0.01, y=0.01
            y=self.dimensions.length * 0.01,
            z=self.dimensions.height - 0.5     # 50cm margin from ceiling
        )

    def get_bottom_left_corner(self) -> Position3D:
        return Position3D(
            x=self.dimensions.width * 0.01,   # Bottom-left: x=0.01, y=0.99
            y=self.dimensions.length * 0.99,
            z=self.dimensions.height - 0.5     # 50cm margin from ceiling
        )

    def get_top_right_corner(self) -> Position3D:
        return Position3D(
            x=self.dimensions.width * 0.99,   # Top-right: x=0.99, y=0.01
            y=self.dimensions.length * 0.01,
            z=self.dimensions.height - 0.5     # 50cm margin from ceiling
        )

    def get_bottom_right_corner(self) -> Position3D:
        return Position3D(
            x=self.dimensions.width * 0.99,   # Bottom-right: x=0.99, y=0.99
            y=self.dimensions.length * 0.99,
            z=self.dimensions.height - 0.5     # 50cm margin from ceiling
        )

    def get_roof_center(self) -> Position3D:
        return Position3D(
            x=self.dimensions.width * 0.50,    # Center: x=width/2, y=length/2
            y=self.dimensions.length * 0.50,
            z=self.dimensions.height - 0.5     # 50cm margin from ceiling for pyroomacoustics compatibility
        )

    def add_furnitures(self, furnitures: dict[str, Furniture]):
        self.furnitures.update(furnitures)

    def get_furnitures(self) -> dict[str, Furniture]:
        return self.furnitures

    def get_square_meters(self) -> float:
        """
        Get the square meters of the room
        """
        return self.dimensions.width * self.dimensions.length

    def get_volume(self) -> float:
        """
        Get the volume of the room
        """
        return self.dimensions.width * self.dimensions.length * self.dimensions.height

    def get_speaker_distances_to_microphone(self, dimensions: int = 3) -> dict[str, float]:
        """
        Get the distances between speakers and the microphone in 2D or 3D.
        """
        if dimensions in [2, 3]:
            return {
                speaker_name: coordinates.distance_to(self.mic_position_3d, dimensions=dimensions)
                for speaker_name, coordinates in self.speakers_positions.items()
            }
        else:
            raise ValueError(f"Invalid dimensions: {dimensions}")

    def to_image(
        self,
        show_speakers: bool = True,
        show_furnitures: bool = True,
        show_microphones: bool = True,
        show_anchors: bool = True,
        show_walls: bool = True
    ):
        """
        Create a room plan (pillow image) based on the "dimensions"
        """
        from PIL import Image, ImageDraw, ImageFont

        # Create a 512x512 image with white background
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)

        # Calculate scaling factors to fit the room in the image
        # Leave some margin (50 pixels on each side)
        margin = 50
        available_width = 512 - 2 * margin
        available_height = 512 - 2 * margin

        # Calculate scale factors for width (x-axis) and length (y-axis)
        scale_x = available_width / self.dimensions.width
        scale_y = available_height / self.dimensions.length

        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)

        # Calculate the actual room dimensions in pixels
        room_width_px = int(self.dimensions.width * scale)
        room_length_px = int(self.dimensions.length * scale)

        # Center the room in the image
        start_x = (512 - room_width_px) // 2
        start_y = (512 - room_length_px) // 2

        if show_walls:
            # Draw the room walls (rectangle)
            # Top wall
            draw.line(
                [(start_x, start_y), (start_x + room_width_px, start_y)],
                fill='black', width=3
            )
            # Right wall
            draw.line(
                [
                    (start_x + room_width_px, start_y),
                    (start_x + room_width_px, start_y + room_length_px)
                ],
                fill='black', width=3
            )
            # Bottom wall
            draw.line(
                [
                    (start_x + room_width_px, start_y + room_length_px),
                    (start_x, start_y + room_length_px)
                ],
                fill='black', width=3
            )
            # Left wall
            draw.line(
                [(start_x, start_y + room_length_px), (start_x, start_y)],
                fill='black', width=3
            )

        # Add room dimensions as text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Add dimension labels
        dim_text = f"{self.dimensions.width:.1f}m x {self.dimensions.length:.1f}m"
        if font:
            # Get text size for centering
            bbox = draw.textbbox((0, 0), dim_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position text at the bottom of the image
            text_x = (512 - text_width) // 2
            text_y = 512 - text_height - 10

            draw.text((text_x, text_y), dim_text, fill='black', font=font)

        # Add room name if available
        if self.name and self.name != f"Room_{self.id}":
            name_text = self.name
            if font:
                bbox = draw.textbbox((0, 0), name_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Position text at the top of the image
                text_x = (512 - text_width) // 2
                text_y = 10

                draw.text((text_x, text_y), name_text, fill='black', font=font)

        if show_furnitures:
            #########################
            # Drawing furnitures
            #########################

            # Add furniture as rectangles using their x, y, width and depth coordinates
            for furniture_name, furniture in self.furnitures.items():
                # Convert furniture coordinates to pixel coordinates
                # Furniture coordinates are in meters, need to convert to pixels
                furniture_x_px = start_x + int(furniture.x * scale)
                furniture_y_px = start_y + int(furniture.y * scale)

                # Convert furniture dimensions to pixels
                furniture_width_px = int(furniture.width * scale)
                furniture_depth_px = int(furniture.depth * scale)

                # Calculate rectangle coordinates (top-left and bottom-right)
                # Furniture position is now the top-left corner
                rect_left = furniture_x_px
                rect_top = furniture_y_px
                rect_right = furniture_x_px + furniture_width_px
                rect_bottom = furniture_y_px + furniture_depth_px

                # Ensure minimum size for visibility
                min_size = 4  # Minimum 4 pixels
                if furniture_width_px < min_size:
                    rect_right = rect_left + min_size
                if furniture_depth_px < min_size:
                    rect_bottom = rect_top + min_size

                # Draw furniture rectangle outline
                draw.rectangle(
                    [rect_left, rect_top, rect_right, rect_bottom],
                    outline=furniture.color.value, width=2
                )

                # Fill the rectangle with a semi-transparent red color
                # Create a temporary image for the fill
                fill_img = Image.new('RGBA', (rect_right - rect_left, rect_bottom - rect_top), furniture.color.value)
                img.paste(fill_img, (rect_left, rect_top), fill_img)

                # Add furniture name as text near the rectangle
                if font:
                    # Get text size for positioning
                    bbox = draw.textbbox((0, 0), furniture_name, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Position text at the center of the rectangle
                    text_x = rect_left + (rect_right - rect_left - text_width) // 2
                    text_y = rect_top + (rect_bottom - rect_top - text_height) // 2

                    # Make sure text doesn't go outside the image bounds
                    if text_x < 0:
                        text_x = 5
                    elif text_x + text_width > 512:
                        text_x = 512 - text_width - 5
                    if text_y < 0:
                        text_y = 5
                    elif text_y + text_height > 512:
                        text_y = 512 - text_height - 5

                    draw.text((text_x, text_y), furniture_name, fill=furniture.color.value, font=font)

        if show_microphones:
            #########################
            # Drawing microphone position
            #########################
            # Convert microphone coordinates to pixel coordinates relative to the room
            # Microphone coordinates are in meters, need to convert to pixels and position relative to room
            mic_x_px = start_x + int(self.mic_position_3d.x * scale)
            mic_y_px = start_y + int(self.mic_position_3d.y * scale)

            # Ensure microphone is within room bounds
            mic_x_px = max(start_x + 5, min(mic_x_px, start_x + room_width_px - 5))
            mic_y_px = max(start_y + 5, min(mic_y_px, start_y + room_length_px - 5))

            # Draw microphone as a circle
            draw.circle(
                (mic_x_px, mic_y_px),
                radius=8,
                fill='red',
                outline='black',
                width=2
            )

            # Add microphone label
            mic_label = 'Mic' if self.mic_position != MicrophonePosition.CUSTOM else 'Custom Mic'
            if font:
                # Get text size for positioning
                bbox = draw.textbbox((0, 0), mic_label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Position text below the microphone circle
                text_x = mic_x_px - text_width // 2
                text_y = mic_y_px + 12  # Position below the circle

                # Make sure text doesn't go outside the image bounds
                if text_x < 0:
                    text_x = 5
                elif text_x + text_width > 512:
                    text_x = 512 - text_width - 5
                if text_y < 0:
                    text_y = 5
                elif text_y + text_height > 512:
                    text_y = 512 - text_height - 5

                draw.text((text_x, text_y), mic_label, fill='red', font=font)

        if show_anchors:
            #########################
            # Drawing corners and center of the room based on get_top_left_corner,
            # get_top_right_corner, get_bottom_left_corner, get_bottom_right_corner, get_roof_center
            #########################

            # Get corner and center positions
            top_left = self.get_top_left_corner()
            top_right = self.get_top_right_corner()
            bottom_left = self.get_bottom_left_corner()
            bottom_right = self.get_bottom_right_corner()
            roof_center = self.get_roof_center()

            # Convert 3D positions to pixel coordinates (ignoring z for 2D view)
            def pos_to_pixels(pos: Position3D) -> Tuple[int, int]:
                x_px = start_x + int(pos.x * scale)
                y_px = start_y + int(pos.y * scale)
                return x_px, y_px

            # Draw corner points with improved label positioning
            corner_positions = [
                (top_left, "TL", "top-left"),
                (top_right, "TR", "top-right"),
                (bottom_left, "BL", "bottom-left"),
                (bottom_right, "BR", "bottom-right"),
                (roof_center, "RC", "center")
            ]

            for pos, label, position_type in corner_positions:
                x_px, y_px = pos_to_pixels(pos)

                # Ensure points are within room bounds (allow reaching exact edges)
                x_px = max(start_x + 5, min(x_px, start_x + room_width_px - 5))
                y_px = max(start_y + 5, min(y_px, start_y + room_length_px - 5))

                # Draw corner point as a small circle
                draw.circle(
                    (x_px, y_px),
                    radius=4,
                    fill='blue',
                    outline='darkblue',
                    width=1
                )

                # Add corner label with improved positioning
                if font:
                    # Get text size for positioning
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Position text based on corner type to avoid overlaps
                    if position_type == "top-left":
                        text_x = x_px + 8  # To the right
                        text_y = y_px + 8  # Below
                    elif position_type == "top-right":
                        text_x = x_px - text_width - 8  # To the left
                        text_y = y_px + 8  # Below
                    elif position_type == "bottom-left":
                        text_x = x_px + 8  # To the right
                        text_y = y_px - text_height - 8  # Above
                    elif position_type == "bottom-right":
                        text_x = x_px - text_width - 8  # To the left
                        text_y = y_px - text_height - 8  # Above
                    else:  # center
                        text_x = x_px - text_width // 2  # Centered horizontally
                        text_y = y_px - text_height - 8  # Above

                    # Make sure text doesn't go outside the image bounds
                    text_x = max(5, min(text_x, 512 - text_width - 5))
                    text_y = max(5, min(text_y, 512 - text_height - 5))

                    draw.text((text_x, text_y), label, fill='blue', font=font)

        if show_speakers:
            #########################
            # Drawing speakers positions from self.speakers_positions
            #########################
            for speaker_name, speaker_position in self.speakers_positions.items():
                # Convert speaker coordinates to pixel coordinates relative to the room
                # Speaker coordinates are in meters, need to convert to pixels and position relative to room
                speaker_x_px = start_x + int(speaker_position.x * scale)
                speaker_y_px = start_y + int(speaker_position.y * scale)

                # Ensure speaker is within room bounds
                speaker_x_px = max(start_x + 5, min(speaker_x_px, start_x + room_width_px - 5))
                speaker_y_px = max(start_y + 5, min(speaker_y_px, start_y + room_length_px - 5))

                # Draw speaker as a circle with a different color for each speaker
                # Use a simple hash of the speaker name to get a consistent color
                color_hash = hash(speaker_name) % 360  # Get hue value
                import colorsys
                rgb = colorsys.hsv_to_rgb(color_hash / 360.0, 0.8, 0.8)
                speaker_color = tuple(int(c * 255) for c in rgb)

                # Draw speaker as a circle
                draw.circle(
                    (speaker_x_px, speaker_y_px),
                    radius=10,
                    fill=speaker_color,
                    outline='black',
                    width=2
                )

                # Add speaker name as label
                if font:
                    # Get text size for positioning
                    bbox = draw.textbbox((0, 0), speaker_name, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Position text below the speaker circle
                    text_x = speaker_x_px - text_width // 2
                    text_y = speaker_y_px + 15  # Position below the circle

                    # Make sure text doesn't go outside the image bounds
                    if text_x < 0:
                        text_x = 5
                    elif text_x + text_width > 512:
                        text_x = 512 - text_width - 5
                    if text_y < 0:
                        text_y = 5
                    elif text_y + text_height > 512:
                        text_y = 512 - text_height - 5

                    draw.text((text_x, text_y), speaker_name, fill=speaker_color, font=font)

        return img

    def set_directivity(
        self,
        direction: DirectivityType = DirectivityType.OMNIDIRECTIONAL,
        directivity: MicrophoneDirectivity = None,
    ):
        """
        Apply a directivity to the microphone based on the directivity type.
        """

        if direction not in [_dt for _dt in DirectivityType]:
            raise ValueError(f"Directivity type {direction} is not supported")

        # Add the microphone directivity if not already set
        if direction == DirectivityType.CUSTOM:

            if directivity is None:
                raise ValueError("Microphone directivity is required for custom directivity type")

            self.directivity_type = direction
            self.microphone_directivity = directivity

        else:

            if directivity is not None:
                logger.warning(
                    "The given directivity is not taken into account for non-custom directivity type"
                )

            # Compute the azimuth and colatitude based on the directivity type
            _azimuth, _colatitude = self.directivity_type_to_azimuth_colatitude(
                direction
                if direction is not None
                else DirectivityType.OMNIDIRECTIONAL
            )

            self.directivity_type = direction

            # Build the microphone directivity
            self.microphone_directivity = MicrophoneDirectivity(
                azimuth=_azimuth,
                colatitude=_colatitude,
                gain=1.0
            )

    def set_mic_position(
        self,
        mic_position: MicrophonePosition,
        position_3D: Optional[Position3D] = None
    ):
        """
        Set the microphone position.

        :param mic_position: The microphone position.
        :type mic_position: MicrophonePosition
        :param position_3D: The 3D position of the microphone.
        :type position_3D: Optional[Position3D]
        :return: None
        """

        self.mic_position = mic_position
        self.mic_position_3d = microphone_to_position(self, self.mic_position, position_3D=position_3D)
        self.set_directivity(direction=self.directivity_type, directivity=self.microphone_directivity)

    def model_post_init(self, __context: Any) -> None:
        """
        Post init function to set the microphone position 3D.
        """

        if len(self.speakers_positions) > 0:
            for _role, _position in self.speakers_positions.items():
                if _role not in [Role.SPEAKER_1, Role.SPEAKER_2]:
                    raise ValueError(f"Speaker name '{_role}' is not valid, the speaker wasn't placed")

        # if the user override the center of the room, add it to the furnitures
        if "center" not in self.furnitures:
            self.furnitures["center"] = Furniture(
                name="center",
                x=self.dimensions.width * 0.50,
                y=self.dimensions.length * 0.50,
                width=0.0,
                height=0.0,
                depth=0.0
            )

        # Initialize the speakers positions if not already set
        if Role.SPEAKER_1 not in self.speakers_positions:
            self.place_speaker_around_furniture(
                Role.SPEAKER_1,
                furniture_name="center",
                side=SpeakerSide.FRONT,
                max_distance=2.0
            )
        if Role.SPEAKER_2 not in self.speakers_positions:
            self.place_speaker_around_furniture(
                Role.SPEAKER_2,
                furniture_name="center",
                side=SpeakerSide.BACK,
                max_distance=2.0
            )

        # Convert the microphone position to 3D coordinates
        self.mic_position_3d = microphone_to_position(
            self,
            self.mic_position,
            position_3D=self.mic_position_3d
        )

        # Set the directivity of the microphone
        self.set_directivity(
            direction=self.directivity_type,
            directivity=self.microphone_directivity
        )

        # Set the name of the room if not already set
        if self.name == "Room":
            self.name = f"{self.name}_{self.id}"

    def get_info(self) -> Dict[str, Any]:
        """
        Get the information about the room in a format that can be serialized.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dimensions": self.dimensions.to_list(),
            "reverberation_time_ratio": self.reverberation_time_ratio,
            "materials": self.materials.model_dump(),
            "mic_type": self.mic_type.value,
            "mic_position": self.mic_position.value,
            "mic_position_3d": self.mic_position_3d.to_list()
        }

    def get_hash(self) -> str:
        """
        Get the hash of the room.
        """
        return hashlib.sha256(str(self.get_info()).encode()).hexdigest()

    def __str__(self):
        return (
            f"{self.id}:  {self.name}, desc: {self.description} "
            f"(dimentions: {str(self.dimensions)}, reverberation_time_ratio: {self.reverberation_time_ratio}"
            f"materials: {self.materials})"
        )


def microphone_to_position(
    room: Room,
    mic_pos: MicrophonePosition,
    position_3D: Optional[Position3D] = None
) -> Position3D:
    """
    Convert semantic microphone position enum to actual 3D coordinates within the room.

    This function maps microphone placement descriptions to concrete 3D coordinates
    that can be used for acoustic simulation.
    """
    width, length, height = (
        room.dimensions.width,
        room.dimensions.length,
        room.dimensions.height,
    )

    def clamp_position(x, y, z):
        """Ensure position is within room bounds with safety margin"""
        margin = 0.1  # 10cm safety margin from walls (except ceiling)
        x = max(margin, min(x, width - margin))
        y = max(margin, min(y, length - margin))
        z = max(0.1, min(z, height - 0.05))  # Smaller top margin for ceiling mics
        return Position3D.from_list([x, y, z])

    # Map microphone positions
    if mic_pos == MicrophonePosition.DESK_SMARTPHONE:

        if "desk" not in room.furnitures:
            raise ValueError((
                "Desk furniture is not found in the room, you can add it with the add_furniture method"
                " or change the mic_position to a different position"
            ))

        return clamp_position(
            room.furnitures["desk"].x + 0.3,
            room.furnitures["desk"].y + 0.2,
            room.furnitures["desk"].get_top_z()
        )

    elif mic_pos == MicrophonePosition.MONITOR:

        if "monitor" not in room.furnitures:
            raise ValueError((
                "Monitor furniture is not found in the room, you can add it with the add_furniture method"
                " or change the mic_position to a different position"
            ))

        return clamp_position(
            room.furnitures["monitor"].x + 0.1,
            room.furnitures["monitor"].y,
            room.furnitures["monitor"].get_top_z()
        )

    elif mic_pos == MicrophonePosition.WALL_MOUNTED:
        return clamp_position(width * 0.01, length * 0.50, BodyPosture.STANDING.value)

    elif mic_pos == MicrophonePosition.CEILING_CENTERED:
        return clamp_position(width * 0.50, length * 0.50, height - 0.1)

    elif mic_pos == MicrophonePosition.MIDDLE_SPEAKERS:
        speaker_1_position = room.speakers_positions[Role.SPEAKER_1]
        speaker_2_position = room.speakers_positions[Role.SPEAKER_2]
        return clamp_position(
            (speaker_1_position.x + speaker_2_position.x) / 2,
            (speaker_1_position.y + speaker_2_position.y) / 2,
            BodyPosture.STANDING.value - 0.3
        )

    elif mic_pos in [MicrophonePosition.CHEST_POCKET_SPEAKER_1, MicrophonePosition.CHEST_POCKET_SPEAKER_2]:
        speaker_position = room.speakers_positions[Role.SPEAKER_1 if "speaker_1" in mic_pos else Role.SPEAKER_2]
        return clamp_position(speaker_position.x, speaker_position.y, BodyPosture.STANDING.value - 0.3)

    elif mic_pos == MicrophonePosition.CUSTOM:
        if position_3D is None:
            raise ValueError("Custom 3D position is required, you can use the mic_position_3d attribute to set it")
        return position_3D

    # Fallback to center position at monitor height
    return clamp_position(
        room.furnitures["center"].x,
        room.furnitures["center"].y,
        room.furnitures["monitor"].get_top_z()
    )
