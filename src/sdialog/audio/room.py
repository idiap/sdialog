"""
This module provides classes for the room specification.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Pawel Cyrta <pawel@cyrta.com>
# SPDX-License-Identifier: MIT
import time
import hashlib
import numpy as np
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field, PrivateAttr
from typing import Dict, Optional, Tuple, List, Any

# from pyroomacoustics.directivities.analytic import Omnidirectional


@dataclass
class Position3D:
    """
    3D position coordinates in meters
    """

    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __post_init__(self):
        if any(coord < 0 for coord in [self.x, self.y, self.z]):
            raise ValueError("Coordinates must be non-negative")

    def __str__(self):
        return f"pos: [{self.x}, {self.y}, {self.z}]"

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_list(self):
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, position_list: List[float]) -> "Position3D":
        if len(position_list) != 3:
            raise ValueError("Position must have exactly 3 coordinates [x, y, z]")
        return cls(x=position_list[0], y=position_list[1], z=position_list[2])


@dataclass
class Dimensions3D:
    """
    3D dimensions in meters
    """

    width: float  # x-axis
    length: float  # y-axis
    height: float  # z-axis

    def __post_init__(self):
        if any(dim <= 0 for dim in [self.width, self.length, self.height]):
            raise ValueError("All dimensions must be positive")

    def __str__(self):
        return f"dim: [{self.width}, {self.length}, {self.height}]"

    def to_list(self):
        return [self.width, self.length, self.height]

    @property
    def volume(self) -> float:
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

    @classmethod
    def from_volume(
        cls,
        volume: float,
        aspect_ratio: Tuple[float, float, float] = (1.5, 1.0, 0.3)
    ):
        """
        Generate dimensions from volume using aspect ratio (width:length:height)
        """
        if volume <= 0:
            raise ValueError("Volume must be positive")

        w_ratio, l_ratio, h_ratio = aspect_ratio
        scale = (volume / (w_ratio * l_ratio * h_ratio)) ** (1 / 3)

        return cls(
            width=w_ratio * scale, length=l_ratio * scale, height=h_ratio * scale
        )


class SoundEventPosition(str, Enum):
    BACKGROUND = "no_type"  # background -
    NOT_DEFINED = "soundevent-not_defined"
    DEFINED = "soundevent-defined"  # [0.0 0.1 0.4]
    # NEXT_TO_DOCTOR
    # NEXT_TO PATIENT


class DoctorPosition(str, Enum):
    """
    Doctor placement locations in examination room
    """

    AT_DESK_SITTING = "doctor-at_desk_sitting"
    AT_DESK_SIDE_STANDING = "doctor-at_desk_side_standing"
    NEXT_TO_BENCH_STANDING = "doctor-next_to_bench_standing"
    NEXT_TO_SINK_FRONT = "doctor-next_to_sink_front"
    NEXT_TO_SINK_BACK = "doctor-next_to_sink_back"
    NEXT_TO_CUPBOARD_FRONT = "doctor-next_to_cupboard_front"
    NEXT_TO_CUPBOARD_BACK = "doctor-next_to_cupboard_back"
    NEXT_TO_DOOR_STANDING = "doctor-next_to_door_standing"


class PatientPosition(str, Enum):
    """
    Patient placement locations in examination room
    """

    AT_DOOR_STANDING = "patient-at_door_standing"
    NEXT_TO_DESK_SITTING = "patient-next_to_desk_sitting"
    NEXT_TO_DESK_STANDING = "patient-next_to_desk_standing"
    SITTING_ON_BENCH = "patient-sitting_on_bench"
    CENTER_ROOM_STANDING = "patient-center_room_standing"


class MicrophonePosition(str, Enum):
    """
    Different microphone placement options
    """

    TABLE_SMARTPHONE = "table_smartphone"
    MONITOR = "monitor"
    WALL_MOUNTED = "wall_mounted"
    CEILING_CENTERED = "ceiling_centered"
    CHEST_POCKET = "chest_pocket"


class WallMaterial(str, Enum):
    """
    Common wall materials with typical absorption coefficients
    """

    DRYWALL = "drywall"
    CONCRETE = "concrete"
    BRICK = "brick"
    WOOD_PANEL = "wood_panel"
    ACOUSTIC_TILE = "acoustic_tile"
    GLASS = "glass"
    METAL = "metal"


class FloorMaterial(str, Enum):
    """
    Floor materials affecting acoustics
    """

    CARPET = "carpet"
    VINYL = "vinyl"
    CONCRETE = "concrete"
    HARDWOOD = "hardwood"
    TILE = "tile"
    RUBBER = "rubber"


# ------------------------------------------------------------------------------


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
    _is_primary: Optional[bool] = PrivateAttr(default=False)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def model_post_init(self, __context: Any) -> None:
        self._is_primary = self._determine_primary_status(self.name)

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

    @staticmethod
    def _determine_primary_status(name: str) -> bool:
        """
        Determine if a source is primary based on its name.
        """
        primary_names = [
            "doctor",
            "physician",
            "main_speaker",
            "speaker_a",
            "primary",
            "médecin",
            "medecin",
            "docteur",
            "lekarz",
            "doktor",
            "lékař",
        ]
        return name.lower() in primary_names


# ------------------------------------------------------------------------------

# related to https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py

def get_room_id():
    """
    Get the room id based on the time in nanoseconds.
    """
    return str(int(time.time_ns()))


class Room(BaseModel):
    """
    Class to handle a room for audio simulation.
    """
    id: str = Field(default_factory=get_room_id)
    name: str = "Room"
    description: str = ""
    reverberation_time_ratio: Optional[float] = 0.5
    aspect_ratio: Optional[Tuple[float, float]] = None

    dimensions: Dimensions3D = Field(default_factory=lambda: Dimensions3D(2, 2.5, 3))

    mic_position: MicrophonePosition = MicrophonePosition.MONITOR

    model_config = {
        "arbitrary_types_allowed": True,
    }

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

    def to_image(self):
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

        return img

    def model_post_init(self, __context: Any) -> None:
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
            # "role": self.role.value,
            "dimensions": self.dimensions.to_list(),
            "reverberation_time_ratio": self.reverberation_time_ratio,
            "mic_type": self.mic_type.value,
            "mic_position": self.mic_position.value
        }

    def get_hash(self) -> str:
        """
        Get the hash of the room.
        """
        return hashlib.sha256(str(self.get_info()).encode()).hexdigest()

    def __str__(self):
        return (
            f"{self.id}:  {self.name}, desc: {self.description} "
            f"(dimentions: {str(self.dimensions)}, reverberation_time_ratio: {self.reverberation_time_ratio})"
            f"(aspect_ratio: {self.aspect_ratio})"
        )
