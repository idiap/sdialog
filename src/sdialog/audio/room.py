"""
This module provides classes for the room specification.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import time
import numpy as np

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List, Any


@dataclass
class Position3D:
    """3D position coordinates in meters"""

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
    """3D dimensions in meters"""

    width: float  # x-axis
    length: float  # y-axis
    height: float  # z-axis

    def __post_init__(self):
        if any(dim <= 0 for dim in [self.width, self.length, self.height]):
            raise ValueError("All dimensions must be positive")

    def __str__(self):
        return f"dim: [{self.width}, {self.length}, {self.height}]"

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

    def to_list(self):
        return [self.length, self.width, self.height]

    @classmethod
    def from_volume(
        cls, volume: float, aspect_ratio: Tuple[float, float, float] = (1.5, 1.0, 0.3)
    ):
        """Generate dimensions from volume using aspect ratio (width:length:height)"""
        if volume <= 0:
            raise ValueError("Volume must be positive")

        w_ratio, l_ratio, h_ratio = aspect_ratio
        scale = (volume / (w_ratio * l_ratio * h_ratio)) ** (1 / 3)

        return cls(
            width=w_ratio * scale, length=l_ratio * scale, height=h_ratio * scale
        )


class RoomRole(Enum):
    """Defines the functional role of the room and dimentions that comes with it."""

    CONSULTATION = "consultation"
    EXAMINATION = "examination"
    TREATMENT = "treatment"
    PATIENT_ROOM = "patient_room"
    SURGERY = "surgery"  # operating_room
    WAITING = "waiting_room"
    EMERGENCY = "emergency"
    OFFICE = "office"

    # def __str__(self):
    #     return self.value


class DoctorPosition(Enum):
    """Doctor placement locations in examination room"""

    AT_DESK_SITTING = "doctor-at_desk_sitting"
    AT_DESK_SIDE_STANDING = "doctor-at_desk_side_standing"
    NEXT_TO_BENCH_STANDING = "doctor-next_to_bench_standing"
    NEXT_TO_SINK_FRONT = "doctor-next_to_sink_front"
    NEXT_TO_SINK_BACK = "doctor-next_to_sink_back"
    NEXT_TO_CUPBOARD_FRONT = "doctor-next_to_cupboard_front"
    NEXT_TO_CUPBOARD_BACK = "doctor-next_to_cupboard_back"
    NEXT_TO_DOOR_STANDING = "doctor-next_to_door_standing"


class PatientPosition(Enum):
    """Patient placement locations in examination room"""

    AT_DOOR_STANDING = "patient-at_door_standing"
    NEXT_TO_DESK_SITTING = "patient-next_to_desk_sitting"
    NEXT_TO_DESK_STANDING = "patient-next_to_desk_standing"
    SITTING_ON_BENCH = "patient-sitting_on_bench"
    CENTER_ROOM_STANDING = "patient-center_room_standing"


class MicrophonePosition(Enum):
    """Different microphone placement options"""

    TABLE_SMARTPHONE = "table_smartphone"
    MONITOR = "monitor"
    WALL_MOUNTED = "wall_mounted"
    CEILING_CENTERED = "ceiling_centered"
    CHEST_POCKET = "chest_pocket"


class RecordingDevice(Enum):
    """Types of recording devices with their characteristics"""

    SMARTPHONE = "smartphone"
    WEBCAM = "webcam"
    TABLET = "tablet"
    HIGH_QUALITY_MIC = "high_quality_mic"
    BEAMFORMING_MIC = "beamforming_mic"
    LAVALIER_MIC = "lavalier_mic"
    SHOTGUN_MIC = "shotgun_mic"


class WallMaterial(Enum):
    """Common wall materials with typical absorption coefficients"""

    DRYWALL = "drywall"
    CONCRETE = "concrete"
    BRICK = "brick"
    WOOD_PANEL = "wood_panel"
    ACOUSTIC_TILE = "acoustic_tile"
    GLASS = "glass"
    METAL = "metal"


class FloorMaterial(Enum):
    """Floor materials affecting acoustics"""

    CARPET = "carpet"
    VINYL = "vinyl"
    CONCRETE = "concrete"
    HARDWOOD = "hardwood"
    TILE = "tile"
    RUBBER = "rubber"


# ------------------------------------------------------------------------------


@dataclass
class AudioSource:
    """Represents an object, speaker that makes sounds in the room"""

    name: str = None
    position: str = None
    snr: float = 0.0  # dB SPL
    source_file: str = None  # audio file e.g wav
    directivity: Optional[str] = "omnidirectional"
    _position3d: Position3D = None
    _is_primary: Optional[bool] = False  # Primary speaker (doctor) vs secondary (patient)

    def __post_init__(self):
        self._is_primary = self._determine_primary_status(self.name)

    @property
    def x(self) -> float:
        return self._position3d.x

    @property
    def y(self) -> float:
        return self._position3d.y

    @property
    def z(self) -> float:
        return self._position3d.z

    def distance_to(self, other_position: Tuple[float, float, float]) -> float:
        return (
            (self.x - other_position[0]) ** 2
            + (self.y - other_position[1]) ** 2
            + (self.z - other_position[2]) ** 2
        ) ** 0.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioSource":
        """
        Create AudioSource from dictionary data.
        """
        if "name" not in data:
            raise ValueError("Missing required field 'name'")

        if "position" not in data:
            raise ValueError("Missing required field 'position'")

        return cls(
            name=data["name"],
            position=data["position"],
            snr=data.get("snr", 0.0),
            directivity=data.get("directivity", "omnidirectional"),
            source_file=data.get("source_file", ""),
        )

    @staticmethod
    def _determine_primary_status(name: str) -> bool:
        """Determine if a source is primary based on its name."""
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


@dataclass
class Room:
    """
    A room is a place where the dialog takes place.
    """

    def __init__(
        self,
        role: RoomRole,
        dimensions: Optional[Dimensions3D],
        name: str = "Room",
        description: str = "",
        rt60: float = 0.5,
        soundsources_position=[],
        mic_type=RecordingDevice.WEBCAM,
        mic_position=MicrophonePosition.MONITOR,
        furnitures=False,
    ):
        self.id: str = str(int(time.time()))[-4:]
        self.name: str = f"{name}_{self.id}"
        self.description = description
        self.role: RoomRole = role if role is not None else RoomRole.CONSULTATION
        self.dimensions: Dimensions3D = (
            dimensions if dimensions is not None else Dimensions3D(2, 2.5, 3)
        )
        self.walls_material: Optional[MaterialProperties] = (
            None  # absorbion_coefficient
        )
        self.rt60: Optional[float] = rt60
        self.mic_type = mic_type
        self.mic_position = mic_position
        self.furnitures = furnitures

    def __str__(self):
        return (
            f"{self.id}:  {self.name}, desc: {self.description} "
            f"(dimentions: {str(self.dimensions)}, rt60: {self.rt60}) role: {self.role})  "
        )


@dataclass
class RoomLayout:
    """Defines the standard layout of furniture in examination room"""

    door_position: Position3D
    desk_position: Position3D
    monitor_position: Optional[Position3D]
    bench_position: Optional[Position3D]
    sink_position: Optional[Position3D]
    cupboard_position: Optional[Position3D]


@dataclass
class MaterialProperties:
    """Acoustic properties of materials"""

    material_type: Union[WallMaterial, FloorMaterial, str]
    absorption_coefficients: Dict[int, float] = field(
        default_factory=dict
    )  # frequency -> coefficient
    scattering_coefficient: float = 0.1

    def __post_init__(self):
        # Set default absorption coefficients if not provided
        if not self.absorption_coefficients:
            self.absorption_coefficients = self._get_default_absorption()

    def _get_default_absorption(self) -> Dict[int, float]:
        """Default absorption coefficients for common frequencies (Hz)"""
        defaults = {
            WallMaterial.DRYWALL: {
                125: 0.05,
                250: 0.06,
                500: 0.08,
                1000: 0.09,
                2000: 0.10,
                4000: 0.11,
            },
            WallMaterial.CONCRETE: {
                125: 0.02,
                250: 0.02,
                500: 0.03,
                1000: 0.04,
                2000: 0.05,
                4000: 0.06,
            },
            WallMaterial.ACOUSTIC_TILE: {
                125: 0.20,
                250: 0.40,
                500: 0.65,
                1000: 0.75,
                2000: 0.80,
                4000: 0.85,
            },
            WallMaterial.WOOD_PANEL: {
                125: 0.10,
                250: 0.15,
                500: 0.20,
                1000: 0.25,
                2000: 0.30,
                4000: 0.35,
            },
            WallMaterial.GLASS: {
                125: 0.03,
                250: 0.03,
                500: 0.03,
                1000: 0.04,
                2000: 0.05,
                4000: 0.05,
            },
            WallMaterial.METAL: {
                125: 0.02,
                250: 0.02,
                500: 0.03,
                1000: 0.04,
                2000: 0.05,
                4000: 0.05,
            },
            FloorMaterial.CARPET: {
                125: 0.05,
                250: 0.10,
                500: 0.20,
                1000: 0.30,
                2000: 0.40,
                4000: 0.50,
            },
            FloorMaterial.VINYL: {
                125: 0.02,
                250: 0.03,
                500: 0.03,
                1000: 0.04,
                2000: 0.04,
                4000: 0.05,
            },
            FloorMaterial.CONCRETE: {
                125: 0.02,
                250: 0.02,
                500: 0.03,
                1000: 0.04,
                2000: 0.05,
                4000: 0.06,
            },
            FloorMaterial.HARDWOOD: {
                125: 0.08,
                250: 0.09,
                500: 0.10,
                1000: 0.11,
                2000: 0.12,
                4000: 0.13,
            },
            FloorMaterial.TILE: {
                125: 0.02,
                250: 0.02,
                500: 0.03,
                1000: 0.03,
                2000: 0.04,
                4000: 0.05,
            },
            FloorMaterial.RUBBER: {
                125: 0.04,
                250: 0.05,
                500: 0.08,
                1000: 0.12,
                2000: 0.15,
                4000: 0.18,
            },
        }

        if isinstance(self.material_type, str):
            # Return generic values for custom materials
            return {125: 0.05, 250: 0.06, 500: 0.08, 1000: 0.09, 2000: 0.10, 4000: 0.11}

        return defaults.get(
            self.material_type,
            {125: 0.05, 250: 0.06, 500: 0.08, 1000: 0.09, 2000: 0.10, 4000: 0.11},
        )


class FurnitureType(Enum):
    """Types of furniture commonly found in medical rooms"""

    DESK = "desk"
    MONITOR = "monitor"
    CHAIR = "chair"
    BENCH = "bench"
    EXAMINATION_TABLE = "examination_table"
    CABINET = "cabinet"
    EQUIPMENT_CART = "equipment_cart"
    BED = "bed"
    DIVIDER_CURTAIN = "divider_curtain"
    BOOKSHELF = "bookshelf"
    SINK = "sink"


@dataclass
class Furniture:
    """Furniture object in the room"""

    name: str
    furniture_type: FurnitureType
    position: Position3D
    dimensions: Dimensions3D
    material: MaterialProperties
    is_movable: bool = True

    @property
    def volume(self) -> float:
        return self.dimensions.volume


@dataclass
class RecordingDeviceSpec:
    """Recording device specifications"""

    device_type: RecordingDevice = None
    sensitivity: float = -40.0  # dBV/Pa
    frequency_response: Tuple[int, int] = (20, 20000)  # Hz range
    snr: float = 60.0  # Signal-to-noise ratio in dB
    directivity_pattern: str = "omnidirectional"  # omnidirectional, cardioid, etc.
    num_channels: int = 1
    position: Position3D = field(default_factory=lambda: Position3D(0, 0, 1.5))

    def __post_init__(self):
        # Set default values based on device type
        device_defaults = {
            RecordingDevice.SMARTPHONE: {
                "sensitivity": -38.0,
                "snr": 50.0,
                "num_channels": 1,
            },
            RecordingDevice.WEBCAM: {
                "sensitivity": -42.0,
                "snr": 45.0,
                "num_channels": 1,
            },
            RecordingDevice.TABLET: {
                "sensitivity": -40.0,
                "snr": 48.0,
                "num_channels": 1,
            },
            RecordingDevice.HIGH_QUALITY_MIC: {
                "sensitivity": -35.0,
                "snr": 70.0,
                "num_channels": 1,
            },
            RecordingDevice.BEAMFORMING_MIC: {
                "sensitivity": -40.0,
                "snr": 65.0,
                "num_channels": 8,
                "directivity_pattern": "beamformed",
            },
            RecordingDevice.LAVALIER_MIC: {
                "sensitivity": -44.0,
                "snr": 55.0,
                "num_channels": 1,
                "directivity_pattern": "omnidirectional",
            },
            RecordingDevice.SHOTGUN_MIC: {
                "sensitivity": -35.0,
                "snr": 65.0,
                "num_channels": 1,
                "directivity_pattern": "shotgun",
            },
        }

        if self.device_type in device_defaults:
            defaults = device_defaults[self.device_type]
            for key, value in defaults.items():
                # Only update if still at default value
                value = getattr(
                    RecordingDeviceSpec.__dataclass_fields__[key], "default", None
                )
                if hasattr(self, key) and getattr(self, key) == value:
                    setattr(self, key, value)
