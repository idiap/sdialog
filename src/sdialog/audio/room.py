"""
This module provides classes for the room specification.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np

#------------------------------------------------
#


@dataclass
class Position3D:
    """3D position coordinates in meters"""
    x: float
    y: float
    z: float

    def __post_init__(self):
        if any(coord < 0 for coord in [self.x, self.y, self.z]):
            raise ValueError("Coordinates must be non-negative")
    def __str__(self):
        return f"pos: [{self.x}, {self.y}, {self.z}]"
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Dimensions3D:
    """3D dimensions in meters"""
    width: float   # x-axis
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
    @classmethod
    def from_volume(cls, volume: float, aspect_ratio: Tuple[float, float, float] = (1.5, 1.0, 0.3)):
        """Generate dimensions from volume using aspect ratio (width:length:height)"""
        if volume <= 0:
            raise ValueError("Volume must be positive")

        w_ratio, l_ratio, h_ratio = aspect_ratio
        scale = (volume / (w_ratio * l_ratio * h_ratio)) ** (1/3)

        return cls(
            width=w_ratio * scale,
            length=l_ratio * scale,
            height=h_ratio * scale
        )

#---------------------------------------------------------------------
# Enums


class RoomRole(Enum):
    """Defines the functional role of the room and dimentions that comes with it."""
    CONSULTATION = "consultation"
    EXAMINATION = "examination"
    TREATMENT = "treatment"
    PATIENT_ROOM = "patient_room"
    SURGERY = "surgery" #operating_room
    WAITING = "waiting_room"
    EMERGENCY = "emergency"
    OFFICE= "office"


class DoctorPosition(Enum):
    """Doctor placement locations in examination room"""
    AT_DESK_SITTING = "at_desk_sitting"
    NEXT_TO_BENCH_STANDING = "next_to_bench_standing"
    NEXT_TO_SINK_FRONT = "next_to_sink_front"
    NEXT_TO_SINK_BACK = "next_to_sink_back"
    NEXT_TO_CUPBOARD_FRONT = "next_to_cupboard_front"
    NEXT_TO_CUPBOARD_BACK = "next_to_cupboard_back"
    NEXT_TO_DOOR_STANDING = "next_to_door_standing"
    AT_DESK_SIDE_STANDING = "at_desk_side_standing"

class PatientPosition(Enum):
    """Patient placement locations in examination room"""
    AT_DOOR_STANDING = "at_door_standing"
    NEXT_TO_DESK_SITTING = "next_to_desk_sitting"
    NEXT_TO_DESK_STANDING = "next_to_desk_standing"
    SITTING_ON_BENCH = "sitting_on_bench"
    CENTER_ROOM_STANDING = "center_room_standing"

class MicrophonePosition(Enum):
    """Different microphone placement options"""
    TABLE_SMARTPHONE = "table_smartphone"
    MONITOR = "monitor"
    WALL_MOUNTED = "wall_mounted"
    CEILING_CENTERED = "ceiling_centered"
    CHEST_POCKET = "chest_pocket"

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


class RecordingDevice(Enum):
    """Types of recording devices with their characteristics"""
    SMARTPHONE = "smartphone"
    WEBCAM = "webcam"
    TABLET = "tablet"
    HIGH_QUALITY_MIC = "high_quality_mic"
    BEAMFORMING_MIC = "beamforming_mic"
    LAVALIER_MIC = "lavalier_mic"
    SHOTGUN_MIC = "shotgun_mic"


@dataclass
class SpeakerSource:
    """Represents a person speaking in the room"""
    name: str
    position: Position3D
    voice_level: float = 60.0  # dB SPL
    fundamental_frequency: float = 150.0  # Hz
    is_primary: bool = True  # Primary speaker (doctor) vs secondary (patient)


#------------------------------------------------
#

# https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py

@dataclass
class Room:
    """
    A room is a place where the dialog takes place.
    """

    def __init__(self, role: RoomRole, dimensions: Optional[Dimensions3D], name: str = "Room", description: str = "", rt60: float = 0.4,
        speaker_position=[], mic_type=RecordingDevice.WEBCAM, mic_position=MicrophonePosition.MONITOR,furnitures=False):
            self.id: str =  str(int(time.time()))[-4:]
            self.name: str = name + self.id
            self.description = description
            self.role: RoomRole = role if role is not None else RoomRole.CONSULTATION
            self.dimensions: Dimensions3D = dimensions if dimensions is not None else Dimensions3D(2, 2.5, 3)
            self.walls_material: Optional[MaterialProperties] = None  #absorbion_coefficient
            self.rt60: Optional[float] = rt60

    def __str__(self):
        return f"Room {self.id}: {self.name}. {self.description}. (dimentions: {str(self.dimensions)}, rt60: {self.rt60} ) "


@dataclass
class RoomLayout:
    """Defines the standard layout of furniture in examination room"""
    door_position: Position3D
    desk_position: Position3D
    monitor_position: Optional[Position3D]
    bench_position: Optional[Position3D]
    sink_position: Optional[Position3D]
    cupboard_position: Optional[Position3D]


#---------------------------------------------------------------------
#


@dataclass
class MaterialProperties:
    """Acoustic properties of materials"""
    material_type: Union[WallMaterial, FloorMaterial, str]
    absorption_coefficients: Dict[int, float] = field(default_factory=dict)  # frequency -> coefficient
    scattering_coefficient: float = 0.1

    def __post_init__(self):
        # Set default absorption coefficients if not provided
        if not self.absorption_coefficients:
            self.absorption_coefficients = self._get_default_absorption()

    def _get_default_absorption(self) -> Dict[int, float]:
        """Default absorption coefficients for common frequencies (Hz)"""
        defaults = {
            WallMaterial.DRYWALL: {125: 0.05, 250: 0.06, 500: 0.08, 1000: 0.09, 2000: 0.10, 4000: 0.11},
            WallMaterial.CONCRETE: {125: 0.02, 250: 0.02, 500: 0.03, 1000: 0.04, 2000: 0.05, 4000: 0.06},
            WallMaterial.ACOUSTIC_TILE: {125: 0.20, 250: 0.40, 500: 0.65, 1000: 0.75, 2000: 0.80, 4000: 0.85},
            WallMaterial.WOOD_PANEL: {125: 0.10, 250: 0.15, 500: 0.20, 1000: 0.25, 2000: 0.30, 4000: 0.35},
            WallMaterial.GLASS: {125: 0.03, 250: 0.03, 500: 0.03, 1000: 0.04, 2000: 0.05, 4000: 0.05},
            WallMaterial.METAL: {125: 0.02, 250: 0.02, 500: 0.03, 1000: 0.04, 2000: 0.05, 4000: 0.05},
            FloorMaterial.CARPET: {125: 0.05, 250: 0.10, 500: 0.20, 1000: 0.30, 2000: 0.40, 4000: 0.50},
            FloorMaterial.VINYL: {125: 0.02, 250: 0.03, 500: 0.03, 1000: 0.04, 2000: 0.04, 4000: 0.05},
            FloorMaterial.CONCRETE: {125: 0.02, 250: 0.02, 500: 0.03, 1000: 0.04, 2000: 0.05, 4000: 0.06},
            FloorMaterial.HARDWOOD: {125: 0.08, 250: 0.09, 500: 0.10, 1000: 0.11, 2000: 0.12, 4000: 0.13},
            FloorMaterial.TILE: {125: 0.02, 250: 0.02, 500: 0.03, 1000: 0.03, 2000: 0.04, 4000: 0.05},
            FloorMaterial.RUBBER: {125: 0.04, 250: 0.05, 500: 0.08, 1000: 0.12, 2000: 0.15, 4000: 0.18},
        }

        if isinstance(self.material_type, str):
            # Return generic values for custom materials
            return {125: 0.05, 250: 0.06, 500: 0.08, 1000: 0.09, 2000: 0.10, 4000: 0.11}

        return defaults.get(self.material_type, {125: 0.05, 250: 0.06, 500: 0.08, 1000: 0.09, 2000: 0.10, 4000: 0.11})


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


#------------------------------------------------
#

@dataclass
class RecordingDeviceSpec:
    """Recording device specifications"""
    device_type: RecordingDevice
    sensitivity: float = -40.0  # dBV/Pa
    frequency_response: Tuple[int, int] = (20, 20000)  # Hz range
    snr: float = 60.0  # Signal-to-noise ratio in dB
    directivity_pattern: str = "omnidirectional"  # omnidirectional, cardioid, etc.
    num_channels: int = 1
    position: Position3D = field(default_factory=lambda: Position3D(0, 0, 1.5))

    def __post_init__(self):
        # Set default values based on device type
        device_defaults = {
            RecordingDevice.SMARTPHONE: {"sensitivity": -38.0, "snr": 50.0, "num_channels": 1},
            RecordingDevice.WEBCAM: {"sensitivity": -42.0, "snr": 45.0, "num_channels": 1},
            RecordingDevice.TABLET: {"sensitivity": -40.0, "snr": 48.0, "num_channels": 1},
            RecordingDevice.HIGH_QUALITY_MIC: {"sensitivity": -35.0, "snr": 70.0, "num_channels": 1},
            RecordingDevice.BEAMFORMING_MIC: {"sensitivity": -40.0, "snr": 65.0, "num_channels": 8, "directivity_pattern": "beamformed"},
            RecordingDevice.LAVALIER_MIC: {"sensitivity": -44.0, "snr": 55.0, "num_channels": 1, "directivity_pattern": "omnidirectional"},
            RecordingDevice.SHOTGUN_MIC: {"sensitivity": -35.0, "snr": 65.0, "num_channels": 1, "directivity_pattern": "shotgun"},
        }

        if self.device_type in device_defaults:
            defaults = device_defaults[self.device_type]
            for key, value in defaults.items():
                # Only update if still at default value
                if hasattr(self, key) and getattr(self, key) == getattr(RecordingDeviceSpec.__dataclass_fields__[key], 'default', None):
                    setattr(self, key, value)


#---------------------------------------------------------------------
#



@dataclass
class EnvironmentalConditions:
    """Environmental factors affecting acoustics"""
    temperature: float = 20.0  # Celsius
    humidity: float = 50.0  # Relative humidity %
    atmospheric_pressure: float = 101325.0  # Pa
    background_noise_level: float = 35.0  # dB SPL

    @property
    def sound_speed(self) -> float:
        """Calculate speed of sound based on temperature"""
        return 331.4 + 0.6 * self.temperature

    def air_absorption_coefficient(self) -> float:
        """
        Calculate air absorption coefficient for sound attenuation.

        Returns coefficient in dB/m for 1kHz frequency.
        Based on ISO 9613-1:1993 standard for atmospheric absorption.
        https://www.iso.org/obp/ui/#iso:std:iso:9613:-1:ed-1:v1:en
        """

        T = self.temperature + 273.15 # in Kelvin

        # Saturation vapor pressure (Pa)
        psat = 10**(8.07131 - 1730.63 / (T - 39.724))

        # Molar concentration of water vapor
        h = self.humidity * psat / self.atmospheric_pressure

        # Simplified calculation for 1kHz
        absorption = (1.84e-11 * (self.atmospheric_pressure / 101325) *
                     (T / 293.15)**(-0.5) +
                     (T / 293.15)**(-2.5) *
                     (0.01275 * h * np.exp(-2239.1 / T) /
                      (0.0963 + h * np.exp(-2239.1 / T))))

        return absorption * 1000  # Convert to dB/m


        def calculate_reverberation_time_adjustment(self, room_volume: float,
                                                  base_rt60: float) -> float:
            """
            Calculate adjustment factor for reverberation time based on environmental conditions.

            Args:
                room_volume: Room volume in cubic meters
                base_rt60: Base reverberation time at standard conditions (20°C, 50% RH)

            Returns:
                Adjusted RT60 in seconds

            Environmental factors affect sound absorption:
            - Higher humidity increases absorption at high frequencies
            - Temperature affects air density and sound propagation
            - Atmospheric pressure affects air density

            ![](https://www.mdpi.com/buildings/buildings-12-01282/article_deploy/html/images/buildings-12-01282-g005.png)
            ![](https://www.mdpi.com/buildings/buildings-12-01282/article_deploy/html/images/buildings-12-01282-g009a.png)
            Reverberation times as a function of humidity and air temperature for octave frequencies,
            where φ stands for relative humidity and θ stands for air temperature.

            """
            # Standard conditions (20°C, 50% RH, 101325 Pa)
            standard_temp = 20.0
            standard_humidity = 50.0
            standard_pressure = 101325.0

            temp_factor = (standard_temp + 273.15) / (self.temperature + 273.15)

            # Higher humidity = more absorption = lower RT60
            humidity_factor = 1.0 + (self.humidity - standard_humidity) * 0.002

            pressure_factor = self.atmospheric_pressure / standard_pressure
            env_factor = temp_factor * pressure_factor / humidity_factor

            adjusted_rt60 = base_rt60 * env_factor

            return max(0.01, adjusted_rt60)
