"""
This module provides specialized room generation for medical environments.

The module includes the MedicalRoomGenerator class that creates realistic
medical room configurations with appropriate dimensions, furniture placement,
and acoustic properties. It supports various medical room types including
consultation rooms, examination rooms, treatment rooms, and surgical suites.

Key Features:

  - Medical room type definitions with standardized dimensions
  - Realistic furniture placement for medical environments
  - Aspect ratio calculations for proper room proportions
  - Support for various medical room configurations
  - Integration with room acoustics simulation

Medical Room Types:

  - Consultation: Small consultation rooms (4.5m²)
  - Examination: Standard examination rooms (6m²)
  - Treatment: Treatment rooms (8m²)
  - Patient Room: Patient rooms (9.5m²)
  - Surgery: Operating rooms (12m²)
  - Waiting: Waiting rooms (15m²)
  - Emergency: Emergency rooms (18m²)
  - Office: Medical offices (20m²)

Example:

    .. code-block:: python

        from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole

        # Create medical room generator
        generator = MedicalRoomGenerator()

        # Generate examination room
        room = generator.generate(args={"room_type": RoomRole.EXAMINATION})

        # Generate random medical room
        random_room = generator.generate(args={"room_type": "random"})
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Pawel Cyrta <pawel@cyrta.com>
# SPDX-License-Identifier: MIT
import time
import math
import random
from enum import Enum
from typing import Tuple, Dict, Any, Optional

from sdialog.audio.room import Room, Dimensions3D
from sdialog.audio.room_generator import RoomGenerator
from sdialog.audio.utils import Furniture, RGBAColor, RoomMaterials


class RoomRole(str, Enum):
    """
    Defines the functional role of medical rooms and their associated dimensions.

    This enumeration provides standardized medical room types with predefined
    dimensions and characteristics. Each room type is designed for specific
    medical functions and includes appropriate size specifications for
    realistic room acoustics simulation.

    :ivar CONSULTATION: Small consultation rooms (4.5m²).
    :vartype CONSULTATION: str
    :ivar EXAMINATION: Standard examination rooms (6m²).
    :vartype EXAMINATION: str
    :ivar TREATMENT: Treatment rooms (8m²).
    :vartype TREATMENT: str
    :ivar PATIENT_ROOM: Patient rooms (9.5m²).
    :vartype PATIENT_ROOM: str
    :ivar SURGERY: Operating rooms (12m²).
    :vartype SURGERY: str
    :ivar WAITING: Waiting rooms (15m²).
    :vartype WAITING: str
    :ivar EMERGENCY: Emergency rooms (18m²).
    :vartype EMERGENCY: str
    :ivar OFFICE: Medical offices (20m²).
    :vartype OFFICE: str
    """

    CONSULTATION = "consultation"
    EXAMINATION = "examination"
    TREATMENT = "treatment"
    PATIENT_ROOM = "patient_room"
    SURGERY = "surgery"  # operating_room
    WAITING = "waiting_room"
    EMERGENCY = "emergency"
    OFFICE = "office"


class MedicalRoomGenerator(RoomGenerator):
    """
    Medical room generator for creating realistic medical environment configurations.

    MedicalRoomGenerator extends the base RoomGenerator to create specialized
    medical room configurations with appropriate dimensions, furniture placement,
    and acoustic properties. It supports various medical room types with
    standardized dimensions and realistic furniture arrangements.

    Key Features:

      - Medical room type definitions with standardized dimensions
      - Realistic furniture placement for medical environments
      - Aspect ratio calculations for proper room proportions
      - Support for various medical room configurations
      - Integration with room acoustics simulation

    :ivar ROOM_SIZES: Dictionary mapping room types to dimensions and descriptions.
    :vartype ROOM_SIZES: Dict[RoomRole, Tuple[float, str, str]]
    :ivar ROOM_ASPECT_RATIOS: Dictionary mapping floor areas to aspect ratios.
    :vartype ROOM_ASPECT_RATIOS: Dict[float, Tuple[float, float]]
    """

    def __init__(self, seed: Optional[int] = time.time_ns()):
        super().__init__(seed)

        # Standard room sizes (floor area in m²): size, name, description
        self.ROOM_SIZES: Dict[RoomRole, Tuple[float, str, str]] = {
            RoomRole.CONSULTATION: (4.5, "consultation_room", "consultation room"),
            RoomRole.EXAMINATION: (6, "examination_room", "examination room"),
            RoomRole.TREATMENT: (8, "treatment_room", "treatment room"),
            RoomRole.PATIENT_ROOM: (9.5, "patient_room", "patient room"),
            RoomRole.SURGERY: (12, "surgery_room", "surgery room"),
            RoomRole.WAITING: (15, "waiting_room", "waiting room"),
            RoomRole.EMERGENCY: (18, "emergency_room", "emergency room"),
            RoomRole.OFFICE: (20, "office_room", "office room"),
        }

        # Standard aspect ratios for different room sizes (width:length)
        self.ROOM_ASPECT_RATIOS = {
            4.5: (1.5, 1.0),  # 2.12 x 2.12m (compact square)
            6: (1.5, 1.0),  # 2.45 x 2.45m
            8: (1.6, 1.0),  # 3.58 x 2.24m (slightly rectangular)
            9.5: (1.7, 1.0),  # 4.0 x 2.35m
            12: (1.8, 1.0),  # 4.65 x 2.58m
            15: (2.0, 1.0),  # 5.48 x 2.74m
            18: (2.2, 1.0),  # 6.26 x 2.87m
            20: (2.5, 1.0),  # 7.07 x 2.83m
            24: (2.4, 1.0),  # 7.59 x 3.16m
            32: (2.8, 1.0),  # 9.49 x 3.37m (long rectangular)
        }

    def calculate_room_dimensions(self, floor_area: float, aspect_ratio: Tuple[float, float]) -> Dimensions3D:
        """
        Calculate room dimensions from floor area and aspect ratio.

        Computes the width, length, and height of a room based on the specified
        floor area and aspect ratio. The height is fixed at 2.5 meters for
        medical rooms to maintain realistic proportions.

        :param floor_area: Floor area of the room in square meters.
        :type floor_area: float
        :param aspect_ratio: Width to length ratio as a tuple (width_ratio, length_ratio).
        :type aspect_ratio: Tuple[float, float]
        :return: Room dimensions with calculated width, length, and height.
        :rtype: Dimensions3D
        """

        w_ratio, l_ratio = aspect_ratio

        length = math.sqrt(floor_area / (w_ratio / l_ratio))
        width = length * (w_ratio / l_ratio)

        return Dimensions3D(width=width, length=length, height=2.5)

    def generate(self, args: Dict[str, Any]) -> Room:
        """
        Generate a medical room based on predefined room type configurations.

        Creates a complete medical room configuration with appropriate dimensions,
        furniture placement, and acoustic properties based on the specified
        room type. The room includes standard medical furniture such as desks,
        monitors, examination benches, sinks, and cupboards.

        :param args: Dictionary containing room generation parameters.
        :type args: Dict[str, Any]
        :return: Complete medical room configuration with furniture and materials.
        :rtype: Room
        """

        if "room_type" not in args:
            raise ValueError("room_type is required")

        if len(args) > 1:
            raise ValueError("Only room_type is allowed")

        if args["room_type"] == "random":
            args["room_type"] = random.choice(list(RoomRole.__members__.values()))

        floor_area, name, description = self.ROOM_SIZES[args["room_type"]]

        if floor_area not in self.ROOM_ASPECT_RATIOS:
            raise ValueError(f"Unsupported room size: {floor_area}m²")

        w_ratio, l_ratio = self.ROOM_ASPECT_RATIOS[floor_area]

        # Time in nanoseconds
        time_in_ns = time.time_ns()

        # Calculate room dimensions
        dims = self.calculate_room_dimensions(floor_area, (w_ratio, l_ratio))

        room = Room(
            name=f"{name} - {time_in_ns}",
            description=f"{description} - {time_in_ns}",
            dimensions=dims,
            # reverberation_time_ratio=0.18,
            materials=RoomMaterials(),
            furnitures={
                "desk": Furniture(
                    name="desk",
                    x=dims.width * 0.01,
                    y=dims.length * 0.15,
                    width=1.22,
                    height=0.76,
                    depth=0.76,
                    color=RGBAColor.GREEN
                ),
                "monitor": Furniture(
                    name="monitor",
                    x=dims.width * 0.01,
                    y=dims.length * 0.15,
                    z=0.8,
                    width=0.5,
                    height=0.4,
                    depth=0.10,
                    color=RGBAColor.BROWN
                ),
                "bench": Furniture(
                    name="bench",
                    x=dims.width * 0.65,
                    y=dims.length * 0.01,
                    width=0.82,
                    height=0.75,
                    depth=1.95,
                    color=RGBAColor.ORANGE
                ),
                "sink": Furniture(
                    name="sink",
                    x=dims.width * 0.35,
                    y=dims.length * 0.75,
                    width=0.4,
                    height=1.0,
                    depth=0.4
                ),
                "cupboard": Furniture(
                    name="cupboard",
                    x=dims.width * 0.01,
                    y=dims.length * 0.75,
                    width=0.9,
                    height=1.85,
                    depth=0.4
                ),
                "door": Furniture(
                    name="door",
                    x=0.01,
                    y=0.01,
                    width=0.70,
                    height=2.10,
                    depth=0.10,
                    color=RGBAColor.BLACK
                )
            }
        )

        return room
