"""
This module provides classes for medical room generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
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
    Defines the functional role of the room and dimentions that comes with it.
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
    A room generator is a class that generates a room to be handled by the dialog.
    creating standardized room personas with different configurations
    """

    def __init__(self, seed: Optional[int] = time.time_ns()):
        super().__init__(seed)

        # Standard room sizes (floor area in m²): size, name, description
        self.ROOM_SIZES: Dict[RoomRole, Tuple[float, str, str]] = {
            RoomRole.CONSULTATION: (4.5, "consultation_room", "consultation room"),
            RoomRole.EXAMINATION:  (6,   "examination_room",  "examination room"),
            RoomRole.TREATMENT:    (8,   "treatment_room",    "treatment room"),
            RoomRole.PATIENT_ROOM: (9.5, "patient_room",      "patient room"),
            RoomRole.SURGERY:      (12,  "surgery_room",      "surgery room"),
            RoomRole.WAITING:      (15,  "waiting_room",      "waiting room"),
            RoomRole.EMERGENCY:    (18,  "emergency_room",    "emergency room"),
            RoomRole.OFFICE:       (20,  "office_room",       "office room"),
        }

        # Standard aspect ratios for different room sizes (width:length)
        self.ROOM_ASPECT_RATIOS = {
            4.5: (1.5, 1.0),  # 2.12 x 2.12m (compact square)
            6:   (1.5, 1.0),  # 2.45 x 2.45m
            8:   (1.6, 1.0),  # 3.58 x 2.24m (slightly rectangular)
            9.5: (1.7, 1.0),  # 4.0 x 2.35m
            12:  (1.8, 1.0),  # 4.65 x 2.58m
            15:  (2.0, 1.0),  # 5.48 x 2.74m
            18:  (2.2, 1.0),  # 6.26 x 2.87m
            20:  (2.5, 1.0),  # 7.07 x 2.83m
            24:  (2.4, 1.0),  # 7.59 x 3.16m
            32:  (2.8, 1.0),  # 9.49 x 3.37m (long rectangular)
        }

    def calculate_room_dimensions(self, floor_area: float, aspect_ratio: Tuple[float, float]) -> Dimensions3D:
        """
        Calculate room dimensions from floor area
        floor_area: float
        aspect_ratio: Tuple[float, float]
        """

        w_ratio, l_ratio = aspect_ratio

        length = math.sqrt(floor_area / (w_ratio / l_ratio))
        width = length * (w_ratio / l_ratio)

        return Dimensions3D(width=width, length=length, height=2.5)

    def generate(self, args: Dict[str, Any]) -> Room:
        """
        Generate a room based on predefined medical room setups.
        args:
            room_type: RoomRole
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
