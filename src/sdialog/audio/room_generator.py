"""
This module provides classes for the room generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import time
import random
from abc import abstractmethod
from typing import Tuple, Dict, Any
from sdialog.audio.room import Room, Dimensions3D


class RoomGenerator:
    """
    A room generator is a class that generates a room to be handled by the dialog.
    creating standardized room personas with different configurations
    """

    @abstractmethod
    def calculate_room_dimensions(self, floor_area: float, aspect_ratio: Tuple[float, float]) -> Dimensions3D:
        """
        Calculate room dimensions from floor area
        """
        return None

    @abstractmethod
    def generate(self, args: Dict[str, Any]) -> Room:
        """
        Generate a room based on predefined setups.
        """
        return None


class BasicRoomGenerator(RoomGenerator):
    """
    Generate a basic room based on the floor area and aspect ratio which is
    selected automatically based on the floor area the user provides.
    """

    def __init__(self):
        super().__init__()
        self.aspect_ratio = [
            (1.0, 1.0),
            (1.5, 1.0),
            (2.0, 1.0)
        ]
        self.floor_heights = [
            2.25,
            2.5,
            3.0,
            3.5
        ]

    def calculate_room_dimensions(self, floor_area: float, aspect_ratio: Tuple[float, float]) -> Dimensions3D:
        """
        Calculate room dimensions from floor area
        """
        width_ratio, length_ratio = aspect_ratio

        # Calculate the scaling factor to achieve the desired floor area
        # floor_area = width * length = (width_ratio * k) * (length_ratio * k) = width_ratio * length_ratio * k²
        # Therefore: k = sqrt(floor_area / (width_ratio * length_ratio))
        k = (floor_area / (width_ratio * length_ratio)) ** 0.5

        width = width_ratio * k
        length = length_ratio * k
        height = random.choice(self.floor_heights)
        return Dimensions3D(width=width, length=length, height=height)

    def generate(self, args: Dict[str, Any]) -> Room:
        """
        Generate a room based on predefined setups.
        """

        if "room_size" not in args:
            raise ValueError("room_size is required in m²")

        if len(args) > 1:
            raise ValueError("Only room_size is allowed")

        aspect_ratio = random.choice(self.aspect_ratio)

        dimensions = self.calculate_room_dimensions(args["room_size"], aspect_ratio)

        return Room(
            name=f"room_{time.time_ns()}",
            description=f"room_{time.time_ns()}",
            dimensions=dimensions,
            reverberation_time_ratio=random.uniform(0.3, 0.7),
            aspect_ratio=aspect_ratio
        )
