"""
This module provides base classes for room generation in the sdialog library.

The module includes abstract and concrete room generator classes that create
realistic room configurations with appropriate dimensions, furniture placement,
and acoustic properties. It provides a flexible framework for generating
various types of rooms for audio simulation.

Key Components:

  - RoomGenerator: Abstract base class for room generation
  - BasicRoomGenerator: Concrete implementation for basic room generation
  - Support for customizable room dimensions and furniture placement
  - Integration with room acoustics simulation

Example:

    .. code-block:: python

        from sdialog.audio.room_generator import BasicRoomGenerator

        # Create basic room generator
        generator = BasicRoomGenerator()

        # Generate a room with specified floor area
        room = generator.generate(args={"room_size": 20.0})
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Pawel Cyrta <pawel@cyrta.com>
# SPDX-License-Identifier: MIT
import time
import random
from abc import abstractmethod
from typing import Tuple, Dict, Any, Optional

from sdialog.audio.utils import Furniture, RoomMaterials
from sdialog.audio.room import Room, Dimensions3D, MicrophonePosition


class RoomGenerator:
    """
    Abstract base class for room generation in audio simulation.

    RoomGenerator provides the interface for creating realistic room configurations
    with appropriate dimensions, furniture placement, and acoustic properties.
    Subclasses should implement the abstract methods to provide specific room
    generation logic for different room types and configurations.

    Key Features:

      - Abstract interface for room generation
      - Support for customizable room dimensions and furniture placement
      - Integration with room acoustics simulation
      - Seed-based randomization for reproducible results

    :ivar seed: Random seed for reproducible room generation.
    :vartype seed: int
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else time.time_ns()
        self.rng = random.Random(self.seed) if self.seed is not None else random

    @abstractmethod
    def calculate_room_dimensions(self, floor_area: float, aspect_ratio: Tuple[float, float]) -> Dimensions3D:
        """
        Calculate room dimensions from floor area and aspect ratio.

        :param floor_area: Floor area of the room in square meters.
        :type floor_area: float
        :param aspect_ratio: Width to length ratio as a tuple (width_ratio, length_ratio).
        :type aspect_ratio: Tuple[float, float]
        :return: Room dimensions with calculated width, length, and height.
        :rtype: Dimensions3D
        """
        return None

    @abstractmethod
    def generate(self, args: Dict[str, Any]) -> Room:
        """
        Generate a room based on predefined configurations.

        :param args: Dictionary containing room generation parameters.
        :type args: Dict[str, Any]
        :return: Complete room configuration with furniture and materials.
        :rtype: Room
        """
        return None


class BasicRoomGenerator(RoomGenerator):
    """
    Basic room generator for creating simple room configurations.

    BasicRoomGenerator creates basic room configurations with customizable
    floor area and automatically selected aspect ratios. It provides a
    simple interface for generating rooms with basic furniture placement
    and standard acoustic properties.

    Key Features:

      - Automatic aspect ratio selection based on floor area
      - Random height selection from predefined options
      - Basic furniture placement (door)
      - Support for customizable room dimensions

    :ivar aspect_ratio: List of available aspect ratios for room generation.
    :vartype aspect_ratio: List[Tuple[float, float]]
    :ivar floor_heights: List of available floor heights for room generation.
    :vartype floor_heights: List[float]
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.aspect_ratio = [
            (1.0, 1.0),
            (1.5, 1.0),
            (2.0, 1.0),
        ]
        self.floor_heights = [
            2.5,
            3.0,
            3.5
        ]

    def calculate_room_dimensions(self, floor_area: float, aspect_ratio: Tuple[float, float]) -> Dimensions3D:
        """
        Calculate room dimensions from floor area and aspect ratio.

        Computes the width, length, and height of a room based on the specified
        floor area and aspect ratio. The height is randomly selected from
        predefined options to add variety to the room configurations.

        :param floor_area: Floor area of the room in square meters.
        :type floor_area: float
        :param aspect_ratio: Width to length ratio as a tuple (width_ratio, length_ratio).
        :type aspect_ratio: Tuple[float, float]
        :return: Room dimensions with calculated width, length, and random height.
        :rtype: Dimensions3D
        """
        width_ratio, length_ratio = aspect_ratio

        # Calculate the scaling factor to achieve the desired floor area
        # floor_area = width * length = (width_ratio * k) * (length_ratio * k) = width_ratio * length_ratio * k²
        # Therefore: k = sqrt(floor_area / (width_ratio * length_ratio))
        k = (floor_area / (width_ratio * length_ratio)) ** 0.5

        width = width_ratio * k
        length = length_ratio * k

        height = self.rng.choice(self.floor_heights)

        return Dimensions3D(width=width, length=length, height=height)

    def generate(self, args: Dict[str, Any]) -> Room:
        """
        Generate a basic room configuration with specified floor area.

        Creates a basic room with the specified floor area, automatically
        selecting an appropriate aspect ratio and random height. The room
        includes basic furniture placement with a standard door configuration.

        :param args: Dictionary containing room generation parameters.
        :type args: Dict[str, Any]
        :return: Complete basic room configuration with furniture and materials.
        :rtype: Room
        """

        if "room_size" not in args:
            raise ValueError("room_size is required in m²")

        if len(args) > 1:
            raise ValueError("Only room_size is allowed")

        aspect_ratio = self.rng.choice(self.aspect_ratio)

        dims = self.calculate_room_dimensions(args["room_size"], aspect_ratio)

        room = Room(
            name=f"room_{time.time_ns()}",
            description=f"room_{time.time_ns()}",
            dimensions=dims,
            reverberation_time_ratio=None,
            materials=RoomMaterials(),
            mic_position=MicrophonePosition.CEILING_CENTERED,
            furnitures={
                "door": Furniture(
                    name="door",
                    x=0.10,
                    y=0.10,
                    width=0.70,
                    height=2.10,
                    depth=0.5
                )
            }
        )

        return room
