"""
This module provides classes for the room generation.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import math

from sdialog.audio.room import Room, Dimensions3D, RoomRole
from typing import List


# Standard room sizes (floor area in m²)
ROOM_SIZES: List[float] = [4.5, 6, 8, 9.5, 12, 15, 18]


# Standard aspect ratios for different room sizes (width:length)
ROOM_ASPECT_RATIOS = {
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


def calculate_room_dimensions(floor_area: float) -> Dimensions3D:
    """Calculate room dimensions from floor area"""
    if floor_area not in ROOM_ASPECT_RATIOS:
        raise ValueError(f"Unsupported room size: {floor_area}m²")

    w_ratio, l_ratio = ROOM_ASPECT_RATIOS[floor_area]
    length = math.sqrt(floor_area / (w_ratio / l_ratio))
    width = length * (w_ratio / l_ratio)

    return Dimensions3D(width=width, length=length, height=3.0)


class RoomGenerator:
    """
    A room generator is a class that generates a room to be handled by the dialog.
    creating standardized room personas with different configurations
    """

    def __init__(self):
        self.generated_rooms = {}

    def generate(self, room_type: RoomRole) -> Room:
        """
        Generate a room based on predefined setups.
        """

        if room_type == RoomRole.OFFICE:
            return Room(
                role=RoomRole.OFFICE,
                name="RoomRole.OFFICE" + " room",
                description="office",
                dimensions=calculate_room_dimensions(ROOM_SIZES[4]),
                rt60=0.3,
            )
        return Room(
            role=RoomRole.CONSULTATION,
            name="RoomRole.CONSULTATION" + " room",
            description="consultation room",
            dimensions=calculate_room_dimensions(ROOM_SIZES[3]),
            rt60=0.5,
        )


if __name__ == "__main__":
    print(" Room Generator creates:")
    generator = RoomGenerator()
    room = generator.generate(RoomRole.CONSULTATION)
    print(f"  Room {room}")
