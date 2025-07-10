"""
This module provides classes for the room generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
from sdialog.audio.room import Room


class RoomGenerator:
    """
    A room generator is a class that generates a room to be handled by the dialog.
    """

    def generate(self, room_type: str) -> Room:
        """
        Generate a room based on predefined setups.
        """

        if room_type == "office":
            return Room(
                name="office",
                description="office"
            )
        return Room(
            name="room",
            description="room"
        )
