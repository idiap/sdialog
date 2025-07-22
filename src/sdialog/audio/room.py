"""
This module provides classes for the room specification.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from typing import List


class Room:
    """
    A room is a place where the dialog takes place.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @staticmethod
    def get_microphone_positions() -> List[str]:
        """
        Get the microphone positions.
        """
        return [
            "table_smartphone",
            "monitor",
            "wall_mounted",
            "ceiling_centered",
            "chest_pocket"
        ]
