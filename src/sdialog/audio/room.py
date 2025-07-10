"""
This module provides classes for the room specification.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Pawel Cyrta <pawel@cyrta.com>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

class Room:
    """
    A room is a place where the dialog takes place.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description