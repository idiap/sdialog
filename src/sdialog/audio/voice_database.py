"""
This module provides a voice database.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import random
from typing import List


class BaseVoiceDatabase:
    
    def __init__(self, data: dict = {}):
        """
        Initialize the voice database.
        """
        self._data = data

    def add_voice(self, genre: str, age: int, identifier: str, path: str):
        """
        Add a voice to the database.
        """
        if (genre, age) not in self._data:
            self._data[(genre, age)] = []
        self._data[(genre, age)].append({"identifier": identifier, "path": path})

    def get_voice(self, genre: str, age: int) -> dict:
        """
        Random sampling of voice from the database.
        """
        return random.choice(self._data[(genre, age)])


class DummyVoiceDatabase(BaseVoiceDatabase):

    def __init__(self):
        BaseVoiceDatabase.__init__(self, data={
            ("male", 45): [
                {"identifier": "af_heart", "path": "af_heart.wav"},
                {"identifier": "af_heart", "path": "af_heart.wav"}
            ],
            ("female", 23): [
                {"identifier": "af_heart", "path": "af_heart.wav"},
                {"identifier": "af_heart", "path": "af_heart.wav"}
            ]
        })
