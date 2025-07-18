"""
This module provides a voice database.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import random


class BaseVoiceDatabase:
    """
    Base class for voice databases.
    """

    def __init__(self):
        """
        Initialize the voice database.
        """
        self._data = {}
        self.populate()

    def populate(self) -> dict:
        """
        Populate the voice database.
        """
        self._data = {}

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
        _subset = self._data[(genre, age)]
        return random.choice(_subset)


class DummyVoiceDatabase(BaseVoiceDatabase):
    """
    Dummy voice database.
    """

    def __init__(self):
        BaseVoiceDatabase.__init__(self)
    
    def get_data(self) -> dict:
        """
        Get the data of the voice database.
        """
        return self._data

    def populate(self) -> dict:
        """
        Populate the voice database.
        """
        self._womans = [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"
        ]
        self._mans = [
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
            "am_onyx", "am_puck"
        ]

        males_voices = {
            ("male", age): [
                {"identifier": voice_name, "path": f"{voice_name}.wav"} for voice_name in self._mans
            ] for age in range(0,150,1)
        }
        females_voices = {
            ("female", age): [
                {"identifier": voice_name, "path": f"{voice_name}.wav"} for voice_name in self._womans
            ] for age in range(0,150,1)
        }
        self._data = {**males_voices, **females_voices}
