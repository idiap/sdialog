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

    def get_data(self) -> dict:
        """
        Get the data of the voice database.
        """
        return self._data

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

        # If the voice is not in the database, find the closest age for this gender
        if (genre, age) not in self._data:

            # Get the list of ages for this gender
            _ages = [age for (genre, age) in self._data.keys() if genre == genre]

            # Get the closest age for this gender
            age = min(_ages, key=lambda x: abs(x - age))

        # Get the voices from the database for this gender and age
        _subset = self._data[(genre, age)]

        # Randomly sample a voice from the database for this gender and age
        return random.choice(_subset)


class DummyVoiceDatabase(BaseVoiceDatabase):
    """
    Dummy voice database.
    """

    def __init__(self):
        BaseVoiceDatabase.__init__(self)

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
            ] for age in range(0, 150, 1)
        }
        females_voices = {
            ("female", age): [
                {"identifier": voice_name, "path": f"{voice_name}.wav"} for voice_name in self._womans
            ] for age in range(0, 150, 1)
        }
        self._data = {**males_voices, **females_voices}


class HuggingfaceVoiceDatabase(BaseVoiceDatabase):
    """
    Huggingface voice database.
    """

    def __init__(
            self,
            dataset_name: str = "sdialog/voices-libritts",
            subset: str = "train"):

        self.dataset_name = dataset_name
        self.subset = subset
        BaseVoiceDatabase.__init__(self)

    def _gender_to_gender(self, gender: str) -> str:
        """
        Convert the gender to the gender.
        """
        return "male" if gender == "M" else "female"

    def populate(self) -> dict:
        """
        Populate the voice database.
        """
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name)[self.subset]

        self._data = {
            (self._gender_to_gender(d["gender"]), d["age"]): [
                {
                    "identifier": d["speaker_id"],
                    "path": d["audio"]["path"]
                }
            ] for d in dataset
        }
