"""
This module provides a voice database.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
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

    def add_voice(
            self,
            genre: str,
            age: int,
            identifier: str,
            path: str):
        """
        Add a voice to the database.
        """
        if (genre, age) not in self._data:
            self._data[(genre, age)] = []

        self._data[(genre, age)].append({
            "identifier": identifier,
            "voice": path
        })

    def get_voice(
            self,
            genre: str,
            age: int) -> dict:
        """
        Random sampling of voice from the database.
        """
        genre = genre.lower()

        # If the voice is not in the database, find the closest age for this gender
        if (genre, age) not in self._data:

            # Get the list of ages for this gender
            _ages = [_age for (_genre, _age) in self._data.keys() if _genre == genre]
            # add shuffle the list
            random.shuffle(_ages)
            random.shuffle(_ages)
            random.shuffle(_ages)

            # Get the closest age for this gender
            age = min(_ages, key=lambda x: abs(x - age))

        # Get the voices from the database for this gender and age
        _subset = self._data[(genre, age)]

        # Randomly sample a voice from the database for this gender and age
        return random.choice(_subset)


class DummyKokoroVoiceDatabase(BaseVoiceDatabase):
    """
    Dummy voice database for Kokoro.
    """

    def __init__(self):
        BaseVoiceDatabase.__init__(self)

    def populate(self) -> dict:
        """
        Populate the voice database with the voices from Kokoro.
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
                {
                    "identifier": voice_name,
                    "voice": voice_name
                } for voice_name in self._mans
            ] for age in range(0, 150, 1)
        }
        females_voices = {
            ("female", age): [
                {
                    "identifier": voice_name,
                    "voice": voice_name
                } for voice_name in self._womans
            ] for age in range(0, 150, 1)
        }
        self._data = {
            **males_voices,
            **females_voices
        }


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

    def _gender_to_gender(
            self,
            gender: str) -> str:
        """
        Convert the gender to the gender.
        """
        gender = gender.lower()

        if gender == "m":
            return "male"

        if gender == "f":
            return "female"

        if gender not in ["male", "female"]:
            raise ValueError(f"Invalid gender: {gender}")

        return gender

    def populate(self) -> dict:
        """
        Populate the voice database.
        """
        from datasets import load_dataset, load_from_disk

        if os.path.exists(self.dataset_name):
            dataset = load_from_disk(self.dataset_name)[self.subset]
        else:
            dataset = load_dataset(self.dataset_name)[self.subset]

        self._data = {
            (self._gender_to_gender(d["gender"]), d["age"]): [
                {
                    "identifier": d["speaker_id"],
                    "voice": d["audio"]["path"]
                }
            ] for d in dataset
        }


class JsaltDummyIndexTtsVoiceDatabase(BaseVoiceDatabase):
    """
    JSALT DummyIndexTts voice database.
    Made for Jean-Zay cluster only.
    """

    def __init__(self):
        BaseVoiceDatabase.__init__(self)

    def _gender_to_gender(
            self,
            gender: str) -> str:
        """
        Convert the gender to the gender.
        """
        gender = gender.lower()

        if gender == "m":
            return "male"

        if gender == "f":
            return "female"

        if gender not in ["male", "female"]:
            raise ValueError(f"Invalid gender: {gender}")

        return gender

    def populate(self) -> dict:
        """
        Populate the voice database.
        """
        dataset = [
            {"speaker_id": 1, "age": 28, "gender": "M", "name": "Sergio Burdisso"},
            {"speaker_id": 2, "age": 26, "gender": "M", "name": "Yanis Labrak"},
            {"speaker_id": 3, "age": 21, "gender": "F", "name": "Isabella Gidi"},
            {"speaker_id": 4, "age": 21, "gender": "F", "name": "Isabella Gidi Texan"}
        ]

        root_voices_path = "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio/Generation/"
        sergio_voice = root_voices_path + "sergio-sample.wav"
        isabella_normal = root_voices_path + "isabella.wav"
        isabella_texan = root_voices_path + "isabella-texan.wav"

        for d in dataset:
            _gender = self._gender_to_gender(d["gender"])
            voice_attributed = sergio_voice if _gender == "m" else random.choice([isabella_normal, isabella_texan])
            self._data[(_gender, d["age"])] = [
                {
                    "identifier": d["speaker_id"],
                    "voice": voice_attributed
                }
            ]


class DummyIndexTtsVoiceDatabase(BaseVoiceDatabase):
    """
    DummyIndexTts voice database.
    Downloaded from git clone https://huggingface.co/datasets/sdialog/voices-libritts
    """

    def __init__(self, path_dir):
        self.path_dir = path_dir
        BaseVoiceDatabase.__init__(self)

    def _gender_to_gender(
            self,
            gender: str) -> str:
        """
        Convert the gender to the gender.
        """
        gender = gender.lower()

        if gender == "m":
            return "male"

        if gender == "f":
            return "female"

        if gender not in ["male", "female"]:
            raise ValueError(f"Invalid gender: {gender}")

        return gender

    def populate(self) -> dict:
        """
        Populate the voice database.
        """
        import pandas as pd

        df = pd.read_csv(os.path.join(self.path_dir, "metadata.csv"))
        print(df)

        for index, row in df.iterrows():

            _gender = self._gender_to_gender(row["gender"])

            if (_gender, row["age"]) not in self._data:
                self._data[(_gender, row["age"])] = []

            self._data[(_gender, row["age"])].append(
                {
                    "identifier": row["speaker_id"],
                    "voice": os.path.join(self.path_dir, row["file_name"])
                }
            )
