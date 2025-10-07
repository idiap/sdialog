"""
This module provides a voice database.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import random
import logging


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
        logging.info(f"Voice database populated with {len(self._data)} voices")


class LocalVoiceDatabase(BaseVoiceDatabase):
    """
    Local voice database.
    """

    def __init__(
            self,
            directory_audios: str,
            metadata_file: str):

        self.directory_audios = directory_audios
        self.metadata_file = metadata_file
        BaseVoiceDatabase.__init__(self)

        # check if the directory audios exists
        if not os.path.exists(self.directory_audios):
            raise ValueError(f"Directory audios does not exist: {self.directory_audios}")

        # check if the metadata file exists
        if not os.path.exists(self.metadata_file):
            raise ValueError(f"Metadata file does not exist: {self.metadata_file}")

        # check if the directory audios is a directory
        if not os.path.isdir(self.directory_audios):
            raise ValueError(f"Directory audios is not a directory: {self.directory_audios}")

        # check if the metadata file is a csv / tsv / json file
        if (
            not self.metadata_file.endswith(".csv") and
            not self.metadata_file.endswith(".tsv") and
            not self.metadata_file.endswith(".json")
        ):
            raise ValueError(f"Metadata file is not a csv / tsv / json file: {self.metadata_file}")

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

        if self.metadata_file.endswith(".csv"):
            df = pd.read_csv(self.metadata_file)
        elif self.metadata_file.endswith(".tsv"):
            df = pd.read_csv(self.metadata_file, delimiter="\t")
        elif self.metadata_file.endswith(".json"):
            df = pd.read_json(self.metadata_file)
        else:
            raise ValueError(f"Metadata file is not a csv / tsv / json file: {self.metadata_file}")

        # check if the audio file column exists
        if "audio_file" not in df.columns:
            raise ValueError(f"Audio file column does not exist in the metadata file: {self.metadata_file}")

        # check if the gender column exists
        if "gender" not in df.columns:
            raise ValueError(f"Gender column does not exist in the metadata file: {self.metadata_file}")

        # check if the age column exists
        if "age" not in df.columns:
            raise ValueError(f"Age column does not exist in the metadata file: {self.metadata_file}")

        # check if the speaker id column exists
        if "speaker_id" not in df.columns:
            raise ValueError(f"Speaker id column does not exist in the metadata file: {self.metadata_file}")

        self._data = {
            (self._gender_to_gender(row["gender"]), row["age"]): [
                {
                    "identifier": row["speaker_id"],
                    "voice": os.path.join(self.directory_audios, row["audio_file"])
                }
            ] for index, row in df.iterrows()
        }
        logging.info(f"Voice database populated with {len(self._data)} voices")
