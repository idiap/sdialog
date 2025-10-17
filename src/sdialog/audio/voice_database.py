"""
This module provides a voice database.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import random
import logging
from pydantic import BaseModel
from typing import List, Union
from sdialog.util import dict_to_table
from collections import defaultdict, Counter


def is_a_audio_file(file: str) -> bool:
    """
    Check if the file is a audio file.
    """
    file = file.lower()
    if (
        ".wav" in file or
        ".mp3" in file or
        ".m4a" in file or
        ".ogg" in file or
        ".flac" in file or
        ".aiff" in file or
        ".aif" in file or
        ".aac" in file
    ):
        return True
    return False


class Voice(BaseModel):
    """
    Voice class.
    """
    gender: str
    age: int
    identifier: str
    voice: str  # Can be a path or the voice string
    language: str = "english"
    language_code: str = "e"


class BaseVoiceDatabase:
    """
    Base class for voice databases.
    """

    def __init__(self):
        """
        Initialize the voice database.
        """

        # Dictionary to keep track of the voices: language -> (gender, age) -> list of voices
        self._data: dict[str, dict[tuple[str, int], List[Voice]]] = {}

        # Dictionary to keep track of the used voices: language -> list of identifiers
        self._used_voices: dict[str, List[str]] = {}

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

    def reset_used_voices(self):
        """
        Reset the used voices.
        """
        self._used_voices = {}

    def get_statistics(self, pretty: bool = False, pretty_format: str = "markdown") -> Union[dict, str]:
        """
        Get comprehensive statistics about the voice database.

        Returns a nested dict including:
            - num_languages: number of languages in the database
            - overall: global stats (total, by_gender, ages)
            - languages: per-language stats with totals, by_gender, age distributions,
              by_gender_age, unique speaker count, observed language codes, and age stats
        If pretty=True, returns a formatted string (Markdown if pretty_format=="markdown", otherwise fancy grid).
        """
        # Global accumulators
        overall_total = 0
        overall_by_gender: Counter = Counter()
        overall_ages: Counter = Counter()

        languages_stats: dict = {}

        for lang, gender_age_to_voices in self._data.items():
            # Per-language accumulators
            lang_total = 0
            lang_by_gender: Counter = Counter()
            lang_ages: Counter = Counter()
            lang_by_gender_age: dict = defaultdict(Counter)  # gender -> Counter(age -> count)
            unique_identifiers = set()
            observed_lang_codes = set()

            for (gender, age), voices in gender_age_to_voices.items():
                count = len(voices)
                lang_total += count
                lang_by_gender[gender] += count
                lang_ages[age] += count
                lang_by_gender_age[gender][age] += count

                # Collect identifiers and language codes
                for v in voices:
                    unique_identifiers.add(v.identifier)
                    if getattr(v, "language_code", None) is not None:
                        observed_lang_codes.add(v.language_code)

            # Update overall accumulators
            overall_total += lang_total
            overall_by_gender.update(lang_by_gender)
            overall_ages.update(lang_ages)

            # Compute simple age stats (weighted by counts)
            if lang_ages:
                ages_list = []
                for a, c in lang_ages.items():
                    ages_list.extend([a] * c)
                age_min = min(lang_ages.keys())
                age_max = max(lang_ages.keys())
                age_mean = sum(ages_list) / len(ages_list)
            else:
                age_min = None
                age_max = None
                age_mean = None

            languages_stats[lang] = {
                "total": lang_total,
                "by_gender": dict(lang_by_gender),
                "ages": dict(lang_ages),
                "by_gender_age": {g: dict(c) for g, c in lang_by_gender_age.items()},
                "unique_speakers": len(unique_identifiers),
                "language_codes": sorted(observed_lang_codes),
                "age_stats": {
                    "min": age_min,
                    "max": age_max,
                    "mean": age_mean,
                },
            }

        stats = {
            "num_languages": len(self._data),
            "overall": {
                "total": overall_total,
                "by_gender": dict(overall_by_gender),
                "ages": dict(overall_ages),
            },
            "languages": languages_stats,
        }

        if not pretty:
            return stats

        # Build pretty tables
        is_markdown = (pretty_format.lower() == "markdown")

        # 1) Languages summary table
        lang_rows = {}
        for lang, info in languages_stats.items():
            row = {
                "total": info.get("total", 0),
                "male": info.get("by_gender", {}).get("male", 0),
                "female": info.get("by_gender", {}).get("female", 0),
                "unique_speakers": info.get("unique_speakers", 0),
                "age_min": (info.get("age_stats", {}) or {}).get("min", None),
                "age_mean": (info.get("age_stats", {}) or {}).get("mean", None),
                "age_max": (info.get("age_stats", {}) or {}).get("max", None),
                "codes": ",".join(info.get("language_codes", [])),
            }
            lang_rows[lang] = row

        summary_table = dict_to_table(
            lang_rows,
            sort_by="total",
            sort_ascending=False,
            markdown=is_markdown,
            format=".2f",
            show=False,
        )

        # 2) Overall summary small block
        overall = stats["overall"]
        overall_lines = []
        overall_lines.append(f"Number of languages: {stats['num_languages']}")
        overall_lines.append(f"Total voices: {overall['total']}")
        # By gender
        og = overall.get("by_gender", {})
        overall_lines.append("By gender: " + ", ".join([f"{g}: {c}" for g, c in og.items()]))
        # Ages (top few)
        oa = overall.get("ages", {})
        if oa:
            # show up to 10 age bins sorted
            top_ages = sorted(oa.items())[:10]
            overall_lines.append("Ages (first 10 bins sorted): " + ", ".join([f"{a}:{c}" for a, c in top_ages]))

        blocks = []
        # Title
        if is_markdown:
            blocks.append("### Voice Database Statistics")
            blocks.append("")
            blocks.append("#### Overall")
        else:
            blocks.append("Voice Database Statistics\n")
            blocks.append("Overall")
        blocks.append("\n".join(overall_lines))
        blocks.append("")

        # Languages table
        if is_markdown:
            blocks.append("#### By Language (summary)")
        else:
            blocks.append("By Language (summary)")
        blocks.append(summary_table)

        # 3) Optional: Per-language gender-age breakdown (compact)
        for lang, info in languages_stats.items():
            if is_markdown:
                blocks.append("")
                blocks.append(f"#### {lang} — gender/age distribution")
            else:
                blocks.append("")
                blocks.append(f"{lang} — gender/age distribution")

            by_gender_age = info.get("by_gender_age", {})
            # Convert to a table with (gender_age) columns or separate rows
            # We'll render as a dict-of-dicts: age rows, gender columns
            ages_set = set()
            for g, counter in by_gender_age.items():
                ages_set.update(counter.keys())
            ages_list = sorted(ages_set)

            table_map = {}
            for age in ages_list:
                row = {}
                for g in sorted(by_gender_age.keys()):
                    row[g] = by_gender_age[g].get(age, 0)
                table_map[str(age)] = row

            if table_map:
                ga_table = dict_to_table(
                    table_map,
                    markdown=is_markdown,
                    show=False,
                )
                blocks.append(ga_table)
            else:
                blocks.append("(no data)")

        return "\n\n".join(blocks)

    def add_voice(
            self,
            gender: str,
            age: int,
            identifier: str,
            voice: str,
            lang: str,
            language_code: str):
        """
        Add a voice to the database.
        """
        if lang not in self._data:
            self._data[lang] = {}

        if (gender, age) not in self._data[lang]:
            self._data[lang][(gender, age)] = []

        self._data[lang][(gender, age)].append(Voice(
            gender=gender.lower(),
            age=age,
            identifier=identifier,
            voice=voice,
            language=lang.lower(),
            language_code=language_code.lower()
        ))

    def get_voice_by_identifier(
        self,
        identifier: str,
        lang: str,
        keep_duplicate: bool = True  # If True, the voice will be returned even if it is already used
    ) -> Voice:
        """
        Get a voice by its identifier.
        """
        if lang not in self._data:
            raise ValueError(f"Language {lang} not found in the database")

        for (gender, age), voices in self._data[lang].items():
            for voice in voices:
                if voice.identifier == identifier:
                    if not keep_duplicate:
                        if voice.identifier in self._used_voices[lang]:
                            raise ValueError(f"Voice with identifier {identifier} is already used")
                        self._used_voices[lang].append(voice.identifier)
                    return voice

        raise ValueError(f"Voice with identifier {identifier} not found in the database")
        return None

    def get_voice(
            self,
            gender: str,
            age: int,
            lang: str = "english",
            keep_duplicate: bool = True) -> Voice:
        """
        Random sampling of voice from the database.
        """

        if lang is not None:
            lang = lang.lower()

        if lang not in self._data:
            raise ValueError(f"Language {lang} not found in the database")

        gender = gender.lower()

        # If the voice is not in the database, find the closest age for this gender
        if (gender, age) not in self._data[lang]:

            # Get the list of ages for this gender
            _ages = [_age for (_gender, _age) in self._data[lang].keys() if _gender == gender]
            # add shuffle the list
            random.shuffle(_ages)
            random.shuffle(_ages)
            random.shuffle(_ages)

            # Get the closest age for this gender
            age = min(_ages, key=lambda x: abs(x - age))

        # Get the voices from the database for this gender, age and language
        _subset: List[Voice] = self._data[lang][(gender, age)]

        # Filter the voices to keep only the ones that are not in the used voices
        if not keep_duplicate:

            if lang not in self._used_voices:
                self._used_voices[lang] = []

            _subset: List[Voice] = [
                voice for voice in _subset
                if voice.identifier not in self._used_voices[lang]
            ]

        # If no voice left, raise an error
        if len(_subset) == 0:
            raise ValueError("No voice found for this gender, age and language")

        # Randomly sample a voice from the database for this gender, age and language
        final_voice: Voice = random.choice(_subset)

        # Add the voice to the list of used voices
        if not keep_duplicate:
            self._used_voices[lang].append(final_voice.identifier)

        return final_voice


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

        counter = 0

        self._data = {}

        for d in dataset:

            if "language" in d and d["language"] is not None:
                lang = d["language"].lower()
            else:
                lang = "english"
                logging.warning("[Voice Database] Language not found, english has been considered by default")

            if "language_code" in d and d["language_code"] is not None:
                lang_code = d["language_code"].lower()
            else:
                lang_code = "e"
                logging.warning("[Voice Database] Language code not found, e has been considered by default")

            if "gender" in d and d["gender"] is not None:
                gender = self._gender_to_gender(d["gender"])
            else:
                gender = random.choice(["male", "female"]).lower()
                logging.warning(
                    f"[Voice Database] Gender not found, a random gender ({gender}) has been considered by default"
                )

            if "age" in d and d["age"] is not None:
                age = int(d["age"])
            else:
                age = random.randint(18, 65)
                logging.warning(f"[Voice Database] Age not found, a random age ({age}) has been considered by default")

            if "identifier" in d and d["identifier"] is not None:
                identifier = str(d["identifier"])
            else:
                identifier = f"voice_{counter}"
                logging.warning(
                    "[Voice Database] Identifier not found, "
                    f"a random identifier ({identifier}) has been considered by default"
                )

            if "audio" in d and d["audio"] is not None:
                _voice = d["audio"]["path"]
            elif "voice" in d and d["voice"] is not None:
                _voice = d["voice"]
            else:
                raise ValueError("No voice found in the dataset")

            if lang not in self._data:
                self._data[lang] = {}

            key = (gender, age)

            if key not in self._data[lang]:
                self._data[lang][key] = []

            self._data[lang][key].append(Voice(
                gender=gender,
                age=age,
                identifier=str(identifier),
                voice=_voice,
                language=lang,
                language_code=lang_code
            ))
            counter += 1

        logging.info(f"[Voice Database] Has been populated with {counter} voices")


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
        The metadata file can be a csv, tsv or json file.
        The metadata file must contain the following columns: identifier, gender, age, voice, language, language_code.
            - "voice" or "file_name" column:
                - file_name: can be a relative path or an absolute path
                - voice: can be the name of the voice like "am_echo"
            - language column can be a string like "english" or "french".
            - language_code column can be a string like "e" or "f".
            - identifier column can be a string like "am_echo" or "am_echo_2".
            - gender column can be a string like "male" or "female".
            - age column can be an integer like 20 or 30.
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

        # check if the voice or file_name column exists
        if "voice" not in df.columns and "file_name" not in df.columns:
            raise ValueError(f"Voice or file_name column does not exist in the metadata file: {self.metadata_file}")

        # check if the gender column exists
        if "gender" not in df.columns:
            raise ValueError(f"Gender column does not exist in the metadata file: {self.metadata_file}")

        # check if the age column exists
        if "age" not in df.columns:
            raise ValueError(f"Age column does not exist in the metadata file: {self.metadata_file}")

        # check if the speaker id column exists
        if "identifier" not in df.columns:
            raise ValueError(f"Speaker id column does not exist in the metadata file: {self.metadata_file}")

        counter = 0

        self._data = {}
        for index, row in df.iterrows():

            lang = row["language"] if "language" in df.columns else "english"
            lang_code = row["language_code"] if "language_code" in df.columns else "e"
            gender = self._gender_to_gender(row["gender"])

            # check if the voice is a audio file
            if "file_name" in row and row["file_name"] is not None:

                # Check if the voice is a relative path
                if not os.path.isabs(row["file_name"]):
                    voice = os.path.abspath(os.path.join(self.directory_audios, row["file_name"]))
                else:
                    # Otherwise it's an absolute path
                    voice = row["file_name"]

            elif "voice" in row and row["voice"] is not None:
                # Otherwise it can be the identifier of the voice like "am_echo"
                voice = row["voice"]

            else:
                raise ValueError(f"Voice or file_name column does not exist in the metadata file: {self.metadata_file}")

            age = int(row["age"])

            if lang not in self._data:
                self._data[lang] = {}

            key = (gender, age)

            if key not in self._data[lang]:
                self._data[lang][key] = []

            self._data[lang][key].append(Voice(
                gender=gender,
                age=age,
                identifier=str(row["identifier"]),
                voice=voice,
                language=lang,
                language_code=lang_code
            ))
            counter += 1

        logging.info(f"[Voice Database] Has been populated with {counter} voices")
