"""
This module provides comprehensive voice database management for the sdialog library.

The module includes a base voice database class and multiple implementations for
different data sources, enabling flexible voice selection and management for
text-to-speech generation with support for multiple languages and speaker
characteristics.

Key Components:

  - BaseVoiceDatabase: Abstract base class for voice database implementations
  - Voice: Data model for individual voice entries with metadata
  - HuggingfaceVoiceDatabase: Implementation using Hugging Face datasets
  - LocalVoiceDatabase: Implementation using local audio files and metadata
  - VoiceDatabase: Implementation using in-memory voice data

Voice Database Features:

  - Multi-language voice support with automatic language detection
  - Speaker characteristics (gender, age, language) for voice selection
  - Voice usage tracking to prevent duplicates (optional)
  - Comprehensive statistics and reporting
  - Support for various data sources (Hugging Face, local files, in-memory)
  - Flexible voice selection based on persona characteristics

Example:

    .. code-block:: python

        from sdialog.audio import HuggingfaceVoiceDatabase, LocalVoiceDatabase

        # Initialize with Hugging Face dataset
        voice_db = HuggingfaceVoiceDatabase("sdialog/voices-libritts")

        # Get voice based on speaker characteristics
        voice = voice_db.get_voice(gender="female", age=25, lang="english", seed=42)

        # Initialize with local files
        local_db = LocalVoiceDatabase(
            directory_audios="voices/",
            metadata_file="voices/metadata.csv"
        )

        # Get statistics
        stats = voice_db.get_statistics(pretty=True)
        print(stats)
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import random
from typing import List, Union
from pydantic import BaseModel
from collections import defaultdict, Counter

from sdialog.audio.utils import logger
from sdialog.util import dict_to_table


def is_a_audio_file(file: str) -> bool:
    """
    Checks if a file is an audio file based on its extension.

    This utility function determines whether a given file path corresponds
    to an audio file by checking for common audio file extensions.
    The check is case-insensitive and supports various audio formats.

    Supported audio formats:
        - WAV (.wav)
        - MP3 (.mp3)
        - M4A (.m4a)
        - OGG (.ogg)
        - FLAC (.flac)
        - AIFF (.aiff, .aif)
        - AAC (.aac)

    :param file: The file path to check.
    :type file: str
    :return: True if the file has an audio extension, False otherwise.
    :rtype: bool
    """
    file = file.lower()
    if (
        ".wav" in file
        or ".mp3" in file
        or ".m4a" in file
        or ".ogg" in file
        or ".flac" in file
        or ".aiff" in file
        or ".aif" in file
        or ".aac" in file
    ):
        return True
    return False


class Voice(BaseModel):
    """
    Data model representing a voice entry in the voice database.

    The Voice class encapsulates all metadata associated with a specific voice,
    including speaker characteristics, language information, and voice identifiers.
    This model is used throughout the voice database system for voice selection
    and management.

    Key Attributes:

      - gender: Speaker gender for voice selection
      - age: Speaker age for voice selection
      - identifier: Unique identifier for the voice
      - voice: Voice data (file path or voice string identifier)
      - language: Human-readable language name
      - language_code: Language code for TTS engines

    :ivar gender: Speaker gender (e.g., "male", "female").
    :vartype gender: str
    :ivar age: Speaker age in years.
    :vartype age: int
    :ivar identifier: Unique identifier for this voice entry.
    :vartype identifier: str
    :ivar voice: Voice data - can be a file path or voice string identifier.
    :vartype voice: str
    :ivar language: Human-readable language name (default: "english").
    :vartype language: str
    :ivar language_code: Language code for TTS engines (default: "a").
    :vartype language_code: str
    """
    gender: str
    age: int
    identifier: str
    voice: str  # Can be a path or the voice string
    language: str = "english"
    language_code: str = "a"


class BaseVoiceDatabase:
    """
    Abstract base class for voice database implementations.

    This class provides the foundation for voice database implementations,
    defining the common interface and data structures used across different
    voice database types. It manages voice data organization, usage tracking,
    and provides utility methods for voice selection and statistics.

    Key Features:

      - Hierarchical voice organization by language, gender, and age
      - Voice usage tracking to prevent duplicates (optional)
      - Comprehensive statistics and reporting capabilities
      - Abstract interface for different data source implementations
      - Flexible voice selection based on speaker characteristics

    Data Structure:
        The voice database uses a nested dictionary structure:
        _data[language][(gender, age)] = [Voice1, Voice2, ...]

    :ivar _data: Nested dictionary organizing voices by language, gender, and age.
    :vartype _data: dict[str, dict[tuple[str, int], List[Voice]]]
    :ivar _used_voices: Dictionary tracking used voice identifiers by language.
    :vartype _used_voices: dict[str, List[str]]
    """

    def __init__(self):
        """
        Initializes the base voice database.

        This constructor sets up the data structures for voice organization
        and usage tracking. Subclasses should call this method and then
        implement the populate() method to load voice data from their
        specific data source.
        """

        # Dictionary to keep track of the voices: language -> (gender, age) -> list of voices
        self._data: dict[str, dict[tuple[str, int], List[Voice]]] = {}

        # Dictionary to keep track of the used voices: language -> list of identifiers
        self._used_voices: dict[str, List[str]] = {}

        # Populate the database with voice data
        self.populate()

    def get_data(self) -> dict:
        """
        Retrieves the complete voice database data structure.

        :return: The nested dictionary containing all voice data organized by language, gender, and age.
        :rtype: dict[str, dict[tuple[str, int], List[Voice]]]
        """
        return self._data

    def populate(self) -> dict:
        """
        Populates the voice database with voice data.

        This abstract method must be implemented by subclasses to load voice
        data from their specific data source (e.g., Hugging Face datasets,
        local files, in-memory data).

        :return: The populated voice data dictionary.
        :rtype: dict
        :raises NotImplementedError: If not implemented by subclass.
        """
        self._data = {}
        raise NotImplementedError("Subclasses must implement the populate method")

    def reset_used_voices(self):
        """
        Resets the tracking of used voices across all languages.

        This method clears the usage tracking, allowing all voices to be
        available for selection again. Useful for starting a new dialogue
        generation session or resetting voice allocation.
        """
        self._used_voices = {}

    def get_statistics(self, pretty: bool = False, pretty_format: str = "markdown") -> Union[dict, str]:
        """
        Generates comprehensive statistics about the voice database.

        This method analyzes the voice database and provides detailed statistics
        about the available voices, including language distribution, gender/age
        breakdowns, and usage patterns. The statistics can be returned as either
        a structured dictionary or a formatted string for display.

        Statistics include:
            - num_languages: Total number of languages in the database
            - overall: Global statistics (total voices, gender distribution, age distribution)
            - languages: Per-language detailed statistics including:
                - total: Total voices for the language
                - by_gender: Voice count by gender (male/female)
                - ages: Voice count by age bins
                - by_gender_age: Cross-tabulation of gender and age
                - unique_speakers: Number of unique voice identifiers
                - language_codes: Observed language codes for TTS engines
                - age_stats: Age statistics (min, max, mean)

        :param pretty: If True, returns a formatted string; if False, returns a dictionary.
        :type pretty: bool
        :param pretty_format: Format for pretty output - "markdown" for Markdown tables,
                             other values for fancy grid format.
        :type pretty_format: str
        :return: Either a dictionary with statistics or a formatted string representation.
        :rtype: Union[dict, str]
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
        Adds a voice entry to the database.

        This method creates a new Voice object and adds it to the appropriate
        location in the hierarchical database structure based on language,
        gender, and age. The voice is organized for efficient retrieval
        during voice selection.

        :param gender: Speaker gender (e.g., "male", "female").
        :type gender: str
        :param age: Speaker age in years.
        :type age: int
        :param identifier: Unique identifier for this voice entry.
        :type identifier: str
        :param voice: Voice data - can be a file path or voice string identifier.
        :type voice: str
        :param lang: Language name (e.g., "english", "french").
        :type lang: str
        :param language_code: Language code for TTS engines (e.g., "a", "f").
        :type language_code: str
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
        Retrieves a voice by its unique identifier.

        This method searches for a voice with the specified identifier within
        the given language. It can optionally enforce uniqueness by preventing
        the reuse of already used voices.

        :param identifier: The unique identifier of the voice to retrieve.
        :type identifier: str
        :param lang: The language to search within.
        :type lang: str
        :param keep_duplicate: If True, allows returning voices that have already been used.
                              If False, raises an error if the voice has already been used.
        :type keep_duplicate: bool
        :return: The Voice object with the specified identifier.
        :rtype: Voice
        :raises ValueError: If the language is not found in the database.
        :raises ValueError: If the voice identifier is not found.
        :raises ValueError: If keep_duplicate is False and the voice has already been used.
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

    def get_voice(
            self,
            gender: str,
            age: int,
            lang: str = "english",
            keep_duplicate: bool = True,
            seed: int = None) -> Voice:
        """
        Retrieves a voice based on speaker characteristics with intelligent matching.

        This method selects a voice from the database based on the specified
        speaker characteristics (gender, age, language). It uses intelligent
        matching to find the closest available voice when an exact match is
        not available, and can optionally enforce uniqueness to prevent
        voice reuse.

        Voice selection process:
        1. Normalize language and gender parameters
        2. Check for exact match (gender, age, language)
        3. If no exact match, find closest age for the specified gender
        4. Filter out used voices if keep_duplicate is False
        5. Randomly select from available voices
        6. Track usage if keep_duplicate is False

        :param gender: Speaker gender (e.g., "male", "female").
        :type gender: str
        :param age: Speaker age in years.
        :type age: int
        :param lang: Language name (default: "english").
        :type lang: str
        :param keep_duplicate: If True, allows voice reuse. If False, ensures each voice is used only once.
        :type keep_duplicate: bool
        :param seed: Seed for random number generator.
        :type seed: int
        :return: A Voice object matching the specified characteristics.
        :rtype: Voice
        :raises ValueError: If the language is not found in the database.
        :raises ValueError: If no voice is found for the specified characteristics.
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

            # Get the voices for the closest age for this gender
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

        # Make a deterministic copy and sort by stable key to remove any source-order nondeterminism
        _subset.sort(key=lambda v: str(v.identifier))

        # Use a local RNG to avoid mutating global state and ensure determinism when seed is provided
        rng = random.Random(seed) if seed is not None else random

        # Shuffle and sample deterministically with the local RNG
        rng.shuffle(_subset)
        final_voice: Voice = rng.choice(_subset)

        # Add the voice to the list of used voices
        if not keep_duplicate:
            self._used_voices[lang].append(final_voice.identifier)

        return final_voice


class HuggingfaceVoiceDatabase(BaseVoiceDatabase):
    """
    Voice database implementation using Hugging Face datasets.

    This implementation loads voice data from Hugging Face datasets, providing
    access to large-scale voice collections with standardized metadata. It
    supports both remote datasets and local dataset caches, making it suitable
    for research and production use cases.

    Key Features:

      - Integration with Hugging Face datasets library
      - Support for both remote and local dataset access
      - Automatic metadata extraction and validation
      - Fallback handling for missing metadata fields
      - Efficient dataset loading and caching

    Expected Dataset Format:
        The dataset should contain the following fields:
        - audio: Audio data with path information
        - voice: Voice identifier (alternative to audio)
        - gender: Speaker gender ("male", "female", "m", "f")
        - age: Speaker age (integer)
        - identifier: Unique voice identifier
        - language: Language name (optional, defaults to "english")
        - language_code: Language code for TTS engines (optional, defaults to "e")

    :ivar dataset_name: Name or path of the Hugging Face dataset.
    :vartype dataset_name: str
    :ivar subset: Dataset subset to use (e.g., "train", "test", "validation").
    :vartype subset: str
    """

    def __init__(
            self,
            dataset_name: str = "sdialog/voices-libritts",
            subset: str = "train"):
        """
        Initializes the Hugging Face voice database.

        This constructor sets up the dataset connection and loads voice data
        from the specified Hugging Face dataset. It supports both remote
        datasets and local dataset caches.

        :param dataset_name: Name or path of the Hugging Face dataset
                            (default: "sdialog/voices-libritts").
        :type dataset_name: str
        :param subset: Dataset subset to use (default: "train").
        :type subset: str
        :raises ImportError: If the datasets library is not installed.
        :raises ValueError: If the dataset or subset is not found.
        """

        self.dataset_name = dataset_name
        self.subset = subset
        BaseVoiceDatabase.__init__(self)

    def populate(self) -> dict:
        """
        Populates the voice database from the Hugging Face dataset.

        This method loads voice data from the specified Hugging Face dataset,
        extracts metadata for each voice entry, and organizes the data in the
        hierarchical database structure. It handles missing metadata by
        providing sensible defaults and logging warnings.

        Data processing steps:
        1. Load dataset from Hugging Face (remote or local cache)
        2. Iterate through dataset entries
        3. Extract and validate metadata fields
        4. Handle missing fields with defaults and warnings
        5. Organize voices by language, gender, and age
        6. Create Voice objects and add to database

        :return: The populated voice data dictionary.
        :rtype: dict
        :raises ImportError: If the datasets library is not installed.
        :raises ValueError: If the dataset or subset is not found.
        :raises ValueError: If required voice data is missing from the dataset.
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
                logger.warning("[Voice Database] Language not found, english has been considered by default")

            if "language_code" in d and d["language_code"] is not None:
                lang_code = d["language_code"].lower()
            else:
                lang_code = "e"
                logger.warning("[Voice Database] Language code not found, e has been considered by default")

            if "gender" in d and d["gender"] is not None:
                gender = self._gender_to_gender(d["gender"])
            else:
                gender = random.choice(["male", "female"]).lower()
                logger.warning(
                    f"[Voice Database] Gender not found, a random gender ({gender}) has been considered by default"
                )

            if "age" in d and d["age"] is not None:
                age = int(d["age"])
            else:
                age = random.randint(18, 65)
                logger.warning(
                    f"[Voice Database] Age not found, a random age ({age}) has been considered by default"
                )

            if "identifier" in d and d["identifier"] is not None:
                identifier = str(d["identifier"])
            else:
                identifier = f"voice_{counter}"
                logger.warning(
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

        logger.info(f"[Voice Database] Has been populated with {counter} voices")


class LocalVoiceDatabase(BaseVoiceDatabase):
    """
    Voice database implementation using local audio files and metadata.

    This implementation loads voice data from local audio files and metadata
    files (CSV, TSV, or JSON), providing a flexible solution for custom voice
    collections. It supports various metadata formats and handles both relative
    and absolute file paths.

    Key Features:

      - Support for multiple metadata formats (CSV, TSV, JSON)
      - Flexible file path handling (relative and absolute paths)
      - Comprehensive metadata validation
      - Support for both file paths and voice identifiers
      - Local file system integration

    Required Metadata Format:
        The metadata file must contain the following columns:
        - identifier: Unique voice identifier
        - gender: Speaker gender ("male", "female", "m", "f")
        - age: Speaker age (integer)
        - voice or file_name: Voice identifier or audio file path
        - language: Language name (optional, defaults to "english")
        - language_code: Language code for TTS engines (optional, defaults to "e")

    :ivar directory_audios: Directory containing audio files.
    :vartype directory_audios: str
    :ivar metadata_file: Path to the metadata file (CSV, TSV, or JSON).
    :vartype metadata_file: str
    """

    def __init__(
            self,
            directory_audios: str,
            metadata_file: str):
        """
        Initializes the local voice database.

        This constructor sets up the local voice database by validating the
        audio directory and metadata file paths, then loading voice data
        from the local files.

        :param directory_audios: Directory path containing audio files.
        :type directory_audios: str
        :param metadata_file: Path to the metadata file (CSV, TSV, or JSON).
        :type metadata_file: str
        :raises ValueError: If the audio directory does not exist or is not a directory.
        :raises ValueError: If the metadata file does not exist or has an unsupported format.
        """

        self.directory_audios = directory_audios
        self.metadata_file = metadata_file

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
            not self.metadata_file.endswith(".csv") and not self.metadata_file.endswith(".tsv")
            and not self.metadata_file.endswith(".json")
        ):
            raise ValueError(f"Metadata file is not a csv / tsv / json file: {self.metadata_file}")

        BaseVoiceDatabase.__init__(self)

    def populate(self) -> dict:
        """
        Populate the voice database.
        The metadata file can be a csv, tsv or json file.
        The metadata file must contain the following columns: identifier, gender, age, voice, language, language_code.

        - "voice" or "file_name" column: path to audio file or voice name (e.g., "am_echo")
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

        logger.info(f"[Voice Database] Has been populated with {counter} voices")


class VoiceDatabase(BaseVoiceDatabase):
    """
    Voice database implementation using in-memory voice data.

    This implementation creates a voice database from a list of voice dictionaries,
    providing a flexible solution for programmatically creating voice databases
    or loading voice data from custom sources. It's particularly useful for
    testing, small voice collections, or when voice data is already available
    in memory.

    Key Features:

      - In-memory voice data processing
      - Support for custom voice data structures
      - Comprehensive data validation
      - Flexible voice data input format
      - No external file dependencies

    Required Data Format:
        The input data should be a list of dictionaries, where each dictionary
        contains the following keys:
        - identifier: Unique voice identifier
        - gender: Speaker gender ("male", "female", "m", "f")
        - age: Speaker age (integer)
        - voice: Voice identifier or data
        - language: Language name (optional, defaults to "english")
        - language_code: Language code for TTS engines (optional, defaults to "e")

    :ivar _input_data: List of voice dictionaries to process.
    :vartype _input_data: list[dict]
    """

    def __init__(self, data: list[dict]):
        """
        Initializes the voice database with in-memory voice data.

        This constructor sets up the voice database by processing the provided
        list of voice dictionaries and organizing them in the hierarchical
        database structure.

        :param data: List of voice dictionaries containing voice metadata.
        :type data: list[dict]
        :raises ValueError: If the input data is not a list of dictionaries.
        :raises ValueError: If required voice data is missing from the input.
        """

        self._input_data = data
        BaseVoiceDatabase.__init__(self)

    def populate(self) -> dict:
        """
        Populates the voice database from in-memory voice data.

        This method processes the list of voice dictionaries provided during
        initialization, validates the data, and organizes it in the hierarchical
        database structure. It performs comprehensive validation to ensure
        data integrity and provides detailed error messages for missing or
        invalid data.

        Data processing steps:
        1. Validate input data format (list of dictionaries)
        2. Iterate through each voice dictionary
        3. Extract and validate required fields
        4. Handle missing optional fields with defaults
        5. Create Voice objects and add to database
        6. Log processing statistics

        Required fields in each voice dictionary:
            - identifier: Unique voice identifier
            - gender: Speaker gender ("male", "female", "m", "f")
            - age: Speaker age (integer)
            - voice: Voice identifier or data
            - language: Language name (optional, defaults to "english")
            - language_code: Language code for TTS engines (optional, defaults to "e")

        :return: The populated voice data dictionary.
        :rtype: dict
        :raises ValueError: If the input data is not a list of dictionaries.
        :raises ValueError: If required voice data is missing from any entry.
        """

        # check if the metadata is a list of dictionaries
        if not isinstance(self._input_data, list) or not all(isinstance(item, dict) for item in self._input_data):
            raise ValueError(f"Data is not a list of dictionaries: {self._input_data}")

        counter = 0

        self._data = {}

        for item in self._input_data:

            if "voice" not in item:
                raise ValueError(f"Voice column does not exist in the data: {item}")
            else:
                voice = item["voice"]

            if "language" not in item:
                raise ValueError(f"Language column does not exist in the data: {item}")
            else:
                lang = item["language"]

            if "language_code" not in item:
                raise ValueError(f"Language code column does not exist in the data: {item}")
            else:
                lang_code = item["language_code"]

            if "identifier" not in item:
                raise ValueError(f"Identifier column does not exist in the data: {item}")
            else:
                identifier = str(item["identifier"])

            if "gender" not in item:
                raise ValueError(f"Gender column does not exist in the data: {item}")
            else:
                gender = self._gender_to_gender(item["gender"])

            if "age" not in item:
                raise ValueError(f"Age column does not exist in the data: {item}")
            else:
                age = int(item["age"])

            if lang not in self._data:
                self._data[lang] = {}

            key = (gender, age)

            if key not in self._data[lang]:
                self._data[lang][key] = []

            self._data[lang][key].append(Voice(
                gender=gender,
                age=age,
                identifier=identifier,
                voice=voice,
                language=lang,
                language_code=lang_code
            ))
            counter += 1

        logger.info(f"[Voice Database] Has been populated with {counter} voices")
