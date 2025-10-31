# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import abc
import json
from enum import Enum
from typing import Union


class RecordingDevice(str, Enum):
    """
    An enumeration of supported recording devices.

    This class provides a standardized way to refer to different recording
    devices, which are used to select the appropriate impulse response for
    audio processing. It inherits from `str` and `Enum` to allow for both
    enum-style access and string-based identifiers.

    Example:
        .. code-block:: python

            from sdialog.audio.impulse_response_database import RecordingDevice
            # Accessing a device by its enum member name
            device = RecordingDevice.LCT_440
            print(device)
            # Using the string value directly
            device_str = "OD-FBVET30-CND-AU-1-P20-50"
            if device_str == RecordingDevice.LCT_440:
                print("Device identified correctly.")

    Note:
        The string values correspond to specific impulse response identifiers
        in the impulse response database.
    """
    LCT_440 = "OD-FBVET30-CND-AU-1-P20-50"
    SHURE_SM57 = "OD-FBVET30-DYN-57-P05-20"
    RE_20 = "OD-FBVET30-DYN-US-8-P10-70"
    RBN_160 = "OD-FBVET30-RBN-160-P10-30"
    SENNHEISER_E906 = "OD-FB-VET30-DYN-906-P12-30"
    AUDIX_I5 = "OD-FB-VET30-DYN-I5-P12-20"
    NEUMANN_TLM_103 = "OD-FB-VET30-LD-103-P09-40-LC"
    SONY_C800G_TUBE = "OD-FB-VET30-LD-800-P09-40"
    ROYER_R_10 = "OD-FB-VET30-RBN-US-1-P10-30-LC75"
    SENNHEISER_MD409_U3 = "OD-FBVET30-DN-409U-09-40"
    SENNHEISER_MD421_II = "OD-FBVET30-DN-421B-10-40-LC"
    NEUMANN_U67_TUBE = "OD-FBVET30-LD-67NOS-P09-40-LC"
    RBN_CN_2 = "OD-FBVET30-RBN-CN-2-P09-100"

    def __str__(self):
        return self.value


class ImpulseResponseDatabase(abc.ABC):
    """
    Abstract base class for an impulse response database.

    This class defines the interface for an impulse response database, which
    is used to store and retrieve impulse responses for audio processing.
    Subclasses must implement the `_populate` method to load the impulse
    response data from a specific source.

    :ivar _data: A dictionary mapping impulse response identifiers to their
                 corresponding audio file paths.
    :vartype _data: dict[str, str]
    """

    def __init__(self):
        self._data: dict[str, str] = {}
        self._populate()

    @abc.abstractmethod
    def _populate(self) -> None:
        """
        Loads all impulse responses into memory.
        """
        raise NotImplementedError

    def get_data(self) -> dict[str, str]:
        """
        Returns the data of the impulse response database.
        :return: The data of the impulse response database.
        :rtype: dict[str, str]
        """
        return self._data

    def get_ir(self, identifier: Union[str, RecordingDevice]) -> str:
        """
        :param identifier: The identifier of the impulse response.
        :type identifier: str
        :return: The path to the impulse response audio file.
        :rtype: str
        :raises ValueError: If the impulse response with the given identifier is not found.
        """

        if isinstance(identifier, RecordingDevice):
            identifier = str(identifier.value)

        if identifier not in self._data:
            raise ValueError(f"Impulse response with identifier '{identifier}' not found.")

        return self._data[identifier]


class LocalImpulseResponseDatabase(ImpulseResponseDatabase):
    """
    An impulse response database that loads data from a local directory.

    This class provides an implementation of `ImpulseResponseDatabase` that
    loads impulse response data from a local filesystem. It expects a
    directory containing the audio files and a metadata file (in JSON, CSV,
    or TSV format) that maps impulse response identifiers to their
    corresponding file names.

    The metadata file must contain 'identifier' and 'file_name' columns.

    Example:
        .. code-block:: python

            from sdialog.audio.impulse_response_database import LocalImpulseResponseDatabase
            # Create a dummy metadata file and audio file
            with open("metadata.csv", "w") as f:
                f.write("identifier,file_name\\n")
                f.write("my_ir,my_ir.wav\\n")
            import soundfile as sf
            import numpy as np
            sf.write("my_ir.wav", np.random.randn(1000), 16000)
            # Initialize the database and retrieve an impulse response
            db = LocalImpulseResponseDatabase(metadata_file="metadata.csv", directory=".")
            ir_path = db.get_ir("my_ir")
            print(ir_path)

    :ivar metadata_file: The path to the metadata file.
    :vartype metadata_file: str
    :ivar directory: The path to the directory containing the audio files.
    :vartype directory: str
    """
    def __init__(self, metadata_file: str, directory: str):
        self.metadata_file = metadata_file
        self.directory = directory
        ImpulseResponseDatabase.__init__(self)

    def _populate(self) -> None:
        """
        Loads metadata and all associated audio files into memory.
        The metadata file can be a csv, tsv or json file.
        The metadata file must contain the following columns: identifier, audio.
        """
        import pandas as pd

        if not os.path.exists(self.metadata_file):
            raise ValueError(f"Metadata file not found at path: {self.metadata_file}")

        if not os.path.isdir(self.directory):
            raise ValueError(f"Audio directory is not a directory: {self.directory}")

        if self.metadata_file.endswith(".csv"):
            metadata = pd.read_csv(self.metadata_file)
        elif self.metadata_file.endswith(".tsv"):
            metadata = pd.read_csv(self.metadata_file, sep='\t')
        elif self.metadata_file.endswith(".json"):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            raise ValueError(f"Metadata file is not a csv / tsv / json file: {self.metadata_file}")

        # Convert the metadata to a list of dictionaries
        if isinstance(metadata, pd.DataFrame):
            metadata = metadata.to_dict(orient="records")

        # Load the metadata into the database
        for row in metadata:

            audio_path = os.path.join(self.directory, str(row["file_name"]))

            # Check if the audio file exists
            if os.path.exists(audio_path):
                self._data[str(row["identifier"])] = audio_path
            else:
                raise ValueError(f"Audio file not found at path: {audio_path}")


class HuggingFaceImpulseResponseDatabase(ImpulseResponseDatabase):
    """
    An impulse response database that loads data from a Hugging Face Hub dataset.

    This class provides an implementation of `ImpulseResponseDatabase` that
    loads impulse response data from a dataset on the Hugging Face Hub. It
    can load a dataset from a repository ID or a local path.

    The dataset is expected to have 'identifier' and 'audio' columns, where
    'audio' is a dictionary containing a 'path' key.

    Example:
        .. code-block:: python

            from sdialog.audio.impulse_response_database import HuggingFaceImpulseResponseDatabase
            # Initialize the database with a Hugging Face Hub dataset
            # Note: This requires the 'datasets' library to be installed.
            # db = HuggingFaceImpulseResponseDatabase(repo_id="your_username/your_dataset_repo")
            # ir_path = db.get_ir("some_ir_identifier")

    :ivar repo_id: The repository ID of the Hugging Face Hub dataset, or a
                   local path to the dataset.
    :vartype repo_id: str
    :ivar subset: The subset of the dataset to use (e.g., "train", "test").
    :vartype subset: str
    """

    def __init__(
        self,
        repo_id: str,
        subset: str = "train"
    ):
        """
        Initializes the Hugging Face impulse response database.
        :param repo_id: The repository identifier of the Hugging Face Hub dataset.
        :type repo_id: str
        :param subset: The subset of the Hugging Face Hub dataset to use.
        :type subset: str
        """
        self.repo_id = repo_id
        self.subset = subset
        ImpulseResponseDatabase.__init__(self)

    def _populate(self) -> None:
        """
        Loads the data from the Hugging Face Hub dataset.
        The dataset must contain the following columns: identifier, audio.
        :raises ValueError: If the dataset does not contain the required columns.
        :raises ValueError: If the audio file is not found.
        :raises ValueError: If the identifier is not found in the dataset.
        """
        from datasets import load_dataset, load_from_disk

        if os.path.exists(self.repo_id):
            _dataset = load_from_disk(self.repo_id)[self.subset]
        else:
            _dataset = load_dataset(self.repo_id)[self.subset]

        for d in _dataset:

            if "identifier" not in d or d["identifier"] is None:
                raise ValueError("Identifier not found in the dataset")

            if "audio" not in d or d["audio"] is None:
                raise ValueError("Audio not found in the dataset")

            if "path" not in d["audio"] or d["audio"]["path"] is None:
                raise ValueError("Path not found in the audio")

            self._data[str(d["identifier"])] = d["audio"]["path"]
