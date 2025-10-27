"""
This module provides an extended dialogue class for audio generation and processing.

The AudioDialog class extends the base Dialog class with audio-specific functionality,
including audio turn management, audio source handling, and room acoustics simulation
support. It maintains compatibility with the base Dialog interface while adding
comprehensive audio processing capabilities.

Key Features:

  - Audio turn management with individual audio data per turn
  - Audio source collection and organization for room acoustics simulation
  - Combined audio generation and management
  - File path tracking for different audio processing stages
  - Speaker role mapping and identification
  - Serialization support for audio dialogue data

Example:

    .. code-block:: python

        from sdialog.audio import AudioDialog
        from sdialog import Dialog

        # Convert regular dialog to audio dialog
        audio_dialog = AudioDialog.from_dialog(dialog)

        # Access audio-specific properties
        print(f"Total duration: {audio_dialog.total_duration}")
        print(f"Audio sources: {len(audio_dialog.get_audio_sources())}")

        # Save audio dialog with metadata
        audio_dialog.to_file("audio_dialog.json")
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import json
import numpy as np
import soundfile as sf
from sdialog import Dialog
from typing import List, Union
from sdialog.audio.utils import Role
from sdialog.audio.room import AudioSource
from sdialog.audio.turn import AudioTurn


class AudioDialog(Dialog):
    """
    Extended dialogue class with comprehensive audio processing capabilities.
    """

    turns: List[AudioTurn] = []
    audio_dir_path: str = ""
    total_duration: float = -1.0
    timeline_name: str = ""

    _combined_audio: np.ndarray = None
    audio_sources: List[AudioSource] = []

    audio_step_1_filepath: str = ""
    audio_step_2_filepath: str = ""
    audio_step_3_filepaths: dict[str, dict] = {}

    speakers_names: dict[str, str] = {}
    speakers_roles: dict[str, str] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_audio_sources(self, audio_sources: List[AudioSource]):
        """
        Sets the audio sources for room acoustics simulation.

        Audio sources represent the spatial positions and characteristics of
        each speaker in the dialogue for room acoustics simulation. This
        method replaces the current list of audio sources.

        :param audio_sources: List of AudioSource objects representing speaker positions.
        :type audio_sources: List[AudioSource]
        """
        self.audio_sources = audio_sources

    def add_audio_source(self, audio_source: AudioSource):
        """
        Adds a single audio source to the dialogue's audio sources list.

        This method appends a new AudioSource to the existing list, allowing
        for incremental building of the audio sources collection.

        :param audio_source: AudioSource object to add to the dialogue.
        :type audio_source: AudioSource
        """
        self.audio_sources.append(audio_source)

    def get_audio_sources(self) -> List[AudioSource]:
        """
        Retrieves the list of audio sources for room acoustics simulation.

        :return: List of AudioSource objects representing speaker positions and characteristics.
        :rtype: List[AudioSource]
        """
        return self.audio_sources

    def set_combined_audio(self, audio: np.ndarray):
        """
        Sets the combined audio data for the entire dialogue.

        The combined audio represents the concatenated audio from all turns
        in the dialogue, typically used for room acoustics simulation or
        final audio export.

        :param audio: Numpy array containing the combined audio data.
        :type audio: np.ndarray
        """
        self._combined_audio = audio

    def get_combined_audio(self) -> np.ndarray:
        """
        Retrieves the combined audio data for the entire dialogue.

        If the combined audio is not already loaded in memory, it will be
        loaded from the audio_step_1_filepath. This method provides lazy
        loading of audio data to optimize memory usage.

        :return: Numpy array containing the combined audio data.
        :rtype: np.ndarray
        :raises FileNotFoundError: If the audio file path is invalid or file doesn't exist.
        """
        if self._combined_audio is None:
            # load the combined audio from the audio_step_1_filepath
            self._combined_audio = sf.read(self.audio_step_1_filepath)[0]
        return self._combined_audio

    @staticmethod
    def from_dialog(dialog: Dialog):
        """
        Creates an AudioDialog object from a base Dialog object.

        This static method converts a regular Dialog object into an AudioDialog
        by copying all attributes and converting Turn objects to AudioTurn objects.
        It also establishes speaker role mappings based on the dialogue structure.

        The conversion process:
        1. Creates a new AudioDialog instance
        2. Copies all attributes from the base Dialog
        3. Converts each Turn to an AudioTurn using from_turn()
        4. Identifies the first two speakers and assigns them roles
        5. Creates bidirectional mappings between speaker names and roles

        :param dialog: The base Dialog object to convert.
        :type dialog: Dialog
        :return: A new AudioDialog object with audio-specific functionality.
        :rtype: AudioDialog
        :raises IndexError: If the dialog has fewer than 2 turns (speakers).
        """

        audio_dialog = AudioDialog()

        # Copy all attributes from the base dialog
        for attr in dialog.__dict__:
            setattr(audio_dialog, attr, getattr(dialog, attr))

        # Convert regular turns to audio turns
        audio_dialog.turns = [AudioTurn.from_turn(turn) for turn in dialog.turns]

        # Identify speakers from the first two turns
        speaker_1 = audio_dialog.turns[0].speaker
        speaker_2 = audio_dialog.turns[1].speaker

        # Create role mappings for speaker identification
        audio_dialog.speakers_names[Role.SPEAKER_1] = speaker_1
        audio_dialog.speakers_names[Role.SPEAKER_2] = speaker_2

        # Create reverse mappings for role lookup
        audio_dialog.speakers_roles[speaker_1] = Role.SPEAKER_1
        audio_dialog.speakers_roles[speaker_2] = Role.SPEAKER_2

        return audio_dialog

    @staticmethod
    def from_dict(data: dict):
        """
        Creates an AudioDialog object from a dictionary representation.

        This method deserializes an AudioDialog from a dictionary containing
        all the dialogue data including audio-specific attributes. It uses
        Pydantic's model validation to ensure data integrity.

        :param data: Dictionary containing serialized AudioDialog data.
        :type data: dict
        :return: A new AudioDialog object created from the dictionary data.
        :rtype: AudioDialog
        :raises ValidationError: If the dictionary data is invalid or incomplete.
        """
        return AudioDialog.model_validate(data)

    @staticmethod
    def from_json(json_str: str):
        """
        Creates an AudioDialog object from a JSON string representation.

        This method deserializes an AudioDialog from a JSON string by first
        parsing the JSON and then using from_dict() to create the object.

        :param json_str: JSON string containing serialized AudioDialog data.
        :type json_str: str
        :return: A new AudioDialog object created from the JSON data.
        :rtype: AudioDialog
        :raises json.JSONDecodeError: If the JSON string is malformed.
        :raises ValidationError: If the parsed data is invalid or incomplete.
        """
        return AudioDialog.from_dict(json.loads(json_str))

    def to_file(self, path: str = None, makedir: bool = True, overwrite: bool = True):
        """
        Saves the AudioDialog object to a JSON file with comprehensive metadata.

        This method serializes the AudioDialog object to JSON format, including
        all audio-specific attributes, file paths, and processing metadata.
        It provides flexible path handling and directory creation options.

        Path resolution:
        1. If path is provided, use it directly
        2. If no path but _path exists (from loading), use _path
        3. Otherwise, raise ValueError

        :param path: Output file path for the JSON file. If None, uses the path
                    from which the dialog was loaded (if available).
        :type path: Optional[str]
        :param makedir: If True, creates parent directories as needed.
        :type makedir: bool
        :param overwrite: If True, overwrites existing files. If False, raises
                         FileExistsError if file already exists.
        :type overwrite: bool
        :raises ValueError: If no path is provided and no loading path is available.
        :raises FileExistsError: If file exists and overwrite is False.
        :raises OSError: If directory creation fails or file writing fails.
        """
        if not path:
            if hasattr(self, '_path') and self._path:
                path = self._path
            else:
                raise ValueError("No path provided to save the audio dialog and no loading path available. "
                                 "Please specify a valid file path.")

        if makedir and os.path.split(path)[0]:
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"File '{path}' already exists. Use 'overwrite=True' to overwrite it.")

        with open(path, "w", newline='') as writer:
            writer.write(self.model_dump_json(indent=2))

    @staticmethod
    def from_file(path: str) -> Union["AudioDialog", List["AudioDialog"]]:
        """
        Loads an audio dialog from a JSON file or a directory of JSON files.

        :param path: Path to the dialogue file or directory. In case of a directory, all dialogues in the directory
                     will be loaded and returned as a list of Dialog objects.
        :type path: str
        :return: The loaded dialogue object or a list of dialogue objects.
        :rtype: Union[Dialog, List[Dialog]]
        """
        if os.path.isdir(path):
            dialogs = [AudioDialog.from_file(os.path.join(path, filename))
                       for filename in sorted(os.listdir(path))
                       if filename.endswith(".json")]
            return dialogs

        with open(path) as reader:
            dialog = AudioDialog.from_dict(json.load(reader))
            dialog._path = path  # Store the path for later use
            return dialog

    def to_string(self):
        return self.model_dump_json(indent=4)

    def display(self):
        """
        Displays the audio dialog.
        """
        from IPython.display import Audio, display

        if len(self.audio_step_1_filepath) > 0:
            print("-"*25)
            print("Step 1:")
            print("-"*25)
            display(Audio(
                self.audio_step_1_filepath,
                autoplay=False
            ))

        if len(self.audio_step_2_filepath) > 0:
            print("-"*25)
            print("Step 2:")
            print("-"*25)
            display(Audio(
                self.audio_step_2_filepath,
                autoplay=False
            ))

        if len(self.audio_step_3_filepaths) > 0:

            print("-"*25)
            print("- Room Configurations")
            print("-"*25)

            # For each room configuration, display the original audio and the processed audio
            for config_name in self.audio_step_3_filepaths:

                print(f"> Room Configuration: {config_name}")
                print("Original audio:")
                display(Audio(
                    self.audio_step_3_filepaths[config_name]["audio_path"],
                    autoplay=False
                ))

                # If the room configuration has processed audio, display it
                if (
                    config_name in self.audio_step_3_filepaths
                    and "audio_paths_post_processing" in self.audio_step_3_filepaths[config_name]
                    and len(self.audio_step_3_filepaths[config_name]["audio_paths_post_processing"]) > 0
                ):
                    print("#"*10)
                    print("Processed audio:")
                    print("#"*10)

                    # For each recording device, display the processed audio
                    for _rd in self.audio_step_3_filepaths[config_name]["audio_paths_post_processing"]:
                        display(Audio(
                            self.audio_step_3_filepaths[config_name]["audio_paths_post_processing"][_rd],
                            autoplay=False
                        ))
