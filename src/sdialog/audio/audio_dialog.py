"""
This module provides a dialog class for audio generation.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import json
import numpy as np
from sdialog import Dialog
from typing import List, Union
from sdialog.audio.room import AudioSource
from sdialog.audio.audio_turn import AudioTurn


class AudioDialog(Dialog):
    """
    Represents a dialogue with audio turns.
    """

    turns: List[AudioTurn] = []
    audio_dir_path: str = None
    total_duration: float = None
    timeline_name: str = None

    _combined_audio: np.ndarray = None
    audio_sources: List[AudioSource] = []

    audio_step_1_filepath: str = None
    audio_step_2_filepath: str = None
    audio_step_3_filepaths: dict[str, dict] = {}

    # Room hash or user input name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_audio_sources(self, audio_sources: List[AudioSource]):
        """
        Set the audio sources of the dialog.
        """
        self._audio_sources = audio_sources

    def add_audio_source(self, audio_source: AudioSource):
        """
        Add an audio source to the dialog.
        """
        self._audio_sources.append(audio_source)

    def get_audio_sources(self) -> List[AudioSource]:
        """
        Get the audio sources of the dialog.
        """
        return self._audio_sources

    def set_combined_audio(self, audio: np.ndarray):
        """
        Set the combined audio of the dialog.
        """
        self._combined_audio = audio

    def get_combined_audio(self) -> np.ndarray:
        """
        Get the combined audio of the dialog.
        """
        return self._combined_audio

    @staticmethod
    def from_dialog(dialog: Dialog):
        audio_dialog = AudioDialog()

        for attr in dialog.__dict__:
            setattr(audio_dialog, attr, getattr(dialog, attr))

        audio_dialog.turns = [AudioTurn.from_turn(turn) for turn in dialog.turns]
        return audio_dialog

    @staticmethod
    def from_dict(data: dict):
        """
        Creates a AudioDialog object from a dictionary.

        :param data: The dictionary containing dialogue data.
        :type data: dict
        :return: The created AudioDialog object.
        :rtype: AudioDialog
        """
        return AudioDialog.model_validate(data)
        # return Dialog.model_validate(data)

    def from_json(self, json_str: str):
        """
        Creates a AudioDialog object from a JSON string.

        :param json_str: The JSON string containing audio dialog data.
        :type json_str: str
        :return: The created AudioDialog object.
        :rtype: AudioDialog
        """
        return AudioDialog.from_dict(json.loads(json_str))

    def to_file(self, path: str = None, makedir: bool = True, overwrite: bool = True):
        """
        Saves the audio dialog to a JSON file.

        :param path: Output file path, if not provided, uses the same path used to load the audio dialog.
        :type path: str
        :param makedir: If True, creates parent directories as needed.
        :type makedir: bool
        :param overwrite: If True, overwrites the file if it already exists.
        :type overwrite: bool
        """
        if not path:
            if self._path:
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
            # writer.write(self.json(string=True))

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
