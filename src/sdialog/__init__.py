"""
sdialog: Synthetic Dialogue Generation Toolkit

This package provides utilities for generating synthetic dialogues using instruction-tuned large language models (LLMs).
Dialogues are generated primarily via role-playing, where each agent is defined by a Persona object. The package
supports dialogue orchestration, scenario management, and flexible serialization for downstream tasks.

Main components:

    - Dialog, Turn, Event: Data structures for representing dialogues and their events.
    - Persona and PersonaAgent: For defining and simulating role-played agents.
    - Orchestrators: For controlling agent behavior during dialogue generation.
    - Utility functions for serialization, pretty-printing, and file I/O.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import json
import logging
import importlib
import subprocess

from print_color import print as cprint
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Any

from .util import make_serializable

__version__ = "0.0.2"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# import config sumbodule as "config" attribute of the package
config = importlib.import_module("sdialog.config")


def _get_dynamic_version() -> str:
    """ Retrieves the current version of the package, appending the current git commit hash if available."""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        # If not a valid commit hash, set to empty string
        if re.match(r"\b[0-9a-f]{5,40}\b", commit_hash):
            return f"{__version__}+{commit_hash}"
    except Exception:
        pass
    return __version__


class Turn(BaseModel):
    """
    Represents a single turn in a dialogue.

    :ivar speaker: The name or role of the speaker.
    :vartype speaker: Optional[str]
    :ivar text: The utterance text for this turn.
    :vartype text: str
    """
    speaker: Optional[str]
    text: str


class Event(BaseModel):
    """
    Represents an event in a dialogue, which may be an utterance, instruction, or other action.

    :ivar agent: The agent responsible for the event (e.g., "user", "system").
    :vartype agent: Optional[str]
    :ivar action: The type of event (e.g., "utter", "instruct").
    :vartype action: str
    :ivar actionLabel: A label describing the action (e.g., type of instruction).
    :vartype actionLabel: Optional[str]
    :ivar text: The content of the event.
    :vartype text: str
    :ivar timestamp: The Unix timestamp of the event.
    :vartype timestamp: int
    """
    agent: Optional[str] = None  # "user", "system"
    action: str  # "utter", "instruct"
    actionLabel: Optional[str] = None  # action label (e.g. type of instruct)
    text: str  # the content of the event
    timestamp: int  # timestemp


class Dialog(BaseModel):
    """
    Represents a full dialogue, including turns, events, and scenario metadata.

    :ivar version: Version of the dialogue format (sdialog version).
    :vartype version: Optional[str]
    :ivar model: The model used to generate the dialogue.
    :vartype model: Optional[str]
    :ivar seed: The random seed used for generation.
    :vartype seed: Optional[int]
    :ivar dialogId: Unique identifier for the dialogue.
    :vartype dialogId: Optional[int]
    :ivar dialogIdParent: ID of the parent dialogue, if any.
    :vartype dialogIdParent: Optional[int]
    :ivar complete: Whether the dialogue is complete.
    :vartype complete: Optional[bool]
    :ivar personas: Personas used in the dialogue, mapping speaker names to their attributes.
    :ivar scenario: Scenario description or metadata.
    :vartype scenario: Optional[Union[dict, str]]
    :ivar turns: List of dialogue turns.
    :vartype turns: List[Turn]
    :ivar events: List of dialogue events (optional).
    :vartype events: Optional[List[Event]]
    :ivar notes: Free-text notes or comments about the dialogue.
    :vartype notes: Optional[str]
    """
    version: Optional[str] = Field(default_factory=_get_dynamic_version)  # Version of the format
    model: Optional[str] = None  # the model used to generate the dialogue
    seed: Optional[int] = None  # the seed used to generate the dialogue
    id: Optional[int] = None  # Unique ID for the dialogue
    parentId: Optional[int] = None  # ID of the parent dialogue, if any
    complete: Optional[bool] = None
    personas: Optional[dict[str, Any]] = None  # Any is a subclass of MetaPersona
    scenario: Optional[Union[dict, str]] = None  # the scenario used to generated the dialogue
    turns: List[Turn]  # the list of turns of the conversation
    events: Optional[List[Event]] = None
    notes: Optional[str] = None  # Free-text notes or comments about the dialogue

    def __len__(self):
        """
        Returns the number of turns in the dialogue.

        :return: Number of turns.
        :rtype: int
        """
        return len(self.turns)

    def length(self, mode: str = "words", words_per_minute: int = 130) -> int:
        """
        Returns the length of the dialogue according to the specified mode (number of words by default).

        :param mode: The mode for measuring length. Options:
            - "turns": Number of turns (default)
            - "words": Total number of words in all turns
            - "minutes" / "time": Approximate duration in minutes (`words_per_minute`/minute)
        :type mode: str
        :param words_per_minute: Words per minute for "minutes" mode (default is 130, which is a common estimate).
        :type words_per_minute: int
        :return: The computed length according to the mode.
        :rtype: int
        :raises ValueError: If an unknown mode is specified.
        """
        mode = mode.lower()
        if mode == "turns":
            return len(self.turns)
        elif mode == "words":
            return sum(len(turn.text.split()) for turn in self.turns)
        elif mode in ["minutes", "time"]:
            total_words = sum(len(turn.text.split()) for turn in self.turns)
            return max(1, int(round(total_words / words_per_minute)))
        else:
            raise ValueError(f"Unknown mode for get_length: {mode}")

    def description(self, turn_template: str = "{speaker}: {text}"):
        """
        Returns a human-readable string representation of the dialogue.

        :param turn_template: Template for formatting each turn.
        :type turn_template: str
        :return: The formatted dialogue.
        :rtype: str
        """
        return "\n".join(turn_template.format(speaker=turn.speaker, text=turn.text.replace("\n", " "))
                         for turn in self.turns)

    def json(self, string: bool = False, indent: int = 2):
        """
        Serializes the dialogue to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :return: The serialized dialogue.
        :rtype: Union[str, dict]
        """
        data = self.model_dump()
        make_serializable(data)
        return json.dumps(data, indent=indent) if string else data

    def print(self, *a, **kw):
        """
        Pretty-prints a dialogue to the console, with optional scenario and orchestration details.

        :param scenario: If True, prints scenario information.
        :type scenario: bool
        :param orchestration: If True, prints orchestration events.
        :type orchestration: bool
        """
        _print_dialog(self, *a, **kw)

    def to_file(self, path: str, type: str = "auto", makedir: bool = True):
        """
        Saves the dialogue to a file in either JSON or plain text format.

        :param path: Output file path.
        :type path: str
        :param type: "json", "txt", or "auto" (determined by file extension).
        :type type: str
        :param makedir: If True, creates parent directories as needed.
        :type makedir: bool
        """
        if type == "auto":
            type = "json" if path.endswith(".json") else "txt"

        if makedir and os.path.split(path)[0]:
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        with open(path, "w") as writer:
            if type == "json":
                writer.write(self.json(string=True))
            else:
                writer.write(self.description())

    @staticmethod
    def from_file(path: str, type: str = "auto"):
        """
        Loads a dialogue from a file.

        :param path: Path to the dialogue file.
        :type path: str
        :param type: "json", "txt", or "auto" (determined by file extension).
        :type type: str
        :return: The loaded dialogue object.
        :rtype: Dialog
        """
        if type == "auto":
            type = "json" if path.endswith(".json") else "txt"

        with open(path) as reader:
            if type == "json":
                return Dialog.from_dict(json.load(reader))

            lines = reader.read().split("\n")

        return Dialog(turns=[Turn(speaker=line[:line.index(":")].strip(),
                                  text=line[line.index(":") + 1:].strip())
                             for line in lines if line])

    @staticmethod
    def from_dict(data: dict):
        """
        Creates a Dialog object from a dictionary.

        :param data: The dictionary containing dialogue data.
        :type data: dict
        :return: The created Dialog object.
        :rtype: Dialog
        """
        return Dialog.model_validate(data)

    def from_json(self, json_str: str):
        """
        Creates a Dialog object from a JSON string.

        :param json_str: The JSON string containing dialogue data.
        :type json_str: str
        :return: The created Dialog object.
        :rtype: Dialog
        """
        return Dialog.from_dict(json.loads(json_str))

    def to_audio(self, path=None):
        """ Converts the dialogue to audio format.

        :param path: If provided, saves the audio to this file path.
        :type path: Optional[str]
        :return: The audio data as a numpy array.
        :rtype: np.ndarray
        """
        from .audio import dialog_to_audio, to_wav

        audio = dialog_to_audio(self)

        if path:
            if not path.endswith(".wav"):
                path += ".wav"
            to_wav(audio, path)

        return audio

    def get_length(self, mode: str = "turns") -> float:
        """
        Returns the length of the dialogue according to the specified mode.

        :param mode: The mode for measuring length. Options are:
            - "turns": Number of turns (default)
            - "words": Total number of words in all turns
            - "minutes": Approximate duration in minutes (assuming 150 words per minute)
        :type mode: str
        :return: The length of the dialogue according to the selected mode.
        :rtype: float
        """
        if mode == "turns":
            return float(len(self.turns))
        elif mode == "words":
            return float(sum(len(turn.text.split()) for turn in self.turns))
        elif mode == "minutes":
            total_words = sum(len(turn.text.split()) for turn in self.turns)
            return float(total_words) / 150.0  # 150 words per minute is a common estimate
        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'turns', 'words', 'minutes'.")

    __str__ = description


class Instruction(BaseModel):
    """
    Represents an instruction to an agent, optionally with associated events.

    :ivar text: The instruction text.
    :vartype text: str
    :ivar events: Associated events (optional).
    :vartype events: Optional[Union[Event, List[Event]]]
    """
    text: str = None
    events: Optional[Union[Event, List[Event]]] = None  # extra events


def _print_dialog(dialog: Union[Dialog, dict], scenario: bool = False, orchestration: bool = False):
    """
    Pretty-prints a dialogue to the console, with optional scenario and orchestration details.

    :param dialog: The dialogue to print.
    :type dialog: Union[Dialog, dict]
    :param scenario: If True, prints scenario information.
    :type scenario: bool
    :param orchestration: If True, prints also orchestration events.
    :type orchestration: bool
    """
    if type(dialog) is dict:
        dialog = Dialog.model_validate(dialog)

    speaker_tag_colors = ["red", "blue", "yellow", "cyan", "green", "magenta", "purple"]
    speaker_utt_colors = ["grey", "white"]
    # speaker_utt_colors = ["black", "grey"]

    if dialog.id:
        cprint(dialog.id, tag="dialog_id", tag_color="purple", color="magenta", format="bold")
    if dialog.complete:
        cprint(dialog.complete, tag="complete", tag_color="purple", color="magenta", format="bold")
    if dialog.model:
        cprint(dialog.model, tag="model", tag_color="purple", color="magenta", format="bold")
    if dialog.seed:
        cprint(dialog.seed, tag="seed", tag_color="purple", color="magenta", format="bold")
    if scenario and dialog.scenario:
        cprint("", tag="scenario", tag_color="purple", color="magenta", format="bold")
        if type(dialog.scenario) is str:
            cprint(dialog.scenario, color="magenta")
        else:
            cprint(json.dumps(dialog.scenario, indent=2), color="magenta")

    cprint("--- Dialogue Begins ---", color="magenta", format="bold")
    speakers = sorted(list(set(turn.speaker for turn in dialog.turns)))
    if orchestration:
        dialog = dialog.model_copy()
        dialog.turns = [Turn(speaker=e.agent, text=e.text) if e.action == "utter"
                        else (
                            Turn(speaker=e.agent, text=f"[pick_suggestion] {e.text}") if e.action == "pick_suggestion"
                            else
                            Turn(speaker=e.action, text=f"({e.agent}) {e.text}"))
                        for e in dialog.events]

    for ix, turn in enumerate(dialog.turns):
        speaker = turn.speaker

        if speaker not in speakers:
            tag_color = "yellow"
            color = "purple"
        else:
            tag_color = speaker_tag_colors[speakers.index(speaker) % len(speaker_tag_colors)]
            color = speaker_utt_colors[speakers.index(speaker) % len(speaker_utt_colors)]

        cprint(turn.text,
               tag=speaker,
               tag_color=tag_color,
               color=color)
    cprint("--- Dialogue Ends ---", color="magenta", format="bold")
