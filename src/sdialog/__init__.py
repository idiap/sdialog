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
import csv
import logging
import importlib
import subprocess

from tqdm.auto import tqdm
from pydantic import BaseModel, Field
from print_color import print as cprint
from typing import List, Union, Optional, Any

from .util import make_serializable, get_timestamp, remove_newlines, get_universal_id

__version__ = "0.1.0"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
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
    speaker: Optional[str] = None
    text: str

    def __str__(self):
        return f"{self.speaker}: {self.text}"


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

    def model_post_init(self, context: Any, /) -> None:        # This runs after __init__
        logger.log(level=logging.DEBUG, msg=f"Event: {self}")


class Dialog(BaseModel):
    """
    Represents a full dialogue, including turns, events, and scenario metadata.

    :ivar version: Version of the dialogue format (sdialog version).
    :vartype version: Optional[str]
    :ivar timestamp: Timestamp of dialogue creation (e.g., "2025-01-01T12:00:00Z").
    :vartype timestamp: Optional[str]
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
    timestamp: Optional[str] = Field(default_factory=get_timestamp)  # Timestamp of dialogue creation
    model: Optional[str] = None  # the model used to generate the dialogue
    seed: Optional[int] = None  # the seed used to generate the dialogue
    id: Optional[Union[int, str]] = Field(default_factory=get_universal_id)  # Unique ID for the dialogue
    parentId: Optional[Union[int, str]] = None  # ID of the parent dialogue, if any
    complete: Optional[bool] = None
    personas: Optional[dict[str, Any]] = None  # Any is a subclass of MetaPersona
    scenario: Optional[Union[dict, str]] = None  # the scenario used to generated the dialogue
    turns: List[Turn]  # the list of turns of the conversation
    events: Optional[List[Event]] = None  # the list of events of the conversation (optional)
    notes: Optional[str] = None  # Free-text notes or comments about the dialogue
    _path: Optional[str] = None  # Path to the file where the dialogue was loaded or saved

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

    def clone(self, new_id: int = None) -> "Dialog":
        """
        Creates a deep copy of the dialogue.

        :return: A new Dialog object that is a copy of this one.
        :rtype: Dialog
        """
        cloned = Dialog.from_dict(self.json())
        cloned.parentId = cloned.id
        cloned.id = new_id if new_id is not None else get_universal_id()

        return cloned

    def description(self, turn_template: str = None):
        """
        Returns a human-readable string representation of the dialogue.

        :param turn_template: Template for formatting each turn (default "{speaker}: {text}").
        :type turn_template: str
        :return: The formatted dialogue.
        :rtype: str
        """
        if turn_template is None:
            return "\n".join(f"{turn.speaker}: " + turn.text.replace('\n', ' ') if turn.speaker else turn.text
                             for turn in self.turns)

        return "\n".join(turn_template.format(speaker="" if turn.speaker is None else turn.speaker,
                                              text=turn.text.replace("\n", " "))
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

    def to_file(self, path: str = None, type: str = "auto", makedir: bool = True, overwrite: bool = True):
        """
        Saves the dialogue to a file in JSON, CSV, or plain text format.

        :param path: Output file path, if not provided, uses the same path used to load the dialogue.
        :type path: str
        :param type: "json", "csv", "txt", or "auto" (determined by file extension).
        :type type: str
        :param makedir: If True, creates parent directories as needed.
        :type makedir: bool
        """
        if not path:
            if self._path:
                path = self._path
            else:
                raise ValueError("No path provided to save the dialogue and no loading path available. "
                                 "Please specify a valid file path.")

        if type == "auto":
            _, ext = os.path.splitext(path)
            ext = ext.lower()[1:]
            type = ext if ext in ["json", "txt", "csv", "tsv"] else "txt"

        if makedir and os.path.split(path)[0]:
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"File '{path}' already exists. Use 'overwrite=True' to overwrite it.")

        with open(path, "w", newline='') as writer:
            if type == "json":
                writer.write(self.json(string=True))
            elif type in ["csv", "tsv"]:
                # set delimiter based on desired type
                delimiter = {"csv": ",", "tsv": "\t"}[type]
                csv_writer = csv.writer(writer, delimiter=delimiter)
                # write the header
                csv_writer.writerow(["speaker", "text"])
                # write the turns
                for turn in self.turns:
                    csv_writer.writerow([turn.speaker, turn.text])
            else:
                writer.write(self.description())

    @staticmethod
    def from_file(path: str,
                  type: str = "auto",
                  txt_template: str = "{speaker}: {text}",
                  csv_speaker_col: Union[int, str] = "speaker",
                  csv_text_col: Union[int, str] = "text") -> Union["Dialog", List["Dialog"]]:
        """
        Loads a dialogue from a file.

        :param path: Path to the dialogue file or directory. In case of a directory, all dialogues in the directory
                     will be loaded and returned as a list of Dialog objects.
        :type path: str
        :param type: "json", "txt", "csv", "tsv", or "auto" (determined by file extension).
        :type type: str
        :param txt_turn_template: Template for parsing text dialogue turns (default "{speaker}: {text}").
        :type txt_turn_template: str
        :param csv_speaker_col: Column identifier for speaker in CSV/TSV files (can be index or header name).
        :type csv_speaker_col: Union[int, str]
        :param csv_text_col: Column identifier for text in CSV/TSV files (can be index or header name).
        :type csv_text_col: Union[int, str]
        :return: The loaded dialogue object.
        :rtype: Dialog
        :raises ValueError: If the file format is not recognized or if required columns are missing
        """
        if os.path.isdir(path):
            # Let's load first all dialogues without a stored ID (all non-json files)
            filenames = sorted([filename
                                for filename in os.listdir(path)
                                if ((type == "auto" and filename.endswith((".txt", ".csv", ".tsv")))
                                    or (type != "json" and filename.endswith(type)))])
            dialogs = [Dialog.from_file(os.path.join(path, filename), type=type,
                                        txt_template=txt_template,
                                        csv_speaker_col=csv_speaker_col,
                                        csv_text_col=csv_text_col)
                       for filename in tqdm(filenames, desc="Loading dialogues from directory", leave=False)]
            # Make sure the ID is always the same, for the same file (as long as no more files are added)
            for ix, dialog in enumerate(dialogs):
                dialog.id = ix + 1
            # Adding json files too, assuming they have an id already
            dialogs.extend([Dialog.from_file(os.path.join(path, filename), type=type,
                                             txt_template=txt_template,
                                             csv_speaker_col=csv_speaker_col,
                                             csv_text_col=csv_text_col)
                            for filename in os.listdir(path)
                            if (type in ["auto", "json"]) and filename.endswith(".json")])
            return dialogs

        type = type.lower()
        if type == "auto":
            _, ext = os.path.splitext(path)
            ext = ext.lower()[1:]
            type = ext if ext in ["json", "txt", "csv", "tsv"] else "txt"

        turns = []
        with open(path) as reader:
            if type == "json":
                dialog = Dialog.from_dict(json.load(reader))
                dialog._path = path  # Store the path for later use
                return dialog
            elif type in ["csv", "tsv"]:
                is_tsv = type == "tsv"
                if isinstance(csv_speaker_col, str) and isinstance(csv_text_col, str):
                    reader = csv.DictReader(reader, delimiter='\t' if is_tsv else ',')
                elif isinstance(csv_speaker_col, int) and isinstance(csv_text_col, int):
                    reader = csv.reader(reader, delimiter='\t' if is_tsv else ',')
                else:
                    raise ValueError(f"File '{path}': `csv_speaker_col` and `csv_text_col` must be either both "
                                     "strings (column names) or both integers (column indices).")

                for ix, row in enumerate(reader):
                    speaker = row[csv_speaker_col]
                    text = row[csv_text_col]
                    if speaker is None:
                        raise ValueError(f"Missing speaker in row {ix}: {row}")
                    if not text:
                        logger.warning(f"File '{path}': Empty text in row {ix}: {row}. Skipping this turn.")
                        continue
                    turns.append((speaker.strip(), text.strip()))
            elif type == "txt":
                try:
                    dialog = Dialog.from_str(reader.read(), template=txt_template)
                    dialog._path = path
                    return dialog
                except ValueError as e:
                    raise ValueError(f"File '{path}': {str(e)}")
            else:
                raise ValueError(f"Unknown file type '{type}'. Supported types: 'json', 'txt', 'csv', 'tsv'.")

            dialog = Dialog(turns=[Turn(speaker=speaker, text=text) for speaker, text in turns])
            dialog._path = path
            return dialog

    @staticmethod
    def from_str(dialog_text: str,
                 template: str = "{speaker}: {text}",
                 default_speakers: List[str] = None,
                 id: Union[str, int] = None) -> "Dialog":
        """
        Creates a Dialog object from a string representation of a dialogue.

        :param dialog_text: The dialogue text, with each turn on a new line.
        :type dialog_text: str
        :param template: The template for parsing each turn. Default is "{speaker}: {text}".
        :type template: str
        :param default_speakers: Optional list of default speakers to use if no present in the text or template.
                                 The speakers will be assigned in order of appearance, in alternating turns.
                                 Default is None (speaker field will be empty in each turn).
        :type default_speakers: List[str]
        :param id: Optional ID for the dialogue. If not provided, a universal ID will be generated.
        :type id: Union[str, int]
        :return: The created Dialog object.
        :rtype: Dialog
        """
        if default_speakers is not None and not isinstance(default_speakers, list):
            raise ValueError("default_speakers must be a list of strings.")

        turns = []
        default_speaker_ix = 0
        lines = dialog_text.split("\n")
        for ix, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Use the template to extract speaker and text for each turn
            # Build a regex from the template
            regex = re.escape(template)
            regex = regex.replace(r'\{speaker\}', r'(?P<speaker>.+?)')
            regex = regex.replace(r'\{text\}', r'(?P<text>.+)')
            m = re.match(regex, line)
            if m:
                try:
                    speaker = m.group('speaker').strip()
                except IndexError:
                    speaker = default_speakers[default_speaker_ix % len(default_speakers)] if default_speakers else None
                    default_speaker_ix += 1
                text = m.group('text').strip()
            else:
                raise ValueError(f"Line {ix + 1} '{line}' does not match the expected "
                                 f"format: {template}. Make sure the template "
                                 "matches the dialogue format.")

            turns.append((speaker, text))
        dialog = Dialog(turns=[Turn(speaker=speaker, text=text) for speaker, text in turns])
        if id is not None:
            dialog.id = id
        return dialog

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

    def rename_speaker(self, old_name: str,
                       new_name: str,
                       case_sensitive: bool = False,
                       in_events: bool = True) -> "Dialog":
        """
        Renames all occurrences of a speaker in the dialogue's turns (and optionally events).

        :param old_name: The current speaker name to replace.
        :type old_name: str
        :param new_name: The new speaker name.
        :type new_name: str
        :param case_sensitive: Whether to match speaker names case-sensitively (default: False).
        :type case_sensitive: bool
        :param in_events: Whether to also rename in events' agent fields (default: True).
        :type in_events: bool
        """
        def match(name):
            if case_sensitive:
                return name == old_name
            else:
                return name.lower() == old_name.lower() if name is not None else False

        # Rename in turns
        for turn in self.turns:
            if turn.speaker is not None and match(turn.speaker):
                turn.speaker = new_name

        # Rename in events (if present and requested)
        if in_events and self.events:
            for event in self.events:
                if hasattr(event, 'agent') and event.agent is not None and match(event.agent):
                    event.agent = new_name

        return self

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
                            Turn(speaker=e.agent,
                                 text=f"[pick_suggestion] {remove_newlines(e.text)}") if e.action == "pick_suggestion"
                            else
                            Turn(speaker=e.action, text=f"({e.agent}) {remove_newlines(e.text)}"))
                        for e in dialog.events]

    for ix, turn in enumerate(dialog.turns):
        speaker = turn.speaker

        if speaker not in speakers:
            tag_color = "yellow"
            color = "purple"
        else:
            tag_color = speaker_tag_colors[speakers.index(speaker) % len(speaker_tag_colors)]
            color = speaker_utt_colors[speakers.index(speaker) % len(speaker_utt_colors)]

        cprint(remove_newlines(turn.text),
               tag=speaker,
               tag_color=tag_color,
               color=color)
    cprint("--- Dialogue Ends ---", color="magenta", format="bold")
