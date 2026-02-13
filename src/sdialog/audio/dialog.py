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
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>, Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

import re
import os
import json
import random
import dscaper
import numpy as np
import soundfile as sf
from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import HumanMessage, SystemMessage

from sdialog import Dialog
from sdialog.config import config
from sdialog.audio.turn import AudioTurn
from sdialog.util import get_llm_model, get_universal_id
from sdialog.audio.tts.base import BaseTTS, BaseVoiceCloneTTS
from sdialog.audio.voice_database import BaseVoiceDatabase, Voice
from sdialog.audio.room import AudioSource, Room, MicrophonePosition, RoomPosition
from sdialog.audio.utils import Role, CaseInsensitiveDict, logger, generate_reference_voices, SourceVolume


class RoomAcousticsConfig(BaseModel):
    audio_path: str
    microphone_position: MicrophonePosition
    room_name: str
    room: Room
    source_volumes: Dict[str, SourceVolume] = Field(default_factory=dict)
    kwargs_pyroom: Dict[str, Any] = Field(default_factory=dict)
    background_effect: str = "white_noise"
    foreground_effect: str = "ac_noise_minimal"
    foreground_effect_position: RoomPosition = RoomPosition.TOP_RIGHT
    audio_paths_post_processing: Dict[str, str] = Field(default_factory=dict)

    @field_validator("source_volumes", "kwargs_pyroom", "audio_paths_post_processing", mode="before")
    def _validate_dicts(cls, v):
        return v if v is not None else {}

    @field_validator("background_effect", mode="before")
    def _validate_background_effect(cls, v):
        return v if v is not None else "white_noise"

    @field_validator("foreground_effect", mode="before")
    def _validate_foreground_effect(cls, v):
        return v if v is not None else "ac_noise_minimal"

    @field_validator("foreground_effect_position", mode="before")
    def _validate_foreground_effect_position(cls, v):
        return v if v is not None else RoomPosition.TOP_RIGHT

    class Config:
        arbitrary_types_allowed = True


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
    audio_step_3_filepaths: Dict[str, RoomAcousticsConfig] = {}

    speakers_names: dict[str, str] = {}  # role2name mapping
    speakers_roles: dict[str, str] = CaseInsensitiveDict()  # name2role mapping (case insensitive)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def clone(self, new_id: int = None) -> "AudioDialog":
        """
        Creates a deep copy of the dialogue.

        :param new_id: Optional ID to assign to the cloned dialog. If None, a new universal ID is generated.
        :type new_id: int, optional
        :return: A new AudioDialog object that is a deep copy of this one, with updated id and parentId.
        :rtype: AudioDialog
        """
        cloned = AudioDialog.from_dict(self.json())
        cloned.parentId = cloned.id
        cloned.id = new_id if new_id is not None else get_universal_id()

        return cloned

    def _update_speakers_roles(self):
        """Compute the mapping between speaker names and roles needed for audio inference pipeline."""
        # Get the unique speakers in the dialog in the order of appearance
        speakers = []
        for turn in self.turns:
            if turn.speaker not in speakers:
                speakers.append(turn.speaker)
                if len(speakers) == 2:
                    break

        # In case dialog was loaded from a file or json
        if not isinstance(self.speakers_roles, CaseInsensitiveDict):
            self.speakers_roles = CaseInsensitiveDict()

        # Create role mappings for speaker identification
        self.speakers_names[Role.SPEAKER_1] = speakers[0]
        self.speakers_names[Role.SPEAKER_2] = speakers[1]

        # Create reverse mappings for role lookup
        self.speakers_roles[speakers[0]] = Role.SPEAKER_1
        self.speakers_roles[speakers[1]] = Role.SPEAKER_2

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
        audio_dialog._update_speakers_roles()

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
        audio_dialog = AudioDialog.model_validate(data)
        audio_dialog._update_speakers_roles()
        return audio_dialog

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
            writer.write(self.model_dump_json_safe(indent=2))

    def to_string(self):
        return self.model_dump_json_safe(indent=4)

    def model_dump_json_safe(self, **kwargs) -> str:
        """
        Safely dumps the AudioDialog object to a JSON string.

        This method overrides the default model_dump_json to ensure that all
        audio-specific attributes are included in the JSON output. It also
        handles any special cases for serializing audio data or file paths.

        :param kwargs: Additional keyword arguments passed to the base model_dump_json method.
        :return: A JSON string representation of the AudioDialog object.
        :rtype: str
        """
        try:
            return self.model_dump_json(**kwargs)
        except Exception:
            # Set all turn.voice to "voice sample (non serializable)" to avoid serialization issues
            for turn in self.turns:
                turn.voice = "voice sample (non serializable)"
            for speaker in self.personas:
                if "voice" in self.personas[speaker]:
                    self.personas[speaker]["voice"] = "voice sample (non serializable)"
            return self.model_dump_json(**kwargs)

    def display(self):
        """
        Displays the audio dialog.
        """
        from IPython.display import Audio, display

        if len(self.audio_step_1_filepath) > 0:
            print("-" * 25)
            print("TTS Audio:")
            print("-" * 25)
            display(Audio(
                self.audio_step_1_filepath,
                autoplay=False
            ))

        if len(self.audio_step_3_filepaths) > 0:

            print("-" * 25)
            print("- Room Configurations")
            print("-" * 25)

            # For each room configuration, display the original audio and the processed audio
            for config_name in self.audio_step_3_filepaths:
                config_data = self.audio_step_3_filepaths[config_name]
                print(f"> Room Configuration: {config_name}")
                print("Room Accoustic Audio:")
                display(Audio(
                    config_data.audio_path,
                    autoplay=False
                ))

                # If the room configuration has processed audio, display it
                if (
                    config_name in self.audio_step_3_filepaths
                    and config_data.audio_paths_post_processing is not None
                    and len(config_data.audio_paths_post_processing) > 0
                ):
                    print("#" * 10)
                    print("Post Processing Audio (e.g. microphone effect):")
                    print("#" * 10)

                    # For each recording device, display the processed audio
                    for _rd in config_data.audio_paths_post_processing:
                        display(Audio(
                            config_data.audio_paths_post_processing[_rd],
                            autoplay=False
                        ))

    def save_utterances_audios(
        self,
        dir_audio: str,
        project_path: str,
        sampling_rate: int = 24_000
    ) -> None:
        """
        Saves individual utterance audio files to the specified directory structure.

        This function creates the necessary directory structure and saves each turn's
        audio as a separate WAV file. It also calculates timing information for each
        utterance and updates the AudioTurn objects with file paths and timing data.

        If the sampling rate of the audio obtained from the TTS engine is not the same
        as the sampling rate of the project, we will resample the audio to the sampling
        rate of the project.

        Directory structure created:
        - {project_path}/utterances/ - Individual utterance audio files
        - {project_path}/exported_audios/ - Combined audio files
        - {project_path}/exported_audios/rooms/ - Room acoustics simulation results

        :param dir_audio: Base directory path for audio storage.
        :type dir_audio: str
        :param project_path: Project-specific path for organizing audio files.
        :type project_path: str
        :param sampling_rate: Audio sampling rate for saving files (default: 24000 Hz).
        :type sampling_rate: int
        """

        self.audio_dir_path = dir_audio.rstrip("/")
        os.makedirs(f"{project_path}/utterances", exist_ok=True)
        os.makedirs(f"{project_path}/exported_audios", exist_ok=True)
        os.makedirs(f"{project_path}/exported_audios/rooms", exist_ok=True)

        for idx, turn in enumerate(self.turns):

            audio_data = turn.get_audio()

            # Build the path to the audio file
            turn.audio_path = f"{project_path}/utterances/{idx}_{turn.speaker}.wav"

            # Calculate the duration of the audio
            turn.audio_duration = audio_data.shape[0] / sampling_rate

            # Save the audio file
            sf.write(turn.audio_path, audio_data, sampling_rate)

    def update_turn_timings(self):
        """
        Updates the start times of turns based on their duration and gap durations.
        Useful after computing overlaps/pauses.
        """
        current_time = 0.0
        for turn in self.turns:
            turn.audio_start_time = current_time
            current_time += turn.audio_duration + turn.gap_duration
            if current_time < 0:
                current_time = 0.0

    # TODO: this method should be rename since does not return anything, maybe more like
    #       "update_voices_from_personas" or "assign_voices_to_personas", and perhaps should be a private method
    def persona_to_voice(
        self,
        voice_database: BaseVoiceDatabase,
        persona_to_voice_desc: Union[str, callable] = None,
        voices: dict[Role, Union[Voice, tuple[str, str]]] = None,
        keep_duplicate: bool = False,
        tts_engine: BaseTTS | BaseVoiceCloneTTS = None,
        seed: int = None
    ) -> None:
        """
        Assigns appropriate voices to speakers based on their persona characteristics.

        This function analyzes each speaker's persona information (gender, age, language)
        and assigns a suitable voice from the voice database. If persona information is
        missing, default values are assigned with appropriate warnings.

        Voice assignment logic:
        1. If explicit voices are provided, use them for the specified roles
        2. If no explicit voices, select from database based on persona characteristics
        3. Handle missing persona information by assigning random/default values
        4. Support both Voice objects and voice identifier tuples

        :param voice_database: Database containing available voices with metadata.
        :type voice_database: BaseVoiceDatabase
        :param persona_to_voice_desc: Jinja2 template string or function that takes persona dictionary
                                      and returns its voice descriptions. Defaults to a template with
                                      gender and age only.
        :type persona_to_voice_desc: Union[str, callable]
        :param voices: Optional dictionary mapping speaker roles to specific voices.
                    Keys are Role enums, values can be Voice objects or (identifier, language) tuples.
        :type voices: Optional[dict[Role, Union[Voice, tuple[str, str]]]]
        :param keep_duplicate: If True, allows voice reuse across speakers.
        :type keep_duplicate: bool
        :param seed: Seed for random number generator.
        :type seed: int
        """
        if voices is None and voice_database is None:
            logger.info("No voices provided, generating them dynamically "
                        "based on the persona definition of each speaker.")
            reference_prompts = generate_reference_voices(dialog=self,
                                                          voice_clone_model=tts_engine,
                                                          persona_to_voice_desc=persona_to_voice_desc)
            voices = {role: reference_prompts.get(speaker) for speaker, role in self.speakers_roles.items()}

        for speaker, persona in self.personas.items():

            # Check if the information about the voice is already in the persona, else add a random information
            if "gender" not in persona or persona["gender"] is None:
                persona["gender"] = random.choice(["male", "female"])
                logger.warning(f"Gender not found in the persona {speaker}, a random gender has been added")

            if "age" not in persona or persona["age"] is None:
                persona["age"] = random.randint(18, 65)
                logger.warning(f"Age not found in the persona {speaker}, a random age has been added")

            if "language" not in persona or persona["language"] is None:
                persona["language"] = "english"
                logger.warning(
                    f"Language not found in the persona {speaker}, english has been considered by default"
                )

            # TODO: Why do we need roles Spekaer 1 or 2?? if we have the speaker name in personas??
            # Get the role of the speaker (speaker_1 or speaker_2)
            role: Role = self.speakers_roles[speaker]

            print(f"voices: {voices}")
            print(f"role: {role}")
            print(f"speaker: {speaker}")

            if voices and role not in voices and speaker not in voices:
                raise ValueError(f"Voice for {str(role)} not found in the voices dictionary")

            # If no voices are provided, get a voice from the voice database based on the gender, age and language
            if voices is None or voices == {}:
                if voice_database is not None:
                    persona["voice"] = voice_database.get_voice(
                        gender=persona["gender"],
                        age=persona["age"],
                        lang=persona["language"],
                        keep_duplicate=keep_duplicate,
                        seed=seed
                    )

            # If the voice of the speaker is provided as a Voice object
            elif isinstance(voices[role], Voice):
                persona["voice"] = voices[role]

            # If the voice of the speaker is provided as an identifier (like "am_echo")
            # check if first item is str to avoid (array, sampling_rate) tuple cases
            elif (isinstance(voices[role], tuple) and isinstance(voices[role][0], str)
                  and voice_database is not None):
                _identifier, _language = voices[role]
                persona["voice"] = voice_database.get_voice_by_identifier(
                    _identifier,
                    _language,
                    keep_duplicate=keep_duplicate
                )
            elif isinstance(voices[role], str):
                persona["voice"] = Voice(voice=voices[role])
            else:
                # Fallback, forward the provided value directly to the TTS engine
                persona["voice"] = voices[role]

    def get_dry_audio(self) -> np.ndarray:
        """
        Get the combination of multiple audio segments into a single master audio track.

        Concatenates all audio segments from the dialogue turns into a single
        continuous audio track. This creates a baseline audio representation
        of the entire dialogue for further processing and analysis.

        :return: Combined audio data as numpy array.
        :rtype: np.ndarray
        """
        _audio, _sr = sf.read(self.audio_step_2_filepath)
        return _audio

    def add_sound_effects(
        self,
        room: Any = None,
        dscaper_manager: dscaper.Dscaper = None,
        available_sound_effects: dict[str, dict] = None,
        model_name_alignment: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        dropout: float = 0.0,
        verbose: bool = False,
        skip_annotation: bool = False
    ) -> None:
        """
        Add sound effects (such as door opening, footsteps, etc.) to the audio.

        :param room: The room object to use for furniture information.
        :type room: Any
        :param dscaper_manager: The dSCAPER manager to use for fetching audio files.
        :type dscaper_manager: dscaper.Dscaper
        :param available_sound_effects: The dictionary of available sound effects.
        :type available_sound_effects: dict[str, dict]
        :param model_name_alignment: The name of the model to use for forced alignment.
        :type model_name_alignment: str
        :param dropout: The dropout rate for sound effects.
        :type dropout: float
        :param verbose: Whether to print verbose output.
        :type verbose: bool
        :param skip_annotation: Whether to skip the annotation of the sound effects.
        :type skip_annotation: bool
        """

        if not skip_annotation:
            # Annotate the turns with sound effect tags using LLM
            decorated_turns = self._annotate_sound_effects_from_turns(
                sound_effects_db=available_sound_effects,
                room=room
            )
        else:
            decorated_turns = self.turns

        # If the dropout rate is greater than 0.0, we will drop some of the sound effects tags
        if dropout > 0.0:
            for turn in decorated_turns:
                # Use a callback to drop tags individually with probability `dropout`
                turn.text = re.sub(
                    r"\[(.*?)\]",
                    lambda m: "" if random.random() < dropout else m.group(0),
                    turn.text
                )
                turn.text = turn.text.strip()

        if not decorated_turns:
            logger.warning("No sound effects tags found. Skipping sound effects.")
            return

        # Load aligner model
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner

            logger.info(f"Loading {model_name_alignment} for sound effect alignment...")
            aligner = Qwen3ForcedAligner.from_pretrained(
                model_name_alignment,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            )
        except ImportError:
            logger.error("`qwen-asr` package not found. Skipping forced alignment for sound effects.")
            logger.error("Please install it via: pip install qwen-asr")
            raise ImportError("`qwen-asr` package not found. Please install it via: pip install qwen-asr")
        except Exception as e:
            logger.error(f"Failed to load {model_name_alignment}: {e}")
            return

        # Process each turn to add sound effects
        for i, turn in enumerate(self.turns):

            new_text = decorated_turns[i].text

            # Process the turn to add sound effects
            self._parse_sound_effects(
                turn=turn,
                new_text=new_text,
                available_sound_effects=available_sound_effects,
                aligner=aligner,
                dscaper_manager=dscaper_manager,
                verbose=verbose
            )

        self.update_turn_timings()

    def _annotate_sound_effects_from_turns(
        self,
        sound_effects_db: Dict[str, Dict],
        room: Any = None
    ) -> List[BaseModel]:
        """
        Uses LLM to add sound effect tags to the dialogue turns.

        :param sound_effects_db: Dictionary of available sound effects.
        :param room: The room object to use for furniture information.
        :return: List of decorated turns or None if failed.
        """

        llm_params = config["llm"].copy()
        model_name = llm_params.pop("model", None)

        if not model_name:
            logger.warning("LLM model not configured. Skipping sound effects generation.")
            return None

        # Warn user about English only focus
        logger.warning("The sound effects feature currently focuses on English dialogues only.")

        class DecoratedTurn(BaseModel):
            text: str = Field(
                description=(
                    "The text of the turn with sound effect tags added. Format:"
                    " [tag|position]. Example: 'Hello [knock|door] world' or 'Hello [knock] world'."
                )
            )

        class DecoratedDialog(BaseModel):
            turns: List[DecoratedTurn] = Field(description="List of turns with sound effects added.")

        # Prepare prompts
        dialog_text = ""
        for i, turn in enumerate(self.turns):
            dialog_text += f"Turn {i+1} ({turn.speaker}): {turn.text}\n"

        db_description = "\n".join([
            f"- [{tag}]: {info['description']} "
            for tag, info in sound_effects_db.items()
        ])

        position_options = (
            "- 'human': The sound originates from the current speaker's position.\n"
            "- Room Anchors: 'room-center', 'room-top_left', "
            "'room-top_right', 'room-bottom_left', 'room-bottom_right'.\n"
        )

        if room:
            furnitures = list(room.furnitures.keys())
            if furnitures:
                position_options += f"- Furniture: {', '.join(furnitures)}.\n"

        # Check if there are stage tags in the dialog
        has_stage_tags = any("<stage>" in turn.text for turn in self.turns)

        # Load the system prompt from the file
        prompt_file = "annotate_sound_effects_with_stage.txt" if has_stage_tags else "annotate_sound_effects.txt"
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_file)

        with open(prompt_path, "r") as f:
            system_prompt_template = f.read()

        system_prompt = system_prompt_template.format(
            position_options=position_options,
            db_description=db_description
        )

        human_prompt = (
            f"Dialogue:\n{dialog_text}\n\n"
            f"Add sound effects tags to the dialogue turns."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # Invoke LLM
        try:
            llm = get_llm_model(
                model_name=model_name,
                output_format=DecoratedDialog,
                **llm_params
            )
            raw_response = llm.invoke(messages)
            structured_response = DecoratedDialog.model_validate(raw_response)
            decorated_turns = structured_response.turns

            if len(decorated_turns) != len(self.turns):
                logger.error(
                    f"LLM returned {len(decorated_turns)} turns, expected {len(self.turns)}. "
                    "Skipping sound effects."
                )
                return None

            return decorated_turns

        except Exception as e:
            logger.error(f"Failed to generate sound effects with LLM: {e}")
            return None

    def _parse_sound_effects(
            self,
            turn: AudioTurn,
            new_text: str,
            available_sound_effects: dict[str, dict],
            aligner: Any,
            dscaper_manager: dscaper.Dscaper,
            verbose: bool = False) -> None:
        """
        Process a single turn to insert/mix sound effects based on tags and forced alignment.

        :param turn: The audio turn object to modify.
        :type turn: AudioTurn
        :param new_text: The text containing sound effect tags.
        :type new_text: str
        :param available_sound_effects: The sound effects database.
        :type available_sound_effects: dict[str, dict]
        :param aligner: The loaded Qwen3ForcedAligner model.
        :type aligner: Any
        :param dscaper_manager: The dSCAPER manager to use for fetching audio files.
        :type dscaper_manager: dscaper.Dscaper
        :param verbose: Whether to print verbose output.
        :type verbose: bool
        """

        if verbose:
            logger.info(new_text)

        # Check for tags
        tags = re.findall(r"\[(.*?)\]", new_text)
        if not tags:
            return

        current_audio = turn.get_audio()
        if current_audio is None or len(current_audio) == 0:
            return

        sr = turn.sampling_rate

        # Clean text for alignment (remove tags)
        clean_text = re.sub(r"\[(.*?)\]", "", new_text)
        # Remove multiple spaces if any
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if not clean_text:
            # If text is only tags, just play them sequentially or mixed
            alignment = []
        else:
            # Perform forced alignment
            try:

                current_audio = current_audio.numpy()

                audio_input = (current_audio.astype(np.float32), sr)

                results = aligner.align(
                    audio=audio_input,
                    text=clean_text,
                    language="English"  # Focusing on English as requested
                )
                # results is a list (batch) of alignments. We sent one item.
                # structure: results[0] is list of WordItem. Each item has .text, .start_time, .end_time
                alignment = results[0]
            except Exception as e:
                logger.error(f"Forced alignment failed: {e}. Skipping sound effects for this turn.")
                return

        # Find tag positions relative to words
        # We map tags in new_text to positions in clean_text, then to time using alignment
        tag_matches = list(re.finditer(r"\[(.*?)\]", new_text))
        valid_tag_infos = []  # (tag_name, position, timestamp_seconds)

        # Process each tag to add sound effects
        for match in tag_matches:

            full_tag = match.group(1)

            # Parse tag and position
            if '|' in full_tag:
                tag, position = full_tag.split('|', 1)
            else:
                tag = full_tag
                position = "human"  # Default position

            if tag not in available_sound_effects:
                continue

            # Reconstruct clean text up to this tag
            full_clean_pre_text = re.sub(r"\[(.*?)\]", "", new_text[:match.start()])

            # Remove the <stage> and </stage> tags and all in between from the full_clean_pre_text
            if "<stage>" in full_clean_pre_text:
                full_clean_pre_text = re.sub(r"<stage>.*?</stage>", "", full_clean_pre_text)

            found_time = 0.0

            # If there is speech, find the time of the tag in the alignment based on the words before the tag
            if alignment is not None:

                # Heuristic: count words before the tag in the clean text representation
                # and map to the word index in the alignment.
                words_before = full_clean_pre_text.split()
                num_words_before = len(words_before)

                if num_words_before == 0:
                    found_time = alignment[0].start_time
                elif num_words_before >= len(alignment):
                    found_time = alignment[-1].end_time
                else:
                    # Insert after the word corresponding to the count
                    # num_words_before corresponds to the index of the *next* word
                    # so (num_words_before - 1) is the index of the word just processed.
                    last_word = alignment[num_words_before - 1]
                    found_time = last_word.end_time

            valid_tag_infos.append((tag, position, found_time))

        # Update turn text to include tags for reference
        turn.text_with_tags = new_text

        # Update turn sound effects
        turn.sound_effects = []
        for tag, position, start_time in valid_tag_infos:
            turn.sound_effects.append({
                "tag": tag,
                "position": position,
                "start_time": start_time,
                "duration": available_sound_effects[tag].get("duration", "unknown"),
                "position": position,
            })

    def compute_overlapping_and_pausing_llm(self, verbose: bool = False):
        """
        Compute the overlapping and pausing between turns using LLM.

        This method computes the overlapping and pausing times between turns using LLM.
        It will return a new AudioDialog object with the overlapping or pausing times for each turn.

        :param verbose: Verbose mode for logging.
        :type verbose: bool
        :return: A new AudioDialog object with the overlapping or pausing times for each turn.
        :rtype: AudioDialog
        """
        llm_params = config["llm"].copy()
        model_name = llm_params.pop("model", None)

        if not model_name:
            raise ValueError("LLM model not configured. Please set sdialog.config.llm('provider:model').")

        # Define schema for structured output
        class GapDurations(BaseModel):
            gaps: List[float] = Field(
                description="List of time intervals (seconds) between turns. "
                            "Positive for pause, negative for overlap."
            )

        # Prepare prompts
        dialog_text = ""
        for i, turn in enumerate(self.turns):
            # dialog_text += f"Turn {i+1} ({turn.speaker}): {turn.text}\n"
            dialog_text += f"Turn {i+1} ({turn.speaker} / {turn.audio_duration} seconds): {turn.text}\n"

        system_prompt = (
            "Analyze the dialogue and determine the natural timing gaps between turns. "
            "For each transition between Turn N and Turn N+1, provide a time duration in seconds:\n"
            "- Positive value (e.g., 0.5): A pause or silence.\n"
            "- Negative value (e.g., -0.2): An overlap.\n"
            "- 0.0: Immediate follow-up."
        )

        # print(system_prompt)

        human_prompt = (
            f"Dialogue:\n{dialog_text}\n\n"
            f"Provide a list of {len(self.turns) - 1} float values representing the gaps between consecutive turns."
        )
        # print(human_prompt)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # Invoke LLM
        try:
            llm = get_llm_model(
                model_name=model_name,
                output_format=GapDurations,
                **llm_params
            )
            raw_response = llm.invoke(messages)
            structured_response = GapDurations.model_validate(raw_response)
            gaps = structured_response.gaps
        except Exception as e:
            logger.error(f"Failed to compute gaps with LLM: {e}")
            return self.clone()

        # Validate length
        expected_gaps = len(self.turns) - 1
        if len(gaps) != expected_gaps:
            logger.warning(
                f"LLM returned {len(gaps)} gaps, expected {expected_gaps}. "
                "Adjusting to match turn count."
            )
            # Pad or truncate
            if len(gaps) < expected_gaps:
                gaps.extend([0.0] * (expected_gaps - len(gaps)))
            else:
                gaps = gaps[:expected_gaps]

        if verbose:
            logger.info("-------------------------------- gaps --------------------------------")
            logger.info(gaps)

        # Apply gaps
        for i in range(len(gaps)):
            self.turns[i].gap_duration = gaps[i]
