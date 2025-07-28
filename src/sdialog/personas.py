"""
personas: Persona and Agent Definitions for Synthetic Dialogue Generation

This module provides classes for defining personas (character profiles) and simulating agents that role-play
these personas in synthetic dialogue generation. Agents interact using LLMs and can be orchestrated for
complex behaviors.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Séverin Baroudi <severin.baroudi@lis-lab.fr>
# SPDX-License-Identifier: MIT
import os
import sys
import json
import torch
import random
import logging
import inspect

from abc import ABC
from time import time
from tqdm.auto import tqdm
from collections import defaultdict
from pydantic import BaseModel, Field
from print_color import print as cprint
from typing import List, Union, Optional

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages.base import messages_to_dict
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import config
from jinja2 import Template
from .orchestrators import BaseOrchestrator
from . import Dialog, Turn, Event, Instruction, _get_dynamic_version
from .interpretability import UtteranceTokenHook, RepresentationHook, Inspector
from .util import get_llm_model, camel_or_snake_to_words, get_timestamp, remove_newlines, get_universal_id
from .util import is_aws_model_name, is_huggingface_model_name


logger = logging.getLogger(__name__)


class PersonaMetadata(BaseModel):
    """
    Wrapper class for persona objects with additional metadata.

    :ivar version: Version of the persona format (matches sdialog version).
    :vartype version: Optional[str]
    :ivar timestamp: Timestamp of when the persona was generated.
    :vartype timestamp: Optional[str]
    :ivar model: The model used to generate the persona.
    :vartype model: Optional[str]
    :ivar seed: The random seed used for persona generation.
    :vartype seed: Optional[int]
    :ivar id: Unique identifier for the persona.
    :vartype id: Optional[int]
    :ivar parentId: ID of the parent persona, if any.
    :vartype parentId: Optional[int]
    :ivar notes: Free-text notes or comments about the generated persona.
    :vartype notes: Optional[str]
    :ivar className: The class name of the persona (a subclass of BasePersona).
    :vartype className: str
    """
    version: Optional[str] = Field(default_factory=_get_dynamic_version)
    timestamp: Optional[str] = Field(default_factory=get_timestamp)
    model: Optional[str] = None
    seed: Optional[int] = None
    id: Optional[Union[int, str]] = Field(default_factory=get_universal_id)
    parentId: Optional[Union[int, str]] = None
    className: str = None
    notes: Optional[str] = None


class BasePersona(BaseModel, ABC):
    """
    Base class for defining a persona (character profile) for role-play.

    :param kwargs: Arbitrary keyword arguments are stored as persona attributes.
    """
    _metadata: Optional[PersonaMetadata] = None

    def clone(self, new_id: int = None, **kwargs) -> "BasePersona":
        """
        Creates a deep copy of the persona, with optional attribute overrides.

        The cloned persona will have all attributes copied from the original, with any provided keyword arguments
        (`kwargs`) used to override or update specific fields. The clone receives a new metadata object:

        - The `parentId` field in the clone's metadata is set to the original persona's `id` (if present).
        - The `id` field in the clone's metadata is set to `new_id` if provided, otherwise to the original's `id`.
        - All other metadata fields are copied from the original.

        This method is useful for generating variations of a persona for ablation, branching, or scenario testing
        without modifying the original instance. The clone is a fully independent object.

        :param new_id: Optional new unique ID for the cloned persona.
        :type new_id: int, optional
        :param kwargs: Attributes to override in the cloned persona.
        :return: A new instance of the persona with updated attributes and metadata.
        :rtype: BasePersona
        """
        data = self.json()
        data.update(kwargs)
        new_persona = self.__class__(**data)
        if self._metadata:
            new_persona._metadata = self._metadata.model_copy()
            new_persona._metadata.parentId = self._metadata.id if self._metadata.id else None
            new_persona._metadata.id = new_id if new_id is not None else get_universal_id()
        else:
            new_persona._metadata = PersonaMetadata(className=self.__class__.__name__,
                                                    id=new_id if new_id is not None else get_universal_id(),
                                                    parentId=self._metadata.id if self._metadata else None)
        return new_persona

    def description(self) -> str:
        """
        Returns a string description of the persona's attributes.

        :return: Description of the persona.
        :rtype: str
        """
        return "\n".join(f"* {camel_or_snake_to_words(key).capitalize()}: {value}"
                         for key, value in self.__dict__.items()
                         if value not in [None, ""])

    def __str__(self) -> str:
        """
        Returns the string representation of the persona.

        :return: Description of the persona.
        :rtype: str
        """
        return self.description()

    def print(self, *a, **kw):
        """
        Pretty-prints the persona, including its metadata information.
        """
        if hasattr(self, "_metadata") and self._metadata is not None:
            for key, value in self._metadata.model_dump().items():
                if value not in [None, ""]:
                    cprint(remove_newlines(value), tag=key, tag_color="purple", color="magenta", format="bold")
        cprint("--- Persona Begins ---", color="magenta", format="bold")
        for key, value in self.__dict__.items():
            if key == "_metadata":
                continue
            cprint(remove_newlines(value),
                   tag=camel_or_snake_to_words(key).capitalize(),
                   tag_color="red",
                   color="white")
        cprint("--- Persona Ends ---", color="magenta", format="bold")

    def json(self, string: bool = False, indent=2, output_metadata: bool = True):
        """
        Serializes the persona to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :param output_metadata: Include the metadata in the serialization.
        :type output_metadata: bool
        :return: The serialized persona.
        :rtype: Union[str, dict]
        """
        data = {key: value for key, value in self.__dict__.items() if value not in [None, ""]}
        if self._metadata and output_metadata:
            data["_metadata"] = self._metadata.model_dump()
        return json.dumps(data, indent=indent) if string else data

    def prompt(self) -> str:
        """
        Returns the textual representation of the persona, used as part of the system prompt.
        """
        return self.json(string=True, output_metadata=False)

    def to_file(self, path: str, makedir: bool = True):
        """
        Saves the persona to a file in either JSON or plain text format.

        :param path: Output file path.
        :type path: str
        :param makedir: If True, creates parent directories as needed.
        :type makedir: bool
        """
        if makedir and os.path.split(path)[0]:
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        if self._metadata is None:
            self._metadata = PersonaMetadata(className=self.__class__.__name__)

        with open(path, "w") as writer:
            writer.write(self.json(string=True))

    @staticmethod
    def from_file(path: str, persona_class: Optional["BasePersona"] = None):
        """
        Loads persona from a file.

        :param path: Path to the persona file.
        :type path: str
        :param persona_class: Optional specific class to use for the persona.
        :type persona_class: Optional[BasePersona]
        :return: The loaded persona object.
        :rtype: MetaPersona
        """
        return BasePersona.from_json(open(path, "r", encoding="utf-8").read(), persona_class)

    @staticmethod
    def from_dict(data: dict, persona_class: Optional["BasePersona"] = None):
        """
        Creates a persona object from a dictionary.

        :param data: The dictionary containing persona data.
        :type data: dict
        :param persona_class: Optional specific class to use for the persona.
        :type persona_class: Optional[BasePersona]
        :return: The created persona object.
        :rtype: MetaPersona
        """
        # Assign to "persona" the instance of the right class using the `className`
        if "_metadata" in data and "className" in data["_metadata"] and data["_metadata"]["className"]:
            persona_class_name = data["_metadata"]["className"]
            if persona_class and issubclass(persona_class, BasePersona):
                # If the user provided a specific class, use it
                persona = persona_class.model_validate(data)
                persona._metadata = PersonaMetadata(**data["_metadata"])
                return persona
            else:  # Assuming the class name is from one of the built-in classes
                # Automatically get all classes in the module that inherit from BasePersona
                current_module = sys.modules[__name__]
                persona_class_map = {
                    cls.__name__: cls
                    for _, cls in inspect.getmembers(current_module, inspect.isclass)
                    if issubclass(cls, BasePersona) and cls is not BasePersona
                }
                persona_class = persona_class_map.get(persona_class_name)
                if persona_class:
                    persona = persona_class.model_validate(data)
                    persona._metadata = PersonaMetadata(**data["_metadata"])
                    return persona
                else:
                    raise ValueError(f"Unknown persona class given in the `className` field: {persona_class_name}.")
        else:
            raise ValueError("Metadata with `className` is required to create a persona from a dict or json.")

    @staticmethod
    def from_json(json_str: str, persona_class: Optional["BasePersona"] = None):
        """
        Creates a persona object from a JSON string.

        :param json_str: The JSON string containing persona data.
        :type json_str: str
        :param persona_class: Optional specific class to use for the persona.
        :type persona_class: Optional[BasePersona]
        :return: The created persona object.
        :rtype: MetaPersona
        """
        return BasePersona.from_dict(json.loads(json_str), persona_class)


class Persona(BasePersona):
    """
    Standard persona class with common attributes for role-play.

    :ivar name: Name of the persona.
    :vartype name: str
    :ivar age: Age of the persona.
    :vartype age: int
    :ivar race: Race of the persona.
    :vartype race: str
    :ivar gender: Gender of the persona.
    :vartype gender: str
    :ivar language: Preferred language.
    :vartype language: str
    :ivar role: Role or occupation.
    :vartype role: str
    :ivar background: Background information.
    :vartype background: str
    :ivar personality: Personality traits.
    :vartype personality: str
    :ivar circumstances: Current circumstances.
    :vartype circumstances: str
    :ivar rules: Rules or constraints.
    :vartype rules: str
    """

    name: str = ""
    age: Union[int, str] = None
    race: str = ""
    gender: str = ""
    language: str = "English"
    role: str = ""
    background: str = ""
    personality: str = ""
    circumstances: str = ""
    rules: str = ""


class ExtendedPersona(BasePersona):
    """
    Extended persona class with additional demographic, personality, and background attributes.

    :ivar name: Name of the persona.
    :vartype name: str
    :ivar age: Age of the persona.
    :vartype age: int
    :ivar race: Race of the persona.
    :vartype race: str
    :ivar gender: Gender of the persona.
    :vartype gender: str
    :ivar language: Preferred language.
    :vartype language: str
    :ivar weight: Weight of the persona.
    :vartype weight: str
    :ivar height: Height of the persona.
    :vartype height: str
    :ivar occupation: Occupation of the persona.
    :vartype occupation: str
    :ivar education: Education background.
    :vartype education: str
    :ivar socioeconomic_status: Socioeconomic status.
    :vartype socioeconomic_status: str
    :ivar interests: Interests of the persona.
    :vartype interests: str
    :ivar hobbies: Hobbies of the persona.
    :vartype hobbies: str
    :ivar politeness: Politeness trait.
    :vartype politeness: str
    :ivar forgetfulness: Forgetfulness trait.
    :vartype forgetfulness: str
    :ivar attentiveness: Attentiveness trait.
    :vartype attentiveness: str
    :ivar communication_style: Communication style.
    :vartype communication_style: str
    :ivar empathy_level: Empathy level.
    :vartype empathy_level: str
    :ivar political_views: Political views (e.g., conservative, liberal, moderate, etc.).
    :vartype political_views: str
    :ivar religious_beliefs: Religious beliefs (e.g., religious, agnostic, atheist, etc.).
    :vartype religious_beliefs: str
    """
    name: str = ""
    # Demographics
    age: Union[int, str] = ""
    race: str = ""
    gender: str = ""
    language: str = "English"
    weight: str = ""
    height: Union[str, float] = ""
    voice_characteristics: str = ""  # e.g., accent, tone, etc.
    # Background
    occupation: str = ""
    education: str = ""
    socioeconomic_status: str = ""
    # Interests and hobbies
    interests: str = ""
    hobbies: str = ""
    # Personality traits
    politeness: str = ""
    forgetfulness: str = ""
    attentiveness: str = ""
    communication_style: str = ""
    empathy_level: str = ""
    # Political and social views
    political_views: str = ""  # conservative, liberal, not polital, moderate, other
    religious_beliefs: str = ""  # religious, agnostic, atheist, etc.


class Patient(ExtendedPersona):
    """
    Patient persona with medical and health-related attributes.

    :ivar reason_for_visit: Reason for visit or chief complaint.
    :vartype reason_for_visit: str
    :ivar symptoms: List of symptoms or health issues.:ivar symptoms: Reason for visit or chief complaint.
    :vartype symptoms: str
    :ivar vital_signs: Vital signs of the patient.
    :vartype vital_signs: str
    :ivar health_literacy: Health literacy level.
    :vartype health_literacy: str
    :ivar medical_conditions: Medical conditions in history.
    :vartype medical_conditions: str
    :ivar medications: Current medications.
    :vartype medications: str
    :ivar allergies: Known allergies.
    :vartype allergies: str
    :ivar family_history: Family medical history.
    :vartype family_history: str
    """
    reason_for_visit: str = ""
    symptoms: str = ""
    vital_signs: str = ""
    health_literacy: str = ""
    medical_conditions: str = ""
    medications: str = ""
    allergies: str = ""
    family_history: str = ""


class MinimalPatient(BasePersona):
    """
    Minimal version of a Patient persona, focusing on essential attributes for dialogue generation.

    :ivar name: Name of the persona.
    :vartype name: str
    :ivar age: Age of the persona.
    :vartype age: int
    :ivar race: Race of the persona.
    :vartype race: str
    :ivar gender: Gender of the persona.
    :vartype gender: str
    :ivar language: Preferred language.
    :vartype language: str
    :ivar reason_for_visit: Reason for visit or chief complaint.
    :vartype reason_for_visit: str
    :ivar medical_history: Medical history of the patient.
    :vartype medical_history: str
    :ivar medical_conditions: Medical conditions in history.
    :vartype medical_conditions: str
    :ivar medications: Current medications.
    :vartype medications: str
    :ivar allergies: Known allergies.
    :vartype allergies: str
    :ivar family_history: Family medical history.
    :vartype family_history: str
    """
    name: str = ""
    age: Union[int, str] = None
    race: str = ""
    gender: str = ""
    language: str = "English"
    forgetfulness: Union[str, float] = ""
    formality: Union[str, float] = ""
    hurriedness: Union[str, float] = ""
    openness: Union[str, float] = ""
    height: Union[int, str] = ""
    weight: Union[int, str] = ""
    occupation: str = ""
    marital_status: str = ""
    insurance: str = ""
    reason_for_visit: str = ""
    medical_history: Union[str, List[str]] = ""
    medical_conditions: Union[str, List[str]] = ""
    medications_current: Union[str, List[str]] = ""
    allergies: Union[str, List[str]] = ""
    family_history: Union[str, List[str]] = ""


class Doctor(ExtendedPersona):
    """
    Doctor persona with medical expertise and professional background.

    :ivar specialty: Medical specialty.
    :vartype specialty: str
    :ivar years_of_experience: Years of experience as a doctor.
    :vartype years_of_experience: int
    :ivar certifications: Certifications held by the doctor.
    :vartype certifications: str
    :ivar work_experience: Professional work experience.
    :vartype work_experience: str
    """
    specialty: str = ""
    years_of_experience: Union[int, str] = ""
    certifications: str = ""
    work_experience: str = ""


class MinimalDoctor(BasePersona):
    """
    This class is a minimal version of a Doctor persona, focusing on essential attributes for dialogue generation.

    :ivar name: Name of the persona.
    :vartype name: str
    :ivar age: Age of the persona.
    :vartype age: int
    :ivar race: Race of the persona.
    :vartype race: str
    :ivar gender: Gender of the persona.
    :vartype gender: str
    :ivar language: Preferred language.
    :vartype language: str
    :ivar years_of_experience: Years of experience as a doctor.
    :vartype years_of_experience: Union[int, str]
    :ivar speciality: Medical specialty.
    :vartype speciality: str
    :ivar forgetfulness: Forgetfulness trait.
    :vartype forgetfulness: str
    :ivar formality: Formality trait.
    :vartype formality: str
    :ivar hurriedness: Hurriedness trait.
    :vartype hurriedness: str
    :ivar openness: Openness trait.
    :vartype openness: str
    """
    name: str = ""
    age: Union[int, str] = ""
    race: str = ""
    gender: str = ""
    language: str = "English"
    years_of_experience: Union[int, str] = ""
    speciality: str = ""
    forgetfulness: str = ""
    formality: str = ""
    hurriedness: str = ""
    openness: str = ""


class Agent:
    """
    Agent that simulates a persona in dialogue using an LLM.

    :cvar STOP_WORD: Special token to indicate end of conversation.
    :vartype STOP_WORD: str
    :cvar STOP_WORD_TEXT: Replacement text for STOP_WORD.
    :vartype STOP_WORD_TEXT: str
    """

    STOP_WORD = "STOP"
    STOP_WORD_TEXT = "(bye bye!)"

    def __init__(self,
                 persona: BasePersona,
                 name: Optional[str] = None,
                 model: Union[str, BaseLanguageModel] = None,
                 example_dialogs: Optional[List['Dialog']] = None,
                 dialogue_details: str = "",
                 response_details: str = "Unless necessary, responses SHOULD NOT be longer than one utterances.",
                 system_prompt: Optional[str] = None,
                 can_finish: bool = True,
                 orchestrators: Optional[Union[BaseOrchestrator, List[BaseOrchestrator]]] = None,
                 inspectors: Optional[Union['Inspector', List['Inspector']]] = None,
                 scenario: Optional[Union[dict, str]] = None,
                 postprocess_fn: Optional[callable] = None,
                 **llm_kwargs):
        """
        Initializes a PersonaAgent for role-play dialogue.

        :param persona: The persona to role-play.
        :type persona: BasePersona
        :param name: Name of the agent (defaults to persona.name if not provided).
        :type name: Optional[str]
        :param model: The LLM or model name to use (defaults to config["llm"]["model"]).
        :type model: Union[str, BaseLanguageModel], optional
        :param example_dialogs: List of example dialogues as a reference for the agent.
        :type example_dialogs: Optional[List[Dialog]]
        :param dialogue_details: Additional details about the dialogue.
        :type dialogue_details: str
        :param response_details: Instructions for response style.
        :type response_details: str
        :param system_prompt: Custom system prompt (optional, otherwise loaded from config).
        :type system_prompt: Optional[str]
        :param can_finish: If True, agent can end the conversation.
        :type can_finish: bool
        :param orchestrators: Orchestrators for agent behavior.
        :type orchestrators: Optional[Union[BaseOrchestrator, List[BaseOrchestrator]]]
        :param inspectors: Inspector(s) to add to the agent.
        :type inspectors: Optional[Union[Inspector, List[Inspector]]]
        :param scenario: Scenario metadata.
        :type scenario: Optional[Union[dict, str]]
        :param postprocess_fn: Optional function to postprocess each utterance (input string, output string).
        :type postprocess_fn: Optional[callable]
        :param **llm_kwargs: Additional parameters for the LLM.
        :type llm_kwargs: dict
        """
        if model is None:
            model = config["llm"]["model"]
        self.model_uri = model

        if postprocess_fn is not None and not callable(postprocess_fn):
            raise ValueError("postprocess_fn must be a callable function that takes a string and outputs a string.")

        if not system_prompt:
            with open(config["prompts"]["persona_agent"], encoding="utf-8") as f:
                system_prompt_template = Template(f.read())
            system_prompt = system_prompt_template.render(
                persona=persona.prompt(),
                example_dialogs=example_dialogs,
                dialogue_details=dialogue_details,
                response_details=response_details,
                can_finish=can_finish,
                stop_word=self.STOP_WORD
            )

        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}
        self.llm = get_llm_model(model_name=model, **llm_kwargs)

        self.memory = [SystemMessage(system_prompt)]

        self.name = name if name is not None else getattr(persona, "name", None)
        self.persona = persona
        self.model_name = str(self.llm)
        self.first_utterances = None
        self.finished = False
        self.scenario = scenario
        self.orchestrators = None
        self.add_orchestrators(orchestrators)
        self.inspectors = None
        self.add_inspectors(inspectors)
        self.postprocess_fn = postprocess_fn

        logger.debug(f"Initialized agent '{self.name}' with model '{self.model_name}' "
                     f"using prompt from '{config['prompts']['persona_agent']}'.")
        logger.debug("Prompt: " + self.prompt())

    @property
    def utterance_list(self):
        return self.utterance_hook.utterance_list

    def __call__(self, utterance: str = "", return_events: bool = False) -> str:
        """
        Processes an input utterance and generates a response.

        :param utterance: The input utterance from the other agent or user.
        :type utterance: str
        :param return_events: If True, returns a list of events instead of just the response string.
        :type return_events: bool
        :return: The agent's response or events, or None if finished.
        :rtype: Union[str, List[Event], None]
        """
        if self.finished:
            return None

        if utterance:
            self.memory.append(HumanMessage(content=utterance))

        if return_events:
            events = []
        if self.orchestrators:
            for orchestrator in self.orchestrators:
                instruction = orchestrator()
                if instruction:

                    if type(instruction) is Instruction:
                        if return_events and instruction.events:
                            if type(instruction.events) is Event:
                                events.append(instruction.events)
                            else:
                                events.extend(instruction.events)
                        instruction = instruction.text

                    persist = orchestrator.is_persistent()
                    self.instruct(instruction, persist=persist)
                    if return_events:
                        events.append(Event(agent=self.get_name(),
                                            action="instruct" + ("-persist" if persist else ""),
                                            actionLabel=orchestrator.get_event_label(),
                                            text=instruction,
                                            timestamp=int(time())))

        if len(self.memory) <= 1 and self.first_utterances:
            response = (random.choice(self.first_utterances)
                        if type(self.first_utterances) is list
                        else self.first_utterances)
            response = AIMessage(content=response)
        else:
            if self.inspectors:
                self.utterance_hook.new_utterance_event(self.memory_dump())

            if (is_huggingface_model_name(self.model_uri) or is_aws_model_name(self.model_uri)) and \
               (not self.memory or not isinstance(self.memory[-1], HumanMessage)):
                # Ensure that the last message is a HumanMessage to avoid
                # "A conversation must start with a user message" (aws)
                # or "Last message must be a HumanMessage!" (huggingface)
                # from langchain_huggingface (which makes no sense, for ollama is OK but for hugging face is not?)
                # https://github.com/langchain-ai/langchain/blob/6d71b6b6ee7433716a59e73c8e859737800a0a86/libs/partners/huggingface/langchain_huggingface/chat_models/huggingface.py#L726
                response = self.llm.invoke(self.memory + [HumanMessage(
                    content="" if is_huggingface_model_name(self.model_uri) else ".")
                ])
                logger.warning(
                    "For HuggingFace or AWS LLMs, the last message in the conversation history must be a HumanMessage. "
                    "A dummy HumanMessage was appended to memory to satisfy this requirement and prevent errors."
                )
            else:
                response = self.llm.invoke(self.memory)

            if self.inspectors:
                self.utterance_hook.end_utterance_event()

        if self.postprocess_fn:
            response.content = self.postprocess_fn(response.content)

        if self.orchestrators:
            self.memory[:] = [msg for msg in self.memory
                              if not (msg.response_metadata
                                      and "persist" in msg.response_metadata
                                      and not msg.response_metadata["persist"])]
        self.memory.append(response)

        response = response.content
        if self.STOP_WORD in response:
            response = response.replace(self.STOP_WORD, self.STOP_WORD_TEXT).strip()
            self.memory[-1].content = self.memory[-1].content.replace(self.STOP_WORD, "").strip()
            self.finished = True

        if return_events:
            if response:
                events.append(Event(agent=self.get_name(),
                                    action="utter",
                                    text=response,
                                    timestamp=int(time())))
            return events
        else:
            return response if response else ""

    def __or__(self, other):
        """
        Adds enitity to the agent using the | operator.

        :param orchestrator: Orchestrator(s) to add.
        :type orchestrator: Union[BaseOrchestrator, List[BaseOrchestrator]]
        :return: The agent with orchestrators added.
        :rtype: PersonaAgent
        """
        if isinstance(other, Inspector):
            self.set_utterance_hook()
            self.add_inspectors(other)
            other.add_agent(agent=self)
        else:
            self.add_orchestrators(other)
        return self

    def response_lookahead(self, utterance: str = None):
        """
        Generates a response to a hypothetical next utterance without updating memory.

        :param utterance: The hypothetical next utterance.
        :type utterance: str
        :return: The predicted response.
        :rtype: str
        """
        if not utterance:
            return self.llm.invoke(self.memory).content
        return self.llm.invoke(self.memory + [HumanMessage(utterance)]).content

    def add_orchestrators(self, orchestrators):
        """
        Adds orchestrators to the agent.

        :param orchestrators: Orchestrator(s) to add.
        :type orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]]
        """
        if not orchestrators:
            return

        if self.orchestrators is None:
            self.orchestrators = []

        if isinstance(orchestrators, BaseOrchestrator):
            orchestrators = [orchestrators]

        self.orchestrators.extend(orchestrators)

        for orchestrator in orchestrators:
            orchestrator._set_target_agent(self)

    def add_inspectors(self, inspectors):
        """
        Adds inspectors to the agent.

        :param inspectors: Inspector(s) to add.
        :type inspectors: Union[Inspector, List[Inspector]]
        """
        if inspectors is None:
            return

        if self.inspectors is None:
            self.inspectors = []

        # Handle both single Inspector and list of Inspectors
        if isinstance(inspectors, Inspector):
            inspectors = [inspectors]
        elif isinstance(inspectors, list):
            inspectors = [ins for ins in inspectors if ins is not None]
            if not inspectors:
                return
        else:
            raise TypeError("inspectors must be an Inspector or a list of Inspectors")

        self.inspectors.extend(inspectors)

        for inspector in inspectors:
            inspector.add_agent(self)
        self.set_utterance_hook()

    def clear_orchestrators(self):
        """
        Removes all orchestrators from the agent.
        """
        self.orchestrators = None

    def add_hooks(self, layer_name_to_key, steering_function=None, steering_interval=(0, -1)):
        """
        Registers RepresentationHooks for each layer in the given mapping.
        Skips already registered layers. Adds new keys to the shared representation_cache.

        Args:
            layer_name_to_key: Dict mapping layer names to cache keys.
            steering_function: Optional function to apply to the output tensor before caching.
            steering_interval: Tuple `(min_token, max_token)` to control steering.
                                   `min_token` tokens are skipped. Steering stops at `max_token`.
                                   A `max_token` of -1 means no upper limit.
        """
        # Get the model (assume HuggingFace pipeline)
        model = self.llm.llm.pipeline.model if hasattr(self.llm, 'llm') and hasattr(self.llm.llm, 'pipeline') else None
        if model is None:
            raise RuntimeError("Model not found or not a HuggingFace pipeline.")

        # Always re-initialize cache and hooks
        self.representation_cache = defaultdict(lambda: defaultdict(list))
        self.rep_hooks = []

        # Register new hooks
        for layer_name, cache_key in layer_name_to_key.items():
            hook = RepresentationHook(
                layer_key=layer_name,
                cache_key=cache_key,
                representation_cache=self.representation_cache,
                utterance_list=self.utterance_list,
                steering_function=steering_function,  # pass the function here,
                steering_interval=steering_interval
            )
            hook.register(model)
            self.rep_hooks.append(hook)

    def clear_hooks(self):
        """
        Resets all representation cached and removes all registered hooks from the agent.
        """
        for hook in getattr(self, 'rep_hooks', []):
            hook.reset()
            hook.remove()
        self.rep_hooks = []
        self.representation_cache = defaultdict(lambda: defaultdict(list))
        self.set_utterance_hook()

    def set_utterance_hook(self):
        # Register UtteranceTokenHook and expose utterance_list
        self.utterance_hook = UtteranceTokenHook(agent=self)
        model_obj = self.llm.llm.pipeline.model
        self.utterance_hook.register(model_obj)
        # Automatically set the tokenizer in the hook
        self.utterance_hook.hook_state['tokenizer'] = self.llm.llm.pipeline.tokenizer

    def instruct(self, instruction: str, persist: bool = False):
        """
        Adds a system instruction to the agent's memory.

        :param instruction: The instruction text.
        :type instruction: str
        :param persist: If True, instruction persists across turns.
        :type persist: bool
        """
        if isinstance(self.memory[-1], HumanMessage):
            # If the last message is a HumanMessage, insert the SystemMessage before it
            # (so the last message is still HumanMessage)
            self.memory.insert(-1, SystemMessage(instruction, response_metadata={"persist": persist}))
        else:
            self.memory.append(SystemMessage(instruction, response_metadata={"persist": persist}))

    def set_first_utterances(self, utterances: Union[str, List[str]]):
        """
        Sets the agent's first utterance(s) for dialogue initialization.

        :param utterances: The greeting(s) to use.
        :type utterances: Union[str, List[str]]
        """
        self.first_utterances = utterances

    def get_name(self, default: str = "Me") -> str:
        """
        Returns the agent's name.

        :return: The agent's name.
        :rtype: str
        """
        return self.name if self.name is not None else default

    def prompt(self) -> str:
        """
        Returns the current system prompt.

        :return: The system prompt.
        :rtype: str
        """
        return self.memory[0].content

    def json(self, string: bool = False, indent=None):
        """
        Serializes the agent's configuration and persona to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :return: The serialized agent.
        :rtype: Union[str, dict]
        """
        data = {}
        if self.name:
            data["name"] = self.get_name()
        data["model_name"] = self.model_name
        if self.first_utterances:
            data["first_utterances"] = self.first_utterances
        data["persona"] = self.persona.json()
        if self.orchestrators:
            data["persona"]["orchestrators"] = [orc.json() for orc in self.orchestrators]
        return json.dumps(data, indent=indent) if string else data

    def reset(self, seed: int = None):
        """
        Resets the agent's memory and orchestrators, optionally reseeding the LLM.
        Clears the interpretability state (utterance_list and representation_cache).

        :param seed: Random seed for reproducibility.
        :type seed: int
        """
        self.memory[:] = self.memory[:1]
        self.finished = False
        if hasattr(self.llm, "seed"):
            self.llm.seed = seed if seed is not None else random.getrandbits(32)
        else:
            logger.warning("The LLM does not support dynamically setting a seed.")

        if self.orchestrators:
            for orchestrator in self.orchestrators:
                orchestrator.reset()

        self._reset_interpretability_state()

        if isinstance(self.llm, ChatOllama):
            # hack to avoid seed bug in prompt cache in Ollama
            # (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
            _ = self.llm.num_predict
            self.llm.num_predict = 1
            self.llm.invoke(self.memory)
            self.llm.num_predict = _
        else:
            if seed is None:
                torch.manual_seed(13)
            else:
                torch.manual_seed(seed)

    def _reset_interpretability_state(self):
        """
        Clears the interpretability state: utterance_list and representation_cache, if present.
        Does not reset or remove hooks.
        """
        if hasattr(self, 'utterance_hook') and self.utterance_hook is not None:
            self.utterance_hook.utterance_list.clear()
        if hasattr(self, 'representation_cache') and self.representation_cache is not None:
            self.representation_cache.clear()
            self.representation_cache.update(defaultdict(lambda: defaultdict(list)))

    def dialog_with(self,
                    agent: "PersonaAgent",
                    max_turns: int = 200,
                    id: int = None,
                    parent_id: int = None,
                    seed: int = None,
                    notes: str = None,
                    keep_bar: bool = True):
        """
        Simulates a dialogue between this agent and another PersonaAgent.

        :param agent: The other agent to converse with.
        :type agent: PersonaAgent
        :param max_turns: Maximum number of dialogue turns.
        :type max_turns: int
        :param id: Dialogue ID.
        :type id: int
        :param parent_id: ID of the parent dialogue, if any.
        :type parent_id: int
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param notes: Optional notes to include in the dialogue.
        :type notes: str
        :param keep_bar: If True, keeps the progress bar visible.
        :type keep_bar: bool
        :return: The generated dialogue object.
        :rtype: Dialog
        """
        seed = seed if seed is not None else random.getrandbits(32)

        random.seed(seed)
        self.reset(seed)
        agent.reset(seed)

        dialog = []
        events = []

        utter = None
        completion = False
        pbar = tqdm(total=max_turns, desc="Dialogue", leave=keep_bar)
        while len(dialog) < max_turns:
            utt_events = self(utter, return_events=True)

            if utt_events and utt_events[-1].action == "utter":
                utter = utt_events[-1].text
                utt_events[-1].text = utter.replace(self.STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].text:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=self.get_name(),
                text=utt_events[-1].text
            ))
            events.extend(utt_events)
            pbar.update(1)

            utt_events = agent(utter, return_events=True)
            if utt_events and utt_events[-1].action == "utter":
                utter = utt_events[-1].text
                utt_events[-1].text = utter.replace(self.STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].text:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=agent.get_name(default="Other"),
                text=utt_events[-1].text
            ))
            events.extend(utt_events)
            pbar.update(1)

        pbar.close()

        if self.scenario:
            scenario = self.scenario
        else:
            scenario = {
                "agents": [
                    self.json(),
                    agent.json()
                ]
            }

        return Dialog(
            id=id if id is not None else get_universal_id(),
            parentId=parent_id,
            complete=completion,  # incomplete if ran out of iterations (reached max_iteration number)
            model=self.model_name,
            seed=seed,
            personas={
                self.get_name(): self.persona.json(),
                agent.get_name(default="Other"): agent.persona.json()},
            scenario=scenario,
            notes=notes,
            turns=dialog,
            events=events
        )

    def memory_dump(self, as_dict: bool = False) -> list:
        """
        Returns a copy of the agent's memory (list of messages).
        :return: A copy of the memory list.
        :rtype: list
        """
        return messages_to_dict(self.memory) if as_dict else self.memory.copy()

    talk_with = dialog_with


PersonaAgent = Agent
