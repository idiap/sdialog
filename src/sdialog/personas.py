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
import transformers

from abc import ABC
from time import time
from tqdm.auto import trange
from pydantic import BaseModel, Field
from typing import List, Union, Optional

from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from . import Dialog, Turn, Event, Instruction, _get_dynamic_version
from .orchestrators import BaseOrchestrator
from .util import camel_or_snake_to_words
from .config import config
from jinja2 import Template


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class PersonaMetadata(BaseModel):
    """
    Wrapper class for persona objects with additional metadata.

    :ivar version: Version of the persona format (matches sdialog version).
    :vartype version: Optional[str]
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
    model: Optional[str] = None
    seed: Optional[int] = None
    id: Optional[int] = None
    parentId: Optional[int] = None
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
            new_persona._metadata.id = new_id if new_id is not None else self._metadata.id
        else:
            new_persona._metadata = PersonaMetadata(className=self.__class__.__name__,
                                                    id=new_id if new_id is not None else None,
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

    def json(self, string: bool = False, indent=2):
        """
        Serializes the persona to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :return: The serialized persona.
        :rtype: Union[str, dict]
        """
        data = {key: value for key, value in self.__dict__.items() if value not in [None, ""]}
        if self._metadata:
            data["_metadata"] = self._metadata.model_dump()
        return json.dumps(data, indent=indent) if string else data

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
    age: Optional[Union[int, str]] = None
    race: str = ""
    gender: str = ""
    language: str = "English"
    role: str = ""
    background: str = ""
    personality: str = ""
    circumstances: str = ""
    rules: str = ""

    @staticmethod
    def from_file(path: str, persona_class: Optional["BasePersona"] = None):
        """
        Loads a persona from a file.

        :param path: Path to the persona file.
        :type path: str
        :param persona_class: Optional specific class to use for the persona.
        :type persona_class: Optional[BasePersona]
        """
        return BasePersona.from_file(path, persona_class)

    @staticmethod
    def from_json(json_str: str, persona_class: Optional["BasePersona"] = None):
        """
        Creates a persona object from a JSON string.

        :param json_str: The JSON string containing persona data.
        :type json_str: str
        :param persona_class: Optional specific class to use for the persona.
        :type persona_class: Optional[BasePersona]
        """
        return BasePersona.from_json(json_str, persona_class)


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
    age: Optional[Union[int, str]] = None
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

    :ivar symptoms: Reason for visit or chief complaint.
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
    symptoms: str = ""
    vital_signs: str = ""
    health_literacy: str = ""
    medical_conditions: str = ""
    medications: str = ""
    allergies: str = ""
    family_history: str = ""


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
    years_of_experience: Optional[Union[int, str]] = None
    certifications: str = ""
    work_experience: str = ""


class PersonaAgent:
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
                 name: str = None,
                 model: Union[str, ChatOllama] = None,
                 dialogue_details: str = "",
                 response_details: str = "responses SHOULD NOT be too long and wordy, should be "
                                         "approximately one utterance long",
                 system_prompt: str = None,
                 can_finish: bool = True,
                 orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]] = None,
                 scenario: Union[dict, str] = None,
                 llm_kwargs: dict = None):

        """
        Initializes a PersonaAgent for role-play dialogue.

        :param persona: The persona to role-play.
        :type persona: BasePersona
        :param model: The LLM or model name to use.
        :type model: Union[str, ChatOllama]
        :param name: Name of the agent.
        :type name: str
        :param dialogue_details: Additional details about the dialogue.
        :type dialogue_details: str
        :param response_details: Instructions for response style.
        :type response_details: str
        :param system_prompt: Custom system prompt (optional).
        :type system_prompt: str
        :param can_finish: If True, agent can end the conversation.
        :type can_finish: bool
        :param orchestrators: Orchestrators for agent behavior.
        :type orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]]
        :param scenario: Scenario metadata.
        :type scenario: Union[dict, str]
        :param llm_kwargs: Additional parameters for the LLM.
        :type llm_kwargs: dict
        """

        if model is None:
            model = config["llm"]["model"]

        if not system_prompt:
            with open(config["prompts"]["persona_agent"], encoding="utf-8") as f:
                system_prompt_template = Template(f.read())
            system_prompt = system_prompt_template.render(
                persona=persona,
                dialogue_details=dialogue_details,
                response_details=response_details,
                can_finish=can_finish,
                stop_word=self.STOP_WORD
            )

        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = llm_kwargs or llm_config_params
        self.hf_model = False
        if isinstance(model, str):
            # If model name has a slash, assume it's a Hugging Face model
            # Otherwise, assume it's an Ollama model
            if "/" in model:
                logging.info(f"Loading Hugging Face model: {model}")
                self.hf_model = True

                # Default HuggingFace parameters
                hf_defaults = dict(
                    model=model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=2048,
                    do_sample=True,
                    repetition_penalty=1.03,
                    return_full_text=False,
                )
                hf_params = {**hf_defaults, **llm_kwargs}

                pipe = transformers.pipeline("text-generation", **hf_params)
                pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
                # TODO: if tokenizer doesn't have a chat template, set a default one

                self.llm = ChatHuggingFace(
                    llm=HuggingFacePipeline(pipeline=pipe,
                                            model_kwargs={'temperature': hf_params.get("temperature", 0.3)})
                )
            else:
                logging.info(f"Loading ChatOllama model: {model}")
                # Collect LLM parameters from config, only if not None
                # llm_kwargs overrides config
                self.llm = ChatOllama(model=model, **llm_kwargs)
        else:
            # Assume model is already an instance
            self.llm = model
            self.hf_model = isinstance(model, ChatHuggingFace)

        self.memory = [SystemMessage(system_prompt)]

        self.name = name if name else (persona.name if hasattr(persona, "name") else None)
        self.persona = persona
        self.model_name = str(self.llm)
        self.first_utterances = None
        self.finished = False
        self.scenario = scenario
        self.orchestrators = None
        self.add_orchestrators(orchestrators)

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
            if self.hf_model and not isinstance(self.memory[-1], HumanMessage):
                # Ensure last message is HumanMessage to avoid "Last message must be a HumanMessage!"
                # from langchain_huggingface (which makes no sense, for ollama is OK but for hugging face is not?)
                # https://github.com/langchain-ai/langchain/blob/6d71b6b6ee7433716a59e73c8e859737800a0a86/libs/partners/huggingface/langchain_huggingface/chat_models/huggingface.py#L726
                response = self.llm.invoke(self.memory + [HumanMessage(content="")])
            else:
                response = self.llm.invoke(self.memory)

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

    def __or__(self, orchestrator: Union[BaseOrchestrator, List[BaseOrchestrator]]):
        """
        Adds orchestrators to the agent using the | operator.

        :param orchestrator: Orchestrator(s) to add.
        :type orchestrator: Union[BaseOrchestrator, List[BaseOrchestrator]]
        :return: The agent with orchestrators added.
        :rtype: PersonaAgent
        """
        self.add_orchestrators(orchestrator)
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

    def clear_orchestrators(self):
        """
        Removes all orchestrators from the agent.
        """
        self.orchestrators = None

    def instruct(self, instruction: str, persist: bool = False):
        """
        Adds a system instruction to the agent's memory.

        :param instruction: The instruction text.
        :type instruction: str
        :param persist: If True, instruction persists across turns.
        :type persist: bool
        """
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

    def get_prompt(self) -> str:
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

        :param seed: Random seed for reproducibility.
        :type seed: int
        """
        self.memory[:] = self.memory[:1]
        self.finished = False
        self.llm.seed = seed

        if self.orchestrators:
            for orchestrator in self.orchestrators:
                orchestrator.reset()

        if not self.hf_model:
            # hack to avoid seed bug in prompt cache
            # (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
            _ = self.llm.num_predict
            self.llm.num_predict = 1
            self.llm.invoke(self.memory)
            self.llm.num_predict = _

    def dialog_with(self,
                    agent: "PersonaAgent",
                    max_turns: int = 80,
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
        tqdm_iterator = trange(max_turns // 2, desc="Dialogue", leave=keep_bar)
        for _ in tqdm_iterator:
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

        if not keep_bar:
            try:
                tqdm_iterator.container.close()
            except AttributeError:
                pass

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
            id=id if id else None,
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

    talk_with = dialog_with
