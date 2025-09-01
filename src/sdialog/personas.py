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
import logging
import inspect

from abc import ABC
from pydantic import BaseModel, Field, ConfigDict
from print_color import print as cprint
from typing import List, Union, Optional

from . import _get_dynamic_version
from .util import camel_or_snake_to_words, get_timestamp, remove_newlines, get_universal_id


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
    """
    model_config = ConfigDict(extra='forbid')
    _metadata: Optional[PersonaMetadata] = None

    # Automatically inject a staticmethod attributes() into every subclass
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def _attributes(_cls=cls):
            return [k for k in _cls.model_fields.keys() if not k.startswith("_")]
        cls.attributes = staticmethod(_attributes)

    @staticmethod
    def attributes():
        """
        Return the list of attribute (field) names for BasePersona itself.
        Subclasses get their own version injected in __init_subclass__.
        """
        return [k for k in BasePersona.model_fields.keys() if not k.startswith("_")]

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
        if "_metadata" in data:
            del data["_metadata"]  # to avoid model validation issues
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
            metadata = PersonaMetadata(**data["_metadata"])
            del data["_metadata"]  # to avoid model_validate(data) issues
            if persona_class and issubclass(persona_class, BasePersona):
                # If the user provided a specific class, use it
                persona = persona_class.model_validate(data)
                persona._metadata = metadata
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
                    persona._metadata = metadata
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


class Patient(BasePersona):
    """
    Patient persona with essential / minimal attributes for dialogue generation.

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


class ExtendedPatient(ExtendedPersona):
    """
    ExtendedPatient persona with medical and health-related attributes.

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


class Doctor(BasePersona):
    """
    Doctor persona with essential / minimal attributes for dialogue generation.

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


class ExtendedDoctor(ExtendedPersona):
    """
    ExtendedDoctor persona with medical expertise and professional background.

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


class Customer(BasePersona):
    """
    Persona for a customer in a customer service interaction.

    Identification / profile:
    :ivar name: Customer name.
    :ivar age: Customer age.
    :ivar gender: Customer gender.
    :ivar language: Preferred language.
    :ivar customer_id: Internal customer identifier.
    :ivar occupation: Customer occupation.

    Loyalty / tenure:
    :ivar account_tenure: How long (e.g., "2 years", "3 months") they have been a customer.
    :ivar membership_level: Plan/tier (e.g., basic, premium, enterprise).
    :ivar loyalty_status: Qualitative loyalty tier (loyal, at-risk, churn-risk, advocate).
    :ivar fidelity_score: Numerical or descriptive fidelity/loyalty score.

    Issue context:
    :ivar issue: Short summary of the current problem (concise).
    :ivar issue_category: High-level category (billing, technical, shipping, cancellation, feedback).
    :ivar issue_description: Free-text detailed description of the issue.
    :vartype issue_description: str
    :ivar issue_history: Brief summary of prior related issues.
    :ivar desired_outcome: What the customer wants resolved.

    Knowledge / competence:
    :ivar knowledge_domain: Domain knowledge area or level (novice, intermediate, expert, billing-savvy, etc.).
    :ivar technical_expertise: (Deprecated overlap) Original field kept for backward compatibility.

    Emotional / behavioral state:
    :ivar sentiment: Overall emotional tone (angry, frustrated, neutral, positive, confused).
    :ivar anger_level: Specific anger intensity (low, medium, high, escalating, etc.).
    :ivar tiredness: Fatigue level affecting patience/comprehension.
    :ivar patience_level: Patience (low, medium, high) or numeric score.
    :ivar politeness: Politeness style (polite, curt, rude, formal, casual).
    :ivar personality: Brief personality descriptor (assertive, analytical, anxious, etc.).
    :ivar instruction_following: Tendency to follow step-by-step instructions (low, medium, high).
    :ivar forgetfulness: Propensity to forget prior guidance (low, medium, high).

    Contact / interaction history:
    :ivar times_called: Number of prior contacts about this (or related) issue.
    :ivar preferred_channel: Preferred support channel (chat, phone, email, portal).
    :ivar prior_interactions_summary: Compressed summary of prior support contacts.

    Meta:
    :ivar urgency: Perceived urgency (low, medium, high, critical).
    :ivar rules: Constraints or special handling notes.
    """
    # Identification / profile
    name: str = ""
    age: Union[int, str] = ""
    gender: str = ""
    language: str = "English"
    customer_id: str = ""
    occupation: str = ""

    # Loyalty / tenure
    account_tenure: str = ""
    membership_level: str = ""
    loyalty_status: str = ""
    fidelity_score: Union[str, float, int] = ""

    # Issue context
    issue: str = ""
    issue_category: str = ""
    issue_description: str = ""
    issue_history: str = ""
    desired_outcome: str = ""

    # Knowledge / competence
    knowledge_domain: str = ""
    technical_expertise: str = ""  # kept for backward compatibility

    # Emotional / behavioral state
    sentiment: str = ""
    anger_level: str = ""
    tiredness: str = ""
    patience_level: str = ""
    politeness: str = ""
    personality: str = ""
    instruction_following: str = ""
    forgetfulness: str = ""

    # Contact / interaction history
    times_called: Union[int, str] = ""
    preferred_channel: str = ""
    prior_interactions_summary: str = ""

    # Meta
    urgency: str = ""
    rules: str = ""


class SupportAgent(BasePersona):
    """
    Persona for a customer service / support agent.

    :ivar name: Agent name.
    :ivar language: Working language.
    :ivar agent_id: Internal agent identifier.
    :ivar role: Role or queue (e.g., frontline, technical support, retention).
    :ivar experience_years: Years (or range) of support experience.
    :ivar product_scope: Products / domains the agent covers.
    :ivar product_knowledge_level: Knowledge depth (basic, proficient, expert).
    :ivar communication_style: Style (formal, friendly, concise, empathetic).
    :ivar empathy_level: Empathy (low, medium, high) or descriptive phrase.
    :ivar politeness: Politeness level descriptor.
    :ivar resolution_authority_level: Authority to refund / replace / escalate (e.g., low, medium, high).
    :ivar escalation_policy: Short description of escalation triggers.
    :ivar average_handle_time: Typical AHT (e.g., "6m", "300s").
    :ivar adherence_notes: Notes on process adherence or QA.
    :ivar stress_tolerance: Stress handling descriptor.
    :ivar performance_notes: KPIs summary.
    :ivar rules: Internal rules / constraints / compliance reminders.
    """
    name: str = ""
    language: str = "English"
    agent_id: str = ""
    role: str = "Customer Support Agent"
    experience_years: str = ""
    product_scope: str = ""
    product_knowledge_level: str = ""
    communication_style: str = ""
    empathy_level: str = ""
    politeness: str = ""
    resolution_authority_level: str = ""
    escalation_policy: str = ""
    average_handle_time: str = ""
    adherence_notes: str = ""
    stress_tolerance: str = ""
    performance_notes: str = ""
    rules: str = ""
