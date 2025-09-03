"""
personas: Persona Definitions for Synthetic Dialogue Generation

This module provides classes for defining personas (character profiles) and simulating agents that role-play
these personas in synthetic dialogue generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Séverin Baroudi <severin.baroudi@lis-lab.fr>
# SPDX-License-Identifier: MIT
import logging

from typing import List, Union

from .base import BaseAttributeModel

logger = logging.getLogger(__name__)

BasePersona = BaseAttributeModel


class Persona(BasePersona):
    """
    Standard persona class with common attributes for role-play.

    :ivar name: Name of the persona.
    :vartype name: str
    :ivar age: Age of the persona (can be an int or a descriptive string like "middle-aged").
    :vartype age: Union[int, str]
    :ivar race: Race / ethnicity of the persona.
    :vartype race: str
    :ivar gender: Gender of the persona.
    :vartype gender: str
    :ivar language: Preferred language of communication.
    :vartype language: str
    :ivar role: Role, profession, or primary identity descriptor.
    :vartype role: str
    :ivar background: Background or life history summary.
    :vartype background: str
    :ivar personality: Personality traits summary (free text).
    :vartype personality: str
    :ivar circumstances: Current situational context (e.g., "recently moved", "under stress").
    :vartype circumstances: str
    :ivar rules: Constraints, style or behavioral rules to enforce.
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
    :ivar age: Age (numeric or descriptive string).
    :vartype age: Union[int, str]
    :ivar race: Race / ethnicity.
    :vartype race: str
    :ivar gender: Gender identity.
    :vartype gender: str
    :ivar language: Preferred language.
    :vartype language: str
    :ivar weight: Weight (numeric with unit or descriptive string).
    :vartype weight: str
    :ivar height: Height (numeric with unit or descriptive string).
    :vartype height: Union[str, float]
    :ivar voice_characteristics: Voice, accent, tone, pacing, etc.
    :vartype voice_characteristics: str
    :ivar occupation: Current occupation or professional role.
    :vartype occupation: str
    :ivar education: Education level or academic background.
    :vartype education: str
    :ivar socioeconomic_status: Socioeconomic status descriptor.
    :vartype socioeconomic_status: str
    :ivar interests: General interests (comma-separated or free text).
    :vartype interests: str
    :ivar hobbies: Hobbies (comma-separated or free text).
    :vartype hobbies: str
    :ivar politeness: Politeness style/level.
    :vartype politeness: str
    :ivar forgetfulness: Forgetfulness tendency.
    :vartype forgetfulness: str
    :ivar attentiveness: Attentiveness or focus tendency.
    :vartype attentiveness: str
    :ivar communication_style: Style of communication (e.g., direct, verbose).
    :vartype communication_style: str
    :ivar empathy_level: Empathy level or descriptor.
    :vartype empathy_level: str
    :ivar political_views: Political alignment (e.g., conservative, moderate, apolitical).
    :vartype political_views: str
    :ivar religious_beliefs: Religious stance (e.g., religious, agnostic, atheist).
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
    Patient persona with essential / minimal plus behavioral and demographic attributes for dialogue generation.

    :ivar name: Patient name.
    :vartype name: str
    :ivar age: Patient age (numeric or descriptive).
    :vartype age: Union[int, str]
    :ivar race: Race / ethnicity.
    :vartype race: str
    :ivar gender: Gender identity.
    :vartype gender: str
    :ivar language: Preferred communication language.
    :vartype language: str
    :ivar forgetfulness: Forgetfulness tendency (qualitative or numeric).
    :vartype forgetfulness: Union[str, float]
    :ivar formality: Formality of speech (qualitative or numeric scale).
    :vartype formality: Union[str, float]
    :ivar hurriedness: Degree of impatience / hurriedness.
    :vartype hurriedness: Union[str, float]
    :ivar openness: Openness to share information.
    :vartype openness: Union[str, float]
    :ivar height: Height (numeric with unit or descriptive).
    :vartype height: Union[int, str]
    :ivar weight: Weight (numeric with unit or descriptive).
    :vartype weight: Union[int, str]
    :ivar occupation: Occupation or employment status.
    :vartype occupation: str
    :ivar marital_status: Marital status.
    :vartype marital_status: str
    :ivar insurance: Insurance provider / status.
    :vartype insurance: str
    :ivar reason_for_visit: Chief complaint / presenting problem.
    :vartype reason_for_visit: str
    :ivar medical_history: Past medical history (string or list of conditions).
    :vartype medical_history: Union[str, List[str]]
    :ivar medical_conditions: Known diagnosed conditions (string or list).
    :vartype medical_conditions: Union[str, List[str]]
    :ivar medications_current: Current medications (string or list).
    :vartype medications_current: Union[str, List[str]]
    :ivar allergies: Known allergies (string or list).
    :vartype allergies: Union[str, List[str]]
    :ivar family_history: Family medical history (string or list).
    :vartype family_history: Union[str, List[str]]
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
    ExtendedPatient persona with additional health-related attributes.
    Inherits all attributes from ExtendedPersona plus the following medical context fields.

    :ivar reason_for_visit: Chief complaint or reason for consultation.
    :vartype reason_for_visit: str
    :ivar symptoms: Reported symptoms (free text or summarized list).
    :vartype symptoms: str
    :ivar vital_signs: Vital signs summary (e.g., "BP 120/80, HR 72").
    :vartype vital_signs: str
    :ivar health_literacy: Health literacy level descriptor.
    :vartype health_literacy: str
    :ivar medical_conditions: Known or chronic conditions (free text summary).
    :vartype medical_conditions: str
    :ivar medications: Current medications summary.
    :vartype medications: str
    :ivar allergies: Allergy list / summary.
    :vartype allergies: str
    :ivar family_history: Family medical history summary.
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
    Doctor persona with essential professional and behavioral attributes.

    :ivar name: Doctor's name.
    :vartype name: str
    :ivar age: Doctor's age (numeric or descriptive).
    :vartype age: Union[int, str]
    :ivar race: Race / ethnicity.
    :vartype race: str
    :ivar gender: Gender identity.
    :vartype gender: str
    :ivar language: Working language.
    :vartype language: str
    :ivar years_of_experience: Years (or range) of medical practice.
    :vartype years_of_experience: Union[int, str]
    :ivar speciality: Medical speciality (as spelled in this class).
    :vartype speciality: str
    :ivar forgetfulness: Forgetfulness tendency.
    :vartype forgetfulness: str
    :ivar formality: Formality level in communication.
    :vartype formality: str
    :ivar hurriedness: Degree of time pressure / haste.
    :vartype hurriedness: str
    :ivar openness: Openness / approachability.
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
    ExtendedDoctor persona adding professional credentials.
    Inherits all attributes from ExtendedPersona.

    :ivar specialty: Medical specialty / domain focus.
    :vartype specialty: str
    :ivar years_of_experience: Years (or range) of clinical experience.
    :vartype years_of_experience: Union[int, str]
    :ivar certifications: Professional certifications / board statuses.
    :vartype certifications: str
    :ivar work_experience: Summary of prior practice settings / roles.
    :vartype work_experience: str
    """
    specialty: str = ""
    years_of_experience: Union[int, str] = ""
    certifications: str = ""
    work_experience: str = ""


class Customer(BasePersona):
    """
    Persona for a customer in a customer service interaction.

    Identification / Profile:
    :ivar name: Customer name.
    :vartype name: str
    :ivar age: Customer age (numeric or descriptive).
    :vartype age: Union[int, str]
    :ivar gender: Customer gender.
    :vartype gender: str
    :ivar language: Preferred language.
    :vartype language: str
    :ivar customer_id: Internal customer identifier.
    :vartype customer_id: str
    :ivar occupation: Customer occupation.
    :vartype occupation: str

    Loyalty / Tenure:
    :ivar account_tenure: How long they have been a customer (e.g., "2 years").
    :vartype account_tenure: str
    :ivar membership_level: Plan/tier (e.g., basic, premium).
    :vartype membership_level: str
    :ivar loyalty_status: Loyalty descriptor (e.g., loyal, at-risk).
    :vartype loyalty_status: str
    :ivar fidelity_score: Loyalty score (numeric or descriptive).
    :vartype fidelity_score: Union[str, float, int]

    Issue Context:
    :ivar issue: Short summary of current problem.
    :vartype issue: str
    :ivar issue_category: High-level category (billing, technical, etc.).
    :vartype issue_category: str
    :ivar issue_description: Detailed issue description.
    :vartype issue_description: str
    :ivar issue_history: Brief summary of related past issues.
    :vartype issue_history: str
    :ivar desired_outcome: Customer's desired resolution / goal.
    :vartype desired_outcome: str

    Knowledge / Competence:
    :ivar knowledge_domain: Subject/domain familiarity (e.g., novice, expert).
    :vartype knowledge_domain: str
    :ivar technical_expertise: Legacy field for backward compatibility.
    :vartype technical_expertise: str

    Emotional / Behavioral State:
    :ivar sentiment: Overall emotional tone (e.g., frustrated, neutral).
    :vartype sentiment: str
    :ivar anger_level: Anger intensity descriptor.
    :vartype anger_level: str
    :ivar tiredness: Fatigue level.
    :vartype tiredness: str
    :ivar patience_level: Patience descriptor.
    :vartype patience_level: str
    :ivar politeness: Politeness style (e.g., polite, curt).
    :vartype politeness: str
    :ivar personality: Personality descriptor (e.g., analytical).
    :vartype personality: str
    :ivar instruction_following: Likelihood of following instructions.
    :vartype instruction_following: str
    :ivar forgetfulness: Tendency to forget prior guidance.
    :vartype forgetfulness: str

    Contact / Interaction History:
    :ivar times_called: Number of prior contacts (numeric or descriptive).
    :vartype times_called: Union[int, str]
    :ivar preferred_channel: Preferred support channel.
    :vartype preferred_channel: str
    :ivar prior_interactions_summary: Summary of earlier interactions.
    :vartype prior_interactions_summary: str

    Meta:
    :ivar urgency: Perceived urgency (e.g., low, high).
    :vartype urgency: str
    :ivar rules: Constraints or special handling notes.
    :vartype rules: str
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
    :vartype name: str
    :ivar language: Working language.
    :vartype language: str
    :ivar agent_id: Internal agent identifier.
    :vartype agent_id: str
    :ivar role: Agent role or queue designation.
    :vartype role: str
    :ivar experience_years: Years (or range) of support experience.
    :vartype experience_years: str
    :ivar product_scope: Products or domains covered.
    :vartype product_scope: str
    :ivar product_knowledge_level: Knowledge depth (e.g., basic, expert).
    :vartype product_knowledge_level: str
    :ivar communication_style: Communication style (e.g., concise, empathetic).
    :vartype communication_style: str
    :ivar empathy_level: Empathy descriptor.
    :vartype empathy_level: str
    :ivar politeness: Politeness level descriptor.
    :vartype politeness: str
    :ivar resolution_authority_level: Authority level for resolutions/escalations.
    :vartype resolution_authority_level: str
    :ivar escalation_policy: Summary of escalation criteria/process.
    :vartype escalation_policy: str
    :ivar average_handle_time: Typical handling time (e.g., "6m").
    :vartype average_handle_time: str
    :ivar adherence_notes: Notes on process or QA adherence.
    :vartype adherence_notes: str
    :ivar stress_tolerance: Stress handling capability descriptor.
    :vartype stress_tolerance: str
    :ivar performance_notes: Performance KPIs or evaluation notes.
    :vartype performance_notes: str
    :ivar rules: Internal rules, compliance reminders, or constraints.
    :vartype rules: str
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
