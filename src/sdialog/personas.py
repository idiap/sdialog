"""
This module provides classes for defining personas (character profiles) and simulating agents that role-play
these personas in synthetic dialogue generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Séverin Baroudi <severin.baroudi@lis-lab.fr>
# SPDX-License-Identifier: MIT
import logging

from typing import List, Union
from pydantic import Field

from .base import BaseAttributeModel

logger = logging.getLogger(__name__)

#: Abstract base class for defining personas. Alias for :class:`sdialog.base.BaseAttributeModel`
BasePersona = BaseAttributeModel


class Persona(BasePersona):
    """
    Standard persona class with common attributes for role-play.

    :param name: Name of the persona.
    :type name: str
    :param age: Age of the persona (can be an int or a descriptive string like "middle-aged").
    :type age: Union[int, str]
    :param race: Race / ethnicity of the persona.
    :type race: str
    :param gender: Gender of the persona.
    :type gender: str
    :param language: Preferred language of communication.
    :type language: str
    :param role: Role, profession, or primary identity descriptor.
    :type role: str
    :param background: Background or life history summary.
    :type background: str
    :param personality: Personality traits summary (free text).
    :type personality: str
    :param circumstances: Current situational context (e.g., "recently moved", "under stress").
    :type circumstances: str
    :param rules: Constraints, style or behavioral rules to enforce.
    :type rules: str
    """

    name: str = Field("", description="Name of the persona.")
    age: Union[int, str] = Field(None, description="Age (integer or descriptive string like 'middle-aged').")
    race: str = Field("", description="Race or ethnicity.")
    gender: str = Field("", description="Gender of the persona.")
    language: str = Field("English", description="Preferred language of communication.")
    role: str = Field("", description="Role, profession, or primary identity descriptor.")
    background: str = Field("", description="Background or life history summary.")
    personality: str = Field("", description="Personality traits summary.")
    circumstances: str = Field("", description="Current situational context.")
    rules: str = Field("", description="Constraints, style, or behavioral rules to enforce.")


class ExtendedPersona(BasePersona):
    """
    Extended persona class with additional demographic, personality, and background attributes.

    :param name: Name of the persona.
    :type name: str
    :param age: Age (numeric or descriptive string).
    :type age: Union[int, str]
    :param race: Race / ethnicity.
    :type race: str
    :param gender: Gender identity.
    :type gender: str
    :param language: Preferred language.
    :type language: str
    :param weight: Weight (numeric with unit or descriptive string).
    :type weight: Union[str, int, float]
    :param height: Height (numeric with unit or descriptive string).
    :type height: Union[str, int, float]
    :param voice_characteristics: Voice, accent, tone, pacing, etc.
    :type voice_characteristics: str
    :param occupation: Current occupation or professional role.
    :type occupation: str
    :param education: Education level or academic background.
    :type education: str
    :param socioeconomic_status: Socioeconomic status descriptor.
    :type socioeconomic_status: str
    :param interests: General interests (comma-separated or free text).
    :type interests: str
    :param hobbies: Hobbies (comma-separated or free text).
    :type hobbies: str
    :param politeness: Politeness style/level.
    :type politeness: str
    :param forgetfulness: Forgetfulness tendency.
    :type forgetfulness: str
    :param attentiveness: Attentiveness or focus tendency.
    :type attentiveness: str
    :param communication_style: Style of communication (e.g., direct, verbose).
    :type communication_style: str
    :param empathy_level: Empathy level or descriptor.
    :type empathy_level: str
    :param political_views: Political alignment (e.g., conservative, moderate, apolitical).
    :type political_views: str
    :param religious_beliefs: Religious stance (e.g., religious, agnostic, atheist).
    :type religious_beliefs: str
    """

    name: str = Field("", description="Name of the persona.")
    # Demographics
    age: Union[int, str] = Field("", description="Age (numeric or descriptive).")
    race: str = Field("", description="Race or ethnicity.")
    gender: str = Field("", description="Gender identity.")
    language: str = Field("English", description="Preferred language.")
    weight: Union[str, int, float] = Field("", description="Weight (value with unit or descriptive text).")
    height: Union[str, int, float] = Field("", description="Height (value with unit or descriptive text).")
    voice_characteristics: str = Field("", description="Voice characteristics (accent, tone, pacing).")
    # Background
    occupation: str = Field("", description="Current occupation or professional role.")
    education: str = Field("", description="Education level or academic background.")
    socioeconomic_status: str = Field("", description="Socioeconomic status descriptor.")
    # Interests and hobbies
    interests: str = Field("", description="General interests (comma-separated or free text).")
    hobbies: str = Field("", description="Hobbies (comma-separated or free text).")
    # Personality traits
    politeness: str = Field("", description="Politeness style or level.")
    forgetfulness: str = Field("", description="Forgetfulness tendency.")
    attentiveness: str = Field("", description="Attentiveness or focus tendency.")
    communication_style: str = Field("", description="Communication style (e.g., direct, verbose).")
    empathy_level: str = Field("", description="Empathy level descriptor.")
    # Political and social views
    political_views: str = Field("", description="Political alignment (e.g., conservative, moderate, apolitical).")
    religious_beliefs: str = Field("", description="Religious stance (e.g., religious, agnostic, atheist).")


class Patient(BasePersona):
    """
    Patient persona with essential / minimal plus behavioral and demographic attributes for dialogue generation.

    :param name: Patient name.
    :type name: str
    :param age: Patient age (numeric or descriptive).
    :type age: Union[int, str]
    :param race: Race / ethnicity.
    :type race: str
    :param gender: Gender identity.
    :type gender: str
    :param language: Preferred communication language.
    :type language: str
    :param forgetfulness: Forgetfulness tendency (qualitative or numeric).
    :type forgetfulness: Union[str, float]
    :param formality: Formality of speech (qualitative or numeric scale).
    :type formality: Union[str, float]
    :param hurriedness: Degree of impatience / hurriedness.
    :type hurriedness: Union[str, float]
    :param openness: Openness to share information.
    :type openness: Union[str, float]
    :param height: Height (numeric with unit or descriptive).
    :type height: Union[str, int, float]
    :param weight: Weight (numeric with unit or descriptive).
    :type weight: Union[str, int, float]
    :param occupation: Occupation or employment status.
    :type occupation: str
    :param marital_status: Marital status.
    :type marital_status: str
    :param insurance: Insurance provider / status.
    :type insurance: str
    :param reason_for_visit: Chief complaint / presenting problem.
    :type reason_for_visit: str
    :param symptoms: Reported symptoms.
    :type symptoms: Union[str, List[str]]
    :param medical_history: Past medical history (string or list of conditions).
    :type medical_history: Union[str, List[str]]
    :param medical_conditions: Known diagnosed conditions (string or list).
    :type medical_conditions: Union[str, List[str]]
    :param medications: Current medications (string or list).
    :type medications: Union[str, List[str]]
    :param allergies: Known allergies (string or list).
    :type allergies: Union[str, List[str]]
    :param family_history: Family medical history (string or list).
    :type family_history: Union[str, List[str]]
    """

    name: str = Field("", description="Patient name.")
    age: Union[int, str] = Field(None, description="Patient age (numeric or descriptive).")
    race: str = Field("", description="Race or ethnicity.")
    gender: str = Field("", description="Gender identity.")
    language: str = Field("English", description="Preferred communication language.")
    forgetfulness: Union[str, float] = Field("", description="Forgetfulness tendency (qualitative or numeric).")
    formality: Union[str, float] = Field("", description="Formality of speech.")
    hurriedness: Union[str, float] = Field("", description="Degree of impatience or hurriedness.")
    openness: Union[str, float] = Field("", description="Openness to share information.")
    height: Union[str, int, float] = Field("", description="Height (value with unit or descriptive).")
    weight: Union[str, int, float] = Field("", description="Weight (value with unit or descriptive).")
    occupation: str = Field("", description="Occupation or employment status.")
    marital_status: str = Field("", description="Marital status.")
    insurance: str = Field("", description="Insurance provider or status.")
    reason_for_visit: str = Field("", description="Chief complaint or presenting problem.")
    symptoms: Union[str, List[str]] = Field("", description="Reported symptoms.")
    medical_history: Union[str, List[str]] = Field("", description="Past medical history.")
    medical_conditions: Union[str, List[str]] = Field("", description="Known diagnosed conditions.")
    medications: Union[str, List[str]] = Field("", description="Current medications.")
    allergies: Union[str, List[str]] = Field("", description="Known allergies.")
    family_history: Union[str, List[str]] = Field("", description="Family medical history.")


class ExtendedPatient(ExtendedPersona):
    """
    ExtendedPatient persona with additional health-related attributes.
    Inherits all attributes from ExtendedPersona plus medical context fields.

    :param reason_for_visit: Chief complaint or reason for consultation.
    :type reason_for_visit: str
    :param symptoms: Reported symptoms (free text or summarized list).
    :type symptoms: Union[str, List[str]]
    :param vital_signs: Vital signs summary (e.g., "BP 120/80, HR 72").
    :type vital_signs: str
    :param health_literacy: Health literacy level descriptor.
    :type health_literacy: str
    :param medical_conditions: Known or chronic conditions (free text summary).
    :type medical_conditions: Union[str, List[str]]
    :param medications: Current medications summary.
    :type medications: Union[str, List[str]]
    :param allergies: Allergy list / summary.
    :type allergies: Union[str, List[str]]
    :param family_history: Family medical history summary.
    :type family_history: Union[str, List[str]]
    """

    reason_for_visit: str = Field("", description="Chief complaint or reason for consultation.")
    symptoms: Union[str, List[str]] = Field("", description="Reported symptoms.")
    vital_signs: str = Field("", description="Vital signs summary (e.g., 'BP 120/80, HR 72').")
    health_literacy: str = Field("", description="Health literacy level.")
    medical_conditions: Union[str, List[str]] = Field("", description="Known or chronic conditions summary.")
    medications: Union[str, List[str]] = Field("", description="Current medications summary.")
    allergies: Union[str, List[str]] = Field("", description="Allergy list or summary.")
    family_history: Union[str, List[str]] = Field("", description="Family medical history summary.")


class Doctor(BasePersona):
    """
    Doctor persona with essential professional and behavioral attributes.

    :param name: Doctor's name.
    :type name: str
    :param age: Doctor's age (numeric or descriptive).
    :type age: Union[int, str]
    :param race: Race / ethnicity.
    :type race: str
    :param gender: Gender identity.
    :type gender: str
    :param language: Working language.
    :type language: str
    :param years_of_experience: Years (or range) of medical practice.
    :type years_of_experience: Union[int, str]
    :param specialty: Medical specialty (as spelled in this class).
    :type specialty: str
    :param forgetfulness: Forgetfulness tendency.
    :type forgetfulness: str
    :param formality: Formality level in communication.
    :type formality: str
    :param hurriedness: Degree of time pressure / haste.
    :type hurriedness: str
    :param openness: Openness / approachability.
    :type openness: str
    """

    name: str = Field("", description="Doctor's name.")
    age: Union[int, str] = Field("", description="Doctor's age (numeric or descriptive).")
    race: str = Field("", description="Race or ethnicity.")
    gender: str = Field("", description="Gender identity.")
    language: str = Field("English", description="Working language.")
    years_of_experience: Union[int, str] = Field("", description="Years or range of medical practice.")
    specialty: str = Field("", description="Medical specialty.")
    forgetfulness: str = Field("", description="Forgetfulness tendency.")
    formality: str = Field("", description="Formality level in communication.")
    hurriedness: str = Field("", description="Degree of time pressure or haste.")
    openness: str = Field("", description="Openness or approachability.")


class ExtendedDoctor(ExtendedPersona):
    """
    ExtendedDoctor persona adding professional credentials.
    Inherits all attributes from ExtendedPersona plus, the following ones.

    :param specialty: Medical specialty / domain focus.
    :type specialty: str
    :param years_of_experience: Years (or range) of clinical experience.
    :type years_of_experience: Union[int, str]
    :param certifications: Professional certifications / board statuses.
    :type certifications: str
    :param work_experience: Summary of prior practice settings / roles.
    :type work_experience: str
    """

    specialty: str = Field("", description="Medical specialty or domain focus.")
    years_of_experience: Union[int, str] = Field("", description="Years (or range) of clinical experience.")
    certifications: str = Field("", description="Professional certifications / board statuses.")
    work_experience: str = Field("", description="Summary of prior practice settings / roles.")


class Customer(BasePersona):
    """
    Persona for a customer in a customer service interaction.

    :param name: Customer name.
    :type name: str
    :param age: Customer age (numeric or descriptive).
    :type age: Union[int, str]
    :param gender: Customer gender.
    :type gender: str
    :param language: Preferred language.
    :type language: str
    :param customer_id: Internal customer identifier.
    :type customer_id: Union[str, int]
    :param occupation: Customer occupation.
    :type occupation: str
    :param account_tenure: How long they have been a customer (e.g., "2 years").
    :type account_tenure: str
    :param membership_level: Plan/tier (e.g., basic, premium).
    :type membership_level: str
    :param loyalty_status: Loyalty descriptor (e.g., loyal, at-risk).
    :type loyalty_status: str
    :param fidelity_score: Loyalty score (numeric or descriptive).
    :type fidelity_score: Union[str, float, int]
    :param issue: Short summary of current problem.
    :type issue: str
    :param issue_category: High-level category (billing, technical, etc.).
    :type issue_category: str
    :param issue_description: Detailed issue description.
    :type issue_description: str
    :param issue_history: Brief summary of related past issues.
    :type issue_history: str
    :param desired_outcome: Customer's desired resolution / goal.
    :type desired_outcome: str
    :param knowledge_domain: Subject/domain familiarity (e.g., novice, expert).
    :type knowledge_domain: str
    :param technical_expertise: Legacy field for backward compatibility.
    :type technical_expertise: str
    :param sentiment: Overall emotional tone (e.g., frustrated, neutral).
    :type sentiment: str
    :param anger_level: Anger intensity descriptor.
    :type anger_level: str
    :param tiredness: Fatigue level.
    :type tiredness: str
    :param patience_level: Patience descriptor.
    :type patience_level: str
    :param politeness: Politeness style (e.g., polite, curt).
    :type politeness: str
    :param personality: Personality descriptor (e.g., analytical).
    :type personality: str
    :param instruction_following: Likelihood of following instructions.
    :type instruction_following: str
    :param forgetfulness: Tendency to forget prior guidance.
    :type forgetfulness: str
    :param times_called: Number of prior contacts (numeric or descriptive).
    :type times_called: Union[int, str]
    :param preferred_channel: Preferred support channel.
    :type preferred_channel: str
    :param prior_interactions_summary: Summary of earlier interactions.
    :type prior_interactions_summary: str
    :param urgency: Perceived urgency (e.g., low, high).
    :type urgency: str
    :param rules: Constraints or special handling notes.
    :type rules: str
    """

    # Identification / profile
    name: str = Field("", description="Customer name.")
    age: Union[int, str] = Field("", description="Customer age (numeric or descriptive).")
    gender: str = Field("", description="Customer gender.")
    language: str = Field("English", description="Preferred language.")
    customer_id: Union[str, int] = Field("", description="Internal customer identifier.")
    occupation: str = Field("", description="Customer occupation.")

    # Loyalty / tenure
    account_tenure: str = Field("", description="Customer tenure (e.g., '2 years').")
    membership_level: str = Field("", description="Subscription or plan tier.")
    loyalty_status: str = Field("", description="Loyalty descriptor (e.g., loyal, at-risk).")
    fidelity_score: Union[str, float, int] = Field("", description="Loyalty / fidelity score.")

    # Issue context
    issue: str = Field("", description="Short summary of current problem.")
    issue_category: str = Field("", description="Issue category (billing, technical, etc.).")
    issue_description: str = Field("", description="Detailed issue description.")
    issue_history: str = Field("", description="Summary of prior related issues.")
    desired_outcome: str = Field("", description="Desired resolution outcome.")

    # Knowledge / competence
    knowledge_domain: str = Field("", description="Domain knowledge level or area.")
    technical_expertise: str = Field("", description="Legacy technical expertise field.")

    # Emotional / behavioral state
    sentiment: str = Field("", description="Overall emotional tone.")
    anger_level: str = Field("", description="Anger intensity descriptor.")
    tiredness: str = Field("", description="Fatigue level.")
    patience_level: str = Field("", description="Patience descriptor.")
    politeness: str = Field("", description="Politeness style.")
    personality: str = Field("", description="Personality descriptor.")
    instruction_following: str = Field("", description="Likelihood of following instructions.")
    forgetfulness: str = Field("", description="Tendency to forget prior guidance.")

    # Contact / interaction history
    times_called: Union[int, str] = Field("", description="Number of prior related contacts.")
    preferred_channel: str = Field("", description="Preferred support channel.")
    prior_interactions_summary: str = Field("", description="Summary of prior interactions.")

    # Meta
    urgency: str = Field("", description="Perceived urgency level.")
    rules: str = Field("", description="Constraints or special handling notes.")


class SupportAgent(BasePersona):
    """
    Persona for a customer service / support agent.

    :param name: Agent name.
    :type name: str
    :param language: Working language.
    :type language: str
    :param agent_id: Internal agent identifier.
    :type agent_id: Union[str, int]
    :param role: Agent role or queue designation.
    :type role: str
    :param experience_years: Years (or range) of support experience.
    :type experience_years: Union[int, str]
    :param product_scope: Products or domains covered.
    :type product_scope: str
    :param product_knowledge_level: Knowledge depth (e.g., basic, expert).
    :type product_knowledge_level: str
    :param communication_style: Communication style (e.g., concise, empathetic).
    :type communication_style: str
    :param empathy_level: Empathy descriptor.
    :type empathy_level: str
    :param politeness: Politeness level descriptor.
    :type politeness: str
    :param resolution_authority_level: Authority level for resolutions/escalations.
    :type resolution_authority_level: str
    :param escalation_policy: Summary of escalation criteria/process.
    :type escalation_policy: str
    :param average_handle_time: Typical handling time (e.g., "6m").
    :type average_handle_time: Union[int, float, str]
    :param adherence_notes: Notes on process or QA adherence.
    :type adherence_notes: str
    :param stress_tolerance: Stress handling capability descriptor.
    :type stress_tolerance: str
    :param performance_notes: Performance KPIs or evaluation notes.
    :type performance_notes: str
    :param rules: Internal rules, compliance reminders, or constraints.
    :type rules: str
    """

    name: str = Field("", description="Agent name.")
    language: str = Field("English", description="Working language.")
    agent_id: Union[str, int] = Field("", description="Internal agent identifier.")
    role: str = Field("Customer Support Agent", description="Agent role or queue designation.")
    experience_years: Union[int, str] = Field("", description="Years (or range) of support experience.")
    product_scope: str = Field("", description="Products or domains covered.")
    product_knowledge_level: str = Field("", description="Knowledge depth (e.g., basic, expert).")
    communication_style: str = Field("", description="Communication style (e.g., concise, empathetic).")
    empathy_level: str = Field("", description="Empathy descriptor.")
    politeness: str = Field("", description="Politeness level descriptor.")
    resolution_authority_level: str = Field("", description="Authority for resolutions/escalations.")
    escalation_policy: str = Field("", description="Escalation criteria / process summary.")
    average_handle_time: Union[int, float, str] = Field("", description="Typical handling time (e.g., '6m').")
    adherence_notes: str = Field("", description="Process adherence / QA notes.")
    stress_tolerance: str = Field("", description="Stress handling capability descriptor.")
    performance_notes: str = Field("", description="Performance KPIs or evaluation notes.")
    rules: str = Field("", description="Internal rules or compliance reminders.")
