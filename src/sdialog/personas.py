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
from pydantic import Field

from .base import BaseAttributeModel

logger = logging.getLogger(__name__)

BasePersona = BaseAttributeModel


class Persona(BasePersona):
    """
    Standard persona class with common attributes for role-play.

    Attributes:
        name (str): Name of the persona.
        age (int | str): Age or descriptive category (e.g., 'middle-aged').
        race (str): Race or ethnicity.
        gender (str): Gender identity or expression.
        language (str): Preferred communication language.
        role (str): Role / profession / primary identity descriptor.
        background (str): Brief life history or contextual background.
        personality (str): Personality traits summary.
        circumstances (str): Current situational context (e.g., stressors, recent changes).
        rules (str): Style constraints, behavioral norms, or hard guidelines.

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

    Attributes:
        name (str): Persona name.
        age (int | str): Numeric age or descriptive label.
        race (str): Race or ethnicity.
        gender (str): Gender identity.
        language (str): Preferred language.
        weight (Union[str, int, float]): Weight (value + unit or descriptive).
        height (Union[str, int, float]): Height (value + unit or descriptive).
        voice_characteristics (str): Voice/accent/tone/pacing notes.
        occupation (str): Current occupation.
        education (str): Education level / academic background.
        socioeconomic_status (str): Socioeconomic status descriptor.
        interests (str): General interests.
        hobbies (str): Hobbies.
        politeness (str): Politeness style or level.
        forgetfulness (str): Forgetfulness tendency.
        attentiveness (str): Attentiveness or focus level.
        communication_style (str): Communication style (e.g., direct, verbose).
        empathy_level (str): Empathy descriptor.
        political_views (str): Political alignment.
        religious_beliefs (str): Religious stance / worldview.

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
    :vartype weight: Union[str, int, float]
    :ivar height: Height (numeric with unit or descriptive string).
    :vartype height: Union[str, int, float]
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

    Attributes:
        name (str): Patient name.
        age (int | str): Numeric age or descriptive label.
        race (str): Race or ethnicity.
        gender (str): Gender identity.
        language (str): Preferred language.
        forgetfulness (str | float): Forgetfulness tendency (qualitative or scaled).
        formality (str | float): Formality in speech.
        hurriedness (str | float): Impatience / urgency level.
        openness (str | float): Willingness to share information.
        height (str | int | float): Height descriptor or value.
        weight (str | int | float): Weight descriptor or value.
        occupation (str): Occupation or employment status.
        marital_status (str): Marital status.
        insurance (str): Insurance status/provider.
        reason_for_visit (str): Chief complaint.
        symptoms (str | list[str]): Reported symptoms.
        medical_history (str | list[str]): Past medical history.
        medical_conditions (str | list[str]): Diagnosed conditions.
        medications (str | list[str]): Current medications.
        allergies (str | list[str]): Known allergies.
        family_history (str | list[str]): Family medical history.

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
    :vartype height: Union[str, int, float]
    :ivar weight: Weight (numeric with unit or descriptive).
    :vartype weight: Union[str, int, float]
    :ivar occupation: Occupation or employment status.
    :vartype occupation: str
    :ivar marital_status: Marital status.
    :vartype marital_status: str
    :ivar insurance: Insurance provider / status.
    :vartype insurance: str
    :ivar reason_for_visit: Chief complaint / presenting problem.
    :vartype reason_for_visit: str
    :ivar symptoms: Reported symptoms.
    :vartype symptoms: Union[str, List[str]]
    :ivar medical_history: Past medical history (string or list of conditions).
    :vartype medical_history: Union[str, List[str]]
    :ivar medical_conditions: Known diagnosed conditions (string or list).
    :vartype medical_conditions: Union[str, List[str]]
    :ivar medications: Current medications (string or list).
    :vartype medications: Union[str, List[str]]
    :ivar allergies: Known allergies (string or list).
    :vartype allergies: Union[str, List[str]]
    :ivar family_history: Family medical history (string or list).
    :vartype family_history: Union[str, List[str]]
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

    Attributes:
        reason_for_visit (str): Chief complaint.
        symptoms (str | list[str]): Reported symptoms.
        vital_signs (str): Vital signs summary.
        health_literacy (str): Health literacy descriptor.
        medical_conditions (str | list[str]): Chronic/known conditions summary.
        medications (str | list[str]): Current medications summary.
        allergies (str | list[str]): Allergy summary.
        family_history (str | list[str]): Family medical history summary.

    :ivar reason_for_visit: Chief complaint or reason for consultation.
    :vartype reason_for_visit: str
    :ivar symptoms: Reported symptoms (free text or summarized list).
    :vartype symptoms: Union[str, List[str]]
    :ivar vital_signs: Vital signs summary (e.g., "BP 120/80, HR 72").
    :vartype vital_signs: str
    :ivar health_literacy: Health literacy level descriptor.
    :vartype health_literacy: str
    :ivar medical_conditions: Known or chronic conditions (free text summary).
    :vartype medical_conditions: Union[str, List[str]]
    :ivar medications: Current medications summary.
    :vartype medications: Union[str, List[str]]
    :ivar allergies: Allergy list / summary.
    :vartype allergies: Union[str, List[str]]
    :ivar family_history: Family medical history summary.
    :vartype family_history: Union[str, List[str]]
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

    Attributes:
        name (str): Doctor name.
        age (int | str): Numeric age or descriptive label.
        race (str): Race or ethnicity.
        gender (str): Gender identity.
        language (str): Working language.
        years_of_experience (int | str): Clinical experience duration.
        specialty (str): Medical specialty (note spelling).
        forgetfulness (str): Forgetfulness tendency.
        formality (str): Communication formality.
        hurriedness (str): Time pressure / pace.
        openness (str): Approachability / openness.

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
    :ivar specialty: Medical specialty (as spelled in this class).
    :vartype specialty: str
    :ivar forgetfulness: Forgetfulness tendency.
    :vartype forgetfulness: str
    :ivar formality: Formality level in communication.
    :vartype formality: str
    :ivar hurriedness: Degree of time pressure / haste.
    :vartype hurriedness: str
    :ivar openness: Openness / approachability.
    :vartype openness: str
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

    Attributes:
        specialty (str): Medical specialty / domain focus.
        years_of_experience (int | str): Clinical experience length.
        certifications (str): Certifications / board statuses.
        work_experience (str): Prior practice settings / roles.

    :ivar specialty: Medical specialty / domain focus.
    :vartype specialty: str
    :ivar years_of_experience: Years (or range) of clinical experience.
    :vartype years_of_experience: Union[int, str]
    :ivar certifications: Professional certifications / board statuses.
    :vartype certifications: str
    :ivar work_experience: Summary of prior practice settings / roles.
    :vartype work_experience: str
    """

    specialty: str = Field("", description="Medical specialty or domain focus.")
    years_of_experience: Union[int, str] = Field("", description="Years (or range) of clinical experience.")
    certifications: str = Field("", description="Professional certifications / board statuses.")
    work_experience: str = Field("", description="Summary of prior practice settings / roles.")


class Customer(BasePersona):
    """
    Persona for a customer in a customer service interaction.

    Attributes (grouped):
        Identification:
            name (str), age (int | str), gender (str), language (str),
            customer_id (Union[str, int]), occupation (str)
        Loyalty:
            account_tenure (str), membership_level (str),
            loyalty_status (str), fidelity_score (Union[str, float, int])
        Issue Context:
            issue (str), issue_category (str), issue_description (str),
            issue_history (str), desired_outcome (str)
        Knowledge:
            knowledge_domain (str), technical_expertise (str)
        Emotional State:
            sentiment (str), anger_level (str), tiredness (str),
            patience_level (str), politeness (str), personality (str),
            instruction_following (str), forgetfulness (str)
        Interaction History:
            times_called (int | str), preferred_channel (str),
            prior_interactions_summary (str)
        Meta:
            urgency (str), rules (str)

    :ivar name: Customer name.
    :vartype name: str
    :ivar age: Customer age (numeric or descriptive).
    :vartype age: Union[int, str]
    :ivar gender: Customer gender.
    :vartype gender: str
    :ivar language: Preferred language.
    :vartype language: str
    :ivar customer_id: Internal customer identifier.
    :vartype customer_id: Union[str, int]
    :ivar occupation: Customer occupation.
    :vartype occupation: str
    :ivar account_tenure: How long they have been a customer (e.g., "2 years").
    :vartype account_tenure: str
    :ivar membership_level: Plan/tier (e.g., basic, premium).
    :vartype membership_level: str
    :ivar loyalty_status: Loyalty descriptor (e.g., loyal, at-risk).
    :vartype loyalty_status: str
    :ivar fidelity_score: Loyalty score (numeric or descriptive).
    :vartype fidelity_score: Union[str, float, int]
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
    :ivar knowledge_domain: Subject/domain familiarity (e.g., novice, expert).
    :vartype knowledge_domain: str
    :ivar technical_expertise: Legacy field for backward compatibility.
    :vartype technical_expertise: str
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
    :ivar times_called: Number of prior contacts (numeric or descriptive).
    :vartype times_called: Union[int, str]
    :ivar preferred_channel: Preferred support channel.
    :vartype preferred_channel: str
    :ivar prior_interactions_summary: Summary of earlier interactions.
    :vartype prior_interactions_summary: str
    :ivar urgency: Perceived urgency (e.g., low, high).
    :vartype urgency: str
    :ivar rules: Constraints or special handling notes.
    :vartype rules: str
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

    Attributes:
        name (str): Agent name.
        language (str): Working language.
        agent_id (str | int): Internal agent ID.
        role (str): Role / queue designation.
        experience_years (int | str): Support experience duration.
        product_scope (str): Products / domains covered.
        product_knowledge_level (str): Knowledge depth level.
        communication_style (str): Communication style.
        empathy_level (str): Empathy descriptor.
        politeness (str): Politeness level.
        resolution_authority_level (str): Resolution authority level.
        escalation_policy (str): Escalation process summary.
        average_handle_time (str | int | float): Typical handling time (e.g., '6m').
        adherence_notes (str): Notes on policy/process adherence.
        stress_tolerance (str): Stress handling capability.
        performance_notes (str): Performance KPIs / notes.
        rules (str): Internal rules / compliance reminders.

    :ivar name: Agent name.
    :vartype name: str
    :ivar language: Working language.
    :vartype language: str
    :ivar agent_id: Internal agent identifier.
    :vartype agent_id: Union[str, int]
    :ivar role: Agent role or queue designation.
    :vartype role: str
    :ivar experience_years: Years (or range) of support experience.
    :vartype experience_years: Union[int, str]
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
    :vartype average_handle_time: Union[int, float, str]
    :ivar adherence_notes: Notes on process or QA adherence.
    :vartype adherence_notes: str
    :ivar stress_tolerance: Stress handling capability descriptor.
    :vartype stress_tolerance: str
    :ivar performance_notes: Performance KPIs or evaluation notes.
    :vartype performance_notes: str
    :ivar rules: Internal rules, compliance reminders, or constraints.
    :vartype rules: str
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
