"""
personas: Persona and Agent Definitions for Synthetic Dialogue Generation

This module provides classes for defining personas (character profiles) and simulating agents that role-play
these personas in synthetic dialogue generation. Agents interact using LLMs and can be orchestrated for
complex behaviors.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Séverin Baroudi <severin.baroudi@lis-lab.fr>
# SPDX-License-Identifier: MIT
import logging

from typing import List, Union

from .base import BaseAttributeModel

logger = logging.getLogger(__name__)


class BasePersona(BaseAttributeModel):
    """
    Base class for defining a persona (character profile) for role-play.
    """
    def print(self):
        """
        Pretty-prints the persona, including its metadata information.
        """
        super().print(object_name="Persona")


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
