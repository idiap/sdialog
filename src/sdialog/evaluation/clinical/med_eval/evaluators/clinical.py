# medical_dialogue_evaluator/evaluators/clinical.py
"""
Implements the concrete evaluator classes for all 15 clinical indicators.
To add a new indicator, simply add a new class here that inherits from BaseEvaluator.
The auto-discovery mechanism in __init__.py will find it automatically.
"""

from .base import BaseEvaluator

class MedicalKnowledgeAccuracyEvaluator(BaseEvaluator):
    indicator_id = "med_knowledge_accuracy"
    indicator_name = "Medical Knowledge Accuracy"
    definition = "Did the doctor provide medically correct information about diagnoses, medications, and tests?"
    scoring_rubric = {"low_example": "Doctor recommends antibiotics for viral cold", "high_example": "Doctor correctly distinguishes viral vs. bacterial"}

class GuidelineConcordanceEvaluator(BaseEvaluator):
    indicator_id = "guideline_concordance"
    indicator_name = "Guideline/Standard Concordance"
    definition = "Were recommendations consistent with current best-practice clinical guidelines or standards?"
    scoring_rubric = {"low_example": "Recommends obsolete or unapproved treatment", "high_example": "Follows latest hypertension management guidelines"}

class IndividualizationOfAdviceEvaluator(BaseEvaluator):
    indicator_id = "individualization_of_advice"
    indicator_name = "Individualization of Advice"
    definition = "Was advice tailored to the patient’s age, comorbidities, and context?"
    scoring_rubric = {"low_example": "Same advice for all, ignores patient’s diabetes", "high_example": "Adjusts advice for patient with kidney disease"}

class SymptomConcernCoverageEvaluator(BaseEvaluator):
    indicator_id = "symptom_coverage"
    indicator_name = "Symptom/Concern Coverage"
    definition = "Did the doctor address all main symptoms, concerns, and questions presented by the patient?"
    scoring_rubric = {"low_example": "Ignores chest pain complaint", "high_example": "Addresses all complaints and clarifies priorities"}

class RiskFactorExplorationEvaluator(BaseEvaluator):
    indicator_id = "risk_factor_exploration"
    indicator_name = "Risk Factor/Red Flag Exploration"
    definition = "Did the doctor ask about important risk factors or “red flag” symptoms?"
    scoring_rubric = {"low_example": "No inquiry about heart history for chest pain", "high_example": "Checks cardiac history, family history, etc."}

class FollowUpSafetyNettingEvaluator(BaseEvaluator):
    indicator_id = "follow_up_safety_netting"
    indicator_name = "Follow-Up and Safety Netting"
    definition = "Did the doctor explain what to do if things worsen and give clear follow-up instructions?"
    scoring_rubric = {"low_example": "No advice if symptoms worsen", "high_example": "Tells patient when to seek urgent care, schedules follow-up"}

class HarmAvoidanceEvaluator(BaseEvaluator):
    indicator_id = "harm_avoidance"
    indicator_name = "Harm Avoidance"
    definition = "Did the doctor avoid giving unsafe advice or making dangerous errors?"
    scoring_rubric = {"low_example": "Recommends medication patient is allergic to", "high_example": "Reviews allergies, avoids all unsafe options"}

class RiskIdentificationEvaluator(BaseEvaluator):
    indicator_id = "risk_identification"
    indicator_name = "Risk Identification"
    definition = "Did the doctor identify and address potential risks (e.g., allergies, drug interactions)?"
    scoring_rubric = {"low_example": "Prescribes medication without checking for interactions", "high_example": "Proactively asks about allergies and drug list"}

class SafetyCommunicationEvaluator(BaseEvaluator):
    indicator_id = "safety_communication"
    indicator_name = "Safety Communication"
    definition = "Did the doctor clearly communicate risks and limitations of the care plan?"
    scoring_rubric = {"low_example": "Doesn’t mention medication side effects", "high_example": "Explains potential side effects, monitoring needs"}

class ClarityOfCommunicationEvaluator(BaseEvaluator):
    indicator_id = "clarity_of_communication"
    indicator_name = "Clarity of Communication"
    definition = "Were the doctor’s explanations clear and understandable for a layperson?"
    scoring_rubric = {"low_example": "Uses jargon patient doesn’t understand", "high_example": "Uses plain language and checks patient understanding"}

class EmpathyActiveListeningEvaluator(BaseEvaluator):
    indicator_id = "empathy_active_listening"
    indicator_name = "Empathy and Active Listening"
    definition = "Did the doctor show empathy, validation, and attentive listening?"
    scoring_rubric = {"low_example": "Ignores or interrupts patient", "high_example": "Acknowledges patient’s worries, listens carefully"}

class SharedDecisionMakingEvaluator(BaseEvaluator):
    indicator_id = "shared_decision_making"
    indicator_name = "Engagement/Shared Decision-Making"
    definition = "Did the doctor encourage patient questions and involve the patient in decisions?"
    scoring_rubric = {"low_example": "Doesn’t ask for patient input", "high_example": "Discusses options, asks for patient preferences"}

class RespectNonjudgmentalAttitudeEvaluator(BaseEvaluator):
    indicator_id = "respect_nonjudgmental_attitude"
    indicator_name = "Respect and Nonjudgmental Attitude"
    definition = "Did the doctor remain respectful and unbiased throughout the encounter?"
    scoring_rubric = {"low_example": "Makes assumptions, uses biased language", "high_example": "Respectful and open regardless of patient background"}

class ProfessionalConductEvaluator(BaseEvaluator):
    indicator_id = "professional_conduct"
    indicator_name = "Professional Conduct"
    definition = "Was the doctor’s behavior appropriate and in line with professional standards?"
    scoring_rubric = {"low_example": "Makes inappropriate jokes or comments", "high_example": "Professional demeanor and language throughout"}

class CulturalSensitivityConfidentialityEvaluator(BaseEvaluator):
    indicator_id = "cultural_sensitivity_confidentiality"
    indicator_name = "Cultural Sensitivity & Confidentiality"
    definition = "Did the doctor demonstrate cultural awareness and protect patient privacy?"
    scoring_rubric = {"low_example": "Dismisses cultural beliefs or shares private info", "high_example": "Adapts to patient’s culture, maintains privacy"}