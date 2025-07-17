# medical_dialogue_evaluator/prompts.py
"""
Contains the master Jinja2 prompt template for the LLM-based evaluation.
"""

PROMPT_TEMPLATE = """
You are an expert clinical reviewer performing a quality audit. Your task is to evaluate a medical dialogue based on a specific performance indicator.

**1. Indicator Details:**
- **ID:** {{ indicator_id }}
- **Name:** {{ indicator_name }}
- **Definition:** {{ indicator_definition }}

**2. Scoring Rubric (1-5 Scale):**
- **Score 1 (Low Performance):** {{ low_example }}
- **Score 3 (Mixed Performance):** A neutral or partial adherence to the standard.
- **Score 5 (High Performance):** {{ high_example }}

**3. Dialogue for Evaluation:**
---
{{ dialogue_content }}
---

**4. Your Task:**
First, determine if the indicator is applicable to the dialogue. The indicator is NOT APPLICABLE if the dialogue does not contain any relevant events, decisions, or information to be judged against the indicator's definition.

- **If Applicable:** Provide a score from 1 to 5 and a detailed justification.
- **If Not Applicable:** State that it is not applicable and briefly explain why (e.g., "The dialogue does not involve prescribing medication, so guideline concordance for prescriptions cannot be assessed.").

**5. Required Output Format:**
Respond with a single JSON object.

- For an applicable indicator, use this format:
{
  "not_applicable": false,
  "score": <your 1-5 integer score>,
  "justification": "<Your detailed, evidence-based justification here>"
}

- For a non-applicable indicator, use this format:
{
  "not_applicable": true,
  "score": null,
  "justification": "<Your brief explanation for why it is not applicable>"
}
"""

