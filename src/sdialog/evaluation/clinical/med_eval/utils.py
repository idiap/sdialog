# medical_dialogue_evaluator/utils.py
"""
Utility functions for interfacing with the Language Model.
"""
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from .logger import logger
from .config import config

class EvaluationOutput(BaseModel):
    """The expected JSON structure for the LLM's response."""
    not_applicable: bool
    score: Optional[int] = Field(default=None, ge=1, le=5)
    justification: str

def get_llm_client():
    # ... (rest of the function remains the same)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    
    logger.info(f"Initializing LLM client with model: {config.llm.model} and temperature: {config.llm.temperature}")
    return ChatOpenAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
        api_key=api_key,
    ).with_structured_output(EvaluationOutput)
