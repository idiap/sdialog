# medical_dialogue_evaluator/config.py
"""
Handles loading and validation of application configuration from a YAML file.
"""
import yaml
from pydantic import BaseModel, Field
from logger import logger


class LLMConfig(BaseModel):
    """Pydantic model for LLM configuration settings."""
    model: str = "gpt-4-turbo"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class AppConfig(BaseModel):
    """Root configuration model."""
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Loads configuration from a YAML file."""
    try:
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        return AppConfig(**config_data)
    except FileNotFoundError:
        logger.warning(f"Config file not found at '{path}'. Using default settings.")
        return AppConfig()
    except Exception as e:
        logger.error(f"Error loading config from '{path}': {e}. Using default settings.")
        return AppConfig()
    

config = load_config()

