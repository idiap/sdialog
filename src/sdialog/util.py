"""
util: Utility Functions for sdialog

This module provides helper functions for the sdialog package, including serialization utilities to ensure
objects can be safely converted to JSON for storage or transmission.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import json
import time
import torch
import logging
import subprocess
import transformers

from typing import Union
from pydantic import BaseModel
from langchain_ollama.chat_models import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

logger = logging.getLogger(__name__)


def get_universal_id() -> str:
    """
    Generates a unique identifier for a dialog or persona using a universal ID generator.

    :return: A unique identifier as a string.
    :rtype: str
    """
    return int(time.time() * 1000)


def remove_newlines(s: str) -> str:
    """
    Removes all newline (\n and \r) characters from a string, replacing them with a single space.

    :param s: The input string.
    :type s: str
    :return: The string with all newlines replaced by spaces.
    :rtype: str
    """
    if type(s) is not str:
        return s
    return re.sub(r'\s+', ' ', s)


def get_timestamp() -> str:
    """
    Returns the current UTC timestamp as an ISO 8601 string (e.g., "2025-01-01T12:00:00Z").

    :return: Current UTC timestamp in ISO 8601 format with 'Z' suffix.
    :rtype: str
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def set_ollama_model_defaults(model_name: str, llm_params: dict) -> float:
    """ Set default parameters for an Ollama model if not already specified in llm_params."""
    defaults = {}
    try:
        result = subprocess.run(
            ["ollama", "show", "--parameters", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        # Look for a line like: "temperature: 0.7"
        for line in result.stdout.splitlines():
            m = re.match(r'(\w+)\s+([0-9]*\.?[0-9]+)', line)  # For now only with numbers
            # TODO: check support strings leter, gives Ollama ValidationError (probably the stop tokens?)
            # m = re.match(r'(\w+)\s+(.+)', line)
            if m:
                param, value = m.groups()
                if value.startswith('"'):
                    if param not in defaults:
                        defaults[param] = value.strip('"')
                    else:
                        if type(defaults[param]) is not list:
                            defaults[param] = [defaults[param]]
                        defaults[param].append(value.strip('"'))
                else:
                    try:
                        defaults[param] = float(value) if "." in value else int(value)
                    except ValueError:
                        logger.warning(f"Could not convert value '{value}' for parameter '{param}' "
                                       "to float or int. Skipping...")
        if "temperature" not in defaults:
            defaults["temperature"] = 0.8
    except Exception as e:
        logger.error(f"Error getting default parameters for model '{model_name}': {e}")

    for k, v in list(defaults.items()):
        if k in llm_params and llm_params[k] is not None:
            continue
        llm_params[k] = v
    return llm_params


def get_llm_model(model_name: Union[ChatOllama, str] = None,
                  output_format: Union[dict, BaseModel] = None,
                  llm_kwargs: dict = {}):
    # If model name has a slash, assume it's a Hugging Face model
    # Otherwise, assume it's an Ollama model
    if "/" in model_name:
        logger.info(f"Loading Hugging Face model: {model_name}")

        # Remove 'seed' from llm_kwargs if present (not supported by HuggingFace pipeline)
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if k != "seed"}
        llm_kwargs["model"] = model_name

        # Default HuggingFace parameters
        hf_defaults = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=2048,
            do_sample=True,
            repetition_penalty=1.03,
            return_full_text=False,
        )
        hf_params = {**hf_defaults, **llm_kwargs}

        pipe = transformers.pipeline("text-generation", **hf_params)
        # TODO: avoid the eos token warning message for certain llm
        # TODO: if tokenizer doesn't have a chat template, set a default one

        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
    else:
        if output_format and isinstance(output_format, BaseModel):
            output_format = output_format.model_json_schema()

        logger.info(f"Loading ChatOllama model: {model_name}")
        ollama_check_and_pull_model(model_name)  # Ensure the model is available locally
        llm_kwargs = set_ollama_model_defaults(model_name, llm_kwargs)
        llm = ChatOllama(model=model_name,
                         format=output_format,
                         **llm_kwargs)
    return llm


def ollama_check_and_pull_model(model_name: str) -> bool:
    """
    Check if an Ollama model is available locally, and if not, pull it from the hub.

    :param model_name: The name of the Ollama model to check/pull.
    :type model_name: str
    :return: True if the model is available (either was already local or successfully pulled), False otherwise.
    :rtype: bool
    """
    try:
        # First, check if the model is available locally
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        # Check if the model name is in the output
        if model_name in result.stdout:
            return True

        # If not available locally, try to pull it
        logger.info(f"Model '{model_name}' not found locally. Pulling it from the hub...")
        pull_result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            check=True
        )

        if pull_result.returncode == 0:
            logger.info(f"Successfully pulled model '{model_name}'.")
            return True
        else:
            logger.error(f"Failed to pull model '{model_name}': {pull_result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error while pulling model '{model_name}' from ollama hub: {e}")
        return False


def make_serializable(data: dict) -> dict:
    """
    Converts non-serializable values in a dictionary to strings so the dictionary can be safely serialized to JSON.

    :param data: The dictionary to process.
    :type data: dict
    :return: The dictionary with all values JSON-serializable.
    :rtype: dict
    """

    if type(data) is not dict:
        raise TypeError("Input must be a dictionary")

    for key, value in data.items():
        if hasattr(value, "json") and callable(value.json):
            data[key] = value.json()
        else:
            try:
                json.dumps(value)
            except (TypeError, OverflowError):
                data[key] = str(value)

    return data


def camel_or_snake_to_words(varname: str) -> str:
    """
    Converts a camelCase or snake_case variable name to a space-separated string of words.

    :param varname: The variable name in camelCase or snake_case.
    :type varname: str
    :return: The variable name as space-separated words.
    :rtype: str
    """
    # Replace underscores with spaces (snake_case)
    s = varname.replace('_', ' ')
    # Insert spaces before capital letters (camelCase, PascalCase)
    s = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', s)
    # Normalize multiple spaces
    return ' '.join(s.split())


def remove_audio_tags(text: str) -> str:
    """
    Remove all the tags that use those formatting: <>, {}, (), []
    """
    return re.sub(r'<[^>]*>', '', text)
