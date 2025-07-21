"""
util: Utility Functions for sdialog

This module provides helper functions for the sdialog package, including serialization utilities to ensure
objects can be safely converted to JSON for storage or transmission.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import json
import uuid
import torch
import pickle
import logging
import subprocess
import transformers
import pandas as pd

from typing import Union
from functools import wraps
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from langchain_ollama.chat_models import ChatOllama

logger = logging.getLogger(__name__)


class KNNModel:
    def __init__(self, items, k=3):
        # items = (item, vector) pair list
        self.model = NearestNeighbors(algorithm='auto',
                                      metric="cosine",
                                      n_jobs=-1)
        self.k = k
        self.model.ix2id = {ix: item for ix, (item, _) in enumerate(items)}
        self.model.fit([vec for _, vec in items])

    def neighbors(self, target_emb, k=None):
        k = k or self.k
        dists, indexes = self.model.kneighbors([target_emb],
                                               min(k, len(self.model.ix2id)),
                                               return_distance=True)
        dists, indexes = dists[0], indexes[0]
        return [(self.model.ix2id[indexes[ix]], dist) for ix, dist in enumerate(dists)]

    __call__ = neighbors


class CacheDialogScore:
    def __init__(self, path, enable_cache=True):
        cache_dir = os.path.expanduser(path)
        os.makedirs(cache_dir, exist_ok=True)
        self.enable_cache = enable_cache
        self.cache_path = os.path.join(cache_dir, "dialog_scores_cache.json")
        self._score_obj_attributes = {}
        # Load cache dict if exists
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    def save(self):
        """
        Save the cache to the file.
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f)

    def cache(self, func):
        @wraps(func)
        def wrapper(score_obj, dialog, *args, **kwargs):
            # Build cache key
            dialog_path = getattr(dialog, "_path", None)
            if not self.enable_cache or dialog_path is None:
                result = func(score_obj, dialog, *args, **kwargs)
            else:
                # Build a hashable key that includes score_obj attributes (sorted, serializable only)
                # Cache attr_items for each score_obj class
                score_obj_class = score_obj.__class__.__name__
                if score_obj_class not in self._score_obj_attributes:
                    attrs = []
                    for attr in sorted(vars(score_obj)):
                        value = getattr(score_obj, attr)
                        try:
                            json.dumps(value)
                            attrs.append(attr)
                        except (TypeError, OverflowError):
                            continue
                    self._score_obj_attributes[score_obj_class] = attrs
                else:
                    attrs = self._score_obj_attributes[score_obj_class]
                attr_items = {attr: getattr(score_obj, attr) for attr in attrs}
                attr_str = json.dumps(attr_items, sort_keys=True)
                key = f"{score_obj_class}:{attr_str}:{dialog_path}"
                if key in self._cache:
                    return self._cache[key]
                result = func(score_obj, dialog, *args, **kwargs)
                self._cache[key] = result
            return result
        return wrapper

    # TODO: add a filter to clear only specific keys
    def clear(self):
        """
        Clear the cache.
        """
        self._cache = {}
        self.save()


def check_valid_model_name(func):
    def wrapper(model_name, *args, **kwargs):
        if not isinstance(model_name, str):
            return False
        return func(model_name, *args, **kwargs)
    return wrapper


def softmax(values, temperature=0.05, as_list=True):
    probs = torch.nn.functional.softmax(torch.tensor(values, dtype=float) / temperature, dim=0)
    return probs.tolist() if as_list else probs


def get_universal_id() -> str:
    """
    Generates a unique identifier for a dialog or persona using a universal ID generator.

    :return: A unique identifier as a string.
    :rtype: str
    """
    return str(uuid.uuid4())


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
    if not is_ollama_model_name(model_name):
        return llm_params

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


@check_valid_model_name
def is_ollama_model_name(model_name: str) -> bool:
    return (
        model_name.startswith("ollama:")
        or not is_huggingface_model_name(model_name)
        and not is_openai_model_name(model_name)
        and not is_google_genai_model_name(model_name)
        and not is_aws_model_name(model_name)
    )


@check_valid_model_name
def is_openai_model_name(model_name: str) -> bool:
    return model_name.startswith("openai:")


@check_valid_model_name
def is_aws_model_name(model_name: str) -> bool:
    return model_name.startswith("aws:")


@check_valid_model_name
def is_google_genai_model_name(model_name: str) -> bool:
    return re.match(r"^google(-genai)?:", model_name, re.IGNORECASE)


@check_valid_model_name
def is_huggingface_model_name(model_name: str) -> bool:
    return model_name.startswith("huggingface:") or "/" in model_name


def get_llm_model(model_name: str,
                  output_format: Union[dict, BaseModel] = None,
                  **llm_kwargs):
    # If model name has a slash, assume it's a Hugging Face model
    # Otherwise, assume it's an Ollama model
    if not isinstance(model_name, str):
        if hasattr(model_name, "invoke") and callable(model_name.invoke):
            llm = model_name
        else:
            raise ValueError("model_name must be a string or a valid Langchain model instance.")
    elif is_openai_model_name(model_name):
        # If the model name is a string, assume it's an OpenAI model
        from langchain_openai import ChatOpenAI
        if ":" in model_name:
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading OpenAI model: {model_name}")

        llm = ChatOpenAI(model=model_name, **llm_kwargs)
    elif is_aws_model_name(model_name):
        from langchain_aws import ChatBedrockConverse
        if ":" in model_name:
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading AWS model: {model_name}")

        if "seed" in llm_kwargs:
            logger.warning("Ignoring 'seed' parameter for AWS Bedrock models, as it is not supported.")
            llm_kwargs.pop("seed")

        llm = ChatBedrockConverse(model=model_name, **llm_kwargs)
    elif is_google_genai_model_name(model_name):
        from langchain_google_genai import ChatGoogleGenerativeAI
        if ":" in model_name:
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading Google GenAI model: {model_name}")

        llm = ChatGoogleGenerativeAI(model=model_name, **llm_kwargs)
    elif is_ollama_model_name(model_name):
        if model_name.startswith("ollama:"):
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading ChatOllama model: {model_name}")

        ollama_check_and_pull_model(model_name)  # Ensure the model is available locally
        llm_kwargs = set_ollama_model_defaults(model_name, llm_kwargs)
        llm = ChatOllama(model=model_name, **llm_kwargs)
    else:
        if model_name.startswith("huggingface:"):
            model_name = model_name.split(":", 1)[-1]
        logger.info(f"Loading Hugging Face model: {model_name}")
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

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

        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    if output_format:
        if isinstance(output_format, type) and issubclass(output_format, BaseModel):
            output_format = output_format.model_json_schema()
        if hasattr(llm, "with_structured_output"):
            llm = llm.with_structured_output(output_format)
        else:
            logger.error(f"The given model '{model_name}' does not support structured output. ")

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


def dict_to_table(data: dict,
                  sort_by: str = None,
                  sort_ascending: bool = True,
                  markdown: bool = False,
                  format: str = ".2f",
                  show: bool = True) -> str:
    """
    Print a dictionary of dictionaries as a table (markdown or plain text).

    :param data: The dictionary to convert to a table.
    :type data: dict
    :param sort_by: The key to sort (column name) the table by. If None, no sorting is applied.
    :type sort_by: str, optional
    :param sort_ascending: If True, sort in ascending order; otherwise, descending.
    :type sort_ascending: bool
    :param markdown: If True, format the table as Markdown; otherwise, as plain text.
    :type markdown: bool
    :param format: The format string for floating-point numbers (e.g., ".2f").
    :type format: str
    :param show: If True, print the table to the console.
    :type show: bool
    :return: The formatted table as a string.
    :rtype: str
    """
    if not data:
        return "(empty table)"
    df = pd.DataFrame(data).T
    df.index.name = "dataset"
    if sort_by:
        df.sort_values(by=sort_by, ascending=sort_ascending, inplace=True)
    if markdown:
        table = df.to_markdown(floatfmt=format)
    else:
        table = df.to_markdown(tablefmt='fancy_grid', floatfmt=format)

    if show:
        print(table)

    return table


def upper_camel_to_dash(name: str) -> str:
    """
    Converts an UpperCamelCase class name to dash-case.

    :param name: The class name in UpperCamelCase.
    :type name: str
    :return: The class name in dash-case.
    :rtype: str
    """
    # Improved to not split consecutive uppercase letters,
    # e.g., "HTTPServer" -> "http-server" instead of "h-t-t-p-server"
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1-\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', name)
    return name.lower()
