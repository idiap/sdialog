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
import yaml
import json


PROMPT_YAML_PATH = os.path.join(os.path.dirname(__file__), "prompts", "config.yaml")
with open(PROMPT_YAML_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Ensure all default paths are absolute
for k, v in config["prompts"].items():
    if not os.path.isabs(v):
        config["prompts"][k] = os.path.join(os.path.dirname(__file__), v)


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
