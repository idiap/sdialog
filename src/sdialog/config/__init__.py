"""
This module loads and processes the configuration for the sdialog package.

It reads a YAML configuration file named 'config.yaml' located in the same directory,
loads its contents, and ensures that all prompt file paths specified in the configuration
are converted to absolute paths if they are not already.

Attributes:
    config (dict): The loaded configuration dictionary with absolute prompt paths.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import yaml

from ..util import ollama_check_and_pull_model, is_ollama_model_name

PROMPT_YAML_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(PROMPT_YAML_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)


def _make_cfg_absolute_path(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            _make_cfg_absolute_path(v)
        elif isinstance(v, str) and not os.path.isabs(v):
            cfg[k] = os.path.join(os.path.dirname(__file__), v)


def set_llm(llm_name, **llm_kwargs):
    """
    Update the LLM model setting in the config.

    :param llm_name: The name of the LLM model to set.
    :type llm_name: str
    """
    if is_ollama_model_name(llm_name):
        ollama_check_and_pull_model(llm_name)
    config["llm"]["model"] = llm_name

    if llm_kwargs:
        set_llm_params(**llm_kwargs)


def set_llm_params(**params):
    """
    Update the LLM hyperparameters in the config.

    :param params: Dictionary of hyperparameter names and values.
    :type params: dict
    """
    if "llm" not in config:
        config["llm"] = {}
    config["llm"].update(params)


# Prompt setters for each prompt type in config.yaml
def set_persona_dialog_generator_prompt(path):
    """
    Set the path for the persona_dialog_generator prompt.

    :param path: The new path for the prompt file.
    :type path: str
    """
    config["prompts"]["persona_dialog_generator"] = path


def set_persona_generator_prompt(path):
    """
    Set the path for the persona_generator prompt.

    :param path: The new path for the prompt file.
    :type path: str
    """
    config["prompts"]["persona_generator"] = path


def set_dialog_generator_prompt(path):
    """
    Set the path for the dialog_generator prompt.

    :param path: The new path for the prompt file.
    :type path: str
    """
    config["prompts"]["dialog_generator"] = path


def set_persona_agent_prompt(path):
    """
    Set the path for the persona_agent prompt.

    :param path: The new path for the prompt file.
    :type path: str
    """
    config["prompts"]["persona_agent"] = path


# Make sure all default prompt paths are absolute
_make_cfg_absolute_path(config["prompts"])
