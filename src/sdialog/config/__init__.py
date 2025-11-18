"""
This module loads and processes the configuration for the sdialog package.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import yaml
import logging

from ..util import CacheDialogScore, ollama_check_and_pull_model, is_ollama_model_name

PROMPT_YAML_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
logger = logging.getLogger(__name__)

with open(PROMPT_YAML_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)


def _make_cfg_absolute_path(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            _make_cfg_absolute_path(v)
        elif isinstance(v, str) and not os.path.isabs(v):
            cfg[k] = os.path.join(os.path.dirname(__file__), v)


def llm(llm_name, **llm_kwargs):
    """
    Update the LLM model setting in the config.

    :param llm_name: The name of the LLM model to set.
    :type llm_name: str
    """
    if is_ollama_model_name(llm_name):
        ollama_check_and_pull_model(llm_name)
    config["llm"]["model"] = llm_name

    if llm_kwargs:
        llm_params(**llm_kwargs)


def llm_params(**params):
    """
    Update the LLM hyperparameters in the config.

    :param params: Dictionary of hyperparameter names and values.
    :type params: dict
    """
    if "llm" not in config:
        config["llm"] = {}
    config["llm"].update(params)


def cache(enable):
    """
    Enable or disable caching.

    :param enable: Whether to enable caching or not.
    :type enable: bool
    """
    config["cache"]["enabled"] = enable
    CacheDialogScore.set_enable_cache(enable)

    if enable:
        logger.info("Caching enabled. Cache path: %s", config["cache"]["path"])
        logger.warning(
            "Caution: Caching may cause outdated results if external or implicit variables affecting score computation "
            "are changed. For example, if you use LLMJudge-based scores without specifying the model (relying on the "
            "global default), the cache will return previous results even if the default model changes. "
            "To avoid inconsistencies, ensure all relevant parameters are explicitly set when caching is enabled.\n"
            "Use with caution! ;)"
        )


def cache_path(path):
    """
    Set the path for the cache directory.

    :param path: The new path for the cache directory.
    :type path: str
    """
    config["cache"]["path"] = path
    CacheDialogScore.set_cache_path(path)


def set_cache(path, enable=True):
    """
    Set the cache path and enable/disable caching.

    :param path: The path to the cache directory.
    :type path: str
    :param enable: Whether to enable caching or not.
    :type enable: bool
    """
    cache(path)
    cache_path(enable)


def clear_cache():
    """
    Clear the cache by deleting all files in the cache directory.
    """
    CacheDialogScore.clear_cache()
    logger.info("Cache cleared.")


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
