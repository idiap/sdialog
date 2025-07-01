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


PROMPT_YAML_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(PROMPT_YAML_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Ensure all default paths are absolute
for k, v in config["prompts"].items():
    if not os.path.isabs(v):
        config["prompts"][k] = os.path.join(os.path.dirname(__file__), v)
