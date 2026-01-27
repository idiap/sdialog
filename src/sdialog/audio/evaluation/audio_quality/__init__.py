"""
This module provides an audio evaluation metric for audio quality.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from .evaluator import AudioQualityEvaluator

__all__ = [
    "AudioQualityEvaluator",
]
