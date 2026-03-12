"""
This module provides an audio evaluation metric for speech analytics.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from .evaluator import SpeechAnalyticsEvaluator

__all__ = [
    "SpeechAnalyticsEvaluator",
]
