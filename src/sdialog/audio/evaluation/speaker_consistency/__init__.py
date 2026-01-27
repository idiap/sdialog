"""
This module provides an audio evaluation metric for speaker consistency.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from .evaluator import SpeakerConsistency

__all__ = [
    "SpeakerConsistency",
]
