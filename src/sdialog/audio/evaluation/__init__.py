"""
Audio evaluation module.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from .base import BaseAudioDialogScore
from typing import Union, List, Dict, Any
from sdialog.audio.dialog import AudioDialog
from .speaker_consistency.evaluator import SpeakerConsistency

__all__ = ["BaseAudioDialogScore", "SpeakerConsistency", "evaluate"]


def evaluate(
    dialogs: Union[AudioDialog, List[AudioDialog]],
    evaluators: List[BaseAudioDialogScore],
    evaluator_args: Dict[str, Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Apply a list of audio evaluation functions to one or many dialogs.
    :param dialogs: The dialog or list of dialogs to evaluate.
    :type dialogs: Union[AudioDialog, List[AudioDialog]]
    :param evaluators: The list of evaluators to apply. Can be strings or instances.
    :type evaluators: List[Union[str, BaseAudioDialogScore]]
    :param evaluator_args: Optional dictionary of arguments for instantiating evaluators from strings.
    :type evaluator_args: Dict[str, Any]
    :return: A dictionary of evaluation results, keyed by dialog ID.
    :rtype: Dict[str, Dict[str, Any]]
    """
    if not isinstance(dialogs, list):
        dialogs = [dialogs]

    if evaluator_args is None:
        evaluator_args = {}

    results = {}

    for dialog in dialogs:

        dialog_results = {}

        for evaluator in evaluators:
            score = evaluator.score(dialog)
            dialog_results[str(evaluator)] = {"score": score}

        results[dialog.id] = dialog_results

    return results
