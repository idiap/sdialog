"""
Audio evaluation module.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from .base import Result
from .base import BaseAudioDialogScore
from typing import Union, List, Dict, Any
from sdialog.audio.dialog import AudioDialog
from .speaker_consistency.evaluator import SpeakerConsistency

__all__ = ["BaseAudioDialogScore", "SpeakerConsistency", "evaluate"]


def evaluate(
    dialogs: Union[AudioDialog, List[AudioDialog]],
    evaluators: List[BaseAudioDialogScore],
    evaluator_args: Dict[str, Any] = None,
    compute_on_all_dialogs: bool = False
) -> Dict[str, any]:
    """
    Apply a list of audio evaluation functions to one or many dialogs.
    :param dialogs: The dialog or list of dialogs to evaluate.
    :type dialogs: Union[AudioDialog, List[AudioDialog]]
    :param evaluators: The list of evaluators to apply. Can be strings or instances.
    :type evaluators: List[Union[str, BaseAudioDialogScore]]
    :param evaluator_args: Optional dictionary of arguments for instantiating evaluators from strings.
    :type evaluator_args: Dict[str, Any]
    :return: A dictionary of evaluation results, keyed by evaluator name.
    :rtype: Dict[str, any]
    """
    if not isinstance(dialogs, list):
        dialogs = [dialogs]

    if evaluator_args is None:
        evaluator_args = {}

    per_dialog_results: Dict[str, Dict[str, Result]] = {}
    overall_results: Dict[str, Result] = {}

    # Evaluate each evaluator on each dialog
    for evaluator in evaluators:

        evaluator_results: Dict[str, Result] = {}

        # Evaluate the evaluator on each dialog
        for dialog in dialogs:
            evaluator_results[dialog.id] = evaluator.score(dialog)

        # Store the results of the evaluator on each dialog
        per_dialog_results[str(evaluator)] = evaluator_results

        # Compute the overall result of the evaluator on all dialogs
        if compute_on_all_dialogs:
            overall_results[str(evaluator)] = evaluator.results2result(evaluator_results)

    return {
        "per_dialog_results": per_dialog_results,
        "overall_results": overall_results
    }
