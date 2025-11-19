"""
This module contains the classes for the annotators and the function to apply a list of annotators to a dialog.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import logging
from sdialog import Dialog
from typing import List, Tuple, Any
from .base import Annotator, TaskModality

from .audio import (
    SpokenQuestionAnsweringAnnotator,
    AutomaticSpeechRecognitionAnnotator
)
from .nlp import (
    QuestionAnsweringAnnotator,
    SummaryAnnotator
)


__all__ = [
    "Annotator",
    "TaskModality",
    "dialog2annotations",
    "dialogs2annotations",
    "SpokenQuestionAnsweringAnnotator",
    "AutomaticSpeechRecognitionAnnotator",
    "QuestionAnsweringAnnotator",
    "SummaryAnnotator",
]


def dialog2annotations(dialog: Dialog, annotators: List[Tuple[Annotator, dict[str, Any]]]) -> Dialog:
    """
    Apply a list of annotators to a dialog.
    Each annotator is applied in the order of the list.
    The dialog is modified in place and the output is returned as a dialog.
    If an annotator has requirements, they are checked before applying the annotator.

    :param dialog: The dialog to annotate.
    :type dialog: Dialog
    :param annotators: The list of annotators to apply.
    :type annotators: List[Tuple[Annotator, dict[str, Any]]]
    :return: The annotated dialog.
    :rtype: Dialog
    """

    # Apply the annotators to the dialog.
    for annotator, args in annotators:

        # if args["skip_if_existing"] and os.path.exists(args["save_path"]):
        #     logging.info(
        #         f"[Annotator] Using existing data from {args["save_path"]}"
        #     )
        #     dialog.set_annotations(
        #         annotator.get_task_name(),
        #         {
        #             "data": pd.read_csv(args["save_path"]),
        #             "modality": annotator.get_modality()
        #         }
        #     )
        #     continue

        # Apply the annotator and save the data.
        dialog = annotator.annotate(
            dialog=dialog,
            args={
                "save_path": args["save_path"]
            }
        )

    return dialog


def dialogs2annotations(
    dialogs: List[Dialog],
    annotators: List[Tuple[Annotator, dict[str, Any]]]
) -> List[Dialog]:
    """
    Apply a list of annotators to a list of dialogs.
    Each annotator is applied in the order of the list.
    The dialogs are modified in place and the output is returned as a list of dialogs.
    If an annotator has requirements, they are checked before applying the annotator.

    :param dialogs: The list of dialogs to annotate.
    :type dialogs: List[Dialog]
    :param annotators: The list of annotators to apply.
    :type annotators: List[Tuple[Annotator, dict[str, Any]]]
    :return: The annotated list of dialogs.
    :rtype: List[Dialog]
    """

    output_dialogs: List[Dialog] = []

    # For each given dialog
    for dialog in dialogs:

        logging.info(f"[Annotator] Applying annotators to dialog {dialog.id}")

        # Apply the annotators to the dialog.
        for annotator, args in annotators:

            # Build the path to save the data.
            _path = f"{args['output_dir']}/{annotator.get_task_name()}/{dialog.id}.csv"

            # Create the directory if it doesn't exist.
            os.makedirs(os.path.dirname(_path), exist_ok=True)

            # if args["skip_if_existing"] and os.path.exists(_path):
            #     logging.info(
            #         f"[Annotator] Using existing data from {_path}"
            #     )
            #     dialog.set_annotations(
            #         annotator.get_task_name(),
            #         {
            #             "data": pd.read_csv(_path),
            #             "modality": annotator.get_modality()
            #         }
            #     )
            #     continue

            # Apply the annotator and save the data.
            dialog = annotator.annotate(
                dialog=dialog,
                args={
                    "save_path": _path
                }
            )

        output_dialogs.append(dialog)

    return output_dialogs
