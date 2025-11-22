"""
This module contains the classes for the tasks and the function to apply a list of tasks to a dialog.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import logging
from sdialog import Dialog
from typing import List, Tuple, Any
from .base import Task, TaskModality

from .audio import (
    SpokenQuestionAnsweringTask,
    AutomaticSpeechRecognitionTask
)
from .nlp import (
    QuestionAnsweringTask,
    SummaryTask
)


__all__ = [
    "Task",
    "TaskModality",
    "dialog2tasks",
    "dialogs2tasks",
    "SpokenQuestionAnsweringTask",
    "AutomaticSpeechRecognitionTask",
    "QuestionAnsweringTask",
    "SummaryTask",
]


def dialog2tasks(dialog: Dialog, tasks: List[Tuple[Task, dict[str, Any]]]) -> Dialog:
    """
    Apply a list of tasks to a dialog.
    Each task is applied in the order of the list.
    The dialog is modified in place and the output is returned as a dialog.
    If an task has requirements, they are checked before applying the task.

    :param dialog: The dialog to annotate.
    :type dialog: Dialog
    :param tasks: The list of tasks to apply.
    :type tasks: List[Tuple[Task, dict[str, Any]]]
    :return: The annotated dialog.
    :rtype: Dialog
    """

    # Apply the tasks to the dialog.
    for task, args in tasks:

        # if args["skip_if_existing"] and os.path.exists(args["save_path"]):
        #     logging.info(
        #         f"[Task] Using existing data from {args['save_path']}"
        #     )
        #     dialog.set_annotations(
        #         task.get_task_name(),
        #         {
        #             "data": pd.read_csv(args["save_path"]),
        #             "modality": task.get_modality()
        #         }
        #     )
        #     continue

        # Apply the task and save the data.
        dialog = task.run(
            dialog=dialog,
            args={
                "save_path": args["save_path"]
            }
        )

    return dialog


def dialogs2tasks(
    dialogs: List[Dialog],
    tasks: List[Tuple[Task, dict[str, Any]]]
) -> List[Dialog]:
    """
    Apply a list of tasks to a list of dialogs.
    Each task is applied in the order of the list.
    The dialogs are modified in place and the output is returned as a list of dialogs.
    If an task has requirements, they are checked before applying the task.

    :param dialogs: The list of dialogs to annotate.
    :type dialogs: List[Dialog]
    :param tasks: The list of tasks to apply.
    :type tasks: List[Tuple[Task, dict[str, Any]]]
    :return: The annotated list of dialogs.
    :rtype: List[Dialog]
    """

    output_dialogs: List[Dialog] = []

    # For each given dialog
    for dialog in dialogs:

        logging.info(f"[Task] Applying tasks to dialog {dialog.id}")

        # Apply the tasks to the dialog.
        for task, args in tasks:

            # Build the path to save the data.
            _path = f"{args['output_dir']}/{task.get_task_name()}/{dialog.id}.csv"

            # Create the directory if it doesn't exist.
            os.makedirs(os.path.dirname(_path), exist_ok=True)

            # if args["skip_if_existing"] and os.path.exists(_path):
            #     logging.info(
            #         f"[Task] Using existing data from {_path}"
            #     )
            #     dialog.set_annotations(
            #         task.get_task_name(),
            #         {
            #             "data": pd.read_csv(_path),
            #             "modality": task.get_modality()
            #         }
            #     )
            #     continue

            # Apply the task and save the data.
            dialog = task.run(
                dialog=dialog,
                args={
                    "save_path": _path
                }
            )

        output_dialogs.append(dialog)

    return output_dialogs
