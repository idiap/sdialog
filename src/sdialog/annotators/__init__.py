"""
This module contains the classes for the annotators and the function to apply a list of annotators to a dialog.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import logging
from enum import Enum
from sdialog import Dialog
from pydantic import BaseModel
from abc import abstractmethod, ABC
from typing import Optional, Any, List, Tuple


class TaskModality(str, Enum):
    """
    Enum for the type of modalities of the task.
    e.g. TEXT_TO_TEXT means that the task takes a text as input and returns a text as output.
    e.g. AUDIO_TO_TEXT means that the task takes an audio as input and returns a text as output.
    """

    # Unimodal tasks
    TEXT_TO_TEXT = "text_to_text"
    AUDIO_TO_AUDIO = "audio_to_audio"

    # Bimodal tasks
    TEXT_TO_AUDIO = "text_to_audio"
    AUDIO_TO_TEXT = "audio_to_text"

    def __str__(self) -> str:
        return self.value


class Annotator(ABC):
    """
    Abstract class for the annotators.
    An annotator is a class that adds a specific annotation to a dialog.
    The annotation is a dictionary that contains the data of the task, and some metadata.
    To create a new annotator, you need to inherit from this class and implement the abstract methods.
    """

    def __init__(self):
        """
        Initialize the annotator.
        """
        self._requirements = self.get_requirements()
        self._task_name = self.get_task_name()
        self._modality = self.get_modality()
        self._structured_model = self.get_structured_model()

    def list_requirements(self) -> list["Annotator"]:
        """
        List the requirements for the task.
        :return: The list of requirements for the task.
        :rtype: list[Annotator]
        """
        return [
            requirement.get_task_name() for requirement in self._requirements
        ]

    @abstractmethod
    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the task.
        :return: The modality of the task.
        :rtype: list[TaskModality]
        """
        raise NotImplementedError("Annotator subclass must implement this method get_modality")

    @abstractmethod
    def get_requirements(self) -> list["Annotator"]:
        """
        Get the requirements for the task.
        The requirements are a list of other annotators that need to be applied before this one.
        :return: The requirements for the task.
        :rtype: list[Annotator]
        """
        raise NotImplementedError("Annotator subclass must implement this method get_requirements")

    @abstractmethod
    def get_task_name(self) -> str:
        """
        Get the name of the task.
        :return: The name of the task.
        :rtype: str
        """
        raise NotImplementedError("Annotator subclass must implement this method get_task_name")

    @abstractmethod
    def annotate(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Annotate a dialog for a specific task.
        :param dialog: The dialog to annotate.
        :type dialog: Dialog
        :param args: Additional arguments to pass to the annotate method.
        :type args: dict[str, Any]
        :return: The annotated dialog.
        :rtype: Dialog
        """
        raise NotImplementedError("Annotator subclass must implement this method annotate")

    def check_requirements(self, dialog: Dialog) -> Dialog:
        """
        Check if the dialog has the requirements for the annotator.
        If the requirements are not met, apply the Annotator to the dialog.
        :param dialog: The dialog to check the requirements for.
        :type dialog: Dialog
        :return: The dialog with the requirements applied.
        :rtype: Dialog
        """
        # For each requirement, check if the annotation is already present in the dialog.
        # If not, apply the requirement to the dialog.
        for requirement in self._requirements:
            if (
                requirement.get_task_name() not in dialog.get_annotations()
                or dialog.get_annotations(requirement.get_task_name()) is None
                or dialog.get_annotations(requirement.get_task_name())["data"] is None
            ):
                logging.info(
                    f"[Annotator] Requirement {requirement.get_task_name()} not met, applying it to the dialog"
                )
                dialog = requirement.annotate(dialog)

        return dialog

    def get_structured_model(self) -> Optional[BaseModel]:
        """
        Get the structured model for the parsing of the data for the task during LLM inference.
        The structured model is a pydantic model that defines the structure of the data for the task,
        so that the LLM can parse the data correctly.
        If the task doesn't have a structured model, return None.
        :return: The structured model for the task.
        :rtype: BaseModel
        """
        return None

    @abstractmethod
    def save(self, data: Any, args: dict[str, Any] = {}) -> None:
        """
        Save the data to a file.
        :param data: The data to save.
        :type data: Any
        :param args: Additional arguments to pass to the saving method.
        It includes the 'save_path' where to save the data.
        :type args: dict[str, Any]
        :return: None
        :rtype: None
        """
        raise NotImplementedError("Annotator subclass must implement this method save")


def apply_annotators(dialog: Dialog, annotators: List[Tuple[Annotator, dict[str, Any]]]) -> Dialog:
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


def apply_annotators_to_dialogs(
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
