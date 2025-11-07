"""
This module contains the classes for the task-specific annotators.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import logging
from enum import Enum
from sdialog import Dialog
from pydantic import BaseModel
from typing import Optional, Any
from abc import abstractmethod, ABC


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
