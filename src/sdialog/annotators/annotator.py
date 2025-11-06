"""
This module contains the classes for the task-specific annotators.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import logging
from enum import Enum
from sdialog import Dialog
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
    def annotate(self, dialog: Dialog) -> Dialog:
        """
        Annotate a dialog for a specific task.
        :param dialog: The dialog to annotate.
        :type dialog: Dialog
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
