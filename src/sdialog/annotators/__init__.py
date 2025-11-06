"""
This module contains the classes for the task-specific annotators.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import logging
from typing import List
from sdialog import Dialog
from sdialog.annotators.annotator import Annotator, TaskModality


def apply_annotators(dialog: Dialog, annotators: List[Annotator]) -> Dialog:
    """
    Apply a list of annotators to a dialog.
    Each annotator is applied in the order of the list.
    The dialog is modified in place.
    If an annotator has requirements, they are checked before applying the annotator.

    :param dialog: The dialog to annotate.
    :type dialog: Dialog
    :param annotators: The list of annotators to apply.
    :type annotators: List[Annotator]
    :return: The annotated dialog.
    :rtype: Dialog
    """

    for annotator in annotators:
        dialog = annotator.annotate(dialog)

    return dialog


class QuestionAnsweringAnnotator(Annotator):
    """
    Annotator to annotate a dialog for question answering tasks.
    The annotation is a list of questions and answers.
    This annotator has no requirements.
    The modality of the task is text-to-text.
    """

    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the question answering task.
        :return: The modality of the question answering task.
        :rtype: list[TaskModality]
        """
        return [
            TaskModality.TEXT_TO_TEXT
        ]

    def get_task_name(self) -> str:
        """
        Get the name of the question answering task.
        :return: The name of the question answering task.
        :rtype: str
        """
        return "question_answering"

    def get_requirements(self) -> list[Annotator]:
        """
        Get the requirements for the question answering task.
        :return: The requirements for the question answering task.
        :rtype: list[Annotator]
        """
        return []

    def annotate(self, dialog: Dialog) -> Dialog:
        """
        Annotate a dialog for question answering tasks.
        The annotation is a list of questions and answers.
        :param dialog: The dialog to annotate.
        :type dialog: Dialog
        :return: The annotated dialog.
        :rtype: Dialog
        """
        logging.info("[QuestionAnsweringAnnotator] Annotating dialog for question answering tasks")

        dialog = self.check_requirements(dialog)

        # For now, we use a dummy list of questions and answers.
        # TODO: Use a real model to generate questions and answers.
        _annotations = {
            "data": [
                {
                    "question": "What is the capital of France?",
                    "answer": "Paris"
                },
                {
                    "question": "What is the capital of Germany?",
                    "answer": "Berlin"
                }
            ],
            "modality": self._modality
        }

        dialog.add_annotations(self._task_name, _annotations)

        logging.info(
            "[QuestionAnsweringAnnotator] "
            f"Annotation done for {len(_annotations['data'])} questions"
        )

        return dialog


class SpokenQuestionAnsweringAnnotator(Annotator):
    """
    Annotator to convert a question answering task to a spoken question answering task.
    This annotator requires the QuestionAnsweringAnnotator to be applied before.
    The modality of the task is audio-to-text.
    This annotator is specific to the AudioDialog class.
    """

    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the spoken question answering task.
        :return: The modality of the spoken question answering task.
        :rtype: list[TaskModality]
        """
        return [
            TaskModality.AUDIO_TO_TEXT
        ]

    def get_task_name(self) -> str:
        """
        Get the name of the spoken question answering task.
        :return: The name of the spoken question answering task.
        :rtype: str
        """
        return "spoken_question_answering"

    def get_requirements(self) -> list[Annotator]:
        """
        Get the requirements for the spoken question answering task.
        :return: The requirements for the spoken question answering task.
        :rtype: list[Annotator]
        """
        return [
            QuestionAnsweringAnnotator()
        ]

    def annotate(self, dialog: Dialog) -> Dialog:
        """
        Annotate a dialog for question answering tasks.
        The annotation is a list of questions and answers, with the audio path and the voice used.
        :param dialog: The dialog to annotate.
        :type dialog: Dialog
        :return: The annotated dialog.
        :rtype: Dialog
        """
        logging.info("[SpokenQuestionAnsweringAnnotator] Annotating dialog for spoken question answering tasks")

        from sdialog.audio.dialog import AudioDialog

        if not isinstance(dialog, AudioDialog):
            raise ValueError("Dialog must be an instance of AudioDialog")

        dialog = self.check_requirements(dialog)

        _annotations = {
            "data": [],
            "modality": self._modality
        }

        # Iterate over the annotations of the question answering task
        for ann in dialog.get_annotations("question_answering")["data"]:

            # For now, we use a dummy audio path and voice.
            # TODO: Generate the audio for the question answering task.
            audio_path = "dummy_audio_path.wav"
            audio_voice = "af_default"

            # Add the annotation to the spoken question answering task
            _annotations["data"].append({
                "question": ann["question"],
                "answer": ann["answer"],
                "audio": audio_path,
                "voice": audio_voice
            })

        dialog.add_annotations(self._task_name, _annotations)

        logging.info(
            "[SpokenQuestionAnsweringAnnotator] "
            f"Annotation done for {len(_annotations['data'])} questions"
        )

        return dialog
