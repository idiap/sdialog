"""
This module contains the classes for the audio-specific annotators.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import logging
from typing import Any
from sdialog import Dialog
from sdialog.annotators import Annotator, TaskModality
from sdialog.annotators.nlp import QuestionAnsweringAnnotator


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
        import pandas as pd

        if args is None or "save_path" not in args or args["save_path"] is None:
            logging.warning("[SpokenQuestionAnsweringAnnotator] No 'save_path' provided, skipping saving")
            return

        if not data:
            logging.info("[SpokenQuestionAnsweringAnnotator] No annotations to save, skipping file creation.")
            return

        df = pd.DataFrame(data)
        df["identifier"] = df.index
        df = df[["identifier", "question", "answer", "audio_path", "voice"]]
        df.to_csv(args["save_path"], index=False)

        logging.info(
            "[SpokenQuestionAnsweringAnnotator] "
            f"Data saved to {args['save_path']}"
        )

    def annotate(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Annotate a dialog for question answering tasks.
        The annotation is a list of questions and answers, with the audio path and the voice used.
        :param dialog: The dialog to annotate.
        :type dialog: Dialog
        :param args: Additional arguments to pass to the annotate method.
        This includes the 'save_path' where to save the data and can contain 'save_args'
        additional arguments to pass to the save method.
        :type args: dict[str, Any]
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
                "audio_path": audio_path,
                "voice": audio_voice
            })

        dialog.add_annotations(self._task_name, _annotations)

        logging.info(
            "[SpokenQuestionAnsweringAnnotator] "
            f"Annotation done for {len(_annotations['data'])} questions"
        )

        self.save(data=_annotations["data"], args=args)

        return dialog
