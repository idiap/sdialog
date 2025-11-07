"""
This module contains the classes for the task-specific annotators.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import logging
from sdialog import Dialog
from sdialog.config import config
from pydantic import BaseModel, Field
from sdialog.util import get_llm_model
from typing import List, Optional, Any, Tuple, Type
from langchain_core.messages import HumanMessage, SystemMessage
from sdialog.annotators.annotator import Annotator, TaskModality


def apply_annotators(dialog: Dialog, annotators: List[Tuple[Annotator, dict[str, Any]]]) -> Dialog:
    """
    Apply a list of annotators to a dialog.
    Each annotator is applied in the order of the list.
    The dialog is modified in place.
    If an annotator has requirements, they are checked before applying the annotator.

    :param dialog: The dialog to annotate.
    :type dialog: Dialog
    :param annotators: The list of annotators to apply.
    :type annotators: List[Tuple[Annotator, dict[str, Any]]]
    :return: The annotated dialog.
    :rtype: Dialog
    """

    for annotator, args in annotators:
        dialog = annotator.annotate(dialog, args=args)

    return dialog


class QuestionAnsweringAnnotator(Annotator):
    """
    Annotator to annotate a dialog for question answering tasks.
    The annotation is a list of questions and answers.
    This annotator has no requirements.
    The modality of the task is text-to-text.
    """

    def get_structured_model(self) -> Optional[Type[BaseModel]]:
        """
        Get the structured model for the parsing of the data for the task during LLM inference.
        The structured model is a pydantic model that defines the structure of the data for the task,
        so that the LLM can parse the data correctly.
        :return: The structured model for the task.
        :rtype: BaseModel
        """
        class QAPair(BaseModel):
            """Represents a single question and its corresponding answer derived from the dialogue."""
            question: str
            answer: str

        class QuestionAnswerList(BaseModel):
            """A list of question and answer pairs extracted from a dialogue."""
            questions_answers: List[QAPair] = Field(default_factory=list)

        return QuestionAnswerList

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

    def save(self, data: Any, args: dict[str, Any] = {}) -> None:
        """
        Save the data to a file. An 'identifier' column will be added with incremental values.
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
            logging.warning("[QuestionAnsweringAnnotator] No 'save_path' provided, skipping saving")
            return

        if not data:
            logging.info("[QuestionAnsweringAnnotator] No annotations to save, skipping file creation.")
            return

        df = pd.DataFrame(data)
        df["identifier"] = df.index
        df = df[["identifier", "question", "answer"]]
        df.to_csv(args["save_path"], index=False)

        logging.info(
            "[QuestionAnsweringAnnotator] "
            f"Data saved to {args['save_path']}"
        )

    def annotate(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Annotate a dialog for question answering tasks.
        The annotation is a list of questions and answers.
        :param dialog: The dialog to annotate.
        :type dialog: Dialog
        :param args: Additional arguments to pass to the annotate method.
        This includes the 'save_path' where to save the data and can contain 'save_args'
        additional arguments to pass to the save method.
        :type args: dict[str, Any]
        :return: The annotated dialog.
        :rtype: Dialog
        """
        logging.info("[QuestionAnsweringAnnotator] Annotating dialog for question answering tasks")

        dialog = self.check_requirements(dialog)

        llm_params = config["llm"].copy()
        if "model" in llm_params:
            del llm_params["model"]

        llm = get_llm_model(
            model_name=config["llm"]["model"],
            output_format=self._structured_model,
            **llm_params,
        )

        system_prompt = "You are an expert at creating question-answer pairs from a dialogue. " \
                        "Your response must be a JSON object, and it must contain at least one question-answer pair."

        dialog_text = "\n".join([f"{turn.speaker}: {turn.text}" for turn in dialog.turns])

        human_prompt = (
            f"Here is a dialogue:\n---\n{dialog_text}\n---\n\n"
            "Based on the dialogue, generate a list of question and answer pairs. "
            "It is mandatory to generate at least one question-answer pair. "
            "If the dialogue contains little information, create a general question about the conversation theme. "
            'The output should be a JSON object with a key "questions_answers". '
            "This key should contain a list of objects, where each object has "
            'two keys: "question" and "answer".\n\n'
            "Example format:\n"
            "{\n"
            '  "questions_answers": [\n'
            "    {\n"
            '      "question": "What is the main topic of the conversation?",\n'
            '      "answer": "The main topic is a general discussion about a recent event."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Now, generate the question and answer pairs for the provided dialogue."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            raw_response = llm.invoke(messages)
            logging.info(f"[QuestionAnsweringAnnotator] Raw LLM response: {raw_response}")
            structured_response = self._structured_model.model_validate(raw_response)
            data = [pair.model_dump() for pair in structured_response.questions_answers]
        except Exception as e:
            logging.error(f"[QuestionAnsweringAnnotator] Failed to generate annotations: {e}")
            data = []

        _annotations = {
            "data": data,
            "modality": self._modality
        }

        dialog.add_annotations(self._task_name, _annotations)

        logging.info(
            "[QuestionAnsweringAnnotator] "
            f"Annotation done for {len(_annotations['data'])} questions"
        )

        self.save(data=_annotations["data"], args=args)

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


class SummaryAnnotator(Annotator):
    """
    Annotator to annotate a dialog for summarization tasks.
    The annotation is a string containing the summary.
    This annotator has no requirements.
    The modality of the task is text-to-text.
    """

    def get_structured_model(self) -> Optional[Type[BaseModel]]:
        """
        Get the structured model for the parsing of the data for the task during LLM inference.
        """
        class SummaryModel(BaseModel):
            """Represents a summary of the dialogue."""
            summary: str = Field(default="")

        return SummaryModel

    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the summarization task.
        """
        return [
            TaskModality.TEXT_TO_TEXT
        ]

    def get_task_name(self) -> str:
        """
        Get the name of the summarization task.
        """
        return "summarization"

    def get_requirements(self) -> list[Annotator]:
        """
        Get the requirements for the summarization task.
        """
        return []

    def save(self, data: Any, args: dict[str, Any] = {}) -> None:
        """
        Save the data to a file.
        :param data: The data to save (summary string).
        :param args: Additional arguments, including 'save_path'.
        """
        if args is None or "save_path" not in args or args["save_path"] is None:
            logging.warning("[SummaryAnnotator] No 'save_path' provided, skipping saving")
            return

        if not data:
            logging.info("[SummaryAnnotator] No summary to save, skipping file creation.")
            return

        save_path = args["save_path"]
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(data)
            logging.info(f"[SummaryAnnotator] Summary saved to {save_path}")
        except Exception as e:
            logging.error(f"[SummaryAnnotator] Failed to save summary: {e}")

    def annotate(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Annotate a dialog for summarization tasks.
        """
        logging.info("[SummaryAnnotator] Annotating dialog for summarization tasks")

        dialog = self.check_requirements(dialog)

        llm_params = config["llm"].copy()
        if "model" in llm_params:
            del llm_params["model"]

        llm = get_llm_model(
            model_name=config["llm"]["model"],
            output_format=self._structured_model,
            **llm_params,
        )

        system_prompt = "You are an expert at summarizing dialogues. Your response must be a JSON object with a 'summary' key."

        dialog_text = "\n".join([f"{turn.speaker}: {turn.text}" for turn in dialog.turns])

        human_prompt = (
            f"Please provide a concise summary of the following dialogue:\n---\n{dialog_text}\n---\n\n"
            "The summary should capture the main points and key information from the conversation while being short and concise.\n"
            "The output must be a JSON object with a single key 'summary' "
            "containing the summary text."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            raw_response = llm.invoke(messages)
            logging.info(f"[SummaryAnnotator] Raw LLM response: {raw_response}")
            structured_response = self._structured_model.model_validate(raw_response)
            data = structured_response.summary
        except Exception as e:
            logging.error(f"[SummaryAnnotator] Failed to generate summary: {e}")
            data = ""

        _annotations = {
            "data": data,
            "modality": self._modality
        }

        dialog.add_annotations(self._task_name, _annotations)

        logging.info("[SummaryAnnotator] Annotation done for summarization.")

        self.save(data=_annotations["data"], args=args)

        return dialog
