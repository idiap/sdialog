"""
This module contains the classes for the NLP-specific tasks.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import logging
from sdialog import Dialog
from sdialog.config import config
from pydantic import BaseModel, Field
from sdialog.util import get_llm_model
from typing import List, Optional, Any, Type
from sdialog.tasks import Task, TaskModality
from langchain_core.messages import HumanMessage, SystemMessage


class QuestionAnsweringTask(Task):
    """
    Task to annotate a dialog for question answering tasks.
    The annotation is a list of questions and answers.
    This task has no requirements.
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

    def get_requirements(self) -> list[Task]:
        """
        Get the requirements for the question answering task.
        :return: The requirements for the question answering task.
        :rtype: list[Task]
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
            logging.warning("[QuestionAnsweringTask] No 'save_path' provided, skipping saving")
            return

        if not data:
            logging.info("[QuestionAnsweringTask] No annotations to save, skipping file creation.")
            return

        df = pd.DataFrame(data)
        df["identifier"] = df.index
        df = df[["identifier", "question", "answer"]]
        df.to_csv(args["save_path"], index=False)

        logging.info(
            "[QuestionAnsweringTask] "
            f"Data saved to {args['save_path']}"
        )

    def run(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Run the question answering task on a dialog.
        The annotation is a list of questions and answers.
        :param dialog: The dialog to annotate.
        :type dialog: Dialog
        :param args: Additional arguments to pass to the run method.
        This includes the 'save_path' where to save the data and can contain 'save_args'
        additional arguments to pass to the save method.
        :type args: dict[str, Any]
        :return: The annotated dialog.
        :rtype: Dialog
        """
        logging.info("[QuestionAnsweringTask] Running dialog for question answering tasks")

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
            logging.info(f"[QuestionAnsweringTask] Raw LLM response: {raw_response}")
            structured_response = self._structured_model.model_validate(raw_response)
            data = [pair.model_dump() for pair in structured_response.questions_answers]
        except Exception as e:
            logging.error(f"[QuestionAnsweringTask] Failed to generate annotations: {e}")
            data = []

        _annotations = {
            "data": data,
            "modality": self._modality
        }

        dialog.add_annotations(self._task_name, _annotations)

        logging.info(
            "[QuestionAnsweringTask] "
            f"Annotation done for {len(_annotations['data'])} questions"
        )

        self.save(data=_annotations["data"], args=args)

        return dialog
