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
from typing import Any, Optional, Type
from sdialog.tasks import Task, TaskModality
from langchain_core.messages import HumanMessage, SystemMessage


class SummaryTask(Task):
    """
    Task to annotate a dialog for summarization tasks.
    The annotation is a string containing the summary.
    This task has no requirements.
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

    def get_requirements(self) -> list[Task]:
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
            logging.warning("[SummaryTask] No 'save_path' provided, skipping saving")
            return

        if not data:
            logging.info("[SummaryTask] No summary to save, skipping file creation.")
            return

        save_path = args["save_path"]
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(data)
            logging.info(f"[SummaryTask] Summary saved to {save_path}")
        except Exception as e:
            logging.error(f"[SummaryTask] Failed to save summary: {e}")

    def run(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Run the summarization task on a dialog.
        """
        logging.info("[SummaryTask] Running dialog for summarization tasks")

        dialog = self.check_requirements(dialog)

        llm_params = config["llm"].copy()
        if "model" in llm_params:
            del llm_params["model"]

        llm = get_llm_model(
            model_name=config["llm"]["model"],
            output_format=self._structured_model,
            **llm_params,
        )

        system_prompt = (
            "You are an expert at summarizing dialogues. "
            "Your response must be a JSON object with a 'summary' key."
        )

        dialog_text = "\n".join([f"{turn.speaker}: {turn.text}" for turn in dialog.turns])

        human_prompt = (
            f"Please provide a concise summary of the following dialogue:\n---\n{dialog_text}\n---\n\n"
            "The summary should capture the main points and key information "
            "from the conversation while being short and concise.\n"
            "The output must be a JSON object with a single key 'summary' "
            "containing the summary text."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            raw_response = llm.invoke(messages)
            logging.info(f"[SummaryTask] Raw LLM response: {raw_response}")
            structured_response = self._structured_model.model_validate(raw_response)
            data = structured_response.summary
        except Exception as e:
            logging.error(f"[SummaryTask] Failed to generate summary: {e}")
            data = ""

        _annotations = {
            "data": data,
            "modality": self._modality
        }

        dialog.add_annotations(self._task_name, _annotations)

        logging.info("[SummaryTask] Annotation done for summarization.")

        self.save(data=_annotations["data"], args=args)

        return dialog
