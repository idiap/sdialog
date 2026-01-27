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
from typing import Any, Optional, Type, List
from sdialog.tasks import Task, TaskModality
from langchain_core.messages import HumanMessage, SystemMessage


class NamedEntityRecognitionTask(Task):
    """
    Task to annotate a dialog for Named Entity Recognition.
    The annotation is a list of entities found in the dialog.
    This task has no requirements.
    The modality of the task is text-to-text.
    """

    def __init__(self, tags: List[str] = None):
        super().__init__()
        self.tags = tags or ["PERSON", "LOCATION", "ORGANIZATION"]

    def get_structured_model(self) -> Optional[Type[BaseModel]]:
        """
        Get the structured model for the parsing of the data for the task during LLM inference.
        """
        class NamedEntity(BaseModel):
            """Represents a single named entity found in the text."""
            text: str = Field(description="The text of the named entity.")
            label: str = Field(description="The label of the named entity.")

        class NamedEntityRecognitionModel(BaseModel):
            """Represents the output of a Named Entity Recognition task."""
            entities: List[NamedEntity] = Field(description="A list of named entities found in the text.")

        return NamedEntityRecognitionModel

    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the Named Entity Recognition task.
        """
        return [
            TaskModality.TEXT_TO_TEXT
        ]

    def get_task_name(self) -> str:
        """
        Get the name of the Named Entity Recognition task.
        """
        return "named_entity_recognition"

    def get_requirements(self) -> list[Task]:
        """
        Get the requirements for the Named Entity Recognition task.
        """
        return []

    def save(self, data: Any, args: dict[str, Any] = {}) -> None:
        """
        Save the data to a file in BIO format.
        :param data: A dictionary containing the dialog and the entities.
        :param args: Additional arguments, including 'save_path'.
        """
        if args is None or "save_path" not in args or args["save_path"] is None:
            logging.warning("[NamedEntityRecognitionTask] No 'save_path' provided, skipping saving")
            return

        dialog = data.get("dialog")
        entities = data.get("entities")

        if not dialog or not entities:
            logging.info("[NamedEntityRecognitionTask] No data to save, skipping file creation.")
            return

        save_path = args["save_path"]

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for turn in dialog.turns:
                    tokens = turn.text.split()
                    labels = ['O'] * len(tokens)

                    for entity in entities:
                        entity_tokens = entity.text.split()
                        num_entity_tokens = len(entity_tokens)
                        if num_entity_tokens == 0:
                            continue

                        # Find all occurrences of the entity in the turn
                        for i in range(len(tokens) - num_entity_tokens + 1):
                            if tokens[i:i + num_entity_tokens] == entity_tokens:
                                # Check if these tokens are not already tagged
                                if all(labels[j] == 'O' for j in range(i, i + num_entity_tokens)):
                                    labels[i] = f'B-{entity.label}'
                                    for j in range(1, num_entity_tokens):
                                        labels[i + j] = f'I-{entity.label}'

                    for token, label in zip(tokens, labels):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")  # Turn break
            logging.info(f"[NamedEntityRecognitionTask] BIO annotations saved to {save_path}")
        except Exception as e:
            logging.error(f"[NamedEntityRecognitionTask] Failed to save BIO annotations: {e}")

    def run(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Run the Named Entity Recognition task on a dialog.
        """
        logging.info("[NamedEntityRecognitionTask] Running dialog for Named Entity Recognition tasks")

        dialog = self.check_requirements(dialog)

        tags_str = ", ".join(self.tags)

        llm_params = config["llm"].copy()
        if "model" in llm_params:
            del llm_params["model"]

        structured_model = self.get_structured_model()
        llm = get_llm_model(
            model_name=config["llm"]["model"],
            output_format=structured_model,
            **llm_params,
        )

        system_prompt = (
            f"You are an expert at Named Entity Recognition. "
            f"Extract entities with the following tags: {tags_str}. "
            "Your response must be a JSON object with an 'entities' key, which is a list of objects, "
            "each with 'text' and 'label'."
        )

        dialog_text = "\n".join([f"{turn.speaker}: {turn.text}" for turn in dialog.turns])

        human_prompt = (
            f"Please extract named entities from the following dialogue:\n---\n{dialog_text}\n---\n\n"
            f"The entities should have one of these labels: {tags_str}.\n"
            "The output must be a JSON object with a single key 'entities' containing a list of found entities."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            raw_response = llm.invoke(messages)
            logging.info(f"[NamedEntityRecognitionTask] Raw LLM response: {raw_response}")
            structured_response = structured_model.model_validate(raw_response)
            data = structured_response.entities
        except Exception as e:
            logging.error(f"[NamedEntityRecognitionTask] Failed to generate NER data: {e}")
            data = []

        _annotations = {
            "data": data,
            "modality": self._modality
        }

        dialog.add_annotations(self._task_name, _annotations)

        logging.info("[NamedEntityRecognitionTask] Annotation done for Named Entity Recognition.")

        data_to_save = {"dialog": dialog, "entities": data}
        self.save(data=data_to_save, args=args)

        return dialog
