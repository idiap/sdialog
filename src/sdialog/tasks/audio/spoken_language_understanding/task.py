"""
This module contains the classes for the audio-specific tasks.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import json
import logging
import librosa
import soundfile as sf
from sdialog import Dialog
from sdialog.config import config
from pydantic import BaseModel, Field
from sdialog.util import get_llm_model
from typing import Any, Optional, Type, List
from sdialog.tasks import Task, TaskModality
from sdialog.audio.dialog import AudioDialog
from langchain_core.messages import HumanMessage, SystemMessage


class SpokenLanguageUnderstandingTask(Task):
    """
    Task to perform Spoken Language Understanding (SLU) on a dialog.
    This involves identifying the user's intent and extracting relevant slots from audio.
    The modality of the task is audio-to-text.
    """

    def __init__(self, intents: List[str] = None, slots: List[str] = None):
        super().__init__()
        self.intents = intents or [
            "inform_symptom",
            "provide_clarification",
            "answer_question_lifestyle",
            "ask_for_diagnosis",
            "affirm",
            "thank_you",
            "ask_for_clarification",
            "propose_procedure",
            "provide_diagnosis",
            "recommend_treatment"
        ]
        self.slots = slots or [
            "SYMPTOM",
            "DURATION",
            "ANATOMY",
            "LIFESTYLE_FACTOR",
            "SIGN",
            "TREATMENT",
            "PERSON"
        ]

    def get_structured_model(self) -> Optional[Type[BaseModel]]:
        """
        Defines the structured output for the Spoken Language Understanding task, including intent and slots.
        """
        class Slot(BaseModel):
            """Represents a single slot with its name and value."""
            slot_name: str = Field(description="The name of the slot.")
            slot_value: str = Field(description="The value of the slot.")

        class SpokenLanguageUnderstandingModel(BaseModel):
            """Represents the Spoken Language Understanding output for a single utterance."""
            intent: str = Field(description="The identified intent.")
            slots: List[Slot] = Field(description="A list of extracted slots.", default=[])

        return SpokenLanguageUnderstandingModel

    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the Spoken Language Understanding task.
        """
        return [TaskModality.AUDIO_TO_TEXT]

    def get_task_name(self) -> str:
        """
        Get the name of the Spoken Language Understanding task.
        """
        return "spoken_language_understanding"

    def get_requirements(self) -> list[Task]:
        """
        This task processes audio directly and does not have external task requirements.
        """
        return []

    def save(self, data: Any, args: dict[str, Any] = {}) -> None:
        """
        Save the Spoken Language Understanding annotations to a JSON file.
        """
        save_path = args.get("save_path")
        if not save_path:
            logging.warning("[SpokenLanguageUnderstandingTask] No 'save_path' provided, skipping saving.")
            return

        if not data:
            logging.info("[SpokenLanguageUnderstandingTask] No data to save, skipping file creation.")
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"[SpokenLanguageUnderstandingTask] Spoken Language Understanding annotations saved to {save_path}")
        except Exception as e:
            logging.error(f"[SpokenLanguageUnderstandingTask] Failed to save Spoken Language Understanding annotations: {e}")

    def run(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Run the Spoken Language Understanding task on a dialog.
        """

        if not isinstance(dialog, AudioDialog):
            raise ValueError("Dialog must be an instance of AudioDialog for SpokenLanguageUnderstandingTask.")

        logging.info("[SpokenLanguageUnderstandingTask] Running dialog for Spoken Language Understanding task")

        if not isinstance(dialog, AudioDialog):
            raise ValueError("Dialog must be an instance of AudioDialog for SpokenLanguageUnderstandingTask.")

        if "output_dir" not in args or not args["output_dir"]:
            raise ValueError("'output_dir' is required in args to save audio segments.")

        dialog = self.check_requirements(dialog)

        if "room_audio_path" in args and args["room_audio_path"]:
            main_audio_path = args["room_audio_path"]
        elif dialog.audio_step_3_filepaths:
            main_audio_path = list(dialog.audio_step_3_filepaths.values())[0]["audio_path"]
            logging.warning(
                "[SpokenLanguageUnderstandingTask] No 'room_audio_path' provided, using the first room audio path found in the dialog."
            )
        else:
            raise ValueError("No audio file path found in the dialog or arguments.")

        wav_file, sampling_rate = librosa.load(main_audio_path, sr=None)

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
            "You are an expert in Spoken Language Understanding. "
            "Your task is to identify the user's intent and extract relevant slots from their utterance. "
            f"Possible intents are: {', '.join(self.intents)}. "
            f"Possible slots are: {', '.join(self.slots)}. "
            "Your response must be a JSON object with 'intent' and 'slots' keys."
        )

        slu_annotations = []
        for turn_id, turn in enumerate(dialog.turns):
            transcription = turn.text
            if not transcription:
                logging.warning(f"[SpokenLanguageUnderstandingTask] No transcription found for turn {turn_id}, skipping.")
                continue

            start_time = turn.audio_start_time
            end_time = turn.audio_start_time + turn.audio_duration
            segment_data = wav_file[int(start_time * sampling_rate):int(end_time * sampling_rate)]

            segment_dir = os.path.join(args['output_dir'], dialog.id)
            os.makedirs(segment_dir, exist_ok=True)
            segment_path = os.path.join(segment_dir, f"utterance_{dialog.id}_{turn_id}.wav")

            sf.write(segment_path, segment_data, sampling_rate)

            human_prompt = (
                f"Please perform Spoken Language Understanding on the following utterance:\n---\n{transcription}\n---\n\n"
                "Identify the intent and extract all relevant slots and their values. "
                "The output must be a JSON object."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            try:
                raw_response = llm.invoke(messages)
                structured_response = structured_model.model_validate(raw_response)
                persona = dialog.personas.get(turn.speaker, {})

                annotation = {
                    "dialog_id": dialog.id,
                    "turn_id": turn_id,
                    "audio_path": segment_path,
                    "transcription": transcription,
                    "speaker_info": {
                        "voice": persona.get("voice").identifier if persona.get("voice") else None,
                        "gender": persona.get("gender"),
                        "age": persona.get("age"),
                        "language": persona.get("language"),
                    },
                    "intent": structured_response.intent,
                    "slots": [s.model_dump() for s in structured_response.slots],
                }
                slu_annotations.append(annotation)

            except Exception as e:
                logging.error(f"[SpokenLanguageUnderstandingTask] Failed to process turn {turn_id}: {e}")

        annotations = {
            "data": slu_annotations,
            "modality": self.get_modality()
        }

        dialog.add_annotations(self.get_task_name(), annotations)
        logging.info("[SpokenLanguageUnderstandingTask] Spoken Language Understanding annotation completed.")

        # Save the annotations inside the dialog-specific folder
        segment_dir = os.path.join(args['output_dir'], dialog.id)
        save_path = os.path.join(segment_dir, "slu_annotations.json")
        save_args = args.copy()
        save_args['save_path'] = save_path
        self.save(data=slu_annotations, args=save_args)

        return dialog
