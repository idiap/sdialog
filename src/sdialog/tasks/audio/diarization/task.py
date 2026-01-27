"""
This module contains the classes for the audio diarization task.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import logging
import json
import shutil
from sdialog import Dialog
from typing import Any
from sdialog.tasks import Task, TaskModality
from sdialog.audio.dialog import AudioDialog


class DiarizationTask(Task):
    """
    Task to generate a ground-truth speaker diarization file from a dialog's audio metadata.
    It uses the turn timings and speaker information within the AudioDialog object.
    The output is saved in the Rich Transcription Time Marked (RTTM) format.
    """

    def __init__(self):
        super().__init__()

    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the diarization task.
        """
        return [TaskModality.AUDIO_TO_TEXT]

    def get_task_name(self) -> str:
        """
        Get the name of the diarization task.
        """
        return "diarization"

    def get_requirements(self) -> list[Task]:
        """
        This task processes audio metadata directly and does not have external task requirements.
        """
        return []

    def save(self, data: Any, args: dict[str, Any] = {}) -> None:
        """
        Save the diarization data to an RTTM file.
        """
        save_path = args.get("save_path")
        if not save_path:
            logging.warning("[DiarizationTask] No 'save_path' provided, skipping saving.")
            return

        if not data:
            logging.info("[DiarizationTask] No data to save, skipping file creation.")
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(data)
            logging.info(f"[DiarizationTask] Diarization saved to RTTM file: {save_path}")
        except Exception as e:
            logging.error(f"[DiarizationTask] Failed to save RTTM file: {e}")

    def run(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Run the diarization task on a dialog.
        """
        logging.info("[DiarizationTask] Generating ground-truth diarization from dialog")

        if not isinstance(dialog, AudioDialog):
            raise ValueError("Dialog must be an instance of AudioDialog for DiarizationTask.")

        if "output_dir" not in args or not args["output_dir"]:
            raise ValueError("'output_dir' is required in args to save audio and annotations.")

        # Find the main audio file path
        if "room_audio_path" in args and args["room_audio_path"]:
            main_audio_path = args["room_audio_path"]
        elif dialog.audio_step_3_filepaths:
            main_audio_path = list(dialog.audio_step_3_filepaths.values())[0].audio_path
        elif dialog.audio_step_1_filepath:
            logging.warning(
                "[DiarizationTask] No 'room_audio_path' or 'audio_step_3_filepaths' provided, "
                "using the anechoic audio (tts only) path found in the dialog."
            )
            main_audio_path = dialog.audio_step_1_filepath
        else:
            raise ValueError("No audio path found in the dialog.")

        rttm_lines = []
        annotations_data = []

        file_id = dialog.id
        speaker_map = {}
        speaker_count = 0

        for turn in dialog.turns:
            start_time = turn.audio_start_time
            duration = turn.audio_duration
            end_time = start_time + duration
            original_speaker_id = turn.speaker

            if start_time is None or duration is None or original_speaker_id is None:
                logging.warning(f"Skipping turn with missing audio metadata: {turn}")
                continue

            if original_speaker_id not in speaker_map:
                speaker_map[original_speaker_id] = f"SPEAKER_{speaker_count:02d}"
                speaker_count += 1

            generic_speaker_id = speaker_map[original_speaker_id]

            # RTTM format: type, file_id, channel, start, duration, <NA>, <NA>, speaker_id, <NA>, <NA>
            rttm_line = f"SPEAKER {file_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {generic_speaker_id} <NA> <NA>"
            rttm_lines.append(rttm_line)

            annotations_data.append({
                "start": start_time,
                "end": end_time,
                "speaker": generic_speaker_id,
            })

        rttm_data = "\n".join(rttm_lines)

        annotations = {
            "data": annotations_data,
            "modality": self.get_modality()
        }

        dialog.add_annotations(self.get_task_name(), annotations)
        logging.info("[DiarizationTask] Diarization annotation completed.")

        # Create a dedicated output directory for the dialog
        dialog_output_dir = os.path.join(args['output_dir'], dialog.id)
        os.makedirs(dialog_output_dir, exist_ok=True)

        # Save the .rttm file inside the dialog-specific directory
        rttm_filename = f"{dialog.id}_diarization.rttm"
        rttm_save_path = os.path.join(dialog_output_dir, rttm_filename)
        self.save(data=rttm_data, args={'save_path': rttm_save_path})

        # Save speaker mapping
        mapping_filename = f"{dialog.id}_speaker_mapping.json"
        mapping_save_path = os.path.join(dialog_output_dir, mapping_filename)
        try:
            with open(mapping_save_path, "w", encoding="utf-8") as f:
                json.dump(speaker_map, f, indent=4)
            logging.info(f"[DiarizationTask] Speaker mapping saved to: {mapping_save_path}")
        except Exception as e:
            logging.error(f"[DiarizationTask] Failed to save speaker mapping: {e}")

        # Copy the main audio file into the same directory
        try:
            audio_filename = f"{dialog.id}.wav"
            dest_audio_path = os.path.join(dialog_output_dir, audio_filename)
            shutil.copy(main_audio_path, dest_audio_path)
            logging.info(f"Copied main audio file to {dest_audio_path}")
        except Exception as e:
            logging.error(f"Failed to copy audio file: {e}")

        return dialog
