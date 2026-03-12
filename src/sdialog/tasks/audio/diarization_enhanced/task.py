"""
This module contains the classes for the audio diarization task.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import numpy as np
import re
import os
import logging
import json
import shutil
from sdialog import Dialog
from typing import Any
from sdialog.tasks import Task, TaskModality
from sdialog.audio.dialog import AudioDialog


class DiarizationEnhancedTask(Task):
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
            logging.warning("[DiarizationEnhancedTask] No 'save_path' provided, skipping saving.")
            return

        if not data:
            logging.info("[DiarizationEnhancedTask] No data to save, skipping file creation.")
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(data)
            logging.info(f"[DiarizationEnhancedTask] Diarization saved to RTTM file: {save_path}")
        except Exception as e:
            logging.error(f"[DiarizationEnhancedTask] Failed to save RTTM file: {e}")

    def run(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Run the diarization task on a dialog.
        """
        logging.info("[DiarizationEnhancedTask] Generating ground-truth diarization from dialog")

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
                "[DiarizationEnhancedTask] No 'room_audio_path' or 'audio_step_3_filepaths' provided, "
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

        # Load aligner model
        aligner = None
        model_name_alignment = args.get("model_name_alignment", "Qwen/Qwen3-ForcedAligner-0.6B")
        merge_threshold = args.get("merge_threshold", 0.1)

        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner

            logging.info(f"Loading {model_name_alignment} for diarization alignment...")
            aligner = Qwen3ForcedAligner.from_pretrained(
                model_name_alignment,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            )
        except ImportError:
            logging.warning("`qwen-asr` package not found. Using default turn timings.")
        except Exception as e:
            logging.error(f"Failed to load {model_name_alignment}: {e}. Using default turn timings.")

        for turn in dialog.turns:
            turn_start_time = turn.audio_start_time
            turn_duration = turn.audio_duration
            original_speaker_id = turn.speaker

            if turn_start_time is None or turn_duration is None or original_speaker_id is None:
                logging.warning(f"Skipping turn with missing audio metadata: {turn}")
                continue

            if original_speaker_id not in speaker_map:
                speaker_map[original_speaker_id] = f"SPEAKER_{speaker_count:02d}"
                speaker_count += 1

            generic_speaker_id = speaker_map[original_speaker_id]

            segments = []

            # Try alignment if model is loaded
            if aligner:
                try:
                    audio_data = turn.get_audio()
                    # Handle tensor vs numpy
                    if hasattr(audio_data, "numpy"):
                        audio_data = audio_data.numpy()

                    if audio_data is not None and len(audio_data) > 0:

                        # Clean text
                        clean_text = re.sub(r"\[(.*?)\]", "", turn.text)
                        clean_text = re.sub(r"\s+", " ", clean_text).strip()

                        if clean_text:
                            sr = turn.sampling_rate
                            audio_input = (audio_data.astype(np.float32), sr)

                            results = aligner.align(
                                audio=audio_input,
                                text=clean_text,
                                language="English"
                            )

                            if results and len(results) > 0:
                                alignment = results[0]

                                # Merge close segments (words)
                                current_segment_start = alignment[0].start_time
                                current_segment_end = alignment[0].end_time

                                for word in alignment[1:]:
                                    if word.start_time - current_segment_end < merge_threshold:
                                        current_segment_end = word.end_time
                                    else:
                                        segments.append((current_segment_start, current_segment_end))
                                        current_segment_start = word.start_time
                                        current_segment_end = word.end_time
                                segments.append((current_segment_start, current_segment_end))

                except Exception as e:
                    logging.warning(f"Alignment failed for turn {turn.speaker}: {e}")

            # Fallback if no segments found (aligner missing, failed, or empty text)
            if not segments:
                segments.append((0.0, turn_duration))

            for seg_start, seg_end in segments:
                abs_start = turn_start_time + seg_start
                abs_duration = seg_end - seg_start
                abs_end = abs_start + abs_duration

                # RTTM format: type, file_id, channel, start, duration, <NA>, <NA>, speaker_id, <NA>, <NA>
                rttm_line = (
                    f"SPEAKER {file_id} 1 {abs_start:.3f} "
                    f"{abs_duration:.3f} <NA> <NA> {generic_speaker_id} <NA> <NA>"
                )
                rttm_lines.append(rttm_line)

                annotations_data.append({
                    "start": abs_start,
                    "end": abs_end,
                    "speaker": generic_speaker_id,
                })

        rttm_data = "\n".join(rttm_lines)

        annotations = {
            "data": annotations_data,
            "modality": self.get_modality()
        }

        dialog.add_annotations(self.get_task_name(), annotations)
        logging.info("[DiarizationEnhancedTask] Diarization annotation completed.")

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
            logging.info(f"[DiarizationEnhancedTask] Speaker mapping saved to: {mapping_save_path}")
        except Exception as e:
            logging.error(f"[DiarizationEnhancedTask] Failed to save speaker mapping: {e}")

        # Copy the main audio file into the same directory
        try:
            audio_filename = f"{dialog.id}.wav"
            dest_audio_path = os.path.join(dialog_output_dir, audio_filename)
            shutil.copy(main_audio_path, dest_audio_path)
            logging.info(f"Copied main audio file to {dest_audio_path}")
        except Exception as e:
            logging.error(f"Failed to copy audio file: {e}")

        return dialog
