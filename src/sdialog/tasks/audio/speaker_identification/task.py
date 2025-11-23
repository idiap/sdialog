"""
This module contains the classes for the audio speaker identification task.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import logging
import json
import librosa
import soundfile as sf
from sdialog import Dialog
from typing import Any
from sdialog.tasks import Task, TaskModality
from sdialog.audio.dialog import AudioDialog


class SpeakerIdentificationTask(Task):
    """
    Task to generate a ground-truth speaker identification dataset from a dialog's audio.
    It uses the turn timings and speaker information within the AudioDialog object.
    The output is saved in a format similar to VoxCeleb, with audio segments
    organized into speaker-specific directories.
    """

    def __init__(self):
        super().__init__()

    def get_modality(self) -> list[TaskModality]:
        """
        Get the modality of the speaker identification task.
        """
        return [TaskModality.AUDIO_TO_TEXT]

    def get_task_name(self) -> str:
        """
        Get the name of the speaker identification task.
        """
        return "speaker_identification"

    def get_requirements(self) -> list[Task]:
        """
        This task processes audio metadata directly and does not have external task requirements.
        """
        return []

    def save(self, data: Any, args: dict[str, Any] = {}) -> None:
        """
        Save speaker identification metadata.
        In this task, audio files are saved directly in the run method.
        This method could be used to save an additional metadata file if needed.
        """
        save_path = args.get("save_path")
        if not save_path:
            logging.warning("[SpeakerIdentificationTask] No 'save_path' provided, skipping saving.")
            return

        if not data:
            logging.info("[SpeakerIdentificationTask] No data to save, skipping file creation.")
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for line in data:
                    f.write(f"{line}\n")
            logging.info(f"[SpeakerIdentificationTask] Metadata saved to: {save_path}")
        except Exception as e:
            logging.error(f"[SpeakerIdentificationTask] Failed to save metadata file: {e}")

    def run(self, dialog: Dialog, args: dict[str, Any] = {}) -> Dialog:
        """
        Run the speaker identification task on a dialog.
        """
        logging.info("[SpeakerIdentificationTask] Generating ground-truth speaker identification dataset from dialog")

        if not isinstance(dialog, AudioDialog):
            raise ValueError("Dialog must be an instance of AudioDialog for SpeakerIdentificationTask.")

        if "output_dir" not in args or not args["output_dir"]:
            raise ValueError("'output_dir' is required in args to save audio and annotations.")

        # Find the main audio file path
        if "room_audio_path" in args and args["room_audio_path"]:
            main_audio_path = args["room_audio_path"]
        elif dialog.audio_step_3_filepaths:
            main_audio_path = list(dialog.audio_step_3_filepaths.values())[0]["audio_path"]
        else:
            raise ValueError("No audio file path found for speaker identification in the dialog or arguments.")

        try:
            waveform, sample_rate = librosa.load(main_audio_path, sr=None, mono=False)
            if waveform.ndim == 1:
                waveform = waveform[None, :]  # Ensure waveform is 2D for consistent slicing
        except Exception as e:
            logging.error(f"[SpeakerIdentificationTask] Failed to load audio file: {e}")
            return dialog

        annotations_data = []
        metadata_lines = []

        dialog_output_dir = os.path.join(args['output_dir'], dialog.id)
        os.makedirs(dialog_output_dir, exist_ok=True)

        speaker_map = {}
        speaker_count = 0

        for i, turn in enumerate(dialog.turns):
            start_time = turn.audio_start_time
            duration = turn.audio_duration
            original_speaker_id = turn.speaker

            if start_time is None or duration is None or original_speaker_id is None:
                logging.warning(f"Skipping turn with missing audio metadata: {turn}")
                continue

            if original_speaker_id not in speaker_map:
                speaker_map[original_speaker_id] = f"SPEAKER_{speaker_count:02d}"
                speaker_count += 1

            generic_speaker_id = speaker_map[original_speaker_id]

            # Create speaker directory
            speaker_dir = os.path.join(dialog_output_dir, str(generic_speaker_id))
            os.makedirs(speaker_dir, exist_ok=True)

            # Segment audio
            start_frame = int(start_time * sample_rate)
            end_frame = int((start_time + duration) * sample_rate)
            segment = waveform[:, start_frame:end_frame]

            # Save segment
            segment_filename = f"turn_{i:04d}.wav"
            segment_path = os.path.join(speaker_dir, segment_filename)
            try:
                sf.write(segment_path, segment.T, sample_rate)
            except Exception as e:
                logging.error(f"[SpeakerIdentificationTask] Failed to save audio segment: {e}")
                continue

            # For annotations and metadata
            annotations_data.append({
                "speaker": generic_speaker_id,
                "audio_path": segment_path,
            })
            metadata_lines.append(f"{generic_speaker_id} {segment_path}")

        annotations = {
            "data": annotations_data,
            "modality": self.get_modality()
        }

        dialog.add_annotations(self.get_task_name(), annotations)
        logging.info("[SpeakerIdentificationTask] Speaker identification annotation completed.")

        # Save metadata file
        metadata_filename = f"{dialog.id}_speaker_identification.txt"
        metadata_save_path = os.path.join(dialog_output_dir, metadata_filename)
        self.save(data=metadata_lines, args={'save_path': metadata_save_path})

        # Save speaker mapping
        mapping_filename = f"{dialog.id}_speaker_mapping.json"
        mapping_save_path = os.path.join(dialog_output_dir, mapping_filename)
        try:
            with open(mapping_save_path, "w", encoding="utf-8") as f:
                json.dump(speaker_map, f, indent=4)
            logging.info(f"[SpeakerIdentificationTask] Speaker mapping saved to: {mapping_save_path}")
        except Exception as e:
            logging.error(f"[SpeakerIdentificationTask] Failed to save speaker mapping: {e}")

        return dialog
