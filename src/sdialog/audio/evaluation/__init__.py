"""
Audio evaluation module.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from typing import Union, List, Dict, Any, Optional
from sdialog.audio.dialog import AudioDialog, RoomAcousticsConfig
from .base import BaseAudioDialogScore, Result
from .speech_signal import SpeechSignalEvaluator
from .audio_quality import AudioQualityEvaluator
from .speaker_consistency import SpeakerConsistency
import os
import json
import uuid
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """
    Structured container for evaluation results, including metadata.
    """
    dialog_id: str
    evaluator_name: str
    room_name: Optional[str] = None
    room_config: Optional[RoomAcousticsConfig] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


__all__ = [
    "evaluate",
    "Result",
    "BaseAudioDialogScore",
    "SpeakerConsistency",
    "SpeechSignalEvaluator",
    "AudioQualityEvaluator",
]


def evaluate(
    dialogs: Union[AudioDialog, List[AudioDialog]],
    evaluators: List[BaseAudioDialogScore],
    evaluator_args: Dict[str, Any] = None,
    compute_on_all_dialogs: bool = False,
    compute_plots: bool = False,
    save_path: str = None,
) -> Dict[str, any]:
    """
    Apply a list of audio evaluation functions to one or many dialogs.
    :param dialogs: The dialog or list of dialogs to evaluate.
    :type dialogs: Union[AudioDialog, List[AudioDialog]]
    :param evaluators: The list of evaluators to apply. Can be strings or instances.
    :type evaluators: List[Union[str, BaseAudioDialogScore]]
    :param evaluator_args: Optional dictionary of arguments for instantiating evaluators from strings.
    :type evaluator_args: Dict[str, Any]
    :param compute_on_all_dialogs: If True, compute the result on all dialogs.
    :type compute_on_all_dialogs: bool
    :param compute_plots: If True, compute the plots for the results.
    :type compute_plots: bool
    :param save_path: If provided, save the results to this path.
    :type save_path: str
    :return: A dictionary of evaluation results, keyed by evaluator name.
    :rtype: Dict[str, any]
    """
    if not isinstance(dialogs, list):
        dialogs = [dialogs]

    if evaluator_args is None:
        evaluator_args = {}

    per_dialog_results: Dict[str, Dict[str, List[Result]]] = {}
    overall_results: Dict[str, Result] = {}

    # Evaluate each evaluator on each dialog
    for evaluator in evaluators:
        evaluator_name = str(evaluator)
        evaluator_scores_by_dialog: Dict[str, Result] = {}

        # Evaluate the evaluator on each dialog
        for dialog in dialogs:

            # Generate a deterministic UUID from the dialog.
            unique_dialog_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(dialog)))
            print("unique_dialog_id", unique_dialog_id)
            print("dialog.id", dialog.id)

            dialog_scores = evaluator.score(dialog)
            evaluator_scores_by_dialog[unique_dialog_id] = dialog_scores

            if unique_dialog_id not in per_dialog_results.get(evaluator_name, {}):
                if evaluator_name not in per_dialog_results:
                    per_dialog_results[evaluator_name] = {}
                per_dialog_results[evaluator_name][unique_dialog_id] = []

            if dialog_scores and dialog_scores.metrics:
                for key, metrics_value in dialog_scores.metrics.items():
                    room_config = None
                    room_name = key

                    if (
                        dialog.audio_step_3_filepaths
                        and (
                            key.startswith("step_")
                            or key in dialog.audio_step_3_filepaths
                        )
                    ):
                        room_key = key.replace("step_", "")
                        if room_key in dialog.audio_step_3_filepaths:
                            room_config = dialog.audio_step_3_filepaths[room_key]
                            room_name = room_config.room_name
                        elif key == "step_1":
                            room_name = "TTS Audio (no room)"

                    per_room_result = Result(
                        metrics=metrics_value,
                        data=dialog_scores.data.get(key, {}),
                        plots=None,
                        dialog_id=dialog.id,
                        evaluator_name=evaluator_name,
                        room_name=room_name,
                        room_config=room_config,
                    )
                    per_dialog_results[evaluator_name][unique_dialog_id].append(per_room_result)

        # Compute the overall result of the evaluator on all dialogs
        if compute_on_all_dialogs:
            overall_results[evaluator_name] = evaluator.results2result(
                evaluator_scores_by_dialog, compute_plots=compute_plots
            )

        if save_path:
            evaluator_save_path = os.path.join(save_path, evaluator_name)
            os.makedirs(evaluator_save_path, exist_ok=True)

            dialog_results = per_dialog_results.get(evaluator_name, {})
            per_dialog_serializable = {
                dialog_id: [result.to_dict() for result in results_list]
                for dialog_id, results_list in dialog_results.items()
            }

            if per_dialog_serializable:
                p = os.path.join(evaluator_save_path, "per_dialog_results.json")
                with open(p, "w") as f:
                    json.dump(per_dialog_serializable, f, indent=4, default=str)

            if compute_on_all_dialogs and evaluator_name in overall_results:
                overall_result_obj = overall_results[evaluator_name]

                # Save plots and collect their paths
                plot_paths = []
                if overall_result_obj.plots:
                    plots_path = os.path.join(evaluator_save_path, "plots")
                    os.makedirs(plots_path, exist_ok=True)
                    for i, fig in enumerate(overall_result_obj.plots):
                        plot_filepath = os.path.join(plots_path, f"plot_{i+1}.png")
                        fig.savefig(plot_filepath)
                        plot_paths.append(plot_filepath)

                # Replace figure objects with paths for serialization
                overall_result_obj.plots = plot_paths

                # Save overall metrics
                overall_serializable = overall_result_obj.to_dict()
                with open(os.path.join(evaluator_save_path, "overall_results.json"), "w") as f:
                    json.dump(overall_serializable, f, indent=4, default=str)

    return {
        "per_dialog_results": per_dialog_results,
        "overall_results": overall_results,
    }
