"""
This module provides an audio evaluation metric for audio quality.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import torch
import librosa
import logging
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, Optional, List
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.evaluation.base import BaseAudioDialogScore, Result

logger = logging.getLogger(__name__)


class AudioQualityEvaluator(BaseAudioDialogScore):
    """
    Evaluator for audio quality using no-reference metrics.

    :param sample_rate: The sample rate of the audio.
    :type sample_rate: int
    :param metrics_to_compute: A dictionary of metrics to compute.
    :type metrics_to_compute: Optional[Dict[str, bool]]
    :param steps_to_evaluate: A list of audio generation steps to evaluate.
    :type steps_to_evaluate: List[int]
    :param compute_plots: Whether to compute plots.
    :type compute_plots: bool
    :param plot_type: The type of plot to generate.
    :type plot_type: str
    :param max_turns_to_evaluate: The maximum number of turns to evaluate.
    :type max_turns_to_evaluate: int
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        metrics_to_compute: Optional[Dict[str, bool]] = None,
        steps_to_evaluate: List[int] = [1, 3],
        compute_plots: bool = False,
        plot_type: str = "boxplot",
        max_turns_to_evaluate: int = 4
    ):
        super().__init__(name="audio-quality-evaluator")
        self.sample_rate = sample_rate
        self.steps_to_evaluate = steps_to_evaluate
        self.compute_plots = compute_plots
        self.plot_type = plot_type

        if max_turns_to_evaluate < 1:
            raise ValueError("max_turns_to_evaluate must be greater than 0")
        self.max_turns_to_evaluate = max_turns_to_evaluate

        default_metrics = {
            "nisqa": True,
            "dnsmos": True,
            "srmr": True,
        }

        if metrics_to_compute:
            self.metrics_to_compute = {**default_metrics, **metrics_to_compute}
        else:
            self.metrics_to_compute = default_metrics

        self._load_metrics()

    def _load_metrics(self):
        """Loads the torchmetrics objects."""
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
        from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

        if self.metrics_to_compute.get("srmr"):
            try:
                from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio
                self.srmr = SpeechReverberationModulationEnergyRatio(self.sample_rate)
            except (ImportError, ModuleNotFoundError):
                self.srmr = None
                warning_msg = (
                    "ModuleNotFoundError: speech_reverberation_modulation_energy_ratio requires you to have `gammatone`"
                    " and `torchaudio>=0.10` installed. Either install as ``pip install torchmetrics[audio]`` or"
                    " ``pip install torchaudio>=0.10`` and ``pip install git+https://github.com/detly/gammatone``"
                )
                logger.warning(warning_msg)
        else:
            self.srmr = None

        if self.metrics_to_compute.get("dnsmos"):
            try:
                self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(
                    self.sample_rate, False
                )
            except (ImportError, ModuleNotFoundError):
                self.dnsmos = None
                warning_msg = (
                    "ModuleNotFoundError: deep_noise_suppression_mean_opinion_score "
                    "requires you to have `onnxruntime` installed."
                    "Either install as ``pip install torchmetrics[audio]`` or ``pip install onnxruntime``"
                )
                logger.warning(warning_msg)
        else:
            self.dnsmos = None

        if self.metrics_to_compute.get("nisqa"):
            self.nisqa = NonIntrusiveSpeechQualityAssessment(self.sample_rate)
        else:
            self.nisqa = None

    def score(self, dialog: AudioDialog) -> Result:
        """
        Computes audio quality metrics for the given audio dialog.
        """
        all_metrics = {}

        # Compute N'th turn end time
        if self.max_turns_to_evaluate < 1 or self.max_turns_to_evaluate > len(dialog.turns):
            raise ValueError(
                "max_turns_to_evaluate must be greater than 0 and "
                "less than or equal to the number of turns in the dialog"
            )
        n_th_turn = dialog.turns[-self.max_turns_to_evaluate - 1]
        n_th_turn_end_time = n_th_turn.audio_start_time + n_th_turn.audio_duration

        audio_steps = {}
        if hasattr(dialog, 'audio_step_1_filepath') and dialog.audio_step_1_filepath:
            audio_steps[1] = dialog.audio_step_1_filepath

        room_configs = {}
        # Handle multiple audio files from step 3
        if dialog.audio_step_3_filepaths:
            for i, (room_config, paths) in enumerate(dialog.audio_step_3_filepaths.items()):
                step_idx = 3 + i
                audio_steps[step_idx] = paths.audio_path
                room_configs[step_idx] = room_config

        for step in sorted(audio_steps.keys()):
            # Determine if we should evaluate this step
            is_step_3_related = step >= 3
            should_evaluate = (
                step in self.steps_to_evaluate
                or (is_step_3_related and 3 in self.steps_to_evaluate)
            )

            if not should_evaluate or not os.path.exists(audio_steps[step]):
                continue

            audio = self._load_audio(audio_steps[step], n_th_turn_end_time=n_th_turn_end_time)
            metrics = {}

            if self.srmr:
                metrics[f"srmr|{step}"] = self.srmr(audio).item()
            if self.nisqa:
                metrics[f"nisqa-mos|{step}"] = self._calculate_nisqa_in_chunks(audio)
            if self.dnsmos:
                dnsmos_scores = self.dnsmos(audio)
                # handle different return types of dnsmos
                if isinstance(dnsmos_scores, dict):
                    metrics[f"dnsmos|{step}"] = dnsmos_scores['mos_pred'].item()
                else:
                    metrics[f"dnsmos|{step}"] = dnsmos_scores.mean().item()

            # Use a more descriptive key if there are multiple step 3 audios
            if step in room_configs:
                config_name = room_configs[step]
                all_metrics[f"step_{config_name}"] = metrics
            else:
                all_metrics[f"step_{step}"] = metrics

        return Result(metrics=all_metrics, data=all_metrics, plots=[])

    def _load_audio(self, path, n_th_turn_end_time: Optional[float] = None):
        audio, sr = sf.read(path)
        if sr != self.sample_rate:
            if audio.ndim > 1:
                audio = audio.T
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            if audio.ndim > 1:
                audio = audio.T

        if n_th_turn_end_time is not None:
            num_samples = int(n_th_turn_end_time * self.sample_rate)
            audio = audio[:num_samples]

        return torch.from_numpy(audio)

    def _calculate_nisqa_in_chunks(self, audio_tensor, chunk_duration_sec=10):
        chunk_size = int(chunk_duration_sec * self.sample_rate)
        if audio_tensor.dim() > 1:
            audio_tensor = torch.mean(audio_tensor, dim=1)
        total_len = audio_tensor.shape[0]
        if total_len < self.sample_rate:
            return None
        if total_len <= chunk_size:
            try:
                score = self.nisqa(audio_tensor)
                if isinstance(score, dict):
                    score = score["mos_pred"]
                return score.mean().item() if score.numel() > 1 else score.item()
            except (RuntimeError, IndexError):
                return None

        num_chunks = int(np.ceil(total_len / chunk_size))
        scores = []
        for i in range(num_chunks):
            chunk = audio_tensor[i * chunk_size:(i + 1) * chunk_size]
            if chunk.shape[0] < self.sample_rate:
                continue
            try:
                score = self.nisqa(chunk)
                if isinstance(score, dict):
                    score = score["mos_pred"]
                scores.extend(score.tolist() if score.numel() > 1 else [score.item()])
            except (RuntimeError, IndexError):
                continue
        return sum(scores) / len(scores) if scores else None

    def results2result(
        self, results: Dict[str, Any], compute_plots: bool = False
    ) -> Result:
        """
        Compute the overall result from the results of the evaluator on all dialogs.
        """
        aggregated_metrics = {}
        for result in results.values():
            for step_metrics in result.metrics.values():
                for metric_name, value in step_metrics.items():
                    if value is None:
                        continue
                    if metric_name not in aggregated_metrics:
                        aggregated_metrics[metric_name] = []
                    aggregated_metrics[metric_name].append(value)

        summary_metrics = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in aggregated_metrics.items()
        }

        plots = []
        if compute_plots or self.compute_plots:
            plots = self.plot(aggregated_metrics)

        return Result(metrics=summary_metrics, data=aggregated_metrics, plots=plots)

    def plot(self, data: Dict[str, Any]) -> list:
        """
        Plot the results of the evaluator.
        """
        if not data:
            return []

        grouped_data = defaultdict(dict)
        for metric_name, values in data.items():
            if '|' in metric_name:
                base_name, step = metric_name.split('|', 1)
                grouped_data[base_name][step] = values
            else:
                grouped_data[metric_name][metric_name] = values

        num_metrics = len(grouped_data)
        if num_metrics == 0:
            return []

        ncols = 2
        nrows = (num_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for i, (base_name, step_data) in enumerate(grouped_data.items()):
            ax = axes[i]

            # Aggregate values from all steps to show a single distribution
            all_values = [
                value for values_for_step in step_data.values() for value in values_for_step
            ]

            if not all_values:
                continue

            labels = [""]
            plot_values = [all_values]

            if self.plot_type == "boxplot":
                ax.boxplot(plot_values, labels=labels)
            elif self.plot_type == "violin":
                ax.violinplot(plot_values, showmeans=True)
                ax.set_xticks(np.arange(1, len(labels) + 1))
                ax.set_xticklabels(labels)
            elif self.plot_type == "histogram":
                ax.hist(all_values, bins="auto", alpha=0.75)

            ax.set_title(f"Distribution of {base_name}")
            ax.set_ylabel("Value")
            ax.set_xlabel("Aggregated across all steps")

        # Hide any unused subplots
        for j in range(num_metrics, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.close(fig)

        return [fig]
