"""
This module provides an audio evaluation metric for audio quality.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import logging
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
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        metrics_to_compute: Optional[Dict[str, bool]] = None,
        steps_to_evaluate: List[int] = [1, 3],
        compute_plots: bool = False,
        plot_type: str = "boxplot",
    ):
        super().__init__(name="audio-quality-evaluator")
        self.sample_rate = sample_rate
        self.steps_to_evaluate = steps_to_evaluate
        self.compute_plots = compute_plots
        self.plot_type = plot_type

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

        audio_steps = {
            1: dialog.audio_step_1_filepath,
        }

        # Handle multiple audio files from step 3
        if dialog.audio_step_3_filepaths:
            for i, (room_config, paths) in enumerate(dialog.audio_step_3_filepaths.items()):
                audio_steps[3 + i] = paths["audio_path"]

        for step in self.steps_to_evaluate:
            if step not in audio_steps or not os.path.exists(audio_steps[step]):
                continue

            audio = self._load_audio(audio_steps[step])
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
            if step > 2:
                config_name = list(dialog.audio_step_3_filepaths.keys())[step - 3]
                all_metrics[f"step_{config_name}"] = metrics
            else:
                all_metrics[f"step_{step}"] = metrics

        return Result(metrics=all_metrics, data=all_metrics, plots=[])

    def _load_audio(self, path):
        audio, sr = sf.read(path)
        if sr != self.sample_rate:
            if audio.ndim > 1:
                audio = audio.T
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            if audio.ndim > 1:
                audio = audio.T
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
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
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

        figs = []
        for metric_name, values in data.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            if self.plot_type == "boxplot":
                ax.boxplot(values)
            elif self.plot_type == "histogram":
                ax.hist(values, bins="auto")
            elif self.plot_type == "violin":
                ax.violinplot(values, showmeans=True)

            ax.set_title(f"Distribution of {metric_name}")
            ax.set_ylabel("Value")
            ax.set_xlabel(metric_name)
            plt.tight_layout()
            figs.append(fig)
            plt.close(fig)

        return figs
