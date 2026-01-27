"""
This module provides an audio evaluation metric for speech signal analysis.
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
from typing import Dict, Any, Optional

from sdialog.audio.dialog import AudioDialog
from sdialog.audio.evaluation.base import BaseAudioDialogScore, Result

logger = logging.getLogger(__name__)


class SpeechSignalEvaluator(BaseAudioDialogScore):
    """
    Evaluator for audio augmentation pipelines.
    It computes various metrics between different steps of an audio generation pipeline.

    :param sample_rate: The sample rate of the audio.
    :type sample_rate: int
    :param metrics_to_compute: A dictionary of metrics to compute.
    :type metrics_to_compute: Optional[Dict[str, bool]]
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
        compute_plots: bool = False,
        plot_type: str = "boxplot",
        max_turns_to_evaluate: int = 4
    ):
        super().__init__(name="speech-signal-evaluator")
        self.sample_rate = sample_rate
        self.compute_plots = compute_plots
        self.plot_type = plot_type

        if max_turns_to_evaluate < 1:
            raise ValueError("max_turns_to_evaluate must be greater than 0")
        self.max_turns_to_evaluate = max_turns_to_evaluate

        default_metrics = {
            "stoi": True,
            "sdr": True,
            "pesq": True,
        }

        if metrics_to_compute:
            self.metrics_to_compute = {**default_metrics, **metrics_to_compute}
        else:
            self.metrics_to_compute = default_metrics

        self._load_metrics()

    def _load_metrics(self):
        """Loads the torchmetrics objects."""
        from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
        from torchmetrics.audio.snr import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio

        if self.metrics_to_compute.get("stoi"):
            try:
                from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
                self.stoi = ShortTimeObjectiveIntelligibility(self.sample_rate, False)
            except (ImportError, ModuleNotFoundError):
                self.stoi = None
                logger.warning(
                    "ModuleNotFoundError: STOI metric requires that `pystoi` is installed. "
                    "Either install as `pip install torchmetrics[audio]` or `pip install pystoi`."
                )
        else:
            self.stoi = None

        if self.metrics_to_compute.get("sdr"):
            self.si_sdr = ScaleInvariantSignalDistortionRatio()
            self.sdr = SignalDistortionRatio()
            self.snr = SignalNoiseRatio()
            self.si_snr = ScaleInvariantSignalNoiseRatio()
        else:
            self.si_sdr, self.sdr, self.snr, self.si_snr = None, None, None, None

        if self.metrics_to_compute.get("pesq"):
            try:
                from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
                self.pesq = PerceptualEvaluationSpeechQuality(fs=self.sample_rate, mode="wb")
            except (ImportError, ModuleNotFoundError):
                self.pesq = None
                logger.warning(
                    "ModuleNotFoundError: PerceptualEvaluationSpeechQuality metric requires that `pesq` is installed. "
                    "Either install as `pip install torchmetrics[audio]` or `pip install pesq`."
                )
        else:
            self.pesq = None

    def score(self, dialog: AudioDialog) -> Result:
        """
        Computes augmentation metrics for the given audio dialog.
        """
        if not dialog.audio_step_1_filepath or not dialog.audio_step_3_filepaths:
            raise ValueError("AudioDialog is missing file paths for one or more augmentation steps.")

        # Compute N'th turn end time
        if self.max_turns_to_evaluate < 1 or self.max_turns_to_evaluate > len(dialog.turns):
            raise ValueError(
                "max_turns_to_evaluate must be greater than 0 and "
                "less than or equal to the number of turns in the dialog"
            )
        n_th_turn = dialog.turns[-self.max_turns_to_evaluate - 1]
        n_th_turn_end_time = n_th_turn.audio_start_time + n_th_turn.audio_duration

        # Load audios
        audio_1 = self._load_audio(
            dialog.audio_step_1_filepath, n_th_turn_end_time=n_th_turn_end_time
        )

        all_metrics = {}

        # Process room acoustics audio if available
        for room_config, paths in dialog.audio_step_3_filepaths.items():
            audio_path = paths.audio_path
            if not os.path.exists(audio_path):
                continue

            audio_3 = self._load_audio(audio_path, n_th_turn_end_time=n_th_turn_end_time)

            # Truncate audios
            min_len = min(audio_1.shape[0], audio_3.shape[0])
            a1, a3 = audio_1[:min_len], audio_3[:min_len]

            metrics = {}

            # STOI
            if self.stoi:
                metrics["stoi|3-1"] = self.stoi(a3, a1).item()

            # SDR metrics
            if self.sdr:
                metrics["si-sdr|3-1"] = self.si_sdr(a3, a1).item()
                metrics["sdr|3-1"] = self.sdr(a3, a1).item()
                metrics["snr|3-1"] = self.snr(a3, a1).item()
                metrics["si_snr|3-1"] = self.si_snr(a3, a1).item()

            if self.pesq:
                metrics["pesq|3-1"] = self.pesq(a1, a3).item()

            all_metrics[room_config] = metrics

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

    def results2result(self, results: Dict[str, Result], compute_plots: bool = False) -> Result:
        """
        Compute the overall result from the results of the evaluator on all dialogs.
        """
        aggregated_metrics = {}
        for result in results.values():
            for room_metrics in result.metrics.values():
                for metric_name, value in room_metrics.items():
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
                "max": float(np.max(values))
            }
            for name, values in aggregated_metrics.items()
        }

        plots = []
        if compute_plots or self.compute_plots:
            plots = self.plot(aggregated_metrics)

        return Result(
            metrics=summary_metrics,
            data=aggregated_metrics,
            plots=plots
        )

    def plot(self, data: Dict[str, Any]) -> list:
        """
        Plot the results of the evaluator.
        """
        if not data:
            return []

        num_metrics = len(data)
        if num_metrics == 0:
            return []

        ncols = 2
        nrows = (num_metrics + ncols - 1) // ncols  # Ceiling division
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for i, (metric_name, values) in enumerate(data.items()):
            ax = axes[i]
            if self.plot_type == "boxplot":
                ax.boxplot(values)
            elif self.plot_type == "histogram":
                ax.hist(values, bins='auto')
            elif self.plot_type == "violin":
                ax.violinplot(values, showmeans=True)

            ax.set_title(f"Distribution of {metric_name}")
            ax.set_ylabel("Value")
            ax.set_xlabel(metric_name)

        # Hide any unused subplots
        for j in range(num_metrics, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.close(fig)

        return [fig]
