"""
This module provides an audio evaluation metric for speech analytics.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from sdialog.audio.dialog import AudioDialog
from sdialog.audio.evaluation.base import BaseAudioDialogScore, Result

logger = logging.getLogger(__name__)


class SpeechAnalyticsEvaluator(BaseAudioDialogScore):
    """
    Evaluator for speech analytics, such as utterance duration statistics.

    :param compute_plots: Whether to compute plots.
    :type compute_plots: bool
    """

    def __init__(
        self,
        compute_plots: bool = True,
    ):
        super().__init__(name="speech-analytics-evaluator")
        self.compute_plots = compute_plots

    def score(self, dialog: AudioDialog) -> Result:
        """
        Computes speech analytics for the given audio dialog.
        """
        turn_data = []
        for turn in dialog.turns:
            if turn.audio_duration is not None:
                speaker_name = turn.speaker
                role = dialog.speakers_roles.get(speaker_name, speaker_name)
                turn_data.append({
                    'role': role,
                    'duration': turn.audio_duration,
                })

        return Result(metrics={}, data=turn_data, plots=[])

    def results2result(
        self, results: Dict[str, Any], compute_plots: bool = False
    ) -> Result:
        """
        Compute the overall result from the results of the evaluator on all dialogs.
        """
        all_turn_data = []
        for result in results.values():
            all_turn_data.extend(result.data)

        if not all_turn_data:
            return Result(metrics={}, data=[], plots=[])

        df = pd.DataFrame(all_turn_data)

        summary_metrics = {
            "overall": {
                "mean_duration": df['duration'].mean(),
                "std_duration": df['duration'].std(),
                "min_duration": df['duration'].min(),
                "max_duration": df['duration'].max(),
                "total_utterances": len(df),
            }
        }

        for role in df['role'].unique():
            role_df = df[df['role'] == role]
            summary_metrics[role] = {
                "mean_duration": role_df['duration'].mean(),
                "std_duration": role_df['duration'].std(),
                "min_duration": role_df['duration'].min(),
                "max_duration": role_df['duration'].max(),
                "total_utterances": len(role_df),
            }

        plots = []
        if compute_plots or self.compute_plots:
            plots = self.plot(all_turn_data)

        return Result(metrics=summary_metrics, data=all_turn_data, plots=plots)

    def plot(self, data: List[Dict[str, Any]]) -> list:
        """
        Plot the results of the evaluator.
        """
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError("seaborn is not installed. Please install it using `pip install seaborn`.")

        if not data:
            return []

        df = pd.DataFrame(data)

        sns.set_theme(style="darkgrid")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Overall Utterance Duration Distribution
        sns.histplot(df['duration'], bins=30, ax=axes[0], color='lightblue', edgecolor='grey')
        mean_duration = df['duration'].mean()
        axes[0].axvline(mean_duration, color='red', linestyle='--', label=f'Mean: {mean_duration:.1f}s')
        axes[0].set_title('Overall Utterance Duration Distribution')
        axes[0].set_xlabel('Duration (seconds)')
        axes[0].set_ylabel('Count')
        axes[0].legend()

        # Plot 2: Utterance Duration by Role
        roles = df['role'].unique()
        colors = sns.color_palette("pastel", len(roles))
        role_palette = {role: color for role, color in zip(roles, colors)}

        sns.histplot(
            data=df,
            x='duration',
            hue='role',
            multiple='stack',
            bins=30,
            ax=axes[1],
            palette=role_palette,
            edgecolor='grey'
        )
        axes[1].set_title('Utterance Duration by Role')
        axes[1].set_xlabel('Duration (seconds)')
        axes[1].set_ylabel('Count')

        # Plot 3: Utterance Duration Distribution by Role
        sns.violinplot(data=df, x='role', y='duration', ax=axes[2], palette='pastel')
        axes[2].set_title('Utterance Duration Distribution by Role')
        axes[2].set_xlabel('Role')
        axes[2].set_ylabel('Duration (seconds)')

        plt.tight_layout()
        plt.close(fig)

        return [fig]
