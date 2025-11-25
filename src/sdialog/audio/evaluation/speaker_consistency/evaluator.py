"""
This module provides an audio evaluation metric for speaker consistency.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

import torch
import logging
import numpy as np
import soundfile as sf
from typing import Dict, Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sdialog.audio.dialog import AudioDialog
from sdialog.audio.evaluation.base import BaseAudioDialogScore, Result


class SpeakerConsistency(BaseAudioDialogScore):
    """
    Computes speaker consistency scores based on x-vector embeddings from pyannote.
    This evaluator calculates two metrics:
    - turn_to_turn_consistency: The average cosine similarity between consecutive turns from the same speaker.
    - global_consistency: The average cosine similarity between all pairs of turns from the same speaker.
    Note:
        This class requires network access to download pretrained models from Hugging Face.
        It also requires 'pyannote.audio', 'torch', and 'scipy' to be installed.
    Example:
        .. code-block:: python
            from sdialog.audio.evaluation.speaker_consistency import SpeakerConsistency
            from sdialog.audio.dialog import AudioDialog
            # Assuming 'audio_dialog' is an instance of AudioDialog with audio data
            consistency_evaluator = SpeakerConsistency()
            scores = consistency_evaluator.score(audio_dialog)
            print(scores)
            # {'turn_to_turn_consistency': 0.85, 'global_consistency': 0.82}
    :param model_name: The name of the pretrained speaker embedding model to use from pyannote.
    :type model_name: str
    :param device: The device to run the model on (e.g., 'cpu', 'cuda'). If None, it will auto-detect.
    :type device: str
    :param use_auth_token: Hugging Face authentication token.
    :type use_auth_token: str
    :param use_acoustic_audio: If True, use audio with acoustic simulation for evaluation.
    :type use_acoustic_audio: bool
    :param compute_plots: Whether to compute plots.
    :type compute_plots: bool
    :param plot_type: The type of plot to generate.
    :type plot_type: str
    """
    def __init__(
        self,
        model_name: str = "pyannote/embedding",
        device: str = None,
        use_auth_token: str = None,
        use_acoustic_audio: bool = False,
        compute_plots: bool = False,
        plot_type: str = "boxplot"
    ):
        super().__init__(name="speaker-consistency")

        try:
            from pyannote.audio import Inference, Model
        except ImportError:
            raise ImportError(
                "The 'pyannote.audio' library is required to use SpeakerConsistency. "
                "Please install it with 'pip install pyannote.audio omegaconf'."
            )

        self.device = device
        self.model_name = model_name
        self.use_auth_token = use_auth_token
        self.use_acoustic_audio = use_acoustic_audio
        self.compute_plots = compute_plots
        self.plot_type = plot_type

        if self.plot_type not in ["histogram", "violin", "boxplot"]:
            raise ValueError("plot_type must be one of 'histogram', 'violin', or 'boxplot'")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained model
        model = Model.from_pretrained(
            self.model_name, use_auth_token=self.use_auth_token
        )
        self.inference = Inference(model, window="whole", device=self.device)

    def results2result(self, results: Dict[str, Result], compute_plots: bool = False) -> Result:
        """
        Compute the overall result from the results of the evaluator on all dialogs.
        :param results: The results of the evaluator.
        :type results: Dict[str, Result]
        :return: The overall result of the evaluator on all dialogs.
        :rtype: Result
        """
        # Aggregate data from all dialogs (generic, ignoring speaker names)
        all_global_consistency_values = []
        all_turn_to_turn_consistency_values = []

        for result in results.values():
            # Flatten all scores from all speakers in the dialog
            for scores in result.data['global_consistency_all'].values():
                all_global_consistency_values.extend(scores)

            for scores in result.data['turn_to_turn_consistency_all'].values():
                all_turn_to_turn_consistency_values.extend(scores)

        # Prepare data for plotting - treat as one generic distribution
        data_global = {"All Dialogs": np.array(all_global_consistency_values)}
        data_turn = {"All Dialogs": np.array(all_turn_to_turn_consistency_values)}

        # Compute overall metrics
        metrics = {
            "global_consistency": (
                np.mean(all_global_consistency_values) if all_global_consistency_values else 0.0
            ),
            "turn_to_turn_consistency": (
                np.mean(all_turn_to_turn_consistency_values) if all_turn_to_turn_consistency_values else 0.0
            )
        }

        # Generate plots
        plots = []
        if compute_plots:
            plots = self.plot(data={
                "num_speakers": 1,
                "per_speaker_all_global_consistency": data_global,
                "per_speaker_all_consistency": data_turn,
            })

        return Result(
            metrics=metrics,
            data={
                "global_consistency_all": data_global,
                "turn_to_turn_consistency_all": data_turn
            },
            plots=plots
        )

    def plot(self, data: Dict[str, Any]) -> list:
        """
        Plot the results of the evaluator.
        :param data: The data to plot.
        :type data: Dict[str, Any]
        :return: A list of plots.
        :rtype: list
        """

        num_speakers = data["num_speakers"]
        per_speaker_all_global_consistency = data["per_speaker_all_global_consistency"]
        per_speaker_all_consistency = data["per_speaker_all_consistency"]

        output_plots = []

        if num_speakers > 0 and per_speaker_all_global_consistency and per_speaker_all_consistency:
            fig, axes = plt.subplots(2, 1, figsize=(10, 10))

            # Plot for the distribution of global_consistency
            all_speakers_global_consistency = list(per_speaker_all_global_consistency.values())
            speaker_labels = list(per_speaker_all_global_consistency.keys())

            if self.plot_type == "histogram":
                for speaker, consistencies in per_speaker_all_global_consistency.items():
                    if consistencies.size > 0:
                        axes[0].hist(consistencies, alpha=0.5, label=f'Speaker {speaker}', bins=10)
                axes[0].legend()
            elif self.plot_type == "violin":
                if all_speakers_global_consistency:
                    axes[0].violinplot(all_speakers_global_consistency, showmeans=True)
                    axes[0].set_xticks(np.arange(1, len(speaker_labels) + 1))
                    axes[0].set_xticklabels(speaker_labels)
            elif self.plot_type == "boxplot":
                if all_speakers_global_consistency:
                    axes[0].boxplot(all_speakers_global_consistency, labels=speaker_labels)

            axes[0].set_title(f'Distribution of Global Speaker Consistency ({self.plot_type.capitalize()})')
            axes[0].set_xlabel('Speaker')
            axes[0].set_ylabel('Cosine Similarity')
            axes[0].set_ylim(0, 1)

            # Plot for the distribution of all_consistency
            all_speakers_consistency = list(per_speaker_all_consistency.values())
            speaker_labels = list(per_speaker_all_consistency.keys())

            if self.plot_type == "histogram":
                for speaker, consistencies in per_speaker_all_consistency.items():
                    if consistencies:
                        axes[1].hist(consistencies, alpha=0.5, label=f'Speaker {speaker}', bins=10)
                axes[1].legend()
            elif self.plot_type == "violin":
                if all_speakers_consistency:
                    axes[1].violinplot(all_speakers_consistency, showmeans=True)
                    axes[1].set_xticks(np.arange(1, len(speaker_labels) + 1))
                    axes[1].set_xticklabels(speaker_labels)
            elif self.plot_type == "boxplot":
                if all_speakers_consistency:
                    axes[1].boxplot(all_speakers_consistency, labels=speaker_labels)

            axes[1].set_title(f'Distribution of Turn-to-Turn Speaker Consistency ({self.plot_type.capitalize()})')
            axes[1].set_xlabel('Speaker')
            axes[1].set_ylabel('Cosine Similarity')
            axes[1].set_ylim(0, 1)

            plt.tight_layout()
            output_plots.append(fig)
            plt.close(fig)

        return output_plots

    def score(self, dialog: "AudioDialog") -> Dict[str, Any]:
        """
        Computes speaker consistency scores for the given audio dialog.
        :param dialog: The audio dialog to evaluate.
        :type dialog: AudioDialog
        :return: A dictionary containing the 'turn_to_turn_consistency',
        'global_consistency' scores and any additional information such as plots.
        :rtype: Dict[str, Any]
        """

        plots = []
        metrics = {}
        data = {}

        # Use the acoustic audio if requested
        if self.use_acoustic_audio:
            if not dialog.audio_step_3_filepaths:
                raise ValueError("Acoustic audio requested, but not found in the dialog.")

            # Use the audio from the first room configuration
            room_audio_path = list(dialog.audio_step_3_filepaths.values())[0]["audio_path"]
            room_audio, sample_rate = sf.read(room_audio_path)

        speaker_turns = {}

        # Group turns by speaker
        for turn in dialog.turns:

            if turn.speaker not in speaker_turns:
                speaker_turns[turn.speaker] = []

            # Use the acoustic audio if requested
            if self.use_acoustic_audio:
                start_sample = int(turn.audio_start_time * sample_rate)
                end_sample = int((turn.audio_start_time + turn.audio_duration) * sample_rate)
                turn_audio = room_audio[start_sample:end_sample]
                waveform = torch.from_numpy(turn_audio).unsqueeze(0).float().to(self.device)
                speaker_turns[turn.speaker].append({"waveform": waveform, "sample_rate": sample_rate})
            # Otherwise, use the original audio for the utterance
            else:
                audio_data, sr = turn.get_audio(), turn.sampling_rate
                if not isinstance(audio_data, torch.Tensor):
                    waveform = torch.from_numpy(audio_data)
                else:
                    waveform = audio_data
                waveform = waveform.unsqueeze(0).float().to(self.device)
                speaker_turns[turn.speaker].append({"waveform": waveform, "sample_rate": sr})

        # Initialize lists to store the global and turn-to-turn speaker consistencies
        per_speaker_local_consistency = {}
        per_speaker_results_global_consistency = {}
        per_speaker_all_consistency = {}
        per_speaker_all_global_consistency = {}

        # Compute embeddings for each turn
        for speaker, turns_data in speaker_turns.items():

            # Check if the speaker has fewer than two turns
            if len(turns_data) < 2:
                logging.warning(
                    f"Speaker '{speaker}' has fewer than two turns, "
                    "skipping consistency calculation for this speaker."
                )
                continue

            # Compute embeddings for each turn
            embeddings = [self.inference(turn_data) for turn_data in turns_data]
            embeddings = np.array(embeddings)
            if embeddings.ndim == 3:
                embeddings = embeddings.squeeze(axis=1)

            # Global consistency
            if len(embeddings) > 1:
                dist_matrix = cdist(embeddings, embeddings, "cosine")
                # Get upper triangle indices, excluding diagonal to avoid self-similarity and duplicates
                triu_indices = np.triu_indices(len(embeddings), k=1)
                global_consistency_distances = dist_matrix[triu_indices]
                per_speaker_all_global_consistency[speaker] = 1 - global_consistency_distances
                global_consistency_dist = np.mean(global_consistency_distances)
                # Convert distance to similarity
                per_speaker_results_global_consistency[speaker] = float(1 - global_consistency_dist)

            # Compute turn-to-turn consistency
            all_consistency = []
            for i in range(len(embeddings) - 1):
                # cdist expects 2D arrays, and Inference already returns a 2D array (1, D)
                distance = cdist(embeddings[i:i + 1], embeddings[i + 1:i + 2], metric="cosine")[0, 0]
                all_consistency.append(1 - distance)
            per_speaker_local_consistency[speaker] = float(np.mean(all_consistency))
            per_speaker_all_consistency[speaker] = all_consistency

        if self.compute_plots:
            plots = self.plot(data={
                "num_speakers": len(speaker_turns),
                "per_speaker_all_global_consistency": per_speaker_all_global_consistency,
                "per_speaker_all_consistency": per_speaker_all_consistency,
            })

        metrics = {
            "global_consistency": per_speaker_results_global_consistency,
            "turn_to_turn_consistency": per_speaker_local_consistency,
        }

        data = {
            "global_consistency_all": {
                s: v.tolist() for s, v in per_speaker_all_global_consistency.items()
            },
            "turn_to_turn_consistency_all": per_speaker_all_consistency,
        }

        return Result(
            metrics=metrics,
            data=data,
            plots=plots,
        )
