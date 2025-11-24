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
from ..base import BaseAudioDialogScore
from scipy.spatial.distance import cdist
from sdialog.audio.dialog import AudioDialog


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
    """
    def __init__(
        self,
        model_name: str = "pyannote/embedding",
        device: str = None,
        use_auth_token: str = None,
        use_acoustic_audio: bool = False
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

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained model
        model = Model.from_pretrained(
            self.model_name, use_auth_token=self.use_auth_token
        )
        self.inference = Inference(model, window="whole", device=self.device)

    def score(self, dialog: "AudioDialog"):
        """
        Computes speaker consistency scores for the given audio dialog.
        :param dialog: The audio dialog to evaluate.
        :type dialog: AudioDialog
        :return: A dictionary containing the 'turn_to_turn_consistency' and 'global_consistency' scores.
        :rtype: dict[str, float]
        """

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

        # Compute embeddings for each turn
        for speaker, turns_data in speaker_turns.items():

            # Check if the speaker has fewer than two turns
            if len(turns_data) < 2:
                logging.warning(
                    f"Speaker '{speaker}' has fewer than two turns, skipping consistency calculation for this speaker."
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
                global_consistency_dist = np.mean(dist_matrix[triu_indices])
                # Convert distance to similarity
                per_speaker_results_global_consistency[speaker] = 1 - global_consistency_dist

            # Compute turn-to-turn consistency
            all_consistency = []
            for i in range(len(embeddings) - 1):
                # cdist expects 2D arrays, and Inference already returns a 2D array (1, D)
                distance = cdist(embeddings[i:i + 1], embeddings[i + 1:i + 2], metric="cosine")[0, 0]
                all_consistency.append(1 - distance)
            per_speaker_local_consistency[speaker] = np.mean(all_consistency)

        final_scores = {
            "global_consistency": per_speaker_results_global_consistency,
            "turn_to_turn_consistency": per_speaker_local_consistency,
        }

        return final_scores
