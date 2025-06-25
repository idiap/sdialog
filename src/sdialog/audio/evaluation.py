"""
This module provides functions to evaluate the consistency of speaker audio across utterances
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import torch
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from scipy.spatial.distance import cdist

from pyannote.audio import Model
from pyannote.audio import Inference

model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model, window="whole")
inference.to(torch.device("cuda"))


def speaker_consistency(utterances_audios: List[Tuple[np.ndarray, str]]) -> float:
    """
    Evaluates the consistency of speaker audio across utterances.
    :param utterances_audios: List of tuples containing audio data and speaker identifiers.
    :return: Consistency score (0.0 to 1.0).
    :rtype: float
    """

    # Initialize a dictionary to hold x-vectors for each speaker utterances
    xvectors = defaultdict(str)

    # Iterate through the utterances and compute x-vectors for each speaker
    for audio, speaker in utterances_audios:

        tensor_audio = torch.Tensor(audio.unsqueeze(0)).unsqueeze(0)
        embedding = inference.infer(tensor_audio)

        if speaker not in xvectors:
            xvectors[speaker] = []

        xvectors[speaker].append(embedding)

    avg_distance = {}

    # For each speaker, compute the cosine distance between consecutive utterances
    for speaker in xvectors:

        _distances = []

        for i in range(len(xvectors[speaker]) - 1):

            # Get the embeddings for two consecutive utterances of the same speaker
            embedding1 = xvectors[speaker][i]
            embedding2 = xvectors[speaker][i + 1]

            # Compute the cosine similarity between two utterance embeddings of the same speaker
            distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]
            _distances.append(distance)

        # Return a score between 0.0 and 1.0, where 1.0 is perfect consistency
        if _distances:
            avg_distance[speaker] = 1.0 - np.mean(_distances)
        else:
            avg_distance[speaker] = 1.0

    return {
        "local_consistency": avg_distance,
    }


def timestamps_alignment(audio: np.ndarray, reference_audio: np.ndarray) -> float:
    """
    Evaluates the alignment of timestamps in the audio against an reference audio for the same dialog.
    """
    pass
