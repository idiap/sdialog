"""
This module provides functions to evaluate the audio generation capabilities of the sdialog library.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import torch
import logging
import numpy as np
from tqdm import tqdm
from jiwer import wer, cer
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

import whisper
from sdialog import Dialog
from pyannote.audio import Model
from pyannote.audio import Inference
from .whisper_normalizer import EnglishTextNormalizer

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

pyannote_model = Model.from_pretrained("pyannote/embedding")
inference = Inference(pyannote_model, window="whole")

normalizer = EnglishTextNormalizer()
whisper_model = whisper.load_model("large-v3", device=device)


def transcript(audios: List[np.ndarray]) -> List[str]:
    """
    Transcript the audios using the whisper model.
    :param audios: The audios to transcript.
    :return: The transcripts.
    :rtype: List[str]
    """
    transcripts = []
    for audio in audios:
        result = whisper_model.transcribe(audio, fp16=False)
        transcripts.append(result["text"])
    return transcripts


def eval_wer(audios: List[Tuple[np.ndarray, str]], dialog: Dialog) -> List[str]:

    # Transcript the audios
    transcripts = transcript([a[0] for a in audios])
    # Get the speakers
    speakers = [a[1] for a in audios]

    # Get the references from the dialog
    references = [t.text for t in dialog.turns]

    data = {}

    # Group the references and transcripts by speaker
    for r, t, s in tqdm(zip(references, transcripts, speakers)):

        if s not in data:
            data[s] = {
                "references": {"normalized": [], "original": []},
                "transcripts": {"normalized": [], "original": []}
            }

        data[s]["references"]["normalized"].append(normalizer(r))
        data[s]["transcripts"]["normalized"].append(normalizer(t))

        data[s]["references"]["original"].append(r)
        data[s]["transcripts"]["original"].append(t)

    # Compute the WER for each speaker
    results = {"wer": {}, "cer": {}, "transcripts": [normalizer(_) for _ in transcripts]}
    for speaker in data:
        data_speaker = data[speaker]
        results["wer"][speaker] = {
            "normalized": wer(
                data_speaker["references"]["normalized"],
                data_speaker["transcripts"]["normalized"]
            ) * 100,
            "original": wer(
                data_speaker["references"]["original"],
                data_speaker["transcripts"]["original"]
            ) * 100
        }
        results["cer"][speaker] = {
            "normalized": cer(
                data_speaker["references"]["normalized"],
                data_speaker["transcripts"]["normalized"]
            ) * 100,
            "original": cer(
                data_speaker["references"]["original"],
                data_speaker["transcripts"]["original"]
            ) * 100
        }

    return results


# TODO: Test this function
def compute_speaker_similarity(
        utterances_audios: List[Tuple[np.ndarray, str]],
        references_voices: List[Tuple[np.ndarray, str]]) -> Dict[str, float]:
    """
    Compute the speaker similarity metrics of the audios.
    :param audios: The audios to compute the speaker similarity metrics.
    :param references_voices: The references voices to compute the speaker similarity metrics.
    :return: The speaker similarity metrics.
    :rtype: Dict[str, float]
    """

    # Initialize a dictionary to hold x-vectors for each speaker utterances
    xvectors = defaultdict(list)

    # Iterate through the utterances and compute x-vectors for each speaker
    for audio, speaker in utterances_audios:

        tensor_audio = torch.Tensor(audio.unsqueeze(0)).unsqueeze(0)
        embedding = inference.infer(tensor_audio)

        xvectors[speaker].append(embedding)

    # Compute the reference voice x-vector for each speaker
    reference_voice_xvectors = {
        speaker: inference.infer(
            torch.Tensor(references_voices[speaker].unsqueeze(0)).unsqueeze(0)
        )
        for speaker in references_voices
    }

    results = {}

    # Compute the speaker similarity between the reference voice x-vector and the utterances
    # audios x-vectors of each speaker
    for speaker in reference_voice_xvectors:

        # Compute the speaker similarity between the reference voice x-vector and the utterances
        # audios x-vectors of the speaker
        speaker_similarities = []

        for audio in xvectors[speaker]:

            speaker_similarities.append(
                cdist(audio, reference_voice_xvectors[speaker], metric="cosine")[0, 0]
            )

        # Return the average score of similarity for the speakers utterances
        results[speaker] = np.mean(speaker_similarities)

    return results


# TODO: Implement the NISQA MOS computation
def compute_mos(audios: List[np.ndarray]) -> List[float]:
    """
    Compute the mean opinion score (UTMOSv2) of the audios.
    :param audios: The audios to compute the UTMOSv2.
    :return: The MOS score, accoustics features (noisiness, discontinuity,
    coloration and loudness) and the figure.
    :rtype: Dict
    """
    nisqa = NonIntrusiveSpeechQualityAssessment(16000)
    scores = []
    for audio in audios:
        _scores = nisqa(audio).to_list()
        scores.append({
            "umos": _scores[0],
            "accoustics": {
                "loudness": _scores[1],
                "pitch": _scores[2],
                "energy": _scores[3],
                "loudness": _scores[4],
            }
        })
    
    print(scores)

    # Compute the mean of each accoustics features for the MOS scores in the ranges (0.0 to 0.25, 0.25 to 0.5, 0.5 to 0.75, 0.75 to 1.0)
    mos_ranges = {
        _range: {
            "loudness": [],
            "pitch": [],
            "energy": [],
            "loudness": [],
        } for _range in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    }

    # Group the scores by the MOS ranges
    for score in scores:
        for _range in mos_ranges:
            if score["umos"] >= _range[0] and score["umos"] < _range[1]:
                for key, value in score["accoustics"].items():
                    mos_ranges[_range][key].append(value)
    
    # Compute the mean of each accoustics features for the MOS scores in the ranges (0.0 to 0.25, 0.25 to 0.5, 0.5 to 0.75, 0.75 to 1.0)
    for _range in mos_ranges:
        for key, value in mos_ranges[_range].items():
            mos_ranges[_range][key] = np.mean(value)

    # Draw the spider chart figure of the accoustics features, where each color is based on the MOS scores ranges 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(0)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlim(0, 1.0)
    ax.set_title("Accoustics Features")
    for _range in mos_ranges:
        ax.plot(np.linspace(0, 2 * np.pi, len(mos_ranges[_range])), mos_ranges[_range])
        ax.fill(np.linspace(0, 2 * np.pi, len(mos_ranges[_range])), mos_ranges[_range], alpha=0.25)
    plt.show()

    return {
        "scores": mos_ranges,
        "figure": fig,
    }


# TODO: Implement the deepfake score computation
def compute_deepfake_score(audios: List[np.ndarray]) -> List[float]:
    """
    Compute the deepfake score of the audios.
    :param audios: The audios to compute the deepfake score.
    :return: The deepfake score.
    :rtype: List[float]
    """
    return [0.0 for _ in audios]


def speaker_consistency(utterances_audios: List[Tuple[np.ndarray, str]]) -> float:
    """
    Evaluates the consistency of speaker audio across utterances.
    :param utterances_audios: List of tuples containing audio data and speaker identifiers.
    :return: Consistency score (0.0 to 1.0).
    :rtype: float
    """

    # Initialize a dictionary to hold x-vectors for each speaker utterances
    xvectors = defaultdict(list)

    # Iterate through the utterances and compute x-vectors for each speaker
    for audio, speaker in utterances_audios:

        tensor_audio = torch.Tensor(audio.unsqueeze(0)).unsqueeze(0)
        embedding = inference.infer(tensor_audio)

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

    # Compute the global consistency by doing a average of the matrix of distances for each speaker
    global_consistency = {
        speaker: 1 - np.mean(cdist(np.vstack(embeddings), np.vstack(embeddings), metric="cosine"))
        for speaker, embeddings in xvectors.items()
    }

    # Compute the centroid of the embeddings for each speaker
    centroids = {speaker: np.mean(embeddings, axis=0) for speaker, embeddings in xvectors.items()}
    # Compute the average distance between the centroids and utterances embeddings of each speaker
    average_distance_with_centroid = {
        speaker: 1 - np.mean(cdist(np.vstack(embeddings), centroids[speaker], metric="cosine"))
        for speaker, embeddings in xvectors.items()
    }

    return {
        "local_consistency": avg_distance,
        "global_consistency": global_consistency,
        "average_distance_with_centroid": average_distance_with_centroid,
    }
