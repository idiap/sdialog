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
from sdialog.audio.audio_dialog import AudioDialog
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

import whisper
from pyannote.audio import Model, Inference
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


def eval_wer_cer(dialog: AudioDialog) -> Dict:
    """
    Evaluate the WER and CER of the dialog.
    :param dialog: The dialog to evaluate.
    :return: The WER and CER of the dialog.
    :rtype: Dict
    """

    # Transcript the audios
    _transcripts = transcript([turn.get_audio() for turn in dialog.turns])
    for idx, turn in enumerate(dialog.turns):
        turn.transcript = _transcripts[idx]

    data = {}

    # Group the references and transcripts by speaker
    for turn in tqdm(dialog.turns):

        if turn.speaker not in data:
            data[turn.speaker] = {
                "references": {"normalized": [], "original": []},
                "transcripts": {"normalized": [], "original": []}
            }

        data[turn.speaker]["references"]["normalized"].append(normalizer(turn.text))
        data[turn.speaker]["transcripts"]["normalized"].append(normalizer(turn.transcript))

        data[turn.speaker]["references"]["original"].append(turn.text)
        data[turn.speaker]["transcripts"]["original"].append(turn.transcript)

    # Compute the WER for each speaker
    results = {"wer": {}, "cer": {}, "transcripts": [normalizer(_) for _ in dialog.turns]}
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


def compute_evaluation_utterances(dialog: AudioDialog) -> Dict:
    """
    Compute the evaluation of the utterances audios.
    :param utterances_audios: The utterances audios to compute the evaluation.
    :param dialog: The dialog to compute the evaluation.
    :return: The evaluation metrics.
    :rtype: Dict
    """

    mos_score = compute_mos(dialog, show_figure=True)
    speaker_consistency_score = speaker_consistency(dialog)
    wer_score = eval_wer_cer(dialog)
    
    return {
        "wer": wer_score,
        "speaker_similarity": 0.0,
        "deepfake_score": 0.0,
        "speaker_consistency": speaker_consistency_score,
        "mos": mos_score,
    }


def compute_evaluation_audio(dialog: AudioDialog) -> Dict:
    """
    Compute the evaluation of the audio.
    :param dialog: The dialog to compute the evaluation.
    :return: The evaluation metrics based on the reference audio before room accoustics.
    :rtype: Dict
    """
    
    return {
        "wer": 0.0,
        "stoi": 0.0,
        "pesq": 0.0,
        "timing_statistics": {
            "mean_word_duration": 0.0,
            "mean_utterance_duration": 0.0,
            "mean_silence_duration": 0.0,
            "mean_speaker_duration": {
                "speaker_1": 0.0,
                "speaker_2": 0.0,
            }
        },
        "f0_statistics": {
            "mean_f0": 0.0,
            "std_f0": 0.0,
            "min_f0": 0.0,
            "max_f0": 0.0,
        },
    }


def compute_mos(dialog: AudioDialog, show_figure: bool = False, output_file: str = None) -> Dict:
    """
    Compute the mean opinion score (MOS) of the audios.
    :param audios: The audios to compute the MOS.
    :return: The MOS score, accoustics features (noisiness, discontinuity,
    coloration and loudness) and the figure.
    :rtype: Dict
    """
    nisqa = NonIntrusiveSpeechQualityAssessment(16000)
    scores = []
    for turn in tqdm(dialog.turns):
        
        if torch.is_tensor(turn.get_audio()):
            input_tensor = turn.get_audio().clone().detach().to(torch.float32)
        else:
            input_tensor = torch.tensor(turn.get_audio(), dtype=torch.float32)
        
        _scores = nisqa(input_tensor).tolist()
        scores.append({
            "umos": _scores[0],
            "accoustics": {
                "noisiness": _scores[1],
                "discontinuity": _scores[2],
                "coloration": _scores[3],
                "loudness": _scores[4],
            }
        })
    
    # Compute the mean of each accoustics features for the MOS scores in the ranges
    mos_ranges = {
        _range: {
            "noisiness": [],
            "discontinuity": [],
            "coloration": [],
            "loudness": [],
        } for _range in [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.01)]
    }

    # Group the scores by the MOS ranges
    for score in scores:
        for _range in mos_ranges:
            if score["umos"] >= _range[0] and score["umos"] < _range[1]:
                for key, value in score["accoustics"].items():
                    mos_ranges[_range][key].append(value)
    
    # Compute the mean of each accoustics features for the MOS scores in the ranges
    for _range in mos_ranges:
        for key, value in mos_ranges[_range].items():
            if value:
                mos_ranges[_range][key] = np.mean(value)
            else:
                mos_ranges[_range][key] = 0

    # Draw the spider chart figure of the accoustics features, where each color is based on the MOS scores ranges 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    labels = list(list(mos_ranges.values())[0].keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_rlabel_position(0)
    ax.set_ylim(0, 5)

    ax.set_title("Acoustics Features per MOS range", size=15, y=1.1)

    for _range in mos_ranges:
        # Check if there is any data to plot for this range
        if any(mos_ranges[_range].values()):
            values = list(mos_ranges[_range].values())
            values += values[:1]
            ax.plot(angles, values, label=f'MOS {_range}')
            ax.fill(angles, values, alpha=0.25)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if show_figure:
        plt.show()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

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


def speaker_consistency(dialog: AudioDialog) -> float:
    """
    Evaluates the consistency of speaker audio across utterances.
    :param utterances_audios: List of tuples containing audio data and speaker identifiers.
    :return: Consistency score (0.0 to 1.0).
    :rtype: float
    """

    # Initialize a dictionary to hold x-vectors for each speaker utterances
    xvectors = defaultdict(list)

    # Iterate through the utterances and compute x-vectors for each speaker
    for turn in dialog.turns:

        tensor_audio = torch.Tensor(turn.get_audio().unsqueeze(0)).unsqueeze(0)
        embedding = inference.infer(tensor_audio)

        xvectors[turn.speaker].append(embedding)

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
