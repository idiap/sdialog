import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
import soundfile as sf

DIR_PATH = (
    "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio/Generation/"
    "outputs-voices-libritts-indextts+dscaper+acoustics+metadata"
)
# "/Generation/outputs-voices-libritts-indextts+dscaper+acoustcs"

# Initialize the speaker embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
pyannote_model = Model.from_pretrained("pyannote/embedding")
inference = Inference(pyannote_model, window="whole")


def speaker_consistency(dialog: dict) -> dict:
    """
    Evaluates the consistency of speaker audio across utterances.
    :param utterances_audios: List of tuples containing audio data and speaker identifiers.
    :return: Consistency score (0.0 to 1.0).
    :rtype: float
    """

    # Initialize a dictionary to hold x-vectors for each speaker utterances
    xvectors = defaultdict(list)
    sample_rate = 16000

    # Iterate through the utterances and compute x-vectors for each speaker
    for turn in dialog['turns']:
        audio, sr = sf.read(turn['audio_path'])

        if sr != sample_rate:
            if audio.ndim > 1:
                audio = audio.T
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            if audio.ndim > 1:
                audio = audio.T

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        tensor_audio = torch.Tensor(audio).unsqueeze(0)
        embedding = inference.infer(tensor_audio)

        xvectors[turn['speaker']].append(embedding)

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
    centroids = {speaker: np.mean(np.vstack(embeddings), axis=0) for speaker, embeddings in xvectors.items()}
    # Compute the average distance between the centroids and utterances embeddings of each speaker
    average_distance_with_centroid = {
        speaker: 1
        - np.mean(
            cdist(
                np.vstack(embeddings),
                centroids[speaker].reshape(1, -1),
                metric="cosine",
            )
        )
        for speaker, embeddings in xvectors.items()
    }

    return {
        "local_consistency": avg_distance,
        "global_consistency": global_consistency,
        "average_distance_with_centroid": average_distance_with_centroid,
    }


_paths = os.listdir(DIR_PATH)

all_results = {}
for dir_dialog in tqdm(_paths):

    if not os.path.isdir(os.path.join(DIR_PATH, dir_dialog)):
        continue

    dialog_id = dir_dialog.split("_")[1]
    print(f"Processing dialog {dialog_id}")

    path_utterances = os.path.join(DIR_PATH, dir_dialog, "utterances")

    if not os.path.isdir(path_utterances):
        continue

    unsorted_turns = []
    all_utterances_paths = [
        os.path.join(path_utterances, f)
        for f in os.listdir(path_utterances)
        if f.endswith(".wav")
    ]
    for utterance_path in all_utterances_paths:
        try:
            filename = os.path.basename(utterance_path)
            turn_id_str, speaker = os.path.splitext(filename)[0].split("_")
            unsorted_turns.append(
                {"id": int(turn_id_str), "speaker": speaker, "audio_path": utterance_path}
            )
        except (ValueError, IndexError):
            print(f"Skipping malformed filename: {os.path.basename(utterance_path)}")
            continue

    if not unsorted_turns:
        print(f"No turns found for dialog {dialog_id}")
        continue

    audio_turns = sorted(unsorted_turns, key=lambda x: x["id"])
    audio_dialog = {"turns": audio_turns}

    results = speaker_consistency(audio_dialog)

    all_results[dialog_id] = results

# Save the results to a json file
with open("metrics_speakers_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Transform the results to a dataframe
df = pd.DataFrame(all_results).T

# Compute and print average metrics for each speaker role
for role in ["DOCTOR", "PATIENT"]:
    print(f"\n----- Metrics for {role} -----")

    # Local Consistency
    local_consistency_scores = (
        df["local_consistency"].apply(lambda x: x.get(role, np.nan)).dropna()
    )
    if not local_consistency_scores.empty:
        print("  Local Consistency:")
        print(f"    - Mean: {local_consistency_scores.mean():.4f}")
        print(f"    - Std:  {local_consistency_scores.std():.4f}")
        min_dialog = local_consistency_scores.idxmin()
        max_dialog = local_consistency_scores.idxmax()
        print(f"    - Min:  {local_consistency_scores.min():.4f} (Dialog: {min_dialog})")
        print(f"    - Max:  {local_consistency_scores.max():.4f} (Dialog: {max_dialog})")

    # Global Consistency
    global_consistency_scores = (
        df["global_consistency"].apply(lambda x: x.get(role, np.nan)).dropna()
    )
    if not global_consistency_scores.empty:
        print("  Global Consistency:")
        print(f"    - Mean: {global_consistency_scores.mean():.4f}")
        print(f"    - Std:  {global_consistency_scores.std():.4f}")
        min_dialog = global_consistency_scores.idxmin()
        max_dialog = global_consistency_scores.idxmax()
        print(f"    - Min:  {global_consistency_scores.min():.4f} (Dialog: {min_dialog})")
        print(f"    - Max:  {global_consistency_scores.max():.4f} (Dialog: {max_dialog})")

    # Average Distance with Centroid
    avg_dist_centroid_scores = (
        df["average_distance_with_centroid"]
        .apply(lambda x: x.get(role, np.nan))
        .dropna()
    )
    if not avg_dist_centroid_scores.empty:
        print("  Average Distance with Centroid:")
        print(f"    - Mean: {avg_dist_centroid_scores.mean():.4f}")
        print(f"    - Std:  {avg_dist_centroid_scores.std():.4f}")
        min_dialog = avg_dist_centroid_scores.idxmin()
        max_dialog = avg_dist_centroid_scores.idxmax()
        print(f"    - Min:  {avg_dist_centroid_scores.min():.4f} (Dialog: {min_dialog})")
        print(f"    - Max:  {avg_dist_centroid_scores.max():.4f} (Dialog: {max_dialog})")
