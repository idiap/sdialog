import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
import soundfile as sf
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc

DIR_PATH = (
    "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio/Generation/"
    "outputs-voices-libritts-indextts+dscaper+acoustics+metadata"
)

XVECTORS_PATH = "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/libritts-voices/xvectors.pkl"

# Initialize the speaker embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
pyannote_model = Model.from_pretrained("pyannote/embedding")
inference = Inference(pyannote_model, window="whole")

# Load reference speaker embeddings
with open(XVECTORS_PATH, "rb") as f:
    reference_xvectors = pickle.load(f)


def evaluate_speaker_identification(
    dialog: dict, dialog_personas: dict, reference_xvectors: dict
) -> dict:
    """
    Evaluates speaker identification by assigning each utterance to the most similar speaker
    and computes metrics for ROC analysis.
    :param dialog: Dictionary containing the audio turns.
    :param dialog_personas: Dictionary with persona information, including voice identifiers.
    :param reference_xvectors: Dictionary with reference speaker embeddings.
    :return: Dictionary with true labels and prediction scores.
    :rtype: dict
    """
    sample_rate = 16000

    speaker_roles = sorted(list(dialog_personas.keys()))
    if len(speaker_roles) != 2:
        return {
            "true_labels": [],
            "pred_scores": [],
            "error": "Not a two-speaker dialogue",
        }

    ref_xvecs = {}
    for role in speaker_roles:
        if role in dialog_personas and "voice" in dialog_personas[role].get(
            "_metadata", {}
        ):
            speaker_id = dialog_personas[role]["_metadata"]["voice"]["identifier"]
            if speaker_id in reference_xvectors:
                ref_xvecs[role] = reference_xvectors[speaker_id]
            else:
                return {
                    "true_labels": [],
                    "pred_scores": [],
                    "error": f"Missing ref x-vector for {role} ({speaker_id})",
                }
        else:
            return {
                "true_labels": [],
                "pred_scores": [],
                "error": f"Missing persona voice info for {role}",
            }

    if len(ref_xvecs) != 2:
        return {
            "true_labels": [],
            "pred_scores": [],
            "error": "Could not retrieve reference x-vectors for both speakers",
        }

    positive_class = speaker_roles[0]

    true_labels = []
    pred_scores = []

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
        embedding = inference({"waveform": tensor_audio, "sample_rate": sample_rate})

        similarities = {}
        for role, ref_xvec in ref_xvecs.items():
            distance = cdist(embedding.reshape(1, -1), ref_xvec.reshape(1, -1), metric="cosine")[0, 0]
            similarities[role] = 1.0 - distance

        true_speaker = turn['speaker']
        score = similarities[positive_class]

        true_labels.append(1 if true_speaker == positive_class else 0)
        pred_scores.append(score)

    return {"true_labels": true_labels, "pred_scores": pred_scores}


_paths = [p for p in os.listdir(DIR_PATH) if os.path.isdir(os.path.join(DIR_PATH, p))]

all_true_labels = []
all_pred_scores = []

for dir_dialog in tqdm(_paths):

    dialog_id = dir_dialog.split("_")[1]

    path_utterances = os.path.join(DIR_PATH, dir_dialog, "utterances")

    path_exported = os.path.join(DIR_PATH, dir_dialog, "exported_audios")
    json_path = os.path.join(path_exported, "audio_pipeline_info.json")

    if not os.path.exists(json_path):
        print(f"Skipping {dialog_id}, no audio_pipeline_info.json")
        continue

    with open(json_path, "r") as f:
        dialog_info = json.load(f)

    dialog_personas = dialog_info.get("personas", {})
    if len(dialog_personas) < 2 or not all(
        "voice" in p.get("_metadata", {}) for p in dialog_personas.values()
    ):
        print(f"Skipping dialog {dialog_id} due to insufficient persona/voice information.")
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

    results = evaluate_speaker_identification(
        audio_dialog, dialog_personas, reference_xvectors
    )

    if "error" in results and results["error"]:
        print(f"Skipping dialog {dialog_id}: {results['error']}")
        continue

    all_true_labels.extend(results["true_labels"])
    all_pred_scores.extend(results["pred_scores"])


if all_true_labels and all_pred_scores:
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_pred_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic for Speaker Identification")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    plot_filename = "speaker_identification_roc_curve.pdf"
    plt.savefig(plot_filename)
    print(f"\nROC curve plot saved to {plot_filename}")
    print(f"AUC: {roc_auc:.4f}")
else:
    print("\nNo results to compute ROC curve.")
