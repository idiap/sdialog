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
from scipy.optimize import brentq
from scipy.interpolate import interp1d

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
    Evaluates speaker identification by comparing each utterance to all reference speakers
    and computes metrics for ROC analysis.
    :param dialog: Dictionary containing the audio turns.
    :param dialog_personas: Dictionary with persona information, including voice identifiers.
    :param reference_xvectors: Dictionary with reference speaker embeddings for all speakers.
    :return: Dictionary with true labels (1 for genuine, 0 for impostor) and prediction scores (similarities).
    :rtype: dict
    """
    sample_rate = 16000

    dialog_voice_ids = {}
    for role, persona in dialog_personas.items():
        if "voice" in persona.get("_metadata", {}):
            dialog_voice_ids[role] = persona["_metadata"]["voice"]["identifier"]
        else:
            return {"error": f"Missing persona voice info for {role}"}
    
    true_labels = []
    pred_scores = []

    for turn in dialog["turns"]:
        try:
            audio, sr = sf.read(turn["audio_path"])
        except Exception as e:
            print(f"Skipping turn, could not read audio file {turn['audio_path']}: {e}")
            continue

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

        true_speaker_role = turn["speaker"]
        if true_speaker_role not in dialog_voice_ids:
            print(
                f"Skipping turn, role '{true_speaker_role}' not in dialog_personas."
            )
            continue
        true_speaker_id = dialog_voice_ids[true_speaker_role]

        if true_speaker_id not in reference_xvectors:
            print(
                f"Skipping turn, speaker_id '{true_speaker_id}' not in reference_xvectors."
            )
            continue
        
        for ref_id, ref_xvec in reference_xvectors.items():
            distance = cdist(
                embedding.reshape(1, -1), ref_xvec.reshape(1, -1), metric="cosine"
            )[0, 0]
            similarity = 1.0 - distance
            pred_scores.append(similarity)
            true_labels.append(1 if ref_id == true_speaker_id else 0)

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

    # Calculate EER
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.scatter(
        eer,
        1 - eer,
        marker="o",
        color="red",
        zorder=5,
        label=f"EER = {eer:.2f} at threshold = {thresh:.2f}",
    )
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
    plot_filename = "speaker_identification_roc_curve.png"
    plt.savefig(plot_filename)
    print(f"\nROC curve plot saved to {plot_filename}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
else:
    print("\nNo results to compute ROC curve.")
