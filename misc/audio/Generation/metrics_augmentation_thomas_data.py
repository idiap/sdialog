import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

# Fix for onnxruntime error on some systems
os.environ["OMP_NUM_THREADS"] = "1"

print("Loading metrics...")


def calculate_nisqa_in_chunks(audio_tensor, nisqa_metric, sample_rate, chunk_duration_sec=10):
    """
    Calculates NISQA score for a long audio file by processing it in chunks.
    """
    chunk_size = int(chunk_duration_sec * sample_rate)

    if audio_tensor.dim() > 1:
        # Average channels for multichannel audio
        audio_tensor = torch.mean(audio_tensor, dim=1)

    total_len = audio_tensor.shape[0]

    if total_len < sample_rate:  # NISQA may fail on very short audio
        print("Warning: audio shorter than 1s, skipping NISQA calculation.")
        return None

    if total_len <= chunk_size:
        try:
            score = nisqa_metric(audio_tensor)
            if isinstance(score, dict):
                score = score["mos_pred"]
            if score.numel() > 1:
                return score.mean().item()
            return score.item()
        except (RuntimeError, IndexError) as e:
            print(f"NISQA failed for a short audio segment: {e}")
            return None

    num_chunks = int(np.ceil(total_len / chunk_size))
    nisqa_scores = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio_tensor[start:end]

        if chunk.shape[0] < sample_rate:
            continue

        try:
            score = nisqa_metric(chunk)
            if isinstance(score, dict):
                score = score["mos_pred"]
            if score.numel() > 1:
                nisqa_scores.extend(score.tolist())
            else:
                nisqa_scores.append(score.item())
        except (RuntimeError, IndexError) as e:
            print(f"NISQA failed on a chunk: {e}")
            continue

    if not nisqa_scores:
        return None

    return sum(nisqa_scores) / len(nisqa_scores)


# Summary of metrics to be run
print("The following metrics will be calculated:")
metrics_status = {
    "NISQA": True,
}

for name, enabled in metrics_status.items():
    if enabled:
        print(f"✅ {name}")
for name, enabled in metrics_status.items():
    if not enabled:
        print(f"❌ {name}")
print("-" * 50)


sample_rate = 16_000

print("Loading NISQA...")
nisqa = NonIntrusiveSpeechQualityAssessment(sample_rate)
print("NISQA loaded")

DIR_PATH = "./thomas_data"

results = {}

for filename in tqdm(os.listdir(DIR_PATH)):
    if not filename.lower().endswith((".wav", ".flac")):
        continue

    wav_path = os.path.join(DIR_PATH, filename)
    print(f"Processing {wav_path}")

    # Load wav file
    try:
        audio, sampling_rate = sf.read(wav_path)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        continue

    if sampling_rate != sample_rate:
        if audio.ndim > 1:
            audio = audio.T
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sample_rate)
        if audio.ndim > 1:
            audio = audio.T

    audio = torch.from_numpy(audio)
    if audio.ndim == 2:
        audio = torch.mean(audio, dim=1)

    # Initialize metric results to None
    srmr_score, nisqa_score, dnsmos_scores = None, None, None

    # Evaluate the audio on NISQA
    try:
        nisqa_score = calculate_nisqa_in_chunks(audio, nisqa, sample_rate)
        print(f"NISQA: {nisqa_score}")
    except Exception as e:
        print(f"NISQA failed for {filename}: {e}")

    _result = {
        "nisqa-mos": nisqa_score,
    }

    results[filename] = _result

print("-" * 50)
print("Data:")
print("-" * 50)

print(results)

print("-" * 50)
print("Results:")
print("-" * 50)

# Save the results to a json file
with open("metrics_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Converting results to dataframe...")
# Create a dataframe from the results
results_df = pd.DataFrame.from_dict(results, orient="index")
print(results_df)

print("Computing statistics...")
# Compute the average, min, max and std for all dialogs for each metrics and submetrics
results_stats = results_df.describe()

print(results_stats)

# Save the results to a csv file
results_df.to_csv("metrics_results.csv", index=True)
results_stats.to_csv("metrics_results_stats.csv", index=True)
print("Saving results to csv file...")
