import os
import json
import torch
import librosa
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.snr import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.sdr import SignalDistortionRatio
from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

# Fix for onnxruntime error on some systems
os.environ["OMP_NUM_THREADS"] = "1"

print("Loading metrics...")


def apply_snr(x, snr):
    x = x.numpy()
    dbfs = 10 ** (snr / 20)
    x *= dbfs / np.abs(x).max(initial=1e-15)
    x = torch.from_numpy(x)
    return x


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


# Add argument parser
parser = argparse.ArgumentParser(description="Calculate audio metrics.")
parser.add_argument('--stoi', action='store_true', help='Enable STOI calculation')
parser.add_argument('--sdr', action='store_true', help='Enable SI-SDR calculation')
parser.add_argument('--nisqa', action='store_true', help='Enable NISQA calculation')
parser.add_argument('--pesq', action='store_false', help='Enable PESQ calculation')
parser.add_argument('--dnsmos', action='store_false', help='Enable DNSMOS calculation')
parser.add_argument('--all', action='store_true', help='Enable all metric calculations')
args = parser.parse_args()

if args.all:
    args.stoi = True
    args.sdr = True
    args.pesq = True
    args.dnsmos = True
    args.nisqa = True

# Summary of metrics to be run
print("The following metrics will be calculated:")
metrics_status = {
    "STOI": args.stoi,
    "SI-SDR": args.sdr,
    "PESQ": args.pesq,
    "DNSMOS": args.dnsmos,
    "NISQA": args.nisqa,
}

for name, enabled in metrics_status.items():
    if enabled:
        print(f"✅ {name}")
for name, enabled in metrics_status.items():
    if not enabled:
        print(f"❌ {name}")
print("-" * 50)


sample_rate = 16_000

stoi = None
if args.stoi:
    stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)
    print("STOI loaded")

si_sdr = None
if args.sdr:
    si_sdr = ScaleInvariantSignalDistortionRatio()
    snr = SignalNoiseRatio()
    sdr = SignalDistortionRatio()
    srmr = SpeechReverberationModulationEnergyRatio(sample_rate)
    si_snr = ScaleInvariantSignalNoiseRatio()
    print("SDR loaded")

pesq_metric = None
if args.pesq:
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode="wb")
    print("PESQ loaded")

dnsmos = None
if args.dnsmos:
    dnsmos = DeepNoiseSuppressionMeanOpinionScore(sample_rate, False)
    print("DNMOS loaded")

nisqa = None
if args.nisqa:
    nisqa = NonIntrusiveSpeechQualityAssessment(sample_rate)
    print("NISQA loaded")

DIR_PATH = (
    "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio"
    "/Generation/outputs-voices-libritts-indextts+dscaper+acoustics+metadata"
)
# "/Generation/outputs-voices-libritts-indextts+dscaper+acoustcs"

results = {}

for dir_dialog in tqdm(os.listdir(DIR_PATH)):

    _, dialog_id = dir_dialog.split("_")
    # print(dialog_id)

    # Get the exported audios paths
    path_exported = os.path.join(DIR_PATH, dir_dialog, "exported_audios")

    json_path = os.path.join(path_exported, "audio_pipeline_info.json")

    wav_path_1 = os.path.join(path_exported, "audio_pipeline_step1.wav")
    wav_path_2 = os.path.join(path_exported, "audio_pipeline_step2.wav")
    wav_path_3 = os.path.join(path_exported, "audio_pipeline_step3.wav")

    if not os.path.exists(wav_path_1) or not os.path.exists(wav_path_2) or not os.path.exists(wav_path_3):
        print(f"Skipping {dir_dialog} because it doesn't have all the audios")
        continue

    # Load wav files
    audio_1, sampling_rate_1 = sf.read(wav_path_1)
    if sampling_rate_1 != sample_rate:
        audio_1 = librosa.resample(audio_1.T, orig_sr=sampling_rate_1, target_sr=sample_rate).T
    audio_1 = torch.from_numpy(audio_1)
    # print("Audio 1 done!")

    audio_2, sampling_rate_2 = sf.read(wav_path_2)
    if sampling_rate_2 != sample_rate:
        audio_2 = librosa.resample(audio_2.T, orig_sr=sampling_rate_2, target_sr=sample_rate).T
    audio_2 = torch.from_numpy(audio_2)
    # print("Audio 2 done!")

    audio_3, sampling_rate_3 = sf.read(wav_path_3)
    if sampling_rate_3 != sample_rate:
        audio_3 = librosa.resample(audio_3.T, orig_sr=sampling_rate_3, target_sr=sample_rate).T
    audio_3 = torch.from_numpy(audio_3)
    # print("Audio 3 done!")

    # Truncate audios to the same length for comparison
    min_len = min(audio_1.shape[0], audio_2.shape[0], audio_3.shape[0])
    audio_1 = audio_1[:min_len]
    audio_2 = audio_2[:min_len]
    audio_3 = audio_3[:min_len]

    # Initialize metric results to None
    stoi_2_1, stoi_3_1 = None, None
    sdr_2_1, sdr_3_1 = None, None
    nisqa_1, nisqa_2, nisqa_3 = None, None, None
    # dnsmos_1, dnsmos_2, dnsmos_3 = None, None, None
    # pesq_2_1, pesq_3_1 = None, None

    # # Evaluate the audio on PESQ
    # if args.pesq:
    #     pesq_2_1 = pesq_metric(audio_1, audio_2)
    #     print(f"PESQ 2-1: {pesq_2_1}")
    #     pesq_3_1 = pesq_metric(audio_1, audio_3)
    #     print(f"PESQ 3-1: {pesq_3_1}")

    # Evaluate the audio on STOI, DNMOS, SDR, SRMR, PESQ, WER
    if args.stoi:
        stoi_2_1 = stoi(audio_2, audio_1)
        # print(f"STOI 2-1: {stoi_2_1}")

        stoi_3_1 = stoi(audio_3, audio_1)
        # print(f"STOI 3-1: {stoi_3_1}")

    # Evaluate the audio on SDR
    if args.sdr:

        si_sdr_2_1 = si_sdr(audio_2, audio_1)
        print(f"SI-SDR 2-1: {si_sdr_2_1}")
        si_sdr_3_1 = si_sdr(audio_3, audio_1)
        print(f"SI-SDR 3-1: {si_sdr_3_1}")
        si_sdr_3_2 = si_sdr(audio_3, audio_2)
        print(f"SI-SDR 3-2: {si_sdr_3_2}")

        print("-"*50)

        sdr_2_1 = sdr(audio_2, audio_1)
        print(f"SDR 2-1: {sdr_2_1}")
        sdr_3_1 = sdr(audio_3, audio_1)
        print(f"SDR 3-1: {sdr_3_1}")
        sdr_3_2 = sdr(audio_3, audio_2)
        print(f"SDR 3-2: {sdr_3_2}")

        print("-"*50)

        snr_1_1 = snr(audio_1, audio_1)
        print(f"SNR 1-1: {snr_1_1}")
        snr_2_1 = snr(audio_2, audio_1)
        print(f"SNR 2-1: {snr_2_1}")
        snr_3_1 = snr(audio_3, audio_1)
        print(f"SNR 3-1: {snr_3_1}")
        snr_3_2 = snr(audio_3, audio_2)
        print(f"SNR 3-2: {snr_3_2}")
        snr_improved = snr_3_1 - snr_2_1
        print(f"SNR improved: {snr_improved}")

        print("-"*50)

        si_snr_2_1 = si_snr(audio_2, audio_1)
        print(f"SI-SNR 2-1: {si_snr_2_1}")
        si_snr_3_1 = si_snr(audio_3, audio_1)
        print(f"SI-SNR 3-1: {si_snr_3_1}")
        si_snr_3_2 = si_snr(audio_3, audio_2)
        print(f"SI-SNR 3-2: {si_snr_3_2}")
        si_snr_improved = si_snr_3_1 - si_snr_2_1
        print(f"SI-SNR improved: {si_snr_improved}")

        print("-"*50)

        srmr_1 = srmr(audio_1)
        print(f"SRMR 1: {srmr_1}")
        srmr_2 = srmr(audio_2)
        print(f"SRMR 2: {srmr_2}")
        srmr_3 = srmr(audio_3)
        print(f"SRMR 3: {srmr_3}")

        # sdr_1-2 = sdr(audio_1, audio_2)
        # sdr_1-3 = sdr(audio_1, audio_3)

        # sdr_i_3_2 = sdr_3 - sdr_2

    # # Evaluate the audio on DNMOS
    # if args.dnsmos:
    #     dnsmos_1 = dnsmos(audio_1)
    #     print(f"DNMOS 1: {dnsmos_1}")
    #     dnsmos_2 = dnsmos(audio_2)
    #     print(f"DNMOS 2: {dnsmos_2}")
    #     dnsmos_3 = dnsmos(audio_3)
    #     print(f"DNMOS 3: {dnsmos_3}")

    # Evaluate the audio on NISQA
    if args.nisqa:
        nisqa_1 = calculate_nisqa_in_chunks(audio_1, nisqa, sample_rate)
        # print(f"NISQA 1: {nisqa_1}")
        nisqa_2 = calculate_nisqa_in_chunks(audio_2, nisqa, sample_rate)
        # print(f"NISQA 2: {nisqa_2}")
        nisqa_3 = calculate_nisqa_in_chunks(audio_3, nisqa, sample_rate)
        # print(f"NISQA 3: {nisqa_3}")

    _result = {
        "stoi|2-1": stoi_2_1.item() if stoi_2_1 is not None else None,
        "stoi|3-1": stoi_3_1.item() if stoi_3_1 is not None else None,

        "si-sdr|2-1": si_sdr_2_1.item() if si_sdr_2_1 is not None else None,
        "si-sdr|3-1": si_sdr_3_1.item() if si_sdr_3_1 is not None else None,
        "si-sdr|3-2": si_sdr_3_2.item() if si_sdr_3_2 is not None else None,

        "srmr|1": srmr_1.item() if srmr_1 is not None else None,
        "srmr|2": srmr_2.item() if srmr_2 is not None else None,
        "srmr|3": srmr_3.item() if srmr_3 is not None else None,

        "sdr|2-1": sdr_2_1.item() if sdr_2_1 is not None else None,
        "sdr|3-1": sdr_3_1.item() if sdr_3_1 is not None else None,
        "sdr|3-2": sdr_3_2.item() if sdr_3_2 is not None else None,

        "snr|2-1": snr_2_1.item() if snr_2_1 is not None else None,
        "snr|3-1": snr_3_1.item() if snr_3_1 is not None else None,
        "snr|3-2": snr_3_2.item() if snr_3_2 is not None else None,
        "snr-improved": snr_improved.item() if snr_improved is not None else None,

        "si_snr|2-1": si_snr_2_1.item() if si_snr_2_1 is not None else None,
        "si_snr|3-1": si_snr_3_1.item() if si_snr_3_1 is not None else None,
        "si_snr|3-2": si_snr_3_2.item() if si_snr_3_2 is not None else None,
        "si_snr_improved": si_snr_improved.item() if si_snr_improved is not None else None,

        "nisqa-mos|1": nisqa_1,
        "nisqa-mos|2": nisqa_2,
        "nisqa-mos|3": nisqa_3,
    }

    results[dialog_id] = _result

print("-"*50)
print("Data:")
print("-"*50)

print(results.values())

print("-"*50)
print("Results:")
print("-"*50)

# Save the results to a json file
with open("metrics_augmentation_results.json", "w") as f:
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
results_df.to_csv("metrics_augmentation_results_stats.csv", index=True)
print("Saving results to csv file...")
