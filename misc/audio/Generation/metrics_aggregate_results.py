import os
import json
import jiwer
import argparse
import pandas as pd
from tqdm import tqdm
from sdialog.audio.whisper_normalizer import EnglishTextNormalizer


def aggregate_results(transcripts_dir, main_output_dir):
    """
    Parses transcript JSON files, computes WER and CER against ground truth,
    and returns a summary of the results.
    """
    # Load all ground truth dialogs first
    ground_truth_dialogs = {}
    print("Loading ground truth dialogs...")
    ground_truth_files = []
    for root, _, files in os.walk(main_output_dir):
        for file in files:
            if file.startswith('dialog_') and file.endswith('.json'):
                ground_truth_files.append(os.path.join(root, file))

    for gt_path in tqdm(ground_truth_files, desc="Loading ground truth files"):
        with open(gt_path, 'r') as f:
            dialog_data = json.load(f)
        dialog_uuid = dialog_data.get("id")
        if dialog_uuid:
            ground_truth_dialogs[dialog_uuid] = dialog_data
    print(f"Loaded {len(ground_truth_dialogs)} ground truth dialogs.")

    # Find all transcript files
    transcript_files = []
    for root, _, files in os.walk(transcripts_dir):
        for file in files:
            if file.endswith('.json'):
                transcript_files.append(os.path.join(root, file))

    grouped_texts = {}

    # Process each transcript file
    for file_path in tqdm(transcript_files, desc="Processing transcript files"):
        with open(file_path, 'r') as f:
            transcripts = json.load(f)

        for transcript_data in transcripts:
            dialog_id = transcript_data.get("dialog_id")
            step = transcript_data.get("step")
            model = transcript_data.get("model")
            hypothesis = transcript_data.get("transcript")

            if not all([dialog_id, step, model, hypothesis]):
                continue

            # Extract short model name
            model_name = os.path.basename(model)

            # Get ground truth from pre-loaded dialogs
            ground_truth_info = ground_truth_dialogs.get(dialog_id)

            if not ground_truth_info:
                print(f"Warning: Ground truth not found for dialog {dialog_id}")
                continue

            # Extract ground truth text for the corresponding step
            turns = ground_truth_info.get("turns", [])
            normalizer = EnglishTextNormalizer()

            reference_raw = " ".join([turn.get("text", "") for turn in turns])

            # print("Reference raw:", reference_raw)
            reference = normalizer(reference_raw)
            # print("Reference:", reference)
            hypothesis_normalized = normalizer(hypothesis)
            # print("Hypothesis normalized:", hypothesis_normalized)

            key = (model_name, step)
            if key not in grouped_texts:
                grouped_texts[key] = {'references': [], 'hypotheses': []}

            grouped_texts[key]['references'].append(reference)
            grouped_texts[key]['hypotheses'].append(hypothesis_normalized)

    if not grouped_texts:
        print("No results to aggregate.")
        return

    # Process grouped texts to compute overall metrics
    aggregated_results = []
    for (model, step), texts in grouped_texts.items():
        all_references = " ".join(texts['references'])
        all_hypotheses = " ".join(texts['hypotheses'])

        # Compute overall WER and CER
        wer = jiwer.wer(all_references, all_hypotheses) * 100
        cer = jiwer.cer(all_references, all_hypotheses) * 100

        aggregated_results.append({
            "model": model,
            "step": step,
            "wer": wer,
            "cer": cer,
            "count": len(texts['references'])
        })

    # Create a DataFrame for a clean display
    df = pd.DataFrame(aggregated_results)
    df = df.sort_values(by=['model', 'step']).reset_index(drop=True)

    print("\n--- Aggregated Results ---")
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate ASR transcript results.')
    parser.add_argument(
        '--transcripts_dir',
        type=str,
        default='outputs_transcripts',
        help='Directory containing the transcript JSON files.'
    )
    parser.add_argument(
        '--main_output_dir',
        type=str,
        default=(
            "200-dialogues-V0"
        ),
        help='Main directory containing dialogue folders with ground truth.'
    )
    args = parser.parse_args()

    # Ensure you have the necessary packages installed:
    # pip install pandas jiwer tqdm

    aggregate_results(args.transcripts_dir, args.main_output_dir)

    # python3 metrics_aggregate_results.py --transcripts_dir=outputs_transcripts --main_output_dir=200-dialogues-V0
