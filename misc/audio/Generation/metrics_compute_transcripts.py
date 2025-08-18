import argparse
import os
import json
import torch
import numpy as np
from transformers import pipeline
from tqdm import tqdm


def compute_transcripts(main_dir, models, args):

    print("main_dir:", main_dir)

    # Initialize the ASR pipelines
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipes = {
        model: pipeline(
            "automatic-speech-recognition",
            model=model,
            device=device,
            torch_dtype=torch.float16
        ) for model in models
    }

    output_dir = "./outputs_transcripts"
    os.makedirs(output_dir, exist_ok=True)

    # Find all dialog directories
    dialog_dirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if
                   os.path.isdir(os.path.join(main_dir, d)) and d.startswith('dialog_')]
    dialog_dirs.sort()

    # Split paths for the current worker
    if args.nbr_worker > 1:
        if args.worker_id >= args.nbr_worker:
            raise ValueError("worker_id must be less than nbr_worker")

        path_splits = np.array_split(dialog_dirs, args.nbr_worker)
        dialog_dirs_to_process = path_splits[args.worker_id]
        print(
            f"Worker {args.worker_id}/{args.nbr_worker} processing "
            f"{len(dialog_dirs_to_process)} of {len(dialog_dirs)} dialogs."
        )
    else:
        dialog_dirs_to_process = dialog_dirs

    # Process each dialog
    for dialog_dir in tqdm(dialog_dirs_to_process, desc="Processing dialogs"):
        dialog_id_from_dir_name = os.path.basename(dialog_dir)
        exported_audios_dir = os.path.join(dialog_dir, "exported_audios")

        if not os.path.exists(exported_audios_dir):
            continue

        # Get the real dialog UUID from audio_pipeline_info.json
        pipeline_info_path = os.path.join(exported_audios_dir, "audio_pipeline_info.json")
        dialog_id = None
        if os.path.exists(pipeline_info_path):
            with open(pipeline_info_path, 'r') as f:
                try:
                    pipeline_info = json.load(f)
                    dialog_id = pipeline_info.get("dialog_id")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {pipeline_info_path}")

        if not dialog_id:
            dialog_id = dialog_id_from_dir_name  # fallback
            print(f"Warning: could not find dialog_id in {pipeline_info_path}, "
                  f"using directory name {dialog_id} instead.")

        dialog_transcripts = []
        # Process each audio step file
        for i in [1, 2, 3]:
            audio_file = os.path.join(exported_audios_dir, f"audio_pipeline_step{i}.wav")
            if os.path.exists(audio_file):
                # Transcribe with each model
                for model_name, pipe in pipes.items():
                    try:
                        result = pipe(audio_file)
                        print("[DONE] Model name:", model_name, ". Step:", f"step{i}")
                        transcript = {
                            "dialog_id": dialog_id,
                            "model": model_name,
                            "step": f"step{i}",
                            "transcript": result["text"]
                        }
                        dialog_transcripts.append(transcript)
                    except Exception as e:
                        print(f"Error processing {audio_file} with {model_name}: {e}")

        # Save transcripts for this dialog to a JSON file
        if dialog_transcripts:
            output_filename = f"transcripts_{dialog_id_from_dir_name}_out_of_{args.nbr_worker}_{args.worker_id}.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump(dialog_transcripts, f, indent=4)
            print(f"Transcripts for dialog {dialog_id} saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute transcripts for dialogue audio files.')
    parser.add_argument(
        '--main_dir',
        type=str,
        default=(
            "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio/"
            "Generation/outputs-voices-libritts-indextts+dscaper+acoustics+metadata"
        ),
        help='Main directory containing dialogue folders.'
    )
    parser.add_argument("--nbr_worker", type=int, default=1, help="Total number of workers.")
    parser.add_argument("--worker_id", type=int, default=0, help="ID of this worker (0-based).")
    args = parser.parse_args()

    models_to_use = [
        "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/models/whisper-tiny",
        "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/models/whisper-base",
        "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/models/whisper-large-v3-turbo"
    ]

    compute_transcripts(args.main_dir, models_to_use, args)

    # python3 metrics_compute_transcripts.py --nbr_worker=30 --worker_id=0
