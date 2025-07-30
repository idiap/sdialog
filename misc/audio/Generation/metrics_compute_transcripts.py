import argparse
import os
import json
import torch
from transformers import pipeline
from tqdm import tqdm


def compute_transcripts(main_dir, models):

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

    # Find all dialog directories
    dialog_dirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if
                   os.path.isdir(os.path.join(main_dir, d)) and d.startswith('dialog_')]

    print(dialog_dirs)

    all_transcripts = []

    # Process each dialog
    for dialog_dir in tqdm(dialog_dirs, desc="Processing dialogs"):
        dialog_id = os.path.basename(dialog_dir).split('_')[1]
        print(dialog_id)
        exported_audios_dir = os.path.join(dialog_dir, "exported_audios")
        print(exported_audios_dir)

        if not os.path.exists(exported_audios_dir):
            continue

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
                        all_transcripts.append(transcript)
                    except Exception as e:
                        print(f"Error processing {audio_file} with {model_name}: {e}")

    # Save all transcripts to a single JSON file
    if all_transcripts:
        output_dir = "./outputs_transcripts"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "transcripts.json")
        with open(output_path, 'w') as f:
            json.dump(all_transcripts, f, indent=4)
        print(f"All transcripts saved to {output_path}")


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
    args = parser.parse_args()

    models_to_use = [
        "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/models/whisper-tiny",
        "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/models/whisper-base",
        "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/models/whisper-large-v3-turbo"
    ]

    compute_transcripts(args.main_dir, models_to_use)
