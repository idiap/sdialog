import os
import argparse
import re
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import json


def get_speaker_info(libritts_path: str) -> Dict[int, Dict[str, str]]:
    """
    Reads speakers.tsv to get speaker IDs and their corresponding information.

    Args:
        libritts_path: Path to the LibriTTS dataset root.

    Returns:
        A dictionary mapping speaker IDs to their information (gender, subset, name).
    """
    speakers_file = os.path.join(libritts_path, 'speakers.tsv')
    print(f"--> Reading speaker info from: {speakers_file}")
    if not os.path.exists(speakers_file):
        raise FileNotFoundError(f"speakers.tsv not found in {libritts_path}")

    try:
        # The speakers.tsv file in LibriTTS can have inconsistent spacing, and the
        # NAME column can contain spaces. A robust way to parse this is to read
        # it line by line and split carefully.
        records = []
        with open(speakers_file, 'r') as f:
            # Skip any leading comments or blank lines
            for line in f:
                if line.strip() and not line.strip().startswith(';'):
                    # We found the first non-comment/empty line. We assume it's a header but ignore it
                    # in favor of a fixed column structure, which is more robust.
                    break

            # The columns are assumed to be: READER, GENDER, SUBSET, NAME
            columns = ['READER', 'GENDER', 'SUBSET', 'NAME']

            # Process rest of the file
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue

                parts = re.split(r'\s+', line, maxsplit=3)
                if len(parts) == 4:
                    records.append(dict(zip(columns, parts)))

        if not records:
            raise ValueError("Could not parse any speaker information from speakers.tsv")

        df = pd.DataFrame(records)

        df['READER'] = pd.to_numeric(df['READER'], errors='coerce')
        df.dropna(subset=['READER'], inplace=True)
        df['READER'] = df['READER'].astype(int)

        return df.set_index('READER').to_dict(orient='index')
    except Exception as e:
        print(f"Error reading or parsing {speakers_file}: {e}")
        raise


def find_audio_files(base_path: str, speaker_id: int, subset: str) -> List[str]:
    """
    Finds all .wav audio files for a given speaker in their subset.

    Args:
        base_path: Path to the LibriTTS dataset root.
        speaker_id: The speaker's ID.
        subset: The subset directory name (e.g., 'train-clean-100').

    Returns:
        A sorted list of paths to the audio files.
    """
    speaker_path = os.path.join(base_path, subset, str(speaker_id))
    audio_files = []
    if os.path.isdir(speaker_path):
        for root, _, files in os.walk(speaker_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
    return sorted(audio_files)


def create_voice_database(libritts_path: str, output_dir: str, duration_s: int = 30):
    """
    Extracts a fixed-length audio clip for each speaker from the LibriTTS dataset.

    Args:
        libritts_path: Path to the LibriTTS dataset root.
        output_dir: Directory to save the extracted audio clips.
        duration_s: Desired duration of the audio clips in seconds.
    """
    print("Starting voice database creation...")
    print(f"  > LibriTTS path: {libritts_path}")
    print(f"  > Output directory: {output_dir}")
    print(f"  > Clip duration: {duration_s}s")
    os.makedirs(output_dir, exist_ok=True)
    audio_output_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_output_dir, exist_ok=True)
    speaker_info = get_speaker_info(libritts_path)
    print(f"--> Found {len(speaker_info)} speakers to process.")

    all_speakers_metadata = []
    missing_speakers = []
    speaker_iterator = tqdm(speaker_info.items(), desc="Processing speakers")
    for speaker_id, info in speaker_iterator:
        subset = info['SUBSET']
        speaker_name = info['NAME'].replace(' ', '_')
        speaker_iterator.set_description(f"Processing speaker {speaker_id} ({subset})")
        audio_files = find_audio_files(libritts_path, speaker_id, subset)

        if not audio_files:
            missing_speakers.append((speaker_id, subset))
            continue

        concatenated_audio = []
        used_utterances_info = []
        current_duration_samples = 0
        sample_rate = -1

        for audio_file in audio_files:
            # Assume 24kHz if not known to estimate target samples before first read
            current_target_sr = sample_rate if sample_rate != -1 else 24000
            target_samples = duration_s * current_target_sr
            if current_duration_samples >= target_samples and sample_rate != -1:
                break
            try:
                file_info = sf.info(audio_file)
                if sample_rate == -1:
                    sample_rate = file_info.samplerate
                elif sample_rate != file_info.samplerate:
                    tqdm.write(f"Warning: Skipping {audio_file}, inconsistent sample rate.")
                    continue

                audio, sr = sf.read(audio_file, dtype='float32')
                concatenated_audio.append(audio)

                utterance_duration_s = len(audio) / sr
                used_utterances_info.append({
                    "path": audio_file,
                    "duration_s": utterance_duration_s
                })
                current_duration_samples += len(audio)

            except Exception as e:
                tqdm.write(f"Error processing file {audio_file}: {e}")

        if not concatenated_audio or sample_rate == -1:
            tqdm.write(f"Warning: Could not process any audio for speaker {speaker_id}")
            continue

        final_audio = np.concatenate(concatenated_audio)
        target_samples = duration_s * sample_rate

        if len(final_audio) > target_samples:
            final_audio = final_audio[:target_samples]
        elif len(final_audio) < target_samples:
            padding = np.zeros(target_samples - len(final_audio), dtype=np.float32)
            final_audio = np.concatenate([final_audio, padding])

        final_duration_s = len(final_audio) / sample_rate
        output_filename = os.path.join(audio_output_dir, f"{speaker_id}_{speaker_name}.wav")
        sf.write(output_filename, final_audio, sample_rate)

        speaker_metadata = {
            "speaker_id": speaker_id,
            "gender": info['GENDER'],
            "name": info['NAME'],
            "subset": subset,
            "age": -1,
            "output_filename": output_filename,
            "total_duration_s": final_duration_s,
            "used_utterances": used_utterances_info
        }
        all_speakers_metadata.append(speaker_metadata)

    # Save metadata to a CSV file
    if all_speakers_metadata:
        df = pd.DataFrame(all_speakers_metadata)
        df.rename(columns={"output_filename": "file_name"}, inplace=True)
        df["file_name"] = df["file_name"].apply(lambda x: os.path.relpath(x, output_dir))
        # The 'used_utterances' column contains complex objects (list of dicts).
        # For CSV, it's best to serialize it to a JSON string.
        df['used_utterances'] = df['used_utterances'].apply(json.dumps)

        metadata_path = os.path.join(output_dir, "metadata.csv")
        df.to_csv(metadata_path, index=False)
        print(f"--> Metadata for {len(all_speakers_metadata)} speakers saved to {metadata_path}")
    else:
        print("No speaker data processed, metadata file not created.")

    # Print summary of missing speakers
    if missing_speakers:
        print(f"\nWarning: Could not find audio files for {len(missing_speakers)} of the "
              f"{len(speaker_info)} total speakers listed in the master file.")
        print("         This can be expected if you only have a subset of the LibriTTS corpus downloaded.")

    print(f"\nVoice database creation complete. {len(all_speakers_metadata)} speaker clips created.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a voice database by extracting 30s clips for each speaker from LibriTTS."
    )
    parser.add_argument(
        "--libritts_path",
        type=str,
        default="/lustre/fsmisc/dataset/LibriTTS",
        help="Path to the root of the LibriTTS dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the speaker audio clips."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration of the speech clips in seconds."
    )
    args = parser.parse_args()

    create_voice_database(args.libritts_path, args.output_dir, args.duration)


if __name__ == "__main__":
    main()
