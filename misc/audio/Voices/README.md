---
license: apache-2.0
language:
- en
tags:
- audio
- speaker-identification
- speaker-verification
- text-to-speech
- voice-cloning
- libritts
pretty_name: "LibriTTS Speaker Voices & Embeddings"
---

# LibriTTS Speaker Voices & Embeddings

## Dataset Description

This dataset provides a collection of speaker voice samples from the [LibriTTS](http://www.openslr.org/60/) corpus. For each speaker, a single 30-second audio clip is provided, created by concatenating their speech segments.

The dataset is designed for tasks such as speaker identification, speaker verification, and as a voice bank for Text-to-Speech (TTS) models, particularly for voice cloning.

In addition to the audio files and their metadata, this dataset includes pre-computed speaker embeddings (x-vectors) for each speaker, generated using the `pyannote/embedding` model.

### Dataset Components

1.  **Audio Clips**: 30-second `.wav` files for each speaker.
2.  **Metadata**: A `metadata.csv` file linking audio files to speaker information (ID, gender, name, etc.).
3.  **Speaker Embeddings**: A `xvectors.pkl` file containing a dictionary that maps each `speaker_id` to its corresponding embedding vector.

## How to use

### Loading the Dataset

The main dataset, containing the audio and metadata, can be loaded using the `datasets` library.

```python
from datasets import load_dataset

# Load the dataset (replace with the correct name on the Hub)
hf_dataset = "sdialog/voices-libritts" 
dataset = load_dataset(hf_dataset)["train"]

# Access a sample
print(dataset[0])
```

Output:
```python
{
    'speaker_id': 14,
    'gender': 'F',
    'name': 'Kristin LeMoine',
    'subset': 'train-clean-360',
    'age': -1,
    'audio': {'path': '...', 'array': array([...]), 'sampling_rate': 24000},
    'total_duration_s': 30.0,
    'used_utterances': '[{"path": "...", "duration_s": 9.02}, ...]'
}
```

### Loading Speaker Embeddings (x-vectors)

The `xvectors.pkl` file is stored in the dataset repository. You can download and load it using `huggingface_hub`.

```python
import pickle
from huggingface_hub import hf_hub_download
import numpy as np

# Replace with your dataset repository ID
repo_id = "sdialog/voices-libritts" 

# Download the pickle file from the hub
pickle_file = hf_hub_download(repo_id=repo_id, filename="xvectors.pkl")

# Load the x-vectors
with open(pickle_file, "rb") as f:
    xvectors = pickle.load(f)

# Access an embedding
speaker_id = 14
if speaker_id in xvectors:
    embedding = xvectors[speaker_id]
    print(f"Embedding for speaker {speaker_id} has shape: {embedding.shape}")
else:
    print(f"Speaker {speaker_id} not found in x-vectors.")

# All embeddings can be stacked into a matrix
all_speaker_ids = sorted(xvectors.keys())
all_embeddings = np.array([xvectors[sid] for sid in all_speaker_ids])
print(f"Shape of stacked embeddings matrix: {all_embeddings.shape}")
```

## Dataset Structure

### Data Fields

The dataset contains the following fields:

*   `speaker_id` (int): A unique identifier for each speaker, corresponding to the LibriTTS speaker ID.
*   `gender` (string): The speaker's gender ('M' or 'F').
*   `name` (string): The speaker's name.
*   `subset` (string): The LibriTTS subset from which the speaker's audio originates (e.g., 'train-clean-100').
*   `age` (int): The speaker's age. This information was not available in the source material and is set to `-1` for all entries.
*   `audio` (Audio): A 30-second audio clip for the speaker, with a sampling rate of 24kHz.
*   `total_duration_s` (float): The final duration of the audio clip in seconds (should be 30.0).
*   `used_utterances` (string): A JSON string containing a list of the original LibriTTS utterance files that were concatenated to create the speaker's audio clip.

### Data Files

*   `./audio/`: A directory containing all the speaker audio clips in `.wav` format.
*   `metadata.csv`: The CSV file containing the metadata for all speakers. The columns correspond to the data fields described above, with `file_name` pointing to the relative path of the audio file.
*   `xvectors.pkl`: A Python pickle file containing a dictionary. The keys are the `speaker_id` (int) and the values are the corresponding speaker embeddings (as NumPy arrays).

## Dataset Creation

### Source Data

The dataset was created from the [LibriTTS](http://www.openslr.org/60/) corpus.

### Preprocessing

The generation process is carried out by the script `misc/audio/Voices/create_libritts_voice_database_csv.py`.

For each speaker in LibriTTS:
1.  Their `.wav` utterance files are located from their respective subset directories.
2.  Utterances are concatenated one by one until a total duration of at least 30 seconds is reached.
3.  The concatenated audio is then trimmed to exactly 30 seconds. If the total audio is less than 30 seconds, it is padded with silence to reach 30 seconds.
4.  The final audio is saved as a single `.wav` file.

The generation process resulted in 2455 speaker clips. During creation, audio files for 28 out of 2483 speakers listed in the master file could not found 30 seconds of audio. This is expected if you are using a subset of the full LibriTTS corpus. The final metadata file contains information for the 2455 processed speakers.

### Embeddings Calculation

The speaker embeddings (x-vectors) were calculated using the `misc/audio/Voices/xvectors_viz_speakers.ipynb` notebook.
The pre-trained `pyannote/embedding` model from `pyannote.audio` was used to extract one embedding for each 30-second audio clip. The resulting embeddings were stored in the `xvectors.pkl` file.

## Citations

If you use this dataset, please cite the original LibriTTS corpus, as well as `pyannote.audio`.

```bibtex
@inproceedings{zen19_interspeech,
  title     = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  author    = {Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
  year      = {2019},
  booktitle = {Interspeech 2019},
  pages     = {1526--1530},
  doi       = {10.21437/Interspeech.2019-2441},
  issn      = {2958-1796},
}

@INPROCEEDINGS{9052974,
  author={Bredin, Hervé and Yin, Ruiqing and Coria, Juan Manuel and Gelly, Gregory and Korshunov, Pavel and Lavechin, Marvin and Fustes, Diego and Titeux, Hadrien and Bouaziz, Wassim and Gill, Marie-Philippe},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Pyannote.Audio: Neural Building Blocks for Speaker Diarization}, 
  year={2020},
  volume={},
  number={},
  pages={7124-7128},
  keywords={Voice activity detection;Conferences;Pipelines;Machine learning;Signal processing;Acoustics;Open source software;speaker diarization;voice activity detection;speaker change detection;overlapped speech detection;speaker embedding},
  doi={10.1109/ICASSP40776.2020.9052974}}
```

## License

This dataset is released under the Apache 2.0 License. 