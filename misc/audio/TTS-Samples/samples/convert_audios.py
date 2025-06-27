import librosa
import soundfile as sf


def convert_audio(input_path, output_path):

    print(f"Conversion of {input_path}...")

    y, sr = librosa.load(input_path, sr=44100, mono=False)

    if len(y.shape) > 1:
        y = librosa.to_mono(y)

    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=24000)

    sf.write(output_path, y_resampled, 24000, subtype='PCM_16')
    print(f"File converted : {output_path}")


files_to_convert = [
    ("doctor_en_reference.wav", "doctor_en_reference_converted.wav"),
    ("patient_en_reference.wav", "patient_en_reference_converted.wav")
]

for input_file, output_file in files_to_convert:
    convert_audio(input_file, output_file)
