from TTS.api import TTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Chargement du modèle...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=torch.cuda.is_available())
print("Modèle chargé.")

# Données du dialogue
data = [
    {
        "text": ("Thank you for sharing that. So have you experienced any other symptoms along"
                 " with the chest pain? Such as shortness of breath, sweating, nausea, or dizziness?"),
        "role": "doctor",
        "voice": "doctor_en_reference_converted.wav"
    },
    {
        "text": ("Yes, I feel short of breath, and I'm sweating a lot. I also feel a bit"
                 " nauseated, but I haven't vomited."),
        "role": "patient",
        "voice": "patient_en_reference_converted.wav"
    },
]

for i, entry in enumerate(data):

    output_filename = f"xttsv2_clone_{entry['role']}.wav"

    tts.tts_to_file(
        text=entry['text'],
        file_path=output_filename,
        speaker_wav=entry['voice'],
        language="en"
    )
    print(f"Fichier généré : {output_filename}")

print("\nSynthèse vocale du dialogue terminée !")
