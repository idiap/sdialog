import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

data = [
    {
        "text": ("Thank you for sharing that. So have you experienced any other symptoms along with the chest pain?"
                 " Such as shortness of breath, sweating, nausea, or dizziness?"),
        "role": "doctor",
        "voice": "doctor_en_reference_converted.wav"
    },
    {
        "text": ("Yes, I feel short of breath, and I'm sweating a lot. I also feel a bit nauseated, but "
                 "I haven't vomited."),
        "role": "patient",
        "voice": "patient_en_reference_converted.wav"
    },
]

for d in data:

    text = d["text"]
    role = d["role"]
    wav = model.generate(text)
    ta.save(f"chatterbox_{role}.wav", wav, model.sr)

    # Voice Cloning
    wav = model.generate(text, audio_prompt_path=d["voice"])
    ta.save(f"chatterbox_clone_{role}.wav", wav, model.sr)
