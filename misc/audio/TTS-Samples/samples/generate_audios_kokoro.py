import soundfile as sf
from kokoro import KPipeline

data = [
    {
        "text": ("Thank you for sharing that. So have you experienced any other symptoms along"
                 " with the chest pain? Such as shortness of breath, sweating, nausea, or dizziness?"),
        "voice": "am_fenrir",
        "role": "doctor"
    },
    {
        "text": ("Yes, I feel short of breath, and I'm sweating a lot. "
                 "I also feel a bit nauseated, but I haven't vomited."),
        "voice": "af_heart",
        "role": "patient"
    },
]
# females = ["af_heart","af_bella"]
# males = ["am_fenrir","am_michael","am_puck"]

pipeline = KPipeline(lang_code='a')

for d in data:

    generator = pipeline(d["text"], voice=d["voice"])
    gs, ps, audio = next(iter(generator))

    output_file = f"kokoro_{d['role']}.wav"
    sf.write(output_file, audio, 24_000)
    print(f"Audio saved to {output_file}")
