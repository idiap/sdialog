"""
This script demonstrates the audio generation capabilities of the sdialog library.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import os
import json

import sdialog
from sdialog import Dialog
from sdialog.audio.tts_engine import KokoroTTS
from sdialog.personas import Doctor, Patient, Agent
from sdialog.audio.voice_database import DummyVoiceDatabase
from sdialog.audio.evaluation import speaker_consistency, eval_wer
from sdialog.audio import dialog_to_audio, to_wav, generate_utterances_audios


dummy_voice_database = DummyVoiceDatabase()
kokoro_tts_pipeline = KokoroTTS()


os.makedirs("./outputs", exist_ok=True)

sdialog.config.set_llm("qwen2.5:14b")

FORCE_DIALOG_GENERATION = False

if FORCE_DIALOG_GENERATION:

    doctor = Agent(
        Doctor(
            name="Dr. Smith",
            gender="male",
            age=52,
            specialty="Family Medicine"
        ),
        name="DOCTOR"
    )
    patient = Agent(
        Patient(
            name="John Doe",
            gender="male",
            age=62
        ),
        name="PATIENT"
    )

    dialog = doctor.talk_with(patient)
    dialog.to_file("dialog_demo.json")

else:
    dialog = Dialog.from_file("dialog_demo.json")

dialog.print()

full_audio = dialog_to_audio(dialog, voice_database=dummy_voice_database, tts_pipeline=kokoro_tts_pipeline)

to_wav(full_audio, "./outputs/first_dialog_audio.wav")

exit(0)

dialog.to_audio("./outputs/first_direct_dialog_audio.wav")

audio_res = dialog.to_audio()

utterances = generate_utterances_audios(dialog)

# os.makedirs("./outputs/utterances", exist_ok=True)

# for idx, (utterance, speaker) in enumerate(utterances):
#     to_wav(utterance, f"./outputs/utterances/{idx}_{speaker}_utterance.wav")

similarity_score = speaker_consistency(utterances)
print("Speaker consistency scores:")
print(similarity_score)

print("WER scores:")
wer_utt = eval_wer(utterances, dialog)
print(json.dumps(wer_utt, indent=4))
