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
from sdialog.personas import Doctor, Patient, Agent
from sdialog.audio.voice_database import DummyVoiceDatabase
from sdialog.audio.tts_engine import KokoroTTS
# ChatterboxTTS, XttsTTS
from sdialog.audio.audio_events_enricher import AudioEventsEnricher
from sdialog.audio import dialog_to_audio, to_wav, generate_utterances_audios
from sdialog.audio.evaluation import compute_evaluation_utterances, compute_evaluation_audio


dummy_voice_database = DummyVoiceDatabase()
tts_pipeline = KokoroTTS()
# tts_pipeline = ChatterboxTTS()
# tts_pipeline = XttsTTS()

save_all = False

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

audio_res = dialog_to_audio(dialog, voice_database=dummy_voice_database, tts_pipeline=tts_pipeline)

if save_all:

    # Saving individual utterances audios
    os.makedirs("./outputs/utterances", exist_ok=True)
    for idx, (utterance, speaker) in enumerate(audio_res["utterances_audios"]):
        to_wav(utterance, f"./outputs/utterances/{idx}_{speaker}_utterance.wav")

    # Saving audio step 1: no room accoustics and audios events
    to_wav(audio_res["audio"], "./outputs/dialog_audio_step1.wav")

# Enriching the dialog with audio events and generate the timeline of audio events and utterances
enricher = AudioEventsEnricher()
timeline = enricher.extract_events(dialog, audio_res["utterances_audios"])
timeline.print()

# Drawing the timeline
timeline.draw("./outputs/timeline.png")

# compute_evaluation_utterances(audio_res["utterances_audios"], dialog)
# compute_evaluation_audio(audio_res["audio"], dialog)
