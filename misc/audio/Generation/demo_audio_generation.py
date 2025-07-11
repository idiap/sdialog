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
from sdialog.audio.tts_engine import KokoroTTS, ChatterboxTTS, XttsTTS
from sdialog.audio.evaluation import speaker_consistency, eval_wer, compute_mos
from sdialog.audio.audio_events_enricher import AudioEventsEnricher
from sdialog.audio import dialog_to_audio, to_wav, generate_utterances_audios


dummy_voice_database = DummyVoiceDatabase()
tts_pipeline = KokoroTTS()
# tts_pipeline = ChatterboxTTS()
# tts_pipeline = XttsTTS()


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

def test_audio_enricher(dialog):

    dialog = AudioEventsEnricher.enrich(dialog)

    dialog.print()

    print("Structuring markup language...")
    tags = AudioEventsEnricher.structure_markup_language(dialog)
    print(tags)

    print("Removing markup language...")
    dialog = AudioEventsEnricher.remove_markup_language(dialog)

    dialog.print()

    exit(0)

def test_save_audio_from_utils(dialog):

    full_audio = dialog_to_audio(dialog, voice_database=dummy_voice_database, tts_pipeline=tts_pipeline)

    to_wav(full_audio, "./outputs/first_dialog_audio.wav")

    exit(0)

def test_save_audio_from_dialog(dialog):

    dialog.to_audio("./outputs/first_direct_dialog_audio.wav")

    audio_res = dialog.to_audio()

def test_save_utterances_audios(utts):

    os.makedirs("./outputs/utterances", exist_ok=True)

    for idx, (utterance, speaker) in enumerate(utts):
        to_wav(utterance, f"./outputs/utterances/{idx}_{speaker}_utterance.wav")

def test_eval_speaker_consistency(utterances):
    similarity_score = speaker_consistency(utterances)
    print("Speaker consistency scores:")
    print(similarity_score)

def test_eval_wer(utterances):
    print("WER scores:")
    wer_utt = eval_wer(utterances, dialog)
    print(json.dumps(wer_utt, indent=4))

def test_eval_cloning_adherence(utterances):
    return None

def test_eval_mos(utterances):
    res = compute_mos(utterances, show_figure=True)
    return res

# test_audio_enricher(dialog)
# test_save_audio_from_utils(dialog)
# test_save_audio_from_dialog(dialog)

utterances = generate_utterances_audios(dialog, voice_database=dummy_voice_database, tts_pipeline=tts_pipeline)
test_eval_mos(utterances)
# test_save_utterances_audios(utterances)
# test_eval_speaker_consistency(utterances)
# test_eval_wer(utterances)
# test_eval_cloning_adherence(utterances)
# test_eval_mos(utterances)
