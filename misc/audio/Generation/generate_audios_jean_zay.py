import os
from tqdm import tqdm

import sdialog
from sdialog import Dialog
from sdialog.audio.tts_engine import IndexTTS
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.audio_pipeline import AudioPipeline
from sdialog.audio.voice_database import HuggingfaceVoiceDatabase

sdialog.config.set_llm("aws:anthropic.claude-3-5-sonnet-20240620-v1:0", region_name="us-east-1")

os.makedirs("./outputs", exist_ok=True)

path_dir = "./200-dialogues-V0"

# from sdialog.audio.voice_database import DummyKokoroVoiceDatabase
# dummy_voice_database = DummyKokoroVoiceDatabase()
# dummy_voice_database.get_voice(genre="male", age=20)

# from sdialog.audio.voice_database import HuggingfaceVoiceDatabase
# voices_libritts = HuggingfaceVoiceDatabase("sdialog/voices-libritts")
# voices_libritts.get_voice(genre="male", age=20)

dummy_voice_database = HuggingfaceVoiceDatabase("./data/voices-jsalt")
dummy_voice_database.get_voice(genre="male", age=20)

# from sdialog.audio.tts_engine import KokoroTTS
# tts_pipeline = KokoroTTS()

tts_pipeline = IndexTTS(device="cpu")

# from sdialog.audio.audio_dialog import AudioDialog
# from sdialog.audio.audio_pipeline import AudioPipeline

audio_pipeline = AudioPipeline(
    voice_database=dummy_voice_database,
    tts_pipeline=tts_pipeline,
    dir_audio="./outputs",
)

for dialog_path in tqdm(os.listdir(path_dir)):

    full_path = os.path.join(path_dir, dialog_path)

    print(full_path)

    original_dialog = Dialog.from_file(full_path)

    original_dialog.print()

    dialog: AudioDialog = AudioDialog.from_dialog(original_dialog)
    # audio_pipeline = AudioPipeline() # Default values are used

    dialog: AudioDialog = audio_pipeline.inference(dialog) # Generate the audio for the dialog
    print(dialog.audio_step_1_filepath) # Path to the audio of the first stage of the audio pipeline

    continue

    import scaper
    DATA_PATH = "./dscaper_data" # Path where the sound events, utterances and timelines database will be saved
    os.makedirs(DATA_PATH, exist_ok=True)

    # %%
    dsc = scaper.Dscaper(dscaper_base_path=DATA_PATH)

    # %%
    audio_pipeline = AudioPipeline(dscaper=dsc)

    # %%
    # Populate the sound events database
    audio_pipeline.populate_dscaper(["sdialog/background","sdialog/foreground"])

    # %%
    dialog: AudioDialog = audio_pipeline.inference(dialog)
    print(dialog.audio_step_1_filepath)
    print(dialog.audio_step_2_filepath)

    # %% [markdown]
    # ## Step 3 : Room Accoustics

    # %%
    audio_pipeline = AudioPipeline(dscaper=dsc) # The audio pipeline doesn't change

    # %%
    from sdialog.audio.room import MicrophonePosition
    from sdialog.audio.room_generator import RoomGenerator, RoomRole

    # %%
    room = RoomGenerator().generate(RoomRole.CONSULTATION)
    print(room)

    # %%
    dialog: AudioDialog = audio_pipeline.inference(
        dialog,
        room=room, # Need to provide a room object to trigger the 3rd step of the audio pipeline
        # microphone_position=MicrophonePosition.MONITOR # Default is MicrophonePosition.CEILING_CENTERED
    )
    print(dialog.audio_step_1_filepath)
    print(dialog.audio_step_2_filepath)
    print(dialog.audio_step_3_filepath)

    # %% [markdown]
    # ## Automatic Position and SNR completion (under construction)

    # %%
    from sdialog.audio.audio_events_enricher import AudioEventsEnricher
    enricher = AudioEventsEnricher()

    # %%
    audio_pipeline = AudioPipeline(enricher=enricher)

    # %%
    for turn in dialog.turns:
        print(turn.text)
        print(turn.position)
        print(turn.snr)
        print("____________________")

    # %% [markdown]
    # ## dialog.to_audio()

    # %%
    # TODO: Add a demo of the audio generation pipeline from the dialog object

    # %% [markdown]
    # # Audio Evaluation

    # %% [markdown]
    # ## Utterance level evaluation

    # %%
    from sdialog.audio.evaluation import compute_evaluation_utterances, compute_evaluation_audio

    # %%
    # Utterances level evaluation
    metrics_utterances_level = compute_evaluation_utterances(dialog)
    for key, value in metrics_utterances_level.items():
        print(f"{key}: {value}")

    # Audio level evaluation
    metrics_audio_level = compute_evaluation_audio(dialog)
    for key, value in metrics_audio_level.items():
        print(f"{key}: {value}")

    # %% [markdown]
    # ## Audio level evaluation

    # %%
    # TODO: Evaluate the final generated audio


