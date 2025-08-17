# module purge

# module load arch/h100
# module load cuda/12.4.1
# module load ffmpeg/6.1.1

# module load miniforge

# conda activate jsalt10

import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import scaper
import sdialog
from sdialog import Dialog
from sdialog.audio.tts_engine import IndexTTS
from sdialog.audio.room import MicrophonePosition
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.audio_pipeline import AudioPipeline
from sdialog.audio.room_generator import RoomGenerator, RoomRole
from sdialog.audio.voice_database import DummyIndexTtsVoiceDatabase

# python generate_audios_jean_zay+dscaper+accoustics.py --nbr_worker=30 --worker_id=0

sdialog.config.set_llm("aws:anthropic.claude-3-5-sonnet-20240620-v1:0", region_name="us-east-1")

parser = argparse.ArgumentParser(description="Generate audios in parallel.")
parser.add_argument("--nbr_worker", type=int, default=1, help="Total number of workers.")
parser.add_argument("--worker_id", type=int, default=0, help="ID of this worker (0-based).")
args = parser.parse_args()

path_dir = "./200-dialogues-V0"

dummy_voice_database = DummyIndexTtsVoiceDatabase(
    path_dir="/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/cache_hf/libritts-voices/"
)
dummy_voice_database.get_voice(genre="male", age=20)

print(dummy_voice_database.get_data())

tts_pipeline = IndexTTS(device="cpu")

DATA_PATH = "./dscaper_data"  # Path where the sound events, utterances and timelines database will be saved
os.makedirs(DATA_PATH, exist_ok=True)

dsc = scaper.Dscaper(dscaper_base_path=DATA_PATH)

rooms_configs = [
    (RoomGenerator().generate(RoomRole.CONSULTATION, room_size=9.5), "consultation_9-5"),
    (RoomGenerator().generate(RoomRole.CONSULTATION, room_size=12), "consultation_12"),
    (RoomGenerator().generate(RoomRole.CONSULTATION, room_size=15), "consultation_15"),
]

microphone_positions = [
    (MicrophonePosition.CEILING_CENTERED, "ceiling_centered"),
    (MicrophonePosition.MONITOR, "webcam")
]

for current_room, current_room_name in rooms_configs:

    for current_microphone_position, current_microphone_position_name in microphone_positions:

        run_dir = f"./outputs|{current_room_name}|{current_microphone_position_name}"

        if not os.path.exists(run_dir):
            shutil.copytree(
                "./outputs-voices-libritts-indextts+dscaper+acoustics+metadata",
                run_dir
            )

        audio_pipeline = AudioPipeline(
            voice_database=dummy_voice_database,
            tts_pipeline=tts_pipeline,
            dir_audio=run_dir,
            dscaper=dsc
        )

        # audio_pipeline.populate_dscaper([
        #     (
        #         "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/"
        #         f"sdialog/misc/audio/Generation/hf_dscaper/{sound_bank_name}"
        #     )
        #     for sound_bank_name in ["foreground", "background"]
        # ])
        # audio_pipeline.populate_dscaper(["sdialog/background", "sdialog/foreground"])

        paths = [_ for _ in os.listdir(path_dir) if ".json" in _]
        paths.sort()

        # Split paths for the current worker
        if args.nbr_worker > 1:
            if args.worker_id >= args.nbr_worker:
                raise ValueError("worker_id must be less than nbr_worker")

            path_splits = np.array_split(paths, args.nbr_worker)
            paths_to_process = path_splits[args.worker_id]
            print(
                f"Worker {args.worker_id}/{args.nbr_worker} processing {len(paths_to_process)} of {len(paths)} dialogs."
            )
        else:
            paths_to_process = paths

        for dialog_path in tqdm(paths_to_process):

            full_path = os.path.join(path_dir, dialog_path)

            print(full_path)

            original_dialog = Dialog.from_file(full_path)

            original_dialog.print()

            dialog: AudioDialog = AudioDialog.from_dialog(original_dialog)
            # audio_pipeline = AudioPipeline()  # Default values are used

            print(current_room)

            # Generate the audio for the dialog
            dialog: AudioDialog = audio_pipeline.inference(
                dialog,
                room=current_room,
                microphone_position=current_microphone_position,
                do_step_1=False,
                do_step_2=False,
                do_step_3=True
            )
            print(dialog.audio_step_1_filepath)
            print(dialog.audio_step_2_filepath)
            print(dialog.audio_step_3_filepath)
