import os

DIR_PATH = (
    "/lustre/fsn1/projects/rech/rtl/uaj63yz/JSALT2025/sdialog/misc/audio/Generation/"
    "outputs-voices-libritts-indextts+dscaper+acoustics+metadata"
)

for dir_dialog in os.listdir(DIR_PATH):

    dialog_id = dir_dialog.split("_")[0]
    print(dialog_id)

    # Get the exported audios paths
    path_exported = os.path.join(DIR_PATH, dir_dialog, "exported_audios")
    json_path = os.path.join(path_exported, "audio_pipeline_info.json")
    wav_path_1 = os.path.join(path_exported, "audio_pipeline_step1.wav")
    wav_path_2 = os.path.join(path_exported, "audio_pipeline_step2.wav")
    wav_path_3 = os.path.join(path_exported, "audio_pipeline_step3.wav")

    # Get all the utterances paths
    path_utterances = os.path.join(DIR_PATH, dir_dialog, "utterances")
    all_utterances_paths = [os.path.join(path_utterances, _) for _ in os.listdir(path_utterances)]

    for utterance_path in all_utterances_paths:
        utterance_id, role = utterance_path.replace(".wav", "").split("_")
        print(utterance_id, role)
