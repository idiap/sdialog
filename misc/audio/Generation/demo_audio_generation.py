import os

from sdialog import Dialog
from sdialog.audio.evaluation import speaker_consistency
from sdialog.audio import dialog_to_audio, to_wav, generate_utterances_audios

dialog = Dialog.from_file("1.json")

dialog.print()

full_audio = dialog_to_audio(dialog)

to_wav(full_audio, "./outputs/first_dialog_audio.wav")

dialog.to_audio("./outputs/first_direct_dialog_audio.wav")

audio_res = dialog.to_audio()

utterances = generate_utterances_audios(dialog)

os.makedirs("./outputs/utterances", exist_ok=True)

for idx, (utterance, speaker) in enumerate(utterances):
    to_wav(utterance, f"./outputs/utterances/{idx}_{speaker}_utterance.wav")

similarity_score = speaker_consistency(utterances)
print(similarity_score)
