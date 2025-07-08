import os
import json

from sdialog import Dialog
from sdialog.audio.evaluation import speaker_consistency, eval_wer
from sdialog.audio import dialog_to_audio, to_wav, generate_utterances_audios

os.makedirs("./outputs", exist_ok=True)

dialog = Dialog.from_file("1.json")

dialog.print()

full_audio = dialog_to_audio(dialog)

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
