from sdialog import Dialog
from sdialog.audio.evaluation import speaker_consistency
from sdialog.audio import dialog_to_audio, to_wav, generate_utterances_audios

dialog = Dialog.from_file("1.json")

dialog.print()

full_audio = dialog_to_audio(dialog)

to_wav(full_audio, "first_dialog_audio.wav")

dialog.to_audio("first_direct_dialog_audio.wav")

audio_res = dialog.to_audio()

similarity_score = speaker_consistency(
    generate_utterances_audios(dialog)
)
print(similarity_score)
