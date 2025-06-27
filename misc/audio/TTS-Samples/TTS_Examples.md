# Samples of TTS models

We generated audio samples for the following sentence in order to mimic the voice of the original doctor and patient:
Doctor: **"Thank you for sharing that. So have you experienced any other symptoms along with the chest pain? Such as shortness of breath, sweating, nausea, or dizziness?"**
Patient: **"Yes, I feel short of breath, and I'm sweating a lot. I also feel a bit nauseated, but I haven't vomited."**

## References samples for voice cloning:
- Doctor said **"So I understand from the emergency room records that you are coming in with a pneumonia. Was there something else that also caused you to come in to day?"**: <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/doctor_en_reference.wav" type="audio/wav"></audio>
- Patient said **"I'd say it was about maybe two weeks ago that I had really bad cold and I started coughing a real lot."**: <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/patient_en_reference.wav" type="audio/wav"></audio>

## TTS models outputs:

| Model name  | Doctor | Patient |
|-------------|--------|---------|
|  Kokoro (w/o cloning)     |  <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/kokoro_doctor.wav" type="audio/wav"></audio>      |  <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/kokoro_patient.wav" type="audio/wav"></audio>       |
|  CosyVoice2 | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/cosyvoice2_doctor.wav" type="audio/wav"></audio>       | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/cosyvoice2_patient.wav" type="audio/wav"></audio>        |
|  ZipVoice   | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/zipvoice_doctor.wav" type="audio/wav"></audio>       | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/zipvoice_patient.wav" type="audio/wav"></audio>        |
|  ChatterBox   | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/chatterbox_clone_doctor.wav" type="audio/wav"></audio>       | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/chatterbox_clone_patient.wav" type="audio/wav"></audio>        |
|  XTTSv2   | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/xttsv2_clone_doctor.wav" type="audio/wav"></audio>       | <audio controls><source src="https://huggingface.co/datasets/Play-Your-Part/SamplesTTS/resolve/main/xttsv2_clone_patient.wav" type="audio/wav"></audio>        |

## Feedback

- Chatterbox: Doesn't pace the speed of the original voice.
