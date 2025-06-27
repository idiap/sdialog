python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice" \
    --prompt-wav doctor_en_reference_converted.wav \
    --prompt-text "So I understand from the emergency room records that you were coming in with a pneumonia. Was there something else that also caused you to come in today?" \
    --text "Thank you for sharing that. So have you experienced any other symptoms along with the chest pain? Such as shortness of breath, sweating, nausea, or dizziness?" \
    --res-wav-path result_doctor.wav

python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice" \
    --prompt-wav patient_en_reference_converted.wav \
    --prompt-text "I'd say it was about maybe two weeks ago that I had a really bad cold and I started coughing a real lot." \
    --text "Yes, I feel short of breath, and I'm sweating a lot. I also feel a bit nauseated, but I haven't vomited." \
    --res-wav-path result_patient.wav
