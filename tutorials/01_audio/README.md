# 🔊 Audio Generation Tutorials for `sdialog`

This directory contains a series of tutorials demonstrating the audio generation capabilities of the `sdialog` library. Each notebook covers a specific aspect of the audio pipeline, from basic text-to-speech synthesis to advanced acoustic simulations.

## Tutorials

Here is a breakdown of the tutorials available in this directory:

### 1. Audio Generation (`1.audio_generation.ipynb`)
This tutorial covers the fundamentals of generating audio from a textual dialogue. It explains how to:
- Set up the Text-to-Speech (TTS) engine.
- Use voice databases to assign voices to speakers.
- Create and run an audio pipeline to synthesize speech for a dialogue.
- Automatically resample output audios to your sampling rate.
- Change audio file format between MP3 / WAV / FLAC. 

### 2. Acoustic Simulation (`2.accoustic_simulation.ipynb`)
Building on the basics, this notebook introduces acoustic simulation. You will learn how to make the generated audio sound as if it were recorded in a specific room or environment, adding realism to the dialogue. Including background and foreground effects, while also improving simulation using ray tracing technology. 

### 3. Acoustic Simulation for Customer Service (`3.accoustic_simulation-customer_service.ipynb`)
This tutorial applies acoustic simulation to a specific use-case: a customer service call. It demonstrates how to configure the environment to match a typical call center or office setting.

### 4. Room Generation (`4.rooms.ipynb`)
Learn how to define and generate different types of rooms for acoustic simulation. This notebook covers:
- Using room generators like `MedicalRoomGenerator` and `BasicRoomGenerator`.
- Customizing room properties (dimensions, wall materials, furnitures placement) to create diverse acoustic environments.
- Placing objects, human speakers and microphone around the room. All of this based on absolute coordinates or relative to room anchors and furnitures placement.
- Creating your own room generator from scratch by extending the `RoomGenerator` class.

### 5. Voice Databases (`5.voices_databases.ipynb`)
This tutorial focuses on managing and using voice databases. It explains how to:
- Load voice databases from Hugging Face, local storage and on the fly using the `VoiceDatabase` class.
- Select voices based on speaker attributes (e.g., age, gender).
- Manage voice selection to ensure variety in generated dialogues.

### 6. Acoustic Variations (`6.accoustics_variations.ipynb`)
Explore how different room configurations and acoustic properties affect the final audio. This tutorial allows you to compare audio generated in various environments and understand the impact of acoustics on speech.

### 7. Impulse Response Simulation (`7.impulse_response.ipynb`)
This notebook covers the final step in the audio pipeline: simulating recording devices. You will learn how to apply microphone impulse responses to the audio to make it sound as if it were recorded on different devices, such as a phone or a high-quality microphone.
