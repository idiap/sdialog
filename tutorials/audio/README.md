# 🔊 Audio Generation Tutorials for `sdialog`

This directory contains tutorials on the audio generation capabilities of the `sdialog` library: turning dialogues into realistic audio with ground-truth annotations. For all the options (e.g. custom rooms), see the [Audio Generation documentation](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#audio-generation).

## Tutorials

### 0. Quickstart (`0.quickstart.ipynb`)
**Start here.** One-line audio generation with `dialog.to_audio()`. It covers:
- Voices matched to the personas, designed, random or your own.
- Ground-truth annotations (segments, word timestamps, RTTM / JSON).
- Scene presets (clean, conversational, room, call center, far field, hard) in mono or stereo.
- Pauses, overlaps, reverberation, noise and telephone channels.
- Parallel batch generation.

### 1. Rooms and Acoustics (`1.rooms_and_acoustics.ipynb`)
Choose the room your dialogues are recorded in with `Room`. It covers:
- Random rooms vs. a room of your choice, and reading back what was simulated.
- How reverberant the room is: a target RT60 or the materials of its surfaces, with plots of the impulse responses.
- Microphone placement (ceiling, table, lapel) and directional microphones (cardioid, ...).
- A different room for each speaker, e.g. both ends of a phone call.
- Measured impulse responses, and varying the rooms across a dataset.
