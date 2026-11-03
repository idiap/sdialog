# 🔊 Audio Generation Tutorials for `sdialog`

This directory contains a tutorial on the audio generation capabilities of the `sdialog` library: turning dialogues into realistic audio with ground-truth annotations. For all the options (e.g. custom rooms), see the [Audio Generation documentation](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#audio-generation).

## Tutorials

### 0. Quickstart (`0.quickstart.ipynb`)
**Start here.** One-line audio generation with `dialog.to_audio()`. It covers:
- Voices matched to the personas, designed, random or your own.
- Ground-truth annotations (segments, word timestamps, RTTM / JSON).
- Scene presets (clean, conversational, room, call center, far field, hard) in mono or stereo.
- Pauses, overlaps, reverberation, noise and telephone channels.
- Parallel batch generation.
