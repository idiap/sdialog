"""
Result of ``dialog.to_audio()``: the dialogue audio together with its ground-truth annotations.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import json
import numpy as np

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


@dataclass
class Segment:
    """
    Ground-truth annotation of one dialogue turn in the audio.

    :ivar turn: Index of the turn in the dialogue.
    :ivar speaker: Speaker name.
    :ivar text: Text of the turn.
    :ivar start: Start time (seconds).
    :ivar end: End time (seconds).
    :ivar channel: Channel holding this speaker in stereo output (``None`` for mono output).
    :ivar words: Word timestamps ``(word, start_seconds)`` (absolute times), when the TTS engine provides them.
    """
    turn: int
    speaker: str
    text: str
    start: float
    end: float
    channel: Optional[int] = None
    words: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DialogAudio:
    """
    Audio of a dialogue with its annotations, returned by ``dialog.to_audio()``.

    :ivar audio: Float32 waveform, shape ``(samples,)`` (mono) or ``(samples, channels)`` (stereo,
                 one channel per speaker in dialogue speaker order).
    :ivar sample_rate: Sample rate (Hz).
    :ivar segments: One :class:`Segment` per turn, in dialogue order.
    :ivar speakers: Speaker names, in channel order.
    :ivar voices: Voice used for each speaker (e.g. the voice database identifier, or "random").
    :ivar config: All the parameters used to generate the audio, including the values drawn at random
                  (room, SNRs, codecs, ...), for reproducibility.
    :ivar dialog_id: Id of the source dialogue, if any.

    Example:

        .. code-block:: python

            audio = dialog.to_audio(preset="call_center", stereo=True, seed=0)
            audio.save("output/call")       # output/call.wav, output/call.json, output/call.rttm
            for seg in audio.segments:
                print(f"{seg.start:6.2f}-{seg.end:6.2f} {seg.speaker}: {seg.text}")
    """
    audio: np.ndarray
    sample_rate: int
    segments: List[Segment]
    speakers: List[str]
    voices: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    dialog_id: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds."""
        return self.audio.shape[0] / self.sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return 1 if self.audio.ndim == 1 else self.audio.shape[1]

    def to_dict(self, words: bool = True) -> dict:
        """Returns the annotations (everything but the waveform) as a JSON-serializable dictionary."""
        segments = []
        for seg in self.segments:
            seg = asdict(seg)
            seg["start"], seg["end"] = round(seg["start"], 3), round(seg["end"], 3)
            if words:
                seg["words"] = [[w, round(t, 3)] for w, t in seg["words"]]
            else:
                seg.pop("words")
            segments.append(seg)
        return {"dialog_id": self.dialog_id, "sample_rate": self.sample_rate, "duration": round(self.duration, 3),
                "channels": self.channels, "speakers": self.speakers, "voices": self.voices,
                "segments": segments, "config": self.config}

    def to_json(self, indent: int = 2, words: bool = True) -> str:
        """Returns the annotations as a JSON string."""
        return json.dumps(self.to_dict(words=words), indent=indent, ensure_ascii=False, default=str)

    def to_rttm(self, file_id: Optional[str] = None) -> str:
        """Returns the speaker turns in RTTM format (speaker diarization ground truth)."""
        file_id = file_id or self.dialog_id or "dialog"
        file_id = str(file_id).replace(" ", "_")
        return "".join(
            f"SPEAKER {file_id} 1 {seg.start:.3f} {seg.duration:.3f} <NA> <NA> {seg.speaker.replace(' ', '_')} "
            "<NA> <NA>\n"
            for seg in self.segments
        )

    def save(self, path: str, audio_format: str = "wav") -> dict:
        """
        Saves the audio and its annotations as ``<path>.<audio_format>``, ``<path>.json`` and ``<path>.rttm``.

        :param path: Output path without extension (an audio extension, if given, is removed).
        :param audio_format: Audio file format ("wav" or "flac").
        :return: The paths of the written files.
        """
        import soundfile as sf
        root, ext = os.path.splitext(path)
        if ext.lower().lstrip(".") in ("wav", "flac", "ogg"):
            path, audio_format = root, ext.lower().lstrip(".")
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        paths = {"audio": f"{path}.{audio_format}", "json": f"{path}.json", "rttm": f"{path}.rttm"}
        sf.write(paths["audio"], self.audio, self.sample_rate)
        with open(paths["json"], "w", encoding="utf-8") as f:
            f.write(self.to_json())
        with open(paths["rttm"], "w", encoding="utf-8") as f:
            f.write(self.to_rttm(file_id=os.path.basename(path)))
        return paths

    def display(self):
        """Shows an audio player (in Jupyter notebooks)."""
        from IPython.display import Audio, display
        display(Audio(self.audio.T if self.audio.ndim == 2 else self.audio, rate=self.sample_rate))

    def _repr_html_(self):
        return (f"<b>DialogAudio</b>: {self.duration:.1f} s, {self.sample_rate} Hz, {self.channels} channel(s), "
                f"{len(self.segments)} turns")
