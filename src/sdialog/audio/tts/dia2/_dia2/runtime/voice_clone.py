from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch

from ..generation import PrefixConfig
from .audio_io import encode_audio_tokens, load_mono_audio
from .state_machine import Entry

if TYPE_CHECKING:  # pragma: no cover
    from .context import RuntimeContext


@dataclass
class WhisperWord:
    text: str
    start: float
    end: float


@dataclass
class PrefixPlan:
    entries: List[Entry]
    new_word_steps: List[int]
    aligned_tokens: torch.Tensor
    aligned_frames: int


def build_prefix_plan(
    runtime: "RuntimeContext",
    prefix: Optional[PrefixConfig],
    *,
    transcribe_fn: Optional[Callable[[str, torch.device], List[WhisperWord]]] = None,
    load_audio_fn: Optional[Callable[[str, int], np.ndarray]] = None,
    encode_fn: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
) -> Optional[PrefixPlan]:
    if prefix is None:
        return None
    if not prefix.speaker_1:
        if prefix.speaker_2:
            raise ValueError("speaker_2 requires speaker_1 to be provided")
        return None

    transcribe = transcribe_fn or (lambda source, device: transcribe_words(source, device))
    load_audio = load_audio_fn or (lambda source, sr: load_mono_audio(source, sr))
    encode_audio = encode_fn or (lambda audio: encode_audio_tokens(runtime.mimi, audio))

    entries1, steps1, tokens1 = _process_prefix_audio(
        runtime=runtime,
        audio_path=prefix.speaker_1,
        speaker_token=runtime.constants.spk1,
        transcribe=transcribe,
        load_audio=load_audio,
        encode_audio=encode_audio,
    )
    offset = 3  # Match legacy BOS/PAD offset
    entries = list(entries1)
    new_word_steps = [step + offset for step in steps1]
    audio_tokens = tokens1.to(runtime.device)

    if prefix.speaker_2:
        entries2, steps2, tokens2 = _process_prefix_audio(
            runtime=runtime,
            audio_path=prefix.speaker_2,
            speaker_token=runtime.constants.spk2,
            transcribe=transcribe,
            load_audio=load_audio,
            encode_audio=encode_audio,
        )
        spk1_frames = audio_tokens.shape[-1]
        new_word_steps.extend(step + spk1_frames for step in steps2)
        entries.extend(entries2)
        audio_tokens = torch.cat([audio_tokens, tokens2.to(runtime.device)], dim=1)

    return PrefixPlan(
        entries=entries,
        new_word_steps=new_word_steps,
        aligned_tokens=audio_tokens,
        aligned_frames=audio_tokens.shape[-1],
    )


def _process_prefix_audio(
    runtime: "RuntimeContext",
    audio_path,
    speaker_token: int,
    *,
    transcribe: Callable[[object, torch.device], List[WhisperWord]],
    load_audio: Callable[[object, int], np.ndarray],
    encode_audio: Callable[[np.ndarray], torch.Tensor],
) -> tuple[List[Entry], List[int], torch.Tensor]:
    words = transcribe(audio_path, runtime.device)
    entries, steps = words_to_entries(
        words=words,
        tokenizer=runtime.tokenizer,
        speaker_token=speaker_token,
        frame_rate=runtime.frame_rate,
    )
    audio = load_audio(audio_path, runtime.mimi.sample_rate)
    tokens = encode_audio(audio)
    return entries, steps, tokens


# [sdialog] The Whisper model is loaded once (instead of on every call) and transcriptions are
# cached by audio content, so the same reference voice is transcribed only once per process.
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
_WHISPER_MODELS: Dict[tuple, object] = {}
_TRANSCRIPTION_CACHE: Dict[tuple, List[WhisperWord]] = {}
_WHISPER_SAMPLE_RATE = 16000


def _get_whisper_model(device: torch.device):
    import whisper_timestamped as wts  # Imported lazily

    key = (WHISPER_MODEL_NAME, str(device))
    if key not in _WHISPER_MODELS:
        _WHISPER_MODELS[key] = wts.load_model(WHISPER_MODEL_NAME, device=str(device))
    return _WHISPER_MODELS[key]


def transcribe_words(
    audio_source,
    device: torch.device,
    language: Optional[str] = None,
) -> List[WhisperWord]:
    import whisper_timestamped as wts  # Imported lazily

    audio = load_mono_audio(audio_source, _WHISPER_SAMPLE_RATE)
    key = (WHISPER_MODEL_NAME, language, hashlib.sha1(audio.tobytes()).hexdigest())
    if key in _TRANSCRIPTION_CACHE:
        return list(_TRANSCRIPTION_CACHE[key])

    model = _get_whisper_model(device)
    result = wts.transcribe(model, audio, language=language)

    words: List[WhisperWord] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("text") or word.get("word") or "").strip()
            if not text:
                continue
            words.append(
                WhisperWord(
                    text=text,
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", 0.0)),
                )
            )
    _TRANSCRIPTION_CACHE[key] = list(words)
    return words


def words_to_entries(
    *,
    words: Sequence[WhisperWord],
    tokenizer,
    speaker_token: int,
    frame_rate: float,
) -> tuple[List[Entry], List[int]]:
    entries: List[Entry] = []
    new_word_steps: List[int] = []
    if not words:
        return entries, new_word_steps

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    speaker_prefix: Optional[str] = None
    if callable(convert):
        s1_id = convert("[S1]")
        s2_id = convert("[S2]")
        if speaker_token == s1_id:
            speaker_prefix = "[S1]"
        elif speaker_token == s2_id:
            speaker_prefix = "[S2]"
    pending_prefix: Optional[str] = speaker_prefix
    current_pos = 0

    for idx, word in enumerate(words):
        tokens = _encode_word(word.text, tokenizer, pending_prefix)
        pending_prefix = None
        start_frame = max(current_pos + 1, int(round(word.start * frame_rate)))
        end_frame = start_frame + len(tokens)
        new_word_steps.append(start_frame - 1)

        if idx < len(words) - 1:
            next_start = int(round(words[idx + 1].start * frame_rate))
            next_word_start = max(end_frame + 1, next_start)
        else:
            end_time = int(round(words[-1].end * frame_rate))
            next_word_start = max(end_frame + 1, end_time)

        padding = max(0, next_word_start - start_frame - 1)
        entries.append(Entry(tokens=tokens, text=word.text, padding=padding))
        current_pos = end_frame

    return entries, new_word_steps


def _encode_word(text: str, tokenizer, prefix: Optional[str]) -> List[int]:
    if prefix:
        return tokenizer.encode(f"{prefix} {text}", add_special_tokens=False)
    return tokenizer.encode(text, add_special_tokens=False)


__all__ = [
    "PrefixPlan",
    "WhisperWord",
    "build_prefix_plan",
    "transcribe_words",
    "words_to_entries",
]
