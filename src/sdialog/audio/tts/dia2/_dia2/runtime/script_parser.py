from __future__ import annotations

import re
from typing import List, Optional, Sequence

from .state_machine import Entry


def parse_script(
    script: Sequence[str],
    tokenizer,
    constants,
    frame_rate: float,
) -> List[Entry]:
    entries: List[Entry] = []
    speaker_tokens = [constants.spk1, constants.spk2]
    padding_between = 1
    event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")
    last_speaker_idx = [None]

    def add_entry(idx: int, word: str, *, pending: Optional[int], first_content: List[bool]):
        tokens: List[int]
        if pending is not None:
            prefix = "[S1]" if pending == constants.spk1 else "[S2]"
            tokens = tokenizer.encode(f"{prefix} {word}", add_special_tokens=False)
        else:
            tokens = tokenizer.encode(word, add_special_tokens=False)
        if first_content[0]:
            if speaker_tokens:
                speaker_idx = idx % len(speaker_tokens)
                speaker_token = speaker_tokens[speaker_idx]
                if speaker_token is not None and last_speaker_idx[0] != speaker_idx:
                    if not tokens or tokens[0] != speaker_token:
                        tokens.insert(0, speaker_token)
                    last_speaker_idx[0] = speaker_idx
            first_content[0] = False
        padding = max(0, padding_between + len(tokens) - 1)
        entries.append(Entry(tokens=tokens, text=word, padding=padding))

    for idx, line in enumerate(script):
        normalized = line.replace("’", "'").replace(":", " ")
        remaining = normalized
        first_content = [True]
        pending_speaker: Optional[int] = None
        while remaining:
            match = event_re.search(remaining)
            if match is None:
                segment = remaining
                remaining = ""
            else:
                segment = remaining[: match.start()]
                remaining = remaining[match.end() :]
            if segment:
                for raw_word in segment.split():
                    if raw_word in ("[S1]", "[S2]"):
                        pending_speaker = (
                            constants.spk1 if raw_word == "[S1]" else constants.spk2
                        )
                        continue
                    add_entry(idx, raw_word, pending=pending_speaker, first_content=first_content)
                    pending_speaker = None
            if match and match.group(1):
                seconds = float(match.group(1))
                padding = int(round(seconds * frame_rate))
                if padding > 0:
                    entries.append(Entry(tokens=[], text="", padding=padding))
        if remaining:
            continue
    return entries
