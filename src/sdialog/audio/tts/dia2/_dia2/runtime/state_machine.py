from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Sequence, Tuple


@dataclass
class TokenIds:
    card: int
    new_word: int
    pad: int
    bos: int
    zero: int
    spk1: int
    spk2: int
    audio_pad: int
    audio_bos: int
    ungenerated: int = -2


@dataclass
class Entry:
    tokens: List[int]
    text: str
    padding: int = 0


@dataclass
class State:
    entries: Deque[Entry]
    padding_budget: int
    forced_padding: int
    pending_tokens: Deque[int] = field(default_factory=deque)
    lookahead_tokens: Deque[int] = field(default_factory=deque)
    end_step: int | None = None
    consumption_times: List[int] = field(default_factory=list)
    transcript: List[Tuple[str, int]] = field(default_factory=list)

    def peek_tokens(self, count: int) -> List[int]:
        """Return tokens from upcoming entries (used for second-stream lookahead)."""
        assert count > 0
        for entry in self.entries:
            if entry.tokens:
                count -= 1
                if count == 0:
                    return entry.tokens
        return []


class StateMachine:
    def __init__(
        self,
        token_ids: TokenIds,
        *,
        second_stream_ahead: int = 0,
        max_padding: int = 6,
        initial_padding: int = 0,
    ) -> None:
        self.token_ids = token_ids
        self.second_stream_ahead = second_stream_ahead
        self.max_padding = max_padding
        self.initial_padding = initial_padding

    def new_state(self, entries: Iterable[Entry]) -> State:
        return State(
            entries=deque(entries),
            padding_budget=self.initial_padding,
            forced_padding=self.initial_padding,
        )

    def process(
        self,
        step: int,
        state: State,
        token: int,
        is_forced: bool = False,
    ) -> Tuple[int, int, bool]:
        token = self._sanitize_token(token)
        token = self._enforce_token_constraints(state, token, is_forced)
        token, consumed_new_word = self._handle_new_word(step, state, token)
        output_token = self._select_output_token(state, token)
        final_main, final_second = self._maybe_multiplex_second_stream(
            state, output_token
        )
        return final_main, final_second, consumed_new_word

    def _sanitize_token(self, token: int) -> int:
        if token == 1:
            token = self.token_ids.new_word
        elif token == 0:
            token = self.token_ids.pad
        if token not in (self.token_ids.new_word, self.token_ids.pad):
            return self.token_ids.pad
        return token

    def _enforce_token_constraints(
        self, state: State, token: int, is_forced: bool
    ) -> int:
        if state.pending_tokens:
            return self.token_ids.pad
        if is_forced:
            return token
        if state.forced_padding > 0:
            if token != self.token_ids.pad:
                token = self.token_ids.pad
            return token
        if state.padding_budget <= 0 and token != self.token_ids.new_word:
            return self.token_ids.new_word
        return token

    def _handle_new_word(
        self, step: int, state: State, token: int
    ) -> Tuple[int, bool]:
        if token != self.token_ids.new_word:
            return token, False
        if state.entries:
            entry = state.entries.popleft()
            state.consumption_times.append(step)
            if entry.tokens:
                state.transcript.append((entry.text, step))
                state.pending_tokens.extend(entry.tokens)
                if self.second_stream_ahead:
                    state.lookahead_tokens.extend(
                        state.peek_tokens(self.second_stream_ahead)
                    )
                state.padding_budget = self.max_padding
            else:
                token = self.token_ids.pad
            state.forced_padding = entry.padding
            return token, True
        token = self.token_ids.pad
        if self.second_stream_ahead and state.end_step is None:
            token = self.token_ids.new_word
        if state.end_step is None:
            state.end_step = step
        return token, False

    def _select_output_token(self, state: State, token: int) -> int:
        if token == self.token_ids.pad:
            if state.padding_budget > 0:
                state.padding_budget -= 1
            if state.forced_padding > 0:
                state.forced_padding -= 1
            if state.pending_tokens:
                return state.pending_tokens.popleft()
            return self.token_ids.pad
        if token == self.token_ids.new_word:
            return self.token_ids.new_word
        if token == self.token_ids.zero:
            return token
        raise RuntimeError(f"Invalid token {token}")

    def _maybe_multiplex_second_stream(
        self, state: State, output: int
    ) -> Tuple[int, int]:
        if not self.second_stream_ahead:
            return output, output
        second = -1
        if output == self.token_ids.new_word:
            second = self.token_ids.new_word
            if state.pending_tokens:
                output = state.pending_tokens.popleft()
            else:
                output = self.token_ids.pad
        elif state.lookahead_tokens:
            second = state.lookahead_tokens.popleft()
        else:
            second = self.token_ids.pad
        return output, second
