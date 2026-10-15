"""
Turn placement (pauses and overlapping speech) for :func:`sdialog.audio.synthesis.to_audio`.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import numpy as np

from typing import List, Optional, Tuple, Union

PausesSpec = Union[str, float, Tuple[float, float]]

DEFAULT_PAUSE_MEDIAN = 0.4  # seconds, used between independently generated turns in "natural" mode
PAUSE_SIGMA = 0.4
MIN_PAUSE, MAX_PAUSE = 0.05, 3.0
OVERLAP_RANGE = (0.15, 1.0)  # seconds
MIN_SELF_GAP = 0.1  # seconds between two consecutive turns of the same speaker


def validate_pauses(pauses: PausesSpec) -> PausesSpec:
    """Checks the ``pauses`` argument; raises ``ValueError`` if it is invalid."""
    if isinstance(pauses, str):
        if pauses != "natural":
            raise ValueError(f"Invalid pauses '{pauses}': use 'natural', a median in seconds, or a (min, max) range.")
    elif isinstance(pauses, (tuple, list)):
        if len(pauses) != 2 or not 0 <= pauses[0] <= pauses[1]:
            raise ValueError(f"Invalid pauses range {pauses}: expected (min, max) with 0 <= min <= max.")
    elif not isinstance(pauses, (int, float)) or pauses <= 0:
        raise ValueError(f"Invalid pauses {pauses}: a median pause must be a positive number of seconds.")
    return pauses


def sample_pause(rng: np.random.Generator, pauses: PausesSpec) -> float:
    """Draws a pause duration (seconds) from a median (log-normal) or a ``(min, max)`` range (uniform)."""
    if isinstance(pauses, (tuple, list)):
        return float(rng.uniform(*pauses))
    median = DEFAULT_PAUSE_MEDIAN if isinstance(pauses, str) else float(pauses)
    return float(np.clip(rng.lognormal(np.log(median), PAUSE_SIGMA), MIN_PAUSE, MAX_PAUSE))


def place_turns(speakers: List[int],
                durations: List[float],
                rng: np.random.Generator,
                pauses: PausesSpec = "natural",
                overlap: float = 0.0,
                max_overlap: float = 0.3,
                windows: Optional[List[int]] = None,
                natural_starts: Optional[List[float]] = None,
                lead_in: float = 0.3) -> Tuple[List[float], List[float]]:
    """
    Computes the start time of every turn.

    :param speakers: Speaker index of each turn.
    :param durations: Duration (seconds) of each turn's audio.
    :param rng: Random generator.
    :param pauses: "natural" keeps the timing of turns generated together (same window) and draws the
                   other pauses around 0.4 s; a number is the median of log-normal pauses; a ``(min, max)``
                   tuple draws pauses uniformly.
    :param overlap: Probability that a turn starts before the previous one ends.
    :param max_overlap: Maximum overlap, as a fraction of the previous turn's duration.
    :param windows: Generation window of each turn (needed for "natural" pauses).
    :param natural_starts: Start time of each turn within its window (needed for "natural" pauses).
    :param lead_in: Silence (seconds) before the first turn.
    :return: The start time of each turn, and the gap (seconds, negative for overlaps) before each turn.
    """
    if not 0 <= overlap <= 1:
        raise ValueError(f"overlap must be a probability in [0, 1], got {overlap}")
    if not 0 <= max_overlap <= 1:
        raise ValueError(f"max_overlap must be a fraction in [0, 1], got {max_overlap}")
    validate_pauses(pauses)
    natural = pauses == "natural" and windows is not None and natural_starts is not None

    starts, gaps = [], []
    last_end = {}
    for ix, (speaker, duration) in enumerate(zip(speakers, durations)):
        if ix == 0:
            start, gap = lead_in, 0.0
        else:
            prev_start, prev_dur = starts[-1], durations[ix - 1]
            prev_end = prev_start + prev_dur
            if natural and windows[ix] == windows[ix - 1]:
                gap = max(0.0, natural_starts[ix] - (natural_starts[ix - 1] + prev_dur))
            else:
                gap = sample_pause(rng, pauses)
            if overlap > 0 and speaker != speakers[ix - 1] and rng.random() < overlap:
                gap = -min(rng.uniform(*OVERLAP_RANGE), max_overlap * prev_dur)
            start = max(prev_end + gap, prev_start + 0.05)
        start = max(start, last_end.get(speaker, -np.inf) + MIN_SELF_GAP)  # nobody overlaps themselves
        if ix > 0:
            gap = start - (starts[-1] + durations[ix - 1])
        starts.append(float(start))
        gaps.append(float(gap))
        last_end[speaker] = start + duration
    return starts, gaps
