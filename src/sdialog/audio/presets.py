"""
Named acoustic scene presets for ``dialog.to_audio(preset=...)``.

A preset only sets *scene* parameters (timing, reverberation, noise, channel and level variation);
output-format parameters (``stereo``, ``sample_rate``, ``path``) are never part of a preset, so any
preset can be rendered in mono or stereo. Arguments passed explicitly to ``to_audio`` override the
preset values.

Per-speaker values are dictionaries keyed by speaker index (0 = first speaker of the dialogue) or by
speaker name. A list of values means "pick one at random".

Example:

    .. code-block:: python

        from sdialog.audio import register_preset

        register_preset("noisy_phone", channel="g711", noise={"type": "pink", "snr": (5, 15)}, overlap=0.2)
        audio = dialog.to_audio(preset="noisy_phone", stereo=True)
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import copy

from typing import Optional

SCENE_PARAMS = ("pauses", "overlap", "max_overlap", "reverb", "noise", "channel", "level_jitter_db")

SCENE_DEFAULTS = {
    "pauses": "natural",
    "overlap": 0.0,
    "max_overlap": 0.3,
    "reverb": None,
    "noise": None,
    "channel": None,
    "level_jitter_db": 0.0,
}

PRESETS = {
    # Studio-quality speech with the TTS engine's own timing
    "clean": {},
    # Spontaneous-like timing: natural pauses and some overlapping speech, no acoustic effects
    "conversational": {"pauses": 0.35, "overlap": 0.15, "level_jitter_db": 2.0},
    # Meeting / office room: both speakers in the same medium room (different positions), light noise
    "room": {"pauses": 0.4, "overlap": 0.1, "reverb": "medium",
             "noise": {"type": "pink", "snr": (20, 30)}, "level_jitter_db": 3.0},
    # Call center: agent (first speaker) on a headset, caller (second speaker) through a phone line
    "call_center": {"pauses": 0.4, "overlap": 0.1,
                    "noise": {0: {"type": "pink", "snr": (30, 40)},
                              1: {"type": ["pink", "brown", "white"], "snr": (10, 25)}},
                    "channel": {1: ["g711", "amr"]}, "level_jitter_db": 3.0},
    # Distant microphone in a large reverberant room, noisy
    "far_field": {"pauses": 0.4, "overlap": 0.2, "reverb": "large",
                  "noise": {"type": "pink", "snr": (5, 15)}, "level_jitter_db": 4.0},
    # Everything at once: large room, very low SNR, narrowband channel, frequent overlaps
    "hard": {"pauses": 0.3, "overlap": 0.3, "reverb": "large",
             "noise": {"type": ["pink", "brown", "white"], "snr": (0, 10)},
             "channel": "g711", "level_jitter_db": 6.0},
}


def register_preset(name: str, **params) -> None:
    """
    Registers (or replaces) a named scene preset.

    :param name: Name of the preset (used as ``dialog.to_audio(preset=name)``).
    :param params: Scene parameters; valid keys are: pauses, overlap, max_overlap, reverb, noise,
                   channel and level_jitter_db.
    :raises ValueError: If an invalid parameter is given (e.g. ``stereo``, which is an output option).
    """
    invalid = [k for k in params if k not in SCENE_PARAMS]
    if invalid:
        raise ValueError(f"Invalid preset parameter(s) {invalid}. Presets only set scene parameters: "
                         f"{list(SCENE_PARAMS)} (output options like `stereo` are passed to to_audio).")
    PRESETS[name] = copy.deepcopy(params)


def list_presets() -> dict:
    """Returns a copy of all registered presets."""
    return copy.deepcopy(PRESETS)


def resolve_scene(preset: Optional[str] = None, **overrides) -> dict:
    """
    Returns the full scene parameters: defaults, updated with the preset, updated with the explicit
    (non-``None``) overrides.
    """
    scene = copy.deepcopy(SCENE_DEFAULTS)
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {list(PRESETS)}")
        scene.update(copy.deepcopy(PRESETS[preset]))
    scene.update({k: v for k, v in overrides.items() if v is not None})
    return scene
