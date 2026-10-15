"""
Dialogue-to-audio synthesis: the implementation behind ``dialog.to_audio()``.

The whole process runs in memory (no temporary files):

  1. Resolve the scene (preset + explicit arguments) and each speaker's voice.
  2. Generate the audio of every turn with a dialogue TTS engine (Dia2 by default).
  3. Trim silences and normalize the speech level of every turn.
  4. Place the turns on a timeline (pauses / overlapping speech).
  5. Apply per-speaker effects (reverberation, noise, telephone channel) on each speaker's track.
  6. Mix to mono or stereo (one channel per speaker) and apply the shared effects.

Example:

    .. code-block:: python

        audio = dialog.to_audio()                                     # clean, Dia2, persona-matched voices
        audio = dialog.to_audio(preset="call_center", stereo=True)    # agent / caller channels
        audio = dialog.to_audio(voices="random", reverb="large", noise=10, overlap=0.2, seed=0)
        audio.save("output/dialog")                                   # .wav + .json + .rttm
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import json
import logging
import numpy as np

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from . import effects
from .presets import resolve_scene
from .result import DialogAudio, Segment
from .timeline import place_turns
from .tts.base import BaseTTS, BaseVoiceCloneTTS
from .tts.dialog_tts import BaseDialogTTS, PerTurnTTS
from .voices import resolve_voices

logger = logging.getLogger(__name__)

SPEECH_LEVEL_DB = -26.0  # active speech level of every turn before effects
LEAD_IN, TAIL = 0.3, 0.5  # seconds of silence before the first and after the last turn

_default_tts = None


def get_default_tts() -> BaseDialogTTS:
    """Returns the default TTS engine (Dia2), loaded once per process."""
    global _default_tts
    if _default_tts is None:
        from .tts.dia2 import Dia2TTS
        _default_tts = Dia2TTS()
    return _default_tts


def _as_dialog_tts(tts) -> BaseDialogTTS:
    if tts is None:
        return get_default_tts()
    if isinstance(tts, BaseDialogTTS):
        return tts
    if isinstance(tts, (BaseTTS, BaseVoiceCloneTTS)):
        return PerTurnTTS(tts)
    raise TypeError(f"tts must be a BaseDialogTTS, BaseTTS or BaseVoiceCloneTTS instance, got {type(tts).__name__}")


def _persona_dict(persona: Any) -> dict:
    if persona is None:
        return {}
    if isinstance(persona, dict):
        return persona
    if hasattr(persona, "model_dump"):
        return persona.model_dump()
    return dict(persona)


def _per_speaker(value: Any, speakers: List[str]) -> Tuple[List[Any], bool]:
    """
    Expands a scene value to one value per speaker.

    :return: The per-speaker values and whether the value was speaker-specific (a dict keyed by speaker
             name or index) rather than shared by all speakers.
    """
    if isinstance(value, dict) and value and all(isinstance(k, int) or k in speakers for k in value):
        return [value.get(ix, value.get(name)) for ix, name in enumerate(speakers)], True
    return [value] * len(speakers), False


def _pick(value: Any, rng: np.random.Generator) -> Any:
    """Picks a random element if ``value`` is a list (a list means "one of these")."""
    return value[int(rng.integers(len(value)))] if isinstance(value, list) else value


def _is_off(value: Any) -> bool:
    return value is None or value is False


def _noise_params(value: Any, rng: np.random.Generator) -> Optional[dict]:
    """Normalizes a noise spec (SNR number, (min, max) range, or {"type", "snr"} dict)."""
    if _is_off(value):
        return None
    if isinstance(value, dict):
        return {"kind": _pick(value.get("type", "white"), rng), "snr_db": value.get("snr", 20.0)}
    if isinstance(value, (int, float, tuple)):
        return {"kind": "white", "snr_db": value}
    if isinstance(value, str):  # a noise type or file/folder path with default SNR
        return {"kind": value, "snr_db": 20.0}
    raise ValueError(f"Invalid noise value: {value}")


def _apply_noise(x: np.ndarray, sr: int, value: Any, rng: np.random.Generator) -> Tuple[np.ndarray, Optional[dict]]:
    params = _noise_params(value, rng)
    if params is None or not np.any(x):
        return x, None
    return effects.add_noise(x, sr, rng, snr_db=params["snr_db"], kind=params["kind"])


def _apply_channel(x: np.ndarray, sr: int, value: Any, rng: np.random.Generator) -> Tuple[np.ndarray, Optional[str]]:
    codec = _pick(value, rng)
    if _is_off(codec):
        return x, None
    codec = "g711" if codec is True else codec
    return effects.telephone(x, sr, codec), codec


def _speaker_rirs(values: List[Any], shared: bool, rng: np.random.Generator,
                  sr: int) -> Tuple[List[Optional[np.ndarray]], Any]:
    """Returns the impulse response of each speaker (``None`` = no reverb) and a description of the room(s)."""
    def one(value, n_sources):
        if isinstance(value, np.ndarray):
            return [value] * n_sources, {"type": "user"}
        if value is True:
            value = "medium"
        if value in effects.ROOM_SIZES:
            return effects.simulate_rirs(rng, sr, size=value, n_sources=n_sources)
        if isinstance(value, str):
            loaded = [effects.load_rir(value, rng, sr) for _ in range(n_sources)]
            return [r for r, _ in loaded], {"type": "file", "files": [f for _, f in loaded]}
        raise ValueError(f"Invalid reverb value: {value}")

    if shared:  # all speakers in the same room (different positions)
        value = _pick(values[0], rng)
        if _is_off(value):
            return [None] * len(values), None
        return one(value, len(values))
    rirs, info = [], []
    for value in values:
        value = _pick(value, rng)
        if _is_off(value):
            rirs.append(None)
            info.append(None)
        else:
            rir, room = one(value, 1)
            rirs.append(rir[0])
            info.append(room)
    return rirs, info


def _get_turns(dialog) -> Tuple[List[str], List[Tuple[int, str]], List[int]]:
    """Returns the speakers (in order of appearance), the (speaker_index, text) turns and their dialogue index."""
    speakers, turns, turn_ids = [], [], []
    for ix, turn in enumerate(dialog.turns or []):
        text = " ".join((turn.text or "").split())
        if not text:
            continue
        speaker = turn.speaker or "Speaker"
        if speaker not in speakers:
            speakers.append(speaker)
        turns.append((speakers.index(speaker), text))
        turn_ids.append(ix)
    return speakers, turns, turn_ids


def to_audio(dialog,
             path: Optional[str] = None,
             preset: Optional[str] = None,
             voices: Union[str, dict] = "db",
             voice_database=None,
             voice_description: Optional[Union[str, Callable]] = None,
             tts=None,
             pauses: Optional[Union[str, float, Tuple[float, float]]] = None,
             overlap: Optional[float] = None,
             max_overlap: Optional[float] = None,
             reverb: Any = None,
             noise: Any = None,
             channel: Any = None,
             level_jitter_db: Optional[float] = None,
             stereo: bool = False,
             sample_rate: int = 16000,
             seed: Optional[int] = None) -> DialogAudio:
    """
    Converts a two-speaker dialogue into audio, with ground-truth annotations.

    Scene parameters left as ``None`` take the value of the ``preset`` (or the clean default); pass ``False``
    to explicitly disable an effect set by the preset. Reverb, noise and channel values can be given per speaker
    as a dictionary keyed by speaker name or index (0 = first speaker); a single value applies to the whole
    recording (all speakers). A list of values means "pick one at random".

    :param dialog: The :class:`~sdialog.Dialog` to convert (exactly two speakers).
    :param path: If given, saves ``<path>.wav``, ``<path>.json`` and ``<path>.rttm``.
    :param preset: Named scene preset: "clean", "conversational", "room", "call_center", "far_field", "hard",
                   or any preset added with :func:`sdialog.audio.register_preset`.
    :param voices: "db" (voice database matching the personas), "design" (designed from the persona
                   description), "random" (made up by the TTS engine), or a ``{speaker: voice}`` dictionary
                   (audio path, ``(array, sample_rate)``, :class:`~sdialog.audio.Voice` or engine voice name).
    :param voice_database: Voice database, or Hugging Face dataset name, used by ``voices="db"``
                           (default: the TTS engine's default, "sdialog/voices-libritts" for Dia2).
    :param voice_description: Jinja2 template over the persona fields (or function of the persona dict)
                              describing each voice, used by ``voices="design"``.
    :param tts: TTS engine: a :class:`~sdialog.audio.tts.BaseDialogTTS` (default: ``Dia2TTS()``), or a classic
                ``BaseTTS``/``BaseVoiceCloneTTS`` engine (wrapped with ``PerTurnTTS``).
    :param pauses: "natural" (the TTS engine's own timing), a median pause in seconds (log-normal), or a
                   ``(min, max)`` range in seconds (uniform).
    :param overlap: Probability (0-1) that a turn starts before the previous one ends.
    :param max_overlap: Maximum overlap, as a fraction of the previous turn's duration (default 0.3).
    :param reverb: Room reverberation: True or "small"/"medium"/"large" (simulated room), a path to an
                   impulse response file or folder of files, or an impulse response array.
    :param noise: Background noise: an SNR in dB, a ``(min, max)`` SNR range, a noise type ("white", "pink",
                  "brown") or audio file/folder path, or a dict ``{"type": ..., "snr": ...}``.
    :param channel: Telephone channel codec: "telephone" (= "g711"), "g711", "g729", "amr" or "amr-wb".
    :param level_jitter_db: Maximum random loudness difference (dB) between speakers.
    :param stereo: If True, output one channel per speaker (in order of appearance); otherwise mono.
    :param sample_rate: Output sample rate (Hz).
    :param seed: Random seed; the same seed reproduces the same audio (a random seed is drawn and saved in
                 the annotations when not given).
    :return: A :class:`~sdialog.audio.DialogAudio` with the waveform and the annotations.
    :raises ValueError: If the dialogue does not have exactly two speakers or a parameter is invalid.
    """
    speakers, turns, turn_ids = _get_turns(dialog)
    if len(speakers) != 2:
        raise ValueError(f"to_audio() supports dialogues with exactly two speakers, found {len(speakers)}: "
                         f"{speakers}")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0] % 2**31)
    rng = np.random.default_rng(seed)
    scene = resolve_scene(preset, pauses=pauses, overlap=overlap, max_overlap=max_overlap, reverb=reverb,
                          noise=noise, channel=channel, level_jitter_db=level_jitter_db)
    tts = _as_dialog_tts(tts)
    personas = {name: _persona_dict(p) for name, p in (dialog.personas or {}).items()}

    # 1. Voices and TTS
    voice_list, voice_info = resolve_voices(speakers, personas, turns, voices, tts, rng,
                                            voice_database=voice_database, voice_description=voice_description)
    clips = tts.synthesize(turns, voice_list, seed=int(rng.integers(2**31)))
    if len(clips) != len(turns):
        raise RuntimeError(f"{type(tts).__name__} returned {len(clips)} clips for {len(turns)} turns")

    # 2. Per-turn processing: resample, trim silences, set level
    jitter = [float(rng.uniform(-1, 1) * scene["level_jitter_db"] / 2) for _ in speakers]
    audios, durations, natural_starts, words = [], [], [], []
    for (speaker, _), clip in zip(turns, clips):
        x = effects.resample(effects.to_mono(clip.audio), clip.sample_rate, sample_rate)
        x, trimmed = effects.trim_silence(x, sample_rate)
        x = effects.set_active_level(x, sample_rate, SPEECH_LEVEL_DB + jitter[speaker])
        audios.append(x)
        durations.append(len(x) / sample_rate)
        natural_starts.append(clip.offset + trimmed / sample_rate)
        words.append([(w, t - trimmed / sample_rate) for w, t in clip.words if t - trimmed / sample_rate >= -0.05])

    # 3. Timeline
    starts, gaps = place_turns([s for s, _ in turns], durations, rng, pauses=scene["pauses"],
                               overlap=scene["overlap"], max_overlap=scene["max_overlap"],
                               windows=[c.window for c in clips], natural_starts=natural_starts, lead_in=LEAD_IN)
    length = int(np.ceil((max(s + d for s, d in zip(starts, durations)) + TAIL) * sample_rate))
    tracks = [np.zeros(length, dtype=np.float32) for _ in speakers]
    for (speaker, _), x, start in zip(turns, audios, starts):
        a = int(round(start * sample_rate))
        tracks[speaker][a:a + len(x)] += x

    # 4. Per-speaker effects: reverberation (same room if shared), then speaker-specific noise and channel
    reverbs, reverb_shared = _per_speaker(scene["reverb"], speakers)
    rirs, room_info = _speaker_rirs(reverbs, not reverb_shared, rng, sample_rate)
    tracks = [effects.apply_rir(t, rir, sample_rate) if rir is not None else t for t, rir in zip(tracks, rirs)]
    length = max(len(t) for t in tracks)
    tracks = [np.pad(t, (0, length - len(t))) for t in tracks]

    noises, noise_specific = _per_speaker(scene["noise"], speakers)
    channels, channel_specific = _per_speaker(scene["channel"], speakers)
    noise_info, channel_info = {}, {}
    for ix, name in enumerate(speakers):
        if noise_specific:
            tracks[ix], noise_info[name] = _apply_noise(tracks[ix], sample_rate, noises[ix], rng)
        if channel_specific:
            tracks[ix], channel_info[name] = _apply_channel(tracks[ix], sample_rate, channels[ix], rng)

    # 5. Mix and shared (recording-level) effects
    outputs = tracks if stereo else [np.sum(tracks, axis=0)]
    shared_noise = None if noise_specific else noises[0]
    shared_channel = None if channel_specific else _pick(channels[0], rng)
    for ix in range(len(outputs)):
        outputs[ix], info = _apply_noise(outputs[ix], sample_rate, shared_noise, rng)
        if info is not None:
            noise_info["all" if not stereo else speakers[ix]] = info
        outputs[ix], codec = _apply_channel(outputs[ix], sample_rate, shared_channel, rng)
        if codec is not None:
            channel_info["all"] = codec
        outputs[ix] = effects.soft_limit(outputs[ix])
    audio = np.stack(outputs, axis=1) if stereo else outputs[0]

    # 6. Annotations
    segments = [Segment(turn=turn_ids[k], speaker=speakers[speaker], text=text,
                        start=round(starts[k], 4), end=round(starts[k] + durations[k], 4),
                        channel=speaker if stereo else None,
                        words=[(w, round(starts[k] + t, 3)) for w, t in words[k]])
                for k, (speaker, text) in enumerate(turns)]
    config = {
        "seed": seed,
        "preset": preset,
        "scene": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in scene.items()},
        "stereo": stereo,
        "sample_rate": sample_rate,
        "tts": tts.json(),
        "level_offsets_db": {name: round(jitter[ix], 2) for ix, name in enumerate(speakers)},
        "room": room_info,
        "noise": noise_info or None,
        "channel": channel_info or None,
    }
    result = DialogAudio(audio=audio.astype(np.float32), sample_rate=sample_rate, segments=segments,
                         speakers=speakers, voices=voice_info, config=config,
                         dialog_id=None if dialog.id is None else str(dialog.id))
    if path is not None:
        result.save(path)
    return result


# ------------------------------------------------------------------------------------------------------------- batch
def _dialog_seed(seed: Optional[int], index: int) -> Optional[int]:
    if seed is None:
        return None
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0] % 2**31)


def _output_name(dialog, index: int) -> str:
    name = str(dialog.id) if getattr(dialog, "id", None) is not None else str(index)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _run_jobs(jobs, out_dir, device, tts_factory, kwargs, results=None):
    try:
        if tts_factory is not None:
            tts = tts_factory(device)
        else:
            from .tts.dia2 import Dia2TTS
            tts = Dia2TTS(device=device)
        tts_error = None
    except Exception as e:
        logger.error(f"Could not create the TTS engine on '{device}': {e}")
        tts, tts_error = None, e
    rows = []
    for index, dialog, seed in jobs:
        path = os.path.join(out_dir, _output_name(dialog, index))
        try:
            if tts_error is not None:
                raise tts_error
            audio = to_audio(dialog, path=path, tts=tts, seed=seed, **kwargs)
            row = {"index": index, "dialog_id": audio.dialog_id, "audio": f"{path}.wav", "json": f"{path}.json",
                   "rttm": f"{path}.rttm", "duration": round(audio.duration, 3), "seed": audio.config["seed"]}
        except Exception as e:  # keep going: one bad dialogue should not stop a large batch
            logger.error(f"Dialog {index} failed: {e}")
            row = {"index": index, "dialog_id": getattr(dialog, "id", None), "error": repr(e)}
        if results is not None:
            results.put(row)
        rows.append(row)
    return rows


def _worker(jobs, out_dir, device, tts_factory, kwargs, results):
    _run_jobs(jobs, out_dir, device, tts_factory, kwargs, results)


def to_audio_batch(dialogs: Sequence,
                   out_dir: str,
                   devices: Optional[List[str]] = None,
                   workers_per_device: int = 1,
                   tts_factory: Optional[Callable[[str], BaseDialogTTS]] = None,
                   seed: Optional[int] = None,
                   **kwargs) -> List[dict]:
    """
    Converts many dialogues to audio in parallel, with one TTS model per worker process.

    Every dialogue is saved under ``out_dir`` (``<dialog_id>.wav``, ``.json`` and ``.rttm``) and an
    ``index.jsonl`` file lists all the outputs. Each dialogue gets its own seed derived from ``seed`` and its
    position in ``dialogs``, so results do not depend on the number of workers.

    :param dialogs: Dialogues to convert.
    :param out_dir: Output folder.
    :param devices: Devices to use, e.g. ``["cuda:0", "cuda:1"]`` (default: all visible GPUs, or the CPU).
    :param workers_per_device: Number of worker processes (each with its own model) per device.
    :param tts_factory: Function ``device -> BaseDialogTTS`` creating each worker's engine (default:
                        ``Dia2TTS(device=device)``). Must be a top-level (picklable) function when using
                        several workers.
    :param seed: Base random seed.
    :param kwargs: Any other :func:`to_audio` argument (preset, voices, stereo, ...).
    :return: The rows of ``index.jsonl`` (one per dialogue, with output paths, or an "error" field).

    Example:

        .. code-block:: python

            from sdialog.audio import to_audio_batch

            rows = to_audio_batch(dialogs, "audios/", devices=["cuda:0", "cuda:1"], preset="call_center",
                                  stereo=True, seed=0)
    """
    if devices is None:
        try:
            import torch
            n_gpus = torch.cuda.device_count()
        except ImportError:
            n_gpus = 0
        devices = [f"cuda:{i}" for i in range(n_gpus)] or ["cpu"]
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(ix, dialog, _dialog_seed(seed, ix)) for ix, dialog in enumerate(dialogs)]
    workers = [d for d in devices for _ in range(workers_per_device)]
    workers = workers[:max(1, len(jobs))]

    if len(workers) == 1:
        rows = _run_jobs(jobs, out_dir, workers[0], tts_factory, kwargs)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # required for CUDA in child processes
        results = ctx.Queue()
        processes = [ctx.Process(target=_worker,
                                 args=(jobs[w::len(workers)], out_dir, device, tts_factory, kwargs, results))
                     for w, device in enumerate(workers)]
        for p in processes:
            p.start()
        rows = [results.get() for _ in jobs]
        for p in processes:
            p.join()

    rows.sort(key=lambda row: row["index"])
    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return rows


__all__ = ["to_audio", "to_audio_batch", "get_default_tts"]
