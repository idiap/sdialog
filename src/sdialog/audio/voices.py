"""
Voice selection for ``dialog.to_audio(voices=...)``.

Voices can be:

  - ``"db"``: picked from a voice database to match each speaker's persona (gender, age, language).
  - ``"design"``: designed from each speaker's persona description with Qwen3-TTS VoiceDesign.
  - ``"random"``: made up by the TTS engine (a different voice on every generation unless seeded).
  - A dictionary ``{speaker: voice}`` with explicit voices (speaker name or index as key), where a voice
    is an audio path, an ``(array, sample_rate)`` tuple, a :class:`Voice`, or an engine voice name.
"""

# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import logging
import numpy as np

from jinja2 import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .voice_database import BaseVoiceDatabase, Voice

logger = logging.getLogger(__name__)

VOICE_MODES = ("db", "design", "random")
DEFAULT_VOICE_DATABASE = "sdialog/voices-libritts"
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_VOICE_DESCRIPTION = ("{{ gender }}{% if age %}, {{ age }} years old{% endif %}"
                             "{% if voice_characteristics %}, {{ voice_characteristics }}{% endif %}"
                             ", speaking naturally.")

AGE_WORDS = {"child": 10, "kid": 10, "teen": 16, "adolescent": 16, "young": 25, "adult": 35,
             "middle": 45, "senior": 70, "elderly": 75, "old": 70}
MALE_WORDS = ("male", "m", "man", "boy", "masculine")
FEMALE_WORDS = ("female", "f", "woman", "girl", "feminine")

_voice_databases: Dict[str, BaseVoiceDatabase] = {}
_voice_design_model = None


def normalize_age(age: Any, rng: np.random.Generator) -> int:
    """Converts a persona age (number, "35", "30s", "middle-aged", "", None, ...) to an integer age."""
    if isinstance(age, (int, float)) and not isinstance(age, bool) and age > 0:
        return int(age)
    text = str(age or "").lower()
    number = re.search(r"\d+", text)
    if number:
        return int(number.group())
    for word, value in AGE_WORDS.items():
        if word in text:
            return value
    return int(rng.integers(20, 66))


def normalize_gender(gender: Any, rng: np.random.Generator) -> str:
    """Converts a persona gender to "male" or "female" (random if unknown)."""
    text = str(gender or "").strip().lower()
    if text in MALE_WORDS or text.startswith("male"):
        return "male"
    if text in FEMALE_WORDS or text.startswith("female"):
        return "female"
    return "male" if rng.random() < 0.5 else "female"


def get_voice_database(name_or_db: Union[str, BaseVoiceDatabase]) -> BaseVoiceDatabase:
    """Returns a voice database, loading (once per process) Hugging Face databases given by name."""
    if isinstance(name_or_db, BaseVoiceDatabase):
        return name_or_db
    if name_or_db not in _voice_databases:
        from .voice_database import HuggingfaceVoiceDatabase
        _voice_databases[name_or_db] = HuggingfaceVoiceDatabase(name_or_db)
    return _voice_databases[name_or_db]


def _speaker_key(voices: dict, speaker: str, index: int):
    for key in (speaker, index, str(index)):
        if key in voices:
            return key
    return None


def _from_database(persona: dict, db: BaseVoiceDatabase, rng: np.random.Generator,
                   exclude: List[str]) -> Tuple[Any, dict]:
    gender = normalize_gender(persona.get("gender"), rng)
    age = normalize_age(persona.get("age"), rng)
    language = str(persona.get("language") or "english").lower()
    voice = None
    for _ in range(10):  # avoid giving the same voice to two speakers
        voice = db.get_voice(gender=gender, age=age, lang=language, keep_duplicate=True,
                             seed=int(rng.integers(2**31)))
        if voice.identifier not in exclude:
            break
    info = {"source": "db", "id": voice.identifier, "gender": voice.gender, "age": voice.age,
            "language": voice.language}
    return voice.voice, info


def _design(persona: dict, text: str, description: Union[str, Callable], seed: Optional[int]) -> Tuple[Any, dict]:
    global _voice_design_model
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise ImportError("voices='design' requires qwen-tts: `pip install qwen-tts`.") from e
    if _voice_design_model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _voice_design_model = Qwen3TTSModel.from_pretrained(VOICE_DESIGN_MODEL, device_map=device,
                                                            dtype=torch.bfloat16)
    desc = description(persona) if callable(description) else Template(description).render(**persona)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    wavs, sr = _voice_design_model.generate_voice_design(text=text, language=persona.get("language") or "English",
                                                         instruct=desc)
    return (np.asarray(wavs[0], dtype=np.float32), int(sr)), {"source": "design", "description": desc}


def resolve_voices(speakers: List[str],
                   personas: Dict[str, dict],
                   turns: List[Tuple[int, str]],
                   voices: Union[str, dict],
                   tts,
                   rng: np.random.Generator,
                   voice_database: Optional[Union[str, BaseVoiceDatabase]] = None,
                   voice_description: Optional[Union[str, Callable]] = None) -> Tuple[List[Any], Dict[str, dict]]:
    """
    Resolves the voice of every speaker.

    :param speakers: Speaker names, indexed as in ``turns``.
    :param personas: Persona dictionary of each speaker (may be missing or incomplete).
    :param turns: The ``(speaker_index, text)`` turns (used as reference text by "design").
    :param voices: "db", "design", "random", or an explicit ``{speaker: voice}`` dictionary (speakers missing
                   from the dictionary get a "db" voice, or a random one if the engine supports it).
    :param tts: The :class:`BaseDialogTTS` engine that will use the voices.
    :param rng: Random generator.
    :param voice_database: Voice database (or Hugging Face dataset name) for "db" voices.
    :param voice_description: Jinja2 template (or function of the persona dict) describing the voice for "design".
    :return: The voice of each speaker (``None`` = random) and a description of each speaker's voice.
    """
    if isinstance(voices, str) and voices not in VOICE_MODES:
        raise ValueError(f"Invalid voices '{voices}': use one of {VOICE_MODES} or a {{speaker: voice}} dictionary.")
    if not isinstance(voices, (str, dict)):
        raise TypeError(f"voices must be a string or a dictionary, got {type(voices).__name__}")

    explicit = voices if isinstance(voices, dict) else {}
    mode = voices if isinstance(voices, str) else ("random" if tts.supports_random_voices else "db")
    if mode == "random" and not tts.supports_random_voices:
        raise ValueError(f"{type(tts).__name__} cannot generate random voices; use voices='db' or 'design'.")
    if mode == "design" and not tts.supports_reference_audio:
        raise ValueError(f"{type(tts).__name__} does not support reference audio voices, required by "
                         "voices='design'.")

    resolved, info, used_ids = [], {}, []
    for ix, speaker in enumerate(speakers):
        persona = dict(personas.get(speaker) or {})
        key = _speaker_key(explicit, speaker, ix)
        if key is not None:
            voice = explicit[key]
            if isinstance(voice, Voice):
                info[speaker] = {"source": "user", "id": voice.identifier}
                voice = voice.voice
            else:
                info[speaker] = {"source": "user", "id": voice if isinstance(voice, str) else "audio"}
        elif mode == "random":
            voice, info[speaker] = None, {"source": "random"}
        elif mode == "design":
            text = next((t for s, t in turns if s == ix), "Hello, how are you doing today?")
            text = " ".join(text.split()[:40])
            seed = int(rng.integers(2**31))
            voice, info[speaker] = _design(persona, text, voice_description or DEFAULT_VOICE_DESCRIPTION, seed)
        else:
            db = get_voice_database(voice_database or tts.default_voice_database or DEFAULT_VOICE_DATABASE)
            voice, info[speaker] = _from_database(persona, db, rng, used_ids)
            used_ids.append(info[speaker]["id"])
        resolved.append(voice)
    return resolved, info
