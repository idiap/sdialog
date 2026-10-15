"""
Vendored copy of the Dia2 inference runtime by Nari Labs (https://github.com/nari-labs/dia2),
commit 8687268f4ed3ed20704638fd353b51491de3b476, licensed under the Apache License 2.0
(see the LICENSE file in this folder).

Modifications made for sdialog (marked with ``[sdialog]`` in the code):

  - ``cli.py`` removed, and the one absolute ``dia2`` import made relative.
  - Compatibility with torch 2.8 (``torch.backends.cudnn.conv`` is only available in newer versions).
  - Speaker prefixes accept in-memory ``(array, sample_rate)`` audio in addition to file paths.
  - The Whisper model used to align prefixes is loaded once, and transcriptions are cached.
  - ``Dia2.generate`` accepts a ``seed`` argument.
"""
from .config import DiaConfig, load_config
from .core.model import Dia2Model
from .engine import Dia2
from .generation import (
    GenerationConfig,
    GenerationResult,
    PrefixConfig,
    SamplingConfig,
)

__all__ = [
    "DiaConfig",
    "Dia2Model",
    "load_config",
    "GenerationConfig",
    "GenerationResult",
    "PrefixConfig",
    "SamplingConfig",
    "Dia2",
]
