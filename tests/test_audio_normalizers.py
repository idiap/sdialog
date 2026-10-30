"""Tests for the text and audio normalizers used by the TTS engines."""

import sys
import types

import numpy as np
import pytest

from sdialog.audio.normalizers import (normalize_audio, normalize_text, TextNormalizer, LowercaseNormalizer,
                                       StageNormalizer, AudioEventTagsNormalizer, DocumentFormatNormalizer,
                                       ReplaceCommaWithDotNormalizer, UnicodeToAsciiNormalizer, WhisperNormalizer)


def test_document_format_keeps_words_apart():
    assert DocumentFormatNormalizer().normalize("Hello\nworld,\thow  are you?\n") == "Hello world, how are you?"


def test_text_normalizers():
    assert LowercaseNormalizer().normalize("HeLLo") == "hello"
    assert StageNormalizer().normalize("Hi <stage>waves</stage> there <STAGE>smiles</STAGE>") == "Hi  there"
    assert AudioEventTagsNormalizer().normalize("[laughs] Sure [pause]") == "Sure"
    assert ReplaceCommaWithDotNormalizer().normalize("a, b") == "a. b"
    assert UnicodeToAsciiNormalizer().normalize("“Hi” — it’s 20°…") == '"Hi" - it\'s 20 degrees...'
    chain = [AudioEventTagsNormalizer(), DocumentFormatNormalizer(), LowercaseNormalizer()]
    assert normalize_text("[sigh]  Hello\nThere", chain) == "hello there"
    with pytest.raises(NotImplementedError):
        TextNormalizer().normalize("x")


def test_whisper_normalizer(monkeypatch):
    module = types.ModuleType("whisper_normalization")
    module.EnglishNumberNormalizer = lambda: (lambda text: text.replace("twenty", "20"))
    module.EnglishTextNormalizer = lambda: (lambda text: text.lower())
    monkeypatch.setitem(sys.modules, "whisper_normalization", module)
    assert WhisperNormalizer().normalize("I am Twenty") == "i am twenty"
    module.EnglishTextNormalizer = lambda: (lambda text: 1 / 0)
    assert WhisperNormalizer().normalize("Kept") == "Kept"  # errors are logged, the text is kept
    monkeypatch.setitem(sys.modules, "whisper_normalization", None)
    with pytest.raises(ImportError):
        WhisperNormalizer().normalize("x")


def test_normalize_audio():
    audio = normalize_audio(np.array([0.5, -0.5, 0.5, -0.5]), target_rms=0.1)
    assert np.sqrt(np.mean(audio ** 2)) == pytest.approx(0.1)
    assert normalize_audio(np.zeros(4)).tolist() == [0, 0, 0, 0]
