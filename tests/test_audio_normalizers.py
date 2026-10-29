"""Tests for the text and audio normalizers used by the TTS engines."""

from sdialog.audio.normalizers import DocumentFormatNormalizer


def test_document_format_keeps_words_apart():
    assert DocumentFormatNormalizer().normalize("Hello\nworld,\thow  are you?\n") == "Hello world, how are you?"
