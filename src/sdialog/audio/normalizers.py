# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import re
import logging
from abc import abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


def normalize_audio(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """
    Perform RMS normalization on an audio signal.

    Centers all outputs around the same gain by normalizing
    to a target RMS level.

    :param audio: Input audio signal as numpy array.
    :type audio: np.ndarray
    :param target_rms: Target RMS level for normalization (default: 0.05).
    :type target_rms: float
    :return: RMS-normalized audio signal.
    :rtype: np.ndarray
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        audio = audio * (target_rms / current_rms)
    return audio


class TextNormalizer:
    """
    Abstract base class for text normalizers.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize the text given as input.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """
        raise NotImplementedError("TextNormalizer subclass must implement this method normalize")


def normalize_text(text: str, text_normalizers: list[TextNormalizer]) -> str:
    """
    Normalize the text given a list of text normalizers.
    :param text: The text to normalize.
    :type text: str
    :param text_normalizers: The list of text normalizers to apply.
    :type text_normalizers: list[TextNormalizer]
    :return: The normalized text.
    :rtype: str
    """
    for _normalizer in text_normalizers:
        text = _normalizer.normalize(text)
    return text


class LowercaseNormalizer(TextNormalizer):
    """
    Normalizer for lowercase.
    """

    def normalize(self, text: str) -> str:
        """
        Normalize the text to lowercase.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """
        return text.lower()


class StageNormalizer(TextNormalizer):
    """
    Normalizer for stage directions.
    This normalizer will remove the <stage>...</stage> tags from the text.
    """

    def normalize(self, text: str) -> str:
        """
        Normalize the text to remove the <stage>...</stage> tags from the text.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """
        text = re.sub(r"<stage>.*?</stage>", "", text).strip()
        text = re.sub(r"<STAGE>.*?</STAGE>", "", text).strip()
        return text


class AudioEventTagsNormalizer(TextNormalizer):
    """
    Normalizer for audio event tags.
    This normalizer will remove the [tag] tags from the text.
    """

    def normalize(self, text: str) -> str:
        """
        Normalize the text to remove the [tag] tags from the text.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """
        text = re.sub(r"\[.*?\]", "", text).strip()
        return text


class DocumentFormatNormalizer(TextNormalizer):
    """
    Normalizer for whitespace.
    This normalizer removes newlines, tabs, and multiple spaces.
    """

    def normalize(self, text: str) -> str:
        """
        Normalize the text to remove the newlines, tabs, and multiple spaces.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """

        text = text.replace("\n", "").replace("\t", "")

        text = re.sub(r"\s+", " ", text).strip()

        return text.strip()


class ReplaceCommaWithDotNormalizer(TextNormalizer):
    """
    Normalizer for replace comma with dot.
    """

    def normalize(self, text: str) -> str:
        """
        Normalize the text to replace comma with dot.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """
        return text.replace(",", ".")


class UnicodeToAsciiNormalizer(TextNormalizer):
    """
    Normalizer that replaces common Unicode characters with ASCII equivalents.
    Handles fancy quotes, dashes, special spaces, ligatures, bullets, fractions,
    and other characters commonly produced by LLMs that may confuse TTS engines.
    """

    UNICODE_REPLACEMENTS = {
        # Quotes
        "\u2018": "'",    # left single quotation mark
        "\u2019": "'",    # right single quotation mark
        "\u201A": "'",    # single low-9 quotation mark
        "\u201C": '"',    # left double quotation mark
        "\u201D": '"',    # right double quotation mark
        "\u201E": '"',    # double low-9 quotation mark
        "\u2039": "'",    # single left-pointing angle quotation
        "\u203A": "'",    # single right-pointing angle quotation
        "\u00AB": '"',    # left guillemet
        "\u00BB": '"',    # right guillemet
        "\u2032": "'",    # prime (feet, minutes)
        "\u2033": '"',    # double prime (inches, seconds)

        # Dashes and hyphens
        "\u2010": "-",    # hyphen
        "\u2011": "-",    # non-breaking hyphen
        "\u2012": "-",    # figure dash
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2015": "-",    # horizontal bar
        "\u2212": "-",    # minus sign

        # Spaces (collapse to normal space)
        "\u00A0": " ",    # non-breaking space
        "\u2002": " ",    # en space
        "\u2003": " ",    # em space
        "\u2009": " ",    # thin space
        "\u200A": " ",    # hair space
        "\u200B": "",     # zero-width space (remove)
        "\u200C": "",     # zero-width non-joiner (remove)
        "\u200D": "",     # zero-width joiner (remove)
        "\uFEFF": "",     # BOM / zero-width no-break space (remove)

        # Ellipsis
        "\u2026": "...",  # horizontal ellipsis

        # Ligatures
        "\uFB01": "fi",   # fi ligature
        "\uFB02": "fl",   # fl ligature

        # Bullets (replace with dash for TTS list reading)
        "\u2022": "-",    # bullet
        "\u2023": "-",    # triangular bullet
        "\u25E6": "-",    # white bullet

        # Fractions (spell out for TTS)
        "\u00BC": " one quarter",
        "\u00BD": " one half",
        "\u00BE": " three quarters",

        # Other common LLM outputs
        "\u00B0": " degrees",  # degree sign
    }

    def normalize(self, text: str) -> str:
        """
        Replace fancy Unicode characters with ASCII equivalents.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """
        for src, dst in self.UNICODE_REPLACEMENTS.items():
            text = text.replace(src, dst)
        return text


class WhisperNormalizer(TextNormalizer):
    """
    Normalizer for whisper.
    """

    def normalize(self, text: str) -> str:
        """
        Normalize the text for whisper.

        :param text: The text to normalize.
        :type text: str
        :return: The normalized text.
        :rtype: str
        """

        try:
            from whisper_normalization import (
                EnglishTextNormalizer,
                EnglishNumberNormalizer
            )

            numbers_normalizer = EnglishNumberNormalizer()
            text_normalizer = EnglishTextNormalizer()

            text = numbers_normalizer(text)
            text = text_normalizer(text)

        except ImportError:
            raise ImportError(
                "whisper_normalization is not installed, please install it with `pip install whisper-normalization`"
            )
        except Exception as e:
            logger.warning(f"Failed to normalize text with whisper_normalization: {e}")

        return text
