# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import re
import logging
from abc import abstractmethod

logger = logging.getLogger(__name__)


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
