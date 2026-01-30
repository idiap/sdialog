"""
Evaluation components for dialogue generation and analysis.

This module provides classes for evaluating dialogues, including LLM judges, metrics, and similarity scores.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import logging
import syllables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from tqdm.auto import tqdm
from math import exp, log, sqrt
from pydantic import Field, create_model
from typing import Optional, Literal, Union, List, Dict, Tuple

from scipy import linalg
from scipy.stats import norm, gaussian_kde
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
from langchain_core.language_models.base import BaseLanguageModel

from .. import Dialog
from ..config import config
from ..personas import BasePersona
from ..util import SentencePairTransformer
from ..util import dict_to_table, upper_camel_to_dash, dialogs_to_utt_pairs

from .base import LLMJudgeYesNoOutput, LLMJudgeScoreOutput
from .base import BaseDatasetEvaluator, BaseDatasetScoreEvaluator, BaseDatasetEmbeddingEvaluator
from .base import CacheDialogScore, BaseLLMJudge, BaseDialogEmbedder, BaseDialogScore, BaseDialogFlowScore

logger = logging.getLogger(__name__)

# Configure matplotlib for publication-quality figures
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 13
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['grid.linewidth'] = 0.6
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['patch.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['xtick.minor.width'] = 0.8
mpl.rcParams['ytick.minor.width'] = 0.8
mpl.rcParams['text.usetex'] = False  # Set to True if LaTeX is available
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better PDF compatibility
mpl.rcParams['ps.fonttype'] = 42


def _add_jitter(x, eps=1e-8):
    return x + np.random.normal(0, eps, size=len(x))


def _cs_divergence(p1, p2, resolution=100, bw_method=1):
    """
    Calculate the Cauchy-Schwarz divergence between two 1D distributions via KDE.

    :param p1: First sample (1D array-like).
    :type p1: array-like
    :param p2: Second sample (1D array-like).
    :type p2: array-like
    :param resolution: Number of evaluation points for KDE grid.
    :type resolution: int
    :param bw_method: KDE bandwidth (scalar or method string).
    :type bw_method: Union[float, str]
    :return: CS divergence (0 means identical distributions); None if either sample empty.
    :rtype: Optional[float]
    """
    if len(p1) == 0 or len(p2) == 0:
        logger.error("Both input distributions must have at least one sample. Returning None")
        return None
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    r = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), resolution)

    try:
        p1_kernel = gaussian_kde(p1, bw_method=bw_method)
        p2_kernel = gaussian_kde(p2, bw_method=bw_method)
    except Exception as e:
        if np.array_equal(p1, p2):
            return 0.0
        logger.error(f"Error computing KDEs for KL divergence: {e}. Trying with jitter...")
        if np.std(p1) == 0:
            p1 = _add_jitter(p1)
        if np.std(p2) == 0:
            p2 = _add_jitter(p2)
        p1_kernel = gaussian_kde(p1, bw_method=bw_method)
        p2_kernel = gaussian_kde(p2, bw_method=bw_method)

    p1_vals = p1_kernel(r)
    p2_vals = p2_kernel(r)
    numerator = np.sum(p1_vals * p2_vals)
    denominator = sqrt(np.sum(p1_vals ** 2) * np.sum(p2_vals ** 2))
    # Avoid log(0) and division by zero by ensuring minimum values
    return -log(max(numerator, 1e-12) / max(denominator, 1e-12))


def _kl_divergence(p1, p2, resolution=100, bw_method=1e-1):
    """
    Estimate KL divergence KL(p1 || p2) between two 1D distributions via KDE.

    KL(p1||p2) is non-symmetric and >= 0 (0 means identical).

    :param p1: First sample (treat as true distribution).
    :type p1: array-like
    :param p2: Second sample (approximate distribution).
    :type p2: array-like
    :param resolution: Number of evaluation points for KDE grid.
    :type resolution: int
    :param bw_method: KDE bandwidth parameter.
    :type bw_method: Union[float, str]
    :return: KL divergence value; None if either sample empty.
    :rtype: Optional[float]
    """
    if len(p1) == 0 or len(p2) == 0:
        logger.error("Both input distributions must have at least one sample. Returning None")
        return None

    r = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), resolution)

    try:
        p1_kernel = gaussian_kde(p1, bw_method=bw_method)
        p2_kernel = gaussian_kde(p2, bw_method=bw_method)
    except Exception as e:
        if np.array_equal(p1, p2):
            return 0.0
        logger.error(f"Error computing KDEs for KL divergence: {e}. Trying with jitter...")
        if np.std(p1) == 0:
            p1 = _add_jitter(p1)
        if np.std(p2) == 0:
            p2 = _add_jitter(p2)
        p1_kernel = gaussian_kde(p1, bw_method=bw_method)
        p2_kernel = gaussian_kde(p2, bw_method=bw_method)

    p1_vals = p1_kernel(r)
    p2_vals = p2_kernel(r)
    # Avoid division by zero and log(0) by adding a small epsilon
    eps = 1e-12
    p1_vals = np.clip(p1_vals, eps, None)
    p2_vals = np.clip(p2_vals, eps, None)

    sum_p1_vals = np.sum(p1_vals)
    # Protect against division by zero
    if sum_p1_vals == 0:
        return 0.0
    return float(np.sum(p1_vals * np.log(p1_vals / p2_vals)) / sum_p1_vals)


class ConversationalFeatures(BaseDialogScore):
    """
    Compute conversational and dialogue-specific features.

    These metrics measure dialogue structure, speech patterns, and interaction dynamics
    rather than text readability.

    Example:

        .. code-block:: python

            from sdialog.evaluation import ConversationalFeatures

            # All conversational features
            scorer_all = ConversationalFeatures()
            # Single feature
            scorer_hes = ConversationalFeatures(feature="hesitation-rate")
            # Multiple features
            scorer_multi = ConversationalFeatures(feature=["question-rate", "lexical-diversity"])

            print(scorer_all(dialog))      # dict with all feature values
            print(scorer_hes(dialog))      # single float (hesitation rate)
            print(scorer_multi(dialog))    # dict with selected features

    :param feature: List of feature names to compute. If ``None`` (default) compute all.
                    If the resulting set has size 1, ``__call__`` / ``score`` returns a single float; otherwise a dict.
                    Available features:

                      - ``"mean-turn-length"``: average number of words per dialogue turn.
                      - ``"hesitation-rate"``: percentage of hesitation tokens over total words (%).
                      - ``"turn-taking-ratio"``: distribution of turns between speakers
                                                 (entropy-based, 0=monopolized, 1=balanced).
                      - ``"question-rate"``: percentage of turns containing questions (%).
                      - ``"lexical-diversity"``: type-token ratio measuring vocabulary richness (0-1).
                      - ``"back-channel-rate"``: percentage of minimal response turns (%).
                      - ``"filler-word-density"``: percentage of filler words over total words (%).

    :type feature: Optional[List[Literal["mean-turn-length", "hesitation-rate", "turn-taking-ratio",
                                         "question-rate", "lexical-diversity", "back-channel-rate",
                                         "filler-word-density"]]]
    :param name: Internal score name (defaults to ``"conversational_features"`` or
                 the single feature name if only one provided).
    :type name: str
    :param speaker: If set, only turns by this speaker (case-insensitive) are considered.
                    Note: turn-taking-ratio ignores this parameter as it requires multiple speakers.
    :type speaker: Optional[str]
    """
    def __init__(self,
                 feature: Optional[List[Literal["mean-turn-length", "hesitation-rate", "turn-taking-ratio",
                                                "question-rate", "lexical-diversity", "back-channel-rate",
                                                "filler-word-density"]]] = None,
                 name: str = None,
                 speaker: Optional[str] = None):
        """Initialize conversational features scorer."""
        if isinstance(feature, str):
            feature = [feature]

        # Check all features valid
        valid_features = {"mean-turn-length", "hesitation-rate", "turn-taking-ratio",
                          "question-rate", "lexical-diversity", "back-channel-rate", "filler-word-density"}
        if feature and not all(f in valid_features for f in feature):
            raise ValueError(f"Invalid feature(s): {feature}. Must be one or more of: {valid_features}")

        # If a single feature is requested, allow name override with that feature for clearer downstream tables
        if name is None:
            if feature and isinstance(feature, list) and len(feature) == 1:
                name = feature[0]

        super().__init__(name=name or "conversational_features")
        self.feature = feature
        self.speaker = speaker

    @staticmethod
    def count_hesitations(text):
        """
        Count hesitation tokens in the provided text (e.g., uh, um, hmm).

        :param text: Input text to search for hesitation markers.
        :type text: str
        :return: Number of detected hesitation tokens in the provided text.
        :rtype: int
        """
        hesitation_patterns = re.compile(
            r'\b(?:uh+|um+|er+|ahh*|ohh*|hmm+|huh+|mm+|mhm+|uh-huh|um-hum+)\b',
            flags=re.IGNORECASE
        )
        return len(hesitation_patterns.findall(text))

    @staticmethod
    def count_filler_words(text):
        """
        Count filler words in the provided text (e.g., like, you know, I mean, basically).

        :param text: Input text to search for filler words.
        :type text: str
        :return: Number of detected filler words in the provided text.
        :rtype: int
        """
        filler_patterns = re.compile(
            r'\b(?:like|you know|i mean|basically|actually|literally|sort of|kind of|'
            r'stuff like that|or something|whatever|anyway)\b',
            flags=re.IGNORECASE
        )
        return len(filler_patterns.findall(text))

    @staticmethod
    def is_back_channel(turn_text):
        """
        Check if a turn is a back-channel response (minimal acknowledgment).

        :param turn_text: Text of a single turn.
        :type turn_text: str
        :return: True if the turn is a back-channel response.
        :rtype: bool
        """
        text = turn_text.strip().lower()
        back_channel_patterns = [
            r'^(?:yeah|yes|yep|yup|uh-huh|mhm|mm-hmm|okay|ok|right|sure|i see|got it|alright)$',
            r'^(?:oh|ah|hmm|huh)$',
        ]
        return any(re.match(pattern, text) for pattern in back_channel_patterns)

    @staticmethod
    def calculate_turn_taking_ratio(dialog):
        """
        Calculate turn-taking balance using normalized entropy.

        Returns a value between 0 (monopolized conversation) and 1 (perfectly balanced).
        Based on Shannon entropy normalized by maximum possible entropy.

        :param dialog: Dialog object with turns from multiple speakers.
        :type dialog: Dialog
        :return: Turn-taking ratio (0-1).
        :rtype: float
        """
        from collections import Counter
        import math

        if len(dialog) == 0:
            return 0.0

        speaker_turns = Counter(turn.speaker for turn in dialog if hasattr(turn, 'speaker'))

        if len(speaker_turns) <= 1:
            return 0.0

        total_turns = sum(speaker_turns.values())

        entropy = 0.0
        for count in speaker_turns.values():
            if count > 0:
                p = count / total_turns
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(speaker_turns))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def score(self, dialog: Dialog) -> Union[float, dict]:
        """Compute one or multiple conversational features for the dialogue.

        :param dialog: Dialogue instance to evaluate.
        :type dialog: Dialog
        :return: If exactly one feature is requested, returns a single ``float``.
                 Otherwise returns a ``dict`` mapping feature-name to numeric value.
        :rtype: Union[float, dict]
        """
        original_dialog = dialog

        if self.speaker and len(dialog) > 0:
            dialog = dialog.filter(speaker=self.speaker)

        results = {}
        all_text = str(dialog)
        turn_lengths = [len(turn) for turn in dialog]
        total_words = sum(turn_lengths)

        if not self.feature or "mean-turn-length" in self.feature:
            results["mean-turn-length"] = float(np.mean(turn_lengths)) if turn_lengths else 0.0

        if not self.feature or "hesitation-rate" in self.feature:
            results["hesitation_rate"] = (self.count_hesitations(all_text) / max(1, total_words) * 100)

        if not self.feature or "turn-taking-ratio" in self.feature:
            results["turn_taking_ratio"] = self.calculate_turn_taking_ratio(original_dialog)

        if not self.feature or "question-rate" in self.feature:
            question_turns = sum(1 for turn in dialog if '?' in str(turn))
            results["question_rate"] = (question_turns / max(1, len(dialog)) * 100)

        if not self.feature or "lexical-diversity" in self.feature:
            words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
            unique_words = len(set(words))
            results["lexical_diversity"] = (unique_words / max(1, len(words))) if words else 0.0

        if not self.feature or "back-channel-rate" in self.feature:
            back_channel_turns = sum(1 for turn in dialog if self.is_back_channel(str(turn)))
            results["back_channel_rate"] = (back_channel_turns / max(1, len(dialog)) * 100)

        if not self.feature or "filler-word-density" in self.feature:
            results["filler_word_density"] = (self.count_filler_words(all_text) / max(1, total_words) * 100)

        return results if len(results) > 1 else list(results.values())[0]


class ReadabilityScore(BaseDialogScore):
    """
    Compute one or multiple readability metrics for a dialogue text: Gunning Fog index,
    Flesch Reading Ease score, Coleman-Liau Index, Linsear Write metric, and Dale-Chall
    Readability Formula.

    These metrics measure text complexity and reading difficulty, not dialogue structure.

    Example:

        .. code-block:: python

            from sdialog.evaluation import ReadabilityScore

            # All readability metrics
            scorer_all = ReadabilityScore()
            # Single metric
            scorer_flesch = ReadabilityScore(feature="flesch-reading-ease")
            # Subset of metrics
            scorer_subset = ReadabilityScore(feature=["gunning-fog", "coleman-liau"])

            print(scorer_all(dialog))      # dict with all metric values
            print(scorer_flesch(dialog))   # single float (Flesch score)
            print(scorer_subset(dialog))   # dict with the two requested metrics

    :param feature: List of feature names to compute. If ``None`` (default) compute all.
                    If the resulting set has size 1, ``__call__`` / ``score`` returns a
                    single float; otherwise a dict.
                    Available features:

                      - ``"gunning-fog"``: Gunning Fog readability index.
                      - ``"flesch-reading-ease"``: Flesch Reading Ease score.
                      - ``"coleman-liau"``: Coleman-Liau Index.
                      - ``"linsear-write"``: Linsear Write readability metric.
                      - ``"dale-chall"``: Dale-Chall Readability Formula.

    :type feature: Optional[List[Literal["gunning-fog", "flesch-reading-ease",
                                         "coleman-liau", "linsear-write", "dale-chall"]]]
    :param name: Internal score name (defaults to ``"readability_score"`` or
                 the single feature name if only one provided).
    :type name: str
    :param speaker: If set, only turns by this speaker (case-insensitive) are considered.
    :type speaker: Optional[str]
    """
    def __init__(self,
                 feature: Optional[List[Literal["gunning-fog", "flesch-reading-ease",
                                                "coleman-liau", "linsear-write", "dale-chall"]]] = None,
                 name: str = None,
                 speaker: Optional[str] = None):
        """Initialize readability scorer."""
        if isinstance(feature, str):
            feature = [feature]

        # Check all features valid
        valid_features = {"gunning-fog", "flesch-reading-ease", "coleman-liau", "linsear-write", "dale-chall"}
        if feature and not all(f in valid_features for f in feature):
            raise ValueError(f"Invalid feature(s): {feature}. Must be one or more of: {valid_features}")

        # If a single feature is requested, allow name override with that feature for clearer downstream tables
        if name is None:
            if feature and isinstance(feature, list) and len(feature) == 1:
                name = feature[0]

        # If a single feature is requested, allow name override with that feature for clearer downstream tables
        if name is None:
            if feature and isinstance(feature, list) and len(feature) == 1:
                name = feature[0]

        super().__init__(name=name or "readability_score")
        self.feature = feature
        self.speaker = speaker

    @staticmethod
    def calculate_gunning_fog(text):
        """
        Compute the Gunning Fog index of the provided text.

        :param text: Input text.
        :type text: str
        :return: Gunning Fog index value.
        :rtype: float
        """
        sentences = re.split(r'[.!?\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        total_words = len(words)
        complex_words = sum(1 for word in words if syllables.estimate(word) >= 3)
        avg_sentence_length = total_words / len(sentences)
        complex_word_ratio = (complex_words / total_words) * 100 if total_words > 0 else 0
        return 0.4 * (avg_sentence_length + complex_word_ratio)

    @staticmethod
    def calculate_flesch_reading_ease(text):
        """
        Compute the Flesch Reading Ease score of the provided text.

        :param text: Input text.
        :type text: str
        :return: Reading ease score.
        :rtype: float
        """
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0
        total_syllables = sum(syllables.estimate(word) for word in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        return 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

    @staticmethod
    def calculate_coleman_liau(text):
        """
        Compute the Coleman-Liau Index of the provided text.

        The Coleman-Liau Index estimates the U.S. grade level needed to understand the text.
        It uses character counts instead of syllable counts.

        :param text: Input text.
        :type text: str
        :return: Coleman-Liau Index value (minimum 0).
        :rtype: float
        """
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0
        # Count letters (characters)
        letters = sum(len(word) for word in words)
        # Calculate averages per 100 words
        L = (letters / len(words)) * 100  # average letters per 100 words
        S = (len(sentences) / len(words)) * 100  # average sentences per 100 words
        # Clamp to minimum of 0 (negative grade levels don't make sense)
        return max(0, 0.0588 * L - 0.296 * S - 15.8)

    @staticmethod
    def calculate_linsear_write(text):
        """
        Compute the Linsear Write readability metric of the provided text.

        The Linsear Write formula estimates the U.S. grade level needed to understand the text.
        It focuses on easy vs. difficult words (based on syllable count).

        :param text: Input text.
        :type text: str
        :return: Linsear Write score.
        :rtype: float
        """
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words or len(words) < 1:
            return 0

        # Count easy (1-2 syllables) and difficult (3+ syllables) words
        easy_words = 0
        difficult_words = 0
        for word in words:
            syllable_count = syllables.estimate(word)
            if syllable_count <= 2:
                easy_words += 1
            else:
                difficult_words += 1

        # Calculate per 100 words, then adjust
        total_words = len(words)
        score = ((easy_words * 1 + difficult_words * 3) / total_words) * 100 / len(sentences)

        # Adjust score according to Linsear Write formula
        if score > 20:
            score = score / 2
        else:
            score = (score - 2) / 2

        return max(0, score)

    @staticmethod
    def calculate_dale_chall(text):
        """
        Compute the Dale-Chall Readability Formula score of the provided text.

        The Dale-Chall formula uses a list of 3000 familiar words that 80% of 4th-grade students
        understand. Words not on this list are considered "difficult".

        Note: This implementation uses a simplified approximation based on word length and
        syllable count as a proxy for the Dale-Chall word list, since the full list is proprietary.

        :param text: Input text.
        :type text: str
        :return: Dale-Chall score.
        :rtype: float
        """
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0

        # Simplified approximation: consider words as "difficult" if they have 3+ syllables
        # or are longer than 7 characters (proxy for Dale-Chall familiar word list)
        difficult_words = 0
        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 7 or syllables.estimate(word) >= 3:
                difficult_words += 1

        # Calculate percentage of difficult words
        percent_difficult = (difficult_words / len(words)) * 100

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)

        # Dale-Chall formula
        score = 0.1579 * percent_difficult + 0.0496 * avg_sentence_length

        # Add constant if more than 5% difficult words
        if percent_difficult > 5:
            score += 3.6365

        return score

    def score(self, dialog: Dialog) -> Union[float, dict]:
        """Compute one or multiple readability metrics for the dialogue.

        :param dialog: Dialogue instance to evaluate.
        :type dialog: Dialog
        :return: If exactly one metric is requested, returns a single ``float``.
                 Otherwise returns a ``dict`` mapping metric-name to numeric value.
        :rtype: Union[float, dict]
        """
        if self.speaker and len(dialog) > 0:
            dialog = dialog.filter(speaker=self.speaker)

        results = {}
        all_text = str(dialog)
        if not self.feature or "gunning-fog" in self.feature:
            results["gunning_fog"] = self.calculate_gunning_fog(all_text)
        if not self.feature or "flesch-reading-ease" in self.feature:
            results["flesch_reading_ease"] = self.calculate_flesch_reading_ease(all_text)
        if not self.feature or "coleman-liau" in self.feature:
            results["coleman_liau"] = self.calculate_coleman_liau(all_text)
        if not self.feature or "linsear-write" in self.feature:
            results["linsear_write"] = self.calculate_linsear_write(all_text)
        if not self.feature or "dale-chall" in self.feature:
            results["dale_chall"] = self.calculate_dale_chall(all_text)

        return results if len(results) > 1 else list(results.values())[0]


class MeanTurnLengthScore(ConversationalFeatures):
    """
    Compute the mean turn length (average number of words per turn) for a dialogue.

    This is a conversational metric that measures dialogue structure, not text readability.

    Example:

        .. code-block:: python

            from sdialog.evaluation import MeanTurnLengthScore

            scorer = MeanTurnLengthScore()
            print(scorer(dialog))  # Outputs mean turn length as float

    :param name: Optional score name (defaults to "mean-turn-length").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize mean turn length scorer."""
        super().__init__(feature="mean-turn-length", name=name, speaker=speaker)


class TurnLength(BaseDialogScore):
    """
    Compute individual turn lengths (number of words per turn) for a dialogue.

    Returns a list of word counts for each turn in the dialogue. This is a granular metric
    that captures turn length distribution, often used as raw input for downstream aggregations
    (e.g., computing mean or median turn length).

    Example:

        .. code-block:: python

            from sdialog.evaluation import TurnLength

            scorer = TurnLength()
            lengths = scorer(dialog)  # Returns list of integers
            print(lengths)  # [5, 12, 3, 18, ...] words per turn

            # Filter by speaker
            scorer_system = TurnLength(speaker="System")
            system_lengths = scorer_system(dialog)

    :param name: Optional score name (defaults to "turn-length").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker (case-insensitive) are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize turn length scorer."""
        super().__init__(name=name or "turn-length", ai_speaker=speaker)

    def score(self, dialog: Dialog) -> List[int]:
        """
        Compute word count for each turn in the dialogue.

        :param dialog: Dialogue instance to evaluate.
        :type dialog: Dialog
        :return: List of integers representing word count per turn.
        :rtype: List[int]
        """
        if self.ai_speaker is None:
            return [len(turn) for turn in dialog]
        else:
            return [len(turn) for turn in dialog if turn.speaker.lower() == self.ai_speaker.lower()]


class HesitationRateScore(ConversationalFeatures):
    """
    Compute the hesitation rate (percentage of hesitation tokens) for a dialogue.

    This is a conversational metric that measures speech disfluencies, not text readability.

    Example:

        .. code-block:: python

            from sdialog.evaluation import HesitationRateScore

            scorer = HesitationRateScore()
            print(scorer(dialog))  # Outputs hesitation rate as percentage

    :param name: Optional score name (defaults to "hesitation-rate").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize hesitation rate scorer."""
        super().__init__(feature="hesitation-rate", name=name, speaker=speaker)


class TurnTakingRatioScore(ConversationalFeatures):
    """
    Compute the turn-taking ratio (balance of conversation between speakers) for a dialogue.

    Returns a value between 0 (monopolized) and 1 (perfectly balanced), based on normalized
    Shannon entropy of turn distribution across speakers.

    Example:

        .. code-block:: python

            from sdialog.evaluation import TurnTakingRatioScore

            scorer = TurnTakingRatioScore()
            print(scorer(dialog))  # Outputs turn-taking ratio (0-1)

    :param name: Optional score name (defaults to "turn-taking-ratio").
    :type name: Optional[str]
    """
    def __init__(self, name: str = None):
        """Initialize turn-taking ratio scorer."""
        super().__init__(feature="turn-taking-ratio", name=name, speaker=None)


class QuestionRateScore(ConversationalFeatures):
    """
    Compute the question rate (percentage of turns containing questions) for a dialogue.

    This metric measures the interrogative nature of the conversation.

    Example:

        .. code-block:: python

            from sdialog.evaluation import QuestionRateScore

            scorer = QuestionRateScore()
            print(scorer(dialog))  # Outputs question rate as percentage

    :param name: Optional score name (defaults to "question-rate").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize question rate scorer."""
        super().__init__(feature="question-rate", name=name, speaker=speaker)


class LexicalDiversityScore(ConversationalFeatures):
    """
    Compute the lexical diversity (type-token ratio) for a dialogue.

    Measures vocabulary richness as the ratio of unique words to total words (0-1).
    Higher values indicate more varied vocabulary.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LexicalDiversityScore

            scorer = LexicalDiversityScore()
            print(scorer(dialog))  # Outputs lexical diversity (0-1)

    :param name: Optional score name (defaults to "lexical-diversity").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize lexical diversity scorer."""
        super().__init__(feature="lexical-diversity", name=name, speaker=speaker)


class BackChannelRateScore(ConversationalFeatures):
    """
    Compute the back-channel rate (percentage of minimal response turns) for a dialogue.

    Back-channels are brief responses like "yeah", "okay", "I see" that indicate active listening
    without contributing substantial content.

    Example:

        .. code-block:: python

            from sdialog.evaluation import BackChannelRateScore

            scorer = BackChannelRateScore()
            print(scorer(dialog))  # Outputs back-channel rate as percentage

    :param name: Optional score name (defaults to "back-channel-rate").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize back-channel rate scorer."""
        super().__init__(feature="back-channel-rate", name=name, speaker=speaker)


class FillerWordDensityScore(ConversationalFeatures):
    """
    Compute the filler word density (percentage of filler words) for a dialogue.

    Filler words include "like", "you know", "I mean", "basically", etc.
    These differ from hesitations and indicate informal speech patterns.

    Example:

        .. code-block:: python

            from sdialog.evaluation import FillerWordDensityScore

            scorer = FillerWordDensityScore()
            print(scorer(dialog))  # Outputs filler word density as percentage

    :param name: Optional score name (defaults to "filler-word-density").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize filler word density scorer."""
        super().__init__(feature="filler-word-density", name=name, speaker=speaker)


class GunningFogScore(ReadabilityScore):
    """
    Compute the Gunning Fog readability index for a dialogue.

    The Gunning Fog index estimates the years of formal education needed to understand
    the text on a first reading. Higher values indicate more complex text.

    Example:

        .. code-block:: python

            from sdialog.evaluation import GunningFogScore

            scorer = GunningFogScore()
            print(scorer(dialog))  # Outputs Gunning Fog index as float

    :param name: Optional score name (defaults to "gunning-fog").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize Gunning Fog index scorer."""
        super().__init__(feature="gunning-fog", name=name, speaker=speaker)


class FleschReadingEaseScore(ReadabilityScore):
    """
    Compute the Flesch Reading Ease score for a dialogue.

    The Flesch Reading Ease score rates text on a 100-point scale. Higher scores indicate
    text that is easier to read. Scores typically range from 0 (very difficult) to 100 (very easy).

    Example:

        .. code-block:: python

            from sdialog.evaluation import FleschReadingEaseScore

            scorer = FleschReadingEaseScore()
            print(scorer(dialog))  # Outputs Flesch Reading Ease score as float

    :param name: Optional score name (defaults to "flesch-reading-ease").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize Flesch Reading Ease scorer."""
        super().__init__(feature="flesch-reading-ease", name=name, speaker=speaker)


class ColemanLiauScore(ReadabilityScore):
    """
    Compute the Coleman-Liau Index for a dialogue.

    The Coleman-Liau Index estimates the U.S. grade level needed to understand the text.
    Unlike other readability formulas, it uses character counts instead of syllable counts,
    making it more suitable for automated text analysis.

    Example:

        .. code-block:: python

            from sdialog.evaluation import ColemanLiauScore

            scorer = ColemanLiauScore()
            print(scorer(dialog))  # Outputs Coleman-Liau Index as float

    :param name: Optional score name (defaults to "coleman-liau").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize Coleman-Liau Index scorer."""
        super().__init__(feature="coleman-liau", name=name, speaker=speaker)


class LinsearWriteScore(ReadabilityScore):
    """
    Compute the Linsear Write readability metric for a dialogue.

    The Linsear Write formula estimates the U.S. grade level needed to understand the text.
    It focuses on easy versus difficult words (based on syllable count) and is particularly
    useful for technical writing assessment.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LinsearWriteScore

            scorer = LinsearWriteScore()
            print(scorer(dialog))  # Outputs Linsear Write score as float

    :param name: Optional score name (defaults to "linsear-write").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize Linsear Write scorer."""
        super().__init__(feature="linsear-write", name=name, speaker=speaker)


class DaleChallScore(ReadabilityScore):
    """
    Compute the Dale-Chall Readability Formula score for a dialogue.

    The Dale-Chall formula uses a list of familiar words that most 4th-grade students understand.
    Words not on this list are considered "difficult". This implementation uses a simplified
    approximation based on word length and syllable count as a proxy for the Dale-Chall word list.

    Example:

        .. code-block:: python

            from sdialog.evaluation import DaleChallScore

            scorer = DaleChallScore()
            print(scorer(dialog))  # Outputs Dale-Chall score as float

    :param name: Optional score name (defaults to "dale-chall").
    :type name: Optional[str]
    :param speaker: If set, only turns by this speaker are considered.
    :type speaker: Optional[str]
    """
    def __init__(self, name: str = None, speaker: Optional[str] = None):
        """Initialize Dale-Chall scorer."""
        super().__init__(feature="dale-chall", name=name, speaker=speaker)


class ToolSequenceValidator(BaseDialogScore):
    """
    Validate that an agent used specific tools in the correct sequence during a dialogue.

    This validator checks whether the agent called the specified tools in the expected order
    based on the dialogue's event history. It returns 1 if the sequence is valid, 0 otherwise.

    Tool names can be prefixed with ``"not:"`` to indicate that the tool must NOT be called
    before subsequent tools in the list. This allows for flexible validation of tool usage patterns.

    Example 1: Basic sequence validation

        .. code-block:: python

            from sdialog.evaluation import ToolSequenceValidator

            # Validate that tools were called in exact order
            validator = ToolSequenceValidator(["search_flights", "book_flight", "confirm_booking"])

            score = validator(dialog)
            print(score)  # 1 if sequence correct, 0 otherwise

    Example 2: Using negative constraints

        .. code-block:: python

            from sdialog.evaluation import ToolSequenceValidator

            # Ensure send_receipt is NOT called before charging payment
            # (don't send receipt before actually charging the customer)
            # But send_receipt may be called after charge_payment, or not at all
            validator = ToolSequenceValidator([
                "not:send_receipt",
                "charge_payment",
                "update_inventory"
            ])

            score = validator(dialog)

    Example 3: With evaluators

        .. code-block:: python

            from sdialog.evaluation import ToolSequenceValidator, FrequencyEvaluator

            validator = ToolSequenceValidator(["authenticate", "fetch_data", "logout"])
            freq_eval = FrequencyEvaluator(validator)

            # Get percentage of dialogues with correct tool sequence
            percentage = freq_eval(dialogs)
            print(f"{percentage * 100:.1f}% of dialogues follow correct sequence")

    :param tool_names: List of tool names defining the expected sequence. Each tool name can be:
                       - A plain string (e.g., ``"search_flights"``): tool must be called in sequence.
                       - Prefixed with ``"not:"`` (e.g., ``"not:verify_account"``): tool must NOT be
                         called before the next required tool in the sequence, though it may be called
                         after or omitted entirely.
    :type tool_names: List[str]
    :param name: Custom score name (defaults to ``"tool-sequence-validator"``).
    :type name: str

    .. note::
        - Tools must appear in the specified order within the dialogue's event history.
        - The first tool in the sequence must come after at least one user utterance.
        - If a required tool (without ``"not:"`` prefix) is missing, the score is 0.
        - Tools with ``"not:"`` prefix that don't appear in the dialogue are ignored.
    """
    def __init__(self, tool_names: List[str], name: str = "tool-sequence-validator"):
        """
        Initialize the tool sequence validator.

        :param tool_names: List of tool names in expected order (may include "not:" prefixes).
        :type tool_names: List[str]
        :param name: Score name for reporting.
        :type name: str
        """
        super().__init__(name=name)
        self.tool_names = tool_names

    def score(self, dialog: Dialog) -> int:
        """
        Compute the validation score for the dialogue's tool usage sequence.

        Extracts tool calls from the dialogue's event history and validates that:
        1. All required tools (without ``"not:"`` prefix) are present.
        2. Tools appear in the specified order.
        3. Tools with ``"not:"`` prefix do not appear before subsequent tools.
        4. The first tool call comes after at least one user utterance.

        :param dialog: Dialogue instance to validate.
        :type dialog: Dialog
        :return: 1 if the tool sequence is valid, 0 otherwise.
        :rtype: int

        .. note::
            Returns 0 if:
            - The dialogue has no events or tool_names is empty.
            - A required tool is missing from the event history.
            - Tools appear in incorrect order.
            - A ``"not:"`` prefixed tool appears before subsequent tools.
        """
        # Check if all tools were used in that order from the event list.
        if not dialog.events or not self.tool_names:
            return 0

        # Extract tool calls from events in order
        event_name_list = []
        for event in dialog.events:
            if event.action == "tool" and event.actionLabel == "call" and isinstance(event.content, dict):
                event_name_list.append(event.content["name"])
            else:
                event_name_list.append(event.action)

        # Process tool names and check ordering constraints
        try:
            indices = []
            for i, tool in enumerate(self.tool_names):
                if tool.startswith("not:"):
                    # Tool must NOT be called before subsequent tools
                    actual_tool_name = tool.split(":", 1)[1]
                    try:
                        tool_index = event_name_list.index(actual_tool_name)
                        # If this tool exists, check it comes after the previous tool (if any)
                        if indices and tool_index <= indices[-1]:
                            return 0  # Tool was called before the previous one
                        indices.append(tool_index)
                    except ValueError:
                        # Tool not found - this is OK for "not:" prefixed tools
                        # Use a placeholder that won't break ordering (previous index or -1)
                        indices.append(indices[-1] if indices else -1)
                else:
                    # Tool must be called
                    tool_index = event_name_list.index(tool)
                    indices.append(tool_index)

            # Check tools are in correct order and first tool comes after utterance
            return 1 if indices == sorted(indices) and (not indices or indices[0] > 0 or indices[0] == -1) else 0
        except ValueError:
            # One or more required tools not found
            return 0


class DialogFlowPPL(BaseDialogFlowScore):
    """
    Compute flow perplexity-like score of a dialogue against reference dialogues.

    Given a collection of reference dialogues, it first builds the dialogue flow graph that
    represent them. Then, given a candidate dialogue, it computes a flow perplexity-like score
    (i.e. "how well it fits on the reference graph in terms of perplexity?").

    Example:

        .. code-block:: python

            from sdialog.evaluation import DialogFlowPPL

            # reference_dialogs = [...]
            flow_ppl = DialogFlowPPL(reference_dialogs)

            value = flow_ppl(candidate_dialog)

            print("Flow Perplexity:", value)

    :param reference_dialogues: List of reference dialogues or file path.
    :type reference_dialogues: Union[str, List[Dialog]]
    :param ai_speaker: If set, restrict scoring to AI/system turns.
    :type ai_speaker: Optional[str]
    :param k_neighbors: Neighbor count for embedding lookup.
    :type k_neighbors: int
    :param use_softmax: Whether to weight neighbors via softmax.
    :type use_softmax: bool
    :param use_only_known_edges: If True, ignore unknown transitions (penalize less).
    :type use_only_known_edges: bool
    :param name: Custom score name override.
    :type name: Optional[str]
    :param verbose: Verbosity flag.
    :type verbose: bool
    :param d2f_kwargs: Extra kwargs to dialog2graph.
    :type d2f_kwargs: dict
    """
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 k_neighbors: int = 64,
                 use_softmax: bool = True,
                 use_only_known_edges: bool = False,
                 name: str = None,
                 verbose: bool = False,
                 **d2f_kwargs):
        """Initialize flow perplexity evaluator."""
        self.use_only_known_edges = use_only_known_edges
        if name is None:
            name = "dfppl" + ("" if use_softmax else "-hard") + ("-ai" if ai_speaker else "")
            name += "-only-known" if use_only_known_edges else ""
        super().__init__(
            reference_dialogues,
            ai_speaker=ai_speaker,
            k_neighbors=k_neighbors,
            use_softmax=use_softmax,
            name=name,
            verbose=verbose,
            **d2f_kwargs
        )

    @CacheDialogScore.cache
    def score(self, dialog: Dialog) -> float:
        """
        Compute flow perplexity-like score (exp of negative average log probability).

        :param dialog: Dialogue to score.
        :type dialog: Dialog
        :return: Perplexity value or None if insufficient transitions.
        :rtype: Optional[float]
        """
        sum_log_p_known, n_turns_known, sum_log_p, n_turns = self.compute_dialog_log_likelihood(dialog)
        if n_turns <= 1:
            dialog_path = getattr(dialog, "_path", None)
            if dialog_path:
                logger.warning(f"Dialog at '{dialog_path}' has no known transitions or valid turns. Skipping.")
            else:
                logger.warning(f"Dialog (id={getattr(dialog, 'id', 'unknown')}) has no known transitions "
                               "or valid turns. Skipping.")
            return None
        if self.use_only_known_edges:
            return exp(-sum_log_p_known / n_turns_known)
        else:
            return exp(-sum_log_p / n_turns)


class DialogFlowScore(BaseDialogFlowScore):
    """
    Compute flow likelihood score of a dialogue against reference dialogues.

    Given a collection of reference dialogues, it first builds the dialogue flow graph that
    represent them. Then, given a candidate dialogue, it computes a flow likelihood score
    based on the geometric mean of edge probabilities (i.e. "how well the dialogue fits on the reference graph").

    Example:

        .. code-block:: python

            from sdialog.evaluation import DialogFlowScore

            flow_score = DialogFlowScore(reference_dialogs)

            print(flow_score(candidate_dialog))

    :param reference_dialogues: List of reference dialogues or file path.
    :type reference_dialogues: Union[str, List[Dialog]]
    :param ai_speaker: Restrict scoring to AI/system turns if provided.
    :type ai_speaker: Optional[str]
    :param k_neighbors: Neighbor count for embedding lookup.
    :type k_neighbors: int
    :param use_softmax: Whether to weight neighbors via softmax.
    :type use_softmax: bool
    :param use_only_ai_speaker: If True, only AI turns are used to build the graph and compute the scores.
    :type use_only_ai_speaker: bool
    :param use_only_known_edges: If True, only known edges contribute.
    :type use_only_known_edges: bool
    :param name: Custom score name.
    :type name: Optional[str]
    :param verbose: Verbosity flag.
    :type verbose: bool
    :param graph: Pre-built graph (optional).
    :type graph: Any
    :param nodes: Pre-built node metadata (optional).
    :type nodes: dict
    :param d2f_kwargs: Extra kwargs to dialog2graph.
    :type d2f_kwargs: dict
    """
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 k_neighbors: int = 64,
                 use_softmax: bool = True,
                 use_only_ai_speaker: bool = False,
                 use_only_known_edges: bool = False,
                 name: str = None,
                 verbose: bool = False,
                 graph=None,
                 nodes=None,
                 **d2f_kwargs):
        """Initialize flow likelihood score evaluator."""
        self.use_only_known_edges = use_only_known_edges
        if name is None:
            name = "dfs" + ("" if use_softmax else "-hard") + ("-ai" if ai_speaker else "")
            name += "-only-known" if use_only_known_edges else ""
        super().__init__(
            reference_dialogues,
            ai_speaker=ai_speaker,
            k_neighbors=k_neighbors,
            use_softmax=use_softmax,
            use_only_ai_speaker=use_only_ai_speaker,
            name=name,
            graph=graph,
            nodes=nodes,
            verbose=verbose,
            **d2f_kwargs
        )

    @CacheDialogScore.cache
    def score(self, dialog: Dialog) -> float:
        """
        Compute geometric mean transition likelihood.

        :param dialog: Dialogue to score.
        :type dialog: Dialog
        :return: Score value or None if insufficient transitions.
        :rtype: Optional[float]
        """
        sum_log_p_known, n_turns_known, sum_log_p, n_turns = self.compute_dialog_log_likelihood(dialog)
        if n_turns <= 1:
            dialog_path = getattr(dialog, '_path', None)
            if dialog_path:
                logger.warning(f"Dialog at '{dialog_path}' has no known transitions or valid turns. Skipping.")
            else:
                logger.warning(f"Dialog (id={getattr(dialog, 'id', 'unknown')}) has no known transitions "
                               "or valid turns. Skipping.")
            return None
        if self.use_only_known_edges:
            return pow(exp(sum_log_p_known), 1 / n_turns_known)
        else:
            return pow(exp(sum_log_p), 1 / n_turns)


class LLMJudgeYesNo(BaseDialogScore, BaseLLMJudge):
    """LLM judge for classifying a dialogue as "yes or no" (boolean) output and reason.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeYesNo

            magic_judge = LLMJudgeYesNo("Is this dialogue magical?", reason=True)

            result = magic_judge.judge(dialog)

            print(result.positive)
            print(result.reason)

    :param prompt_template: Jinja2 template for judging prompt.
    :type prompt_template: str
    :param reason: Whether to request reason field.
    :type reason: bool
    :param model: Model instance or model name.
    :type model: Optional[Union[BaseLanguageModel, str]]
    :param llm_kwargs: Extra LLM initialization kwargs.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 prompt_template: str,
                 reason: bool = False,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initialize yes/no LLM judge."""
        with open(config["prompts"]["evaluation"]["llm_judge"]["yesno"]["base"], encoding="utf-8") as f:
            prompt_template = f.read().replace("{{ prompt_template }}", prompt_template)

        BaseDialogScore.__init__(self,
                                 name=upper_camel_to_dash(self.__class__.__name__))
        BaseLLMJudge.__init__(self,
                              model=model,
                              output_format=LLMJudgeYesNoOutput,
                              prompt_template=prompt_template,
                              **llm_kwargs)
        self.reason = reason

    def judge(self,
              dialogs: Union[Dialog, List[Dialog]],
              reason: bool = None,
              **template_kwargs) -> Union[LLMJudgeYesNoOutput, int]:
        """
        Run judgment over one or multiple dialogues.

        :param dialogs: A single Dialog or list of Dialogs.
        :type dialogs: Union[Dialog, List[Dialog]]
        :param reason: Override reason flag (falls back to self.reason).
        :type reason: Optional[bool]
        :param template_kwargs: Extra template kwargs.
        :type template_kwargs: dict
        :return: Structured yes/no output model.
        :rtype: LLMJudgeYesNoOutput
        """
        if isinstance(dialogs, Dialog):
            dialogs = [dialogs]  # Wrap single dialog in a list

        # Prepare default template variables
        render_kwargs = {
            'dialogs': dialogs,
            'dialog': dialogs[0],
            'reason': reason if reason is not None else self.reason
        }

        # Merge with any additional template kwargs
        render_kwargs.update(template_kwargs)

        prompt = self.prompt_template.render(**render_kwargs)
        output = BaseLLMJudge.__call__(self, prompt)
        output = self.output_format.model_validate(output)

        if isinstance(output.positive, list) and not output.positive:
            if len(dialogs) > 1:
                output.positive = [False] * len(dialogs)
            else:
                output.positive = False

        return output

    @CacheDialogScore.cache
    def score(self, dialog: Dialog) -> int:
        """
        Computes the score for the provided dialog, 1 if dialogues is judged as real, 0 otherwise.

        :param dialog: The dialog to score.
        :return: An int representing the score of the dialog.
        """
        output = self.judge(dialog)
        try:
            return int(output.positive[0]) if isinstance(output.positive, list) else int(output.positive)
        except TypeError:
            raise ValueError(f"LLMJudgeYesNo output '{output.positive}' is not a boolean or list of booleans, "
                             f"cannot convert to integer score.")


class LLMJudgeScore(BaseDialogScore, BaseLLMJudge):
    """LLM judge for scoring a dialogue with a numerical score and optional reason.

    Example 1:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeScore

            magic_judge = LLMJudgeScore("From 1 to 5, how magical is this dialogue?", reason=True)

            result = magic_judge.judge(dialog)

            print(result.score)
            print(result.reason)

    Example 2:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeScore

            # You can use the `min_score`, `max_score`, `score_type` and/or `reason` parameters
            # as variables in your prompt template.
            prompt = (
                "On a scale from {{ min_score }} to {{ max_score }}, "
                "how magical is this dialogue?"
                "Provide a {{ score_type }} score."
            )
            magic_judge = LLMJudgeScore(prompt,
                                        min_score=1,
                                        max_score=10,
                                        score_type=int)
            result = magic_judge.judge(dialog)
            print(result.score)
            print(result.reason)

    :param prompt_template: Jinja2 template text.
    :type prompt_template: str
    :param min_score: Minimum allowed score.
    :type min_score: float
    :param max_score: Maximum allowed score.
    :type max_score: float
    :param score_type: int or float score type.
    :type score_type: type
    :param reason: Whether to request reason field.
    :type reason: bool
    :param model: Model instance or model name.
    :type model: Optional[Union[BaseLanguageModel, str]]
    :param llm_kwargs: Extra LLM kwargs.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 prompt_template: str,
                 min_score: float = 1,
                 max_score: float = 5,
                 score_type: type = int,
                 reason: bool = False,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """
        Initialize numeric score judge.

        :raises ValueError: If score_type invalid.
        """
        if isinstance(score_type, str):
            score_type = {"int": int, "float": float}.get(score_type.lower(), score_type)

        if score_type not in [int, float]:
            raise ValueError(f"Invalid score_type: {score_type}. Must be int or float.")
        elif score_type is float:
            logger.warning(
                "Using float as `score_type` may cause boundary issues (min_score, max_score). "
                "Consider using int for discrete scales."
            )

        # Build the model dynamically with the user provided score_type and range (min_score, max_score)
        LLMJudgeScoreRangeOutput = create_model(
            "LLMJudgeScoreRangeOutput",
            __base__=LLMJudgeScoreOutput,
            score=(score_type, Field(ge=min_score, le=max_score)),
        )

        with open(config["prompts"]["evaluation"]["llm_judge"]["score"]["base"], encoding="utf-8") as f:
            prompt_template = f.read().replace("{{ prompt_template }}", prompt_template)

        BaseDialogScore.__init__(self,
                                 name=upper_camel_to_dash(self.__class__.__name__))
        BaseLLMJudge.__init__(self,
                              model=model,
                              output_format=LLMJudgeScoreRangeOutput,
                              prompt_template=prompt_template,
                              **llm_kwargs)

        self.score_type = score_type
        self.min_score = min_score
        self.max_score = max_score
        self.reason = reason

    def judge(self,
              dialogs: Union[Dialog, List[Dialog]],
              reason: bool = None,
              **template_kwargs) -> LLMJudgeScoreOutput:
        """
        Produce a numeric judgment for one or more dialogues.

        :param dialogs: Dialogue or list of dialogues.
        :type dialogs: Union[Dialog, List[Dialog]]
        :param reason: Override reason flag.
        :type reason: Optional[bool]
        :param template_kwargs: Extra template kwargs.
        :type template_kwargs: dict
        :return: Structured output containing the score and an optional reason.
        :rtype: LLMJudgeScoreOutput
        """

        if isinstance(dialogs, Dialog):
            dialogs = [dialogs]  # Wrap single dialog in a list

        # Prepare default template variables
        render_kwargs = {
            'dialogs': dialogs,
            'dialog': dialogs[0],
            'min_score': self.min_score,
            'max_score': self.max_score,
            'reason': reason if reason is not None else self.reason
        }

        # Merge with any additional template kwargs
        render_kwargs.update(template_kwargs)

        prompt = self.prompt_template.render(**render_kwargs)
        output = self.output_format.model_validate(BaseLLMJudge.__call__(self, prompt))

        return output

    @CacheDialogScore.cache
    def score(self, dialog: Dialog, **template_kwargs) -> Union[float, int]:
        """
        Return the numeric score.

        :param dialog: Dialogue to score.
        :type dialog: Dialog
        :param template_kwargs: Extra template kwargs.
        :type template_kwargs: dict
        :return: Score value.
        :rtype: Union[int, float]
        :raises ValueError: If model output malformed.
        """
        output = self.judge(dialog, **template_kwargs)
        try:
            score = output.score[0] if isinstance(output.score, list) else output.score
            # Clamp score to [min_score, max_score] if out of bounds
            if score < self.min_score or score > self.max_score:
                old_score = score
                score = max(self.min_score, min(score, self.max_score))
                logger.warning(
                    f"Generated score {old_score} is out of bounds [{self.min_score}, {self.max_score}]. "
                    f"Clamping to valid range: {score}."
                )
            return score
        except TypeError:
            raise ValueError(f"LLMJudgeScore output ({output.score}) is not a {self.score_type} or list of booleans, "
                             "cannot convert to integer score.")


class LLMJudgeRealDialog(LLMJudgeYesNo):
    """
    LLM judge for classifying a dialogue as real (human) or synthetic (machine-generated), with boolean output and reason.
    Returns an instance of LLMJudgeYesNoOutput.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeRealDialog

            judge_real = LLMJudgeRealDialog(reason=True)

            result = judge_real.judge(dialog)

            print("Real?", result.positive)
            print("Reason:", result.reason)

    :param reason: Whether to request reason.
    :type reason: bool
    :param model: Model instance or name.
    :type model: Optional[Union[BaseLanguageModel, str]]
    :param llm_kwargs: Additional LLM kwargs.
    :type llm_kwargs: dict
    """  # noqa: E501
    def __init__(self,
                 reason: bool = False,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initialize real vs synthetic judge (boolean)."""
        with open(config["prompts"]["evaluation"]["llm_judge"]["yesno"]["real_dialog"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         reason=reason,
                         **llm_kwargs)


class LLMJudgeRealDialogLikertScore(LLMJudgeScore):
    """
    LLM judge for evaluating whether a dialogue appears real (human) or synthetic (machine-generated),
    providing a Likert score between 1 (definitely synthetic) and 5 (definitely real), with optional reason.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeRealDialogLikertScore

            judge_real = LLMJudgeRealDialogLikertScore(reason=True)

            result = judge_real.judge(dialog)
            # score = judge_real(dialog)

            print("Likert Score:", result.score)  # score from 1 to 5
            print("Reason:", result.reason)

    :param reason: Request reason flag.
    :type reason: bool
    :param model: Model instance or name.
    :type model: Optional[Union[BaseLanguageModel, str]]
    :param llm_kwargs: Extra LLM kwargs.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 reason: bool = False,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initialize Likert realism scorer (1-5)."""
        with open(config["prompts"]["evaluation"]["llm_judge"]["score"]["real_dialog_likert"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         score_type=int,
                         min_score=1,
                         max_score=5,
                         reason=reason,
                         **llm_kwargs)


class LLMJudgeRealDialogScore(LLMJudgeScore):
    """
    LLM judge for evaluating how "real" (human-like) or "synthetic" a dialogue appears
    on a configurable numeric range.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeRealDialogScore

            judge_real = LLMJudgeRealDialogScore(min_score=0, max_score=10, reason=True)

            result = judge_real.judge(dialog)
            # score = judge_real(dialog)

            print("Score:", result.score)  # score from 0 to 10
            print("Reason:", result.reason)

    :param min_score: Minimum realism score.
    :type min_score: int
    :param max_score: Maximum realism score.
    :type max_score: int
    :param reason: Request reason flag.
    :type reason: bool
    :param model: Model instance or name.
    :type model: Optional[Union[BaseLanguageModel, str]]
    :param llm_kwargs: Extra LLM kwargs.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 min_score: int = 0,
                 max_score: int = 10,
                 reason: bool = False,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initialize realism score judge (custom numeric range)."""
        with open(config["prompts"]["evaluation"]["llm_judge"]["score"]["real_dialog"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         score_type=int,
                         min_score=min_score,
                         max_score=max_score,
                         reason=reason,
                         **llm_kwargs)


class LLMJudgeRefusal(LLMJudgeYesNo):
    """
    LLM judge for evaluating if a dialogue contains a refusal response.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeRefusal

            judge_refusal = LLMJudgeRefusal(reason=True)

            result = judge_refusal.judge(dialog)

            print("Refused?", result.positive)
            print("Reason:", result.reason)

    :param reason: Request reason flag.
    :type reason: bool
    :param model: Model instance or name.
    :type model: Optional[Union[BaseLanguageModel, str]]
    :param llm_kwargs: Extra LLM kwargs.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 reason: bool = False,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initialize refusal detector."""
        with open(config["prompts"]["evaluation"]["llm_judge"]["yesno"]["refusal"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         reason=reason,
                         **llm_kwargs)


class LLMJudgePersonaAttributes(LLMJudgeYesNo):
    """LLM judge for evaluating if a speaker follows the persona attributes in a dialogue.

    Example:

        .. code-block:: python

            from sdialog.personas import Doctor
            from sdialog.evaluation import LLMJudgePersonaAttributes

            reference_persona = Doctor(name="Dr. Smith", specialty="cardiology")
            judge_persona = LLMJudgePersonaAttributes(persona=reference_persona,
                                                      speaker="Doctor",
                                                      reason=True)
            result = judge_persona.judge(dialog)

            print("Matches persona?", result.positive)
            print("Reason:", result.reason)

    :param persona: Persona definition object.
    :type persona: BasePersona
    :param speaker: Target speaker in dialogue.
    :type speaker: str
    :param reason: Request reason flag.
    :type reason: bool
    :param model: Model instance or name.
    :type model: Optional[Union[BaseLanguageModel, str]]
    :param llm_kwargs: Additional LLM kwargs.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 persona: BasePersona,
                 speaker: str,
                 reason: bool = False,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initialize persona adherence judge."""
        with open(config["prompts"]["evaluation"]["llm_judge"]["yesno"]["persona_attributes"], encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_template = prompt_template.render(persona=persona, speaker=speaker)

        super().__init__(prompt_template,
                         model=model,
                         reason=reason,
                         **llm_kwargs)


class SentenceTransformerDialogEmbedder(BaseDialogEmbedder):
    """
    Dialog embedder using SentenceTransformer.
    Can embed a dialog as the mean of turn embeddings or as a single embedding of the whole dialog text.

    Example:

        .. code-block:: python

            from sdialog.evaluation import SentenceTransformerDialogEmbedder

            dialog_embedder = SentenceTransformerDialogEmbedder(model_name="sentence-transformers/LaBSE")

            emb = dialog_embedder(dialog)

            print(emb.shape)

    :param model_name: SentenceTransformer model name.
    :type model_name: str
    :param mean: If True average per-turn embeddings; else encode concatenated text.
    :type mean: bool
    :param ai_speaker: If set, restrict embedding to AI/system turns only.
    :type ai_speaker: Optional[str]
    :param name: Optional custom embedder name.
    :type name: Optional[str]
    :param verbose: Show progress bars for encoding.
    :type verbose: bool
    """
    def __init__(self,
                 model_name: str = "sentence-transformers/LaBSE",
                 mean: bool = True,
                 ai_speaker: str = None,
                 name: str = None,
                 verbose: bool = False):
        """Initialize dialog embedder."""

        mode_str = "mean-" if mean else ""
        super().__init__(name=name or f"{mode_str}{model_name.split('/')[-1]}" + ("-ai" if ai_speaker else ""))
        self.model = SentenceTransformer(model_name)
        self.mean = mean
        self.verbose = verbose
        self.ai_speaker = ai_speaker

    def embed(self, dialog: Dialog) -> np.ndarray:
        """
        Generate embedding for a dialog.

        :param dialog: Dialog instance.
        :type dialog: Dialog
        :return: Embedding vector.
        :rtype: np.ndarray
        """
        if self.mean:
            if self.ai_speaker:
                texts = [turn.text for turn in dialog
                         if hasattr(turn, "text") and turn.speaker.lower() == self.ai_speaker.lower()]
            else:
                texts = [turn.text for turn in dialog if hasattr(turn, "text")]
            if not texts:
                return np.zeros(self.model.get_sentence_embedding_dimension())
            embs = self.model.encode(texts, show_progress_bar=self.verbose)
            return np.mean(embs, axis=0)
        else:
            if self.ai_speaker:
                dialog_text = "\n".join([turn.text for turn in dialog
                                         if turn.speaker.lower() == self.ai_speaker.lower()])
            else:
                dialog_text = "\n".join([turn.text for turn in dialog])
            if not dialog_text:
                return np.zeros(self.model.get_sentence_embedding_dimension())
            emb = self.model.encode([dialog_text], show_progress_bar=self.verbose)[0]
            return emb


class ReferenceCentroidEmbeddingEvaluator(BaseDatasetEmbeddingEvaluator):
    """
    Evaluator comparing candidate centroid to a reference centroid via cosine similarity.

    Example:

        .. code-block:: python

            from sdialog.evaluation import SentenceTransformerDialogEmbedder
            from sdialog.evaluation import ReferenceCentroidEmbeddingEvaluator

            dialog_embedder = SentenceTransformerDialogEmbedder()

            evaluator = ReferenceCentroidEmbeddingEvaluator(dialog_embedder, reference_dialogs)

            # How far are the candidate dialogs from the reference dialogues? (centroid-wise)
            print(evaluator(candidate_dialogs))

    :param dialog_embedder: Dialog embedding component.
    :type dialog_embedder: BaseDialogEmbedder
    :param reference_dialogues: List of reference Dialog objects or path.
    :type reference_dialogues: Union[str, List[Dialog]]
    :param name: Optional evaluator name.
    :type name: Optional[str]
    :param enable_plotting: Store embeddings for plotting if True.
    :type enable_plotting: bool
    :param verbose: Verbosity flag.
    :type verbose: bool
    """
    def __init__(self,
                 dialog_embedder: BaseDialogEmbedder,
                 reference_dialogues: Union[str, List[Dialog]],
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 plot_title: str = None,
                 plot_xlabel: str = None,
                 plot_ylabel: str = None):
        """Initialize centroid similarity evaluator."""
        # Compute reference centroid
        name = name or f"centroid-similarity-{dialog_embedder.name}"
        super().__init__(dialog_embedder, name=name, enable_plotting=enable_plotting, verbose=verbose,
                         plot_title=plot_title, plot_xlabel=plot_xlabel, plot_ylabel=plot_ylabel)

        if isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        reference_embs = np.array([self.dialog_embedder(dialog)
                                   for dialog in tqdm(reference_dialogues,
                                                      desc="Computing reference embeddings",
                                                      leave=verbose)])
        self.reference_embs = reference_embs if enable_plotting else None
        self.reference_centroid = np.mean(reference_embs, axis=0)

    def __plot__(self, dialog_embs: Dict[str, np.ndarray]):
        """
        Plot t-SNE projection of embeddings and centroids.

        :param dialog_embs: Mapping dataset name -> embedding matrix.
        :type dialog_embs: Dict[str, np.ndarray]
        :return: None
        :rtype: None
        """
        # Concatenate all embeddings and keep track of dataset labels
        all_embs = [self.reference_centroid.reshape(1, -1)]
        all_labels = ["centroid-reference"]
        all_embs.append(self.reference_embs)
        all_labels.extend(["reference"] * len(self.reference_embs))
        for dataset_name, embs in dialog_embs.items():
            all_embs.append(embs)
            all_labels.extend([dataset_name] * len(embs))
            all_embs.append(np.mean(embs, axis=0).reshape(1, -1))
            all_labels.append("centroid-" + dataset_name)
        all_embs = np.vstack(all_embs)
        all_labels = np.array(all_labels)

        # Compute t-SNE (2D)
        logger.info("Computing t-SNE for embeddings...")
        tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30, metric="cosine")
        tsne_embs = tsne.fit_transform(all_embs)

        # Plot
        unique_labels = [label for label in np.unique(all_labels).tolist() if "centroid-" not in label]
        colors = plt.cm.tab10.colors if len(unique_labels) <= 10 else plt.cm.tab20.colors
        for i, label in enumerate(unique_labels):
            idx = all_labels == label
            plt.scatter(tsne_embs[idx, 0], tsne_embs[idx, 1],
                        label=label,
                        alpha=0.15 if label == "reference" else 0.7,
                        color="black" if label == "reference" else colors[i % len(colors)])
        for label in ["reference"] + list(dialog_embs.keys()):
            idx = all_labels == f"centroid-{label}"
            plt.scatter(tsne_embs[idx, 0], tsne_embs[idx, 1],
                        label="reference centroid" if label == "reference" else None,
                        linewidths=3 if label == "reference" else 2,
                        alpha=1,  # if label == "reference" else 0.7,
                        color="black" if label == "reference" else colors[unique_labels.index(label) % len(colors)],
                        s=100,
                        marker="x")

        plt.xlabel(self.plot_xlabel if self.plot_xlabel else "t-SNE Component 1")
        plt.ylabel(self.plot_ylabel if self.plot_ylabel else "t-SNE Component 2")
        plt.title(self.plot_title
                  if self.plot_title
                  else f"Dialog Embeddings with Centroids\n({self.dialog_embedder.name})")
        plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        plt.tight_layout(pad=0.5)

    def __eval__(self, dialog_embs: List[np.ndarray]) -> float:
        """
        Compute cosine similarity between candidate centroid and reference centroid.

        :param dialog_embs: Embeddings to evaluate.
        :type dialog_embs: List[np.ndarray]
        :return: Cosine similarity (−1 to 1).
        :rtype: float
        """
        if isinstance(dialog_embs, list):
            dialog_embs = np.array(dialog_embs)
        if dialog_embs.ndim == 1:
            dialog_embs = dialog_embs.reshape(1, -1)
        centroid = np.mean(dialog_embs, axis=0)
        # Cosine similarity
        dot = np.dot(self.reference_centroid, centroid)
        norm_ref = np.linalg.norm(self.reference_centroid)
        norm_cand = np.linalg.norm(centroid)
        if norm_ref == 0 or norm_cand == 0:
            return 0.0
        return float(dot / (norm_ref * norm_cand))


class KDEDistanceEvaluator(BaseDatasetScoreEvaluator):
    """
    Evaluate distribution divergence between reference and candidate dialog scores using KDE.

    Example:

        .. code-block:: python

            from sdialog.evaluation import KDEDistanceEvaluator, GunningFogScore

            # Any dialog score can be used, let's use `GunningFogScore` as an example
            kde_eval = KDEDistanceEvaluator(dialog_score=GunningFogScore(),
                                            reference_dialogues=reference_dialogs)

            print("KL divergence:", kde_eval(candidate_dialogs))

    :param dialog_score: Per-dialog scoring object.
    :type dialog_score: BaseDialogScore
    :param reference_dialogues: Reference Dialog list or path (optional if score object has attribute).
    :type reference_dialogues: Optional[Union[str, List[Dialog]]]
    :param metric: Divergence metric: "kl", "cs", or "all".
    :type metric: str
    :param kde_bw: Bandwidth override for KDE.
    :type kde_bw: Optional[float]
    :param name: Evaluator name.
    :type name: Optional[str]
    :param enable_plotting: Keep distributions for plotting.
    :type enable_plotting: bool
    :param verbose: Verbosity flag.
    :type verbose: bool
    :param evaluator_kwargs: Extra kwargs to parent initializer.
    :type evaluator_kwargs: dict
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 reference_dialogues: Union[str, List[Dialog]] = None,
                 metric: str = "kl",
                 kde_bw: float = None,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 plot_title: str = None,
                 plot_xlabel: str = None,
                 plot_ylabel: str = None,
                 **evaluator_kwargs):
        """Initialize KDE-based divergence evaluator."""
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose,
                         plot_title=plot_title, plot_xlabel=plot_xlabel, plot_ylabel=plot_ylabel, **evaluator_kwargs)

        if reference_dialogues is None:
            if hasattr(dialog_score, "reference_dialogues"):
                reference_dialogues = dialog_score.reference_dialogues
            else:
                raise ValueError("Reference dialogues must be provided or "
                                 "the dialog_score must have a reference_dialogues attribute.")
        elif isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        elif not isinstance(reference_dialogues, list):
            raise ValueError("Reference dialogues must be provided as a list of Dialog objects or a file path.")

        self.metric = metric
        self.kde_bw = kde_bw
        self.reference_scores = [self.dialog_score(dialogue)
                                 for dialogue in tqdm(reference_dialogues,
                                                      desc=f"Computing reference {self.name} scores",
                                                      leave=verbose)]
        self.reference_scores = np.array([s for s in self.reference_scores if s is not None])

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None, zoom: bool = False):
        """
        Plot KDE curves of reference and candidate score distributions.

        :param dialog_scores: Mapping dataset name -> scores array.
        :type dialog_scores: Dict[str, np.ndarray]
        :param plot: Matplotlib Axes or pyplot module.
        :type plot: Optional[plt.Axes]
        :return: None
        :rtype: None
        """
        colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_idx = 0

        if "reference" not in dialog_scores and self.reference_scores is not None:
            pd.Series(self.reference_scores, name="Reference").plot.kde(bw_method=self.kde_bw,
                                                                        lw=2,
                                                                        color=colors[0],
                                                                        linestyle='-')
            color_idx = 1
        for dataset_name, scores in dialog_scores.items():
            try:
                pd.Series(scores, name=dataset_name).plot.kde(bw_method=self.kde_bw,
                                                              lw=1.8,
                                                              color=colors[color_idx % len(colors)])
                color_idx += 1
            except ValueError as e:
                logger.error(f"Error plotting KDE for {dataset_name}: {e}")

        if zoom:
            # Percentile-based zoom
            all_scores = []
            if self.reference_scores is not None:
                all_scores.append(self.reference_scores)
            for scores in dialog_scores.values():
                all_scores.append(scores)

            if all_scores:
                all_scores = np.concatenate(all_scores)
                low, high = np.percentile(all_scores, [2, 98])  # tweak if needed
                pad = 0.05 * (high - low)
                plt.gca().set_xlim(low - pad, high + pad)

        plot.xlabel(self.plot_xlabel if self.plot_xlabel else self.dialog_score.name)
        plot.ylabel(self.plot_ylabel if self.plot_ylabel else "Density")
        plot.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        plot.title(self.plot_title if self.plot_title else f"Kernel Density Estimation: {self.dialog_score.name}")
        plot.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        plt.tight_layout(pad=0.5)

    def __eval__(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        """
        Compute divergence(s) between reference and candidate score distributions.

        :param dialog_scores: Candidate score list/array.
        :type dialog_scores: List[Union[float, int]]
        :return: Single divergence value or dict if metric == "all".
        :rtype: Union[float, dict]
        """
        if self.metric == "kl":
            result = _kl_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw)
        elif self.metric == "cs":
            result = _cs_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw)
        else:
            result = {
                "cs": _cs_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw),
                "kl": _kl_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw)
            }
        return result


class FrechetDistanceEvaluator(BaseDatasetScoreEvaluator):
    """
    Evaluate Frechet distance between Gaussian fits of reference and candidate score distributions.

    Example:

        .. code-block:: python

            from sdialog.evaluation import FrechetDistanceEvaluator, ConversationalFeatures

            # Any dialog score can be used, let's use `ConversationalFeatures` as an example
            turn_length = ConversationalFeatures(feature="mean-turn-length")
            fd_eval = FrechetDistanceEvaluator(dialog_score=turn_length,
                                               reference_dialogues=reference_dialogs)

            print("Frechet distance:", fd_eval(candidate_dialogs))

    :param dialog_score: Per-dialog scoring object.
    :type dialog_score: BaseDialogScore
    :param reference_dialogues: List or path of reference dialogues.
    :type reference_dialogues: Optional[Union[str, List[Dialog]]]
    :param name: Evaluator name.
    :type name: Optional[str]
    :param enable_plotting: Retained for API parity (not used directly here).
    :type enable_plotting: bool
    :param verbose: Verbosity flag.
    :type verbose: bool
    :param evaluator_kwargs: Extra parent kwargs.
    :type evaluator_kwargs: dict
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 reference_dialogues: Union[str, List[Dialog]] = None,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 plot_title: str = None,
                 plot_xlabel: str = None,
                 plot_ylabel: str = None,
                 **evaluator_kwargs):
        """Evaluate Frechet distance between Gaussian fits of reference and candidate score distributions."""
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose,
                         plot_title=plot_title, plot_xlabel=plot_xlabel, plot_ylabel=plot_ylabel, **evaluator_kwargs)

        if reference_dialogues is None:
            if hasattr(dialog_score, "reference_dialogues"):
                reference_dialogues = dialog_score.reference_dialogues
            else:
                raise ValueError("Reference dialogues must be provided or "
                                 "the dialog_score must have a reference_dialogues attribute.")

        reference_scores = np.array([self.dialog_score(dialogue)
                                     for dialogue in tqdm(reference_dialogues,
                                                          desc=f"Computing reference {self.name} scores",
                                                          leave=verbose)])
        self.reference_norm_dist = norm(loc=np.mean(reference_scores), scale=np.std(reference_scores))

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        """
        Plot fitted normal distributions for each dataset.

        :param dialog_scores: Mapping dataset name -> score array.
        :type dialog_scores: Dict[str, np.ndarray]
        :param plot: Matplotlib Axes or pyplot module.
        :type plot: Optional[plt.Axes]
        :return: None
        :rtype: None
        """
        colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_idx = 0

        if "reference" not in dialog_scores and self.reference_norm_dist is not None:
            x = np.linspace(self.reference_norm_dist.ppf(0.001), self.reference_norm_dist.ppf(0.999), 100)
            plot.plot(x, self.reference_norm_dist.pdf(x), color=colors[0], lw=2, label="Reference", linestyle='-')
            color_idx = 1
        for dataset_name, scores in dialog_scores.items():
            x = np.linspace(np.min(scores), np.max(scores), 100)
            plot.plot(x,
                      norm.pdf(x, loc=np.mean(scores), scale=np.std(scores)),
                      label=dataset_name, lw=1.8, color=colors[color_idx % len(colors)])
            color_idx += 1
        plot.xlabel(self.plot_xlabel if self.plot_xlabel else self.dialog_score.name)
        plot.ylabel(self.plot_ylabel if self.plot_ylabel else "Probability Density")
        plot.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        plot.title(self.plot_title if self.plot_title else f"Gaussian Distributions: {self.dialog_score.name}")
        plot.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        plt.tight_layout(pad=0.5)

    def __eval__(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        """
        Compute Frechet distance between reference normal and candidate normal distributions.

        :param dialog_scores: Candidate score list/array.
        :type dialog_scores: List[Union[float, int]]
        :return: Frechet distance (>= 0).
        :rtype: float
        """
        # Compute the Frechet distance between two 1D Gaussians: sqrt((mu_src-mu_tgt)^2 + (sigma_src-sigma_tgt)^2)
        if not isinstance(dialog_scores, np.ndarray):
            dialog_scores = np.array(dialog_scores)
        mu_src, sigma_src = self.reference_norm_dist.mean(), self.reference_norm_dist.std()
        mu_tgt, sigma_tgt = np.mean(dialog_scores), np.std(dialog_scores)
        return np.sqrt((mu_src - mu_tgt) ** 2 + (sigma_src - sigma_tgt) ** 2)


class FrechetBERTDistanceEvaluator(BaseDatasetEvaluator):
    """
    Frechet distance evaluator based on BERT sentence-pair embeddings.
    See: https://aclanthology.org/2021.findings-acl.193/

    Example:

        .. code-block:: python

            from sdialog.evaluation import FrechetBERTDistanceEvaluator

            fb_distance = FrechetBERTDistanceEvaluator(reference_dialogs)

            print(fb_distance(candidate_dialogs))

    :param reference_dialogues: Reference dialogues (list or path).
    :type reference_dialogues: Union[str, List[Dialog]]
    :param ai_speaker: If set, restrict to AI response pairs.
    :type ai_speaker: Optional[str]
    :param name: Evaluator name.
    :type name: Optional[str]
    :param model_name: Underlying transformer model.
    :type model_name: str
    :param batch_size: Batch size for encoding.
    :type batch_size: int
    :param device: Torch device override.
    :type device: Optional[str]
    :param enable_plotting: Store embeddings for later plotting.
    :type enable_plotting: bool
    :param verbose: Verbosity flag.
    :type verbose: bool
    """
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 name: str = None,
                 model_name: str = "roberta-base",
                 batch_size: int = 128,
                 device: str = None,
                 enable_plotting: bool = False,
                 verbose: bool = False):
        """Initialize Frechet BERT evaluator."""
        self.reference_embs = None
        self.datasets_embs = {}
        self.enable_plotting = enable_plotting
        self.verbose = verbose
        self.ai_speaker = ai_speaker
        self.name = name or "frechet-bert-distance" + ("-ai" if ai_speaker else "")
        self.batch_size = batch_size
        self.model = SentencePairTransformer(model_name=model_name,
                                             device=device,
                                             verbose=verbose)

        if isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        if not reference_dialogues or not isinstance(reference_dialogues, list):
            raise ValueError("Reference dialogues must be provided as a list of Dialog objects or a file path.")

        self.reference_mu, self.reference_sigma = self._get_multidim_gaussian_mu_sigma(reference_dialogues)

    def _get_multidim_gaussian_mu_sigma(self,
                                        dialogs: List[Dialog],
                                        dataset_name: str = "reference") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of sentence-pair embeddings for a set of dialogs.

        :param dialogs: List of Dialog objects.
        :type dialogs: List[Dialog]
        :param dataset_name: Dataset label for logging / storage.
        :type dataset_name: str
        :return: (mean vector, covariance matrix).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        utts, utts_next = dialogs_to_utt_pairs(dialogs, self.ai_speaker)

        embs = self.model.encode(utts, utts_next,
                                 batch_size=self.batch_size,
                                 progress_bar_desc=f"Computing embeddings for FrechetBERT on {dataset_name}")

        if self.enable_plotting and dataset_name:
            if dataset_name == "reference":
                self.reference_embs = embs
            else:
                self.datasets_embs[dataset_name] = embs

        mu = np.mean(embs, axis=0)
        sigma = np.cov(embs, rowvar=False)
        return mu, sigma

    def __call__(self, dialogues: Union[str, List[Dialog]], dataset_name: str = "candidate") -> float:
        """
        Compute Frechet distance between reference embedding distribution and candidate.

        :param dialogues: Candidate dialogues (list or path).
        :type dialogues: Union[str, List[Dialog]]
        :param dataset_name: Label for candidate dataset.
        :type dataset_name: str
        :return: Frechet distance (>= 0).
        :rtype: float
        """
        mu_src, sigma_src = np.atleast_1d(self.reference_mu), np.atleast_2d(self.reference_sigma)
        mu_tgt, sigma_tgt = self._get_multidim_gaussian_mu_sigma(dialogues, dataset_name=dataset_name)

        mu_tgt = np.atleast_1d(mu_tgt)
        sigma_tgt = np.atleast_2d(sigma_tgt)

        diff = mu_src - mu_tgt

        covmean, _ = linalg.sqrtm(sigma_src.dot(sigma_tgt), disp=False)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.imag(covmean), 0, atol=1e-6):
                logger.warning("linalg.sqrtm returned complex values; taking real part of result.")
            covmean = np.real(covmean)

        tr_covmean = np.trace(covmean)
        fid = float(diff.dot(diff) + np.trace(sigma_src) + np.trace(sigma_tgt) - 2 * tr_covmean)
        return max(fid, 0.0)

    def plot(self, show: bool = True, save_path: str = None):
        """
        Plot t-SNE projection of sentence-pair embeddings for reference and candidates.

        :param show: Display the figure.
        :type show: bool
        :param save_path: Path to save figure (if provided).
        :type save_path: Optional[str]
        :return: None
        :rtype: None
        """
        if not self.enable_plotting or not self.datasets_embs:
            return
        # Publication-ready figure size (single column: 3.5", double column: 7")
        plt.figure(figsize=(7, 5))
        # Concatenate all embeddings and keep track of dataset labels
        all_embs = [self.reference_embs]
        all_labels = ["reference"] * len(self.reference_embs)
        for dataset_name, embs in self.datasets_embs.items():
            all_embs.append(embs)
            all_labels.extend([dataset_name] * len(embs))
        all_embs = np.vstack(all_embs)
        all_labels = np.array(all_labels)

        # Compute t-SNE (2D)
        logger.info("Computing t-SNE for embeddings...")
        tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30, metric="cosine")
        tsne_embs = tsne.fit_transform(all_embs)

        # Plot with professional color scheme
        unique_labels = np.unique(all_labels).tolist()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p']

        for i, label in enumerate(unique_labels):
            idx = all_labels == label
            if label == "reference":
                plt.scatter(tsne_embs[idx, 0], tsne_embs[idx, 1],
                            label="Reference",
                            alpha=0.3,
                            color="#808080",
                            s=20,
                            marker='o',
                            edgecolors='none')
            else:
                plt.scatter(tsne_embs[idx, 0], tsne_embs[idx, 1],
                            label=label,
                            alpha=0.6,
                            color=colors[i % len(colors)],
                            s=30,
                            marker=markers[i % len(markers)],
                            edgecolors='black',
                            linewidths=0.5)

        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title(f"Sentence-Pair Embeddings\n({self.name})")
        plt.legend(loc='best', frameon=True, fancybox=False,
                   edgecolor='black', framealpha=1.0,
                   ncol=1 if len(unique_labels) <= 4 else 2)
        plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        plt.tight_layout(pad=0.5)

        if save_path:
            # Save in multiple formats for publication
            base_path = os.path.splitext(save_path)[0]
            plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight', format='pdf')
            plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', format='png')
            logger.info(f"Saved publication-quality plots: {base_path}.pdf and {base_path}.png")
        if show:
            plt.show()


class PrecisionRecallDistanceEvaluator(BaseDatasetEvaluator):
    """
    Precision-Recall distance evaluator based on BERT embeddings.
    See: https://aclanthology.org/2021.findings-acl.193/

    Example:

        .. code-block:: python

            from sdialog.evaluation import PrecisionRecallDistanceEvaluator

            pr_distance = PrecisionRecallDistanceEvaluator(reference_dialogs)

            print(pr_distance(candidate_dialogs))

    :param reference_dialogues: Reference dialogues (list or path).
    :type reference_dialogues: Union[str, List[Dialog]]
    :param ai_speaker: If set, restrict to AI response pairs.
    :type ai_speaker: Optional[str]
    :param num_clusters: Number of k-means clusters.
    :type num_clusters: int
    :param num_angles: Angular resolution for PRD curve.
    :type num_angles: int
    :param num_runs: Repetition count when distributions unbalanced.
    :type num_runs: int
    :param name: Evaluator name.
    :type name: Optional[str]
    :param model_name: Underlying transformer model.
    :type model_name: str
    :param batch_size: Batch size for embedding.
    :type batch_size: int
    :param device: Torch device override.
    :type device: Optional[str]
    :param verbose: Verbosity flag.
    :type verbose: bool
    """
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 num_clusters=20,
                 num_angles=1001,
                 num_runs=10,
                 name: str = None,
                 model_name: str = "roberta-base",
                 batch_size: int = 128,
                 device: str = None,
                 verbose: bool = False):
        """Initialize PR distance evaluator."""
        if isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        if not reference_dialogues or not isinstance(reference_dialogues, list):
            raise ValueError("Reference dialogues must be provided as a list of Dialog objects or a file path.")

        self.name = name or f"pr-distance-{model_name.split('/')[-1]}" + ("-ai" if ai_speaker else "")
        self.verbose = verbose
        self.ai_speaker = ai_speaker
        self.num_clusters = num_clusters
        self.num_angles = num_angles
        self.num_runs = num_runs
        self.batch_size = batch_size
        self.model = SentencePairTransformer(model_name=model_name,
                                             device=device,
                                             verbose=verbose)
        self.reference_embs = self._encode_utterance_pairs(reference_dialogues)

    def _encode_utterance_pairs(self, dialogues: List[Dialog], dataset_name: str = "reference") -> np.ndarray:
        """
        Encode aligned utterance pairs.

        :param dialogues: List of dialogues.
        :type dialogues: List[Dialog]
        :param dataset_name: Label for logging.
        :type dataset_name: str
        :return: Embedding matrix.
        :rtype: np.ndarray
        """
        return self.model.encode(*dialogs_to_utt_pairs(dialogues, self.ai_speaker),
                                 batch_size=self.batch_size,
                                 progress_bar_desc=f"Computing embeddings for PRD on {dataset_name}")

    def _cluster_histograms(self, target_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings and build normalized histograms for reference and target.

        :param target_embs: Candidate embedding matrix.
        :type target_embs: np.ndarray
        :return: (reference_histogram, target_histogram)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        cluster_data = np.vstack([target_embs, self.reference_embs])
        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, n_init=10)
        labels = kmeans.fit(cluster_data).labels_

        reference_labels = labels[len(target_embs):]
        target_labels = labels[:len(target_embs)]

        reference_histogram = np.histogram(reference_labels, bins=self.num_clusters,
                                           range=[0, self.num_clusters], density=True)[0]
        target_histogram = np.histogram(target_labels, bins=self.num_clusters,
                                        range=[0, self.num_clusters], density=True)[0]
        return reference_histogram, target_histogram

    def _precision_recall_distance(self,
                                   reference_histogram: np.ndarray,
                                   target_histogram: np.ndarray,
                                   epsilon=1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute precision and recall arrays over angle sweep (PRD curve).

        :param reference_histogram: Reference distribution histogram.
        :type reference_histogram: np.ndarray
        :param target_histogram: Target distribution histogram.
        :type target_histogram: np.ndarray
        :param epsilon: Small positive epsilon to avoid boundary issues.
        :type epsilon: float
        :return: (precision array, recall array), each in [0,1].
        :rtype: Tuple[np.ndarray, np.ndarray]
        :raises ValueError: If epsilon or num_angles out of range.
        """
        if not (epsilon > 0 and epsilon < 0.1):
            raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
        if not (self.num_angles >= 3 and self.num_angles <= 1e6):
            raise ValueError('num_angles must be in [3, 1e6] but is %d.' % self.num_angles)

        angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=self.num_angles)
        slopes = np.tan(angles)

        slopes_2d = np.expand_dims(slopes, 1)

        ref_dist_2d = np.expand_dims(reference_histogram, 0)
        eval_dist_2d = np.expand_dims(target_histogram, 0)

        precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
        recall = precision / slopes

        return np.clip(precision, 0, 1), np.clip(recall, 0, 1)

    def __call__(self, dialogues: Union[str, List[Dialog]], dataset_name: str = None) -> Union[dict, float]:
        """
        Compute maximum F1 score along PRD curve (averaged if size mismatch).

        :param dialogues: Candidate dialogues (list or path).
        :type dialogues: Union[str, List[Dialog]]
        :param dataset_name: Label for candidate dataset.
        :type dataset_name: Optional[str]
        :return: Max F1 value.
        :rtype: float
        """
        target_embs = self._encode_utterance_pairs(dialogues, dataset_name)

        if len(target_embs) != len(self.reference_embs):
            logger.warning("The total number of utterance pairs in the reference dialogues "
                           f"({len(self.reference_embs)}) and those of the evaluation dialogues "
                           f"({len(target_embs)}) are not equal. "
                           "This may lead to misleading results since unbalanced distributions bias "
                           "the clustering towards the larger dataset.")

        precisions = []
        recalls = []
        for _ in range(self.num_runs):
            reference_histogram, target_histogram = self._cluster_histograms(target_embs)
            precision, recall = self._precision_recall_distance(reference_histogram, target_histogram)
            precisions.append(precision)
            recalls.append(recall)
        precision = np.mean(precisions, axis=0).tolist()
        recall = np.mean(recalls, axis=0).tolist()

        max_f1 = max([2 * p * r / (p + r) if (p + r) > 0 else 0
                      for p, r in zip(precision, recall)])
        # Convert F1 similarity score to distance metric (lower is better)
        return 1 - max_f1


class StatsEvaluator(BaseDatasetScoreEvaluator):
    """Statistics evaluator (mean/std/min/max/median).

    Example:

        .. code-block:: python

            from sdialog.evaluation import StatsEvaluator, LexicalDiversityScore

            # Any dialog score can be used, let's use `LexicalDiversityScore` as an example
            lexical_diversity = LexicalDiversityScore()
            stats_eval = StatsEvaluator(lexical_diversity)
            mean_eval = StatsEvaluator(lexical_diversity, stat="mean")

            stats = stats_eval(candidate_dialogs)
            mean = mean_eval(candidate_dialogs)

            # Print descriptive statistics for hesitation rate
            print(stats)  # {'mean': ..., 'std': ..., ...}
            print("Mean hesitation rate:", mean)  # Mean hesitation rate: ...

    :param dialog_score: Dialog scoring component.
    :type dialog_score: BaseDialogScore
    :param stat: Target statistic to return (one of 'mean', 'std', 'min', 'max', 'median'). If None, return all.
    :type stat: Optional[Literal["mean", "std", "min", "max", "median"]]
    :param metric: Deprecated alias for `stat`.
    :type metric: Optional[Literal["mean", "std", "min", "max", "median"]]
    :param name: Evaluator name.
    :type name: Optional[str]
    :param enable_plotting: Keep per-dataset scores for plotting.
    :type enable_plotting: bool
    :param verbose: Verbosity flag.
    :type verbose: bool
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 stat: Optional[Literal["mean", "std", "min", "max", "median"]] = None,
                 metric: Optional[Literal["mean", "std", "min", "max", "median"]] = None,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 plot_title: str = None,
                 plot_xlabel: str = None,
                 plot_ylabel: str = None):
        """Initialize statistics evaluator (mean/std/min/max/median).
        If `stat` is provided, only that value is returned. `metric` is deprecated and kept for backward compatibility.
        """
        valid = {"mean", "std", "min", "max", "median"}
        if stat is not None and stat not in valid:
            raise ValueError(f"Invalid stat: {stat}. Must be one of 'mean', 'std', 'min', 'max', 'median'.")
        if metric is not None:
            warnings.warn("`metric` is deprecated; use `stat` instead.", DeprecationWarning)
            if stat is None:
                stat = metric
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose,
                         plot_title=plot_title, plot_xlabel=plot_xlabel, plot_ylabel=plot_ylabel)
        self.stat = stat

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None, metric: str = None):
        """
        Plot boxplots showing score distributions.

        :param dialog_scores: Mapping dataset -> score array.
        :type dialog_scores: Dict[str, np.ndarray]
        :param plot: Matplotlib Axes or pyplot module.
        :type plot: Optional[plt.Axes]
        :param metric: Optional metric name override.
        :type metric: Optional[str]
        :return: None
        :rtype: None
        """
        # Plot boxplots for score distributions
        name = metric or self.name or self.stat
        title = name or f"{self.dialog_score.name} scores"

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#bcbd22', '#17becf']

        bp = plot.boxplot(list(dialog_scores.values()),
                          labels=list(dialog_scores.keys()),
                          patch_artist=True,
                          widths=0.6,
                          medianprops=dict(color='red', linewidth=1.5),
                          boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=0.8, alpha=0.7),
                          whiskerprops=dict(color='black', linewidth=0.8),
                          capprops=dict(color='black', linewidth=0.8),
                          flierprops=dict(marker='o', markerfacecolor='gray', markersize=4,
                                          linestyle='none', markeredgecolor='black', alpha=0.5))

        # Color each box differently
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        plt.xticks(rotation=45, ha='right')
        plot.xlabel(self.plot_xlabel if self.plot_xlabel else "Datasets")
        plot.ylabel(self.plot_ylabel if self.plot_ylabel else (name or self.dialog_score.name))
        plot.title(self.plot_title if self.plot_title else f"Distribution of {title}")
        plot.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')
        ax = plt.gca() if plot == plt else plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=0.5)

    def __eval__(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        """
        Compute descriptive statistics.

        :param dialog_scores: List/array of numeric scores.
        :type dialog_scores: List[Union[float, int]]
        :return: Dict of statistics, or a single value if a target statistic was specified.
        :rtype: Union[dict, float]
        """
        if len(dialog_scores) == 0:
            logger.warning(f"No valid scores to compute statistics for {self.name}. Returning zeros.")

        stats = {
            "mean": np.mean(dialog_scores) if len(dialog_scores) > 0 else 0.0,
            "std": np.std(dialog_scores) if len(dialog_scores) > 0 else 0.0,
            "min": np.min(dialog_scores) if len(dialog_scores) > 0 else 0.0,
            "max": np.max(dialog_scores) if len(dialog_scores) > 0 else 0.0,
            "median": np.median(dialog_scores) if len(dialog_scores) > 0 else 0.0
        }
        return stats[self.stat] if self.stat in stats else stats


class MeanEvaluator(StatsEvaluator):
    """
    Evaluator for computing the mean of dialog scores.
    This class is a thin wrapper around `StatsEvaluator` with `stat="mean"`.

    Example:

        .. code-block:: python

            from sdialog.evaluation import MeanEvaluator, ReadabilityScore

            flesch_score = ReadabilityScore(feature="flesch-reading-ease")

            mean_eval = MeanEvaluator(flesch_score)

            print("Average Flesch reading ease:", mean_eval(candidate_dialogs))

    :param dialog_score: Dialog scoring component.
    :type dialog_score: BaseDialogScore
    :param name: Evaluator name.
    :type name: Optional[str]
    :param enable_plotting: Keep scores for plotting.
    :type enable_plotting: bool
    :param verbose: Verbosity flag.
    :type verbose: bool
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 plot_title: str = None,
                 plot_xlabel: str = None,
                 plot_ylabel: str = None):
        """Initialize mean-only evaluator using parent `stat` mechanism."""
        super().__init__(dialog_score,
                         stat="mean",
                         name=name,
                         enable_plotting=enable_plotting,
                         verbose=verbose,
                         plot_title=plot_title,
                         plot_xlabel=plot_xlabel,
                         plot_ylabel=plot_ylabel)

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None, metric: str = None):
        """
        Plot bar chart of mean scores.

        :param dialog_scores: Mapping dataset -> score array.
        :type dialog_scores: Dict[str, np.ndarray]
        :param plot: Matplotlib Axes or pyplot module.
        :type plot: Optional[plt.Axes]
        :param metric: Optional metric name override.
        :type metric: Optional[str]
        :return: None
        :rtype: None
        """
        # Plot bar chart with mean values for each dataset
        name = metric or self.name or self.stat
        title = name or f"{self.dialog_score.name} scores"
        means = {k: np.mean(v) for k, v in dialog_scores.items()}

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#bcbd22', '#17becf']
        bar_colors = [colors[i % len(colors)] for i in range(len(means))]

        bars = plot.bar(means.keys(), means.values(), color=bar_colors,
                        alpha=0.85, edgecolor='black', linewidth=0.8)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plot.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}",
                      ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plot.xlabel(self.plot_xlabel if self.plot_xlabel else "Datasets")
        plot.ylabel(self.plot_ylabel if self.plot_ylabel else (name or self.dialog_score.name))
        plot.title(self.plot_title if self.plot_title else f"Mean {title}")
        plot.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

        # Automatically adjust y-axis to highlight differences
        ax = plt.gca() if plot == plt else plot
        values = list(means.values())
        if values:
            y_min, y_max = min(values), max(values)
            y_range = y_max - y_min
            # Add padding (10% below min, 15% above max for labels)
            padding_bottom = max(y_range * 0.1, 0.01) if y_range > 0 else y_max * 0.1
            padding_top = max(y_range * 0.15, 0.01) if y_range > 0 else y_max * 0.15
            ax.set_ylim([y_min - padding_bottom, y_max + padding_top])
        ax = plt.gca() if plot == plt else plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=0.5)


class FrequencyEvaluator(BaseDatasetScoreEvaluator):
    """
    Evaluator for computing the frequency or percentage of dialogues scored as 1 / True (e.g., refusal responses).

    Example:

        .. code-block:: python

            from sdialog.evaluation import FrequencyEvaluator, LLMJudgeRealDialog

            judge_real = LLMJudgeRealDialog()

            freq = FrequencyEvaluator(judge_real)

            print(freq(dialogs))  # Outputs proportion of dialogues judged as real

    :param dialog_score: Dialog scoring component producing binary outputs.
    :type dialog_score: BaseDialogScore
    :param name: Evaluator name.
    :type name: Optional[str]
    :param enable_plotting: Retained for API parity (not used directly).
    :type enable_plotting: bool
    :param verbose: Verbosity flag.
    :type verbose: bool
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 plot_title: str = None,
                 plot_xlabel: str = None,
                 plot_ylabel: str = None):
        """Initialize frequency evaluator."""
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose,
                         plot_title=plot_title, plot_xlabel=plot_xlabel, plot_ylabel=plot_ylabel)

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None, metric: str = None):
        """
        Plot bar chart of positive proportions.

        :param dialog_scores: Mapping dataset -> binary score array.
        :type dialog_scores: Dict[str, np.ndarray]
        :param plot: Matplotlib Axes or pyplot module.
        :type plot: Optional[plt.Axes]
        :param metric: Optional label override.
        :type metric: Optional[str]
        :return: None
        :rtype: None
        """
        # Bar plot for frequency/percentage
        percentages = {k: np.mean(v) * 100 for k, v in dialog_scores.items()}

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#bcbd22', '#17becf']
        bar_colors = [colors[i % len(colors)] for i in range(len(percentages))]

        bars = plot.bar(percentages.keys(), percentages.values(),
                        color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plot.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%",
                      ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plot.ylabel(self.plot_ylabel if self.plot_ylabel else "Percentage (%)")
        plot.xlabel(self.plot_xlabel if self.plot_xlabel else "Datasets")
        plot.title(self.plot_title if self.plot_title else f"{metric or self.dialog_score.name} Frequency")
        plot.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')
        ax = plt.gca() if plot == plt else plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, min(105, max(percentages.values()) * 1.15)])  # Set y-axis limit with some headroom
        plt.tight_layout(pad=0.5)

    def __eval__(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        """
        Compute proportion of positive dialogues.

        :param dialog_scores: Binary scores list (0/1 or False/True).
        :type dialog_scores: List[Union[float, int]]
        :return: Fraction in [0,1].
        :rtype: float
        """
        # Assumes dialog_scores are binary (0/1 or True/False)
        total = len(dialog_scores)
        count = np.sum(dialog_scores)
        percentage = count / total if total > 0 else 0
        return percentage


class DatasetComparator:
    """
    Run multiple evaluators over several dialog datasets and collect results.

    Example:

        .. code-block:: python

            from sdialog.evaluation import LLMJudgeRealDialog, DialogFlowPPL
            from sdialog.evaluation import FrequencyEvaluator, MeanEvaluator
            from sdialog.evaluation import DatasetComparator

            # Dialog scores
            judge_real = LLMJudgeRealDialog()
            flow_score = DialogFlowScore(reference_dialogs)

            # Comparator with two evaluators
            comparator = DatasetComparator(evaluators=[FrequencyEvaluator(judge_real),
                                                       MeanEvaluator(flow_score)])

            comparator({"modelA": modelA_dialogs,  # print table by default
                        "modelB": modelB_dialogs})
            # results = comparator({"modelA": modelA_dialogs,
            #                       "modelB": modelB_dialogs},
            #                      output="dict")  # return results as dict

            comparator.plot()  # plot results for each evaluator that support it

    :param evaluators: Single evaluator instance or list of evaluator instances.
    :type evaluators: Union[BaseDatasetEvaluator, List[BaseDatasetEvaluator]]
    """
    def __init__(self, evaluators: Union[BaseDatasetEvaluator, List[BaseDatasetEvaluator]]):
        """Initialize dataset comparator."""
        if not evaluators:
            raise ValueError("No evaluators provided for comparison.")
        if not isinstance(evaluators, list):
            evaluators = [evaluators]
        for evaluator in evaluators:
            if not isinstance(evaluator, BaseDatasetEvaluator):
                raise TypeError(f"Evaluator {evaluator} is not an instance of `BaseDatasetEvaluator`")

        self.evaluators = evaluators

    def __call__(
        self,
        candidates: Union[str, List[Dialog], List[str], List[List[Dialog]], Dict[str, str], Dict[str, List[Dialog]]],
        digits: int = 2,
        output: Union[str, type] = "markdown",
    ) -> dict:
        """
        Evaluate multiple candidate datasets with all evaluators.

        :param candidates: Collection of datasets (lists/paths/dicts of Dialog objects).
        :type candidates: Union[str, List[Dialog], List[str], List[List[Dialog]], Dict[str, str], Dict[str, List[Dialog]]]
        :param digits: Decimal precision for tabular output.
        :type digits: int
        :param output: Output format: 'dict', 'markdown', or 'table'.
        :type output: Union[str, type]
        :return: Results mapping (dataset -> metric -> value) if output='dict'; otherwise prints a table.
        :rtype: Optional[dict]
        :raises ValueError: If candidates empty or output format unsupported.
        """  # noqa: E501
        if not candidates:
            raise ValueError("No candidates provided for comparison.")

        if isinstance(candidates, str) or isinstance(candidates, list) and isinstance(candidates[0], Dialog):
            candidates = [candidates]  # Ensure candidates is always a list of datasets (set of dialogues)

        # Clear the historical results of each evaluator
        for evaluator in self.evaluators:
            if hasattr(evaluator, "clear"):
                evaluator.clear()

        results = {}
        dataset_iterator = candidates.items() if isinstance(candidates, dict) else enumerate(candidates)
        for dataset_name, dataset in tqdm(dataset_iterator, desc="Evaluating datasets", leave=False):
            if isinstance(dataset_name, int):
                dataset_name += 1
            results[dataset_name] = {}
            for evaluator in tqdm(self.evaluators, desc="Running evaluators", leave=False):
                score = evaluator(dataset, dataset_name=dataset_name)
                if isinstance(score, dict):
                    for metric, value in score.items():
                        metric = f"{evaluator.name}-{metric}" if evaluator.name else metric
                        results[dataset_name][metric] = value
                else:
                    results[dataset_name][evaluator.name] = score

        if output == "dict" or output is dict:
            return results
        elif output in ["markdown", "table"]:
            dict_to_table(results, markdown=output == "markdown", format=f".{digits}f")  # sort_by="evaluator_name"
        else:
            raise ValueError(f"Unsupported output format: {output}. Supported formats are "
                             "'dict', 'markdown', and 'table'.")
        return results

    def plot(self, show: bool = True, save_folder_path: str = None):
        """
        Call plot() on each evaluator that supports it.

        :param show: Whether to display plots.
        :type show: bool
        :param save_folder_path: Directory to save plots (one file per evaluator).
        :type save_folder_path: Optional[str]
        :return: None
        :rtype: None
        """
        """
        Plot the results of the evaluators.
        """
        if not self.evaluators:
            logger.info("No evaluators to plot.")
            return

        for evaluator in self.evaluators:
            if hasattr(evaluator, "plot"):
                evaluator.plot(show=show,
                               save_path=os.path.join(save_folder_path,
                                                      f"{evaluator.name}.png") if save_folder_path else None)


# Alias for backward compatibility and simplicity
Comparator = DatasetComparator
