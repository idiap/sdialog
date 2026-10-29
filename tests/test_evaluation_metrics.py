"""Tests for the dialogue scores and dataset evaluators of sdialog.evaluation (no models are downloaded)."""
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

from sdialog import Dialog, Turn
from sdialog.evaluation import (ConversationalFeatures, ReadabilityScore, MeanTurnLengthScore, TurnLength,
                                HesitationRateScore, TurnTakingRatioScore, QuestionRateScore, LexicalDiversityScore,
                                BackChannelRateScore, FillerWordDensityScore, GunningFogScore, FleschReadingEaseScore,
                                ColemanLiauScore, LinsearWriteScore, DaleChallScore, KDEDistanceEvaluator,
                                FrechetDistanceEvaluator, FrechetBERTDistanceEvaluator,
                                PrecisionRecallDistanceEvaluator, StatsEvaluator, MeanEvaluator, FrequencyEvaluator,
                                DatasetComparator, Comparator)
from sdialog.evaluation.base import BaseDialogScore, BaseDatasetEvaluator

plt.switch_backend("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def make_dialog(*texts, speakers=("Alice", "Bob")):
    return Dialog(turns=[Turn(speaker=speakers[i % len(speakers)], text=t) for i, t in enumerate(texts)])


DIALOG = make_dialog("Um, do you like coffee?", "yeah", "I mean, basically I drink it every day.", "Okay.")


class LengthScore(BaseDialogScore):
    """Number of turns of the dialogue."""

    def __init__(self):
        super().__init__(name="length")

    def score(self, dialog):
        return len(dialog)


class PoliteScore(BaseDialogScore):
    """1 if the dialogue says "thanks", 0 otherwise."""

    def __init__(self):
        super().__init__(name="polite")

    def score(self, dialog):
        return int("thanks" in str(dialog).lower())


def datasets():
    short = [make_dialog(*["Hi."] * n) for n in (2, 3, 3, 4, 2)]
    long = [make_dialog(*["Hello, thanks."] * n) for n in (6, 7, 8, 7, 9)]
    return short, long


# Dialogue scores

def test_conversational_features_values():
    features = ConversationalFeatures()(DIALOG)
    assert features["mean-turn-length"] == pytest.approx(np.mean([len(t) for t in DIALOG]))
    assert features["back_channel_rate"] == 50.0  # "yeah" and "Okay."
    assert features["question_rate"] == 25.0
    assert features["turn_taking_ratio"] == 1.0  # two speakers, two turns each
    assert features["hesitation_rate"] > 0 and features["filler_word_density"] > 0
    assert 0 < features["lexical_diversity"] <= 1


def test_conversational_features_selection_and_speaker():
    score = ConversationalFeatures(feature="question-rate", speaker="Alice")
    assert score.name == "question-rate"
    assert score(DIALOG) == 50.0  # one of Alice's two turns is a question
    both = ConversationalFeatures(feature=["question-rate", "back-channel-rate"])(DIALOG)
    assert set(both) == {"question_rate", "back_channel_rate"}
    with pytest.raises(ValueError):
        ConversationalFeatures(feature="nope")


def test_conversational_features_edge_cases():
    empty = Dialog(turns=[])
    assert ConversationalFeatures.calculate_turn_taking_ratio(empty) == 0.0
    assert ConversationalFeatures.calculate_turn_taking_ratio(make_dialog("a", "b", speakers=("A",))) == 0.0
    assert TurnTakingRatioScore()(make_dialog("a", "b", "c", "d", "e", speakers=("A", "A", "A", "B"))) < 1
    assert LexicalDiversityScore()(empty) == 0.0


@pytest.mark.parametrize("score_class, expected", [
    (MeanTurnLengthScore, np.mean([len(t) for t in DIALOG])),
    (HesitationRateScore, None),
    (QuestionRateScore, 25.0),
    (LexicalDiversityScore, None),
    (BackChannelRateScore, 50.0),
    (FillerWordDensityScore, None),
])
def test_conversational_feature_scores(score_class, expected):
    value = score_class()(DIALOG)
    assert isinstance(value, float)
    if expected is not None:
        assert value == pytest.approx(expected)
    assert score_class(speaker="Bob")(DIALOG) is not None


def test_turn_length():
    assert TurnLength()(DIALOG) == [len(t) for t in DIALOG]
    assert TurnLength(speaker="Bob")(DIALOG) == [len(DIALOG.turns[1]), len(DIALOG.turns[3])]


def test_readability_scores():
    simple = make_dialog("The cat sat.", "It is red.")
    hard = make_dialog("Institutionalization necessitates extraordinarily comprehensive administrative coordination.",
                       "Unquestionably, organizational transformation is considerably complicated.")
    all_scores = ReadabilityScore()(simple)
    assert set(all_scores) == {"gunning_fog", "flesch_reading_ease", "coleman_liau", "linsear_write", "dale_chall"}
    assert GunningFogScore()(hard) > GunningFogScore()(simple)
    assert FleschReadingEaseScore()(hard) < FleschReadingEaseScore()(simple)
    assert ColemanLiauScore()(hard) > ColemanLiauScore()(simple)
    assert LinsearWriteScore()(hard) > LinsearWriteScore()(simple)
    assert DaleChallScore()(hard) > DaleChallScore()(simple)
    assert ReadabilityScore(feature="dale-chall", speaker="Alice").name == "dale-chall"
    with pytest.raises(ValueError):
        ReadabilityScore(feature="nope")


@pytest.mark.parametrize("method", ["calculate_gunning_fog", "calculate_flesch_reading_ease", "calculate_coleman_liau",
                                    "calculate_linsear_write", "calculate_dale_chall"])
def test_readability_empty_text(method):
    assert getattr(ReadabilityScore, method)("") == 0
    assert getattr(ReadabilityScore, method)("...") == 0


# Dataset evaluators

def test_stats_evaluator_stats_and_plot(tmp_path):
    short, long = datasets()
    evaluator = StatsEvaluator(LengthScore(), stat="max")
    assert evaluator(short, dataset_name="short") == 4
    stats = StatsEvaluator(LengthScore())(long)
    assert stats["min"] == 6 and stats["median"] == 7 and stats["std"] > 0
    with pytest.warns(DeprecationWarning):
        assert StatsEvaluator(LengthScore(), metric="min")(short) == 2
    with pytest.raises(ValueError):
        StatsEvaluator(LengthScore(), stat="mode")
    evaluator(long, dataset_name="long")
    evaluator.plot(show=False, save_path=str(tmp_path / "stats.png"))
    assert (tmp_path / "stats.png").exists()


def test_mean_and_frequency_evaluators_plot(tmp_path):
    short, long = datasets()
    mean = MeanEvaluator(LengthScore())
    assert mean(short, dataset_name="short") == pytest.approx(2.8)
    mean(long, dataset_name="long")
    mean.plot(show=False, save_path=str(tmp_path / "mean.png"))
    frequency = FrequencyEvaluator(PoliteScore())
    assert frequency(short + long[:1], dataset_name="mixed") == pytest.approx(1 / 6)
    frequency(long, dataset_name="long")
    frequency.plot(show=False, save_path=str(tmp_path / "frequency.png"))
    assert (tmp_path / "mean.png").exists() and (tmp_path / "frequency.png").exists()


def test_kde_distance_evaluator(tmp_path):
    short, long = datasets()
    kl = KDEDistanceEvaluator(LengthScore(), reference_dialogues=short)
    assert kl(short) == pytest.approx(0, abs=1e-6)
    assert kl(long, dataset_name="long") > 0.1
    both = KDEDistanceEvaluator(LengthScore(), reference_dialogues=short, metric="all")(long)
    assert set(both) == {"cs", "kl"}
    assert KDEDistanceEvaluator(LengthScore(), reference_dialogues=short, metric="cs")(long) > 0
    kl.plot(show=False, save_path=str(tmp_path / "kde.png"))
    assert (tmp_path / "kde.png").exists()
    with pytest.raises(ValueError):
        KDEDistanceEvaluator(LengthScore())
    with pytest.raises(ValueError):
        KDEDistanceEvaluator(LengthScore(), reference_dialogues=short[0])


def test_score_evaluators_from_files(tmp_path):
    short, long = datasets()
    for i, dialog in enumerate(short):
        dialog.to_file(str(tmp_path / f"{i}.json"))
    evaluator = KDEDistanceEvaluator(LengthScore(), reference_dialogues=str(tmp_path))
    assert len(evaluator.reference_scores) == len(short)
    assert MeanEvaluator(LengthScore())(str(tmp_path)) == pytest.approx(2.8)


def test_frechet_distance_evaluator(tmp_path):
    short, long = datasets()
    evaluator = FrechetDistanceEvaluator(LengthScore(), reference_dialogues=short)
    assert evaluator(short) == pytest.approx(0)
    assert evaluator(long, dataset_name="long") > 3
    evaluator.plot(show=False, save_path=str(tmp_path / "frechet.png"))
    assert (tmp_path / "frechet.png").exists()
    with pytest.raises(ValueError):
        FrechetDistanceEvaluator(LengthScore())


class FakeSentencePairTransformer:
    """Embeds an utterance pair as a deterministic random vector seeded by its text."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, utts, utts_next, batch_size=None, progress_bar_desc=None):
        def emb(a, b):
            rng = np.random.default_rng(sum(map(ord, a + "|" + b)))
            return rng.normal(size=4) + (5 if "thanks" in a.lower() else 0)
        return np.array([emb(a, b) for a, b in zip(utts, utts_next)])


def varied_datasets():
    rng = np.random.default_rng(0)
    words = ["coffee", "tea", "milk", "sugar", "cup", "please", "now", "later"]
    short = [make_dialog(*[" ".join(rng.choice(words, 3)) for _ in range(4)]) for _ in range(10)]
    long = [make_dialog(*["thanks " + " ".join(rng.choice(words, 3)) for _ in range(4)]) for _ in range(10)]
    return short, long


def test_frechet_bert_distance_evaluator(monkeypatch, tmp_path):
    monkeypatch.setattr("sdialog.evaluation.SentencePairTransformer", FakeSentencePairTransformer)
    short, long = varied_datasets()
    evaluator = FrechetBERTDistanceEvaluator(short, enable_plotting=True)
    assert evaluator.name == "frechet-bert-distance"
    assert evaluator(short, dataset_name="same") == pytest.approx(0, abs=1e-6)
    assert evaluator(long, dataset_name="different") > 10
    evaluator.plot(show=False, save_path=str(tmp_path / "fbd.png"))
    assert (tmp_path / "fbd.png").exists() and (tmp_path / "fbd.pdf").exists()
    assert FrechetBERTDistanceEvaluator(short, ai_speaker="Bob").name == "frechet-bert-distance-ai"
    with pytest.raises(ValueError):
        FrechetBERTDistanceEvaluator([])


def test_precision_recall_distance_evaluator(monkeypatch):
    monkeypatch.setattr("sdialog.evaluation.SentencePairTransformer", FakeSentencePairTransformer)
    short, long = varied_datasets()
    evaluator = PrecisionRecallDistanceEvaluator(short, num_clusters=3, num_runs=2, num_angles=11)
    assert evaluator.name == "pr-distance-roberta-base"
    same = evaluator(short)
    different = evaluator(long + long[:1])  # different number of pairs: warns, still works
    assert 0 <= same < different <= 1
    with pytest.raises(ValueError):
        evaluator._precision_recall_distance(np.ones(3), np.ones(3), epsilon=1)
    evaluator.num_angles = 2
    with pytest.raises(ValueError):
        evaluator._precision_recall_distance(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        PrecisionRecallDistanceEvaluator([])


def test_dataset_comparator(tmp_path, capsys):
    short, long = datasets()
    comparator = DatasetComparator([MeanEvaluator(LengthScore()), StatsEvaluator(LengthScore(), name="stats"),
                                    FrequencyEvaluator(PoliteScore(), name="polite")])
    results = comparator({"short": short, "long": long}, output="dict")
    assert results["short"]["mean-length"] == pytest.approx(2.8)
    assert results["long"]["stats-max"] == 9 and results["long"]["polite"] == 1
    results = comparator([short, long])
    assert set(results) == {1, 2}
    assert "polite" in capsys.readouterr().out  # markdown table printed
    assert set(comparator(short, output="table")) == {1}
    comparator.plot(show=False, save_folder_path=str(tmp_path))
    assert os.path.exists(tmp_path / "polite.png")
    with pytest.raises(ValueError):
        comparator({"short": short}, output="csv")
    with pytest.raises(ValueError):
        comparator([])
    with pytest.raises(ValueError):
        DatasetComparator([])
    with pytest.raises(TypeError):
        DatasetComparator([LengthScore()])
    assert Comparator is DatasetComparator


def test_base_dataset_evaluator_is_abstract():
    with pytest.raises(TypeError):
        BaseDatasetEvaluator()
