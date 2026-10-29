import pytest
import numpy as np

from sdialog.evaluation import (
    _cs_divergence,
    _kl_divergence,
    LLMJudgeYesNoOutput,
    SentenceTransformerDialogEmbedder,
    ReferenceCentroidEmbeddingEvaluator,
    StatsEvaluator,
    ConversationalFeatures,
    GunningFogScore,
)
from sdialog import Dialog, Turn

dummy_dialog = Dialog(
    turns=[Turn(text="hello world", speaker="A"),
           Turn(text="goodbye world", speaker="B")],
)


def test_cs_divergence_identical():
    arr = np.random.normal(0, 1, 100)
    assert _cs_divergence(arr, arr) == pytest.approx(0, abs=1e-6)


def test_kl_divergence_identical():
    arr = np.random.normal(0, 1, 100)
    assert _kl_divergence(arr, arr) == pytest.approx(0, abs=1e-6)


def test_llmjudgeyesno_output():
    out = LLMJudgeYesNoOutput(positive=True, reason="ok")
    assert out.positive is True
    assert out.reason == "ok"


def test_sentence_transformer_dialog_embedder_mean(monkeypatch):
    # Patch SentenceTransformer to avoid loading model
    class DummyModel:

        def encode(self, texts, show_progress_bar=False):
            return np.ones((len(texts), 3))

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setattr("sdialog.evaluation.SentenceTransformer", lambda *a, **k: DummyModel())
    embedder = SentenceTransformerDialogEmbedder(model_name="dummy", mean=True)
    emb = embedder.embed(dummy_dialog)
    assert emb.shape == (3,)
    assert np.allclose(emb, 1)


def test_reference_centroid_embedding_evaluator(monkeypatch):
    class DummyModel:

        def encode(self, texts, show_progress_bar=False):
            return np.ones((len(texts), 3))

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setattr("sdialog.evaluation.SentenceTransformer", lambda *a, **k: DummyModel())
    embedder = SentenceTransformerDialogEmbedder(model_name="dummy", mean=True)
    dialogs = [dummy_dialog for _ in range(3)]
    evaluator = ReferenceCentroidEmbeddingEvaluator(embedder, dialogs)
    # Centroid similarity with itself should be 1.0
    embs = np.array([embedder.embed(d) for d in dialogs])
    sim = evaluator.__eval__(embs)
    assert sim == pytest.approx(1.0)


def test_stats_evaluator():
    class DummyScore:
        name = "dummy"

        def __call__(self, dialog):
            return 1.0

    evaluator = StatsEvaluator(DummyScore())
    dialogs = [dummy_dialog for _ in range(3)]
    stats = evaluator(dialogs)
    assert "mean" in stats
    assert stats["mean"] == 1.0


def test_conversational_features_ignore_speaker_names():
    dialog = Dialog(turns=[Turn(speaker="Hmm", text="Do you like it?"), Turn(speaker="B", text="yeah")])
    features = ConversationalFeatures()(dialog)
    assert features["back_channel_rate"] == 50.0  # "yeah" is a back-channel turn
    assert features["hesitation_rate"] == 0.0  # the speaker name "Hmm" is not a hesitation
    assert features["lexical_diversity"] == 1.0  # speaker names are not counted as words


def test_readability_ignores_speaker_names():
    dialog = Dialog(turns=[Turn(speaker="Extraordinarily", text="I see."), Turn(speaker="B", text="Me too.")])
    assert GunningFogScore()(dialog) == pytest.approx(0.4 * 2)  # 4 words, 2 sentences, no complex words
