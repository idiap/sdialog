"""Tests for dialog2flow graphs and the flow-based scores (DialogFlowPPL, DialogFlowScore), with a fake encoder."""
import zlib

import numpy as np
import pytest

from sdialog import Dialog, Turn
from sdialog.evaluation.dialog2flow import dialog2graph


class FakeEncoder:
    """Bag-of-words sentence encoder: similar sentences get similar vectors, no model is loaded."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        embs = []
        for text in ([texts] if single else list(texts)):
            v = np.zeros(32, dtype=np.float32)
            for word in text.lower().split():
                v[zlib.crc32(word.encode()) % 32] += 1
            embs.append(v / (np.linalg.norm(v) or 1))
        embs = np.array(embs)
        return embs[0] if single else embs


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    monkeypatch.setattr("sdialog.evaluation.dialog2flow.extract_trajectories.SentenceTransformer", FakeEncoder)
    monkeypatch.setattr("sdialog.evaluation.base.SentenceTransformer", FakeEncoder)


def support_dialog(i=0):
    return Dialog(id=f"d{i}", turns=[
        Turn(speaker="Customer", text="hello i need help with my order"),
        Turn(speaker="Bot", text="sure what is your order number"),
        Turn(speaker="Customer", text=f"my order number is {i}"),
        Turn(speaker="Bot", text="thanks your order will arrive tomorrow"),
        Turn(speaker="Customer", text="great thank you bye"),
        Turn(speaker="Bot", text="you are welcome goodbye"),
    ])


REFERENCE = [support_dialog(i) for i in range(4)]


def flow_kwargs(tmp_path):
    return {"output_path": str(tmp_path / "d2f"), "verbose": False}


def test_dialog2graph_system_speaker_not_first(tmp_path):
    # The customer speaks first, but the bot is the system: it must not be mapped to "user"
    _, nodes = dialog2graph(REFERENCE, system_speaker_name="Bot", node_llm_labels_enabled=False, out_png=False,
                            **flow_kwargs(tmp_path))
    assert nodes["_metadata"]["speakers"]["bot"] == "system"
