"""Tests for dialog2flow graphs and the flow-based scores (DialogFlowPPL, DialogFlowScore), with a fake encoder."""
import os
import zlib

import numpy as np
import pytest

from sdialog import Dialog, Turn
from sdialog.evaluation import DialogFlowPPL, DialogFlowScore
from sdialog.evaluation.dialog2flow import dialog2graph, DEFAULT_TOKEN_START
from sdialog.evaluation.dialog2flow.extract_trajectories import dialog2trajectories


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
UNRELATED = Dialog(turns=[
    Turn(speaker="Customer", text="what is the weather on mars"),
    Turn(speaker="Bot", text="purple elephants dance quietly"),
    Turn(speaker="Customer", text="seven bananas sing loudly"),
    Turn(speaker="Bot", text="mountains swim in green rivers"),
])


def flow_kwargs(tmp_path):
    return {"output_path": str(tmp_path / "d2f"), "verbose": False}


def test_dialog2graph(tmp_path):
    graph, nodes = dialog2graph(REFERENCE, system_speaker_name="Bot", node_llm_labels_enabled=False,
                                out_png=False, **flow_kwargs(tmp_path))
    node_ids = [n for n in nodes if not n.startswith("_")]
    assert any(n.startswith("s") for n in node_ids) and any(n.startswith("u") for n in node_ids)
    assert DEFAULT_TOKEN_START in graph.nodes
    assert graph.number_of_edges() > 0
    assert nodes["_metadata"]["speakers"] == {"bot": "system", "customer": "user"}
    for info in (nodes[n] for n in node_ids):
        assert len(info["centroid-embedding"]) == 32


def test_dialog2graph_system_speaker_not_first(tmp_path):
    # The customer speaks first, but the bot is the system: it must not be mapped to "user"
    _, nodes = dialog2graph(REFERENCE, system_speaker_name="Bot", node_llm_labels_enabled=False, out_png=False,
                            **flow_kwargs(tmp_path))
    assert nodes["_metadata"]["speakers"]["bot"] == "system"


def test_dialog2graph_from_folder_and_interactive(tmp_path):
    folder = tmp_path / "dialogs"
    folder.mkdir()
    for dialog in REFERENCE:
        dialog.to_file(str(folder / f"{dialog.id}.json"))
    (folder / "notes.md").write_text("ignored")
    graph, nodes = dialog2graph(str(folder), node_llm_labels_enabled=False, out_png=False, out_interactive=True,
                                use_closest_as_centroid_emb=True, verbose=False)
    assert graph.number_of_nodes() > 2
    assert os.path.exists(folder / "dialog2flow")
    html = [os.path.join(root, f) for root, _, files in os.walk(folder / "dialog2flow") for f in files
            if f == "graph.html"]
    assert html and "graph_happy_path" in open(html[0]).read()
    assert all("closest-embedding" in nodes[n] for n in nodes if not n.startswith("_"))


def test_dialog2trajectories(tmp_path):
    path = dialog2trajectories(REFERENCE, system_speaker_name="Bot", thresholds=[0.6, 0.6], dendrogram=True,
                               **flow_kwargs(tmp_path))
    assert os.path.exists(path) and path.endswith(".json")
    with pytest.raises(ValueError):
        dialog2trajectories([{"not": "a dialog"}], **flow_kwargs(tmp_path))


def test_dialog_flow_ppl(tmp_path):
    ppl = DialogFlowPPL(REFERENCE, ai_speaker="Bot", **flow_kwargs(tmp_path))
    assert ppl.name == "dfppl-ai"
    in_flow, out_of_flow = ppl(support_dialog(9)), ppl(UNRELATED)
    assert 0 < in_flow < out_of_flow
    known = DialogFlowPPL(REFERENCE, ai_speaker="Bot", use_only_known_edges=True, use_softmax=False,
                          **flow_kwargs(tmp_path))
    assert known.name == "dfppl-hard-ai-only-known"
    assert known(support_dialog(9)) > 0
    assert ppl(Dialog(turns=[])) is None  # no turns to score


def test_dialog_flow_score(tmp_path):
    score = DialogFlowScore(REFERENCE, ai_speaker="Bot", **flow_kwargs(tmp_path))
    assert score.name == "dfs-ai"
    assert 0 < score(UNRELATED) < score(support_dialog(9)) <= 1
    nodes, probs = score.get_node_sequence(support_dialog(9), probs=True)
    assert len(nodes) == len(probs) == 6 and nodes[0].startswith("u") and nodes[1].startswith("s")
    assert score.get_node_sequence(support_dialog(9)) == nodes
    with pytest.raises(ValueError, match="not found in the graph"):
        score(Dialog(turns=[Turn(speaker="Stranger", text="hi")]))

    # Reusing the graph and nodes of another score does not rebuild the graph
    reused = DialogFlowScore(REFERENCE, ai_speaker="Bot", graph=score.graph, nodes=score.nodes,
                             use_only_known_edges=True)
    assert reused.encoder is score.encoder and reused.name == "dfs-ai-only-known"
    assert reused(support_dialog(9)) > 0


def test_dialog_flow_score_only_ai_speaker(tmp_path):
    score = DialogFlowScore(REFERENCE, ai_speaker="Bot", use_only_ai_speaker=True, use_softmax=False,
                            **flow_kwargs(tmp_path))
    assert "user" not in score.knn_models
    assert all(n.startswith("s") for n in score.get_node_sequence(support_dialog(9)))
    assert score(support_dialog(9)) > score(UNRELATED)


def test_dialog_flow_score_from_folder(tmp_path):
    for dialog in REFERENCE:
        dialog.to_file(str(tmp_path / f"{dialog.id}.json"))
    assert DialogFlowScore(str(tmp_path), ai_speaker="Bot", verbose=False,
                           output_path=str(tmp_path / "d2f"))(support_dialog(9)) > 0
    with pytest.raises(ValueError):
        DialogFlowScore([])
