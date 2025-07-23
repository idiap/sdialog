import pytest

from sdialog.util import make_serializable, dialogs_to_utt_pairs
from sdialog import Dialog


def test_make_serializable_dict():
    d = {"a": 1, "b": [1, 2], "c": {"d": 3}}
    make_serializable(d)
    assert isinstance(d, dict)
    assert isinstance(d["b"], list)
    assert isinstance(d["c"], dict)


def test_make_serializable_non_dict():
    lt = [1, 2, 3]
    with pytest.raises(TypeError):
        make_serializable(lt)


def test_dialogs_to_utt_pairs():
    original_dialogs = [
        Dialog(turns=[{"speaker": "Human", "text": "Hello"},
                      {"speaker": "System", "text": "Hi there!"},
                      {"speaker": "Human", "text": "How are ya?"}]),
        Dialog(turns=[{"speaker": "Human", "text": "How are you?"},
                      {"speaker": "System", "text": "I'm fine, thanks!"}])
    ]

    utts, utts_next = dialogs_to_utt_pairs(original_dialogs)
    assert utts == ["Hello", "Hi there!"] + ["How are you?"]
    assert utts_next == ["Hi there!", "How are ya?"] + ["I'm fine, thanks!"]

    utts, utts_next = dialogs_to_utt_pairs(original_dialogs, ai_speaker="System")
    assert utts_next == ["Hi there!"] + ["I'm fine, thanks!"]
    assert utts == ["Hello"] + ["How are you?"]
