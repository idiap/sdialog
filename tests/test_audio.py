import pytest
import numpy as np
from sdialog import Dialog, Turn
from sdialog.audio.audio_turn import AudioTurn
from sdialog.audio.audio_dialog import AudioDialog


@pytest.fixture
def mock_dialog():
    """Pytest fixture for a mock Dialog object."""
    turns = [
        Turn(speaker="Alice", text="Hello Bob"),
        Turn(speaker="Bob", text="Hi Alice")
    ]
    return Dialog(turns=turns, id="test_dialog")


def test_audio_dialog_from_dialog(monkeypatch, mock_dialog):
    """
    Test the creation of an AudioDialog from a Dialog object.
    """
    def mock_from_turn(turn):
        # Directly create an AudioTurn instance to avoid recursion.
        return AudioTurn(speaker=turn.speaker, text=turn.text)

    monkeypatch.setattr(AudioTurn, 'from_turn', mock_from_turn)

    audio_dialog = AudioDialog.from_dialog(mock_dialog)

    assert isinstance(audio_dialog, AudioDialog)
    assert audio_dialog.id == "test_dialog"
    assert len(audio_dialog.turns) == 2
    assert all(isinstance(turn, AudioTurn) for turn in audio_dialog.turns)
    assert audio_dialog.turns[0].speaker == "Alice"
    assert audio_dialog.turns[1].text == "Hi Alice"


def test_audio_dialog_audio_methods():
    """
    Test the set_combined_audio and get_combined_audio methods.
    """
    dialog = Dialog(turns=[])
    audio_dialog = AudioDialog.from_dialog(dialog)

    # Test with None
    audio_dialog.set_combined_audio(None)
    assert audio_dialog.get_combined_audio() is None

    # Test with numpy array
    audio_data = np.random.randn(16000 * 5)
    audio_dialog.set_combined_audio(audio_data)
    assert np.array_equal(audio_dialog.get_combined_audio(), audio_data)
