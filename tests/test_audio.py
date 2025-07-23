import pytest
import numpy as np
from sdialog import Dialog, Turn
from sdialog.audio.audio_turn import AudioTurn
from sdialog.audio.audio_dialog import AudioDialog
from sdialog.audio.audio_events import AudioEvent, Timeline


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


def test_audio_event_creation():
    """Test creation of AudioEvent."""
    event1 = AudioEvent()
    assert event1.label is None
    assert event1.source_file is None
    assert event1.start_time is None
    assert event1.duration is None
    assert event1.role is None

    event2 = AudioEvent(
        label="test_label",
        source_file="test.wav",
        start_time=100,
        duration=1000,
        role="speaker"
    )
    assert event2.label == "test_label"
    assert event2.source_file == "test.wav"
    assert event2.start_time == 100
    assert event2.duration == 1000
    assert event2.role == "speaker"


def test_audio_event_str():
    """Test the __str__ method of AudioEvent."""
    event = AudioEvent(
        label="greeting",
        source_file="hello.wav",
        start_time=50,
        duration=1500,
        role="user"
    )
    expected_str = "greeting user 50 1500 hello.wav"
    assert str(event) == expected_str


def test_timeline_initialization():
    """Test Timeline initialization."""
    timeline = Timeline()
    assert timeline.events == []


def test_timeline_add_event():
    """Test adding an event to a Timeline."""
    timeline = Timeline()
    event = AudioEvent(label="test")
    timeline.add_event(event)
    assert len(timeline.events) == 1
    assert timeline.events[0] == event
