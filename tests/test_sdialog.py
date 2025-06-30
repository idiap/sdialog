import os

from sdialog.util import config
from sdialog import Dialog, Turn, Event, Instruction, _get_dynamic_version


def test_prompt_paths():
    for path in config["prompts"].values():
        assert os.path.isabs(path)


def test_turn_and_event():
    turn = Turn(speaker="Alice", text="Hello!")
    assert turn.speaker == "Alice"
    assert turn.text == "Hello!"

    event = Event(agent="system", action="utter", text="Hi", timestamp=123)
    assert event.agent == "system"
    assert event.action == "utter"
    assert event.text == "Hi"
    assert event.timestamp == 123


def test_dialog_serialization_and_str():
    turns = [Turn(speaker="A", text="Hi"), Turn(speaker="B", text="Hello")]
    dialog = Dialog(turns=turns)
    json_obj = dialog.json()
    assert dialog.formatVersion == _get_dynamic_version()
    assert isinstance(json_obj, dict)
    assert "turns" in json_obj
    assert dialog.description().startswith("A: Hi")
    assert str(dialog) == dialog.description()


def test_dialog_to_file_and_from_file(tmp_path):
    turns = [Turn(speaker="A", text="Hi"), Turn(speaker="B", text="Hello")]
    dialog = Dialog(turns=turns)
    json_path = tmp_path / "dialog.json"
    txt_path = tmp_path / "dialog.txt"

    dialog.to_file(str(json_path))
    dialog.to_file(str(txt_path))

    loaded_json = Dialog.from_file(str(json_path))
    loaded_txt = Dialog.from_file(str(txt_path))

    assert isinstance(loaded_json, Dialog)
    assert isinstance(loaded_txt, Dialog)
    assert loaded_json.turns[0].speaker == "A"
    assert loaded_txt.turns[1].text == "Hello"


def test_instruction_event():
    event = Event(agent="user", action="instruct", text="Do this", timestamp=1)
    instr = Instruction(text="Do this", events=event)
    assert instr.text == "Do this"
    assert instr.events == event


def test_dialog_print(capsys):
    turns = [Turn(speaker="A", text="Hi"), Turn(speaker="B", text="Hello")]
    dialog = Dialog(turns=turns)
    dialog.print()
    out = capsys.readouterr().out
    assert "Dialogue Begins" in out
    assert "A" in out
    assert "Hi" in out
