import csv
import os

from sdialog.config import config
from sdialog import Dialog, Turn, Event, Instruction, _get_dynamic_version, Context


PATH_TEST_DATA = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


def test_prompt_paths():
    for path in config["prompts"].values():
        if isinstance(path, str):
            assert os.path.isabs(path)


def test_turn_and_event():
    turn = Turn(speaker="Alice", text="Hello!")
    assert turn.speaker == "Alice"
    assert turn.text == "Hello!"

    event = Event(agent="system", action="utter", content="Hi", timestamp=123)
    assert event.agent == "system"
    assert event.action == "utter"
    assert event.content == "Hi"
    assert event.timestamp == 123


def test_dialog_serialization_and_str():
    turns = [Turn(speaker="A", text="Hi"), Turn(speaker="B", text="Hello")]
    dialog = Dialog(turns=turns)
    json_obj = dialog.json()
    assert dialog.version == _get_dynamic_version()
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
    event = Event(agent="user", action="instruct", content="Do this", timestamp=1)
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


def test_dialog_length():
    turns = [Turn(speaker="A", text="Hi there!"), Turn(speaker="B", text="Hello world, how are you?")]
    dialog = Dialog(turns=turns)
    # turns: 2
    assert dialog.length("turns") == 2
    # words: 2 + 5 = 7
    assert dialog.length("words") == 7
    # minutes: 7/150 ≈ 0.0467, rounded to 1 by default (see implementation)
    assert dialog.length("minutes") == 1


def test_set_llm():
    from sdialog.config import config, llm
    llm("test-model")
    assert config["llm"]["model"] == "test-model"


def test_set_llm_params():
    from sdialog.config import config, llm_params, llm
    llm_params(temperature=0.5, seed=42)
    assert config["llm"]["temperature"] == 0.5
    assert config["llm"]["seed"] == 42
    llm("test-model", temperature=0.3, seed=33)
    assert config["llm"]["temperature"] == 0.3
    assert config["llm"]["seed"] == 33


def test_set_persona_dialog_generator_prompt():
    from sdialog.config import config, set_persona_dialog_generator_prompt
    rel_path = "../prompts/test_persona_dialog.j2"
    set_persona_dialog_generator_prompt(rel_path)
    assert config["prompts"]["persona_dialog_generator"] == rel_path


def test_set_persona_generator_prompt():
    from sdialog.config import config, set_persona_generator_prompt
    rel_path = "../prompts/test_persona.j2"
    set_persona_generator_prompt(rel_path)
    assert config["prompts"]["persona_generator"] == rel_path


def test_set_dialog_generator_prompt():
    from sdialog.config import config, set_dialog_generator_prompt
    rel_path = "../prompts/test_dialog.j2"
    set_dialog_generator_prompt(rel_path)
    assert config["prompts"]["dialog_generator"] == rel_path


def test_set_persona_agent_prompt():
    from sdialog.config import config, set_persona_agent_prompt
    rel_path = "../prompts/test_agent.j2"
    set_persona_agent_prompt(rel_path)
    assert config["prompts"]["persona_agent"] == rel_path


def test_get_dialog_from_csv_file():
    csv_path = os.path.join(PATH_TEST_DATA, "dialog_with_headers.csv")
    dialog = Dialog.from_file(csv_path,
                              csv_speaker_col="speaker",
                              csv_text_col="text")
    assert len(dialog) == 3
    assert dialog.turns[0].speaker == "Alice"
    assert dialog.turns[0].text == "Hello! How are you?"
    assert dialog.turns[1].speaker == "Bob"
    csv_path = os.path.join(PATH_TEST_DATA, "dialog_no_headers.csv")
    dialog = Dialog.from_file(
        csv_path,
        csv_speaker_col=0,
        csv_text_col=1
    )
    assert len(dialog) == 3
    assert dialog.turns[2].speaker == "Alice"
    assert dialog.turns[2].text == "Doing great, thanks for asking."


def test_save_dialog_as_csv_file(tmp_path):
    dialog_path = os.path.join(PATH_TEST_DATA, "dialog_with_headers.csv")
    temp_path = tmp_path / "temporary_dialog_save.csv"

    dialog = Dialog.from_file(str(dialog_path))
    dialog.to_file(str(temp_path))

    with open(temp_path, "r") as reader:
        reader = csv.reader(reader)
        rows = list(reader)
        assert len(rows) == 4
        assert rows[0] == ["speaker", "text"]
        assert rows[1] == ["Alice", "Hello! How are you?"]
        assert rows[2] == ["Bob", "I'm good, thanks! And you?"]
        assert rows[3] == ["Alice", "Doing great, thanks for asking."]


def test_dialog_rename_speaker():
    turns = [Turn(speaker="Alice", text="Hello!"),
             Turn(speaker="Bob", text="Hi Alice!"),
             Turn(speaker="Alice", text="How are you?")]
    events = [Event(agent="Alice", action="utter", content="Hello!", timestamp=1),
              Event(agent="Bob", action="utter", content="Hi Alice!", timestamp=2)]
    dialog = Dialog(turns=turns, events=events)

    # Rename Alice to Carol (case-sensitive)
    dialog.rename_speaker("Alice", "Carol", case_sensitive=True)
    assert all(turn.speaker != "Alice" for turn in dialog.turns)
    assert dialog.turns[0].speaker == "Carol"
    assert dialog.turns[-1].speaker == "Carol"
    assert dialog.events[0].agent == "Carol"

    # Bob should remain unchanged
    assert dialog.turns[1].speaker == "Bob"
    assert dialog.events[1].agent == "Bob"

    # Rename Bob to Dave (case-insensitive)
    dialog.rename_speaker("bob", "Dave")
    assert dialog.turns[1].speaker == "Dave"
    assert dialog.events[1].agent == "Dave"


def test_dialog_clone():
    dialog = Dialog(turns=[Turn(speaker="Alice", text="Hello, Bob!"),
                           Turn(speaker="Bob", text="Hi, Alice! How are you?")])

    dialog_clone = dialog.clone()
    dialog_clone.turns.append(Turn(speaker="Alice", text="I'm fine, thanks!"))

    assert len(dialog) == 2
    assert len(dialog_clone) == 3


def test_context_initialization():
    ctx = Context(location="Office",
                  goals=["Discuss project"],
                  constraints="Be concise",
                  topics=["AI", "ML"],
                  notes="Kickoff")
    assert ctx.location == "Office"
    assert ctx.goals == ["Discuss project"]
    assert ctx.constraints == "Be concise"
    assert ctx.topics == ["AI", "ML"]
    assert ctx.notes == "Kickoff"


def test_context_print(capsys):
    ctx = Context(location="Lab", goals=["Study"])
    ctx.print()
    out = capsys.readouterr().out
    assert "Context" in out
    assert "location" in out.lower()
    assert "Lab" in out
    assert "Study" in out
