"""More tests for the core data model (sdialog.Dialog and Turn): text transforms, file I/O and printing."""
import json
import re

import pytest

from sdialog import Dialog, Turn, Event


def make_dialog():
    return Dialog(id=7, turns=[Turn(speaker="Alice", text="  Hello there  "), Turn(speaker="Bob", text="hi, alice")])


def test_turn_prompt_and_print(capsys):
    turn = Turn(speaker="Alice", text="Hi!")
    assert json.loads(turn.prompt()) == {"speaker": "Alice", "text": "Hi!"}
    turn.print()
    assert "Hi!" in capsys.readouterr().out


def test_indexing():
    dialog = make_dialog()
    assert dialog[0].speaker == "Alice"
    sliced = dialog[1:]
    assert isinstance(sliced, Dialog) and len(sliced) == 1 and len(dialog) == 2
    with pytest.raises(TypeError):
        dialog["a"]


def test_text_transforms():
    dialog = make_dialog()
    copy = dialog.upper(in_place=False)
    assert copy.turns[1].text == "HI, ALICE" and dialog.turns[1].text == "hi, alice"
    assert dialog.strip().turns[0].text == "Hello there"
    assert dialog.strip("Hel").turns[0].text == "o ther"
    dialog = make_dialog().strip()
    assert dialog.lower().turns[0].text == "hello there"
    assert dialog.title().turns[1].text == "Hi, Alice"
    assert dialog.capitalize().turns[0].text == "Hello there"
    assert dialog.replace("e", "E", count=1).turns[0].text == "HEllo there"
    assert dialog.replace("e", "3").turns[0].text == "HEllo th3r3"  # in place: chained with the previous one
    assert dialog.re_sub(r"\d", "#").turns[0].text == "HEllo th#r#"
    assert dialog.re_sub(re.compile("H"), "h", count=1).turns[0].text == "hEllo th#r#"
    assert dialog.re_sub("ALICE", "Ann", flags=re.IGNORECASE).turns[1].text == "hi, Ann"


def test_length_description_and_prompt():
    dialog = make_dialog()
    assert dialog.length("minutes") == 1
    with pytest.raises(ValueError):
        dialog.length("hours")
    assert dialog.description(turn_template="[{speaker}] {text}").splitlines()[1] == "[Bob] hi, alice"
    no_speaker = Dialog(turns=[Turn(text="narration")])
    assert no_speaker.description() == "narration"
    assert no_speaker.description(turn_template="{speaker}>{text}") == ">narration"
    assert json.loads(dialog.prompt())[0]["speaker"] == "Alice"


def test_to_file_and_load(tmp_path):
    dialog = make_dialog()
    with pytest.raises(ValueError):
        dialog.to_file()
    path = str(tmp_path / "sub" / "dialog.json")
    dialog.to_file(path)
    with pytest.raises(FileExistsError):
        dialog.to_file(path, overwrite=False)
    loaded = Dialog.from_file(path)
    assert loaded.turns[1].text == "hi, alice"
    loaded.turns[0].text = "changed"
    loaded.to_file()  # saved back to where it was loaded from
    assert Dialog.from_file(path).turns[0].text == "changed"
    dialog.to_file(str(tmp_path / "dialog.unknown"))  # unknown extension: saved as text
    assert Dialog.from_file(str(tmp_path / "dialog.unknown")).turns[0].speaker == "Alice"
    for ext in ("txt", "csv", "tsv"):
        dialog.to_file(str(tmp_path / f"dialog.{ext}"))
        assert Dialog.from_file(str(tmp_path / f"dialog.{ext}")).turns[1].speaker == "Bob"


def test_from_file_csv_options(tmp_path):
    path = tmp_path / "dialog.csv"
    path.write_text("who,what\nA,Hi\nA,there\nB,\nB,Bye\n")
    dialog = Dialog.from_file(str(path), csv_speaker_col="who", csv_text_col="what",
                              collapse_consecutive_speakers=True)
    assert [(t.speaker, t.text) for t in dialog.turns] == [("A", "Hi\nthere"), ("B", "Bye")]  # empty text skipped
    by_index = Dialog.from_file(str(path), csv_speaker_col=0, csv_text_col=1)
    assert by_index.turns[0].speaker == "who"  # the header is a row when using column indices
    with pytest.raises(ValueError, match="Missing required columns"):
        Dialog.from_file(str(path), csv_speaker_col="speaker", csv_text_col="text")
    with pytest.raises(ValueError, match="both"):
        Dialog.from_file(str(path), csv_speaker_col="who", csv_text_col=1)
    with pytest.raises(ValueError, match="Missing required columns at row"):
        Dialog.from_file(str(path), csv_speaker_col=0, csv_text_col=5)
    with pytest.raises(ValueError, match="Unknown file type"):
        Dialog.from_file(str(path), type="xml")
    bad_txt = tmp_path / "bad.txt"
    bad_txt.write_text("no speaker here")
    with pytest.raises(ValueError, match="bad.txt"):
        Dialog.from_file(str(bad_txt))


def test_from_folder(tmp_path):
    (tmp_path / "b.txt").write_text("A: hi\nB: hello")
    (tmp_path / "a.txt").write_text("A: one\nA: two")
    Dialog(id="json-dialog", turns=[Turn(speaker="C", text="json")]).to_file(str(tmp_path / "c.json"))
    dialogs = Dialog.from_folder(str(tmp_path), collapse_consecutive_speakers=True, collapse_separator=" / ")
    assert [d.id for d in dialogs] == [1, 2, "json-dialog"]
    assert dialogs[0].turns[0].text == "one / two"
    assert [len(d) for d in Dialog.from_file(str(tmp_path), type="txt")] == [2, 2]
    with pytest.raises(ValueError, match="not a directory"):
        Dialog.from_folder(str(tmp_path / "a.txt"))


def test_from_str_templates():
    dialog = Dialog.from_str("[A] hi\n\n[B] hello", template="[{speaker}] {text}", id=3)
    assert dialog.id == 3 and [t.speaker for t in dialog.turns] == ["A", "B"]
    no_speakers = Dialog.from_str("- hi\n- hello\n- bye", template="- {text}", default_speakers=["X", "Y"])
    assert [t.speaker for t in no_speakers.turns] == ["X", "Y", "X"]
    with pytest.raises(ValueError):
        Dialog.from_str("A: hi", default_speakers="X")


def _fake_snapshot_download(files):
    def snapshot_download(repo_id, repo_type, local_dir):
        assert repo_type == "dataset"
        for rel_path, dialog in files.items():
            dialog.to_file(f"{local_dir}/{rel_path}")
    return snapshot_download


def test_from_huggingface(monkeypatch, tmp_path):
    import huggingface_hub
    d1, d2 = make_dialog(), Dialog(turns=[Turn(speaker="A", text="x"), Turn(speaker="A", text="y")])
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_snapshot_download({"data/train/1.json": d1, "data/test/2.json": d2}))
    splits = Dialog.from_huggingface("org/dataset", collapse_consecutive_speakers=True)
    assert set(splits) == {"train", "test"} and splits["test"][0].turns[0].text == "x\ny"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download({"data/1.json": d1}))
    dialogs = Dialog.from_huggingface("org/dataset", local_dir=str(tmp_path / "hf"))
    assert len(dialogs) == 1 and (tmp_path / "hf" / "data" / "1.json").exists()  # local_dir is kept
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download({"other/1.json": d1}))
    with pytest.raises(ValueError, match="not a valid sdialog dataset"):
        Dialog.from_huggingface("org/not-sdialog")


def test_filter_with_events_and_print(capsys):
    dialog = Dialog(id=1, complete=True, seed=3, model={"name": "m"}, scenario={"goal": "x"},
                    turns=[Turn(speaker="Alice", text="hi"), Turn(speaker="Bob", text="hello")],
                    events=[Event(agent="Alice", action="utter", content="hi", timestamp=0),
                            Event(agent="Bob", action="think", content="hmm", timestamp=1),
                            Event(agent="Bob", action="tool", actionLabel="output", content={"r": 1}, timestamp=2),
                            Event(agent="Bob", action="instruct", actionLabel="Orch", content="be nice", timestamp=3),
                            Event(agent="Bob", action="pick_suggestion", content="hello", timestamp=4),
                            Event(agent="Bob", action="request_suggestions", content="list", timestamp=5)])
    bob = dialog.filter("bob")
    assert [t.text for t in bob.turns] == ["hello"] and all(e.agent == "Bob" for e in bob.events)
    dialog.filter("nobody")  # logs an error, returns an empty dialogue
    dialog.print(all=True, scenario=True)
    out = capsys.readouterr().out
    for text in ("thinking", "tool-output", "be nice", "pick_suggestion", "goal", "Dialogue Ends"):
        assert text in out
    Dialog(turns=[Turn(speaker="A", text="hi")], scenario="A plain scenario").print(scenario=True)
    assert "A plain scenario" in capsys.readouterr().out
