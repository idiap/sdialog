"""More tests for sdialog.generators: Paraphraser and personas with required fields (with fake LLMs)."""
from typing import Dict, List, Optional, Tuple, Union

import pytest

from sdialog import Dialog, Turn
from sdialog.generators import Paraphraser, PersonaGenerator
from sdialog.personas import BasePersona


class FakeStructuredLLM:
    """Returns the given structured outputs in turn (the last one is repeated) and records the prompts."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


DIALOG = Dialog(turns=[Turn(speaker="Agent", text="hi"), Turn(speaker="Customer", text="my order is late"),
                       Turn(speaker="Agent", text="checking")])


def test_paraphraser_whole_dialog():
    turns = [{"speaker": "Agent", "text": "Hello!"}, {"speaker": "Customer", "text": "My order is delayed."},
             {"speaker": "Agent", "text": "Let me check."}]
    llm = FakeStructuredLLM({"dialog": turns})
    paraphraser = Paraphraser(model=llm, extra_instructions="Be formal")
    new = paraphraser(DIALOG, seed=1)
    assert [t.text for t in new.turns] == ["Hello!", "My order is delayed.", "Let me check."]
    assert DIALOG.turns[0].text == "hi"  # the original is not modified
    assert new.events is None and new.timestamp
    assert "Be formal" in llm.prompts[-1]
    assert "{{ extra_instructions }}" in paraphraser.prompt() and "<Template" not in paraphraser.prompt()


def test_paraphraser_target_speaker():
    llm = FakeStructuredLLM({"dialog": [{"speaker": "Agent", "text": "Hello!"},
                                        {"speaker": "Agent", "text": "One sec."}]})
    new = Paraphraser(model=llm, target_speaker="agent")(DIALOG)
    assert [t.text for t in new.turns] == ["Hello!", "my order is late", "One sec."]
    assert "my order is late" not in llm.prompts[-1]  # only the target speaker's turns are sent


def test_paraphraser_turn_by_turn():
    llm = FakeStructuredLLM({"speaker": "Agent", "text": "Hello!"}, {"speaker": "Agent", "text": "One sec."})
    new = Paraphraser(model=llm, turn_by_turn=True)(DIALOG, target_speaker="Agent")
    assert [t.text for t in new.turns] == ["Hello!", "my order is late", "One sec."]
    assert len(llm.prompts) == 2


def test_paraphraser_wrong_number_of_turns():
    llm = FakeStructuredLLM({"dialog": [{"speaker": "Agent", "text": "Hello!"}]})
    with pytest.raises(ValueError, match="Number of turns"):
        Paraphraser(model=llm)(DIALOG)
    assert len(Paraphraser(model=llm)(DIALOG, raise_on_validation_failure=False)) == 1


class RequiredPersona(BasePersona):
    name: str
    age: int
    height: float
    married: bool
    hobbies: List[str]
    scores: Dict[str, int]
    pair: Tuple[int, int]
    nickname: Optional[str]
    code: Union[int, str]
    city: str = "Martigny"


def test_persona_generator_with_required_fields():
    output = {"name": "Ana", "age": 30, "height": 1.7, "married": False, "hobbies": ["chess"], "scores": {},
              "pair": [1, 2], "nickname": None, "code": 7, "city": "Bern"}
    generator = PersonaGenerator(RequiredPersona, model=FakeStructuredLLM(output))
    persona = generator.generate(seed=0)
    assert persona.model_dump() == {**output, "pair": (1, 2), "city": "Martigny"}  # the default city is kept
