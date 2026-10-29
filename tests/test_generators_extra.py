"""More tests for sdialog.generators (with fake LLMs)."""
from typing import Dict, List, Optional, Tuple, Union

from sdialog.generators import PersonaGenerator
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
