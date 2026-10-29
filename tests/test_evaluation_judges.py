"""Tests for the LLM judges of sdialog.evaluation, with a fake structured-output model."""
from sdialog import Dialog, Turn
from sdialog.personas import Persona
from sdialog.evaluation import LLMJudgePersonaAttributes


class FakeJudgeLLM:
    """Returns the given structured outputs in turn (the last one is repeated) and records the prompts."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


DIALOG = Dialog(turns=[Turn(speaker="Alice", text="Hi, how are you?"), Turn(speaker="Bob", text="Fine, thanks.")])
OTHER = Dialog(turns=[Turn(speaker="Alice", text="Bye."), Turn(speaker="Bob", text="See you.")])


def test_persona_attributes_judge():
    llm = FakeJudgeLLM({"positive": True})
    judge = LLMJudgePersonaAttributes(persona=Persona(name="Bob", role="barista"), speaker="Bob", model=llm)
    assert judge(DIALOG) == 1
    prompt = llm.prompts[-1]
    assert '"role": "barista"' in prompt and 'speaker "Bob"' in prompt
    assert "the following dialogue" in prompt  # rendered for one dialogue
    judge.judge([DIALOG, OTHER])
    assert "the following dialogues" in llm.prompts[-1]
