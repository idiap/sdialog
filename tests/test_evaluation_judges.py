"""Tests for the LLM judges of sdialog.evaluation, with a fake structured-output model."""
import pytest

from pydantic import ValidationError

from sdialog import Dialog, Turn
from sdialog.personas import Persona
from sdialog.evaluation import (LLMJudgeYesNo, LLMJudgeScore, LLMJudgeRealDialog, LLMJudgeRealDialogLikertScore,
                                LLMJudgeRealDialogScore, LLMJudgeRefusal, LLMJudgePersonaAttributes,
                                FrequencyEvaluator, MeanEvaluator)


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


def test_yes_no_judge():
    llm = FakeJudgeLLM({"positive": True, "reason": "polite"})
    judge = LLMJudgeYesNo("Is the dialogue polite?", reason=True, model=llm)
    output = judge.judge(DIALOG)
    assert output.positive is True and output.reason == "polite"
    assert "Is the dialogue polite?" in llm.prompts[-1] and "Hi, how are you?" in llm.prompts[-1]
    assert "polite" in judge.prompt() and judge.prompt(system=True)
    assert judge(DIALOG) == 1
    assert judge.name == "llm-judge-yes-no"


def test_yes_no_judge_lists():
    judge = LLMJudgeYesNo("Polite?", model=FakeJudgeLLM({"positive": []}, {"positive": [False, True]}))
    assert judge.judge([DIALOG, OTHER]).positive == [False, False]  # empty list: all negative
    assert judge.score(OTHER) == 0  # first of the list
    judge = LLMJudgeYesNo("Polite?", model=FakeJudgeLLM({"positive": []}))
    assert judge.judge(DIALOG).positive is False


def test_score_judge_range():
    judge = LLMJudgeScore("How polite (1-5)?", min_score=1, max_score=5, model=FakeJudgeLLM({"score": 9}))
    with pytest.raises(ValidationError):  # out of the range of the output schema
        judge.judge(DIALOG)
    judge = LLMJudgeScore("How polite, from {{ min_score }} to {{ max_score }}?", min_score=1, max_score=5,
                          model=FakeJudgeLLM({"score": 3}))
    assert judge(DIALOG) == 3 and "How polite, from 1 to 5?" in judge.prompt()
    with pytest.raises(ValueError):
        LLMJudgeScore("How polite?", score_type=str)


def test_score_judge_several_dialogues():
    llm = FakeJudgeLLM({"score": [2, 5], "reason": ["rude", "polite"]})
    judge = LLMJudgeScore("How polite?", min_score=1, max_score=5, model=llm, reason=True)
    output = judge.judge([DIALOG, OTHER])
    assert output.score == [2, 5] and output.reason == ["rude", "polite"]
    assert "a list of scores" in llm.prompts[-1]
    with pytest.raises(ValidationError):  # each score must be in the range
        LLMJudgeScore("How polite?", max_score=5, model=FakeJudgeLLM({"score": [2, 6]})).judge([DIALOG, OTHER])


def test_score_judge_float():
    judge = LLMJudgeScore("How polite?", score_type=float, min_score=0, max_score=1,
                          model=FakeJudgeLLM({"score": 0.5, "reason": "ok"}), reason=True)
    assert judge(DIALOG) == 0.5


@pytest.mark.parametrize("judge_class, output, expected", [
    (LLMJudgeRealDialog, {"positive": False}, 0),
    (LLMJudgeRefusal, {"positive": True}, 1),
    (LLMJudgeRealDialogLikertScore, {"score": 4}, 4),
    (LLMJudgeRealDialogScore, {"score": 7}, 7),
])
def test_builtin_judges(judge_class, output, expected):
    llm = FakeJudgeLLM(output)
    judge = judge_class(model=llm)
    assert judge(DIALOG) == expected
    assert "Hi, how are you?" in llm.prompts[-1]


def test_persona_attributes_judge():
    llm = FakeJudgeLLM({"positive": True})
    judge = LLMJudgePersonaAttributes(persona=Persona(name="Bob", role="barista"), speaker="Bob", model=llm)
    assert judge(DIALOG) == 1
    prompt = llm.prompts[-1]
    assert '"role": "barista"' in prompt and 'speaker "Bob"' in prompt
    assert "the following dialogue" in prompt  # rendered for one dialogue
    judge.judge([DIALOG, OTHER])
    assert "the following dialogues" in llm.prompts[-1]


def test_judges_with_dataset_evaluators():
    judge = LLMJudgeYesNo("Polite?", model=FakeJudgeLLM({"positive": True}, {"positive": False}))
    assert FrequencyEvaluator(judge)([DIALOG, OTHER]) == 0.5
    judge = LLMJudgeScore("Polite?", model=FakeJudgeLLM({"score": 2}, {"score": 4}))
    assert MeanEvaluator(judge)([DIALOG, OTHER]) == 3
