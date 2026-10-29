"""Tests for the STAR dataset loader and the orchestrators it uses, on a tiny fake copy of the dataset."""
import json
import zlib
from typing import Optional

import numpy as np
import pytest
import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sdialog import Dialog, Turn, Instruction
from sdialog.agents import Agent
from sdialog.datasets import STAR
from sdialog.orchestrators import SimpleResponseOrchestrator, InstructionListOrchestrator
from sdialog.personas import Persona


class FixedChatModel(BaseChatModel):
    """Chat model that always replies with the same text."""

    seed: Optional[int] = None
    reply: str = "Hello, what is the name of the apartment?"

    @property
    def _llm_type(self) -> str:
        return "fixed"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


class FakeEncoder:
    """Bag-of-words sentence encoder with the SentenceTransformer methods the orchestrators use."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        embs = []
        for text in ([texts] if single else list(texts)):
            v = np.zeros(32, dtype=np.float32)
            for word in text.lower().replace("?", "").replace(".", "").split():
                v[zlib.crc32(word.encode()) % 32] += 1
            embs.append(v / (np.linalg.norm(v) or 1))
        return np.array(embs)[0] if single else np.array(embs)

    def similarity(self, a, b):
        return torch.tensor(np.atleast_2d(a) @ np.atleast_2d(b).T)


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    monkeypatch.setattr("sdialog.orchestrators.SentenceTransformer", FakeEncoder)


TASK = "apartment_search"
GRAPH = {"hello": "ask_name", "ask_name": "inform_found"}
RESPONSES = {"hello": "Hello, how can I help?", "ask_name": "What is the name of the {Name:apartment}?",
             "inform_found": "I found {Name} for you.", "out_of_scope": "Sorry, I can't help with that."}
SCENARIO = {"Domains": ["apartment"], "Happy": True, "MultiTask": False, "UserTask": "Find an apartment.",
            "WizardTask": "Help the user.", "WizardCapabilities": [{"Task": TASK, "Domain": "apartment"}]}


@pytest.fixture
def star(tmp_path):
    task_dir = tmp_path / "tasks" / TASK
    task_dir.mkdir(parents=True)
    (task_dir / f"{TASK}.json").write_text(json.dumps({"graph": GRAPH}))
    (task_dir / "responses.json").write_text(json.dumps(RESPONSES))
    (tmp_path / "tasks" / "README.txt").write_text("not a task")
    (tmp_path / "dialogues").mkdir()
    events = [
        {"Agent": "UserGuide", "Action": "instruct", "Text": "Ask for an apartment.", "UnixTime": 0},
        {"Agent": "User", "Action": "utter", "Text": "Hi, I need an apartment.", "UnixTime": 1},
        {"Agent": "Wizard", "Action": "pick_suggestion", "Text": "What is the name of the APARTMENT?",
         "ActionLabel": "ask_name", "UnixTime": 2},
        {"Agent": "UserGuide", "Action": "instruct", "Text": "Say it's Maple.", "UnixTime": 2},
        {"Agent": "User", "Action": "utter", "Text": "Maple.", "UnixTime": 3},
        {"Agent": "Wizard", "Action": "query", "UnixTime": 4},
    ]
    (tmp_path / "dialogues" / "1.json").write_text(json.dumps({"Scenario": SCENARIO, "Events": events}))
    sad = dict(SCENARIO, Happy=False, MultiTask=True, Domains=[""])
    wizard_first = [{"Agent": "Wizard", "Action": "utter", "Text": "Welcome!", "UnixTime": 1}] + events[1:]
    (tmp_path / "dialogues" / "2.json").write_text(json.dumps({"Scenario": sad, "Events": wizard_first}))
    (tmp_path / "dialogues" / "notes.txt").write_text("ignored")
    STAR.set_path(str(tmp_path))
    yield tmp_path
    STAR.set_path(None)


def test_graphs_and_responses(star):
    assert STAR.read_graph(TASK, as_dot=False) == GRAPH
    dot = STAR.read_graph(TASK)
    assert dot.startswith(f"digraph {TASK}") and "hello -> ask_name" in dot
    assert STAR.read_graph({"graph": GRAPH}).startswith("digraph   {")
    responses = STAR.read_graph_responses(TASK, as_dict=True)
    assert responses["ask_name"] == "What is the name of the NAME?" and "out_of_scope" not in responses
    assert json.loads(STAR.read_graph_responses(RESPONSES)) == responses
    assert STAR.get_task_names() == [TASK]


def test_dialogs(star):
    dialog = STAR.get_dialog(1)
    assert isinstance(dialog, Dialog) and dialog.id == 1
    assert [t.speaker for t in dialog.turns] == ["User", "System", "User"]
    assert dialog.events[2].actionLabel == "ask_name"
    assert len(STAR.get_dialogs()) == 2
    assert [d.id for d in STAR.get_dialogs(happy=False)] == [2]
    assert [d.id for d in STAR.get_dialogs(multitask=False, domain="apartment", task_name=TASK)] == [1]
    assert STAR.get_dialogs(task_name="other") == []


def test_dialog_details(star):
    assert STAR.get_dialog_scenario(1) == SCENARIO
    assert STAR.get_dialog_first_turn(1).text == "Hi, I need an apartment."
    assert STAR.get_dialog_first_turn(1, speaker="Wizard").text == "What is the name of the APARTMENT?"
    assert STAR.get_dialog_task_names(1) == [TASK]
    graphs, responses = STAR.get_dialog_graphs_and_responses(1)
    assert graphs == [GRAPH] and "hello" in responses[0]
    assert STAR.get_dialog_user_instructions(1) == {0: "Ask for an apartment.", 1: "Say it's Maple."}
    assert len(STAR.get_dialog_events(1)) == 6


def test_descriptions_and_personas(star):
    scenario, description = STAR.get_dialog_scenario_description(1)
    assert "apartment" in description and "happy path" in description and "only the defined task" in description
    sad = STAR.get_dialog_scenario(2)
    sad_description = STAR.get_scenario_description(sad)
    assert "does NOT follow" in sad_description and "multiple tasks" in sad_description
    user = STAR.get_user_persona_for_scenario(sad)
    assert isinstance(user, Persona) and TASK in user.role  # no domains: tasks are used instead
    assert "not just one" in user.rules
    assert STAR.get_system_persona_for_scenario(scenario).rules == "Help the user."
    assert "PLACEHOLDER" in STAR.get_flowchart_description_for_scenario(scenario)


def test_agents(star):
    system, user = STAR.get_agents_for_scenario(SCENARIO, model_name=FixedChatModel())
    assert system.get_name() == "System" and user.get_name() == "User"
    system, user = STAR.get_agents_from_dialogue(2, model_name=FixedChatModel(), set_first_utterance=True)
    assert system._first_utterances == "Welcome!"
    system, user = STAR.get_agents_from_dialogue(1, model_name=FixedChatModel(), set_first_utterance=True)
    assert system._first_utterances == "Hello, how can I help?"
    system, user = STAR.get_agents_from_dialogue_with_orchestration(1, model_name=FixedChatModel())
    assert any(isinstance(o, SimpleResponseOrchestrator) for o in system._orchestrators)
    assert any(isinstance(o, InstructionListOrchestrator) for o in user._orchestrators)


def _agent(**kwargs):
    return Agent(persona=Persona(name="System"), name="System", model=FixedChatModel(**kwargs))


def test_simple_response_orchestrator_with_actions():
    responses = STAR.read_graph_responses(RESPONSES, as_dict=True)
    orchestrator = SimpleResponseOrchestrator(responses, graph=GRAPH, top_k=2)
    agent = _agent() | orchestrator
    # No previous turn of the agent: the lookahead response is used
    instruction = orchestrator.instruct([Turn(speaker="User", text="Hi")], "Hi")
    assert isinstance(instruction, Instruction)
    assert "Action: ask_name" in instruction.text.split(";")[0]
    assert "Lookahead response" in instruction.events[0].content
    # With a previous turn, the next actions come from the graph
    dialog = [Turn(speaker="System", text="Hello, how can I help?"), Turn(speaker="User", text="An apartment")]
    instruction = orchestrator.instruct(dialog, "An apartment")
    assert "Graph next actions: ask_name" in instruction.events[-1].content
    assert "Action: ask_name" in instruction.text
    assert agent.get_name() == "System"


def test_simple_response_orchestrator_with_list():
    orchestrator = SimpleResponseOrchestrator(["What is the name of the apartment?", "Goodbye"], top_k=1)
    _agent() | orchestrator
    instruction = orchestrator.instruct([], "")
    assert instruction.text.endswith("list: (1) What is the name of the apartment?")


def test_simple_response_orchestrator_top_1_action():
    responses = STAR.read_graph_responses(RESPONSES, as_dict=True)
    orchestrator = SimpleResponseOrchestrator(responses, graph=GRAPH, top_k=1)
    _agent() | orchestrator
    expected = '(1) Action: ask_name. Response: "What is the name of the NAME?"'
    assert orchestrator.instruct([], "").text.endswith(expected)


def test_instruction_list_orchestrator():
    agent = Agent(persona=Persona(name="User"), name="User", model=FixedChatModel())
    by_turn = InstructionListOrchestrator({0: "Say hi", 2: "Say bye"})
    agent | by_turn
    assert by_turn.instruct([], "") == "Say hi"
    assert by_turn.instruct([Turn(speaker="User", text="Hi")], "") is None
    as_list = InstructionListOrchestrator(["First", "Second"], persistent=True)
    agent | as_list
    assert as_list.is_persistent()
    assert as_list.instruct([Turn(speaker="User", text="Hi")], "") == "Second"
    assert as_list.instruct([Turn(speaker="User", text="a"), Turn(speaker="User", text="b")], "") is None
