from sdialog import Dialog, Context
from sdialog.agents import Agent
from sdialog.personas import Persona
from sdialog.generators import LLMDialogOutput, Turn

example_dialog = Dialog(turns=[Turn(speaker="A", text="This is an example!"), Turn(speaker="B", text="Hi!")])


# Patch LLM call
class DummyLLM:
    seed = 0
    num_predict = 1

    def __init__(self, *a, **kw):
        pass

    def invoke(self, memory):
        return type(
            "Msg", (),
            {"content": LLMDialogOutput(
                dialog=[Turn(speaker="A", text="Hi")]).model_dump_json()}
        )()

    def __str__(self):
        return "dummy"


def test_persona_agent_init(monkeypatch):
    persona = Persona(name="Alice")
    agent = Agent(persona=persona, name="Alice", model=DummyLLM())
    assert agent.get_name() == "Alice"
    assert "role" in agent.prompt().lower()
    agent.set_first_utterances("Hi!")
    assert agent._first_utterances == "Hi!"
    agent.clear_orchestrators()
    agent.reset(seed=42)


def test_persona_agent_init_and_prompt():
    persona = Persona(name="Alice", role="barista")
    agent = Agent(persona, "Alice", model=DummyLLM())
    assert agent.get_name() == "Alice"
    prompt = agent.prompt()
    assert "role" in prompt.lower()


def test_persona_agent_dialog_with():
    persona1 = Persona(name="A")
    persona2 = Persona(name="B")
    agent1 = Agent(persona=persona1, name="A", model=DummyLLM())
    agent2 = Agent(persona2, "B", model=DummyLLM())
    dialog = agent1.dialog_with(agent2, max_turns=4, keep_bar=False)
    assert isinstance(dialog, Dialog)
    assert len(dialog.turns) > 0
    assert "A" in dialog.personas
    assert "B" in dialog.personas


def test_agent_postprocessing_fn():
    persona1 = Persona(name="A")
    persona2 = Persona(name="B")
    agent1 = Agent(persona=persona1, name="A", model=DummyLLM())
    agent2 = Agent(persona2, "B", model=DummyLLM(), postprocess_fn=lambda x: x.upper())
    dialog = agent1.dialog_with(agent2, max_turns=4, keep_bar=False)
    assert dialog.turns[1].text.isupper(), "Postprocessing function did not apply correctly."
    assert not dialog.turns[0].text.isupper(), "Postprocessing function should not have effect."


def test_agent_example_dialogs():
    persona = Persona(name="Alice")
    agent = Agent(persona=persona, name="Alice", model=DummyLLM(), example_dialogs=[example_dialog])
    assert example_dialog.turns[0].text in agent.memory[0].content


def test_agents_dialog_with_context():
    ctx = Context(location="Office", goals=["Discuss project"])
    persona1 = Persona(name="A")
    persona2 = Persona(name="B")
    agent1 = Agent(persona=persona1, name="A", model=DummyLLM(), context=ctx)
    agent2 = Agent(persona=persona2, name="B", model=DummyLLM(), context=ctx)
    dialog = agent1.dialog_with(agent2, max_turns=2)
    assert isinstance(dialog, Dialog)
    assert len(dialog.turns) > 0
    assert agent1._context is ctx
    assert agent2._context is ctx
