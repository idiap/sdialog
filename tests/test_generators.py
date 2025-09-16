import os

from sdialog.generators import DialogGenerator, PersonaDialogGenerator, LLMDialogOutput, Turn
from sdialog.generators import PersonaGenerator
from sdialog.personas import BasePersona, Persona
from sdialog.agents import Agent
from sdialog import Dialog, Context
from sdialog.generators import ContextGenerator  # added


MODEL = "smollm:135m"
PATH_TEST_DATA = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
example_dialog = Dialog(turns=[Turn(speaker="A", text="This is an example!"), Turn(speaker="B", text="Hi!")])


# Patch LLM call
class DummyLLM:
    seed = 0
    num_predict = 1
    temperature = None

    def __init__(self, *a, **kw):
        pass

    def invoke(self, memory):
        return type(
            "Msg", (),
            {"content": "Hi there!"}
        )()

    def __str__(self):
        return "dummy"


# Patch LLM call for structured output
class DummyLLMDialogOutput:
    seed = 0
    num_predict = 1
    temperature = None

    def __init__(self, *a, **kw):
        pass

    def invoke(self, memory):
        return LLMDialogOutput(dialog=[Turn(speaker="A", text="Hi")]).model_dump()

    def __str__(self):
        return "dummy"


# Patch LLM for PersonaGenerator
class DummyPersonaLLM:
    seed = 0
    num_predict = 1
    temperature = None

    def __init__(self, *a, **kw):
        pass

    def with_structured_output(self, output_format):
        return self

    def invoke(self, memory):
        return {"name": "Dummy",
                "age": 30,
                "city": "Unknown",
                "hobby": "Reading",
                "occupation": "Engineer"}

    def __str__(self):
        return "dummy"


# Patch LLM for ContextGenerator
class DummyContextLLM:
    seed = 0
    num_predict = 1
    temperature = None

    def __init__(self, *a, **kw):
        pass

    def with_structured_output(self, output_format):
        return self

    def invoke(self, memory):
        # Assumes Context model has at least location and goals
        return {"location": "Office", "goals": ["Small talk"]}

    def __str__(self):
        return "dummy"


class DummyPersona(BasePersona):
    name: str = None
    age: int = None
    city: str = None
    hobby: str = None
    occupation: str = None


def test_dialog_generator(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    gen = DialogGenerator(dialogue_details="test", model=MODEL)
    dialog = gen()
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(persona_a=persona_a, persona_b=persona_b, model=MODEL)
    dialog = gen()
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator_personas(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(persona_a=persona_a,
                                 speaker_a="Speaker1",
                                 persona_b=persona_b,
                                 speaker_b="Speaker2",
                                 model=MODEL)
    dialog = gen()
    assert "A" == dialog.personas["Speaker1"]["name"]
    assert "B" == dialog.personas["Speaker2"]["name"]


def test_persona_dialog_generator_with_agents(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLM)
    persona_a = Agent(Persona(), "A", model=DummyLLM())
    persona_b = Agent(Persona(), "B", model=DummyLLM())
    gen = PersonaDialogGenerator(persona_a=persona_a, persona_b=persona_b, model=MODEL)
    dialog = gen()
    assert hasattr(dialog, "turns")
    assert "A" in dialog.personas
    assert "B" in dialog.personas


def test_persona_generator_function(monkeypatch):
    def random_age():
        return 42
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    gen = PersonaGenerator(DummyPersona, generated_attributes={"age": random_age})
    persona = gen.generate()
    assert persona.age == 42


def test_persona_generator_function_dependency(monkeypatch):
    def get_hobby(**attributes):
        if attributes["name"].split()[0][-1] == "a":
            return "Party"
        return "Dancying"
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    gen = PersonaGenerator(DummyPersona)
    gen.set(name=["Loco Polaco", "Loca Polaca"], hobby=get_hobby)

    p = gen.generate()
    assert (p.name[-1] == "a" and p.hobby == "Party") or (p.name[-1] == "o" and p.hobby == "Dancying")


def test_persona_generator_list(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    gen = PersonaGenerator(DummyPersona, generated_attributes={"city": ["Paris", "London"]})
    persona = gen.generate()
    assert persona.city in ["Paris", "London"]


def test_persona_generator_fixed_value(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    gen = PersonaGenerator(DummyPersona, generated_attributes={"hobby": "reading"})
    persona = gen.generate()
    assert persona.hobby == "reading"


def test_persona_generator_txt_template(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    txt_path = os.path.join(PATH_TEST_DATA, "occupations.txt")
    gen = PersonaGenerator(DummyPersona, generated_attributes={"occupation": "{{txt:%s}}" % txt_path})
    persona = gen.generate()
    with open(txt_path) as f:
        occupations = f.read().splitlines()
    assert persona.occupation in occupations


def test_persona_generator_csv_template(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    csv_path = os.path.join(PATH_TEST_DATA, "personas.csv")
    gen = PersonaGenerator(DummyPersona)
    gen.set(
        name="{{csv:name:%s}}" % csv_path,
        age="{{20-30}}"
    )
    persona = gen.generate()
    with open(csv_path) as f:
        names = [ln.split(',')[0] for ln in f.read().splitlines() if ln]
    assert persona.name in names


def test_persona_generator_tsv_template(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    csv_path = os.path.join(PATH_TEST_DATA, "personas.tsv")
    gen = PersonaGenerator(DummyPersona)
    gen.set(
        name="{{tsv:name:%s}}" % csv_path,
        age="{{20-30}}"
    )
    persona = gen.generate()
    with open(csv_path) as f:
        names = [ln.split('\t')[0] for ln in f.read().splitlines() if ln]
    assert persona.name in names


def test_persona_generator_range_template():
    gen = PersonaGenerator(DummyPersona, generated_attributes={"age": "{{18-99}}"})
    persona = gen.generate()
    assert 18 <= persona.age <= 99


def test_persona_generator_defaults(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyPersonaLLM)
    gen = PersonaGenerator(DummyPersona)
    persona = gen.generate()
    persona2 = Persona.from_dict(persona.json(), DummyPersona)
    assert persona.name == persona2.name


def test_dialog_generator_example_dialogs(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    gen = DialogGenerator(dialogue_details="test", example_dialogs=[example_dialog])
    assert gen.example_dialogs[0] == example_dialog
    _ = gen()
    assert example_dialog.turns[0].text in gen.messages[0].content


def test_persona_dialog_generator_example_dialogs(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(persona_a, persona_b, example_dialogs=[example_dialog])
    assert gen.example_dialogs[0] == example_dialog
    _ = gen()
    assert example_dialog.turns[0].text in gen.messages[0].content


def test_persona_dialog_generator_with_context_in_constructor(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    ctx = Context(location="Cafe", goals=["Casual chat"])
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(persona_a, persona_b, context=ctx)
    dialog = gen()
    assert "Cafe" in gen.dialogue_details
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator_with_context_at_generate(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    ctx = Context(location="Library", goals=["Study"])
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(persona_a, persona_b)
    dialog = gen(context=ctx)
    assert "Library" in gen.prompt()
    assert hasattr(dialog, "turns")


def test_context_generator_basic(monkeypatch):
    """
    Verify that ContextGenerator generates a Context object via structured LLM output.
    """
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyContextLLM)
    ctx = Context(location="Library", goals=["Study"])
    gen = ContextGenerator(context=ctx)
    ctx = gen.generate()
    assert ctx.location == "Library"
    assert isinstance(ctx.goals, list)


def test_context_generator_attribute_overrides(monkeypatch):
    """
    Verify that set works (list / LLM delegation) and fills values.
    """
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyContextLLM)
    ctx = Context()
    gen = ContextGenerator(context=ctx)
    gen.set(
        location=["Cafe", "Library"],
        # Delegate goals to LLM ("*" means fill via LLM) so DummyContextLLM sets it
        goals="*"
    )
    ctx = gen.generate()
    assert ctx.location in ["Cafe", "Library"]
    assert isinstance(ctx.goals, list) and len(ctx.goals) > 0
