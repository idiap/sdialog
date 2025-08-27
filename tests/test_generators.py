import os

from sdialog.generators import DialogGenerator, PersonaDialogGenerator, LLMDialogOutput, Turn
from sdialog.generators import PersonaGenerator
from sdialog.personas import BasePersona, Persona
from sdialog.agents import Agent
from sdialog import Dialog


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
    gen = PersonaDialogGenerator(persona_a, persona_b, MODEL)
    dialog = gen()
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator_personas(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLMDialogOutput)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(persona_a, persona_b, MODEL)
    dialog = gen()
    assert "A" in dialog.personas
    assert "B" in dialog.personas


def test_persona_dialog_generator_with_agents(monkeypatch):
    monkeypatch.setattr("sdialog.util.ChatOllama", DummyLLM)
    persona_a = Agent(Persona(), "A", DummyLLM())
    persona_b = Agent(Persona(), "B", DummyLLM())
    gen = PersonaDialogGenerator(persona_a, persona_b, MODEL)
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
    gen.set_attribute_generators(name=["Loco Polaco", "Loca Polaca"],
                                 hobby=get_hobby)

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
    gen.set_attribute_generators(
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
    gen.set_attribute_generators(
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
