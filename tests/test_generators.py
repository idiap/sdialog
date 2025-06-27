import os
import json

from sdialog.generators import DialogGenerator, PersonaDialogGenerator, LLMDialogOutput, Turn
from sdialog.personas import Persona, PersonaAgent
from sdialog.generators import PersonaGenerator
from sdialog.personas import BasePersona


MODEL = "smollm:135m"
PATH_TEST_DATA = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


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


# Patch LLM for PersonaGenerator
class DummyPersonaLLM:
    seed = 0
    num_predict = 1

    def __init__(self, *a, **kw):
        pass

    def invoke(self, memory):
        return type(
            "Msg", (),
            {"content": json.dumps({
                "name": "Dummy",
                "age": 30,
                "city": "Unknown",
                "hobby": "Reading",
                "occupation": "Engineer"
            })}
        )()

    def __str__(self):
        return "dummy"


class DummyPersona(BasePersona):
    name: str = None
    age: int = None
    city: str = None
    hobby: str = None
    occupation: str = None


def test_dialog_generator(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    gen = DialogGenerator(MODEL, dialogue_details="test")
    dialog = gen()
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(MODEL, persona_a, persona_b)
    dialog = gen()
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator_personas(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(MODEL, persona_a, persona_b)
    dialog = gen()
    assert "A" in dialog.personas
    assert "B" in dialog.personas


def test_persona_dialog_generator_with_agents(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    persona_a = PersonaAgent(DummyLLM(), name="A")
    persona_b = PersonaAgent(DummyLLM(), name="B")
    gen = PersonaDialogGenerator(MODEL, persona_a, persona_b)
    dialog = gen()
    assert hasattr(dialog, "turns")
    assert "A" in dialog.personas
    assert "B" in dialog.personas


def test_persona_generator_function():
    def random_age():
        return 42
    gen = PersonaGenerator(DummyPersona, default_attributes={"age": random_age})
    persona = gen.generate()
    assert persona.age == 42


def test_persona_generator_function_dependency():
    def get_hobby(**attributes):
        if attributes["name"].split()[0][-1] == "a":
            return "Party"
        return "Dancying"
    gen = PersonaGenerator(DummyPersona)
    gen.set_random_attributes(name=["Loco Polaco", "Loca Polaca"],
                              hobby=get_hobby)

    p = gen.generate()
    assert (p.name[-1] == "a" and p.hobby == "Party") or (p.name[-1] == "o" and p.hobby == "Dancying")


def test_persona_generator_list():
    gen = PersonaGenerator(DummyPersona, default_attributes={"city": ["Paris", "London"]})
    persona = gen.generate()
    assert persona.city in ["Paris", "London"]


def test_persona_generator_fixed_value():
    gen = PersonaGenerator(DummyPersona, default_attributes={"hobby": "reading"})
    persona = gen.generate()
    assert persona.hobby == "reading"


def test_persona_generator_txt_template(monkeypatch):
    txt_path = os.path.join(PATH_TEST_DATA, "occupations.txt")
    gen = PersonaGenerator(DummyPersona, default_attributes={"occupation": "{{txt:%s}}" % txt_path})
    persona = gen.generate()
    with open(txt_path) as f:
        occupations = f.read().splitlines()
    assert persona.occupation in occupations


def test_persona_generator_csv_template(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyPersonaLLM)
    csv_path = os.path.join(PATH_TEST_DATA, "personas.csv")
    gen = PersonaGenerator(DummyPersona)
    gen.set_random_attributes(
        name="{{csv:name:%s}}" % csv_path,
        age="{{20-30}}"
    )
    persona = gen.generate()
    with open(csv_path) as f:
        names = [ln.split(',')[0] for ln in f.read().splitlines() if ln]
    assert persona.name in names


def test_persona_generator_tsv_template(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyPersonaLLM)
    csv_path = os.path.join(PATH_TEST_DATA, "personas.tsv")
    gen = PersonaGenerator(DummyPersona)
    gen.set_random_attributes(
        name="{{tsv:name:%s}}" % csv_path,
        age="{{20-30}}"
    )
    persona = gen.generate()
    with open(csv_path) as f:
        names = [ln.split('\t')[0] for ln in f.read().splitlines() if ln]
    assert persona.name in names


def test_persona_generator_range_template():
    gen = PersonaGenerator(DummyPersona, default_attributes={"age": "{{18-99}}"})
    persona = gen.generate()
    assert 18 <= persona.age <= 99


def test_persona_generator_defaults(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyPersonaLLM)
    gen = PersonaGenerator(DummyPersona)
    persona = gen.generate()
    assert persona == DummyPersona.model_validate({
        "name": "Dummy",
        "age": 30,
        "city": "Unknown",
        "hobby": "Reading",
        "occupation": "Engineer"
    })
