from sdialog.personas import PersonaAgent
from sdialog.personas import Persona, ExtendedPersona, Doctor, Patient, PersonaMetadata
from sdialog.generators import LLMDialogOutput, Turn
from sdialog import Dialog

MODEL = "smollm:135m"


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


def test_persona_description_and_json():
    p = Persona(name="Test", role="tester")
    desc = p.description()
    assert "Test" in desc
    js = p.json()
    assert isinstance(js, dict)
    js_str = p.json(string=True)
    assert isinstance(js_str, str)


def test_persona_fields():
    p = Persona(name="Alice", role="barista", background="Cafe")
    assert p.name == "Alice"
    assert p.role == "barista"
    assert p.background == "Cafe"


def test_persona_agent_init(monkeypatch):
    persona = Persona(name="Alice")
    agent = PersonaAgent(DummyLLM(), persona=persona, name="Alice")
    assert agent.get_name() == "Alice"
    assert "role" in agent.get_prompt().lower()
    agent.set_first_utterances("Hi!")
    assert agent.first_utterances == "Hi!"
    agent.clear_orchestrators()
    agent.reset(seed=42)


def test_persona_and_json():
    persona = Persona(name="Alice", role="barista", background="Works at a cafe")
    desc = persona.description()
    assert "Alice" in desc
    js = persona.json()
    assert isinstance(js, dict)
    js_str = persona.json(string=True)
    assert isinstance(js_str, str)
    assert "Alice" in js_str


def test_persona_agent_init_and_prompt():
    persona = Persona(name="Alice", role="barista")
    agent = PersonaAgent(MODEL, persona=persona, name="Alice")
    assert agent.get_name() == "Alice"
    prompt = agent.get_prompt()
    assert "role" in prompt.lower()


def test_persona_agent_dialog_with():
    persona1 = Persona(name="A")
    persona2 = Persona(name="B")
    agent1 = PersonaAgent(DummyLLM(), persona=persona1, name="A")
    agent2 = PersonaAgent(DummyLLM(), persona=persona2, name="B")
    dialog = agent1.dialog_with(agent2, max_iterations=2, keep_bar=False)
    assert isinstance(dialog, Dialog)
    assert len(dialog.turns) > 0
    assert "A" in dialog.personas
    assert "B" in dialog.personas


def test_extended_persona_fields_and_description():
    p = ExtendedPersona(
        name="Bob",
        age=40,
        race="Asian",
        gender="male",
        language="English",
        weight="70kg",
        height=175.0,
        occupation="Engineer",
        education="PhD",
        socioeconomic_status="middle",
        interests="AI, robotics",
        hobbies="chess",
        politeness="high",
        forgetfulness="low",
        attentiveness="high",
        communication_style="direct",
        empathy_level="medium",
        political_views="moderate",
        religious_beliefs="agnostic"
    )
    desc = p.description()
    assert "Bob" in desc
    assert "Engineer" in desc
    js = p.json()
    assert isinstance(js, dict)
    assert js["name"] == "Bob"
    assert js["occupation"] == "Engineer"


def test_patient_fields_and_description():
    p = Patient(
        name="Jane",
        age=30,
        symptoms="cough, fever",
        vital_signs="BP 120/80",
        health_literacy="high",
        medical_conditions="asthma",
        medications="inhaler",
        allergies="penicillin",
        family_history="diabetes"
    )
    desc = p.description()
    assert "Jane" in desc
    assert "cough" in desc
    js = p.json()
    assert isinstance(js, dict)
    assert js["symptoms"] == "cough, fever"
    assert js["medical_conditions"] == "asthma"


def test_doctor_fields_and_description():
    d = Doctor(
        name="Dr. Smith",
        age=50,
        specialty="Cardiology",
        years_of_experience=25,
        certifications="Board Certified",
        work_experience="Hospital A, Hospital B"
    )
    desc = d.description()
    assert "Dr. Smith" in desc
    assert "Cardiology" in desc
    js = d.json()
    assert isinstance(js, dict)
    assert js["specialty"] == "Cardiology"
    assert js["years_of_experience"] == 25


def test_persona_to_file_and_from_file(tmp_path):
    """
    Test saving and loading a Persona (BasePersona subclass) to/from file, including metadata.
    """
    persona = Persona(name="Alice", role="barista", background="Works at a cafe")
    file_path = tmp_path / "persona.json"
    persona.to_file(str(file_path))
    loaded = Persona.from_file(str(file_path))
    assert isinstance(loaded, Persona)
    assert loaded.name == "Alice"
    assert loaded.role == "barista"
    assert loaded.background == "Works at a cafe"
    assert hasattr(loaded, "_metadata")
    assert loaded._metadata is not None
    assert loaded._metadata.className == Persona.__name__


def test_extended_persona_to_file_and_from_file(tmp_path):
    """
    Test saving and loading an ExtendedPersona (BasePersona subclass) to/from file, including metadata.
    """
    p = ExtendedPersona(
        name="Bob",
        age=40,
        race="Asian",
        gender="male",
        language="English",
        weight="70kg",
        height=175.0,
        occupation="Engineer",
        education="PhD",
        socioeconomic_status="middle",
        interests="AI, robotics",
        hobbies="chess",
        politeness="high",
        forgetfulness="low",
        attentiveness="high",
        communication_style="direct",
        empathy_level="medium",
        political_views="moderate",
        religious_beliefs="agnostic"
    )
    file_path = tmp_path / "extended_persona.json"
    p.to_file(str(file_path))
    loaded = ExtendedPersona.from_file(str(file_path))
    assert isinstance(loaded, ExtendedPersona)
    assert loaded.name == "Bob"
    assert loaded.occupation == "Engineer"
    assert loaded._metadata is not None
    assert loaded._metadata.className == ExtendedPersona.__name__


def test_persona_clone():
    """
    Test the .clone() method for Persona and ExtendedPersona, ensuring deep copy and metadata preservation.
    """
    persona = Persona(name="Alice", role="barista", background="Works at a cafe")
    clone = persona.clone()
    assert isinstance(clone, Persona)
    assert clone is not persona
    assert clone.name == persona.name
    assert clone.role == persona.role
    assert clone.background == persona.background
    assert hasattr(clone, "_metadata")
    assert clone._metadata is not None
    assert clone._metadata.className == Persona.__name__

    # ExtendedPersona
    p = ExtendedPersona(
        name="Bob",
        age=40,
        occupation="Engineer"
    )
    p._metadata = None
    clone2 = p.clone()
    assert isinstance(clone2, ExtendedPersona)
    assert clone2 is not p
    assert clone2.name == p.name
    assert clone2.occupation == p.occupation
    assert clone2._metadata is not None
    assert clone2._metadata.className == ExtendedPersona.__name__


# def test_persona_clone_parent_id():
#     """
#     Test that .clone() sets the clone's _metadata.parentId to the original's _metadata.id.
#     """
#     persona = Persona(name="Alice", role="barista")
#     persona._metadata = PersonaMetadata(id=123)
#     clone = persona.clone()
#     assert clone._metadata is not None
#     assert clone._metadata.parentId == 123


# def test_persona_clone_with_changes():
#     """
#     Test that .clone() can produce a modified clone when attributes are changed after cloning.
#     """
#     persona = Persona(name="Alice", role="barista", background="Works at a cafe")
#     clone = persona.clone(role="manager")
#     assert clone.role == "manager"
#     assert persona.name == clone.name
