<img src="https://raw.githubusercontent.com/idiap/sdialog/master/docs/_static/logo-banner.png" alt="SDialog Logo" title="SDialog" height="150" />

[![Documentation Status](https://app.readthedocs.org/projects/sdialog/badge/?version=latest)](https://sdialog.readthedocs.io)
[![CI](https://img.shields.io/github/actions/workflow/status/idiap/sdialog/ci.yml?label=CI)](https://github.com/idiap/sdialog/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/idiap/sdialog/graph/badge.svg?token=2210USI8I0)](https://app.codecov.io/gh/idiap/sdialog?displayType=list)
[![PyPI version](https://badge.fury.io/py/sdialog.svg)](https://badge.fury.io/py/sdialog)
[![Downloads](https://static.pepy.tech/badge/sdialog)](https://pepy.tech/project/sdialog)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/idiap/sdialog/)

---
**SDialog** is a modular Python library for dialogue modeling, generation, evaluation, and analysis with LLMs. It provides a standard Dialog format with rich metadata, persona-driven multi-agent simulation, orchestration for fine control, evaluation metrics, and built-in mechanistic interpretability support.

Quick links: [Docs](https://sdialog.readthedocs.io) • [API](https://sdialog.readthedocs.io/en/latest/api/index.html) • [Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials) • [Demo (Colab)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/0.demo.ipynb) • [Issues](https://github.com/idiap/sdialog/issues)

## 🚀 Motivation

**S**ynthetic **dialogue** generation is increasingly central to creating training data, augmenting datasets, stress-testing systems, and simulating both task-oriented and open-domain interactions. Teams need precise control over personas, contexts, tools, and orchestration to cover long-tail scenarios at scale while preserving privacy and reproducibility. Yet dialogue work is fragmented: every dataset has its own format, every project reinvents agents and prompts, and reproducibility is hard.

The purpose of this project is to make synthetic dialogue generation practical—built with and for the community—by enabling:

- **Standardization and reproducibility:** a well-defined schema for Dialog, Personas, Context, Agents, etc., with JSON import/export serialization for auditability, sharing, and benchmarking.
- **Abstractions:** simple, composable building blocks for personas, agents, orchestrators, generators, evaluation, and interpretability.
- **Interoperability:** the same code works with multiple LLM backends (Ollama, HuggingFace, OpenAI, Google Generative AI, AWS, etc.).
- **Controllability:** persona-, context-, and orchestration-driven generation for targeted scenarios and long-tail distribution exploration.
- **Evaluation loop:** built-in metrics and LLM-as-judge interfaces to compare synthetic against reference data and guide iteration.
- **Interpretability and safety:** native mechanistic interpretability to inspect and steer activations/tokens; supports debugging, bias mitigation, and safe behavior adjustments.

See the quick examples below and our [demo notebook](tutorials/0.demo.ipynb) for a simple demo of the core workflow and basic capabilities. For task-focused guides, see the [Tutorials folder](tutorials).

## ⚡ Installation

```bash
pip install sdialog
```

## 🏁 Quick start

Define personas and context, create agents, and generate a dialogue:

```python
from sdialog import Context
from sdialog.agents import Agent
from sdialog.personas import Persona

# Personas (built-ins like Doctor/Patient/Customer are also available)
alice = Persona(name="Alice", role="friendly barista", personality="cheerful and helpful")
bob = Persona(name="Bob", role="customer", personality="curious and polite")

# Optional shared context
ctx = Context(location="Downtown cafe", topics=["coffee", "recommendations"]) 

# Agents
alice_agent = Agent(persona=alice)
bob_agent = Agent(persona=bob)

# Dialogue
dialog = alice_agent.dialog_with(bob_agent, context=ctx)
dialog.print()  # Pretty print the dialog
# dialog.to_file("my_dialog.json")  # Save it as a JSON file
```

Make the same agents talk in a different context:

```python
starship = Context(
  location="Starship",
  environment="futuristic cafeteria",
  objects=[
    "holographic menu board",
    "service droid",
    "zero-g drink dispenser",
  ],
  circumstances="Customer complains the delivered drink isn’t the one ordered"
)

dialog = alice_agent.dialog_with(bob_agent, context=starship)
dialog.print()
```

Check out our [demo notebook](tutorials/0.demo.ipynb) for a simple demo of the core workflow and capabilities (generation, evaluation, and interpretability).

## 🔧 Interoperability

- SDialog supports many backends (Ollama, HuggingFace, OpenAI, Google Generative AI, AWS), specified as a model string: "BACKEND:MODEL", e.g.:
  - "openai:gpt-4.1"
  - "ollama:gemma3:27b"
  - "aws:anthropic.claude-3-5-sonnet-20240620-v1:0"
  - "huggingface:meta-llama/Llama-3.2-3B-Instruct"

Set a global default LLM for all components:

```python
import sdialog

sdialog.config.llm("ollama:qwen3:14b")
```

Optionally pass parameters:

```python
sdialog.config.llm("ollama:qwen3:14b", temperature=0.9)
```

Any parameter supported by the selected backend is allowed, for instance:

```python
sdialog.config.llm(
  "aws:anthropic.claude-3-5-sonnet-20240620-v1:0",
  region_name="us-east-1"
)
```

## 👤 Personas

Personas are lists of attributes that define who an agent is: a structured profile with role, background/expertise, goals, tone, and other metadata that conditions how the agent speaks and acts. In SDialog, personas are first-class, serializable objects used by Agents, Generators, and Orchestrators; they can be built-in or custom.

Built-in personas include `Persona` (generic) and typed ones like `Doctor`, `Patient`, `Customer`, and `SupportAgent`. For example:

```python
from sdialog.personas import Customer, SupportAgent

customer_persona = Customer(customer_id="12345",
                            issue="Cannot log in to my account",
                            anger_level="high")
support_persona = SupportAgent(politeness="high")

# Pretty print the personas
customer_persona.print()
support_persona.print()
```

Define your own persona type (simple Pydantic-style fields):

```python
from pydantic import Field
from sdialog.personas import BasePersona

class Librarian(BasePersona):
  name: str = ""
  expertise: str = Field("", description="Primary subject area")
  personality: str = Field("patient and meticulous", description="Key traits shaping tone and behavior")

lib = Librarian(name="Morgan",
                expertise="history")
lib.print()
```

## 🤖 Agents

Agents are persona-conditioned conversational actors; they take a persona object when created. They can also support hidden thinking and tool use (if the chosen LLM supports it):

```python
from sdialog.agents import Agent

# As example, let's define wwo simple (mock) tools our support agent can call
# 1) Fake RAG-like tool
def get_product_documentation(product: str, model: str) -> dict:
    """Retrieve product documentation for a specific product and model."""
    # In a real tool, query your documentation store and return top-k snippets.
    snippets = [
        f"Overview for {product} {model}",
        f"Troubleshooting guide for {product} {model}",
        f"FAQ for {product} {model}"
    ]
    return {"snippets": snippets}

# 2) Fake verification account tool
def verify_account(customer_id: str) -> dict:
  """Verify customer account and return minimal details."""
  return {"customer_id": customer_id, "exists": True}

support_agent = Agent(persona=support_persona,
                      think=True,  # Enable reasoning
                      tools=[get_product_documentation, verify_account],  # And tools!
                      name="AGENT")
customer = Agent(persona=customer_persona,
                 first_utterance="Hi there!",
                 name="USER")

dialog = customer.dialog_with(support_agent, max_turns=10)
dialog.print()
```

See the [Agents with tools and thoughts tutorial](tutorials/7.agents_with_tools_and_thoughts.ipynb) for details.

## 🧪 Generators (personas, context, dialogues)

Context and personas can be generated using an LLM or custom functions to populate their attributes. For instance, create doctors (whose specialty is cardiology) and patients (whose symptom is chest pain):

```python
from sdialog.personas import Doctor, Patient
from sdialog.generators import PersonaGenerator

# Persona generators (by default fill all unspecified fields via LLM)
doc_gen = PersonaGenerator(Doctor(specialty="Cardiology"))
pat_gen = PersonaGenerator(Patient(symptoms="mild chest pain"))

# New doctor and patient each time `generate()` is called
doctor = doc_gen.generate()
patient = pat_gen.generate()

# Pretty print generated personas
doctor.print()
patient.print()
```

We can then generate a dialogue using Agents as above, or use a single LLM with `PersonaDialogGenerator`:

```python
from sdialog.generators import PersonaDialogGenerator

# Full dialogue generator for the given personas (no agents)
dlg_gen = PersonaDialogGenerator(doctor, patient, dialogue_details="Keep it short and reassuring.")

# Generate a new dialogue (each call returns a new one)
dialog = dlg_gen()

dialog.print()
```

The `PersonaGenerator` takes any persona as input, including user-defined ones:

```python
# Using the Librarian class defined above
lib_gen = PersonaGenerator(Librarian())  # unspecified fields are LLM-filled by default
new_lib = lib_gen.generate()
new_lib.print()
```

Other utilities: `ContextGenerator` to generate contexts, and `Paraphraser` for dataset augmentation.

## 🎛️ Orchestration in one minute

Add simple rules or constraints by composing orchestrators; the `|` operator attaches them to agents.

```python
from sdialog.orchestrators import SimpleReflexOrchestrator, LengthOrchestrator

# Make Alice react if Bob mentions "cupcakes" and keep the chat between 6 and 10 turns
react = SimpleReflexOrchestrator(
  condition=lambda utt: "cupcakes" in utt.lower(),
  instruction="Politely explain cupcake policy and suggest an alternative"
)
keep_length = LengthOrchestrator(min=6, max=10)

alice_agent = alice_agent | react | keep_length

dialog = alice_agent.dialog_with(bob_agent)
dialog.print(orchestration=True)   # show injected instructions/events
```

Define your own orchestrator:

```python
from sdialog.orchestrators import BaseOrchestrator

class EncourageDetailOrchestrator(BaseOrchestrator):
  def instruct(self, dialog, utterance):
    if utterance and len(utterance.split()) < 5:
      return "Add a bit more detail in your next reply."
    return None

alice_agent = alice_agent | EncourageDetailOrchestrator()
```

See the [orchestration tutorial](tutorials/3.multi-agent+orchestrator_generation.ipynb) for more details.

## 📊 Evaluation and analysis

Evaluate and compare generated dialogues against a reference set (e.g., human reference) using built-in metrics and evaluators (LLM-as-judge, linguistic features, dialog-flow), for example:

```python
import sdialog

from sdialog.evaluation import LLMJudgeRealDialog, LinguisticFeatureScore  # scores
from sdialog.evaluation import FrequencyEvaluator, MeanEvaluator           # evaluators
from sdialog.evaluation import DatasetComparator                           # comparator

reference = [...]   # list of reference Dialogs
candidate_a = [...] # list of first candidate Dialogs
candidate_b = [...] # list of second candidate Dialogs

# Instantiate scores
judge = LLMJudgeRealDialog(feedback=True)
flesch = LinguisticFeatureScore(feature="flesch-reading-ease")
gunning = LinguisticFeatureScore(feature="gunning-fog")

# Instantiate comparator with evaluators
comparator = DatasetComparator(evaluators=[
  FrequencyEvaluator(judge, name="Realistic dialog rate"),
  MeanEvaluator(flesch, name="Mean Flesch Reading Ease"),
  MeanEvaluator(gunning, name="Mean Gunning Fog"),
])

# Compare the dialog sets
comparator({
  "reference": reference,
  "candidate_a": candidate_a,
  "candidate_b": candidate_b,
})

# Plot the comparison
comparator.plot()
```

Create your own score by inheriting from `BaseDialogScore` and implementing `score(dialog)`:

```python
from sdialog.core import Dialog
from sdialog.evaluation import BaseDialogScore

# Simple custom metric: turn length
class DialogLength(BaseDialogScore):
    def score(self, dialog: Dialog) -> int:
        return len(dialog)
```

See the [evaluation tutorial](tutorials/5.evaluation.ipynb) and the [demo](tutorials/0.demo.ipynb) for more.

## 🧠 Mechanistic Interpretability

SDialog natively supports mechanistic interpretability. For instance, it provides an `Inspector` class to capture and steer internal activations at specific layers/tokens. This enables per-token inspection and controlled, ethical behavior adjustments.

Observe internal activations:

```python
import sdialog
from sdialog.agents import Agent
from sdialog.interpretability import Inspector

sdialog.config.llm("huggingface:meta-llama/Llama-3.2-3B-Instruct")

agent = Agent(name="Bob")
# Inspect activations in the residual stream at layer 16
inspector = Inspector(target="model.layers.16.post_attention_layernorm")
agent = agent | inspector

agent("How are you?")
act = inspector[0][0].act  # activations for the first generated token
```

Steer the model with a user-provided direction, e.g., remove anger expression:

```python
import torch

# Target all layers of a 28-layer model
targets = []
for i in range(28):
    targets.append(f"model.layers.{i}.post_attention_layernorm")
    targets.append(f"model.layers.{i}.mlp")
    targets.append(f"model.layers.{i}")

intruder = Inspector(target=targets)

anger_direction = torch.load("anger_direction.pt")  # your direction vector
agent_steered = agent | intruder - anger_direction  # ablate the anger direction across layers

agent_steered("You are an extremely upset assistant")  # anger is no longer part of the activation space
```

See tutorials for worked examples: our [demo notebook](tutorials/0.demo.ipynb) (Mechanistic Interpretability example) and the [tutorial to remove refusal capacity from Llama 3](tutorials/6.agent+inspector_refusal.ipynb).

**Notes:** Use these tools for research and safety improvements only; do not attempt to bypass model safety mechanisms.

## 📖 Documentation and tutorials

- Documentation: https://sdialog.readthedocs.io
- API reference: https://sdialog.readthedocs.io/en/latest/api/index.html
- Tutorials (Jupyter): https://github.com/idiap/sdialog/tree/main/tutorials

## 💪 Contributors 😎👍

We welcome issues, feature requests, and pull requests. If you want to add personas, agents, orchestrators, generators, evaluators, or tutorials, please open [an issue](https://github.com/idiap/sdialog/issues) or submit a PR.

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

All-contributors list:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://sergioburdisso.github.io/" target="_blank"><img src="https://avatars.githubusercontent.com/u/12646542?v=4?s=100" width="100px;" alt="Sergio Burdisso"/><br /><sub><b>Sergio Burdisso</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=sergioburdisso" title="Code" target="_blank">💻</a> <a href="#ideas-sergioburdisso" title="Ideas, Planning, & Feedback" target="_blank">🤔</a> <a href="https://github.com/idiap/sdialog/commits?author=sergioburdisso" title="Documentation" target="_blank">📖</a> <a href="#tutorial-sergioburdisso" title="Tutorials" target="_blank">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://linkedin.com/in/yanis-labrak-8a7412145/" target="_blank"><img src="https://avatars.githubusercontent.com/u/19389475?v=4?s=100" width="100px;" alt="Labrak Yanis"/><br /><sub><b>Labrak Yanis</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=qanastek" title="Code" target="_blank">💻</a> <a href="#ideas-qanastek" title="Ideas, Planning, & Feedback" target="_blank">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SevKod" target="_blank"><img src="https://avatars.githubusercontent.com/u/123748182?v=4?s=100" width="100px;" alt="Séverin"/><br /><sub><b>Séverin</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=SevKod" title="Code" target="_blank">💻</a> <a href="#ideas-SevKod" title="Ideas, Planning, & Feedback" target="_blank">🤔</a> <a href="#tutorial-SevKod" title="Tutorials" target="_blank">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.ricardmarxer.com" target="_blank"><img src="https://avatars.githubusercontent.com/u/15324?v=4?s=100" width="100px;" alt="Ricard Marxer"/><br /><sub><b>Ricard Marxer</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=rikrd" title="Code" target="_blank">💻</a> <a href="#ideas-rikrd" title="Ideas, Planning, & Feedback" target="_blank">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thschaaf" target="_blank"><img src="https://avatars.githubusercontent.com/u/42753790?v=4?s=100" width="100px;" alt="Thomas Schaaf"/><br /><sub><b>Thomas Schaaf</b></sub></a><br /><a href="#ideas-thschaaf" title="Ideas, Planning, & Feedback" target="_blank">🤔</a> <a href="https://github.com/idiap/sdialog/commits?author=thschaaf" title="Code" target="_blank">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/enderzhangpro" target="_blank"><img src="https://avatars.githubusercontent.com/u/41446535?v=4?s=100" width="100px;" alt="David Liu"/><br /><sub><b>David Liu</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=enderzhangpro" title="Code" target="_blank">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ahassoo1" target="_blank"><img src="https://avatars.githubusercontent.com/u/46629954?v=4?s=100" width="100px;" alt="ahassoo1"/><br /><sub><b>ahassoo1</b></sub></a><br /><a href="#ideas-ahassoo1" title="Ideas, Planning, & Feedback" target="_blank">🤔</a> <a href="https://github.com/idiap/sdialog/commits?author=ahassoo1" title="Code" target="_blank">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cyrta.com" target="_blank"><img src="https://avatars.githubusercontent.com/u/83173?v=4?s=100" width="100px;" alt="Pawel Cyrta"/><br /><sub><b>Pawel Cyrta</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=cyrta" title="Code" target="_blank">💻</a> <a href="#ideas-cyrta" title="Ideas, Planning, & Feedback" target="_blank">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Amyyyyeah" target="_blank"><img src="https://avatars.githubusercontent.com/u/122391422?v=4?s=100" width="100px;" alt="ABCDEFGHIJKL"/><br /><sub><b>ABCDEFGHIJKL</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=Amyyyyeah" title="Code" target="_blank">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## 🙏 Acknowledgments

This work was supported by the EU Horizon 2020 project [ELOQUENCE](https://eloquenceai.eu/) (grant number 101070558).

The initial development of this project began in preparation for the 2025 Jelinek Memorial Summer Workshop on Speech and Language Technologies ([JSALT 2025](https://jsalt2025.fit.vut.cz/)). Further improvements and enhancements were made during the Workshop as part of the ["Play your Part" research group](https://jsalt2025.fit.vut.cz/play-your-part).

## 📝 License

[MIT License](LICENSE)  
Copyright (c) 2025 Idiap Research Institute
