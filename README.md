<img src="https://raw.githubusercontent.com/idiap/sdialog/master/docs/_static/logo-banner.png" alt="SDialog Logo" title="SDialog" height="150" />

[![Documentation Status](https://app.readthedocs.org/projects/sdialog/badge/?version=latest)](https://sdialog.readthedocs.io)
[![CI](https://img.shields.io/github/actions/workflow/status/idiap/sdialog/ci.yml?label=CI)](https://github.com/idiap/sdialog/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/idiap/sdialog/graph/badge.svg?token=2210USI8I0)](https://app.codecov.io/gh/idiap/sdialog?displayType=list)
[![PyPI version](https://badge.fury.io/py/sdialog.svg)](https://badge.fury.io/py/sdialog)
[![Downloads](https://static.pepy.tech/badge/sdialog)](https://pepy.tech/project/sdialog)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/idiap/sdialog/)

---
SDialog is a modular Python toolkit for synthetic dialog generation, evaluation, and analysis. It standardizes a Dialog schema and offers persona‑driven multi‑agent simulation with LLMs, composable orchestration, built‑in metrics, and mechanistic interpretability—so you can generate reliable, controllable dialog data at scale.

Quick links: [Docs](https://sdialog.readthedocs.io) • [API](https://sdialog.readthedocs.io/en/latest/api/sdialog.html) • [Demo (Colab)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/0.demo.ipynb) • [Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials) • [Datasets (HF)](https://huggingface.co/datasets/sdialog) • [Issues](https://github.com/idiap/sdialog/issues)

## ✨ Key features
- Standard [Dialog](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#dialog) schema with JSON import/export _(aiming to standardize dialog datasets format [with your help 🙏](#-project-vision--community-call))_
- Persona‑driven multi‑agent simulation with contexts, tools, and thoughts
- Composable orchestration for precise control over behavior and flow
- Built‑in evaluation (metrics + LLM‑as‑judge) for comparison and iteration
- Native mechanistic interpretability (inspect and steer activations)
- Easy creation of user-defined components by inheriting from base classes (personas, metrics, orchestrators, etc.)
- Interoperability across OpenAI, HuggingFace, Ollama, AWS, and more

If you are building controlled multi‑agent conversational systems, benchmarking dialog models, producing synthetic training corpora, simulating diverse users to test or probe conversational systems, or analyzing internal model behavior, SDialog provides an end‑to‑end workflow.

## ⚡ Installation

```bash
pip install sdialog
```

## 🏁 Quickstart tour

Here's a short, hands-on example showing personas, agents, a simple rule (orchestrator), and a tool.

```python
import sdialog
from sdialog import Context
from sdialog.agents import Agent
from sdialog.personas import Persona
from sdialog.orchestrators import SimpleReflexOrchestrator

# First, let's set our preferred backend/model and parameters
sdialog.config.llm("openai:gpt-4.1", temperature=0.9)

# Let's define our personas
alice = Persona(name="Alice", role="barista", personality="cheerful")
bob   = Persona(name="Bob", role="customer", personality="curious")

# (Optional) Let's add a concrete conversational context
ctx = Context(
  location="Downtown cafe",
  environment="noisy, aromatic cafe with occasional grinder sounds",
  circumstances="Morning rush hour",
  objects=["espresso machine", "menu board", "tip jar"]
)

# (Optional) Let's add a simple tool (just a plain Python function)
# We'll use a tiny mock function our agent can call as a tool
def lookup_menu(item: str) -> dict:
    return {"item": item, "specials": ["vanilla latte", "cold brew"]}

# (Optional) Let's include a small rule-based orchestrator
react = SimpleReflexOrchestrator(
    condition=lambda utt: "decaf" in utt.lower(),
    instruction="Explain decaf options and suggest one."
)

# Now we create the agents
barista = Agent(persona=alice, tools=[lookup_menu])
customer = Agent(persona=bob, first_utterance="Hi!")

# (Optional) We can attach orchestrators to an agent using pipe-like composition
barista = barista | react

# Let's generate three dialogs!
for ix in range(3):
    dialog = customer.dialog_with(barista, context=ctx)
    dialog.print(orchestration=True)
    dialog.to_file(f"dialog_{ix}.json")
```
> [!NOTE]
> - See [orchestration tutorial](https://github.com/idiap/sdialog/blob/main/tutorials/3.multi-agent%2Borchestrator_generation.ipynb) and [agents with tools and thoughts](https://github.com/idiap/sdialog/blob/main/tutorials/7.agents_with_tools_and_thoughts.ipynb).
> - Dialogs are [rich objects](https://sdialog.readthedocs.io/en/latest/api/sdialog.html#sdialog.Dialog) with helper methods (filter, slice, transform, etc.) that can be easily exported and loaded.
> - Next: see [Loading and saving dialogs](#loading-and-saving-dialogs) and [Auto-generating personas and contexts](#auto-generating-personas-and-contexts) for persistence and controlled diversity.

### 💾 Loading and saving dialogs

[Dialog](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#dialog)s are JSON‑serializable and can be created from multiple formats. After generating one you can persist it, then reload later for evaluation, transformation, or mixing with real data.

```python
from sdialog import Dialog

# Load from JSON (generated by SDialog)
dialog = Dialog.from_file("dialog_0.json")

# Create from plain text files or strings - perfect for converting existing datasets!
dialog_from_txt = Dialog.from_str("""
Alice: Hello there! How are you today?
Bob: I'm doing great, thanks for asking.
Alice: That's wonderful to hear!
""")
# Or, equivalently if the content is in a txt file
dialog_from_txt = Dialog.from_file("conversation.txt")

# Load from CSV files with custom templates
dialog_from_csv = Dialog.from_file("conversation.csv",
                                   csv_speaker_col="speaker",
                                   csv_text_col="value",)

# All Dialog objects have rich manipulation methods
dialog.filter("Alice").rename_speaker("Alice", "Customer").upper().to_file("processed.json")
avg_words_turn = sum(len(turn) for turn in dialog) / len(dialog)
```

### 🧬 Auto-generating personas and contexts

Use [generators](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#attribute-generators) to fill in (or selectively control) persona/context attributes using LLMs or other data sources (functions, CSV files, inline prompts). The `.set()` method lets you override how individual attributes are produced.

```python
from sdialog.personas import Doctor, Patient
from sdialog.generators import PersonaGenerator, ContextGenerator
from sdialog import Context

# By default, unspecified attributes are LLM generated
doc = PersonaGenerator(Doctor(specialty="Cardiology")).generate()
pat = PersonaGenerator(Patient(symptoms="chest pain")).generate()

# Optionally specify generation sources per attribute
ctx_gen = ContextGenerator(Context(location="emergency room"))
ctx_gen.set(
  objects=get_random_object,                        # user-defined function
  circumstances="{csv:circumstance:./data/circumstances.csv}",  # CSV file values
  goals="{llm:Suggest a realistic goal for the context}"         # targeted LLM instruction
)
ctx = ctx_gen.generate()
```

> [!TIP]
> 🕹️ 👉 Try the [demo notebook](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/0.demo.ipynb) to experiment with generators.

### 🧪 Testing remote systems with simulated users

SDialog can also easily act as a controllable test harness for any (OpenAI‑compatible) conversational backend. Create realistic or adversarial user personas to role‑play against your deployed system:

* Black‑box functional checks (Does the system follow instructions? Handle edge cases?)
* Persona / use‑case coverage (Different goals, emotions, domains)
* Regression testing (Run the same persona batch each release; diff dialogs)
* Safety / robustness probing (Angry, confused, or noisy users)
* Automated evaluation (Pipe generated dialogs directly into evaluators below)

Core idea: your remote system is wrapped as an `Agent`; simulated users are `Agent`s with personas producing diverse conversation trajectories, all recorded as `Dialog` objects you can save, diff, and score.

Below is a minimal example where an "angry customer" interacts once with a mock remote endpoint:

```python
# Our remote system (your conversational backend exposing an OpenAI-compatible API)
system = Agent(
  model="my/super-llm",  # Model name exposed by your server
  openai_api_base="http://my-endpoint.com:8000/v1",  # Base URL of the service
  openai_api_key="EMPTY",  # Or a real key if required
  name="System"
)

# Let's manually define one (minimal) synthetic customer persona
angry_customer = Customer(
  name="Riley",
  issue="Billing error on last invoice",
  issue_description="Charged twice for the same month",
  anger_level="high",
  times_called=3,
)

simulated_customer = Agent(persona=angry_customer, name="Customer")

# Let's make the system talk to our simulated customer once
dialog = system.dialog_with(simulated_customer)
dialog.to_file("dialog_0.json")
```

Next, evaluate these dialogs or orchestrate agents with more complex flows using rule/LLM hybrid orchestrators (see [tutorials 3 & 7](https://github.com/idiap/sdialog/tree/main/tutorials)).

## 📊 Evaluate and compare

Use [built‑in metrics](https://sdialog.readthedocs.io/en/latest/api/sdialog.html#module-sdialog.evaluation) (readability, flow, linguistic features, LLM judges) or easily create new ones, then aggregate and compare datasets via `DatasetComparator`.

```python
from sdialog.evaluation import LLMJudgeRealDialog, LinguisticFeatureScore
from sdialog.evaluation import FrequencyEvaluator, MeanEvaluator
from sdialog.evaluation import DatasetComparator

reference = [...]   # list[Dialog]
candidate = [...]   # list[Dialog]

judge  = LLMJudgeRealDialog()
flesch = LinguisticFeatureScore(feature="flesch-reading-ease")

comparator = DatasetComparator([
  FrequencyEvaluator(judge, name="Realistic dialog rate"),
  MeanEvaluator(flesch, name="Mean Flesch Reading Ease"),
])

results = comparator({"reference": reference, "candidate": candidate})

# Plot results for each evaluator
comparator.plot()
```
> [!TIP]
> See [evaluation tutorial](https://github.com/idiap/sdialog/blob/main/tutorials/5.evaluation.ipynb).

## 🧠 Mechanistic interpretability

Attach Inspectors to capture per‑token activations and optionally steer (add/ablate directions) to analyze or intervene in model behavior.

```python
import sdialog
from sdialog.interpretability import Inspector
from sdialog.agents import Agent

sdialog.config.llm("huggingface:meta-llama/Llama-3.2-3B-Instruct")

agent = Agent(name="Bob")
inspector = Inspector(target="model.layers.16.post_attention_layernorm")
agent = agent | inspector

agent("How are you?")
agent("Cool!")

# Let's get the last response's first token activation vector!
act = inspector[-1][0].act # [response index][token index]
```

Steering intervention (subtracting a direction):
```python
anger_direction = torch.load("anger_direction.pt")  # A direction vector (e.g., PCA / difference-in-mean vector)
agent_steered = agent | inspector - anger_direction  # Ablate the anger direction from the target activations

agent_steered("You are an extremely upset assistant")  # Agent "can't get angry anymore" :)
```
> [!TIP]
> See [the tutorial](https://github.com/idiap/sdialog/blob/main/tutorials/6.agent%2Binspector_refusal.ipynb) on using SDialog to remove the refusal capability from LLaMA 3.2.

## 🔧 Interoperability

Many [backends supported](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#configuration-layer), just use `"BACKEND:MODEL"` string format to either set a global default LLM for all components or pass one to each component:

```python
import sdialog

# Change the default global LLM
sdialog.config.llm("ollama:qwen3:14b")
# Any argument supported by the chosen backend/model can also be given, for example
sdialog.config.llm("ollama:qwen3:14b",
                   temperature=0.7,
                   base_url="https://my-ollama-endpoint.com:123")  # Remote Ollama server
```
Any LLM-powered component can also take a specific model and its parameters as argument, to overwrite the default one:
```python
from sdialog.agents import Agent

my_agent = Agent(model="amazon:anthropic.claude-3-5-sonnet-20240620-v1:0",
                 region_name="us-east-1")
```

## 📖 Documentation and tutorials

- [Demo notebook](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/0.demo.ipynb)
- [Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials)
- [API reference](https://sdialog.readthedocs.io/en/latest/api/sdialog.html)
- [Documentation](https://sdialog.readthedocs.io)
- [LLM-friendly docs](https://sdialog.readthedocs.io/en/latest/llm.txt) for AI coding assistants (**GitHub Copilot**, etc.) following the [llm.txt specification](https://llmstxt.org/), in your chat use:
  ```
  #fetch https://sdialog.readthedocs.io/en/latest/llm.txt
  Your prompt to use sdialog here...
  ```

## 🌍 Project Vision & Community Call

To accelerate open, rigorous, and reproducible conversational AI research, SDialog invites the community to collaborate and help shape the future of open dialogue generation.

### 🤝 How You Can Help

Contributions of any size are welcome and help shape the future of open dialogue generation:

- **🗂️ Dataset Standardization**: Help convert existing dialogue datasets to SDialog format. Currently, each dataset stores dialogues in different formats, making cross-dataset analysis and model evaluation challenging. **Converted datasets are made available as Hugging Face datasets** in the [SDialog organization](https://huggingface.co/datasets/sdialog/) for easy access and integration.
- **🔧 Component Development**: Create new personas, orchestrators, evaluators, generators, or backend integrations
- **📊 Evaluation & Benchmarks**: Design new metrics, evaluation frameworks, or comparative studies
- **🧠 Interpretability Research**: Develop new analysis tools, steering methods, or mechanistic insights
- **📖 Documentation & Tutorials**: Improve guides, add examples, or create educational content
- **🐛 Issues & Discussions**: Report bugs, request features, or share research ideas and use cases

> [!NOTE]
> **Example**: Check out [Primock-57](https://huggingface.co/datasets/sdialog/Primock-57), a sample dataset already available in SDialog format on Hugging Face.
> 
> If you have a dialogue dataset you'd like to convert to SDialog format, need help with the conversion process, or want to contribute in any other way, please [open an issue](https://github.com/idiap/sdialog/issues) or reach out to us. We're happy to help and collaborate!

### 💪 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome issues, feature requests, and pull requests. If you want to **contribute to the project**, please open an [issue](https://github.com/idiap/sdialog/issues) or submit a PR, and help us make SDialog better 👍

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. All-contributors list:

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thschaaf" target="_blank"><img src="https://avatars.githubusercontent.com/u/42753790?v=4?s=100" width="100px;" alt="Thomas Schaaf"/><br /><sub><b>Thomas Schaaf</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=thschaaf" title="Code" target="_blank">💻</a></td>
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

<!-- ## 📚 Citation

If you use SDialog in academic work, please cite:
```bibtex
@misc{sdialog2025,
  title  = {SDialog: A Toolkit for Synthetic Dialog Generation, Evaluation, and Interpretability},
  author = {Contributors of the SDialog Project},
  year   = {2025},
  url    = {https://github.com/idiap/sdialog}
}
``` -->

## 🙏 Acknowledgments

This work was supported by the European Union Horizon 2020 project [ELOQUENCE](https://eloquenceai.eu/about/) (grant number 101070558).

The initial development of this project began in preparation for the 2025 Jelinek Memorial Summer Workshop on Speech and Language Technologies ([JSALT 2025](https://jsalt2025.fit.vut.cz/)) as part of the ["Play your Part" research group](https://jsalt2025.fit.vut.cz/play-your-part).

## 📝 License

[MIT License](LICENSE)  
Copyright (c) 2025 Idiap Research Institute
