<img src="https://raw.githubusercontent.com/idiap/sdialog/master/docs/_static/logo-banner.png" alt="SDialog Logo" title="SDialog" height="150" />

[![Documentation Status](https://app.readthedocs.org/projects/sdialog/badge/?version=latest)](https://sdialog.readthedocs.io)
[![CI](https://img.shields.io/github/actions/workflow/status/idiap/sdialog/ci.yml?label=CI)](https://github.com/idiap/sdialog/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/idiap/sdialog/graph/badge.svg?token=2210USI8I0)](https://app.codecov.io/gh/idiap/sdialog?displayType=list)
[![PyPI version](https://badge.fury.io/py/sdialog.svg)](https://badge.fury.io/py/sdialog)
[![Downloads](https://static.pepy.tech/badge/sdialog)](https://pepy.tech/project/sdialog)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idiap/sdialog/)

---
SDialog is an MIT-licensed open-source toolkit for building, simulating, and evaluating LLM-based conversational agents end-to-end. It aim to bridge agent construction → dialog generation → evaluation → (optionally) interpretability in a single reproducible workflow, so you can generate reliable, controllable dialog systems or data at scale.

It standardizes a Dialog schema and offers persona‑driven multi‑agent simulation with LLMs, composable orchestration, built‑in metrics, and mechanistic interpretability.

Quick links: [GitHub](https://github.com/idiap/sdialog) • [Docs](https://sdialog.readthedocs.io) • [API](https://sdialog.readthedocs.io/en/latest/api/sdialog.html) • [Demo (Colab)](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/demo.ipynb) • [Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials) • [Datasets (HF)](https://huggingface.co/datasets/sdialog) • [Issues](https://github.com/idiap/sdialog/issues)

## ✨ Key features
- Standard dialog schema with JSON import/export _(aiming to standardize dialog dataset formats [with your help 🙏](#project-vision--community-call))_
- Persona‑driven multi‑agent simulation with contexts, tools, and thoughts
- Composable orchestration for precise control over behavior and flow
- Built‑in evaluation (metrics + LLM‑as‑judge) for comparison and iteration
- Native mechanistic interpretability (inspect and steer activations)
- Easy creation of user-defined components by inheriting from base classes (personas, metrics, orchestrators, etc.)
- Interoperability across OpenAI, Hugging Face, Ollama, AWS Bedrock, Google GenAI, Anthropic, and more.

If you are building conversational systems, benchmarking dialog models, producing synthetic training corpora, simulating diverse users to test or probe conversational systems, or analyzing internal model behavior, SDialog provides an end‑to‑end workflow.


## ⚡ Installation

```bash
pip install sdialog
```

> [!IMPORTANT]
> For audio generation (`sdialog.audio` submodule), install with audio dependencies:
> ```bash
> pip install sdialog[audio]
> ```

Alternatively, a ready-to-use Apptainer image (.sif) with SDialog and all dependencies is available on Hugging Face and can be downloaded [here](https://huggingface.co/datasets/sdialog/apptainer/resolve/main/sdialog.sif).

```bash
apptainer exec --nv sdialog.sif python3 -c "import sdialog; print(sdialog.__version__)"
```

> [!NOTE]
> This Apptainer image also has the Ollama server preinstalled.


## 🏁 Quickstart tour

Here's a short, hands‑on example: a support agent helps a customer disputing a double charge. We add a small refund rule and two simple tools, generate three dialogs for evaluation, then serve the agent on port 1333 for Open WebUI or any OpenAI‑compatible client.

```python
import sdialog
from sdialog import Context
from sdialog.agents import Agent
from sdialog.personas import SupportAgent, Customer
from sdialog.orchestrators import SimpleReflexOrchestrator

# First, let's set our preferred default backend:model and parameters
sdialog.config.llm("openai:gpt-4.1", temperature=1, api_key="YOUR_KEY")  # or export OPENAI_API_KEY=YOUR_KEY
# sdialog.config.llm("ollama:qwen3:14b")  # etc.

# Let's define our personas (use built-ins like in this example, or create your own!)
support_persona = SupportAgent(name="Ava", politeness="high", communication_style="friendly")
customer_persona = Customer(name="Riley", issue="double charge", desired_outcome="refund")

# (Optional) Let's define two mock tools (just plain Python functions) for our support agent
def account_verification(user_id):
    """Verify user account by user id."""
    return {"user_id": user_id, "verified": True}
def refund(amount):
    """Process a refund for the given amount."""
    return {"status": "refunded", "amount": amount}

# (Optional) Let's also include a small rule-based orchestrator for our support agent
react_refund = SimpleReflexOrchestrator(
  condition=lambda utt: "refund" in utt.lower(),
  instruction="Follow refund policy; verify account, apologize, refund.",
)

# Now, let's create the agents!
support_agent = Agent(
  persona=support_persona,
  think=True,  # Let's also enable thinking mode
  tools=[account_verification, refund],
  name="Support"
)
simulated_customer = Agent(
  persona=customer_persona,
  first_utterance="Hi!",
  name="Customer"
)

# Since we have one orchestrator, let's attach it to our target agent
support_agent = support_agent | react_refund

# Let's generate 3 dialogs between them! (we can evaluate them later)
# (Optional) Let's also define a concrete conversational context for the agents in these dialogs
web_chat = Context(location="chat", environment="web", circumstances="billing")
for ix in range(3):
  dialog = simulated_customer.dialog_with(support_agent, context=web_chat)  # Generate the dialog
  dialog.to_file(f"dialog_{ix}.json")  # Save it
  dialog.print(all=True)  # And pretty print it with all its events (thoughts, orchestration, etc.)

# Finally, let's serve our support agent to interact with real users (OpenAI-compatible API)
#    Point Open WebUI or any OpenAI-compatible client to: http://localhost:1333
support_agent.serve(port=1333)
```
> [!TIP]
> - Choose your [LLMs and backends freely](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#configuration-layer).
> - Personas and context can be [automatically generated](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#attribute-generators) (e.g. generate different customer profiles!).

> [!NOTE]
> - See ["agents with tools and thoughts" tutorial](https://github.com/idiap/sdialog/blob/main/tutorials/00_overview/7.agents_with_tools_and_thoughts.ipynb) for a more complete example.
> - See [Serving Agents via REST API](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#serving-agents) for more details on server options.

### 🧪 Testing remote systems with simulated users

<details>
<summary>Probe OpenAI‑compatible deployed systems with controllable simulated users and capture dialogs for evaluation.</summary>

You can also use SDialog as a controllable test harness for any OpenAI‑compatible system such as **vLLM**-based ones by role‑playing realistic or adversarial users against your deployed system:

* Black‑box functional checks (Does the system follow instructions? Handle edge cases?)
* Persona / use‑case coverage (Different goals, emotions, domains)
* Regression testing (Run the same persona batch each release; diff dialogs)
* Safety / robustness probing (Angry, confused, or noisy users)
* Automated evaluation (Pipe generated dialogs directly into evaluators - See Evaluation section below)

Core idea: wrap your system as an `Agent` using `openai:` as the prefix of your model name string, talk to it with simulated user `Agent`s, and capture `Dialog`s you can save, diff, and score.

Below is a minimal example where our simulated customer interacts once with your hypothetical remote endpoint:

```python
# Our remote system (your conversational backend exposing an OpenAI-compatible API)
system = Agent(
  model="openai:your/model",  # Model name exposed by your server
  openai_api_base="http://your-endpoint.com:8000/v1",  # Base URL of the service
  openai_api_key="EMPTY",  # Or a real key if required
  name="System"
)

# Let's make our simulated customer talk with the system
dialog = simulated_customer.dialog_with(system)
dialog.to_file("dialog_0.json")
```
</details>

### 💾 Loading and saving dialogs

<details>
<summary>Import, export, and transform dialogs from JSON, text, CSV, or Hugging Face datasets.</summary>

Dialogs are rich objects with helper methods (filter, slice, transform, etc.) that can be easily exported and loaded using different methods:

```python
from sdialog import Dialog

# Load from JSON (generated by SDialog using `to_file()`)
dialog = Dialog.from_file("dialog_0.json")

# Load from HuggingFace Hub datasets
dialogs = Dialog.from_huggingface("sdialog/Primock-57")

# Create from plain text files or strings - perfect for converting existing datasets!
dialog_from_txt = Dialog.from_str("""
Alice: Hello there! How are you today?
Bob: I'm doing great, thanks for asking.
Alice: That's wonderful to hear!
""")
# Or, equivalently if the content is in a txt file
dialog_from_txt = Dialog.from_file("conversation.txt")

# Load from CSV files with custom column names
dialog_from_csv = Dialog.from_file("conversation.csv",
                                   csv_speaker_col="speaker",
                                   csv_text_col="value",)

# All Dialog objects have rich manipulation methods
dialog.filter("Alice").rename_speaker("Alice", "Customer").upper().to_file("processed.json")
avg_words_turn = sum(len(turn) for turn in dialog) / len(dialog)
```

See [Dialog section](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#dialog) in the documentation for more information.
</details>

## 📊 Evaluate and compare

<details>
<summary>Score dialogs with built‑in metrics and LLM judges, and compare datasets with aggregators and plots.</summary>

Dialogs can be evaluated using the different components available inside the `sdialog.evaluation` module.
Use [built‑in metrics](https://sdialog.readthedocs.io/en/latest/api/sdialog.html#module-sdialog.evaluation) (readability, flow, linguistic features, LLM judges) or easily create new ones, then aggregate and compare datasets (sets of dialogs) via `DatasetComparator`.

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
</details>

> [!TIP]
> See [evaluation tutorial](https://github.com/idiap/sdialog/blob/main/tutorials/00_overview/5.evaluation.ipynb).

## 🧠 Mechanistic interpretability

<details>
<summary>Capture per‑token activations and steer models via Inspectors for analysis and interventions.</summary>

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
import torch
anger_direction = torch.load("anger_direction.pt")  # A direction vector (e.g., PCA / difference-in-mean vector)
agent_steered = agent | inspector - anger_direction  # Ablate the anger direction from the target activations

agent_steered("You are an extremely upset assistant")  # Agent "can't get angry anymore" :)
```
</details>

> [!TIP]
> See [the tutorial](https://github.com/idiap/sdialog/blob/main/tutorials/00_overview/6.agent%2Binspector_refusal.ipynb) on using SDialog to remove the refusal capability from LLaMA 3.2.


## 🔊 Audio generation

<details>
<summary>Convert text dialogs to audio conversations with speech synthesis, voice assignment, and acoustic simulation.</summary>

SDialog can transform text dialogs into audio conversations with a simple one-line command. The audio module supports:

* **Text-to-Speech (TTS)**: Multiple TTS engines including Kokoro and HuggingFace models
* **Voice databases**: Automatic or manual voice assignment based on persona attributes (age, gender, language)
* **Acoustic simulation**: Room acoustics simulation for realistic spatial audio
* **Microphone simulation**: Professional microphones simulation from brands like Shure, Sennheiser, and Sony
* **Multiple formats**: Export to WAV, MP3, or FLAC with custom sampling rates
* **Multi-stage pipeline**: Step 1 (tts and concatenate utterances) and Step 2/3 (position based timeline generation and room acoustics)

Generate audio with room acoustics from any dialog with a single line:

```python
from sdialog import Dialog

dialog = Dialog.from_file("my_dialog.json")

# Convert to audio with default settings (HuggingFace TTS)
audio_dialog = dialog.to_audio(perform_room_acoustics=True)
print(audio_dialog.display())

# Or customize the audio generation
audio_dialog = dialog.to_audio(
  perform_room_acoustics=True,
  audio_file_format="mp3",
  re_sampling_rate=16000,
)
print(audio_dialog.display())
```

</details>

> [!TIP]
> - See the [audio tutorials](https://github.com/idiap/sdialog/tree/main/tutorials/01_audio) for examples including acoustic simulation, room generation, and voice databases. Full documentation is available at [Audio Generation](https://sdialog.readthedocs.io/en/latest/sdialog/index.html#audio-generation).
> - You can find examples with more controlability in [this tutorial](https://github.com/idiap/sdialog/blob/main/tutorials/01_audio/2.accoustic_simulation.ipynb).


## 📖 Documentation and tutorials

- [Demo notebook](https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/demo.ipynb)
- [Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials)
- [API reference](https://sdialog.readthedocs.io/en/latest/api/sdialog.html)
- [Documentation](https://sdialog.readthedocs.io)
- Documentation for **AI coding assistants** like Copilot is also available at `https://sdialog.readthedocs.io/en/latest/llm.txt` following the [llm.txt specification](https://llmstxt.org/). In your Copilot chat, simply use:
  ```
  #fetch https://sdialog.readthedocs.io/en/latest/llm.txt

  Your prompt goes here...(e.g. Write a python script using sdialog to have an agent for
  criminal investigation, define its persona, tools, orchestration...)
  ```


## 🌍 Project Vision & Community Call

To accelerate open, rigorous, and reproducible conversational AI research, SDialog invites the community to collaborate and help shape the future of open dialog generation.

### 🤝 How You Can Help

- **🗂️ Dataset Standardization**: Help convert existing dialog datasets to SDialog format. Currently, each dataset stores dialogs in different formats, making cross-dataset analysis and model evaluation challenging. **Converted datasets are made available as Hugging Face datasets** in the [SDialog organization](https://huggingface.co/datasets/sdialog/) for easy access and integration.
- **🔧 Component Development**: Create new personas, orchestrators, evaluators, generators, or backend integrations
- **📊 Evaluation & Benchmarks**: Design new metrics, evaluation frameworks, or comparative studies
- **🧠 Interpretability Research**: Develop new analysis tools, steering methods, or mechanistic insights
- **📖 Documentation & Tutorials**: Improve guides, add examples, or create educational content
- **🐛 Issues & Discussions**: Report bugs, request features, or share research ideas and use cases

> [!NOTE]
> **Example**: Check out [Primock-57](https://huggingface.co/datasets/sdialog/Primock-57), a sample dataset already available in SDialog format on Hugging Face.
> 
> If you have a dialog dataset you'd like to convert to SDialog format, need help with the conversion process, or want to contribute in any other way, please [open an issue](https://github.com/idiap/sdialog/issues) or reach out to us. We're happy to help and collaborate!


## 💪 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome issues, feature requests, and pull requests. If you want to **contribute to the project**, please open an [issue](https://github.com/idiap/sdialog/issues) or submit a PR, and help us make SDialog better 👍.
If you find SDialog useful, please consider starring ⭐ the GitHub repository to support the project and increase its visibility 😄.

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. All-contributors list:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<div style="display:block; width:100%; overflow-x:auto;">
<table style="table-layout:fixed; width:100%; border-collapse:collapse;">
  <tbody>
    <tr>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="https://sergioburdisso.github.io/" target="_blank"><img src="https://avatars.githubusercontent.com/u/12646542?v=4?s=100" width="100px;" alt="Sergio Burdisso"/><br /><sub><b>Sergio Burdisso</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=sergioburdisso" title="Code" target="_blank">💻</a> <a href="#ideas-sergioburdisso" title="Ideas, Planning, & Feedback" target="_blank">🤔</a> <a href="https://github.com/idiap/sdialog/commits?author=sergioburdisso" title="Documentation" target="_blank">📖</a> <a href="#tutorial-sergioburdisso" title="Tutorials" target="_blank">✅</a></td>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="http://linkedin.com/in/yanis-labrak-8a7412145/" target="_blank"><img src="https://avatars.githubusercontent.com/u/19389475?v=4?s=100" width="100px;" alt="Labrak Yanis"/><br /><sub><b>Labrak Yanis</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=qanastek" title="Code" target="_blank">💻</a> <a href="#ideas-qanastek" title="Ideas, Planning, & Feedback" target="_blank">🤔</a></td>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="https://github.com/SevKod" target="_blank"><img src="https://avatars.githubusercontent.com/u/123748182?v=4?s=100" width="100px;" alt="Séverin"/><br /><sub><b>Séverin</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=SevKod" title="Code" target="_blank">💻</a> <a href="#ideas-SevKod" title="Ideas, Planning, & Feedback" target="_blank">🤔</a> <a href="#tutorial-SevKod" title="Tutorials" target="_blank">✅</a></td>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="http://www.ricardmarxer.com" target="_blank"><img src="https://avatars.githubusercontent.com/u/15324?v=4?s=100" width="100px;" alt="Ricard Marxer"/><br /><sub><b>Ricard Marxer</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=rikrd" title="Code" target="_blank">💻</a> <a href="#ideas-rikrd" title="Ideas, Planning, & Feedback" target="_blank">🤔</a></td>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="https://github.com/thschaaf" target="_blank"><img src="https://avatars.githubusercontent.com/u/42753790?v=4?s=100" width="100px;" alt="Thomas Schaaf"/><br /><sub><b>Thomas Schaaf</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=thschaaf" title="Code" target="_blank">💻</a></td>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="https://github.com/enderzhangpro" target="_blank"><img src="https://avatars.githubusercontent.com/u/41446535?v=4?s=100" width="100px;" alt="David Liu"/><br /><sub><b>David Liu</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=enderzhangpro" title="Code" target="_blank">💻</a></td>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="https://github.com/ahassoo1" target="_blank"><img src="https://avatars.githubusercontent.com/u/46629954?v=4?s=100" width="100px;" alt="ahassoo1"/><br /><sub><b>ahassoo1</b></sub></a><br /><a href="#ideas-ahassoo1" title="Ideas, Planning, & Feedback" target="_blank">🤔</a> <a href="https://github.com/idiap/sdialog/commits?author=ahassoo1" title="Code" target="_blank">💻</a></td>
    </tr>
    <tr>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="http://www.cyrta.com" target="_blank"><img src="https://avatars.githubusercontent.com/u/83173?v=4?s=100" width="100px;" alt="Pawel Cyrta"/><br /><sub><b>Pawel Cyrta</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=cyrta" title="Code" target="_blank">💻</a> <a href="#ideas-cyrta" title="Ideas, Planning, & Feedback" target="_blank">🤔</a></td>
      <td style="text-align:center; vertical-align:top; width:14.28%; word-break:break-word; overflow-wrap:anywhere;"><a href="https://github.com/Amyyyyeah" target="_blank"><img src="https://avatars.githubusercontent.com/u/122391422?v=4?s=100" width="100px;" alt="ABCDEFGHIJKL"/><br /><sub><b>ABCDEFGHIJKL</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=Amyyyyeah" title="Code" target="_blank">💻</a></td>
    </tr>
  </tbody>
</table>
</div>
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

This work was supported by the European Union Horizon 2020 project [ELOQUENCE](https://eloquenceai.eu/about/) and originated during the **Johns Hopkins University** ([JSALT 2025](https://jsalt2025.fit.vut.cz/)) workshop, specifically within the ["Play your Part" research group](https://jsalt2025.fit.vut.cz/play-your-part).


## 📝 License

[MIT License](LICENSE)  
Copyright (c) 2025 Idiap Research Institute
