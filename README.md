<img src="https://raw.githubusercontent.com/idiap/sdialog/master/docs/_static/logo-banner.png" alt="SDialog Logo" title="SDialog" height="150" />

[![Documentation Status](https://app.readthedocs.org/projects/sdialog/badge/?version=latest)](https://sdialog.readthedocs.io)
[![CI](https://img.shields.io/github/actions/workflow/status/idiap/sdialog/ci.yml?label=CI)](https://github.com/idiap/sdialog/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/idiap/sdialog/graph/badge.svg?token=2210USI8I0)](https://app.codecov.io/gh/idiap/sdialog?displayType=list)
[![PyPI version](https://badge.fury.io/py/sdialog.svg)](https://badge.fury.io/py/sdialog)
[![Downloads](https://static.pepy.tech/badge/sdialog)](https://pepy.tech/project/sdialog)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/idiap/sdialog/)
---

> ⚠️ This library is currently undergoing significant updates as part of the JSALT workshop. As a result, some examples in this README and the tutorials may be outdated. Many new features have been added (including audio generation, evaluation, interpretability modules, persona generators, and more). We will update the README, documentation, and tutorials once the codebase stabilizes and matures. Thank you for your understanding and patience. **If you are interested in this project, we recommend clicking the “watch” icon above to stay informed about future updates and changes.**

**SDialog** is a modular, extensible Python toolkit for synthetic dialogue generation and analysis, designed for research and development with instruction-tuned Large Language Models (LLMs). It enables flexible, persona-driven, multi-agent dialogue simulation, orchestration, and scenario management, making it ideal for building, evaluating, and experimenting with conversational agents.

## 🚀 Motivation

Modern conversational AI research and applications increasingly require high-quality, flexible, and reproducible synthetic dialogues for training, evaluation, and benchmarking. SDialog addresses the need for:

- **Standardization:** Clear definitions for dialogue, persona, and event structures.
- **Abstraction:** Abstract interfaces for both single-agent and multi-agent dialogue generation.
- **Fine-grained Control:** Orchestration to inject instructions, simulate user behaviors, and enforce scenario constraints.
- **LLM Integration:** Seamless integration with instruction-tuned LLMs, prompt management, and memory handling.
- **Scenario and Dataset Management:** Tools for managing complex scenarios, flowcharts, and persona definitions.

## ✨ Features

- **Persona-based Role-Playing:** Define rich agent personas to simulate realistic conversations.
- **Multi-Agent Dialogue:** Generate dialogues between multiple agents, each with their own persona and behavior.
- **Dialogue Orchestration:** Control agent actions and inject instructions dynamically using orchestrators.
- **Scenario Management:** Easily describe and manage dialogue scenarios, including flowcharts and user/system goals.
- **Flexible Serialization:** Export dialogues and events in JSON or plain text for downstream tasks.
- **Integration with LLMs:** Out-of-the-box support for [Ollama](https://ollama.com/) and [LangChain](https://python.langchain.com/), with planned support for HuggingFace models.

## ⚡ Installation

```bash
pip install sdialog
```

> **Note:** You must have [Ollama](https://ollama.com/download) running on your system to use the default LLM integration.
> ```bash
> curl -fsSL https://ollama.com/install.sh | sh
> ```

## 🏁 Quick Start

Define personas, create agents, and generate a dialogue:

```python
from sdialog.personas import Persona, PersonaAgent

# Define personas
alice = Persona(name="Alice", role="friendly barista", personality="cheerful and helpful")
bob = Persona(name="Bob", role="customer", personality="curious and polite")

# Create agents
alice_agent = Agent(persona=alice, name="Alice")
bob_agent = Agent(persona=bob, name="Bob")

# Generate a dialogue
dialog = alice_agent.dialog_with(bob_agent)
dialog.print()
```

## 🎛️ Orchestration Example

Add orchestration to control dialogue length or simulate agent behaviors:

```python
from sdialog.orchestrators import LengthOrchestrator, ChangeMindOrchestrator

length_orch = LengthOrchestrator(min=3, max=6)
mind_orch = ChangeMindOrchestrator(probability=0.5, reasons=["changed plans", "new information"], max_times=1)
alice_agent = alice_agent | length_orch | mind_orch
```

## 📚 STAR Dataset Integration

Work with the STAR dataset for scenario-driven dialogue generation:

```python
from sdialog.datasets import STAR

STAR.set_path("/path/to/star-dataset")

scenario = {
    "Domains": ["banking"],
    "UserTask": "Open a new account",
    "WizardTask": "Assist with account opening",
    "Happy": True,
    "MultiTask": False,
    "WizardCapabilities": [{"Task": "open_account", "Domain": "banking"}]
}

system_agent, user_agent = STAR.get_agents_for_scenario(scenario, "llama2")

dialog = system_agent.dialog_with(user_agent)
dialog.print()
```

## 📖 Documentation

- **[Documentation](https://sdialog.readthedocs.io)** - Full package documentation, including installation, API reference, usage guides, and advanced examples available.
- **[API Reference](https://sdialog.readthedocs.io/en/latest/api/index.html):** See docstrings in the codebase for detailed documentation of all classes and functions.
- **[Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials):** Tutorials for hands-on examples as Jupyter Notebooks.


## :muscle: Contributors :sunglasses::+1:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://scholar.google.com/citations?user=XOD8lrAAAAAJ"><img src="https://avatars.githubusercontent.com/u/12646542?v=4?s=100" width="100px;" alt="Sergio Burdisso"/><br /><sub><b>Sergio Burdisso</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=sergioburdisso" title="Code">💻</a> <a href="#ideas-sergioburdisso" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/idiap/sdialog/commits?author=sergioburdisso" title="Documentation">📖</a> <a href="#tutorial-sergioburdisso" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://linkedin.com/in/yanis-labrak-8a7412145/"><img src="https://avatars.githubusercontent.com/u/19389475?v=4?s=100" width="100px;" alt="Labrak Yanis"/><br /><sub><b>Labrak Yanis</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=qanastek" title="Code">💻</a> <a href="#ideas-qanastek" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SevKod"><img src="https://avatars.githubusercontent.com/u/123748182?v=4?s=100" width="100px;" alt="Séverin"/><br /><sub><b>Séverin</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=SevKod" title="Code">💻</a> <a href="#ideas-SevKod" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tutorial-SevKod" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.ricardmarxer.com"><img src="https://avatars.githubusercontent.com/u/15324?v=4?s=100" width="100px;" alt="Ricard Marxer"/><br /><sub><b>Ricard Marxer</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=rikrd" title="Code">💻</a> <a href="#ideas-rikrd" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thschaaf"><img src="https://avatars.githubusercontent.com/u/42753790?v=4?s=100" width="100px;" alt="Thomas Schaaf"/><br /><sub><b>Thomas Schaaf</b></sub></a><br /><a href="#ideas-thschaaf" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/idiap/sdialog/pulls?q=is%3Apr+reviewed-by%3Athschaaf" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/enderzhangpro"><img src="https://avatars.githubusercontent.com/u/41446535?v=4?s=100" width="100px;" alt="David Liu"/><br /><sub><b>David Liu</b></sub></a><br /><a href="https://github.com/idiap/sdialog/commits?author=enderzhangpro" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ahassoo1"><img src="https://avatars.githubusercontent.com/u/46629954?v=4?s=100" width="100px;" alt="ahassoo1"/><br /><sub><b>ahassoo1</b></sub></a><br /><a href="#ideas-ahassoo1" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!


## 🙏 Acknowledgments

This work was supported by the EU Horizon 2020 project [ELOQUENCE](https://eloquenceai.eu/) (grant number 101070558).

The initial development of this project began in preparation for the 2025 Jelinek Memorial Summer Workshop on Speech and Language Technologies ([JSALT 2025](https://jsalt2025.fit.vut.cz/)). Further improvements and enhancements were made during the Workshop as part of the ["Play your Part" research group](https://jsalt2025.fit.vut.cz/play-your-part).


## 📝 License

MIT License  
Copyright (c) 2025 Idiap Research Institute
