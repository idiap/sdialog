<img src="https://raw.githubusercontent.com/idiap/sdialog/master/docs/_static/logo-banner.png" alt="SDialog Logo" title="SDialog" height="150" />

[![Documentation Status](https://app.readthedocs.org/projects/sdialog/badge/?version=latest)](https://sdialog.readthedocs.io)
[![CI](https://img.shields.io/github/actions/workflow/status/idiap/sdialog/ci.yml?label=CI)](https://github.com/idiap/sdialog/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/idiap/sdialog/graph/badge.svg?token=2210USI8I0)](https://app.codecov.io/gh/idiap/sdialog?displayType=list)
[![PyPI version](https://badge.fury.io/py/sdialog.svg)](https://badge.fury.io/py/sdialog)
[![Downloads](https://static.pepy.tech/badge/sdialog)](https://pepy.tech/project/sdialog)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/idiap/sdialog/)
---

> ⚠️ This library is currently undergoing significant updates as part of the JSALT workshop. As a result, some examples in this README and the tutorials may be outdated. Many new features have been added (including audio generation, evaluation, interpretability modules, personas generators, and more). We will update the README, documentation, and tutorials once the codebase stabilizes and matures. Thank you for your understanding and patience.

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
from sdialog import Persona, PersonaAgent

# Define personas
alice = Persona(name="Alice", role="friendly barista", personality="cheerful and helpful")
bob = Persona(name="Bob", role="customer", personality="curious and polite")

# Create agents
alice_agent = PersonaAgent(persona=alice, name="Alice")
bob_agent = PersonaAgent(persona=bob, name="Bob")

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

## 📄 Description

SDialog is a Python toolkit for synthetic dialogue generation and analysis, designed for research and development with instruction-tuned Large Language Models (LLMs). It provides abstractions for persona-driven, multi-agent dialogue simulation, orchestration for fine-grained control, and scenario management. SDialog enables users to easily define agent personas, simulate realistic conversations, inject dynamic instructions, and manage complex scenarios or datasets (such as STAR). The toolkit supports integration with LLMs via Ollama and LangChain, and offers flexible serialization, export, and analysis utilities for generated dialogues.

## 📖 Documentation

- **[Documentation](https://sdialog.readthedocs.io)** - Full package documentation, including installation, API reference, usage guides, and advanced examples available.
- **[API Reference](https://sdialog.readthedocs.io/en/latest/api/index.html):** See docstrings in the codebase for detailed documentation of all classes and functions.
- **[Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials):** Tutorials for hands-on examples as Jupyter Notebooks.


## 🙏 Acknowledgments

This work was supported by the EU Horizon 2020 project [ELOQUENCE](https://eloquenceai.eu/) (grant number 101070558).

This work was also initially created in preparation for the 2025 Jelinek Memorial Summer Workshop on Speech and Language Technologies ([JSALT 2025](https://jsalt2025.fit.vut.cz/)) as part of the work done by the ["Play your Part" research group](https://jsalt2025.fit.vut.cz/play-your-part).


## 📝 License

MIT License  
Copyright (c) 2025 Idiap Research Institute
