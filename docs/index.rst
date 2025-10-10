.. image:: _static/logo-banner.png
    :target: https://github.com/idiap/sdialog
    :height: 100px
    :align: right


Synthetic Dialogue Generation with SDialog
==========================================

Conversational AI research and applications increasingly rely on high-quality, flexible, and reproducible synthetic dialogues for training, evaluation, and benchmarking. However, generating such dialogues presents several challenges:

- **Standardization:** There is a lack of standard definitions for dialogue, persona, and event structures, making it difficult to compare results across systems or datasets.
- **Abstraction:** Researchers and developers need abstract interfaces for dialogue generation that support both single-agent and multi-agent scenarios, enabling modular experimentation.
- **Fine-grained Control:** Realistic dialogue simulation often requires fine-grained orchestration, such as injecting instructions, simulating user behaviors, or enforcing scenario constraints.
- **LLM Integration:** Leveraging instruction-tuned Large Language Models (LLMs) for dialogue generation requires seamless integration, prompt management, and memory handling.
- **Scenario and Dataset Management:** Managing complex scenarios, flowcharts, and persona definitions is essential for reproducible research and controlled experimentation.


Project Vision & Community Call
-------------------------------

To accelerate open, rigorous, and reproducible conversational AI research, SDialog invites the community to collaborate and help shape the future of open dialogue generation.

Contributions of any size are welcome and help shape the future of open dialogue generation:

- **🗂️ Dataset Standardization**: Help convert existing dialogue datasets to SDialog format. Currently, each dataset stores dialogues in different formats, making cross-dataset analysis and model evaluation challenging. **Converted datasets are made available as Hugging Face datasets** in the [SDialog organization](https://huggingface.co/datasets/sdialog/) for easy access and integration.
- **🔧 Component Development**: Create new personas, orchestrators, evaluators, generators, or backend integrations
- **📊 Evaluation & Benchmarks**: Design new metrics, evaluation frameworks, or comparative studies
- **🧠 Interpretability Research**: Develop new analysis tools, steering methods, or mechanistic insights
- **📖 Documentation & Tutorials**: Improve guides, add examples, or create educational content
- **🐛 Issues & Discussions**: Report bugs, request features, or share research ideas and use cases

See `SDialog Github page <https://github.com/idiap/sdialog/>` for more info. We welcome issues, feature requests, and pull requests. If you want to **contribute to the project**, please open an [issue](https://github.com/idiap/sdialog/issues) or submit a PR, and help us make SDialog better 👍

Installation
------------
.. code-block:: bash

    pip install sdialog

Quickstart
----------
A minimal 60-second tour combining personas, an orchestrator, and a dialog simulation:

.. code-block:: python

    import sdialog
    from sdialog import Context
    from sdialog.agents import Agent
    from sdialog.personas import Persona
    from sdialog.orchestrators import SimpleReflexOrchestrator

    # Configure default LLM
    sdialog.config.llm("openai:gpt-4.1", temperature=0.9)

    # Personas & context
    barista = Persona(name="Alice", role="barista", personality="cheerful")
    customer = Persona(name="Bob", role="customer", personality="curious")
    ctx = Context(location="Downtown cafe", topics=["coffee"]) 

    # Simple tool (optional)
    def lookup_menu(item: str) -> dict:
        return {"item": item, "specials": ["vanilla latte", "cold brew"]}

    # Reflex orchestrator
    react = SimpleReflexOrchestrator(
        condition=lambda utt: "decaf" in utt.lower(),
        instruction="Explain decaf options and suggest one."
    )

    # Agents
    a_barista = Agent(persona=barista, tools=[lookup_menu])
    a_customer = Agent(persona=customer, first_utterance="Hi!")

   # Attach orchestrator(s) to barista
   a_barista = a_barista | react

    # Generate dialog(s)
    dialog = a_customer.dialog_with(a_barista, context=ctx)
    dialog.print(orchestration=True)


Core Concepts (At a Glance)
---------------------------

- **Dialog / Turn / Event:** Structured conversation container with metadata, transforms, serialization.
- **Persona / Context:** Attribute models (:class:`~sdialog.personas.Persona`, :class:`~sdialog.Context`) defining actors & shared environment.
- **Agent:** LLM-backed conversational actor with memory, optional tools, orchestration pipeline, and hooks.
- **Orchestrators:** Plug-ins that inject instructions or suggestions dynamically (e.g., :class:`~sdialog.orchestrators.SimpleReflexOrchestrator`, :class:`~sdialog.orchestrators.LengthOrchestrator`).
- **Generators:** LLM + rule hybrid attribute or dialogue generation (:class:`~sdialog.generators.PersonaGenerator`, :class:`~sdialog.generators.ContextGenerator`, :class:`~sdialog.generators.DialogGenerator`, :class:`~sdialog.generators.Paraphraser`).
- **Interpretability:** :class:`~sdialog.interpretability.Inspector` captures token/layer activations; steer with directions.
- **Evaluation:** Scalar metrics (:class:`~sdialog.evaluation.LinguisticFeatureScore`), flow scores, embedding distances, LLM judges, dataset aggregators (:class:`~sdialog.evaluation.DatasetComparator`).
- **Configuration:** Unified LLM backend selection & caching via :mod:`sdialog.config`.


Additional Resources
--------------------
Quick links:

- `GitHub Repository <https://github.com/idiap/sdialog>`_
- `Tutorials Collection <https://github.com/idiap/sdialog/tree/main/tutorials>`_
- `Colab Demo <https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/0.demo.ipynb>`_

Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   sdialog/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   api/sdialog

.. toctree::
   :maxdepth: 1
   :caption: About

   about/changelog
   about/contributing
   about/license
