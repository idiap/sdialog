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

`SDialog <https://github.com/idiap/sdialog>`__ addresses these needs by providing a comprehensive, extensible framework for synthetic dialogue generation and analysis, supporting:

- **Persona-based role-playing** with LLMs for realistic, diverse conversations.
- **Multi-agent and orchestrated dialogues** for complex, scenario-driven simulations.
- **Scenario and dataset integration** for reproducible research and benchmarking.
- **Flexible serialization and visualization** for downstream tasks and analysis.
- **Custom orchestration and extensibility** for advanced research and experimentation.

Whether you are building conversational datasets, evaluating dialogue models, or experimenting with new conversational AI techniques, SDialog offers the abstractions and tools you need.


User Guide
==========

.. toctree::
   :maxdepth: 3
   :caption: SDialog

   sdialog/index

.. toctree::
   :maxdepth: 3
   :caption: Simple Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/sdialog

.. toctree::
   :maxdepth: 2
   :caption: About

   about/changelog
   about/contributing
   about/license
