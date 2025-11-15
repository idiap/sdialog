.. image:: _static/logo-banner.png
    :target: https://github.com/idiap/sdialog
    :height: 100px
    :align: right


SDialog: Synthetic Dialog Generation, Evaluation, and Interpretability
=======================================================================

SDialog is an MIT-licensed open-source toolkit for building, simulating, and evaluating LLM-based conversational agents end-to-end. It aims to bridge **agent construction → dialog generation → evaluation → (optionally) interpretability** in a single reproducible workflow, so you can generate reliable, controllable dialog systems or data at scale.

It standardizes a Dialog schema and offers persona‑driven multi‑agent simulation with LLMs, composable orchestration, built‑in metrics, and mechanistic interpretability.

✨ Key Features
---------------

- **Standard dialog schema** with JSON import/export *(aiming to standardize dialog dataset formats with your help 🙏)*
- **Persona‑driven multi‑agent simulation** with contexts, tools, and thoughts
- **Composable orchestration** for precise control over behavior and flow
- **Built‑in evaluation** (metrics + LLM‑as‑judge) for comparison and iteration
- **Native mechanistic interpretability** (inspect and steer activations)
- **Easy creation of user-defined components** by inheriting from base classes (personas, metrics, orchestrators, etc.)
- **Interoperability** across OpenAI, Hugging Face, Ollama, AWS Bedrock, Google GenAI, Anthropic, and more
- **Audio generation** for converting text dialogs to realistic audio conversations

If you are building conversational systems, benchmarking dialog models, producing synthetic training corpora, simulating diverse users to test or probe conversational systems, or analyzing internal model behavior, SDialog provides an end‑to‑end workflow.

Quick Links
-----------

- `GitHub <https://github.com/idiap/sdialog>`_
- `API Reference <https://sdialog.readthedocs.io/en/latest/api/sdialog.html>`_
- `Demo (Colab) <https://colab.research.google.com/github/idiap/sdialog/blob/main/tutorials/demo.ipynb>`_
- `Tutorials <https://github.com/idiap/sdialog/tree/main/tutorials>`_
- `Datasets (HF) <https://huggingface.co/datasets/sdialog>`_
- `Issues <https://github.com/idiap/sdialog/issues>`_


Installation
------------

.. code-block:: bash

    pip install sdialog

.. important::
   If you plan to use the audio capabilities of SDialog via its audio sub-module (``sdialog.audio``), you must install SDialog with audio dependencies:
   
   .. code-block:: bash
   
       pip install sdialog[audio]

Alternatively, a ready-to-use Apptainer image (.sif) with SDialog and all dependencies is available on `Hugging Face <https://huggingface.co/datasets/sdialog/apptainer/resolve/main/sdialog.sif>`_.

.. code-block:: bash

    apptainer exec --nv sdialog.sif python3 -c "import sdialog; print(sdialog.__version__)"

.. note::
   This Apptainer image also has the Ollama server preinstalled.


🏁 Quickstart Tour
------------------

Here's a short, hands‑on example: a support agent helps a customer disputing a double charge. We add a small refund rule and two simple tools, generate three dialogs for evaluation, then serve the agent on port 1333 for Open WebUI or any OpenAI‑compatible client.

.. code-block:: python

    import sdialog
    from sdialog import Context
    from sdialog.agents import Agent
    from sdialog.personas import SupportAgent, Customer
    from sdialog.orchestrators import SimpleReflexOrchestrator

    # First, let's set our preferred default backend:model and parameters
    sdialog.config.llm("openai:gpt-4.1", temperature=1, api_key="YOUR_KEY")
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
        dialog = simulated_customer.dialog_with(support_agent, context=web_chat)
        dialog.to_file(f"dialog_{ix}.json")
        dialog.print(all=True)

    # Finally, let's serve our support agent to interact with real users (OpenAI-compatible API)
    #    Point Open WebUI or any OpenAI-compatible client to: http://localhost:1333
    support_agent.serve(port=1333)

.. tip::
   - Choose your `LLMs and backends freely <https://sdialog.readthedocs.io/en/latest/sdialog/index.html#configuration-layer>`_.
   - Personas and context can be `automatically generated <https://sdialog.readthedocs.io/en/latest/sdialog/index.html#attribute-generators>`_ (e.g. generate different customer profiles!).

.. note::
   - See `"agents with tools and thoughts" tutorial <https://github.com/idiap/sdialog/blob/main/tutorials/00_overview/7.agents_with_tools_and_thoughts.ipynb>`_ for a more complete example.
   - See `Serving Agents via REST API <https://sdialog.readthedocs.io/en/latest/sdialog/index.html#serving-agents>`_ for more details on server options.


Core Capabilities
-----------------

Testing Remote Systems
^^^^^^^^^^^^^^^^^^^^^^

Probe OpenAI‑compatible deployed systems with controllable simulated users and capture dialogs for evaluation.

You can use SDialog as a controllable test harness for any OpenAI‑compatible system such as **vLLM**-based ones by role‑playing realistic or adversarial users against your deployed system:

- Black‑box functional checks (Does the system follow instructions? Handle edge cases?)
- Persona / use‑case coverage (Different goals, emotions, domains)
- Regression testing (Run the same persona batch each release; diff dialogs)
- Safety / robustness probing (Angry, confused, or noisy users)
- Automated evaluation (Pipe generated dialogs directly into evaluators)

.. code-block:: python

    # Our remote system (your conversational backend exposing an OpenAI-compatible API)
    system = Agent(
        model="openai:your/model",  # Model name exposed by your server
        openai_api_base="http://your-endpoint.com:8000/v1",
        openai_api_key="EMPTY",
        name="System"
    )

    # Let's make our simulated customer talk with the system
    dialog = simulated_customer.dialog_with(system)
    dialog.to_file("dialog_0.json")


Loading and Saving Dialogs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import, export, and transform dialogs from JSON, text, CSV, or Hugging Face datasets.

.. code-block:: python

    from sdialog import Dialog

    # Load from JSON (generated by SDialog using `to_file()`)
    dialog = Dialog.from_file("dialog_0.json")

    # Load from HuggingFace Hub datasets
    dialogs = Dialog.from_huggingface("sdialog/Primock-57")

    # Create from plain text files or strings
    dialog_from_txt = Dialog.from_str("""
    Alice: Hello there! How are you today?
    Bob: I'm doing great, thanks for asking.
    Alice: That's wonderful to hear!
    """)

    # All Dialog objects have rich manipulation methods
    dialog.filter("Alice").rename_speaker("Alice", "Customer").upper().to_file("processed.json")


Evaluation and Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^

Score dialogs with built‑in metrics and LLM judges, and compare datasets with aggregators and plots.

.. code-block:: python

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
    comparator.plot()

.. tip::
   See `evaluation tutorial <https://github.com/idiap/sdialog/blob/main/tutorials/00_overview/5.evaluation.ipynb>`_.


Mechanistic Interpretability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Capture per‑token activations and steer models via Inspectors for analysis and interventions.

.. code-block:: python

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
    act = inspector[-1][0].act  # [response index][token index]

Steering intervention:

.. code-block:: python

    import torch
    anger_direction = torch.load("anger_direction.pt")
    agent_steered = agent | inspector - anger_direction

    agent_steered("You are an extremely upset assistant")  # Agent "can't get angry anymore" :)

.. tip::
   See `the tutorial <https://github.com/idiap/sdialog/blob/main/tutorials/00_overview/6.agent%2Binspector_refusal.ipynb>`_ on using SDialog to remove the refusal capability from LLaMA 3.2.


Audio Generation
^^^^^^^^^^^^^^^^

Convert text dialogs to realistic audio conversations with speech synthesis, voice assignment, and acoustic simulation.

SDialog can transform text dialogs into realistic audio conversations with a simple one-line command:

.. code-block:: python

    from sdialog import Dialog

    dialog = Dialog.from_file("my_dialog.json")

    # Convert to audio with default settings (Kokoro TTS)
    audio_dialog = dialog.to_audio()

    # Or customize the audio generation
    audio_dialog = dialog.to_audio(
        perform_room_acoustics=True,
        audio_file_format="mp3",
        re_sampling_rate=16000,
    )

.. tip::
   See the `audio tutorials <https://github.com/idiap/sdialog/tree/main/tutorials/01_audio>`_ for examples including acoustic simulation, room generation, and voice databases. Full documentation is available at `Audio Generation <https://sdialog.readthedocs.io/en/latest/sdialog/index.html#audio-generation>`_.


Project Vision & Community Call
--------------------------------

To accelerate open, rigorous, and reproducible conversational AI research, SDialog invites the community to collaborate and help shape the future of open dialog generation.

How You Can Help
^^^^^^^^^^^^^^^^

- **🗂️ Dataset Standardization**: Help convert existing dialog datasets to SDialog format. Currently, each dataset stores dialogs in different formats, making cross-dataset analysis and model evaluation challenging. **Converted datasets are made available as Hugging Face datasets** in the `SDialog organization <https://huggingface.co/datasets/sdialog/>`_ for easy access and integration.
- **🔧 Component Development**: Create new personas, orchestrators, evaluators, generators, or backend integrations
- **📊 Evaluation & Benchmarks**: Design new metrics, evaluation frameworks, or comparative studies
- **🧠 Interpretability Research**: Develop new analysis tools, steering methods, or mechanistic insights
- **📖 Documentation & Tutorials**: Improve guides, add examples, or create educational content
- **🐛 Issues & Discussions**: Report bugs, request features, or share research ideas and use cases

.. note::
   **Example**: Check out `Primock-57 <https://huggingface.co/datasets/sdialog/Primock-57>`_, a sample dataset already available in SDialog format on Hugging Face.
   
   If you have a dialog dataset you'd like to convert to SDialog format, need help with the conversion process, or want to contribute in any other way, please `open an issue <https://github.com/idiap/sdialog/issues>`_ or reach out to us. We're happy to help and collaborate!


Documentation for AI Coding Assistants
---------------------------------------

Documentation for **AI coding assistants** like Copilot is also available at `llm.txt <https://sdialog.readthedocs.io/en/latest/llm.txt>`_ following the `llm.txt specification <https://llmstxt.org/>`_. In your Copilot chat, simply use:

.. code-block:: text

    #fetch https://sdialog.readthedocs.io/en/latest/llm.txt

    Your prompt goes here...(e.g. Write a python script using sdialog to have an agent for
    criminal investigation, define its persona, tools, orchestration...)


Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   sdialog/index

.. toctree::
   :maxdepth: 1
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
