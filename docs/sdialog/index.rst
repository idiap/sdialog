Introduction
============

Overview
--------
SDialog is a modular Python toolkit for generating, steering, evaluating, and analyzing **synthetic dialogues** with instruction-tuned Large Language Models (LLMs). It standardizes core abstractions (:class:`~sdialog.Dialog`, :class:`~sdialog.Turn`, :class:`~sdialog.Event`, :class:`~sdialog.personas.Persona`, :class:`~sdialog.Context`, :class:`~sdialog.agents.Agent`, :class:`~sdialog.orchestrators.base.BaseOrchestrator`) and supplies composable components for:

- Persona-driven multi-agent role-play
- Dynamic orchestration (rule / probabilistic / semantic response suggestion)
- Attribute & dialogue generation (LLM + light rule hybrids)
- Evaluation (linguistic, flow / coherence, embedding similarity, LLM judges)
- Mechanistic interpretability & activation steering
- Dataset integration (e.g. STAR) and reproducible experiment pipelines

If you are building controlled conversational simulations, benchmarking dialog models, producing synthetic training corpora, or probing internal model behavior, SDialog provides an end-to-end workflow.

Architecture at a Glance
------------------------
::

    +-----------+             +-------------------+
    | Personas  |----+------->|      Agents       |----+
    | Context   |    |        |  (LLM core +      |    |
    +-----------+    |        |  Orchestrators /  |    v
          |          |        |  Inspectors /     | +------------------+
          |          |        |  Tools / Thinking)| |  Dialog Objects  |---->
          |          |        +-------------------+ +------------------+
          |          |                   ^                    |
          |          |                   |                    v
          |    +-------------------+     |          +--------------------+
          +--->|  Generation Layer |-----+          |  Evaluation Stack  |
               | (Attr & Dialog    |                | (consumes Dialogs  |
               |  Generators)      |                |  + optional        |
               +-------------------+                |  persona/context)  |
                                                    +--------------------+

Note: Attribute & Dialog generators may produce Personas / Contexts / prototype Dialogs that seed Agents; Agents then produce the canonical Dialog objects evaluated downstream.

----

Core Concepts
=============

Dialog
------
A :class:`~sdialog.Dialog` object contains an ordered list of :class:`~sdialog.Turn` instances and an optional list of :class:`~sdialog.Event` objects, plus metadata and utility methods:

- **Turn**: Has ``speaker`` and ``text`` fields.
- **Event**: Generic action record (utterance, instruction, tool invocation, etc.).
- **Metadata**: Provenance for reproducibility (version, timestamp, id, parentId, seed, model, etc.).
- **Methods**: Text transforms (:meth:`~sdialog.Dialog.lower`, :meth:`~sdialog.Dialog.upper`, :meth:`~sdialog.Dialog.replace`, :meth:`~sdialog.Dialog.re_sub`, etc.), utilities (:meth:`~sdialog.Dialog.prompt`, :meth:`~sdialog.Dialog.rename_speaker`, :meth:`~sdialog.Dialog.filter`, :meth:`~sdialog.Dialog.get_speakers`, etc.), serialization (:meth:`~sdialog.Dialog.to_file`, :meth:`~sdialog.Dialog.from_file`), cloning with lineage (:meth:`~sdialog.Dialog.clone`), and :meth:`~sdialog.Dialog.length` estimation (words / turns / minutes).

**Operations Example**:

.. code-block:: python

    from sdialog import Dialog, Turn

    # Create a sample dialog
    dialog = Dialog(turns=[
        Turn(speaker="Alice", text="Hello there! How are you doing today?"),
        Turn(speaker="Bob", text="I'm doing great, thanks for asking."),
        Turn(speaker="Alice", text="That's wonderful to hear!"),
        Turn(speaker="Bob", text="What about you? How's YOUR DAY going?")
    ])

    # Slicing creates new Dialog with fresh ID and parentId linkage
    first_half = dialog[:2]  # First 2 turns → new Dialog with dialog.id as parentId
    print(f"Original ID: {dialog.id}")
    print(f"Slice ID: {first_half.id}, Parent: {first_half.parentId}")

    # Text transformations examples
    normalized = dialog.lower().replace("!", ".")             # Chain operations
    filtered_alice = dialog.filter("Alice")                   # Only Alice's turns
    dialog_alicia = dialog.rename_speaker("Alice", "Alicia")  # Rename speaker

    # Print some information
    print(f"Length: {len(dialog)} turns")  # len(dialog) == dialog.length('turns')
    print(f"Length: {dialog.length('words')} words, approx. {dialog.length('time')} minutes")
    print(f"Speakers: {dialog.get_speakers()}")

    # Iterate over dialog turns
    for ix, turn in enumerate(dialog_alicia):
        print(f"Turn {ix+1}: {turn.speaker} - {turn.text}")

Instead of creating the Dialog object programmatically as above, you can also load it from text files or strings using a simple template format with :meth:`~sdialog.Dialog.from_file` or :meth:`~sdialog.Dialog.from_str` methods, respectively.
Both methods accept the same arguments and ``Dialog.from_str(text)`` is equivalent to ``Dialog.from_file("file.txt")`` when the file contains plain text.
The most common cases are shown below with three examples:

**Creating Dialog from Text**:

.. code-block:: python

    from sdialog import Dialog, Turn

    # Basic usage - Text in default "{speaker}: {text}" format
    dialog_text = """Alice: Hello there! How are you today?
    Bob: I'm doing great, thanks for asking.
    Alice: That's wonderful to hear!
    Bob: What about you? How's your day going?"""

    dialog = Dialog.from_str(dialog_text)
    dialog.print()

    # Text in custom format
    chat_log = """[2024-01-15 14:30] @user123: Hey everyone!
    [2024-01-15 14:31] @moderator: Welcome to the chat
    [2024-01-15 14:32] @user123: Thanks, excited to be here!
    [2024-01-15 14:33] @helper_bot: How can I assist you today?"""

    # Define your custom template to parse each turn
    dialog_from_chat = Dialog.from_str(
        chat_log,
        template="[{timestamp}] @{speaker}: {text}"
    )
    dialog_from_chat.print()

    # Text with no speaker tags
    simple_conversation = """Hello there!
    Hi, how are you?
    I'm doing well, thanks!
    That's great to hear."""

    # Provide default speakers to assign alternately
    dialog_with_defaults = Dialog.from_str(
        simple_conversation,
        template="{text}",  # No speaker in text turns
        default_speakers=["Alice", "Bob"]  # Alternating assignment
    )
    dialog_with_defaults.print()


Personas & Context
------------------
Personas are structured, typed attribute bundles that specify role, style, goals, background knowledge, and behavioral constraints used to condition LLM prompts for Agents in a reproducible, inspectable way. Context objects complement Personas with shared situational grounding so multiple agents can coordinate.

SDialog formalizes this socio-cognitive conditioning through attribute models:

- **Persona** / **ExtendedPersona** (:class:`~sdialog.personas.Persona`, :class:`~sdialog.personas.ExtendedPersona`): Baseline and expanded demographic + behavioral traits.
- **Domain-specific Personas**: :class:`~sdialog.personas.Doctor`, :class:`~sdialog.personas.Patient`, :class:`~sdialog.personas.ExtendedDoctor`, :class:`~sdialog.personas.ExtendedPatient`, :class:`~sdialog.personas.Customer`, :class:`~sdialog.personas.SupportAgent`.
- :class:`~sdialog.Context`: Shared situational grounding (location, environment, objects, goals, constraints, topics, style_guidelines, shared knowledge, circumstances).

All inherit :class:`~sdialog.base.BaseAttributeModel` and, as such, they support:

- Cloning with lineage tracking (:meth:`~sdialog.base.BaseAttributeModel.clone`).
- Serialization and I/O methods (:meth:`~sdialog.base.BaseAttributeModel.json`, :meth:`~sdialog.base.BaseAttributeModel.prompt`, :meth:`~sdialog.base.BaseAttributeModel.to_file`, :meth:`~sdialog.base.BaseAttributeModel.from_file`).

**Creating your own Personas**:

Create custom personas by inheriting from :class:`~sdialog.personas.BasePersona` (equivalent to :class:`~sdialog.base.BaseAttributeModel`) or from any existing persona class to extend it further.

.. code-block:: python

    from sdialog.personas import BasePersona

    class TravelAgentPersona(BasePersona):
        role: str = ""
        expertise: str = ""
        tone: str = ""
        goals: str = ""
        constraints: str = "Make sure to provide realistic travel options"

    # Instantiate with defaults or override specific attributes
    my_persona = TravelAgentPersona(
        role="Travel Agent",
        expertise="European rail itineraries",
        tone="friendly and concise",
        goals="suggest routes, optimize cost",
    )
    my_persona.print()  # Pretty-print my travel agent persona

    # Clone with overrides while preserving lineage metadata
    variant = my_persona.clone(tone="enthusiastic and energetic")
    assert variant.parentId == my_persona.id

**Advanced Persona Creation with Field Documentation**:

You can use Pydantic Field descriptions to document each attribute in your custom persona. These descriptions serve as guides for LLMs when generating persona objects via :class:`~sdialog.generators.PersonaGenerator`:

.. code-block:: python

    from sdialog.generators import PersonaGenerator
    from sdialog.personas import BasePersona
    from pydantic import Field


    class TravelAgentPersona(BasePersona):
        role: str = Field("", description="The role of the persona, for example 'Travel Agent' or 'Tour Guide'")
        expertise: str = Field("", description="The area of expertise")
        tone: str = Field("", description="The tone of the persona")
        goals: str = Field("", description="A short description of the goals")
        constraints: str = Field("", description="Operational constraints, e.g., 'budget limitations'")

    # Alternatively, use PersonaGenerator to create personas based on field descriptions
    generator = PersonaGenerator(TravelAgentPersona, model="openai:gpt-4")
    generated_persona = generator.generate()
    generated_persona.print()  # Pretty-print the generated travel agent persona


----

Agents & Orchestration
======================

Agents
------
:class:`~sdialog.agents.Agent` encapsulates an LLM-backed conversational actor:

Features:

- Persona + optional Context + exemplar Dialogs → prompt assembly.
- Memory (list of role-tagged messages) with additive system instructions.
- First utterance selection (fixed or random from list).
- Orchestrator pipeline (dynamic instruction injection).
- Optional tools (callable functions) integrated into LLM responses (if backend supports function/tool calling).
- Optional "thinking" (hidden reasoning segments) extraction & filtering.
- Lookahead capability (:meth:`~sdialog.agents.Agent.response_lookahead`) without mutating memory.
- JSON serialization of configuration and persona.

Key Methods:

- :meth:`~sdialog.agents.Agent.__call__`: invoke the agent with a message and get a response (e.g. ``resp = agent("Hello!")``).
- :meth:`~sdialog.agents.Agent.dialog_with` / alias :meth:`~sdialog.agents.Agent.talk_with`: multi-turn role-play.
- :meth:`~sdialog.agents.Agent.instruct`: inject immediate or persistent system instructions.
- :meth:`~sdialog.agents.Agent.add_orchestrators` / pipeline composition via ``agent | orchestrator``.
- :meth:`~sdialog.agents.Agent.add_inspectors` or ``agent | inspector`` for activation capture (mechanistic interpretability).
- :meth:`~sdialog.agents.Agent.reset`: reproducible restart to mark the beginning of a new conversation.
- :meth:`~sdialog.agents.Agent.memory_dump`: snapshot conversation for logging.
- :meth:`~sdialog.agents.Agent.prompt`: get the underlying system prompt used by the agent.
- :meth:`~sdialog.agents.Agent.json`: export the agent as a JSON object.

Orchestration
-------------
Orchestrators are lightweight controllers that examine the current dialog state and the last utterance from the other agent, optionally returning an instruction. They can be **ephemeral** (one-time) or **persistent** (lasting across multiple turns). Orchestrators are composed using the pipe operator:

.. code-block:: python

    # Instantiate orchestrators first
    length_orch = LengthOrchestrator(min=8, max=12)
    reflex_orch = SimpleReflexOrchestrator(
        condition=lambda utt: "confused" in utt.lower() or "don't understand" in utt.lower(),
        instruction="Slow down and explain with a concrete example to clarify."
    )
    agent = agent | length_orch | reflex_orch


Built-in Orchestrators:

- :class:`~sdialog.orchestrators.SimpleReflexOrchestrator`: condition trigger in input utterance → fixed instruction for the agent to generate the response.
- :class:`~sdialog.orchestrators.LengthOrchestrator`: enforce min length / encourage wrap at max.
- :class:`~sdialog.orchestrators.ChangeMindOrchestrator`: probabilistic revision injection (limited times).
- :class:`~sdialog.orchestrators.SimpleResponseOrchestrator`: semantic similarity suggestions from canned responses / action graph.
- :class:`~sdialog.orchestrators.InstructionListOrchestrator`: deterministic scripted step sequence.

**Creating your own Orchestrators**:

You can create custom orchestrators by inheriting from :class:`~sdialog.orchestrators.base.BaseOrchestrator` (for one-time instructions) or :class:`~sdialog.orchestrators.base.BasePersistentOrchestrator` (for persistent instructions). Implement the ``instruct(self, dialog, utterance)`` method to define your orchestration logic. The method receives the current :class:`~sdialog.Dialog` and the last utterance from the opposite agent (or ``None`` if it's the first turn).

- **One-time Orchestrator Example**:

.. code-block:: python

    from sdialog.orchestrators import BaseOrchestrator

    class EmpathyBooster(BaseOrchestrator):
        def __init__(self, emotion_keywords):
            self.emotion_keywords = emotion_keywords

        def instruct(self, dialog, utterance):
            if utterance and any(keyword in utterance.lower() for keyword in self.emotion_keywords):
                return ("The user seems to be experiencing difficult emotions. "
                        "Acknowledge their feelings with empathy and offer gentle support.")
            return None

    empathy_booster = EmpathyBooster(["worried", "stressed", "frustrated",
                                      "upset", "anxious", "sad", "disappointed", "overwhelmed"])
    agent = agent | empathy_booster

- **Persistent Orchestrator Example**:

This simple example shows the minimal pattern: return an instruction once when a condition is first met; the instruction then persists automatically in the agent's system messages and does not need to be returned again. A more advanced "LLM-as-a-Judge"-based persistent orchestrator is provided :ref:`here <advanced_context_persistent_orchestrator>`.

.. code-block:: python

    from sdialog.orchestrators import BasePersistentOrchestrator

    class EarlyRecapOrchestrator(BasePersistentOrchestrator):
        """Inject one persistent instruction once it's clear the user is asking too many questions."""
        def __init__(self, question_threshold=3):
            super().__init__()
            self.question_threshold = question_threshold
            self.instructed = False

        def instruct(self, dialog, utterance):
            if self.instructed or len(dialog.turns) < 4:
                return None
            # Count '?' in all turns from the other speaker (not this agent)
            other_qmarks = sum(t.text.count('?') for t in dialog.turns
                               if t.speaker.lower() != self.agent.name.lower())
            if other_qmarks >= self.question_threshold:
                self.instructed = True
                return ("From now on begin each reply with a concise bullet-style recap of the key questions so far, "
                        "then answer the latest question directly, and ask ONE clarifying follow-up only if needed.")
            return None

        def reset(self):  # allow reuse in later conversations
            super().reset()
            self.instructed = False

    recap_orch = EarlyRecapOrchestrator(question_threshold=3)
    agent = agent | recap_orch

----

Generation
==========

Generation Components
---------------------

The generation system in SDialog provides powerful tools for creating synthetic content at different levels of complexity. Whether you need to generate individual personas, contextual settings, or complete dialogues, these components work together to produce diverse and realistic outputs.

Attribute Generators
~~~~~~~~~~~~~~~~~~~~

Attribute generators combine LLM intelligence with rule-based patterns to create structured objects with randomized or AI-generated content. They are perfect for creating diverse personas and contexts for your dialogue simulations.

Both generators derive from :class:`~sdialog.generators.base.BaseAttributeModelGenerator` and support flexible attribute generation rules:

**PersonaGenerator** (:class:`~sdialog.generators.PersonaGenerator`)
    Creates diverse character profiles with demographic, behavioral, and professional attributes. Ideal for generating varied participants in dialogue scenarios.

    .. code-block:: python

        from sdialog.personas import Doctor
        from sdialog.generators import PersonaGenerator

        # Create a generator for doctor personas
        doctor_gen = PersonaGenerator(Doctor)
        # Alternatively, setup generation for specific attributes (other than LLM-based)
        doctor_gen.set(
            specialty=["cardiology", "neurology", "oncology"],
            years_of_experience="{5-20}",
            formality=["formal", "casual", "friendly"]
        )

        # Generate diverse doctors
        doctor1 = doctor_gen.generate()
        doctor2 = doctor_gen.generate()

        # Ask the LLM to generate multiple doctors at once
        doctors_batch = doctor_gen.generate(n=3)  # Returns list of 3 doctors

**ContextGenerator** (:class:`~sdialog.generators.ContextGenerator`)
    Generates rich contextual frameworks that define the setting, environment, and situational constraints for dialogues. Essential for creating realistic and consistent conversation backgrounds.

    .. code-block:: python

        from sdialog import Context
        from sdialog.generators import ContextGenerator

        # Create varied hospital contexts
        ctx_gen = ContextGenerator(Context(location="hospital ward"))
        ctx_gen.set(
            environment="{llm:Describe a realistic medical environment}",
            constraints=["time pressure", "privacy concerns", "urgent case"]
        )
        context = ctx_gen.generate()

**Attribute Rule Patterns**

The :meth:`~sdialog.generators.base.BaseAttributeModelGenerator.set` method accepts various rule types for flexible content generation:

- **LLM Delegation**: ``"*"`` or ``"{llm}"`` - Let the AI decide the value (default)
- **Guided LLM**: ``"{llm:Create a professional background}"`` - AI with specific instructions  
- **Numeric Ranges**: ``"{5-25}"`` - Random integer between bounds (inclusive)
- **File Sources**: ``"{txt:names.txt}"`` - Random line from text file
- **Data Tables**: ``"{csv:specialty:doctors.csv}"`` - Random value from CSV/TSV column
- **Python Functions**: ``callable(**attributes)`` - Execute function (receives partial object if compatible)
- **Choice Lists**: ``["option1", "option2"]`` - Random selection from list
- **Fixed Values**: ``"specific_value"`` - Direct assignment

Dialogue Generators
~~~~~~~~~~~~~~~~~~~

Dialogue generators create complete conversations using different approaches, from direct LLM instruction to sophisticated persona-driven interactions.

**DialogGenerator** (:class:`~sdialog.generators.DialogGenerator`)
    The foundational dialogue generator that creates conversations based on free-form instructions. Great for quick prototyping and simple dialogue generation tasks.

    .. code-block:: python

        from sdialog.generators import DialogGenerator

        # Generate a consultation dialogue
        gen = DialogGenerator("Generate a brief medical consultation about headaches")
        
        dialog = gen.generate()
        dialog.print()

**PersonaDialogGenerator** (:class:`~sdialog.generators.PersonaDialogGenerator`)
    Creates sophisticated dialogues by having two distinct personas or agents interact naturally. This generator produces more realistic and character-consistent conversations.

    .. code-block:: python

        from sdialog.personas import Doctor, Patient
        from sdialog.generators import PersonaDialogGenerator

        doctor = Doctor(name="Dr. Smith", specialty="cardiology")
        patient = Patient(name="John", reason_for_visit="chest pain")

        # Generate persona-driven dialogue
        gen = PersonaDialogGenerator(
            doctor, patient, 
            dialogue_details="Discuss symptoms and initial diagnosis"
        )
        
        dialog = gen.generate()
        dialog.print()

**Paraphraser** (:class:`~sdialog.generators.Paraphraser`)
    Transforms existing dialogues following user-provided instructions. Useful for improving synthetic dialogues, adapting content for different styles, or data augmentation.

    .. code-block:: python

        from sdialog.generators import Paraphraser

        # Make bot responses sound more natural
        paraphraser = Paraphraser(
            extra_instructions="Make responses sound more empathetic and natural",
            target_speaker="Bot"  # Only paraphrase "Bot"'s turns
        )

        improved_dialog = paraphraser(original_dialog)
        improved_dialog.print()


Reproducibility & Seeding
-------------------------
Every generation entry point accepts ``seed`` where feasible. Metadata persists ``seed`` and ``model`` so output artifacts remain auditable. Identical model + parameters + seed ⇒ deterministic attribute choices & (backend permitting) stable dialog trajectories. Use ``clone(new_id=...)`` to branch derived objects with recorded lineage.

----

Evaluation & Interpretability
=============================

Evaluation Suite
----------------
Categories:

1. **Linguistic / Readability**: :class:`~sdialog.evaluation.LinguisticFeatureScore`
   - mean-turn-length, hesitation-rate, gunning-fog, flesch-reading-ease.
2. **Flow Graph Based**: :class:`~sdialog.evaluation.DialogFlowScore` (likelihood) & :class:`~sdialog.evaluation.DialogFlowPPL` (perplexity-like) against reference dialog graph.
3. **Embedding Similarity**: :class:`~sdialog.evaluation.SentenceTransformerDialogEmbedder` + evaluators such as :class:`~sdialog.evaluation.ReferenceCentroidEmbeddingEvaluator`.
4. **Distribution Divergence** (score-level): :class:`~sdialog.evaluation.KDEDistanceEvaluator`, :class:`~sdialog.evaluation.FrechetDistanceEvaluator`.
5. **LLM Judges**: Binary or numeric scoring with structured output
    - :class:`~sdialog.evaluation.LLMJudgeYesNo` / :class:`~sdialog.evaluation.LLMJudgeScore`
    - Specialized: :class:`~sdialog.evaluation.LLMJudgeRealDialog`, Likert variant (:class:`~sdialog.evaluation.LLMJudgeRealDialogLikertScore`), numeric range variant (:class:`~sdialog.evaluation.LLMJudgeRealDialogScore`), :class:`~sdialog.evaluation.LLMJudgeRefusal`, :class:`~sdialog.evaluation.LLMJudgePersonaAttributes`.
6. **Advanced Embedding Evaluators**: :class:`~sdialog.evaluation.FrechetBERTDistanceEvaluator` (Fréchet distance with BERT embeddings), :class:`~sdialog.evaluation.PrecisionRecallDistanceEvaluator` (Precision-Recall curves based on BERT).
7. **Dataset Aggregators**: :class:`~sdialog.evaluation.DatasetComparator` to compare multiple evaluators.
8. **Statistics / Frequency**: :class:`~sdialog.evaluation.StatsEvaluator` (mean/std/min/max/median), :class:`~sdialog.evaluation.MeanEvaluator`, :class:`~sdialog.evaluation.FrequencyEvaluator`.

**Evaluation Examples:**

.. code-block:: python

    from sdialog.evaluation import (
        LinguisticFeatureScore, DialogFlowScore, LLMJudgeRealDialog, LLMJudgeRealDialogLikertScore,
        MeanEvaluator, FrequencyEvaluator, StatsEvaluator,
        DatasetComparator
    )

    # Linguistic analysis
    hesitation_score = LinguisticFeatureScore(feature="hesitation-rate")
    readability_score = LinguisticFeatureScore(feature="flesch-reading-ease")

    # Flow-based evaluation (requires reference dialogues)
    flow_evaluator = DialogFlowScore(reference_dialogs)

    # LLM-based judgment
    realism_judge = LLMJudgeRealDialog(feedback=True)
    realism_judge_likert = LLMJudgeRealDialogLikertScore()  # 1-5 scale

    # Multi-metric comparison across different models
    comparator = DatasetComparator([
        FrequencyEvaluator(realism_judge),
        StatsEvaluator(readability_score),
        MeanEvaluator(hesitation_score),
        MeanEvaluator(flow_evaluator)
    ])

    # Evaluate multiple dialog sets
    results = comparator({
        "model_A": dialogs_a,
        "model_B": dialogs_b,
        "human_baseline": human_dialogs
    })
    comparator.plot()

Caching: Some score computations, so a good practice is to enable caching (``sdialog.config.cache(True)``) to accelerate repeated runs across large corpora.

Interpretability & Steering
---------------------------

SDialog provides advanced tools for mechanistic interpretability and activation steering to understand and control model behavior during dialogue generation.

**Activation Inspection:**

:class:`~sdialog.interpretability.Inspector` registers PyTorch forward hooks on specified model layers to capture per-token activations during generation:

.. code-block:: python

    from sdialog.interpretability import Inspector
    from sdialog.agents import Agent

    # Create agent and inspector
    agent = Agent(persona=my_persona)
    inspector = Inspector(target='model.layers.15.post_attention_layernorm')
    
    # Attach inspector to agent
    agent = agent | inspector
    
    # Generate responses (activations are captured automatically)
    response1 = agent("Tell me about the weather")
    response2 = agent("What's your favorite color?")
    
    # Access captured activations
    print(f"Captured {len(inspector)} responses")
    first_token_activation = inspector[-1][0].act  # Last response, first token
    print(f"Activation shape: {first_token_activation.shape}")
    
    # Inspect system instructions
    inspector.find_instructs(verbose=True)

**Activation Steering:**

:class:`~sdialog.interpretability.DirectionSteerer` enables targeted manipulation of model activations:

.. code-block:: python

    import torch
    from sdialog.interpretability import DirectionSteerer

    # Create a direction vector (e.g., from activation differences)
    empathy_direction = torch.randn(4096)  # Model hidden size
    
    # Create steerer and attach to inspector
    steerer = DirectionSteerer(empathy_direction)
    inspector = inspector + steerer  # Add direction
    # or: inspector = inspector - steerer  # Subtract direction
    
    # Generate with steering applied
    steered_response = agent("I'm feeling really sad today")
    
    # Compare with baseline
    inspector.clear()  # Reset to remove steering
    baseline_response = agent("I'm feeling really sad today")

**Advanced Usage:**

.. code-block:: python

    # Multi-layer inspection
    multi_inspector = Inspector({
        'early': 'model.layers.5.post_attention_layernorm',
        'middle': 'model.layers.15.post_attention_layernorm', 
        'late': 'model.layers.25.post_attention_layernorm'
    },
    steering_interval=(0, 2))  # steer only the first two generated tokens

    agent = agent | multi_inspector - direction

----

Configuration & Control
=======================

Configuration Layer
-------------------
Centralized configuration lives in ``sdialog.config``:

- :func:`~sdialog.config.llm`: set global backend & hyperparameters.
- :func:`~sdialog.config.llm_params`: update hyperparameters only.
- :func:`~sdialog.config.cache` / :func:`~sdialog.config.cache_path` / :func:`~sdialog.config.set_cache`.
- Prompt overrides: :func:`~sdialog.config.set_persona_dialog_generator_prompt`, :func:`~sdialog.config.set_persona_generator_prompt`, :func:`~sdialog.config.set_dialog_generator_prompt`, :func:`~sdialog.config.set_persona_agent_prompt`.

**Supported Backend Formats:**

- ``openai:MODEL`` - OpenAI models (GPT-3.5, GPT-4, etc.)
- ``huggingface:MODEL`` - HuggingFace transformers models
- ``ollama:MODEL`` - Local Ollama models
- ``aws:MODEL`` - AWS Bedrock models (Anthropic Claude, etc.)
- Local model instances passed directly

**Configuration Examples:**

.. code-block:: python

    import sdialog.config as config

    # Set global LLM backend
    config.llm("openai:gpt-4")
    config.llm_params(temperature=0.7, max_tokens=1000)

    # Enable caching for expensive operations
    config.set_cache("/path/to/cache", enable=True)

    # Use local Ollama model
    config.llm("ollama:llama2")

    # Configure AWS Bedrock
    config.llm("aws:anthropic.claude-v2")
    config.llm_params(temperature=0.5)

Tools & Function Calling
------------------------
Provide a list of Python callables to an Agent via ``tools=[fn_a, fn_b]``. For backends that expose tool/function-calling semantics, outputs may be (a) executed and inserted into memory, (b) used to refine final completions depending on backend support. Ensure return values are JSON-serializable.

Thinking Segments
-----------------
If ``think=True`` on Agent (and backend supports surrogate reasoning tokens), SDialog can preserve or strip these segments. If not natively supported by the underlying, SDialog can still recognize thinking segments using the provided value in ``thinking_pattern`` (regex capturing group). Suitable for experiments analyzing chain-of-thought style reasoning without leaking it downstream.


----

Extensibility & I/O
===================

Serialization & Persistence
---------------------------
All core objects support JSON serialization with metadata. Typical patterns:

.. code-block:: python

    dialog.to_file("session.json")
    restored = Dialog.from_file("session.json")

    context.to_file("ccontext.json")
    restored = context.from_file("ccontext.json")

    persona.to_file("persona.json")
    restored = Persona.from_file("persona.json")

For plain text or CSV import/export: :meth:`~sdialog.Dialog.to_file` (``type='auto'|'txt'|'csv'|'json'``); similarly :meth:`~sdialog.Dialog.from_file` (``type='auto'|'txt'|'csv'|'json'``). The default is ``'auto'`` (infers the format from the file extension).


Extensibility Patterns
----------------------
Create new components by subclassing:

- **Persona** variants → subclass :class:`~sdialog.personas.BasePersona`.
- **Orchestrator** → implement ``instruct(dialog, utterance)``.
- **Persistent Orchestrator** → subclass :class:`~sdialog.orchestrators.base.BasePersistentOrchestrator`.
- **Dialog Score** → subclass :class:`~sdialog.evaluation.base.BaseDialogScore` implementing ``score(dialog)``.
- **Dataset Score Evaluator** → subclass :class:`~sdialog.evaluation.base.BaseDatasetScoreEvaluator` implementing ``__eval__`` (and optional ``__plot__``).
- **Embedding Evaluator** → subclass :class:`~sdialog.evaluation.base.BaseDatasetEmbeddingEvaluator` implementing ``__eval__`` (+ ``__plot__`` for visualization).
- **LLM Judge** → subclass :class:`~sdialog.evaluation.base.BaseLLMJudge` implementing ``judge(dialogs)``.

----

Operations & Recipes
====================

Testing & Reproducibility Tips
------------------------------
- Use ``prompt()`` to get the underlying prompt sent to the LLM for any LLM-dependant component (Agent and generators) for paper reporting.
- Use ``json()`` to export any LLM-dependant component (Agent, Persona, Context, Generators) for reproducibility.
- Use ``clone()`` to derive variations while preserving ancestry for auditing/controllability.
- Cache expensive score computations (flow graph building, embedding passes) across experimental sweeps.


Performance Considerations
--------------------------

- Batch evaluation: Pre-build flow graphs once (reuse in multiple scorers) or supply precomputed ``graph`` / ``nodes`` to Flow scores.
- Embedding evaluators: Control batch size & device in SentenceTransformer / BERT evaluators to optimize GPU utilization.
- Caching: Enable disk cache for repeated LLMJudge or flow evaluations over large static corpora.

Common Recipes
--------------

1. **Generate Multi-Variant Persona Set**: :class:`~sdialog.generators.PersonaGenerator` with rule lists and numeric ranges + seed cycle.
2. **Scenario Simulation at Scale**: For each scenario → build Agents → attach orchestrators → run ``dialog_with`` for N seeds.
3. **Quality Filtering Pipeline**: Generate dialogs → apply LLM judges (realism > threshold) → compute flow score percentile → retain top quantile.
4. **Style Harmonization**: Paraphrase dialogs targeting only system speaker with controlled extra instructions.
5. **Activation Steering Study**: Attach :class:`~sdialog.interpretability.Inspector` → collect baseline activations → compute direction (e.g., mean difference) → apply :class:`~sdialog.interpretability.DirectionSteerer` → compare linguistic + refusal metrics pre/post.
