A hands-on collection of short, composable examples covering dialog generation, orchestration, evaluation, analysis, and interpretability in SDialog.
Happy experimenting!

-------------
Prerequisites
-------------

Install the library:

.. code-block:: bash

    pip install sdialog

Then in your scripts below make sure to set your default LLM backend, model, and/or parameters, for example:

.. code-block:: python

    import sdialog
    sdialog.config.llm("openai:gpt-4.1", temperature=0.7)

(You may substitute :ref:`any supported backend string <backend_list>`: ``huggingface:...``, ``ollama:...``, ``amazon:...``, ``google:...``.)

----------
Generation
----------

.. _ex-reproducibility:

Reproducibility & Seeding
~~~~~~~~~~~~~~~~~~~~~~~~~

SDialog allows you to control randomness to make generations repeatable.
All generation-based components accept a ``seed`` argument, and will be effective as long as the selected underlying LLM backend supports it.
Some examples:

Agent-based dialogue (role-play) reproducibility:

.. code-block:: python

    # Same seed -> identical dialogue turns (given fixed model + params)
    dialog_a = alice.dialog_with(mentor, max_turns=6, seed=12345)
    dialog_b = alice.dialog_with(mentor, max_turns=6, seed=12345)
    assert [t.text for t in dialog_a.turns] == [t.text for t in dialog_b.turns]

    # Different seed -> likely different variation (if no seed, one is chosen randomly)
    dialog_c = alice.dialog_with(mentor, max_turns=6)

``DialogGenerator`` seeding:

.. code-block:: python

    from sdialog.generators import DialogGenerator

    gen = DialogGenerator("Short friendly greeting between two speakers")
    d1 = gen.generate(seed=42)
    d2 = gen.generate(seed=42)
    # d1.turns == d2.turns (stable under same backend + params)

Persona / Context attribute generation seeding:

.. code-block:: python

    from sdialog.personas import Doctor
    from sdialog.generators import PersonaGenerator

    pg = PersonaGenerator(Doctor(specialty="Cardiology"))
    pg.set(years_of_experience="{5-10}")
    p1 = pg.generate(seed=99)
    p2 = pg.generate(seed=99)   # deterministic attribute choices

Seed value, as long as model and parameters are always logged and saved as part of the metadata of the generated objects (e.g., ``dialog.seed``, ``persona.seed``), which is very useful when they are exported to JSON.

.. _ex-basic-dialog:

Basic Persona-to-Persona Dialogue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a quick dialog between two simple personas using :class:`~sdialog.agents.Agent`.

.. code-block:: python

    from sdialog.personas import Persona
    from sdialog.agents import Agent

    alice = Agent(persona=Persona(name="Alice", role="curious student"), first_utterance="Hi!")
    mentor = Agent(persona=Persona(name="Mentor", role="helpful tutor"))

    dialog = alice.dialog_with(mentor, max_turns=6)
    dialog.print()

Few-Shot Learning with Example Dialogs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SDialog supports in-context few-shot learning by supplying ``example_dialogs`` to generation components. These exemplar dialogs are injected into the system prompt to steer style, structure, tone, or task format.

1. Agent Role-Play with Exemplars

.. code-block:: python

    from sdialog.personas import Persona
    from sdialog.agents import Agent

    # The example dialogues, e.g. real dialogues or handcrafted samples
    my_example_dialogs = [...]

    student = Agent(persona=Persona(name="Learner", role="math student"))
    tutor   = Agent(persona=Persona(name="Guide", role="math tutor"))

    # The exemplar style (concise, explanatory) biases responses
    fewshot_dialog = student.dialog_with(tutor, example_dialogs=my_example_dialogs)
    fewshot_dialog.print()

2. DialogGenerator with Exemplars

.. code-block:: python

    from sdialog.generators import DialogGenerator

    # We can use the from_file() function to load all the dialogues in a folder
    my_example_dialogs = Dialog.from_file("path/to/reference_dialogs/")

    gen = DialogGenerator(
        "Provide a short educational exchange about black holes.",
        example_dialogs=my_example_dialogs  # alternatively, can be also passed in `generate()`
    )
    generated = gen.generate()
    generated.print()

3. PersonaDialogGenerator Few-Shot

.. code-block:: python

    from sdialog.personas import Persona
    from sdialog.generators import PersonaDialogGenerator

    # Alternatively, we can use the type argument to only load specific file types
    my_example_dialogs = Dialog.from_file("path/to/reference_dialogs/",
                                          type="csv")

    p1 = Persona(name="Coach", role="productivity mentor")
    p2 = Persona(name="Client", role="knowledge worker")

    pd_gen = PersonaDialogGenerator(p1, p2,
                                    dialogue_details="Exchange exactly three tips about deep work.",
                                    example_dialogs=my_example_dialogs)
    fewshot_pd = pd_gen.generate()
    fewshot_pd.print()


Multi-Agent Orchestration (Reflex + Length Control)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :mod:`sdialog.orchestrators` to dynamically steer turns. Orchestrators can be composed with the pipe operator.

.. code-block:: python

    from sdialog.orchestrators import SimpleReflexOrchestrator, LengthOrchestrator
    from sdialog.personas import Persona
    from sdialog.agents import Agent

    # Reflex: trigger extra instruction if keyword appears
    reflex = SimpleReflexOrchestrator(
        condition=lambda utt: "deadline" in utt.lower(),
        instruction="Acknowledge the deadline and ask for specifics.")

    # Encourage at least 8, wrap by 12
    length_ctrl = LengthOrchestrator(min=8, max=12)

    planner = Agent(persona=Persona(name="Planner", role="project manager"))
    dev = Agent(persona=Persona(name="Dev", role="engineer"), first_utterance="Any updates?")

    planner = planner | reflex | length_ctrl

    dialog = dev.dialog_with(planner)
    dialog.print(orchestration=True)


.. _advanced_context_persistent_orchestrator:

Advanced Persistent Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This advanced persistent orchestrator demonstrates how to:

- Inspect the entire accumulated dialogue (not only the last utterance).
- Emit different one-time persistent instructions for distinct detected conditions (domain expertise vs emotional context).
- Avoid re-emitting the same instruction thanks to internal state flags.

We will use two :class:`~sdialog.evaluation.LLMJudgeYesNo` judges (LLM-based yes/no classifiers) to detect conditions in the ongoing dialogue:

* ``expertise_judge`` - fires when the other speaker has likely demonstrated professional / domain expertise (mentions of role, advanced methods, implementation specifics, etc.).
* ``sensitive_context_judge`` - fires when recent dialogue turns suggest emotionally sensitive or celebratory life events that warrant tone adaptation.

Both judges are only invoked once (each) thanks to internal flags. After an instruction is emitted it persists automatically (the orchestrator returns ``None`` thereafter), this helps control latency / cost.

.. code-block:: python

    from sdialog.orchestrators import BasePersistentOrchestrator
    from sdialog.evaluation import LLMJudgeYesNo

    class ConversationContextOrchestrator(BasePersistentOrchestrator):
        """Advanced persistent orchestrator using LLM judges.

        Emits at most two persistent instructions:
        1. Domain expertise adaptation (peer-level engagement)
        2. Emotional / situational sensitivity adaptation (empathy or celebration)
        """
        def __init__(self, model: str | None = None):
            super().__init__()
            self.context_set = False
            self.expertise_activated = False

            # Judge templates kept short so they remain inexpensive; they receive the current dialog.
            self.expertise_judge = LLMJudgeYesNo(
                "Has the speaker demonstrated professional domain expertise "
                "(e.g., explicitly stating their job, years of experience, or "
                "discussing implementation/technical methodology) in the dialogue so far?\n\n"
                "Dialogue:\n{{ dialog }}",
                model=model
            )
            self.sensitive_context_judge = LLMJudgeYesNo(
                "Do the recent turns indicate either "
                "(a) a sensitive emotional situation (e.g., loss, illness, personal hardship) or "
                "(b) clearly positive celebratory news (promotion, graduation, birth)?\n\n"
                "Dialogue:\n{{ dialog }}",
                model=model
            )

        def instruct(self, dialog, utterance):
            # Keep only the other speaker's turns in the dialog
            other_speaker_name = [speaker for speaker in dialog.get_speakers()
                                  if speaker != self.agent.name]
            dialog = dialog.filter(other_speaker_name)

            # If no utterances from the other speaker, nothing to do
            if len(dialog) <= 0:
                return None

            # 1) Domain expertise detection
            if not self.expertise_activated:
                result = self.expertise_judge.judge(dialog)
                if result.positive:
                    self.expertise_activated = True
                    return (
                        "The interlocutor has demonstrated domain expertise. "
                        "From now on: engage at a peer level, skip basic explanations, "
                        "focus on advanced concepts, and invite their professional insights."
                    )

            # 2) Emotional / situational sensitivity.
            if not self.context_set:
                result_ctx = self.sensitive_context_judge.judge(dialog)
                if result_ctx.positive:
                    self.context_set = True
                    return (
                        "Recent dialogue content suggests notable emotional "
                        "context (sensitive or celebratory). Acknowledge it explicitly, "
                        "mirror appropriate tone (empathy or enthusiasm), and balance emotional "
                        "support with any informational guidance."
                    )
            return None

        def reset(self):
            super().reset()
            self.context_set = False
            self.expertise_activated = False

    # Example usage where our orchestrator is instantiated with OpenAI's GPT-4o-mini
    ctx_orch = ConversationContextOrchestrator(model="openai:gpt-4o-mini")
    agent = agent | ctx_orch

Explanation: The first time a condition is met we return the instruction; SDialog stores it as a persistent system instruction. Subsequent calls return `None` because once injected the instruction remains in effect automatically.


Attribute Generation (Personas & Contexts)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :class:`~sdialog.generators.PersonaGenerator` and :class:`~sdialog.generators.ContextGenerator` with rule + LLM hybrid specification.

.. code-block:: python

    from sdialog.personas import Doctor, Patient
    from sdialog.generators import PersonaGenerator, ContextGenerator
    from sdialog import Context

    doc_gen = PersonaGenerator(Doctor(specialty="Cardiology"))
    pat_gen = PersonaGenerator(Patient(symptoms="chest pain"))

    # Apply simple attribute rules (random range & list choices)
    doc_gen.set(years_of_experience="{5-15}")
    pat_gen.set(age="{35-70}")

    doctor = doc_gen.generate()
    patient = pat_gen.generate()

    ctx_base = Context(location="Emergency room")
    ctx_gen = ContextGenerator(ctx_base)
    ctx_gen.set(topics=["triage", "diagnosis", "stabilization"],
                goals="{llm:State one succinct medical goal}")
    context = ctx_gen.generate()

    doctor.print(); patient.print(); context.print()

Paraphrasing an Existing Dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Apply :class:`~sdialog.generators.Paraphraser` to rephrase turns (optionally one speaker only).

.. code-block:: python

    from sdialog.generators import Paraphraser

    # Assume `dialog` produced earlier
    paraphraser = Paraphraser(extra_instructions="Lightly simplify wording", target_speaker="Bob")
    dialog_paraphrased = paraphraser(dialog)
    dialog_paraphrased.print()


-----------------------
Evaluation and Analysis
-----------------------

Linguistic Feature Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute readability / style indicators with :class:`~sdialog.evaluation.LinguisticFeatureScore`.

.. code-block:: python

    from sdialog.evaluation import LinguisticFeatureScore

    feat_all = LinguisticFeatureScore()  # all features
    hes_rate = LinguisticFeatureScore(feature="hesitation-rate")

    print(feat_all(dialog))  # dict of metrics
    print(hes_rate(dialog))  # single float

Flow-Based Scores (Perplexity & Likelihood)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use existing dialogs as a reference graph to assess structural fit.

.. code-block:: python

    from sdialog.evaluation import DialogFlowPPL, DialogFlowScore

    reference_dialogs = [...]  # normally a larger corpus
    flow_ppl = DialogFlowPPL(reference_dialogs)
    flow_score = DialogFlowScore(reference_dialogs)

    print("Flow PPL:", flow_ppl(candidate_dialog))
    print("Flow Score:", flow_score(candidate_dialog))

Embedding + Centroid Similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compare candidate dialogs to reference centroid with embeddings.

.. code-block:: python

    from sdialog.evaluation import SentenceTransformerDialogEmbedder, ReferenceCentroidEmbeddingEvaluator

    embedder = SentenceTransformerDialogEmbedder(model_name="sentence-transformers/LaBSE")
    centroid_eval = ReferenceCentroidEmbeddingEvaluator(embedder, reference_dialogs)

    print("Centroid similarity:", centroid_eval([dialog]))

LLM Judges (Realism + Persona Adherence)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Judge realism and persona consistency with built-in yes/no and Likert judges.

.. code-block:: python

    from sdialog.evaluation import LLMJudgeRealDialogLikertScore, LLMJudgePersonaAttributes
    from sdialog.personas import Persona

    realism_judge = LLMJudgeRealDialogLikertScore(reason=True)
    persona_ref = Persona(name="Mentor", role="helpful tutor")
    persona_judge = LLMJudgePersonaAttributes(persona=persona_ref, speaker="Mentor", reason=True)

    realism_result = realism_judge.judge(dialog)
    persona_result = persona_judge.judge(dialog)

    print("Realism score:", realism_result.score, realism_result.reason)
    print("Persona match:", persona_result.positive, persona_result.reason)


Dataset-Level Comparison (Frequency + Mean)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Aggregate metrics over datasets using :class:`~sdialog.evaluation.DatasetComparator`.

.. code-block:: python

    from sdialog.evaluation import FrequencyEvaluator, MeanEvaluator, DatasetComparator, LLMJudgeRealDialog, LinguisticFeatureScore

    judge_real = LLMJudgeRealDialog()
    flesch = LinguisticFeatureScore(feature="flesch-reading-ease")

    comparator = DatasetComparator([
        FrequencyEvaluator(judge_real, name="Realistic rate"),
        MeanEvaluator(flesch, name="Avg Flesch")
    ])

    results = comparator({"modelA": [dialog], "modelB": [dialog_paraphrased]})
    comparator.plot(show=False)  # generate plots silently

Distribution Divergence (KDE / Frechet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compare score distributions with statistical evaluators.

.. code-block:: python

    from sdialog.evaluation import KDEDistanceEvaluator, FrechetDistanceEvaluator

    turn_len_score = LinguisticFeatureScore(feature="mean-turn-length")
    kde_eval = KDEDistanceEvaluator(dialog_score=turn_len_score, reference_dialogues=[dialog])
    frechet_eval = FrechetDistanceEvaluator(dialog_score=turn_len_score, reference_dialogues=[dialog])

    print("KDE divergence:", kde_eval([dialog_paraphrased]))
    print("Frechet distance:", frechet_eval([dialog_paraphrased]))

----------------
Interpretability
----------------

Capturing Activations with an Inspector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Attach :class:`~sdialog.interpretability.Inspector` to an agent to record token-level activations.

.. code-block:: python

    from sdialog.interpretability import Inspector
    from sdialog.agents import Agent
    from sdialog.personas import Persona

    thinker = Agent(persona=Persona(name="Analyzer", role="critic"))
    insp = Inspector(target='model.layers.2.post_attention_layernorm')
    thinker = thinker | insp

    thinker("Summarize the project goals in one sentence.")
    thinker("Refine it further.")

    print("Responses captured:", len(insp))
    first_token_act = insp[-1][0].act  # last response, first token activation

Steering with a Direction Vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use :class:`~sdialog.interpretability.DirectionSteerer` to nudge (add) or ablate (subtract) a semantic direction.

.. code-block:: python

    import torch
    from sdialog.interpretability import DirectionSteerer

    # Assume inspector already attached as `insp`
    direction = torch.randn(first_token_act.shape[-1])  # dummy direction
    steer = DirectionSteerer(direction)

    # Push activations along direction
    insp = insp + steer
    thinker("Provide a concise optimistic remark.")

    # Remove (ablate) the same direction
    insp = insp - steer
    thinker("Provide a concise neutral remark.")

Finding Injected Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inspect dynamic system instructions that were added during a dialog.

.. code-block:: python

    instruct_events = insp.find_instructs(verbose=False)
    for e in instruct_events:
        print(e["index"], e["content"])

Custom Orchestrator + Interpretability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define a bespoke orchestrator to encourage elaboration, while observing effects with an inspector.

.. code-block:: python

    from sdialog.orchestrators import BaseOrchestrator

    class EncourageDetailOrchestrator(BaseOrchestrator):
        def instruct(self, dialog, utterance):
            if utterance and len(utterance.split()) < 5:
                return "Add one more concrete detail in your reply."
            return None

    detail_orch = EncourageDetailOrchestrator()
    explainer_agent = Agent(persona=Persona(name="Explainer", role="assistant"))
    inspector_detail = Inspector(target='model.layers.1.post_attention_layernorm')

    verbose_agent = explainer_agent | detail_orch | inspector_detail

    verbose_dialog = verbose_agent.dialog_with(thinker, max_turns=6)
    verbose_dialog.print(orchestration=True)

Multiple Inspectors (Layer Comparison)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Attach two inspectors to compare early vs late layer activations.

.. code-block:: python

    insp_early = Inspector(target='model.layers.0.post_attention_layernorm')
    insp_late = Inspector(target='model.layers.10.post_attention_layernorm')
    probe_agent = Agent(persona=Persona(name="Probe", role="analyzer")) | insp_early | insp_late
    probe_agent("Explain the purpose of orchestration briefly.")

    print(len(insp_early[-1]), len(insp_late[-1]))  # token counts captured
