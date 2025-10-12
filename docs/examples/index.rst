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
Let's start with something fun and straightforward—creating a simple dialogue between two personas! We'll use :class:`~sdialog.agents.Agent` to bring these characters to life and watch them have a conversation.

.. code-block:: python

    from sdialog.personas import Persona
    from sdialog.agents import Agent

    alice = Agent(persona=Persona(name="Alice", role="curious student"), first_utterance="Hi!")
    mentor = Agent(persona=Persona(name="Mentor", role="helpful tutor"))

    dialog = alice.dialog_with(mentor, max_turns=6)
    dialog.print()

Few-Shot Learning with Example Dialogs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now let's explore one of SDialog's most powerful features! We can guide our dialogues by providing examples that show the system what style, structure, or format we want. This technique, called few-shot learning, works by supplying ``example_dialogs`` to generation components. These exemplar dialogs are injected into the system prompt to steer tone, task format, and conversation flow.

1. Agent Role-Play with Exemplars

Here's how we can influence the conversation style by providing example dialogues:

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

Let's see how we can use example dialogues to guide the generation of entirely new conversations:

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

Now let's combine personas with few-shot learning to create highly targeted conversations:

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
Ready to add some intelligence to our conversations? Let's explore how to use :mod:`sdialog.orchestrators` to dynamically steer dialogue turns. Think of orchestrators as conversation directors—they watch what's happening and can give instructions to guide the flow. The best part? We can combine multiple orchestrators using the pipe operator!

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

Advanced Orchestrator: LLM Judges + Persistent Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's build something really sophisticated! In this advanced example, we'll create a persistent orchestrator that demonstrates the full power of SDialog's orchestration capabilities. Our orchestrator will:

- Inspect the entire accumulated dialogue (not just the last utterance)
- Use LLM judges to detect different conversation contexts
- Emit different persistent instructions for distinct detected conditions
- Avoid re-emitting the same instruction thanks to internal state management

We'll use two :class:`~sdialog.evaluation.LLMJudgeYesNo` judges (LLM-based yes/no classifiers) to detect interesting patterns in our ongoing dialogue:

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
                "discussing implementation/technical methodology) in the dialogue so far?",
                model=model
            )
            self.sensitive_context_judge = LLMJudgeYesNo(
                "Do the recent turns indicate either "
                "(a) a sensitive emotional situation (e.g., loss, illness, personal hardship) or "
                "(b) clearly positive celebratory news (promotion, graduation, birth)?",
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
Let's dive into creating diverse characters and settings! We'll see how to use :class:`~sdialog.generators.PersonaGenerator` and :class:`~sdialog.generators.ContextGenerator` with a hybrid approach that combines rules with LLM intelligence for flexible content generation.

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
Sometimes we have a good dialogue but want to refine or adapt it for different audiences or styles. Let's see how to apply :class:`~sdialog.generators.Paraphraser` to rephrase conversations, with the option to target specific speakers only.

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
Let's start measuring the quality of our dialogues! We'll compute readability and style indicators using :class:`~sdialog.evaluation.LinguisticFeatureScore` to understand how our generated conversations compare to natural human speech.

.. code-block:: python

    from sdialog.evaluation import LinguisticFeatureScore

    feat_all = LinguisticFeatureScore()  # all features
    hes_rate = LinguisticFeatureScore(feature="hesitation-rate")

    print(feat_all(dialog))  # dict of metrics
    print(hes_rate(dialog))  # single float

Flow-Based Scores (Perplexity & Likelihood)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now let's evaluate how well our dialogues follow natural conversation patterns! We can use existing dialogues as a reference to assess structural fit and flow.

.. code-block:: python

    from sdialog.evaluation import DialogFlowPPL, DialogFlowScore

    reference_dialogs = [...]  # normally a larger corpus
    flow_ppl = DialogFlowPPL(reference_dialogs)
    flow_score = DialogFlowScore(reference_dialogs)

    print("Flow PPL:", flow_ppl(candidate_dialog))
    print("Flow Score:", flow_score(candidate_dialog))

Embedding + Centroid Similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's explore semantic similarity! We can compare our candidate dialogues to a reference centroid using embeddings to understand how semantically close they are to our target conversations.

.. code-block:: python

    from sdialog.evaluation import SentenceTransformerDialogEmbedder, ReferenceCentroidEmbeddingEvaluator

    embedder = SentenceTransformerDialogEmbedder(model_name="sentence-transformers/LaBSE")
    centroid_eval = ReferenceCentroidEmbeddingEvaluator(embedder, reference_dialogs)

    print("Centroid similarity:", centroid_eval([dialog]))

LLM Judges: Realism + Persona Adherence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now let's bring in the big guns—LLM judges! These are sophisticated evaluators that can assess dialogue realism and check whether characters stay true to their personas throughout the conversation.

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


Custom LLM Judges: Document Relevance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ready to build your own evaluation logic? In this example, we'll explore how to create custom LLM judges that can evaluate document relevance in the context of a dialogue. This is particularly useful when you want to assess whether retrieved information or generated content actually matches what users are discussing.

Let's start by setting up our scenario with a sample dialogue and two documents (one relevant, one not):

.. code-block:: python

    from sdialog import Dialog

    # Example dialogue where the user is looking for a laptop
    dialog = Dialog.from_str("""
    AI: Hello! How can I assist you today?
    Robert: Hi! I'm looking for a new laptop. Can you help me find one?
    AI: Of course! What are your main requirements for the laptop?
    Robert: I need it for gaming and graphic design, so it should have a powerful GPU and a high-resolution display.
    AI: Got it. Do you have a preferred brand or budget in mind?""")

    # Document with relevant information matching user needs
    good_document = """
    The latest gaming laptops come with powerful GPUs and high-resolution displays.
    They are designed to handle demanding tasks like gaming and graphic design with ease.
    Many models also offer customizable options to fit your specific needs and budget.
    """

    # Document with irrelevant information not matching user needs (e.g., about cooking)
    bad_document = """
    Cooking is an essential skill that everyone should learn.
    It allows you to prepare healthy meals at home and can be a fun hobby.
    Many people enjoy experimenting with new recipes and ingredients.
    """

Example 1: Yes/No relevance judgment with reasoning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start with a binary approach! We'll create a judge using :class:`~sdialog.evaluation.LLMJudgeYesNo` that determines whether a document is relevant to what the user was discussing in the dialogue. The cool thing about SDialog is that we can add any placeholder in our judge template and then pass their values when calling ``judge()``. In this example, we'll use ``{{ document }}`` as a placeholder to pass a document to be evaluated (you're free to add as many placeholders with whatever names as needed).

By setting ``reason=True``, we enable reasoning, which provides detailed explanations along with the binary verdict.

.. code-block:: python

    from sdialog.evaluation import LLMJudgeYesNo

    # Let's create our custom yes/no judge
    doc_rel = LLMJudgeYesNo(
        "Does this document relate to what the user was looking for?\n\n"
        "Document:\n{{ document }}",
        reason=True  # Enable explanations
    )

    # Let's use our judge to evaluate our example documents
    good_result = doc_rel.judge(dialog, document=good_document)
    bad_result = doc_rel.judge(dialog, document=bad_document)

    print("Good document verdict:", good_result.positive)
    print("Good document reason:", good_result.reason)
    print("---")
    print("Bad document verdict:", bad_result.positive)
    print("Bad document reason:", bad_result.reason)

Output:
::

    Good document verdict: True
    Good document reason: The document directly addresses the user's stated needs - a laptop with a powerful GPU and high-resolution display for gaming and graphic design - as expressed in the dialogue.
    ---
    Bad document verdict: False
    Bad document reason: The document discusses cooking, while the user is looking for information about laptops. These topics are unrelated.


Example 2: Likert-style relevance score with reasoning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now let's explore a more nuanced approach! Instead of binary yes/no decisions, we can use numerical scores to evaluate relevance on a scale using :class:`~sdialog.evaluation.LLMJudgeScore`. In this example, we'll implement a 1-5 Likert scale judgment that provides a more granular assessment of document relevance.

.. code-block:: python

    from sdialog.evaluation import LLMJudgeScore

    # Let's create now our custom score judge
    doc_rel = LLMJudgeScore(
        "From 1 to 5, how closely does the following document match user needs?\n\n"
        "Document:\n{{ document }}",
        reason=True
    )

    # Again, let's use our judge to evaluate our example documents
    good_result = doc_rel.judge(dialog, document=good_document)
    bad_result = doc_rel.judge(dialog, document=bad_document)

    print("Good document score:", good_result.score)
    print("Good document reason:", good_result.reason)
    print("---")
    print("Bad document score:", bad_result.score)
    print("Bad document reason:", bad_result.reason)

Output:
::

    Good document score: 5
    Good document reason: The document directly addresses the user's needs as expressed in the dialogue. Robert specifically states he needs a laptop for gaming and graphic design with a powerful GPU and high-resolution display, and the document highlights these exact features in gaming laptops. It's a perfect match for the user's stated requirements.
    ---
    Bad document score: 1
    Bad document reason: The provided document discusses cooking, while the user dialogue is about finding a laptop. There is absolutely no overlap in topic or user need fulfillment; therefore, the match is extremely poor.

Sometimes we only need the numerical score without the detailed explanation. In such cases, we can call the judge object directly to return just the score value:

.. code-block:: python

    print("Good document score:", doc_rel(dialog, document=good_document))
    print("Bad document score:", doc_rel(dialog, document=bad_document))


Dataset-Level Comparison (Frequency + Mean)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When working with multiple datasets or comparing different models, we need comprehensive evaluation tools! Let's see how we can aggregate metrics using :class:`~sdialog.evaluation.DatasetComparator` to get a bird's-eye view of our results.

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
Let's get statistical! To compare score distributions between different dialogue sets, we can use sophisticated statistical evaluators that measure how different our generated content is from reference material.

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
Ready to peek inside the mind of our language model? Let's start by exploring how to attach :class:`~sdialog.interpretability.Inspector` to an agent to record token-level activations. This gives us a window into what the model is "thinking" as it generates responses!

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
Now for the exciting part—actually influencing the model's behavior! We can use :class:`~sdialog.interpretability.DirectionSteerer` to nudge (add) or ablate (subtract) semantic directions in the model's internal representations.

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
Let's trace what's happening under the hood! We can inspect dynamic system instructions that were added during a dialogue to understand how orchestrators influenced the conversation.

.. code-block:: python

    instruct_events = insp.find_instructs(verbose=False)
    for e in instruct_events:
        print(e["index"], e["content"])

Custom Orchestrator + Interpretability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's bring it all together! We'll combine concepts by defining a custom orchestrator to encourage elaboration, while observing its effects with an inspector. This demonstrates how orchestration and interpretability work hand-in-hand.

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
For our final trick, let's attach multiple inspectors to compare what's happening at different depths in the model! This can reveal how information flows and transforms through the neural network layers.

.. code-block:: python

    insp_early = Inspector(target='model.layers.0.post_attention_layernorm')
    insp_late = Inspector(target='model.layers.10.post_attention_layernorm')
    probe_agent = Agent(persona=Persona(name="Probe", role="analyzer")) | insp_early | insp_late
    probe_agent("Explain the purpose of orchestration briefly.")

    print(len(insp_early[-1]), len(insp_late[-1]))  # token counts captured
