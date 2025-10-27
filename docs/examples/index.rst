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

----------------
Audio Generation
----------------

Quick Audio Generation
~~~~~~~~~~~~~~~~~~~~~~~
Let's start with the simplest way to generate audio from your dialogues! SDialog provides convenient one-function audio generation that handles everything automatically.

.. code-block:: python

    from sdialog.audio.pipeline import to_audio
    from sdialog import Dialog

    # Load an existing dialogue
    dialog = Dialog.from_file("path/to/your/dialog.json")
    
    # Generate complete audio in one call
    audio_dialog = to_audio(
        dialog,
        do_step_1=True,  # Combine utterances into single audio
        do_step_2=True,  # Generate dSCAPER timeline with background effects
        do_step_3=True,  # Apply room acoustics simulation
        audio_file_format="mp3"  # or "wav", "flac"
    )
    
    # Access generated files
    print(f"Combined audio: {audio_dialog.audio_step_1_filepath}")
    print(f"Timeline audio: {audio_dialog.audio_step_2_filepath}")
    print(f"Room acoustics: {audio_dialog.audio_step_3_filepaths}")

**Using Dialog's built-in method**:

.. code-block:: python

    # Convert dialog directly to audio using the built-in method
    audio_dialog = dialog.to_audio(
        do_step_1=True,
        do_step_2=True, 
        do_step_3=True
    )
    
    # Play the generated audio (in Jupyter notebooks)
    from IPython.display import Audio, display
    
    if audio_dialog.audio_step_1_filepath:
        display(Audio(audio_dialog.audio_step_1_filepath, autoplay=False))

Room Generation and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SDialog provides powerful room generation capabilities for creating realistic acoustic environments. Let's explore different room types and configurations!

**Medical Room Generator** - Create specialized medical environments:

.. code-block:: python

    from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole
    
    # Generate different types of medical rooms
    generator = MedicalRoomGenerator()
    
    # Various medical room types
    consultation_room = generator.generate({"room_type": RoomRole.CONSULTATION})
    examination_room = generator.generate({"room_type": RoomRole.EXAMINATION})
    # ... other room types available: TREATMENT, PATIENT_ROOM, SURGERY, etc.
    
    # Get room properties
    print(f"Room area: {examination_room.get_square_meters():.1f} m²")
    print(f"Room volume: {examination_room.get_volume():.1f} m³")

**Basic Room Generator** - Create simple rectangular rooms:

.. code-block:: python

    from sdialog.audio.room_generator import BasicRoomGenerator
    
    # Generate rooms with different sizes
    generator = BasicRoomGenerator(seed=123)  # For reproducible results
    
    small_room = generator.generate({"room_size": 8})   # 8 m²
    large_room = generator.generate({"room_size": 20})  # 20 m²
    
    print(f"Small room: {small_room.get_square_meters():.1f} m²")
    print(f"Large room: {large_room.get_square_meters():.1f} m²")

**Room Visualization** - Visualize room layouts and configurations:

.. code-block:: python

    # Generate and visualize a room
    room = MedicalRoomGenerator().generate({"room_type": RoomRole.EXAMINATION})
    
    # Create detailed visualization
    img = room.to_image(
        show_anchors=True,
        show_walls=True,
        show_furnitures=True,
        show_speakers=True,
        show_microphones=True
    )
    
    # Display or save the image
    img.show()  # Display in notebook
    img.save("room_layout.png")  # Save to file

**Microphone Positioning** - Configure microphone placement:

.. code-block:: python

    from sdialog.audio.room import Room, MicrophonePosition, Position3D, Dimensions3D
    
    # Different microphone positions
    room = Room(
        name="Demo Room",
        dimensions=Dimensions3D(width=10, length=10, height=3),
        mic_position=MicrophonePosition.CHEST_POCKET_SPEAKER_1
    )
    
    # Position microphone on desk
    room_with_desk = Room(
        name="Office Room",
        dimensions=Dimensions3D(width=5, length=4, height=3),
        mic_position=MicrophonePosition.DESK_SMARTPHONE,
        furnitures={
            "desk": Furniture(
                name="desk",
                x=2.0, y=2.0,
                width=1.5, height=0.8, depth=1.0
            )
        }
    )
    
    # Custom 3D position
    room_custom = Room(
        name="Custom Mic Room",
        dimensions=Dimensions3D(width=8, length=6, height=3),
        mic_position=MicrophonePosition.CUSTOM,
        mic_position_3d=Position3D(x=4.0, y=3.0, z=1.5)
    )

Voice Database Management
~~~~~~~~~~~~~~~~~~~~~~~~~
SDialog supports multiple voice database types for flexible voice selection. Let's explore how to work with different voice sources!

**HuggingFace Voice Databases** - Use pre-trained voice collections:

.. code-block:: python

    from sdialog.audio.voice_database import HuggingfaceVoiceDatabase
    
    # LibriTTS voices
    voices_libritts = HuggingfaceVoiceDatabase("sdialog/voices-libritts")
    
    # Kokoro voices
    voices_kokoro = HuggingfaceVoiceDatabase("sdialog/voices-kokoro")
    
    # Get voice statistics
    print(voices_kokoro.get_statistics(pretty=True))
    
    # Select voices based on characteristics
    female_voice = voices_libritts.get_voice(gender="female", age=25, seed=42)
    # Prevent voice reuse
    male_voice = voices_libritts.get_voice(gender="male", age=30, keep_duplicate=False)
    
    # Reset used voices for reuse
    voices_libritts.reset_used_voices()

**Local Voice Databases** - Use your own voice files:

.. code-block:: python

    from sdialog.audio.voice_database import LocalVoiceDatabase
    
    # Create database from local files with CSV metadata
    voice_database = LocalVoiceDatabase(
        directory_audios="./my_custom_voices/",
        metadata_file="./my_custom_voices/metadata.csv"
    )
    
    # Add custom voices programmatically
    voice_database.add_voice(
        gender="female",
        age=42,
        identifier="french_female_42",
        voice="./my_custom_voices/french_female_42.wav",
        lang="french",
        language_code="f"
    )
    
    # Get voice by language and prevent voice reuse
    french_voice = voice_database.get_voice(gender="female", age=20, lang="french", keep_duplicate=False)
    
    # Get statistics
    print(voice_database.get_statistics(pretty=True))

**Quick Voice Database** - Create databases from dictionaries:

.. code-block:: python

    from sdialog.audio.voice_database import VoiceDatabase
    
    # Create database from predefined voice list
    quick_voices = VoiceDatabase(
        data=[
            {
                "voice": "am_echo",
                "language": "english",
                "language_code": "a",
                "identifier": "am_echo",
                "gender": "male",
                "age": 20
            },
            {
                "voice": "af_heart",
                "language": "english", 
                "language_code": "a",
                "identifier": "af_heart",
                "gender": "female",
                "age": 25
            }
        ]
    )
    
    # Use the voices
    male_voice = quick_voices.get_voice(gender="male", age=20)
    female_voice = quick_voices.get_voice(gender="female", age=25)
    
    # Unavailable voice for this language (an error will be raised)
    try:
        female_voice_spanish = quick_voices.get_voice(gender="female", age=25, lang="spanish")
    except ValueError as e:
        print("Expected error:", e)


Impulse Response Database and Microphone Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SDialog allows you to simulate different microphone effects by convolving audio with impulse responses.
You can use a local database or one from the Hugging Face Hub.

**Using a Local Impulse Response Database**:

.. code-block:: python

    from sdialog.audio.processing import AudioProcessor
    from sdialog.audio.impulse_response_database import LocalImpulseResponseDatabase, RecordingDevice
    import soundfile as sf
    import numpy as np

    # Create a dummy metadata file and audio file for the example
    with open("metadata.csv", "w") as f:
        f.write("identifier,file_name\\n")
        f.write("my_ir,my_ir.wav\\n")
    sf.write("my_ir.wav", np.random.randn(16000), 16000)

    # Initialize the database
    impulse_response_database = LocalImpulseResponseDatabase(
        metadata_file="metadata.csv",
        directory="."
    )
    # Assume input.wav exists
    sf.write("input.wav", np.random.randn(16000 * 3), 16000)

    AudioProcessor.apply_microphone_effect(
        input_audio_path="input.wav",
        output_audio_path="output_mic_effect.wav",
        device="my_ir", # or RecordingDevice.SHURE_SM57 for built-in devices
        impulse_response_database=impulse_response_database
    )


**Using a HuggingFace Impulse Response Database**:

.. code-block:: python

    from sdialog.audio.impulse_response_database import HuggingFaceImpulseResponseDatabase

    # This requires the 'datasets' library
    hf_db = HuggingFaceImpulseResponseDatabase(repo_id="your_username/your_ir_dataset")
    ir_path = hf_db.get_ir("some_ir_identifier")


Advanced Audio Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
For more control over the audio generation process, let's use the full AudioPipeline with custom configurations!

**Complete Audio Pipeline with Room Acoustics**:

.. code-block:: python

    from sdialog.audio import AudioDialog, KokoroTTS, HuggingfaceVoiceDatabase
    from sdialog.audio.pipeline import AudioPipeline
    from sdialog.audio.room import DirectivityType
    from sdialog.audio.utils import SourceVolume, SourceType, Role
    from sdialog.audio.jsalt import MedicalRoomGenerator, RoomRole
    from sdialog.personas import Persona
    from sdialog.agents import Agent

    # 1. Create a base text dialogue
    doctor = Persona(name="Dr. Smith", role="doctor", age=40, gender="male", language="english")
    patient = Persona(name="John", role="patient", age=45, gender="male", language="english")
    
    doctor_agent = Agent(persona=doctor)
    patient_agent = Agent(persona=patient, first_utterance="Hello doctor, I have chest pain.")
    
    dialog = patient_agent.dialog_with(doctor_agent, max_turns=6)
    
    # 2. Convert to audio dialogue
    audio_dialog = AudioDialog.from_dialog(dialog)
    
    # 3. Configure TTS engine and voice database
    tts_engine = KokoroTTS(lang_code="a")  # American English
    voice_database = HuggingfaceVoiceDatabase("sdialog/voices-kokoro")
    
    # 4. Setup audio pipeline
    audio_pipeline = AudioPipeline(
        voice_database=voice_database,
        tts_pipeline=tts_engine,
        dir_audio="./audio_outputs"
    )
    
    # 5. Generate a medical examination room
    room = MedicalRoomGenerator().generate(args={"room_type": RoomRole.EXAMINATION})
    
    # 6. Position speakers around furniture in the room
    room.place_speaker_around_furniture(
        speaker_name=Role.SPEAKER_1, 
        furniture_name="desk", 
        max_distance=1.0
    )
    room.place_speaker_around_furniture(
        speaker_name=Role.SPEAKER_2, 
        furniture_name="desk", 
        max_distance=1.0
    )
    
    # 7. Set microphone directivity
    room.set_directivity(direction=DirectivityType.OMNIDIRECTIONAL)
    
    # 8. Run the complete audio pipeline
    audio_dialog = audio_pipeline.inference(
        audio_dialog,
        environment={
            "room": room,
            "background_effect": "white_noise",
            "foreground_effect": "ac_noise_minimal",
            "source_volumes": {
                SourceType.ROOM: SourceVolume.HIGH,
                SourceType.BACKGROUND: SourceVolume.VERY_LOW
            },
            "kwargs_pyroom": {
                "ray_tracing": True,
                "air_absorption": True
            }
        },
        do_step_1=True,  # Combine utterances into a single dialogue audio
        do_step_2=True,  # Generate dSCAPER timeline
        do_step_3=True,  # Apply room acoustics simulation
        dialog_dir_name="medical_consultation",
        room_name="examination_room"
    )
    
    # 9. Access the generated audio files
    print(f"Combined utterances: {audio_dialog.audio_step_1_filepath}")
    print(f"DScaper timeline: {audio_dialog.audio_step_2_filepath}")
    print(f"Room acoustics simulation: {audio_dialog.audio_step_3_filepaths}")

**Speaker and Furniture Placement** - Position speakers around furniture:

.. code-block:: python

    from sdialog.audio.utils import SpeakerSide, Role
    from sdialog.audio.room import Room, Dimensions3D, MicrophonePosition

    room = Room(
        name="Demo Room with Speakers and Furniture",
        dimensions=Dimensions3D(width=10, length=10, height=3),
        mic_position=MicrophonePosition.CEILING_CENTERED
    )
    
    # Add furniture to room
    room.add_furnitures({
        "lamp": Furniture(
            name="lamp",
            x=6.5, y=1.5,
            width=0.72, height=1.3, depth=0.72
        ),
        "chair": Furniture(
            name="chair",
            x=2.5, y=4.5,
            width=0.2, height=1.3, depth=0.2
        )
    })
    
    # Position speakers around furniture
    room.place_speaker_around_furniture(
        speaker_name=Role.SPEAKER_1, 
        furniture_name="lamp"
    )
    room.place_speaker_around_furniture(
        speaker_name=Role.SPEAKER_2, 
        furniture_name="chair",
        max_distance=2.0,
        side=SpeakerSide.BACK
    )
    
    # Calculate distances
    distances = room.get_speaker_distances_to_microphone(dimensions=2)
    print(f"Speaker 2D distances to the microphone: {distances}")

Multilingual Audio Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SDialog supports multilingual audio generation with custom TTS engines. Let's create a custom TTS engine for Spanish!

**Custom TTS Engine** - Create your own TTS implementation:

.. code-block:: python

    import torch
    import numpy as np
    from sdialog.audio.tts_engine import BaseTTS
    
    class XTTSEngine(BaseTTS):
        def __init__(self, lang_code: str = "en", model="xtts_v2"):
            from TTS.api import TTS
            self.lang_code = lang_code
            self.pipeline = TTS(model).to("cuda" if torch.cuda.is_available() else "cpu")
        
        def generate(self, text: str, voice: str) -> tuple[np.ndarray, int]:
            wav_data = self.pipeline.tts(
                text=text,
                speaker_wav=voice,
                language=self.lang_code
            )
            return (wav_data, 24000)
    
    # Use custom TTS for Spanish
    spanish_tts = XTTSEngine(lang_code="es")
    
    # Create spanish voice database
    spanish_voices = LocalVoiceDatabase(
        directory_audios="./spanish_voices/",
        metadata_file="./spanish_voices/metadata.csv"
    )
    
    # Generate Spanish audio
    audio_pipeline = AudioPipeline(
        voice_database=spanish_voices,
        tts_pipeline=spanish_tts,
        dir_audio="./spanish_audio_outputs"
    )

    spanish_dialog = AudioDialog.from_dialog(dialog)
    
    spanish_audio = audio_pipeline.inference(
        spanish_dialog,
        do_step_1=True,
        do_step_2=True,
        do_step_3=True,
        dialog_dir_name="spanish_dialogue"
    )

**Language-specific Voice Assignment**:

.. code-block:: python

    from sdialog.audio.utils import Role
    
    # Assign specific voices from your voice database for different languages
    spanish_voices = {
        Role.SPEAKER_1: ("spanish_male_1", "spanish"),
        Role.SPEAKER_2: ("spanish_female_1", "spanish")
    }
    
    spanish_audio = audio_pipeline.inference(
        spanish_dialog,
        voices=spanish_voices
    )

Custom Room Generator
~~~~~~~~~~~~~~~~~~~~~~
Ready to create your own specialized room types? Let's build a custom room generator for warehouses!

.. code-block:: python

    from sdialog.audio.room import Room
    from sdialog.audio.utils import Furniture, RGBAColor
    from sdialog.audio.room_generator import RoomGenerator, Dimensions3D
    import random
    import time
    
    class WarehouseRoomGenerator(RoomGenerator):
        def __init__(self):
            super().__init__()
            self.ROOM_SIZES = {
                "big_warehouse": ([1000, 2500], 0.47, "big_warehouse"),
                "small_warehouse": ([100, 200, 300], 0.75, "small_warehouse"),
            }
        
        def generate(self, args):
            warehouse_type = args["warehouse_type"]
            floor_area, reverberation_ratio, name = self.ROOM_SIZES[warehouse_type]
            
            # Calculate dimensions
            dims = Dimensions3D(width=20, length=25, height=10)
            
            room = Room(
                name=f"Warehouse: {name}",
                dimensions=dims,
                reverberation_time_ratio=reverberation_ratio,
                furnitures={
                    "door": Furniture(
                        name="door",
                        x=0.10, y=0.10,
                        width=0.70, height=2.10, depth=0.5
                    )
                }
            )
            return room
    
    # Use custom generator
    warehouse_gen = WarehouseRoomGenerator()
    warehouse = warehouse_gen.generate({"warehouse_type": "big_warehouse"})
    
    print(f"Warehouse area: {warehouse.get_square_meters():.1f} m²")
    print(f"Warehouse volume: {warehouse.get_volume():.1f} m³")
