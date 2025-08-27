The SDialog Library
-------------------

SDialog is organized around several core abstractions and modules, each designed to provide flexibility, extensibility, and ease of use. Below, the main components of the library and their roles in the synthetic dialogue generation workflow are introduced.

1. Dialogue Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the heart of SDialog are data structures that represent the building blocks of a dialogue. These classes provide a standard way to represent utterances, events, and entire conversations, making it easy to manipulate, serialize, and analyze dialogues.

Turn
^^^^

The ``Turn`` class models a single utterance in a dialogue, capturing both the speaker and the content of the utterance. This is the fundamental unit for constructing dialogue sequences.

- **Purpose:** Represents a single utterance in a dialogue.
- **Fields:**

  - ``speaker``: The name or role of the speaker (e.g., "User", "System", "Alice").
  - ``text``: The utterance text for this turn.
- **Usage:** Used to build up the sequence of utterances in a dialogue.

**Example:**

.. code-block:: python

    from sdialog import Turn

    turn = Turn(speaker="Alice", text="Hello, how can I help you?")
    print(turn)

Event
^^^^^

The ``Event`` class captures actions within a dialogue, which may include utterances, instructions, or other events. Events provide metadata for tracking and analyzing the flow and orchestration of conversations.

- **Purpose:** Represents an action in the dialogue, which may be an utterance, instruction, or other event.
- **Fields:**

  - ``agent``: The agent responsible for the event (e.g., "user", "system").
  - ``action``: The type of event (e.g., "utter", "instruct", "pick_suggestion").
  - ``actionLabel``: A label describing the action (e.g., type of instruction).
  - ``text``: The content of the event.
  - ``timestamp``: The Unix timestamp of the event.
- **Usage:** Enables fine-grained tracking of dialogue actions, including orchestration and scenario events.

**Example:**

.. code-block:: python

    from sdialog import Event
    import time

    event = Event(agent="System", action="utter", actionLabel=None, text="Welcome!", timestamp=int(time.time()))
    print(event)

Dialog
^^^^^^

The ``Dialog`` class represents a complete conversation, including its sequence of turns, associated events, and scenario metadata. It provides methods for serialization, pretty-printing, and file I/O, supporting both human-readable and machine-processable formats.

- **Purpose:** Represents a full dialogue, including turns, events, and scenario metadata.
- **Fields:**

  - ``formatVersion``: Version of the dialogue format.
  - ``model``: The model used to generate the dialogue.
  - ``seed``: The random seed used for generation.
  - ``dialogId``: Unique identifier for the dialogue.
  - ``complete``: Whether the dialogue is complete.
  - ``scenario``: Scenario description or metadata (can be a dict or string).
  - ``turns``: List of ``Turn`` objects representing the conversation.
  - ``events``: Optional list of ``Event`` objects for detailed event tracking.
- **Methods:**

  - ``description()``: Returns a human-readable string representation of the dialogue.
  - ``json()``: Serializes the dialogue to JSON or dict.
  - ``print()``: Pretty-prints the dialogue to the console.
  - ``to_file()``, ``from_file()``: Save/load dialogues in JSON or text format.

**Example:**

.. code-block:: python

    from sdialog import Dialog, Turn

    dialog = Dialog(
        turns=[
            Turn(speaker="Alice", text="Hi!"),
            Turn(speaker="Bob", text="Hello, Alice!")
        ]
    )
    print(dialog)
    dialog.print()

----

2. Personas and Agents
~~~~~~~~~~~~~~~~~~~~~~

SDialog enables rich, persona-driven dialogue generation by allowing the definition of detailed character profiles and the simulation of agents that role-play these personas. This abstraction supports the creation of realistic, diverse, and controllable conversational agents.

Persona
^^^^^^^

The ``Persona`` class defines a character profile for role-playing in dialogue generation. It includes fields for name, role, background, personality, circumstances, rules, and language, which are used to generate system prompts and maintain consistent agent behavior.

- **Purpose:** Defines a character profile for role-playing in dialogue generation.
- **Fields:**

  - ``name``: Name of the persona.
  - ``role``: Role or occupation (e.g., "barista", "customer").
  - ``background``: Background information.
  - ``personality``: Personality traits.
  - ``circumstances``: Current circumstances or context.
  - ``rules``: Rules or constraints for the persona.
  - ``language``: Preferred language.
- **Usage:** Used to generate system prompts and maintain consistent agent behavior.

**Example:**

.. code-block:: python

    from sdialog import Persona

    alice = Persona(
        name="Alice",
        role="barista",
        background="Works at a busy coffee shop.",
        personality="cheerful and helpful",
        circumstances="Morning shift",
        rules="Always greet the customer",
        language="English"
    )
    print(alice)

Agent
^^^^^

The ``Agent`` class simulates an agent that role-plays a given Persona using an LLM. It maintains a memory of the conversation, supports orchestration for injecting instructions or controlling behavior, and can be seeded for reproducible dialogue generation.

- **Purpose:** Simulates an agent that role-plays a given Persona using an LLM.
- **Features:**

  - Maintains a memory of the conversation (system, user, and AI messages).
  - Supports orchestration for injecting instructions or controlling behavior.
  - Can be seeded for reproducible dialogue generation.
  - Supports flexible greeting/first utterance configuration.
  - Can serialize its configuration and persona for reproducibility.
- **Methods:**

  - ``__call__()``: Processes an input utterance and generates a response.
  - ``dialog_with()``: Simulates a dialogue with another Agent.
  - ``add_orchestrators()``, ``clear_orchestrators()``: Manage orchestration.
  - ``reset()``: Reset memory and orchestrators.
  - ``json()``: Serialize agent configuration and persona.

**Example:**

.. code-block:: python

    from sdialog.personas import Persona
    from sdialog.agents import Agent

    alice = Persona(name="Alice", role="barista", personality="cheerful")
    bob = Persona(name="Bob", role="customer", personality="curious")

    alice_agent = Agent("llama2", persona=alice, name="Alice")
    bob_agent = Agent("llama2", persona=bob, name="Bob")

    # Simulate a dialogue
    dialog = alice_agent.dialog_with(bob_agent)
    dialog.print()

----

3. Orchestration
~~~~~~~~~~~~~~~~

To enable fine-grained control over dialogue generation, SDialog introduces the concept of orchestrators. Orchestrators are modular components that can inject instructions, enforce constraints, or simulate specific behaviors in agents during a conversation. This section describes the orchestration mechanism and provides examples of built-in orchestrators.

BaseOrchestrator
^^^^^^^^^^^^^^^^

The ``BaseOrchestrator`` is the abstract base class for all orchestrators. It provides methods for generating instructions, managing persistence, event labeling, and serialization. Orchestrators can be attached to a ``Agent`` to influence its behavior during dialogue generation.

- **Purpose:** Abstract base class for all orchestrators.
- **Features:**

  - Can be attached to a Agent.
  - Provides methods for generating instructions, managing persistence, and event labeling.
  - Supports serialization for reproducibility.

**Example:**

.. code-block:: python

    from sdialog.orchestrators import BaseOrchestrator

    class AlwaysSayHelloOrchestrator(BaseOrchestrator):
        def instruct(self, dialog, utterance):
            if len(dialog) == 0:
                return "Say 'Hello!' as your first utterance."

Example Orchestrators
^^^^^^^^^^^^^^^^^^^^^

SDialog provides several built-in orchestrators for common dialogue control patterns. These orchestrators can be used to trigger instructions based on conditions, control dialogue length, simulate mind changes, suggest responses, or provide a sequence of instructions.

- **SimpleReflexOrchestrator:** Triggers instructions based on a condition (e.g., if a certain keyword is present in the utterance).

  **Example:**

  .. code-block:: python

      from sdialog.orchestrators import SimpleReflexOrchestrator

      # Instruct agent to apologize if the word "problem" appears in the user's utterance
      orch = SimpleReflexOrchestrator(
          condition=lambda utt: "problem" in utt.lower(),
          instruction="Apologize for the inconvenience."
      )

- **LengthOrchestrator:** Controls dialogue length by providing instructions to continue or finish the conversation based on the number of turns.

  **Example:**

  .. code-block:: python

      from sdialog.orchestrators import LengthOrchestrator

      length_orch = LengthOrchestrator(min=3, max=6)

- **ChangeMindOrchestrator:** Simulates agents changing their mind, optionally with a list of reasons and a probability.

  **Example:**

  .. code-block:: python

      from sdialog.orchestrators import ChangeMindOrchestrator

      mind_orch = ChangeMindOrchestrator(probability=0.5, reasons=["changed plans", "new information"], max_times=1)

- **SimpleResponseOrchestrator:** Suggests responses based on similarity to a set of possible responses, using sentence embeddings.

  **Example:**

  .. code-block:: python

      from sdialog.orchestrators import SimpleResponseOrchestrator

      responses = ["Sure, I can help!", "Could you clarify?", "Thank you for your patience."]
      resp_orch = SimpleResponseOrchestrator(responses)

- **InstructionListOrchestrator:** Provides a sequence of instructions at specific turns, useful for simulating guided user behavior.

  **Example:**

  .. code-block:: python

      from sdialog.orchestrators import InstructionListOrchestrator

      instructions = ["Greet the assistant.", "Ask about the weather.", "Say thank you and goodbye."]
      instr_list_orch = InstructionListOrchestrator(instructions)

**Usage Example:**

.. code-block:: python

    from sdialog.personas import Persona
    from sdialog.agents import Agent
    from sdialog.orchestrators import LengthOrchestrator

    assistant = Persona(name="Assistant", role="support agent")
    assistant_agent = Agent("llama2", persona=assistant, name="Assistant")
    length_orch = LengthOrchestrator(min=3, max=6)
    assistant_agent = assistant_agent | length_orch  # Add orchestrator using the | operator

----

4. Dialogue Generation
~~~~~~~~~~~~~~~~~~~~~~

SDialog provides high-level generators to automate the creation of synthetic dialogues, either between arbitrary personas or following specific scenario instructions. These generators leverage LLMs and the abstractions above to produce realistic, structured conversations.

DialogGenerator
^^^^^^^^^^^^^^^

The ``DialogGenerator`` class generates synthetic dialogues using an LLM, given dialogue details and output format. It supports arbitrary system and user prompts, output schemas, and reproducibility via seeding.

- **Purpose:** Generates synthetic dialogues using an LLM, given dialogue details and output format.
- **Features:**

  - Supports arbitrary system and user prompts.
  - Can be configured with output schemas (e.g., Pydantic models).
  - Handles seeding and prompt management for reproducibility.

**Example:**

.. code-block:: python

    from sdialog.generators import DialogGenerator

    details = "Generate a conversation between a customer and a barista about ordering coffee."
    generator = DialogGenerator("llama2", dialogue_details=details)
    dialog = generator()
    dialog.print()

PersonaDialogGenerator
^^^^^^^^^^^^^^^^^^^^^^

The ``PersonaDialogGenerator`` class generates dialogues between two personas, enforcing role-play and scenario constraints. It automatically constructs system prompts for both personas and ensures the dialogue starts with a greeting and follows scenario instructions.

- **Purpose:** Generates dialogues between two personas, enforcing role-play and scenario constraints.
- **Features:**

  - Automatically constructs system prompts for both personas.
  - Ensures the dialogue starts with a greeting and follows scenario instructions.
  - Supports scenario metadata and output formatting.

**Example:**

.. code-block:: python

    from sdialog.generators import PersonaDialogGenerator, Persona

    persona_a = Persona(name="Alice", role="barista")
    persona_b = Persona(name="Bob", role="customer")

    generator = PersonaDialogGenerator("llama2", persona_a, persona_b)
    dialog = generator()
    dialog.print()

----

5. Datasets and Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~

SDialog includes utilities for working with external datasets and for managing complex conversational scenarios. This enables reproducible research and the simulation of realistic, goal-driven dialogues.

STAR Dataset Utilities
^^^^^^^^^^^^^^^^^^^^^^

The STAR dataset utilities provide functions for loading, parsing, and describing dialogues, scenarios, flowcharts, and personas from the STAR dataset. These tools support scenario-driven dialogue generation and analysis.

- **Purpose:** Provides functions for loading, parsing, and describing dialogues, scenarios, flowcharts, and personas from the STAR dataset.
- **Features:**

  - Load dialogues by ID, filter by domain, task, or scenario attributes.
  - Extract scenario descriptions, flowcharts (in DOT format), and example responses.
  - Construct Agent objects for simulation and evaluation.
  - Support for scenario-driven dialogue generation and analysis.

**Example:**

.. code-block:: python

    from sdialog.datasets import STAR

    STAR.set_path("/path/to/star-dataset")
    dialog = STAR.get_dialog(123)
    dialog.print(scenario=True)

    # Get scenario description and flowcharts
    scenario, description = STAR.get_dialog_scenario_description(123)
    print(description)

    # Get agents for a scenario
    system_agent, user_agent = STAR.get_agents_for_scenario(scenario, "llama2")

Scenario Management
^^^^^^^^^^^^^^^^^^^

Scenario management tools in SDialog allow for the generation of natural language descriptions of scenarios, extraction and visualization of flowcharts, and construction of personas and agents based on scenario metadata.

- **Purpose:** Easily describe and manage dialogue scenarios, including flowcharts and user/system goals.
- **Features:**

  - Generate natural language descriptions of scenarios.
  - Extract and visualize flowcharts for tasks.
  - Construct personas and agents based on scenario metadata.

**Example:**

.. code-block:: python

    scenario = {
        "Domains": ["banking"],
        "UserTask": "Open a new account",
        "WizardTask": "Assist with account opening",
        "Happy": True,
        "MultiTask": False,
        "WizardCapabilities": [{"Task": "open_account", "Domain": "banking"}]
    }

    from sdialog.datasets import STAR
    system_agent, user_agent = STAR.get_agents_for_scenario(scenario, "llama2")
    dialog = system_agent.dialog_with(user_agent)
    dialog.print()

----

6. Utilities
~~~~~~~~~~~~

To support the full workflow, SDialog provides utility functions for serialization, pretty-printing, and file I/O. These tools make it easy to save, load, and visualize dialogues for downstream tasks and analysis.

Serialization
^^^^^^^^^^^^^

The serialization utilities in SDialog allow for exporting dialogues and events as JSON or plain text for downstream tasks, training, or analysis. Flexible file I/O is supported for saving and loading dialogues.

- **Export dialogues and events** as JSON or plain text for downstream tasks, training, or analysis.
- **Flexible file I/O**: Save and load dialogues using ``Dialog.to_file()`` and ``Dialog.from_file()``.

**Example:**

.. code-block:: python

    # Save a dialogue to JSON
    dialog.to_file("output/dialogue_001.json")
    # Save a dialogue to TXT
    dialog.to_file("output/dialogue_001.txt")

    # Load a dialogue from JSON
    from sdialog import Dialog

    dialog = Dialog.from_file("output/dialogue_001.json")
    # dialog = Dialog.from_file("output/dialogue_001.txt")


Pretty-printing
^^^^^^^^^^^^^^^

SDialog provides pretty-printing utilities to visualize dialogues in the console with color-coded speakers and events for easy inspection and debugging. Scenario and orchestration visualization is also supported.

- **Visualize dialogues** in the console with color-coded speakers and events for easy inspection and debugging.
- **Scenario and orchestration visualization**: Print scenario metadata and orchestration events alongside dialogue turns.

**Example:**

.. code-block:: python

    dialog.print(scenario=True, orchestration=True)
