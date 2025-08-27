Practical Examples
------------------

Basic Persona-based Dialogue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to define two personas, instantiate agents for each persona, and generate a simple dialogue between them using an LLM. The agents will role-play their respective personas and interact for a fixed number of turns.

.. code-block:: python

    from sdialog.personas import Persona
    from sdialog.agents import Agent

    # Define personas
    alice = Persona(name="Alice", role="friendly barista", personality="cheerful and helpful")
    bob = Persona(name="Bob", role="customer", personality="curious and polite")

    # Create agents
    alice_agent = Agent("llama2", persona=alice, name="Alice")
    bob_agent = Agent("llama2", persona=bob, name="Bob")

    # Generate a dialogue
    dialog = alice_agent.dialog_with(bob_agent, max_iterations=10)
    dialog.print()

Multi-Agent Dialogue with Orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to add orchestration to the dialogue generation process. Orchestrators can control aspects such as dialogue length or simulate behaviors like an agent changing its mind. Here, we use both a length orchestrator and a mind-changing orchestrator to influence the assistant agent's behavior.

.. code-block:: python
    from sdialog.personas import Persona
    from sdialog.agents import Agent
    from sdialog.orchestrators import LengthOrchestrator, ChangeMindOrchestrator

    # Define personas
    user = Persona(name="User", role="customer")
    assistant = Persona(name="Assistant", role="support agent")

    # Create agents
    user_agent = Agent("llama2", persona=user, name="User")
    assistant_agent = Agent("llama2", persona=assistant, name="Assistant")

    # Add orchestrators to control dialogue length and simulate mind changes
    length_orch = LengthOrchestrator(min=3, max=6)
    mind_orch = ChangeMindOrchestrator(probability=0.5, reasons=["changed plans", "new information"], max_times=1)
    assistant_agent = assistant_agent | length_orch | mind_orch

    # Generate a dialogue
    dialog = assistant_agent.dialog_with(user_agent, max_iterations=10)
    dialog.print()

Using STAR Dataset Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to use SDialog's STAR dataset utilities. You can set the dataset path, load a dialogue by its ID, print it with scenario information, and extract scenario descriptions and agents for simulation.

.. code-block:: python

    from sdialog.datasets import STAR

    # Set the STAR dataset path
    STAR.set_path("/path/to/star-dataset")

    # Load a dialogue by ID
    dialog = STAR.get_dialog(123)
    dialog.print(scenario=True)

    # Get scenario for a given dialogue ID (123 in this case)
    scenario  = STAR.get_dialog_scenario(123)

    # Get agents for a given scenario
    system_agent, user_agent = STAR.get_agents_for_scenario(scenario, "llama2")

Exporting and Loading Dialogues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to export a generated dialogue to disk in JSON format and later load it for analysis or further processing.

.. code-block:: python

    # Save a dialogue to JSON
    dialog.to_file("output/dialogue_001.json")
    # Save a dialogue to TXT
    dialog.to_file("output/dialogue_001.txt")

    # Load a dialogue from JSON
    from sdialog import Dialog

    dialog = Dialog.from_file("output/dialogue_001.json")
    # dialog = Dialog.from_file("output/dialogue_001.txt")

    dialog.print()

Advanced Usage: Custom Orchestrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example illustrates how to define your own orchestrator by subclassing ``BaseOrchestrator``. The custom orchestrator ensures the agent starts every conversation with a unique greeting.

.. code-block:: python

    from sdialog.orchestrators import BaseOrchestrator

    class CustomGreetingOrchestrator(BaseOrchestrator):
        def instruct(self, dialog, utterance):
            if len(dialog) == 0:
                return "Start the conversation with a unique greeting!"

Attach your orchestrator to an agent:

.. code-block:: python

    agent = Agent("llama2", persona=Persona(name="Bot"))
    agent = agent | CustomGreetingOrchestrator()

Advanced Usage: Scenario-Driven Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to define a scenario and generate a dialogue between agents created from that scenario, ensuring the conversation follows specific paths or constraints.

.. code-block:: python

    scenario = {
        "Domains": ["banking"],
        "UserTask": "Open a new account",
        "WizardTask": "Assist with account opening",
        "Happy": True,
        "MultiTask": False,
        "WizardCapabilities": [{"Task": "open_account", "Domain": "banking"}]
    }

    system_agent, user_agent = STAR.get_agents_for_scenario(scenario, "llama2")
    dialog = system_agent.dialog_with(user_agent, max_iterations=8)
    dialog.print()