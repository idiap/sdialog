"""
This module provides utilities for loading, parsing, and describing dialogue datasets, including the STAR dataset.
It supports extracting scenarios, flowcharts, personas, and constructing dataset-specific Agent objects for simulation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import json

from tqdm.auto import tqdm

from . import Dialog, Turn, Event
from .agents import Agent
from .personas import Persona
from .orchestrators import InstructionListOrchestrator, SimpleResponseOrchestrator


class STAR:
    """
    Utility class for interacting with the STAR dialogue dataset.

        - Paper: https://arxiv.org/abs/2010.11853
        - Github: https://github.com/RasaHQ/STAR

    Provides methods for loading dialogues, extracting scenarios, flowcharts, responses, and constructing
    Agent objects for simulation and evaluation.
    """

    _path = None
    _speakers = ["User", "Wizard"]

    @staticmethod
    def set_path(path):
        """
        Sets the root path for the STAR dataset.

        :param path: Path to the STAR dataset root directory.
        :type path: str
        :return: None
        :rtype: None
        """
        STAR._path = path

    @staticmethod
    def read_graph(task_name, as_dot: bool = True):
        """
        Read the action graph for a given task.

        :param task_name: Name of the task (folder name under tasks/).
        :type task_name: str
        :param as_dot: If True, return a DOT string; else return the raw graph dict.
        :type as_dot: bool
        :return: Graph in DOT format or raw dictionary mapping edges.
        :rtype: Union[str, dict]
        """
        with open(os.path.join(STAR._path, f"tasks/{task_name}/{task_name}.json")) as reader:
            if not as_dot:
                return json.load(reader)["graph"]
            dot_edges = ";\n".join(f"    {a} -> {b}" for a, b in json.load(reader)["graph"].items())

        return "digraph %s  {\n%s\n}" % (task_name, dot_edges)

    @staticmethod
    def read_graph_responses(task_name, as_dict: bool = False):
        """
        Read example responses associated with each node/action in a task graph.

        Placeholders of the form {variable[:format]} are uppercased for visibility.

        :param task_name: Name of the task.
        :type task_name: str
        :param as_dict: If True, return a dict; otherwise a JSON-formatted string.
        :type as_dict: bool
        :return: Mapping node -> example response, or JSON dump.
        :rtype: Union[dict, str]
        """
        with open(os.path.join(STAR._path, f"tasks/{task_name}/responses.json")) as reader:
            responses = json.load(reader)
            responses = {key: re.sub(r"{(.+?)(?::\w+?)?}", lambda m: m.group(1).upper(), value)
                         for key, value in responses.items()
                         if key != "out_of_scope"}
            return responses if as_dict else json.dumps(responses, indent=2)

    @staticmethod
    def get_task_names():
        """
        List all available task names (directory names under tasks/).

        :return: List of task names.
        :rtype: List[str]
        """
        return [
            task_name
            for task_name in os.listdir(os.path.join(STAR._path, "tasks"))
            if os.path.isdir(os.path.join(STAR._path, "tasks", task_name))
        ]

    @staticmethod
    def get_dialog(id):
        """
        Load a dialogue by numeric ID.

        :param id: Dialogue ID (filename without extension).
        :type id: int
        :return: Dialog object with turns and events populated.
        :rtype: Dialog
        """
        dialog_path = os.path.join(STAR._path, f"dialogues/{id}.json")
        with open(dialog_path) as reader:
            dialog = json.load(reader)

        for e in dialog["Events"]:
            if e["Agent"] == "Wizard":
                e["Agent"] = "System"

        dialog = Dialog(
            id=id,
            scenario=dialog["Scenario"],
            turns=[
                Turn(speaker=e["Agent"], text=e["Text"])
                for e in dialog["Events"]
                if e["Action"] in ["utter", "pick_suggestion"]
            ],
            events=[
                Event(
                    agent=e["Agent"],
                    action=e["Action"],
                    actionLabel=e["ActionLabel"] if "ActionLabel" in e else None,
                    content=e["Text"],
                    timestamp=e["UnixTime"],
                )
                for e in dialog["Events"]
                if "Text" in e
            ],
        )
        dialog._path = dialog_path
        return dialog

    @staticmethod
    def get_dialogs(domain: str = None, task_name: str = None, happy: bool = None, multitask: bool = None):
        """
        Load all dialogues matching optional filter criteria.

        :param domain: Domain filter (must appear in Scenario['Domains']).
        :type domain: Optional[str]
        :param task_name: Task name filter (must appear in WizardCapabilities).
        :type task_name: Optional[str]
        :param happy: Filter by 'happy path' flag.
        :type happy: Optional[bool]
        :param multitask: Filter by multitask flag.
        :type multitask: Optional[bool]
        :return: List of Dialog objects matching filters.
        :rtype: List[Dialog]
        """
        dialogs = []
        for fname in tqdm(os.listdir(os.path.join(STAR._path, "dialogues/")), desc="Reading dialogs", leave=False):
            if not fname.endswith(".json"):
                continue
            dialog_id = int(os.path.splitext(fname)[0])
            scenario = STAR.get_dialog_scenario(dialog_id)

            if (
                (domain is None or domain in scenario["Domains"])
                and (happy is None or scenario["Happy"] == happy)
                and (multitask is None or scenario["MultiTask"] == multitask)
                and (
                    task_name is None
                    or any(capability["Task"] == task_name for capability in scenario["WizardCapabilities"])
                )
            ):
                dialogs.append(STAR.get_dialog(dialog_id))
        return dialogs

    @staticmethod
    def get_dialog_scenario(id):
        """
        Load scenario metadata for a given dialogue.

        :param id: Dialogue ID.
        :type id: int
        :return: Scenario dictionary.
        :rtype: dict
        """
        with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
            return json.load(reader)["Scenario"]

    @staticmethod
    def get_dialog_first_turn(id, speaker: str = None):
        """
        Get the first turn for a given dialogue (optionally constrained to a speaker).

        :param id: Dialogue ID.
        :type id: int
        :param speaker: Speaker name filter (e.g., 'User' or 'Wizard'); if None, first participant turn is returned.
        :type speaker: Optional[str]
        :return: First matching turn or None if not found.
        :rtype: Optional[Turn]
        """
        with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
            for event in json.load(reader)["Events"]:
                turn_speaker = event["Agent"]
                if speaker is None and turn_speaker in STAR._speakers:
                    return Turn(speaker=turn_speaker, text=event["Text"])
                elif turn_speaker == speaker:
                    return Turn(speaker=turn_speaker, text=event["Text"])

    @staticmethod
    def get_dialog_task_names(id):
        """
        Get all task names (WizardCapabilities -> Task) for a dialogue.

        :param id: Dialogue ID.
        :type id: int
        :return: List of task names.
        :rtype: List[str]
        """
        scenario = STAR.get_dialog_scenario(id)
        return [task["Task"] for task in scenario["WizardCapabilities"]]

    @staticmethod
    def get_dialog_responses(id):
        """
        Get example response dictionaries for each task in a dialogue.

        :param id: Dialogue ID.
        :type id: int
        :return: List of response dicts (one per task).
        :rtype: List[dict]
        """
        tasks = STAR.get_dialog_task_names(id)
        return [STAR.read_graph_responses(task, as_dict=True) for task in tasks]

    @staticmethod
    def get_dialog_graphs(id):
        """
        Get raw action graphs (dict form) for all tasks in a dialogue.

        :param id: Dialogue ID.
        :type id: int
        :return: List of graph dicts.
        :rtype: List[dict]
        """
        tasks = STAR.get_dialog_task_names(id)
        return [STAR.read_graph(task, as_dot=False) for task in tasks]

    @staticmethod
    def get_dialog_events(id):
        """
        Get all events for a dialogue.

        :param id: Dialogue ID.
        :type id: int
        :return: List of event dictionaries from the JSON file.
        :rtype: List[dict]
        """
        with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
            return json.load(reader)["Events"]

    @staticmethod
    def get_dialog_user_instructions(id):
        """
        Get user guide instructions mapped to the (user) turn index where each applies.

        :param id: Dialogue ID.
        :type id: int
        :return: Mapping user_turn_index -> instruction text.
        :rtype: Dict[int, str]
        """

        def get_user_n_turns_before(turn_ix, events):
            return len([e for e in events[:turn_ix] if e["Agent"] == "User" and e["Action"] == "utter"])

        events = STAR.get_dialog_events(id)
        return {
            get_user_n_turns_before(ix, events): e["Text"]
            for ix, e in enumerate(events)
            if e["Action"] == "instruct" and e["Agent"] == "UserGuide"
        }

    @staticmethod
    def get_dialog_graphs_and_responses(id):
        """
        Convenience loader returning both graphs and responses for all tasks.

        :param id: Dialogue ID.
        :type id: int
        :return: Tuple (graphs list, responses list).
        :rtype: Tuple[List[dict], List[dict]]
        """
        return STAR.get_dialog_graphs(id), STAR.get_dialog_responses(id)

    @staticmethod
    def get_scenario_description(scenario):
        """
        Build a natural language description of a scenario including embedded DOT graphs and example responses.

        :param scenario: Scenario dictionary (as returned by get_dialog_scenario).
        :type scenario: dict
        :return: Multi-section textual description.
        :rtype: str
        """
        # Let's generate the graph description for each task:
        flowcharts = ""
        for task in scenario["WizardCapabilities"]:
            task_name = task["Task"]
            flowcharts += f"""
The graph for the task '{task_name}' with domain '{task["Domain"]}' is:
```dot
{STAR.read_graph(task_name)}
```
and one example responses for each node is provided in the following json:
```json
{STAR.read_graph_responses(task_name)}
```

# flake8: noqa: E501

---
"""
        # Finally, let's return the scenario object and natural language description for it.
        return f"""The conversation is between a User and a AI assistant in the following domains: {
            ", ".join(scenario["Domains"])
        }.

The User instructions are: {scenario["UserTask"]}
The AI assistant instructions are: {scenario["WizardTask"]}

In addition, the AI assistant is instructed to follow specific flowcharts to address the tasks.
Flowcharts are defined as graph described using DOT.
The actual DOT for the current tasks are:
{flowcharts}

Finally, the following should be considered regarding the conversation:
   1. {
            "The conversation follows the 'happy path', ] meaning the conversations goes according to what it is described in the flowcharts"
            if scenario["Happy"]
            else "The conversation does NOT follow a 'happy path', meaning something happend to the user to change its mind or something happend "
            "in the environment for the conversation to not flow as expected, as described in the flowchart"
        }.
   2. {
            "The user is calling to perform multiple tasks, involving all the tasks defined as flowcharts above ("
            + ", ".join(task["Task"] for task in scenario["WizardCapabilities"])
            + ")"
            if scenario["MultiTask"]
            else "The user is calling to perform only the defined task ("
            + scenario["WizardCapabilities"][0]["Task"]
            + "), nothing else"
        }.
"""  # noqa: E501

    @staticmethod
    def get_dialog_scenario_description(id):
        """
        Retrieve scenario metadata and its natural language description.

        :param id: Dialogue ID.
        :type id: int
        :return: Tuple (scenario dict, description string).
        :rtype: Tuple[dict, str]
        """
        scenario = STAR.get_dialog_scenario(id)
        return scenario, STAR.get_scenario_description(scenario)

    @staticmethod
    def get_user_persona_for_scenario(scenario):
        """
        Construct a Persona object representing the user under the given scenario.

        :param scenario: Scenario metadata.
        :type scenario: dict
        :return: User persona object.
        :rtype: Persona
        """
        dialogue_details = f"""
The following should be considered regarding the conversation:
   1. {
            "The conversation follows a 'happy path', meaning the conversations goes smoothly without any unexpected behavior"
            if scenario["Happy"]
            else "The conversation does NOT follow a 'happy path', meaning you have to simulate something happend in the middle of the conversation, "
            "perhaps you changed your mind at some point or something external happend in the environment for the conversation to not flow as expected"
        }.
   2. {
            "The conversation involves multiple tasks, that is, you want the assistant to perform multiple tasks ("
            + ", ".join(task["Task"] for task in scenario["WizardCapabilities"])
            + "), not just one."
            if scenario["MultiTask"]
            else "The conversation involves only one task you were instructed to ("
            + scenario["WizardCapabilities"][0]["Task"]
            + "), nothing else"
        }"""  # noqa: E501

        return Persona(
            role="user calling a AI assistant that can perform multiple tasks in the following domains: "
            f"{', '.join(scenario['Domains'])}.\n" + dialogue_details,
            circumstances=scenario["UserTask"],
        )

    @staticmethod
    def get_flowchart_description_for_scenario(scenario):
        """
        Build a markdown-like description with DOT graphs and example responses for each task.

        :param scenario: Scenario metadata.
        :type scenario: dict
        :return: Combined flowchart description text.
        :rtype: str
        """
        flowcharts = ""
        for task in scenario["WizardCapabilities"]:
            task_name = task["Task"]
            flowcharts += f"""
## {task_name} ({task["Domain"]})

The flowchart described as an action transition graph for the task '{task_name}' with domain '{task["Domain"]}' is:
```dot
{STAR.read_graph(task_name)}
```
Response example for each action is provided in the following json:
```json
{STAR.read_graph_responses(task_name)}
```
where UPPERCASE words above are just example placeholders. You MUST fill in those with any coherent values in the actual conversation.
"""  # noqa: E501
        return flowcharts

    @staticmethod
    def get_system_persona_for_scenario(scenario):
        """
        Construct a Persona object representing the system/assistant for the scenario.

        :param scenario: Scenario metadata.
        :type scenario: dict
        :return: System persona object.
        :rtype: Persona
        """
        dialogue_details = f"""In the conversation, the AI assistant is instructed to follow specific action flowcharts to address the tasks. Flowcharts are defined as graph described using DOT.
The actual DOT for the current tasks are:
{STAR.get_flowchart_description_for_scenario(scenario)}
"""  # noqa: E501
        return Persona(
            role="AI assistant.\n" + dialogue_details,
            circumstances=scenario["WizardTask"],
        )

    @staticmethod
    def get_agents_for_scenario(scenario, model_name: str = None):
        """
        Create (system, user) Agent objects for a scenario (personas only; no orchestration).

        :param scenario: Scenario metadata.
        :type scenario: dict
        :param model_name: Optional model name / identifier for agent LLM configuration.
        :type model_name: Optional[str]
        :return: Tuple (system_agent, user_agent).
        :rtype: Tuple[Agent, Agent]
        """
        user = Agent(STAR.get_user_persona_for_scenario(scenario), name="User", can_finish=True)

        system = Agent(STAR.get_system_persona_for_scenario(scenario), name="System")

        return system, user

    @staticmethod
    def get_agents_from_dialogue(id, model_name: str = None, set_first_utterance: bool = False):
        """
        Create (system, user) Agent objects derived from a dialogue's scenario.

        Optionally set an initial first system utterance (heuristic).

        :param id: Dialogue ID.
        :type id: int
        :param model_name: Optional model name / identifier for agent LLM configuration.
        :type model_name: Optional[str]
        :param set_first_utterance: If True, assign a first system utterance.
        :type set_first_utterance: bool
        :return: Tuple (system_agent, user_agent).
        :rtype: Tuple[Agent, Agent]
        """
        scenario = STAR.get_dialog_scenario(id)
        system, user = STAR.get_agents_for_scenario(scenario, model_name)

        if set_first_utterance:
            first_turn = STAR.get_dialog_first_turn(id)
            if first_turn.speaker == "Wizard":
                system.set_first_utterances(first_turn.text)
            else:
                system.set_first_utterances("Hello, how can I help?")

        return system, user

    @staticmethod
    def get_agents_from_dialogue_with_orchestration(id, model_name: str = None, set_first_utterance: bool = False):
        """
        Create (system, user) Agent objects with attached orchestrators for responses/instructions.

        :param id: Dialogue ID.
        :type id: int
        :param model_name: Optional model name / identifier for agent LLM configuration.
        :type model_name: Optional[str]
        :param set_first_utterance: If True, assign a first system utterance.
        :type set_first_utterance: bool
        :return: Tuple (system_agent_with_orchestrator, user_agent_with_orchestrator).
        :rtype: Tuple[Agent, Agent]
        """
        system, user = STAR.get_agents_from_dialogue(id, model_name, set_first_utterance)

        graphs, responses = STAR.get_dialog_graphs_and_responses(id)
        response_action_orchestrator = SimpleResponseOrchestrator(responses[0], graph=graphs[0])
        instr_list_orchestrator = InstructionListOrchestrator(STAR.get_dialog_user_instructions(id), persistent=True)

        return system | response_action_orchestrator, user | instr_list_orchestrator
