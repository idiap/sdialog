"""
This module provides base and concrete classes for orchestrating agent behavior during synthetic dialogue generation.
Orchestrators can inject instructions, control agent responses, and manage dialogue flow for more complex scenarios.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json
import random
import inspect
import numpy as np

from time import time
from abc import ABC, abstractmethod
from typing import List, Union, Dict
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage, AIMessage

from . import Turn, Event, Instruction
from .util import make_serializable
# from .agents import Agent  # Circular import error


class BaseOrchestrator(ABC):
    """
    Base abstract class to create orchestrators that control or influence Agent behavior during dialogue generation.

    Responsibilities:
      * Observe dialogue (agent memory) and produce turn-level instructions.
      * Optionally emit events describing guidance injected.
      * Support persistence across turns when marked persistent.

    Example:

        .. code-block:: python

            from sdialog.orchestrators import BaseOrchestrator
            from sdialog.personas import Persona
            from sdialog.agents import Agent

            # Let's create our own orchestrator
            class EncourageDetailOrchestrator(BaseOrchestrator):
                def instruct(self, dialog, utterance):
                    if utterance and len(utterance.split()) < 5:
                        return "Add a bit more detail in your next reply."
                    return None

            orch_encourage = EncourageDetailOrchestrator()

            bob = Agent(persona=Persona(role="Guide"))
            alice = Agent(persona=Persona(role="User"))

            # Let's orchestrate bob to provide more detailed answers if alice is brief
            bob = bob | orch_encourage

            dialog = bob.talk_with(alice)
            dialog.print(orchestration=True)

    :param target_agent: Agent instance to orchestrate (can be set later).
    :type target_agent: Agent
    :param persistent: Whether produced instructions should persist each turn automatically.
    :type persistent: bool
    :param event_label: Optional label to tag generated events; defaults to class name.
    :type event_label: str
    """

    _target = None
    _event_label = None
    _persistent = False

    def __init__(self, target_agent=None, persistent: bool = None, event_label: str = None):
        """Initialize the orchestrator."""
        self._target = target_agent
        self._persistent = persistent
        self._event_label = event_label

    def __call__(self):
        """
        Produce an instruction for the target agent given current dialog state.

        Uses the agent memory to reconstruct a simplified dialog and calls instruct().

        :return: Instruction object/string or None if no instruction is produced.
        :rtype: Union[str, Instruction, None]
        """
        dialog = self.__get_current_dialog()
        return self.instruct(dialog, dialog[-1].text
                             if dialog and dialog[-1].speaker != self._target.get_name()
                             else "")

    def __str__(self) -> str:
        """
        String representation including constructor arguments.

        :return: Human-readable signature-like string.
        :rtype: str
        """
        data = self.json()
        attrs = " ".join(f"{key}={value}" for key, value in data["args"].items())
        return f"{data['name']}({attrs})"

    def __get_current_dialog(self) -> List[Turn]:
        """
        Reconstruct current dialog from agent memory (excluding system messages).

        :return: List of reconstructed turns.
        :rtype: List[Turn]
        """
        return [Turn(speaker=self._target.get_name() if type(message) is AIMessage else None, text=message.content)
                for message in self._target.memory if type(message) is not SystemMessage]

    def _set_target_agent(self, agent):  # agent: Agent):
        """
        Assign the target agent (late binding).

        :param agent: Agent instance to attach.
        :type agent: Agent
        """
        self._target = agent

    def json(self, string: bool = False, indent: int = None):
        """
        Serialize orchestrator configuration.

        :param string: If True returns JSON string; otherwise a dict.
        :type string: bool
        :param indent: Indentation for pretty JSON output (only if string=True).
        :type indent: int
        :return: Serialized configuration.
        :rtype: Union[str, dict]
        """
        sig = inspect.signature(self.__init__)
        data = {"name": type(self).__name__,
                "args": {key: self.__dict__[key] for key in sig.parameters
                         if key in self.__dict__ and self.__dict__[key] is not None}}
        make_serializable(data["args"])
        return json.dumps(data, indent=indent) if string else data

    def get_event_label(self) -> str:
        """
        Get the label used for events generated by this orchestrator.

        :return: Event label.
        :rtype: str
        """
        return self._event_label if self._event_label else type(self).__name__

    def get_target_agent(self):
        """
        Get the currently assigned target agent.

        :return: Agent instance or None.
        :rtype: Agent
        """
        return self._target

    def is_persistent(self):
        """
        Whether this orchestrator is persistent.

        :return: True if persistent.
        :rtype: bool
        """
        return self._persistent

    def set_persistent(self, value: bool):
        """
        Set persistence flag.

        :param value: New persistence state.
        :type value: bool
        """
        self._persistent = value

    def agent_response_lookahead(self):
        """
        Retrieve the agent's lookahead response (preview of next response if available).

        :return: Lookahead response string.
        :rtype: str
        """
        return self._target.response_lookahead()

    @abstractmethod
    def instruct(self, dialog: List[Turn], utterance: str) -> str:
        """
        *Abstract method:* Subclasses are expected to implement this method.
        Implementations should analyze the dialog state and optionally the most recent utterance
        to produce an instruction for the target agent.

        :param dialog: Current reconstructed dialog (list of turns).
        :type dialog: List[Turn]
        :param utterance: Last opposite-party utterance (may be empty string).
        :type utterance: str
        :return: Instruction text, Instruction object, or None if no action needed.
        :rtype: Union[str, Instruction, None]
        """
        pass

    def reset(self):
        """
        Reset any internal state (overridden in stateful orchestrators).

        :return: None
        :rtype: None
        """
        pass


class BasePersistentOrchestrator(BaseOrchestrator):
    """
    Persistent orchestrator base class to create custom persistent orchestrators.

    Automatically sets persistence to True; intended for orchestrators that maintain state
    across the whole dialogue unless explicitly removed.

    Example:

        .. code-block:: python

            from sdialog.orchestrators import BasePersistentOrchestrator
            from sdialog.personas import Persona
            from sdialog.agents import Agent

            # Let's create our custom persistent orchestrator to permanently flip tone after a trigger word
            class FlipToneOrchestrator(BasePersistentOrchestrator):
                def __init__(self, trigger=None):
                    self.trigger = trigger
                def instruct(self, dialog, utterance):
                    if self.trigger and self.trigger in utterance.lower():
                        return ("From now on adopt an annoyed, curt tone; keep answers short and a bit irritable.")

            # Let's create our agents
            alice = Agent(persona=Persona(name="Alice", role="daughter"))
            bob = Agent(persona=Persona(name="Bob", role="dad"))

            # Let's create our orchestrator using "sweet" as the trigger word
            orchestrator = FlipToneOrchestrator(trigger="sweet")

            # Let's attach the orchestrator to Alice
            alice = alice | orchestrator

            dialog = alice.dialog_with(bob)
            dialog.print(orchestration=True)
    """
    _persistent = True

    @abstractmethod
    def instruct(self, dialog: List[Turn], utterance: str) -> str:
        """
        Persistent variant of :func:`BaseOrchestrator.instruct`.

        :param dialog: Current dialog state.
        :type dialog: List[Turn]
        :param utterance: Last opposite-party utterance.
        :type utterance: str
        :return: Instruction text/object or None.
        :rtype: Union[str, Instruction, None]
        """
        pass

    def reset(self):
        """
        Reset internal persistent state (override as needed).

        :return: None
        :rtype: None
        """
        pass


class SimpleReflexOrchestrator(BaseOrchestrator):
    """
    Simple reflex orchestrator that provides fixed instructions when a condition matches.

    Example:

        .. code-block:: python

            from sdialog.orchestrators import SimpleReflexOrchestrator
            from sdialog.agents import Agent
            from sdialog.personas import Persona

            # If the last utterance contains 'quest', steer to suggest party planning (tutorial style)
            reflex = SimpleReflexOrchestrator(
                condition=lambda utt: "quest" in utt.lower(),
                instruction="Acknowledge the quest idea, then suggest one concrete themed activity."
            )

            bob = Agent(persona=Persona(name="Bob", role="dad")) | reflex
            alice = Agent(persona=Persona(name="Alice", role="daughter"))
            dialog = bob.talk_with(alice)
            dialog.print(orchestration=True)

    :param condition: Predicate function receiving the last utterance; returns True to trigger.
    :type condition: callable
    :param instruction: Instruction text to return when condition is satisfied.
    :type instruction: str
    :param persistent: Whether orchestrator persists across turns.
    :type persistent: bool
    :param event_label: Optional event label override.
    :type event_label: str
    """

    def __init__(self, condition: callable, instruction: str, persistent: bool = False, event_label: str = None):
        """Initialize SimpleReflexOrchestrator."""
        super().__init__(persistent=persistent, event_label=event_label)
        self.condition = condition
        self.instruction = instruction

    def instruct(self, dialog: List[Turn], utterance: str) -> str:
        """
        Return the configured instruction if condition holds.

        :param dialog: Current dialog (unused except for extensibility).
        :type dialog: List[Turn]
        :param utterance: Last opposite-party utterance.
        :type utterance: str
        :return: Instruction text or None.
        :rtype: Union[str, None]
        """
        if self.condition(utterance):
            return self.instruction


class LengthOrchestrator(BaseOrchestrator):
    """
    Orchestrator that encourages continuation or termination based on current number of turns.

    Example:

        .. code-block:: python

            from sdialog.orchestrators import LengthOrchestrator
            from sdialog.personas import Persona
            from sdialog.agents import Agent

            # Keep dialogue going at least 8 turns; try to wrap by turn 12
            len_orch = LengthOrchestrator(min=8, max=12)

            planner = Agent(persona=Persona(name="Planner", role="organizer"))
            guest = Agent(persona=Persona(name="Guest", role="participant"))

            # Attach orchestrator to planner
            planner = planner | len_orch

            dialog = planner.dialog_with(guest)
            dialog.print(orchestration=True)

    :param min: Minimum turns before allowing termination (encourages continuation if not reached).
    :type min: int
    :param max: Maximum turns threshold after which termination is enforced.
    :type max: int
    :param persistent: Whether orchestrator persists.
    :type persistent: bool
    :param event_label: Optional event label.
    :type event_label: str
    """

    def __init__(self, min: int = None, max: int = None, persistent: bool = False, event_label: str = None):
        """Initialize LengthOrchestrator."""
        super().__init__(persistent=persistent, event_label=event_label)
        self.max = max
        self.min = min

    def instruct(self, dialog: List[Turn], utterance: str) -> str:
        """
        Provide an instruction to continue or finish based on dialog length.

        :param dialog: Current dialog state.
        :type dialog: List[Turn]
        :param utterance: Last opposite-party utterance.
        :type utterance: str
        :return: Instruction text or None.
        :rtype: Union[str, None]
        """
        if self.min is not None and len(dialog) < self.min and len(dialog) > 1:
            return "Make sure you DO NOT finish the conversation, keep it going!"
        elif self.max and len(dialog) >= self.max - 1:  # + answer
            return "Now FINISH the conversation AS SOON AS possible, if possible, RIGHT NOW!"


class ChangeMindOrchestrator(BaseOrchestrator):
    """
    Orchestrator that probabilistically injects a 'change your mind' instruction a limited number of times.

    Example:

        .. code-block:: python

            from sdialog.orchestrators import ChangeMindOrchestrator
            from sdialog.personas import Persona
            from sdialog.agents import Agent

            # 40% chance (once) to pivot party theme justification
            changer = ChangeMindOrchestrator(probability=0.4,
                                             reasons=["a better surprise", "budget constraints"],
                                             max_times=1)

            alice = Agent(persona=Persona(name="Alice", role="daughter"))
            bob = Agent(persona=Persona(name="Bob", role="dad"))

            # Attach orchestrator to Bob
            bob = bob | changer

            dialog = alice.dialog_with(bob)
            dialog.print(orchestration=True)

    :param probability: Probability (0-1) each eligible turn to trigger a mind-change.
    :type probability: float
    :param reasons: Optional reason(s) appended; single string or list.
    :type reasons: Union[str, List[str]]
    :param max_times: Maximum number of injections allowed.
    :type max_times: int
    :param persistent: Persistence flag.
    :type persistent: bool
    :param event_label: Event label override.
    :type event_label: str
    """

    def __init__(self, probability: float = 0.3,
                 reasons: Union[str, List[str]] = None,
                 max_times: int = 1,
                 persistent: bool = False,
                 event_label: str = None):
        """Initialize ChangeMindOrchestrator."""
        super().__init__(persistent=persistent, event_label=event_label)
        self.probability = probability
        self.reasons = [reasons] if type(reasons) is str else reasons
        self.max_times = max_times
        self.times = 0

    def reset(self):
        """
        Reset internal counter of times triggered.

        :return: None
        :rtype: None
        """
        self.times = 0

    def instruct(self, dialog: List[Turn], utterance: str) -> str:
        """
        Possibly return a mind-change instruction based on probability and remaining allowance.

        :param dialog: Current dialog state.
        :type dialog: List[Turn]
        :param utterance: Last opposite-party utterance.
        :type utterance: str
        :return: Instruction text or None.
        :rtype: Union[str, None]
        """
        if self.max_times and self.times >= self.max_times:
            return

        if random.random() <= self.probability:
            self.times += 1
            instruction = "Change your mind completely, in your next utterance, suggest something completely different!"
            if self.reasons:
                instruction += f" **Reason:** {random.choice(self.reasons)}."
            return instruction


class SimpleResponseOrchestrator(BaseOrchestrator):
    """
    Orchestrator that suggests next responses based on semantic similarity against a response set (or action graph).

    Example:

        .. code-block:: python

            from sdialog.orchestrators import SimpleResponseOrchestrator
            from sdialog.agents import Agent
            from sdialog.personas import Persona

            canned = [
                "Could you clarify that?",
                "Let me summarize the plan.",
                "That sounds exciting—tell me more.",
                "Maybe we should adjust the theme.",
                "Can you give one concrete example?"
            ]

            sugg = SimpleResponseOrchestrator(responses=canned, top_k=3)

            guide = Agent(persona=Persona(name="Guide", role="facilitator"))
            user = Agent(persona=Persona(name="User", role="participant"))

            # Attach orchestrator to guide
            guide = guide | sugg

            dialog = guide.dialog_with(user)
            dialog.print(orchestration=True)

    :param responses: List (plain strings) or dict (action -> response) entries.
    :type responses: List[Union[str, Dict[str, str]]]
    :param graph: Optional action transition graph (current_action -> next_action).
    :type graph: Dict[str, str]
    :param sbert_model: SentenceTransformer model name.
    :type sbert_model: str
    :param top_k: Number of top similar responses/actions to surface.
    :type top_k: int
    """

    def __init__(self,
                 responses: List[Union[str, Dict[str, str]]],
                 graph: Dict[str, str] = None,
                 #  sbert_model: str = "sentence-transformers/LaBSE",
                 sbert_model: str = "sergioburdisso/dialog2flow-joint-bert-base",
                 top_k: int = 5):
        """Initialize SimpleResponseOrchestrator."""
        self.sent_encoder = SentenceTransformer(sbert_model)
        self.responses = responses
        self.top_k = top_k

        if type(responses) is dict:
            self.resp_utts = np.array([resp for resp in responses.values()])
            self.resp_acts = np.array([act for act in responses.keys()])
            self.graph = graph
        else:
            self.resp_utts = np.array(responses)
            self.resp_acts = None
            self.graph = None

        self.resp_utt_embs = self.sent_encoder.encode(self.resp_utts)

    def instruct(self, dialog: List[Turn], utterance: str) -> str:
        """
        Build an Instruction containing candidate responses (and events for traceability).

        :param dialog: Current dialog.
        :type dialog: List[Turn]
        :param utterance: Last opposite-party utterance (unused directly; similarity uses lookahead / last turn).
        :type utterance: str
        :return: Instruction object with suggestion list.
        :rtype: Instruction
        """
        agent = self.get_target_agent()

        agent_last_turn = None
        if self.graph and dialog:
            for turn in dialog[::-1]:
                if turn.speaker == agent.get_name():
                    agent_last_turn = turn.text
                    break

        response = agent_last_turn if agent_last_turn else agent.response_lookahead()

        events = [Event(agent=agent.get_name(),
                        action="request_suggestions",
                        actionLabel=self.get_event_label(),
                        content=f'Previous response: "{response}"'
                                if agent_last_turn
                                else f'Lookahead response: "{response}"',
                        timestamp=int(time()))]

        sims = self.sent_encoder.similarity(self.sent_encoder.encode(response), self.resp_utt_embs)[0]
        top_k_ixs = sims.argsort(descending=True)[:self.top_k]

        if self.resp_acts is None:
            instruction = ("If applicable, try to pick your next response from the following list: "
                           + "; ".join(f'({ix + 1}) {resp}' for ix, resp in enumerate(self.resp_utts[top_k_ixs])))
        else:
            next_actions = self.resp_acts[top_k_ixs].tolist()
            events.append(Event(agent=agent.get_name(),
                                action="request_suggestions",
                                actionLabel=self.get_event_label(),
                                content="Actions for the response: " + ", ".join(action for action in next_actions),
                                timestamp=int(time())))
            if agent_last_turn:
                next_actions = [self.graph[action] if action in self.graph else action
                                for action in next_actions]
                events.append(Event(agent=agent.get_name(),
                                    action="request_suggestions",
                                    actionLabel=self.get_event_label(),
                                    content="Graph next actions: " + ", ".join(action for action in next_actions),
                                    timestamp=int(time())))

            # TODO: remove repeated actions! (make it a set()?)
            next_actions = [action for action in next_actions if action in self.responses]
            instruction = (
                "If applicable, pick your next response from the following action list in order of importance: "
                + "; ".join(f'({ix + 1}) Action: {action}. Response: "{self.responses[action]}"'
                            for ix, action in enumerate(next_actions))
            )

        return Instruction(text=instruction, events=events)


class InstructionListOrchestrator(BaseOrchestrator):
    """
    Orchestrator that dispenses predefined instructions sequentially or by turn index mapping.

    Example:

        .. code-block:: python

            from sdialog.orchestrators import InstructionListOrchestrator
            from sdialog.personas import Persona
            from sdialog.agents import Agent

            steps = [
                "Greet warmly and ask about preferred theme.",
                "Ask for constraints (budget / space).",
                "Suggest one fitting activity.",
                "Confirm decisions and wrap up politely."
            ]

            coach = Agent(persona=Persona(name="Coach", role="planner"))
            client = Agent(persona=Persona(name="Client", role="requester"))

            # Attach a new instance of InstructionListOrchestrator to the coach
            coach = coach | InstructionListOrchestrator(steps)

            dialog = coach.dialog_with(client)
            dialog.print(orchestration=True)

    :param instructions: Either list (indexed per agent turn) or dict mapping agent turn index -> instruction.
    :type instructions: List[Union[str, Dict[int, str]]]
    :param persistent: Persistence flag.
    :type persistent: bool
    """

    def __init__(self,
                 instructions: List[Union[str, Dict[int, str]]],
                 persistent: bool = False):
        """Initialize InstructionListOrchestrator."""
        super().__init__(persistent=persistent)
        self.instructions = instructions

    def instruct(self, dialog: List[Turn], utterance: str) -> str:
        """
        Return the next scheduled instruction if available.

        :param dialog: Current dialog.
        :type dialog: List[Turn]
        :param utterance: Last opposite-party utterance.
        :type utterance: str
        :return: Instruction text or None.
        :rtype: Union[str, None]
        """
        agent = self.get_target_agent()

        if dialog:
            current_user_len = len([t for t in dialog if t.speaker == agent.get_name()])
        else:
            current_user_len = 0

        if (type(self.instructions) is dict and current_user_len in self.instructions) or \
           (type(self.instructions) is list and current_user_len < len(self.instructions)):
            return self.instructions[current_user_len]
