"""
Base classes for creating custom orchestrators to guide Agent behavior during dialogue generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json
import inspect

from typing import List
from abc import ABC, abstractmethod
from langchain_core.messages import SystemMessage, AIMessage

from .. import Turn
from ..util import make_serializable
# from ..agents import Agent  # Circular import error


class BaseOrchestrator(ABC):
    """
    Base abstract class to create orchestrators that control or influence Agent behavior during dialogue generation.
    Abstract method :func:`instruct` must be implemented by subclasses.

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
    Abstract method :func:`instruct` must be implemented by subclasses.

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
