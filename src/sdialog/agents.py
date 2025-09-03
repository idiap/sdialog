"""
agents: Agent Definitions for Synthetic Dialogue Generation

This module provides classes for Agents and related utilities for simulating persona-conditioned dialogue
with Large Language Models (LLMs). Agents maintain structured conversation memory, integrate orchestrators
that inject dynamic (persistent or ephemeral) system instructions, and expose inspection / interpretability
hooks for token- and layer-level analysis and optional representation steering.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json
import random
import logging

from time import time
from tqdm.auto import tqdm
from collections import defaultdict
from typing import List, Union, Optional

from langchain_core.messages.base import messages_to_dict
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import config
from jinja2 import Template

from . import Dialog, Turn, Event, Instruction, Context
from .personas import BasePersona, Persona
from .orchestrators import BaseOrchestrator
from .interpretability import UtteranceTokenHook, RepresentationHook, Inspector
from .util import get_llm_model, is_aws_model_name, is_huggingface_model_name, set_generator_seed, get_universal_id

logger = logging.getLogger(__name__)


class Agent:
    """
    Agent that simulates a persona in dialogue using an LLM.

    :cvar STOP_WORD: Special token to indicate end of conversation.
    :vartype STOP_WORD: str
    :cvar STOP_WORD_TEXT: Replacement text for STOP_WORD.
    :vartype STOP_WORD_TEXT: str
    """

    STOP_WORD = "STOP"
    STOP_WORD_TEXT = "(bye bye!)"

    def __init__(self,
                 persona: BasePersona = Persona(),
                 name: Optional[str] = None,
                 model: Union[str, BaseLanguageModel] = None,
                 example_dialogs: Optional[List['Dialog']] = None,
                 context: Optional[Union[str, Context]] = None,
                 dialogue_details: str = "",
                 response_details: str = ("Unless necessary, responses SHOULD be only one utterance long, and SHOULD "
                                          "NOT contain many questions or topics in one single turn."),
                 system_prompt: Optional[str] = None,
                 can_finish: bool = True,
                 orchestrators: Optional[Union[BaseOrchestrator, List[BaseOrchestrator]]] = None,
                 inspectors: Optional[Union['Inspector', List['Inspector']]] = None,
                 postprocess_fn: Optional[callable] = None,
                 **llm_kwargs):
        """
        Initializes a Agent for role-play dialogue.

        :param persona: The persona to role-play.
        :type persona: BasePersona
        :param name: Name of the agent (defaults to persona.name if not provided).
        :type name: Optional[str]
        :param model: The LLM or model name to use (defaults to config["llm"]["model"]).
        :type model: Union[str, BaseLanguageModel], optional
        :param example_dialogs: List of example dialogues as a reference for the agent.
        :type example_dialogs: Optional[List[Dialog]]
        :param context: The context for the agent (optional).
        :type context: Optional[Union[str, Context]]
        :param dialogue_details: Additional details about the dialogue.
        :type dialogue_details: str
        :param response_details: Instructions for response style.
        :type response_details: str
        :param system_prompt: Custom system prompt (optional, otherwise loaded from config).
        :type system_prompt: Optional[str]
        :param can_finish: If True, agent can end the conversation.
        :type can_finish: bool
        :param orchestrators: Orchestrators for agent behavior.
        :type orchestrators: Optional[Union[BaseOrchestrator, List[BaseOrchestrator]]]
        :param inspectors: Inspector(s) to add to the agent.
        :type inspectors: Optional[Union[Inspector, List[Inspector]]]
        :param postprocess_fn: Optional function to postprocess each utterance (input string, output string).
        :type postprocess_fn: Optional[callable]
        :param **llm_kwargs: Additional parameters for the LLM.
        :type llm_kwargs: dict
        """
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}
        if model is None:
            model = config["llm"]["model"]
        self.model_uri = model

        if postprocess_fn is not None and not callable(postprocess_fn):
            raise ValueError("postprocess_fn must be a callable function that takes a string and outputs a string.")

        if not system_prompt:
            with open(config["prompts"]["persona_agent"], encoding="utf-8") as f:
                self.system_prompt_template = Template(f.read())
        self.context = context
        self.example_dialogs = example_dialogs
        self.dialogue_details = dialogue_details
        self.response_details = response_details
        self.can_finish = can_finish

        self.llm = get_llm_model(model_name=model, **llm_kwargs)
        self.name = name if name is not None else getattr(persona, "name", None)
        self.persona = persona
        self.model_name = str(model)  # TODO: improve by adding llm params str(self.llm)
        self.first_utterances = None
        self.finished = False
        self.orchestrators = None
        self.add_orchestrators(orchestrators)
        self.inspectors = None
        self.add_inspectors(inspectors)
        self.postprocess_fn = postprocess_fn
        self.utterance_hook = None
        self.representation_cache = defaultdict(lambda: defaultdict(list))
        self.memory = [SystemMessage(self.system_prompt_template.render(
            persona=self.persona.prompt(),
            context=self.context,
            example_dialogs=self.example_dialogs,
            dialogue_details=self.dialogue_details,
            response_details=self.response_details,
            can_finish=self.can_finish,
            stop_word=self.STOP_WORD
        ))]

        logger.debug(f"Initialized agent '{self.name}' with model '{self.model_name}' "
                     f"using prompt from '{config['prompts']['persona_agent']}'.")
        logger.debug("Prompt: " + self.prompt())

        self.reset()

    def __call__(self, utterance: str = "", return_events: bool = False) -> str:
        """
        Processes an input utterance and generates a response.

        :param utterance: The input utterance from the other agent or user.
        :type utterance: str
        :param return_events: If True, returns a list of events instead of just the response string.
        :type return_events: bool
        :return: The agent's response or events, or None if finished.
        :rtype: Union[str, List[Event], None]
        """
        if self.finished:
            return None

        if utterance:
            self.memory.append(HumanMessage(content=utterance))

        if return_events:
            events = []
        if self.orchestrators:
            for orchestrator in self.orchestrators:
                instruction = orchestrator()
                if instruction:

                    if type(instruction) is Instruction:
                        if return_events and instruction.events:
                            if type(instruction.events) is Event:
                                events.append(instruction.events)
                            else:
                                events.extend(instruction.events)
                        instruction = instruction.text

                    persist = orchestrator.is_persistent()
                    self.instruct(instruction, persist=persist)
                    if return_events:
                        events.append(Event(agent=self.get_name(),
                                            action="instruct" + ("-persist" if persist else ""),
                                            actionLabel=orchestrator.get_event_label(),
                                            text=instruction,
                                            timestamp=int(time())))

        if len(self.memory) <= 1 and self.first_utterances:
            response = (random.choice(self.first_utterances)
                        if type(self.first_utterances) is list
                        else self.first_utterances)
            response = AIMessage(content=response)
        else:
            if self.inspectors:
                self.utterance_hook.new_utterance_event(self.memory_dump())

            if (is_huggingface_model_name(self.model_uri) or is_aws_model_name(self.model_uri)) and \
               (not self.memory or not isinstance(self.memory[-1], HumanMessage)):
                # Ensure that the last message is a HumanMessage to avoid
                # "A conversation must start with a user message" (aws)
                # or "Last message must be a HumanMessage!" (huggingface)
                # from langchain_huggingface (which makes no sense, for ollama is OK but for hugging face is not?)
                # https://github.com/langchain-ai/langchain/blob/6d71b6b6ee7433716a59e73c8e859737800a0a86/libs/partners/huggingface/langchain_huggingface/chat_models/huggingface.py#L726
                response = self.llm.invoke(self.memory + [HumanMessage(
                    content="" if is_huggingface_model_name(self.model_uri) else ".")
                ])
                logger.warning(
                    "For HuggingFace or AWS LLMs, the last message in the conversation history must be a HumanMessage. "
                    "A dummy HumanMessage was appended to memory to satisfy this requirement and prevent errors."
                )
            else:
                response = self.llm.invoke(self.memory)

            if self.inspectors:
                self.utterance_hook.end_utterance_event()

        if self.postprocess_fn:
            response.content = self.postprocess_fn(response.content)

        if self.orchestrators:
            self.memory[:] = [msg for msg in self.memory
                              if not (msg.response_metadata
                                      and "persist" in msg.response_metadata
                                      and not msg.response_metadata["persist"])]
        self.memory.append(response)

        response = response.content
        if self.STOP_WORD in response:
            response = response.replace(self.STOP_WORD, self.STOP_WORD_TEXT).strip()
            self.memory[-1].content = self.memory[-1].content.replace(self.STOP_WORD, "").strip()
            self.finished = True

        if return_events:
            if response:
                events.append(Event(agent=self.get_name(),
                                    action="utter",
                                    text=response,
                                    timestamp=int(time())))
            return events
        else:
            return response if response else ""

    def __or__(self, other):
        """
        Adds enitity to the agent using the | operator.

        :param orchestrator: Orchestrator(s) to add.
        :type orchestrator: Union[BaseOrchestrator, List[BaseOrchestrator]]
        :return: The agent with orchestrators added.
        :rtype: Agent
        """
        if isinstance(other, Inspector):
            self.add_inspectors(other)
        else:
            self.add_orchestrators(other)
        return self

    @property
    def utterance_list(self):
        return self.utterance_hook.utterance_list

    @property
    def base_model(self):
        """
        Return the underlying base (wrapped) model object (e.g., a HuggingFace Transformers model).

        Resolution order:
          1. ChatHuggingFace wrapper: self.llm.llm.pipeline.model
          2. Objects exposing pipeline.model
          3. Objects exposing model

        If none are found, self.llm is returned as a fallback.
        """
        try:
            if hasattr(self.llm, "llm") and hasattr(self.llm.llm, "pipeline"):
                return self.llm.llm.pipeline.model
            if hasattr(self.llm, "pipeline") and hasattr(self.llm.pipeline, "model"):
                return self.llm.pipeline.model
            if hasattr(self.llm, "model"):
                return self.llm.model
        except Exception:
            pass
        return self.llm

    @property
    def tokenizer(self):
        """
        Return the underlying tokenizer object (e.g., a HuggingFace Transformers tokenizer).

        Resolution order:
          1. ChatHuggingFace wrapper: self.llm.llm.tokenizer
          2. Objects exposing pipeline.tokenizer
          3. Objects exposing tokenizer
        """
        try:
            if hasattr(self.llm, "llm") and hasattr(self.llm.llm, "pipeline"):
                return self.llm.llm.pipeline.tokenizer
            if hasattr(self.llm, "pipeline") and hasattr(self.llm.pipeline, "tokenizer"):
                return self.llm.pipeline.tokenizer
            if hasattr(self.llm, "tokenizer"):
                return self.llm.tokenizer
        except Exception:
            pass
        return None

    def response_lookahead(self, utterance: str = None):
        """
        Generates a response to a hypothetical next utterance without updating memory.

        :param utterance: The hypothetical next utterance.
        :type utterance: str
        :return: The predicted response.
        :rtype: str
        """
        if not utterance:
            return self.llm.invoke(self.memory).content
        return self.llm.invoke(self.memory + [HumanMessage(utterance)]).content

    def add_orchestrators(self, orchestrators):
        """
        Adds orchestrators to the agent.

        :param orchestrators: Orchestrator(s) to add.
        :type orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]]
        """
        if not orchestrators:
            return

        if self.orchestrators is None:
            self.orchestrators = []

        if isinstance(orchestrators, BaseOrchestrator):
            orchestrators = [orchestrators]

        self.orchestrators.extend(orchestrators)

        for orchestrator in orchestrators:
            orchestrator._set_target_agent(self)

    def add_inspectors(self, inspectors):
        """
        Adds inspectors to the agent.

        :param inspectors: Inspector(s) to add.
        :type inspectors: Union[Inspector, List[Inspector]]
        """
        if inspectors is None:
            return

        if self.inspectors is None:
            self.inspectors = []

        # Handle both single Inspector and list of Inspectors
        if isinstance(inspectors, Inspector):
            inspectors = [inspectors]
        elif isinstance(inspectors, list):
            inspectors = [ins for ins in inspectors if ins is not None]
            if not inspectors:
                return
        else:
            raise TypeError("inspectors must be an Inspector or a list of Inspectors")

        self.inspectors.extend(inspectors)
        self.set_utterance_hook()
        for inspector in inspectors:
            inspector.add_agent(self)

    def add_hooks(self, key_to_layer_name, steering_function=None, steering_interval=(0, -1)):
        """
        Registers RepresentationHooks for each layer in the given mapping.
        Skips already registered layers. Adds new keys to the shared representation_cache.

        Args:
            key_to_layer_name: Dict mapping cache keys to layer names.
            steering_function: Optional function to apply to the output tensor before caching.
            steering_interval: Tuple `(min_token, max_token)` to control steering.
                                   `min_token` tokens are skipped. Steering stops at `max_token`.
                                   A `max_token` of -1 means no upper limit.
        """
        # Get the model (assume HuggingFace pipeline)
        model = self.base_model
        if model is self.llm:
            raise RuntimeError("Base model not found or not a HuggingFace pipeline.")

        # Always re-initialize cache and hooks
        self.rep_hooks = []

        # Register new hooks
        for cache_key, layer_name in key_to_layer_name.items():
            hook = RepresentationHook(
                cache_key=cache_key,
                layer_key=layer_name,
                agent=self,
                utterance_hook=self.utterance_hook,
                steering_function=steering_function,  # pass the function here,
                steering_interval=steering_interval
            )
            hook.register(model)
            self.rep_hooks.append(hook)

    def clear_orchestrators(self):
        """
        Removes all orchestrators from the agent.
        """
        self.orchestrators = None

    def clear_inspectors(self):
        """
        Removes all inspectors from the agent.
        """
        self.inspectors = None
        self.utterance_hook = None
        self.clear_hooks()

    def clear_hooks(self):
        """
        Resets all representation cached and removes all registered hooks from the agent.
        """
        for hook in getattr(self, 'rep_hooks', []):
            hook.remove()
        self.rep_hooks = []
        if self.utterance_hook is not None:
            self.utterance_hook.reset()
        self.set_utterance_hook()

    def set_utterance_hook(self):
        # Register UtteranceTokenHook and expose utterance_list
        if self.utterance_hook is None:
            self.utterance_hook = UtteranceTokenHook(agent=self)
        self.utterance_hook.register(self.base_model)
        # Automatically set the tokenizer in the hook
        self.utterance_hook.hook_state['tokenizer'] = self.tokenizer

    def instruct(self, instruction: str, persist: bool = False):
        """
        Adds a system instruction to the agent's memory.

        :param instruction: The instruction text.
        :type instruction: str
        :param persist: If True, instruction persists across turns.
        :type persist: bool
        """
        if isinstance(self.memory[-1], HumanMessage):
            # If the last message is a HumanMessage, insert the SystemMessage before it
            # (so the last message is still HumanMessage)
            self.memory.insert(-1, SystemMessage(instruction, response_metadata={"persist": persist}))
        else:
            self.memory.append(SystemMessage(instruction, response_metadata={"persist": persist}))

    def set_first_utterances(self, utterances: Union[str, List[str]]):
        """
        Sets the agent's first utterance(s) for dialogue initialization.

        :param utterances: The greeting(s) to use.
        :type utterances: Union[str, List[str]]
        """
        self.first_utterances = utterances

    def get_name(self, default: str = "Me") -> str:
        """
        Returns the agent's name.

        :return: The agent's name.
        :rtype: str
        """
        return self.name if self.name is not None else default

    def prompt(self) -> str:
        """
        Returns the current system prompt.

        :return: The system prompt.
        :rtype: str
        """
        return self.memory[0].content

    def json(self, string: bool = False, indent=None):
        """
        Serializes the agent's configuration and persona to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :return: The serialized agent.
        :rtype: Union[str, dict]
        """
        data = {}
        if self.name:
            data["name"] = self.get_name()
        data["model_name"] = self.model_name
        if self.first_utterances:
            data["first_utterances"] = self.first_utterances
        data["persona"] = self.persona.json()
        if self.orchestrators:
            data["persona"]["orchestrators"] = [orc.json() for orc in self.orchestrators]
        return json.dumps(data, indent=indent) if string else data

    def reset(self, seed: int = None, context: Union[str, Context] = None, example_dialogs: List['Dialog'] = None):
        """
        Resets the agent's memory and orchestrators, optionally reseeding the LLM.
        Clears the interpretability state (utterance_list and representation_cache).

        :param seed: Random seed for reproducibility.
        :type seed: int
        :param context: Optional context for the agent.
        :type context: Union[str, Context]
        """
        # Remove history
        self.memory[:] = self.memory[:1]
        # Update system prompt if needed
        if self.memory and (context or example_dialogs):
            system_prompt = self.system_prompt_template.render(
                persona=self.persona.prompt(),
                context=context or self.context,
                example_dialogs=example_dialogs or self.example_dialogs,
                dialogue_details=self.dialogue_details,
                response_details=self.response_details,
                can_finish=self.can_finish,
                stop_word=self.STOP_WORD
            )
            self.memory[0].content = system_prompt

        self.finished = False
        seed = set_generator_seed(self, seed)

        if self.orchestrators:
            for orchestrator in self.orchestrators:
                orchestrator.reset()

        if self.utterance_hook is not None:
            self.utterance_hook.reset()

    def dialog_with(self,
                    agent: "Agent",
                    context: Union[str, Context] = None,
                    example_dialogs: List['Dialog'] = None,
                    scenario: Optional[Union[dict, str]] = None,
                    max_turns: int = 200,
                    id: int = None,
                    parent_id: int = None,
                    seed: int = None,
                    notes: str = None,
                    keep_bar: bool = True):
        """
        Simulates a dialogue between this agent and another Agent.

        :param agent: The other agent to converse with.
        :type agent: Agent
        :param context: The context for the dialogue (optional).
        :type context: Optional[Union[str, Context]]
        :param example_dialogs: Example dialogues to guide the conversation (optional).
        :type example_dialogs: Optional[List[Dialog]]
        :param scenario: Optional scenario metadata for the dialogue.
        :type scenario: Optional[Union[dict, str]]
        :param max_turns: Maximum number of dialogue turns.
        :type max_turns: int
        :param id: Dialogue ID.
        :type id: int
        :param parent_id: ID of the parent dialogue, if any.
        :type parent_id: int
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param notes: Optional notes to include in the dialogue.
        :type notes: str
        :param keep_bar: If True, keeps the progress bar visible.
        :type keep_bar: bool
        :return: The generated dialogue object.
        :rtype: Dialog
        """
        seed = seed if seed is not None else random.getrandbits(32)

        random.seed(seed)
        self.reset(seed, context, example_dialogs)
        agent.reset(seed, context, example_dialogs)

        dialog = []
        events = []

        utter = None
        completion = False
        pbar = tqdm(total=max_turns, desc="Dialogue", leave=keep_bar)
        while len(dialog) < max_turns:
            utt_events = self(utter, return_events=True)

            if utt_events and utt_events[-1].action == "utter":
                utter = utt_events[-1].text
                utt_events[-1].text = utter.replace(self.STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].text:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=self.get_name(),
                text=utt_events[-1].text
            ))
            events.extend(utt_events)
            pbar.update(1)

            utt_events = agent(utter, return_events=True)
            if utt_events and utt_events[-1].action == "utter":
                utter = utt_events[-1].text
                utt_events[-1].text = utter.replace(self.STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].text:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=agent.get_name(default="Other"),
                text=utt_events[-1].text
            ))
            events.extend(utt_events)
            pbar.update(1)

        pbar.close()

        context = context or self.context
        return Dialog(
            id=id if id is not None else get_universal_id(),
            parentId=parent_id,
            complete=completion,  # incomplete if ran out of iterations (reached max_iteration number)
            model=self.model_name,
            seed=seed,
            personas={
                self.get_name(): self.persona.json(),
                agent.get_name(default="Other"): agent.persona.json()},
            context=context.json() if context and isinstance(context, Context) else context,
            scenario=scenario,
            notes=notes,
            turns=dialog,
            events=events
        )

    def memory_dump(self, as_dict: bool = False) -> list:
        """
        Returns a copy of the agent's memory (list of messages).
        :return: A copy of the memory list.
        :rtype: list
        """
        return messages_to_dict(self.memory) if as_dict else self.memory.copy()

    talk_with = dialog_with
