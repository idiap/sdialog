"""
This module provides classes for Agents and related utilities for simulating persona-conditioned dialogue
with Large Language Models (LLMs). Agents maintain structured conversation memory, integrate orchestrators
that inject dynamic (persistent or ephemeral) system instructions, and expose inspection / interpretability
hooks for token- and layer-level analysis and optional representation steering.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import json
import random
import logging

from time import time
from tqdm.auto import tqdm
from collections import defaultdict
from typing import List, Union, Optional, Tuple

from langchain_core.tools import tool
from langchain_core.messages.base import messages_to_dict
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import config
from jinja2 import Template

from . import Dialog, Turn, Event, Instruction, Context
from .personas import BasePersona
from .orchestrators import BaseOrchestrator
from .interpretability import ResponseHook, ActivationHook, Inspector
from .util import get_llm_model, is_amazon_model_name, is_huggingface_model_name, set_generator_seed, get_universal_id

logger = logging.getLogger(__name__)


class Agent:
    """
    Agent that simulates a persona-driven conversational actor using an LLM.

    This class wraps:

     * A persona (traits / role)
     * Optional context + exemplar dialogues
     * Orchestrators (dynamic / persistent injected instructions)
     * Interpretability hooks (token / layer events, steering)
     * Simple dialogue loop utilities (dialog_with)

    Example:

        .. code-block:: python

            from sdialog import Persona, Context
            from sdialog.agents import Agent

            # Create two agents
            user = Agent(persona=Persona(name="Dr. Nebula",
                                         role="Astrobotanist seeking alien spores"),
                         name="Scientist")
            bot = Agent(persona=Persona(name="StationCore",
                                        role="Sarcastic habitat control AI"),
                        name="Bot")

            # Create an (optional) context for the conversation
            context = Context(location="Orbiting Research Station Theta-9",
                              environment="Zero-gravity greenhouse",
                              objects=["alien spores", "hydroponic garden", "research equipment"])

            # Create a dialogue
            dialog = user.dialog_with(bot, context=context)

            # Print dialog
            dialog.print()

    :param persona: The persona to role-play.
    :type persona: BasePersona
    :param name: Name of the agent (defaults to persona.name if not provided).
    :type name: Optional[str]
    :param context: Optional default context for the agent's conversations.
    :type context: Optional[Union[str, Context]]
    :param first_utterance: Optional fixed first utterance or list of possible first utterances.
    :type first_utterance: Optional[Union[str, List[str]]]
    :param dialogue_details: Additional details about the dialogue.
    :type dialogue_details: str
    :param response_details: Instructions for response style.
    :type response_details: str
    :param example_dialogs: Optional list of default example dialogues as a reference for the agent.
    :type example_dialogs: Optional[List[Dialog]]
    :param tools: List of functions to be used as tools by the agent (if supported by the LLM).
    :type tools: Optional[List[callable]]
    :param think: If True, enables "thinking" segments in responses (if supported by the LLM).
    :type think: bool
    :param thinking_pattern: Regex pattern to manually identify "thinking" segments in responses.
    :type thinking_pattern: Optional[str]
    :param can_finish: If True, agent can end the conversation.
    :type can_finish: bool
    :param orchestrators: Orchestrators for agent behavior.
    :type orchestrators: Optional[Union[BaseOrchestrator, List[BaseOrchestrator]]]
    :param inspectors: Inspector(s) to add to the agent.
    :type inspectors: Optional[Union[Inspector, List[Inspector]]]
    :param preprocessing_fn: Optional function to preprocess each input utterance before calling the LLM
                                (input string, output string).
    :type preprocessing_fn: Optional[callable]
    :param postprocess_fn: Optional function to postprocess each output utterance after calling the LLM
                            (input string, output string).
    :type postprocess_fn: Optional[callable]
    :param system_prompt: Custom system prompt to use as-is (takes precedence over persona;
                          if provided, persona is disabled and this prompt is used directly).
    :type system_prompt: Optional[str]
    :param model: The LLM or model name to use (defaults to config["llm"]["model"]).
    :type model: Union[str, BaseLanguageModel], optional
    :param llm_kwargs: Additional parameters for the LLM.
    :type llm_kwargs: dict
    """
    _STOP_WORD = "STOP"
    _STOP_WORD_TEXT = "(bye bye!)"

    def __init__(self,
                 persona: BasePersona = None,
                 name: Optional[str] = None,
                 context: Optional[Union[str, Context]] = None,
                 first_utterance: Optional[Union[str, List[str]]] = None,
                 dialogue_details: str = "",
                 response_details: str = ("Unless necessary, responses SHOULD be only one utterance long, and SHOULD "
                                          "NOT contain many questions or topics in one single turn."),
                 example_dialogs: Optional[List['Dialog']] = None,
                 tools: Optional[List] = None,
                 think: bool = False,
                 thinking_pattern: Optional[str] = r"<think>(.*?)</think>",
                 can_finish: bool = True,
                 orchestrators: Optional[Union[BaseOrchestrator, List[BaseOrchestrator]]] = None,
                 inspectors: Optional[Union['Inspector', List['Inspector']]] = None,
                 preprocessing_fn: Optional[callable] = None,
                 postprocess_fn: Optional[callable] = None,
                 system_prompt: Optional[str] = None,
                 model: Union[str, BaseLanguageModel] = None,
                 **llm_kwargs):
        """
        Initializes an Agent for role-play dialogue.
        """
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}
        if model is None:
            model = config["llm"]["model"]
        if postprocess_fn is not None and not callable(postprocess_fn):
            raise ValueError("postprocess_fn must be a callable function that takes a string and outputs a string.")
        if preprocessing_fn is not None and not callable(preprocessing_fn):
            raise ValueError("preprocessing_fn must be a callable function that takes a string and outputs a string.")

        # Handle system_prompt parameter - if provided, it takes precedence over persona
        if system_prompt:
            if persona:
                logger.warning("Both system_prompt and persona provided. system_prompt takes precedence; "
                               "persona will be ignored.")
            persona = None  # Disable persona logic when custom system_prompt is provided
            system_prompt_template = None
        else:
            with open(config["prompts"]["persona_agent"], encoding="utf-8") as f:
                system_prompt_template = Template(f.read() if persona else "")

        # Private attributes
        self._system_prompt_template = system_prompt_template
        self._thinking_pattern = thinking_pattern
        self._tools = {fn.__name__: tool(fn) for fn in tools} if tools else None
        self._model_uri = model
        self._context = context
        self._example_dialogs = example_dialogs
        self._dialogue_details = dialogue_details
        self._response_details = response_details
        self._can_finish = can_finish
        self._first_utterances = first_utterance
        self._finished = False
        self._orchestrators = None
        self._inspectors = None
        self._postprocess_fn = postprocess_fn
        self._preprocessing_fn = preprocessing_fn
        self._hook_response_data = None
        self._hook_response_act = defaultdict(lambda: defaultdict(list))

        # Public attributes
        self.llm, llm_kwargs = get_llm_model(model_name=model,
                                             return_model_params=True,
                                             think=think,
                                             tools=list(self._tools.values()) if self._tools else None,
                                             **llm_kwargs)
        self.model_info = {"name": str(model), **llm_kwargs}
        self.name = name if name is not None else getattr(persona, "name", None)
        self.persona = persona

        # Initialize memory based on whether we have a custom system_prompt or persona
        if system_prompt:
            self.memory = [SystemMessage(system_prompt)]
        elif persona and self._system_prompt_template:
            self.memory = [SystemMessage(self._system_prompt_template.render(
                persona=self.persona.prompt(),
                context=self._context,
                example_dialogs=self._example_dialogs,
                dialogue_details=self._dialogue_details,
                response_details=self._response_details,
                can_finish=self._can_finish,
                stop_word=self._STOP_WORD
            ))]
        else:
            self.memory = []

        if system_prompt:
            logger.debug(f"Initialized agent '{self.name}' with model '{str(model)}' "
                         f"using custom system prompt (persona disabled).")
            logger.debug("Prompt: " + self.prompt())
        elif persona:
            logger.debug(f"Initialized agent '{self.name}' with model '{str(model)}' "
                         f"using prompt from '{config['prompts']['persona_agent']}'.")
            logger.debug("Prompt: " + self.prompt())
        else:
            logger.debug("Initialized agent with no persona and no system prompt.")

        self.add_orchestrators(orchestrators)
        self.add_inspectors(inspectors)
        self.reset()

    @property
    def _hooked_responses(self):
        """
        Generated responses as list of {"mem": {...}, "output": ...} dicts.
        This attribute is meant to be used internally by the attached Inspector object(s) only.
        """
        return self._hook_response_data.responses

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
        if self._finished:
            return None

        if utterance:
            utterance = self._preprocessing_fn(utterance) if self._preprocessing_fn else utterance
            self.memory.append(HumanMessage(content=utterance))

        if return_events:
            events = []
        if self._orchestrators:
            for orchestrator in self._orchestrators:
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
                                            content=instruction,
                                            timestamp=int(time())))

        if len(self.memory) <= 1 and self._first_utterances:
            response = (random.choice(self._first_utterances)
                        if type(self._first_utterances) is list
                        else self._first_utterances)
            response = AIMessage(content=response)
            response_events = None
        else:
            if self._inspectors:
                self._hook_response_data.response_begin(self.memory_dump())

            if (is_huggingface_model_name(self._model_uri) or is_amazon_model_name(self._model_uri)) and \
               (not self.memory or not isinstance(self.memory[-1], HumanMessage)):
                # Ensure that the last message is a HumanMessage to avoid
                # "A conversation must start with a user message" (aws)
                # or "Last message must be a HumanMessage!" (huggingface)
                # from langchain_huggingface (which makes no sense, for ollama is OK but for hugging face is not?)
                # https://github.com/langchain-ai/langchain/blob/6d71b6b6ee7433716a59e73c8e859737800a0a86/libs/partners/huggingface/langchain_huggingface/chat_models/huggingface.py#L726
                response, response_events = self._get_llm_response(self.memory + [HumanMessage(
                    content="" if is_huggingface_model_name(self._model_uri) else ".")
                ])
                logger.warning(
                    "For HuggingFace or AWS LLMs, the last message in the conversation history must be a HumanMessage. "
                    "A dummy HumanMessage was appended to memory to satisfy this requirement and prevent errors."
                )
            else:
                response, response_events = self._get_llm_response(self.memory)

            if self._inspectors:
                self._hook_response_data.response_end()

        if self._postprocess_fn:
            response.content = self._postprocess_fn(response.content)

        if return_events and response_events:
            events.extend(response_events)

        if self._orchestrators:
            self.memory[:] = [msg for msg in self.memory
                              if not (msg.response_metadata
                                      and "persist" in msg.response_metadata
                                      and not msg.response_metadata["persist"])]
        self.memory.append(response)

        response = response.content
        if self._STOP_WORD in response:
            response = response.replace(self._STOP_WORD, self._STOP_WORD_TEXT).strip()
            self.memory[-1].content = self.memory[-1].content.replace(self._STOP_WORD, "").strip()
            self._finished = True

        if return_events:
            if response:
                events.append(Event(agent=self.get_name(),
                                    action="utter",
                                    content=response,
                                    timestamp=int(time())))
            return events
        else:
            return response if response else ""

    def _get_llm_response(self, messages, update_tool_memory: bool = False) -> Tuple[AIMessage, List[Event]]:
        """
        Generate a single LLM turn from the given message history, handling tool-calls and
        extracting optional "thinking" traces.

        :param messages: The message history to send to the LLM.
        :type messages: List[BaseMessage]
        :param update_tool_memory: If True, appends tool results to the agent's memory.
        :type update_tool_memory: bool
        :return: The LLM response and list of events.
        :rtype: Tuple[AIMessage, List[Event]]
        """
        events = []
        thinking = None

        response = self.llm.invoke(messages)

        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            thinking = response.additional_kwargs.get("reasoning_content", None)

        if self._thinking_pattern:
            think_segments = re.findall(self._thinking_pattern, response.content, flags=re.DOTALL)
            if think_segments:
                think_segments = "\n".join(think_segments)
                if thinking:
                    thinking += "\n" + think_segments
                else:
                    thinking = think_segments
                response.content = re.sub(self._thinking_pattern, "", response.content, flags=re.DOTALL).strip()

        if thinking:
            events.append(Event(agent=self.get_name(),
                                action="think",
                                content=thinking,
                                timestamp=int(time())))

        if hasattr(response, "tool_calls") and response.tool_calls and self._tools:
            messages_n = len(messages)
            for tool_call in response.tool_calls:
                events.append(Event(agent=self.get_name(),
                                    action="tool",
                                    actionLabel="call",
                                    content={"name": tool_call["name"],
                                             "args": tool_call["args"],
                                             "id": tool_call["id"]},
                                    timestamp=int(time())))
                if tool_call["name"] in self._tools:
                    selected_tool = self._tools[tool_call["name"]]
                    tool_msg = selected_tool.invoke(tool_call)
                    messages.append(tool_msg)
                    if update_tool_memory:
                        self.memory.append(tool_msg)
                    events.append(Event(agent=self.get_name(),
                                        action="tool",
                                        actionLabel="output",
                                        content={"name": tool_msg.name,
                                                 "output": tool_msg.content,
                                                 "call_id": tool_msg.tool_call_id},
                                        timestamp=int(time())))
                else:
                    logger.warning(f"Tool '{tool_call['name']}' not found among bound tools.")

            # If tools were called, re-query the LLM with the updated messages
            if messages_n != len(messages):
                response, new_events = self._get_llm_response(messages, update_tool_memory)
                events.extend(new_events)
                return response, events

        if self._postprocess_fn:
            response.content = self._postprocess_fn(response.content)

        return response, events

    def __or__(self, other):
        """
        Overloaded | operator to attach orchestration or interpretability components.

        :param other: BaseOrchestrator, list[BaseOrchestrator], Inspector or list[Inspector].
        :type other: Union[BaseOrchestrator, List[BaseOrchestrator], Inspector, List[Inspector]]
        :return: Self (Agent) for chaining.
        """
        if isinstance(other, Inspector):
            self.add_inspectors(other)
        else:
            self.add_orchestrators(other)
        return self

    def _add_activation_hooks(self, key_to_layer_name, steering_function=None, steering_interval=(0, -1)):
        """
        Registers ActivationHooks for each layer in the given mapping.
        Skips already registered layers.

        :param key_to_layer_name: Mapping from cache key (str) to target layer name (str).
        :type key_to_layer_name: Dict[str, str]
        :param steering_function: Optional function applied to layer output before caching/steering.
        :type steering_function: Optional[Callable]
        :param steering_interval: (min_token, max_token). Skip first min_token; stop after max_token (-1 = no limit).
        :type steering_interval: Tuple[int, int]
        """
        # Get the model (assume HuggingFace pipeline)
        if self.base_model is self.llm:
            raise RuntimeError("Base model not found or not a HuggingFace pipeline.")

        # Always re-initialize cache and hooks
        self._hook_acts = []

        # Register new hooks
        for cache_key, layer_name in key_to_layer_name.items():
            hook = ActivationHook(
                cache_key=cache_key,
                layer_key=layer_name,
                agent=self,
                response_hook=self._hook_response_data,
                steering_function=steering_function,  # pass the function here,
                steering_interval=steering_interval
            )
            self._hook_acts.append(hook)

    def _clear_hooks(self):
        """
        Resets all cached representations and removes registered hooks from the agent.
        """
        for hook in getattr(self, '_hook_acts', []):
            hook.remove()
        self._hook_acts = []
        if self._hook_response_data is not None:
            self._hook_response_data.reset()
        self._set_hook_response_data()

    def _set_hook_response_data(self):
        """
        Ensures a ResponseHook is registered (idempotent).
        """
        if self._hook_response_data is None:
            self._hook_response_data = ResponseHook(agent=self)

    def response_lookahead(self, message: str = None):
        """
        Generates a response without updating the agent's memory.

        - If message is None, predicts the next reply given current memory.
        - If message is provided, predicts a reply to that hypothetical input.

        Notes:
        - Orchestrators and inspectors are not invoked.
        - Tools may be called, but their outputs are not persisted.
        - Only postprocess_fn is applied (no preprocessing).

        :param message: The hypothetical message to reply to (optional).
        :type message: Optional[str]
        :return: The predicted response text.
        :rtype: str
        """
        if not message:
            response, _ = self._get_llm_response(self.memory, update_tool_memory=False)
            return response.content

        response, _ = self._get_llm_response(self.memory + [HumanMessage(message)],
                                             update_tool_memory=False)
        return response.content

    def add_orchestrators(self, orchestrators):
        """
        Adds orchestrators to the agent.

        :param orchestrators: Orchestrator(s) to add.
        :type orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]]
        """
        if not orchestrators:
            return

        if self._orchestrators is None:
            self._orchestrators = []

        if isinstance(orchestrators, BaseOrchestrator):
            orchestrators = [orchestrators]

        self._orchestrators.extend(orchestrators)

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

        if self._inspectors is None:
            self._inspectors = []

        # Handle both single Inspector and list of Inspectors
        if isinstance(inspectors, Inspector):
            inspectors = [inspectors]
        elif isinstance(inspectors, list):
            inspectors = [ins for ins in inspectors if ins is not None]
            if not inspectors:
                return
        else:
            raise TypeError("inspectors must be an Inspector or a list of Inspectors")

        self._inspectors.extend(inspectors)
        self._set_hook_response_data()
        for inspector in inspectors:
            inspector.add_agent(self)

    def clear_orchestrators(self):
        """
        Removes all orchestrators from the agent.
        """
        self._orchestrators = None

    def clear_inspectors(self):
        """
        Removes all inspectors from the agent.
        """
        self._inspectors = None
        self._hook_response_data = None
        self._clear_hooks()

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
        self._first_utterances = utterances

    def get_name(self, default: str = "Me") -> str:
        """
        Returns the agent's name.

        :param default: Fallback name if agent has no explicit name.
        :type default: str
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
        data["model"] = self.model_info
        if self._first_utterances:
            data["first_utterances"] = self._first_utterances
        data["persona"] = self.persona.json() if self.persona else {}
        if self._orchestrators:
            data["persona"]["orchestrators"] = [orc.json() for orc in self._orchestrators]
        return json.dumps(data, indent=indent) if string else data

    def reset(self, seed: int = None, context: Union[str, Context] = None, example_dialogs: List['Dialog'] = None):
        """
        Resets the agent's memory and orchestrators, optionally reseeding the LLM.
        Also clears interpretability state and components if any.

        :param seed: Random seed for reproducibility (if None, generated).
        :param context: Optional context override.
        :param example_dialogs: Optional replacement example dialogs for prompt regeneration.
        """
        # Remove history
        self.memory[:] = self.memory[:1]
        # Update system prompt if needed
        if self.persona and self.memory and (context or example_dialogs):
            system_prompt = self._system_prompt_template.render(
                persona=self.persona.prompt(),
                context=context or self._context,
                example_dialogs=example_dialogs or self._example_dialogs,
                dialogue_details=self._dialogue_details,
                response_details=self._response_details,
                can_finish=self._can_finish,
                stop_word=self._STOP_WORD
            )
            self.memory[0].content = system_prompt

        self._finished = False
        seed = set_generator_seed(self, seed)

        if self._orchestrators:
            for orchestrator in self._orchestrators:
                orchestrator.reset()

        if self._hook_response_data is not None:
            self._hook_response_data.reset()

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
                utter = utt_events[-1].content
                utt_events[-1].content = utter.replace(self._STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].content:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=self.get_name(),
                text=utt_events[-1].content
            ))
            events.extend(utt_events)
            pbar.update(1)

            utt_events = agent(utter, return_events=True)
            if utt_events and utt_events[-1].action == "utter":
                utter = utt_events[-1].content
                utt_events[-1].content = utter.replace(self._STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].content:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=agent.get_name(default="Other"),
                text=utt_events[-1].content
            ))
            events.extend(utt_events)
            pbar.update(1)

        pbar.close()

        context = context or self._context
        return Dialog(
            id=id if id is not None else get_universal_id(),
            parentId=parent_id,
            complete=completion,  # incomplete if ran out of iterations (reached max_iteration number)
            model=self.model_info,
            seed=seed,
            personas={
                self.get_name(): self.persona.json() if self.persona else {},
                agent.get_name(default="Other"): agent.persona.json() if agent.persona else {}
            },
            context=context.json() if context and isinstance(context, Context) else context,
            scenario=scenario,
            notes=notes,
            turns=dialog,
            events=events
        )

    def memory_dump(self, as_dict: bool = False) -> list:
        """
        Returns a copy of the agent's memory (list of messages).

        :param as_dict: If True, returns list of message dicts (serialization-friendly).
        :type as_dict: bool
        :return: Conversation memory snapshot.
        :rtype: list
        """
        return messages_to_dict(self.memory) if as_dict else self.memory.copy()

    #: Alias for :func:`Agent.dialog_with`.
    talk_with = dialog_with
