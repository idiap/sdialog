"""
This module provides classes for generating synthetic dialogues using LLMs, including support for persona-based
role-play and context-driven dialogue generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import logging

from tqdm.auto import tqdm
from jinja2 import Template
from pydantic import BaseModel
from typing import Union, List, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from ..agents import Agent
from ..config import config
from .. import Dialog, Turn, Context
from ..personas import BasePersona, Persona
from ..util import get_universal_id, get_timestamp, set_generator_seed, get_llm_model

from .base import BaseAttributeModelGenerator, LLMDialogOutput

logger = logging.getLogger(__name__)


class DialogGenerator:
    """
    Base class for generating synthetic dialogues using an LLM.

    Typical workflow:

      1. Instantiate with default dialogue instructions and optional context / examples.
      2. Call generate(...) to produce a Dialog (or raw structured output).

    Example:

        .. code-block:: python

            from sdialog.generators import DialogGenerator

            gen = DialogGenerator("Generate a short friendly greeting between two speakers")

            dialog = gen.generate()
            dialog.print()

    :param dialogue_details: Instructions or details for the dialogue.
    :type dialogue_details: str
    :param context: The default context for the dialogue (optional).
    :type context: Optional[Union[str, Context]]
    :param example_dialogs: Optional default list of example dialogues to guide the generation.
    :type example_dialogs: List[Dialog]
    :param scenario: Default scenario metadata for the dialogue.
    :type scenario: Optional[Union[dict, str]]
    :param personas: Optional personas (serialized) involved in the dialogue (e.g., for logging).
    :type personas: dict[str, dict[str, Any]]
    :param output_format: Output schema / model used to parse LLM output (or None for raw text).
    :type output_format: Union[dict, BaseModel]
    :param model: The LLM instance or model name to use.
    :type model: Union[BaseLanguageModel, str]
    :param llm_kwargs: Additional keyword arguments for the LLM (override config).
    :type llm_kwargs: dict
    """
    def __init__(self,
                 dialogue_details: str,
                 context: Optional[Union[str, Context]] = None,
                 example_dialogs: List['Dialog'] = None,
                 scenario: Optional[Union[dict, str]] = None,
                 personas: dict[str, dict[str, Any]] = None,
                 output_format: Union[dict, BaseModel] = LLMDialogOutput,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initializes a DialogGenerator."""
        if model is None:
            model = config["llm"]["model"]

        # Collect LLM parameters from config, only if not None
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}

        self.output_format = output_format

        self.llm, llm_kwargs = get_llm_model(model_name=model,
                                             output_format=self.output_format,
                                             return_model_params=True,
                                             **llm_kwargs)
        self.model_info = {"name": str(model), **llm_kwargs}

        with open(config["prompts"]["dialog_generator"], encoding="utf-8") as f:
            self.system_prompt_template = Template(f.read())

        self._personas = personas
        self.context = context
        self.example_dialogs = example_dialogs
        self.dialogue_details = dialogue_details
        self.scenario = scenario
        self.messages = [SystemMessage(""), HumanMessage("")]

    def _set_prompt(self,
                    dialogue_details: str,
                    context: Optional[Union[str, Context]] = None,
                    example_dialogs: List['Dialog'] = None):
        """
        Sets the dialogue details and scenario for generation.

        :param dialogue_details: Instructions or details for the dialogue.
        :type dialogue_details: str
        :param context: The context for the dialogue (optional).
        :type context: Optional[Union[str, Context]]
        :param example_dialogs: Optional list of example dialogues to guide the generation.
        :type example_dialogs: List[Dialog]
        """
        # Load system message from prompt file
        system_message = self.system_prompt_template.render(example_dialogs=example_dialogs, context=context)

        self.messages[0].content = system_message
        self.messages[1].content = dialogue_details

    def prompt(self) -> str:
        """
        Returns the current system prompt used for dialogue generation.

        :return: The system prompt string.
        :rtype: str
        """
        return self.messages[0].content

    def generate(self,
                 dialogue_details: str = None,
                 context: Optional[Union[str, Context]] = None,
                 example_dialogs: List[Dialog] = None,
                 scenario: Optional[Union[dict, str]] = None,
                 seed: int = None,
                 id: int = None,
                 parent_id: int = None,
                 notes: str = None):
        """
        Generates a synthetic dialogue using the LLM.

        :param dialogue_details: Override instructions / details for this generation.
        :type dialogue_details: str
        :param context: Override context for this generation.
        :type context: Optional[Union[str, Context]]
        :param example_dialogs: Override example dialogues for few-shot style guidance.
        :type example_dialogs: List[Dialog]
        :param scenario: Override scenario metadata.
        :type scenario: Optional[Union[dict, str]]
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param id: Optional dialogue ID to assign (otherwise autogenerated).
        :type id: int
        :param parent_id: Optional parent dialogue ID (thread linkage).
        :type parent_id: int
        :param notes: Optional free-form notes stored in metadata.
        :type notes: str
        :return: Dialog instance if output_format is LLMDialogOutput; BaseModel if custom schema;
                 raw string if output_format is falsy.
        :rtype: Union[Dialog, BaseModel, str]
        """
        self._set_prompt(dialogue_details or self.dialogue_details,
                         context or self.context,
                         example_dialogs or self.example_dialogs)
        seed = set_generator_seed(self, seed)

        dialogue = self.llm.invoke(self.messages)

        logger.log(logging.DEBUG, f"System prompt used: {self.messages[0]}")

        if not self.output_format:
            return dialogue.content
        else:
            llm_output = self.output_format.model_validate(dialogue)

            if self.output_format is LLMDialogOutput:
                context = context or self.context
                return Dialog(id=id if id is not None else get_universal_id(),
                              parentId=parent_id,
                              model=self.model_info,
                              seed=seed,
                              personas=self._personas,
                              context=context.json() if context and isinstance(context, Context) else context,
                              scenario=scenario or self.scenario,
                              notes=notes,
                              turns=llm_output.dialog)
            else:
                return llm_output

    __call__ = generate  # alias for generate method


class PersonaDialogGenerator(DialogGenerator):
    """
    Generates dialogues between two personas (or Agents wrapping personas) using an LLM.

    Example:

        .. code-block:: python

            from sdialog.personas import Persona
            from sdialog.generators import PersonaDialogGenerator

            p1 = Persona(name="Alice", role="Curious student")
            p2 = Persona(name="Mentor", role="Helpful tutor")

            gen = PersonaDialogGenerator(p1, p2, dialogue_details="Explain one concept briefly.")

            dialog = gen()
            dialog.print()

    :param persona_a: The first persona or an Agent containing one.
    :type persona_a: Union[Persona, Agent]
    :param persona_b: The second persona or an Agent containing one.
    :type persona_b: Union[Persona, Agent]
    :param speaker_a: Name/ID of the first speaker in the dialogue.
    :type speaker_a: str
    :param speaker_b: Name/ID of the second speaker in the dialogue.
    :type speaker_b: str
    :param context: Default context for the dialogue (optional).
    :type context: Optional[Union[str, Context]]
    :param example_dialogs: Optional list of example dialogues for guidance.
    :type example_dialogs: List[Dialog]
    :param dialogue_details: Additional dialogue-level instructions.
    :type dialogue_details: str
    :param response_details: Style / formatting instructions for responses.
    :type response_details: str
    :param scenario: Default scenario metadata.
    :type scenario: Optional[Union[dict, str]]
    :param model: LLM instance or model name.
    :type model: Union[BaseLanguageModel, str]
    :param llm_kwargs: Extra LLM keyword arguments (override config).
    :type llm_kwargs: dict
    """
    _agent_a = None
    _agent_b = None

    def __init__(self,
                 persona_a: Union[Persona, Agent],
                 persona_b: Union[Persona, Agent],
                 speaker_a: str = "SPEAKER_A",
                 speaker_b: str = "SPEAKER_B",
                 context: Optional[Union[str, Context]] = None,
                 example_dialogs: List['Dialog'] = None,
                 dialogue_details: str = "",
                 response_details: str = "",
                 scenario: Optional[Union[dict, str]] = None,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """Initializes a PersonaDialogGenerator."""
        if isinstance(persona_a, Agent) and isinstance(persona_b, Agent):
            self._agent_a = persona_a
            self._agent_b = persona_b
            persona_a = persona_a.persona
            persona_b = persona_b.persona
            if dialogue_details:
                logger.warning("The provided `dialogue_details` argument will be ignored because both personas are "
                               "`Agent` instances; dialogue behavior is determined by the agents themselves.")

        # Load persona dialog prompt template from file
        with open(config["prompts"]["persona_dialog_generator"], encoding="utf-8") as f:
            dialogue_details_template = Template(f.read())
        dialogue_details = dialogue_details_template.render(
            persona_a=persona_a.prompt(),
            persona_b=persona_b.prompt(),
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            context=context,
            dialogue_details=dialogue_details,
            response_details=response_details
        )

        super().__init__(dialogue_details=dialogue_details,
                         example_dialogs=example_dialogs,
                         personas={
                             speaker_a: persona_a.json(),
                             speaker_b: persona_b.json()
                         },
                         scenario=scenario,
                         model=model,
                         **llm_kwargs)

    def generate(self,
                 context: Optional[Union[str, Context]] = None,
                 example_dialogs: List[Dialog] = None,
                 scenario: Optional[Union[dict, str]] = None,
                 seed: int = None,
                 id: int = None,
                 parent_id: int = None,
                 max_turns: int = 200,
                 notes: str = None):
        """
        Generates a dialogue between two personas (or drives an Agent-to-Agent interaction).

        :param context: Override context.
        :type context: Optional[Union[str, Context]]
        :param example_dialogs: Override example dialogues.
        :type example_dialogs: List[Dialog]
        :param scenario: Override scenario metadata.
        :type scenario: Optional[Union[dict, str]]
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param id: Dialogue ID override.
        :type id: int
        :param parent_id: Parent dialogue ID (thread).
        :type parent_id: int
        :param max_turns: Max turns (only applies when both participants are Agents).
        :type max_turns: int
        :param notes: Optional metadata notes.
        :type notes: str
        :return: Generated dialogue object.
        :rtype: Dialog
        """
        if self._agent_a and self._agent_b:
            return self._agent_a.dialog_with(self._agent_b,
                                             context=context,
                                             example_dialogs=example_dialogs,
                                             scenario=scenario,
                                             max_turns=max_turns,
                                             id=id,
                                             seed=seed,
                                             notes=notes,
                                             parent_id=parent_id)
        else:
            return super().generate(context=context,
                                    example_dialogs=example_dialogs,
                                    scenario=scenario,
                                    seed=seed,
                                    id=id,
                                    notes=notes,
                                    parent_id=parent_id)

    __call__ = generate  # alias for generate method


class PersonaGenerator(BaseAttributeModelGenerator):
    """
    Generates persona objects (subclasses of :class:`sdialog.personas.BasePersona`) with randomized or
    LLM-populated attributes (see :class:`sdialog.generators.BaseAttributeModelGenerator` for more information).

    Example:

        .. code-block:: python

            from sdialog.personas import Doctor
            from sdialog.generators import PersonaGenerator

            base_persona = Doctor(specialty="Cardiology")

            doctor_generator = PersonaGenerator(base_persona)

            doctor_generator.set(
                years_of_experience="{4-10}",
                gender=["male", "female", "non-binary"]
            )

            doctor = doctor_generator.generate()
            doctor.print()

    :param persona: Persona instance or class to generate.
    :type persona: BasePersona
    :param generated_attributes: Strategy specifying which attributes to fill ("all", list, or dict).
    :type generated_attributes: Union[str, list, dict]
    :param extra_instructions: Additional instructions to include in the LLM prompt.
    :type extra_instructions: str
    :param model: LLM model name (optional).
    :type model: str
    :param llm_kwargs: Extra LLM keyword arguments.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 persona: BasePersona,
                 generated_attributes: str = "all",
                 extra_instructions: str = "Attributes must be in English",
                 model: str = None,
                 **llm_kwargs):
        """Initialize a PersonaGenerator."""
        if isinstance(persona, BasePersona):
            persona_instance = persona
        elif isinstance(persona, type) and issubclass(persona, BasePersona):
            persona_instance = persona()
        else:
            raise ValueError("persona must be a BasePersona instance or subclass.")
        system_prompt = "You are an expert at generating persona JSON objects for synthetic dialogue generation."
        with open(config["prompts"]["persona_generator"], encoding="utf-8") as f:
            llm_prompt = f.read()
        with open(config["prompts"]["persona_generator_n"], encoding="utf-8") as f:
            llm_prompt_n = f.read()
        super().__init__(attribute_model=persona_instance,
                         generated_attributes=generated_attributes,
                         extra_instructions=extra_instructions,
                         model=model,
                         system_prompt=system_prompt,
                         llm_prompt=llm_prompt,
                         llm_prompt_n=llm_prompt_n,
                         **llm_kwargs)


class ContextGenerator(BaseAttributeModelGenerator):
    """
    Generates Context objects with randomized or LLM-populated attributes
    (see :class:`sdialog.generators.BaseAttributeModelGenerator` for more information).

    Example:

        .. code-block:: python

            from sdialog import Context
            from sdialog.generators import ContextGenerator

            base_context = Context(location="Mars Forward Base Alpha")

            ctx_generator = ContextGenerator(base_context)

            ctx_generator.set(
                objects=get_objects_from_db,  # callable function
                topics=["terraforming", "resource logistics", "crew morale"]
                circumstances="{csv:circumstances:./data/circumstances.csv}",
                goals="{llm:Suggest a realistic goal for the context}"
            )

            my_context = ctx_generator.generate()
            my_context.print()

    :param context: Context instance or subclass to generate.
    :type context: Context
    :param generated_attributes: Attribute selection strategy ("all", list, or dict).
    :type generated_attributes: Union[str, list, dict]
    :param extra_instructions: Additional instructions to include in the LLM prompt.
    :type extra_instructions: str
    :param model: LLM model name (optional).
    :type model: str
    :param llm_kwargs: Extra LLM keyword arguments.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 context: Context = None,
                 generated_attributes: str = "all",
                 extra_instructions: str = "Attributes must be in English",
                 model: str = None,
                 **llm_kwargs):
        """
        Initialize a ContextGenerator.

        :raises ValueError: If context is not a Context or subclass.
        """
        if context is None:
            context = Context()
        if isinstance(context, type) and issubclass(context, Context):
            context = context()
        elif not isinstance(context, Context):
            raise ValueError("context must be a `Context` instance or subclass.")
        system_prompt = (
            "You are an expert at generating shared dialogue context as well-structured JSON objects "
            "for synthetic dialogue generation."
        )
        with open(config["prompts"]["context_generator"], encoding="utf-8") as f:
            llm_prompt = f.read()
        with open(config["prompts"]["context_generator_n"], encoding="utf-8") as f:
            llm_prompt_n = f.read()
        super().__init__(attribute_model=context,
                         generated_attributes=generated_attributes,
                         extra_instructions=extra_instructions,
                         model=model,
                         system_prompt=system_prompt,
                         llm_prompt=llm_prompt,
                         llm_prompt_n=llm_prompt_n,
                         **llm_kwargs)


class Paraphraser:
    """
    Paraphrases dialogue turns while preserving semantic entities/values.

    Usage modes:

      * Whole dialogue paraphrasing (default, returns full set of possibly modified turns).
      * Turn-by-turn paraphrasing (stream-like, for smaller LLMs).

    Example:

        .. code-block:: python

            from sdialog.generators import Paraphraser

            # Assume 'original_dialog' is an existing `Dialog` with one of the speaker being "Bot"
            paraphraser = Paraphraser("Make the text sound more natural and less robotic",
                                      target_speaker="Bot")

            new_dialog = paraphraser(original_dialog)
            new_dialog.print()

    :param extra_instructions: Additional style or behavior instructions for the paraphrase.
    :type extra_instructions: str
    :param target_speaker: If provided, only paraphrases turns spoken by this speaker.
    :type target_speaker: Optional[str]
    :param turn_by_turn: Whether to paraphrase one turn at a time.
    :type turn_by_turn: bool
    :param model: The LLM instance or model name to use (falls back to config if None).
    :type model: Union[str, BaseLanguageModel]
    :param llm_kwargs: Additional keyword arguments for the LLM.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 extra_instructions: str = "Keep entities and values identical while making it sound more natural",
                 target_speaker: str = None,
                 turn_by_turn: bool = False,
                 model: Union[str, BaseLanguageModel] = None,
                 **llm_kwargs):
        """Initializes a Paraphraser that rewrites dialog turns while preserving entities and values."""
        if model is None:
            model = config["llm"]["model"]

        self.model = model
        self.output_format = Turn if turn_by_turn else LLMDialogOutput
        self.llm, llm_kwargs = get_llm_model(model_name=model,
                                             output_format=self.output_format,
                                             return_model_params=True,
                                             **llm_kwargs)
        self.model_info = {"name": str(model), **llm_kwargs}
        self.extra_instructions = extra_instructions
        self.target_speaker = target_speaker

        with open(config["prompts"]["paraphraser_system"], encoding="utf-8") as f:
            system_message = Template(f.read()).render(only_turn=turn_by_turn)

        if turn_by_turn:
            with open(config["prompts"]["paraphraser_turn"], encoding="utf-8") as f:
                self.instruction_template = Template(f.read())
        else:
            with open(config["prompts"]["paraphraser"], encoding="utf-8") as f:
                self.instruction_template = Template(f.read())

        self.messages = [SystemMessage(system_message), HumanMessage("")]

    def __call__(self,
                 dialog: Dialog,
                 target_speaker: str = None,
                 seed: int = None) -> Dialog:
        """
        Paraphrase a dialog (entirely or selectively by speaker).

        :param dialog: Source dialogue to paraphrase.
        :type dialog: Dialog
        :param target_speaker: Override target speaker filter for this call.
        :type target_speaker: Optional[str]
        :param seed: Optional random seed (used for reproducibility where supported).
        :type seed: Optional[int]
        :return: New Dialog instance with paraphrased turns.
        :rtype: Dialog
        :raises ValueError: (Indirectly) if underlying validation fails.
        """
        target_speaker = target_speaker or self.target_speaker
        seed = set_generator_seed(self, seed)
        new_dialog = dialog.clone()

        if self.output_format is LLMDialogOutput:
            if target_speaker:
                new_dialog.turns = [turn
                                    for turn in dialog.turns
                                    if turn.speaker.lower() == target_speaker.lower()]
            self.messages[1].content = self.instruction_template.render(dialog=new_dialog,
                                                                        extra_instructions=self.extra_instructions,
                                                                        target_speaker=target_speaker)

            output = self.output_format.model_validate(self.llm.invoke(self.messages))

            if not target_speaker:
                new_dialog.turns = output.dialog
            else:
                new_dialog.turns = [output.dialog.pop(0) if turn.speaker.lower() == target_speaker.lower() else turn
                                    for turn in dialog.turns]
        else:
            new_dialog.turns.clear()
            for turn in tqdm(dialog.clone().turns, desc="Paraphrasing turns", leave=False):
                new_dialog.turns.append(turn)
                if not target_speaker or turn.speaker.lower() == target_speaker.lower():
                    self.messages[1].content = self.instruction_template.render(
                        dialog=new_dialog,
                        extra_instructions=self.extra_instructions,
                        target_speaker=target_speaker
                    )
                    output = self.output_format.model_validate(self.llm.invoke(self.messages))
                    new_dialog.turns[-1].text = output.text

        new_dialog.events = None  # TODO: replace each "utt" event by the new paraphrased utterance
        # Update metadata
        new_dialog.seed = seed
        new_dialog.timestamp = get_timestamp()
        new_dialog.model = self.model_info
        if len(new_dialog) != len(dialog):
            logger.warning("Number of turns in the new paraphrased dialog does not match the original!")

        return new_dialog

    def prompt(self) -> str:
        """
        Returns the combined system prompt and current instruction template.

        :return: Combined prompt preview.
        :rtype: str
        """
        return f"{self.messages[0].content}\n\n{self.instruction_template}"
