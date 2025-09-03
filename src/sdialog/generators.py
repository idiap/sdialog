"""
generators: Dialogue Generation Utilities for sdialog

This module provides classes for generating synthetic dialogues using LLMs, including support for persona-based
role-play and context-driven dialogue generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import csv
import json
import random
import logging

from abc import ABC
from tqdm.auto import tqdm
from jinja2 import Template
from typing import Union, List, Any, Optional
from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from . import Dialog, Turn, Context
from .agents import Agent
from .config import config
from .personas import BasePersona, Persona
from .base import Metadata, BaseAttributeModel
from .util import get_llm_model, set_generator_seed, set_ollama_model_defaults, get_universal_id, is_ollama_model_name

logger = logging.getLogger(__name__)


class LLMDialogOutput(BaseModel):
    """
    Pydantic model for LLM-generated dialogue output.

    :ivar dialog: Ordered list of generated dialogue turns.
    :vartype dialog: List[Turn]
    """
    dialog: List[Turn]


class AttributeObject(BaseAttributeModel):
    """
    Generic attribute object used for structured generation of placeholder-bearing entities.

    :ivar placeholder: placeholder attribute to be removed and replaced by real attributes at generation time.
    :vartype placeholder: str
    """
    placeholder: str = None


class ListOfAttributeObjects(BaseModel):
    """
    Container model for validating a list of AttributeObject instances.

    :ivar objects: Collection of attribute objects.
    :vartype objects: List[AttributeObject]
    """
    objects: List[AttributeObject]


_objects_schema = ListOfAttributeObjects.model_json_schema()


class DialogGenerator:
    """
    Base class for generating synthetic dialogues using an LLM.

    Typical workflow:
      1. Instantiate with default dialogue instructions and optional context / examples.
      2. Call generate(...) to produce a Dialog (or raw structured output).
      3. Access system prompt via prompt() for debugging / inspection.

    Example:
    ```python
        from sdialog.generators import DialogGenerator

        gen = DialogGenerator("Generate a short friendly greeting between two speakers")

        dialog = gen.generate()
        dialog.print()
    ```
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
        """
        Initializes a DialogGenerator.

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
        :param output_format: Output schema / model used to parse LLM output (pass falsy to return raw text).
        :type output_format: Union[dict, BaseModel]
        :param model: The LLM instance or model name to use.
        :type model: Union[BaseLanguageModel, str]
        :param llm_kwargs: Additional keyword arguments for the LLM (override config).
        :type llm_kwargs: dict
        """
        if model is None:
            model = config["llm"]["model"]

        # Collect LLM parameters from config, only if not None
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}

        self.output_format = output_format

        self.llm = get_llm_model(model_name=model,
                                 output_format=self.output_format,
                                 **llm_kwargs)

        with open(config["prompts"]["dialog_generator"], encoding="utf-8") as f:
            self.system_prompt_template = Template(f.read())

        self._personas = personas
        self.context = context
        self.example_dialogs = example_dialogs
        self.dialogue_details = dialogue_details
        self.model_name = str(model)  # TODO: improve by adding llm params str(self.llm)
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
                              model=self.model_name,
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
    ```python
        from sdialog.personas import Persona
        from sdialog.generators import PersonaDialogGenerator

        p1 = Persona(name="Alice", role="Curious student")
        p2 = Persona(name="Mentor", role="Helpful tutor")

        gen = PersonaDialogGenerator(p1, p2, dialogue_details="Explain one concept briefly.")

        dialog = gen()
        dialog.print()
    ```
    """
    _agent_a = None
    _agent_b = None

    def __init__(self,
                 persona_a: Union[Persona, Agent],
                 persona_b: Union[Persona, Agent],
                 context: Optional[Union[str, Context]] = None,
                 example_dialogs: List['Dialog'] = None,
                 dialogue_details: str = "",
                 response_details: str = "",
                 scenario: Optional[Union[dict, str]] = None,
                 model: Union[BaseLanguageModel, str] = None,
                 **llm_kwargs):
        """
        Initializes a PersonaDialogGenerator.

        :param persona_a: The first persona or an Agent containing one.
        :type persona_a: Union[Persona, Agent]
        :param persona_b: The second persona or an Agent containing one.
        :type persona_b: Union[Persona, Agent]
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
            context=context,
            dialogue_details=dialogue_details,
            response_details=response_details
        )

        super().__init__(dialogue_details=dialogue_details,
                         example_dialogs=example_dialogs,
                         personas={
                             persona_a.name: persona_a.json(),
                             persona_b.name: persona_b.json()
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


class BaseAttributeModelGenerator(ABC):
    """
    Abstract class to create subclasses for generators with randomized and/or LLM-populated attributes.

    Workflow:
      1. Provide a target attribute model instance or class.
      2. Configure attribute generation rules (set_attribute_generators or generated_attributes='all').
      3. Call generate(n=...) to produce validated instances.
    """
    def __init__(self,
                 attribute_model: BaseAttributeModel,
                 generated_attributes: str = "all",
                 model: str = None,
                 system_prompt: str = None,
                 llm_prompt: str = None,
                 llm_prompt_n: str = None,
                 **llm_kwargs):
        """
        Initialize a BaseAttributeModelGenerator.

        :param attribute_model: Instance or subclass of BaseAttributeModel to generate.
        :type attribute_model: BaseAttributeModel
        :param generated_attributes: Attribute selection strategy ("all", iterable, or dict of rules).
        :type generated_attributes: Union[str, list, dict]
        :param model: LLM model name (overrides config if provided).
        :type model: str
        :param system_prompt: Override system prompt for generation.
        :type system_prompt: str
        :param llm_prompt: Template for single-object generation.
        :type llm_prompt: str
        :param llm_prompt_n: Template for multi-object generation (n > 1).
        :type llm_prompt_n: str
        :param llm_kwargs: Extra LLM instantiation parameters.
        :type llm_kwargs: dict
        """
        if isinstance(attribute_model, BaseAttributeModel):
            self._attribute_model = attribute_model
        elif isinstance(attribute_model, type) and issubclass(attribute_model, BaseAttributeModel):
            self._attribute_model = attribute_model()

        if isinstance(generated_attributes, (list, dict)):
            self._check_attributes(generated_attributes)

        self._rnd_attributes = generated_attributes if isinstance(generated_attributes, dict) else {}
        self.generated_attributes = generated_attributes
        self.llm_model = model if model is not None else config["llm"]["model"]
        self.llm_kwargs = llm_kwargs
        self.system_prompt = system_prompt or ("You are an expert at generating structured JSON objects "
                                               "for synthetic dialogue workflows.")
        self.llm_prompt = llm_prompt
        self.llm_prompt_n = llm_prompt_n

    def _check_attributes(self, attribute_keys):
        """
        Validate that provided attribute keys exist in the target model instance.

        :param attribute_keys: Iterable of attribute keys to validate.
        :type attribute_keys: Iterable
        :raises ValueError: If any attribute is not defined on the target model.
        """
        for key in attribute_keys:
            if key not in self._attribute_model.__dict__:
                raise ValueError(f"Default attribute '{key}' not found in "
                                 f"class '{type(self._attribute_model).__name__}'. "
                                 f"Expected attributes are: {list(self._attribute_model.__dict__.keys())}.")

    def _extract_field_descriptions(self, schema, target_attributes=None):
        """
        Extract field descriptions from a Pydantic model JSON schema.

        :param schema: JSON schema dictionary produced by model_json_schema().
        :type schema: dict
        :param target_attributes: Optional iterable restricting extraction to these fields.
        :type target_attributes: Optional[Iterable[str]]
        :return: Mapping of field name -> description text.
        :rtype: dict[str, str]
        """
        descriptions = {}
        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            if target_attributes is None or field_name in target_attributes:
                d = field_schema.get("description")
                if d:
                    descriptions[field_name] = d

        return descriptions

    def prompt(self) -> str:
        """
        Returns the single-object prompt template text.

        :return: The prompt string.
        :rtype: str
        """
        return self.llm_prompt

    def set_attribute_generators(self, **attributes):
        """
        Define per-attribute randomization / generation specifications.

        Specification value semantics:
          * "*": Defer to LLM.
          * callable: Invoked (with current partial object as kwargs if compatible).
          * list: Random element chosen.
          * fixed scalar / str: Assigned directly.
          * templated string "{...}":
              "{min-max}": Random int in inclusive range.
              "{txt:PATH}": Random non-empty line from file.
              "{csv:COLUMN:PATH}": Random value from CSV column (name or index).
              "{tsv:COLUMN:PATH}": Same for TSV.
              "{llm}": Defer to LLM.
              "{llm:INSTRUCTION}": Defer with custom instruction.

        :param attributes: Mapping of attribute name -> generation rule.
        :type attributes: dict
        :raises ValueError: If an attribute key is not valid.
        """
        self._check_attributes(attributes)
        self._rnd_attributes = attributes

    def generate(self,
                 n: int = 1,
                 temperature: float = None,
                 seed: int = None,
                 id: int = None,
                 parent_id: int = None,
                 notes: str = None,
                 max_attempts: int = 3) -> BaseAttributeModel:
        """
        Generate one or many model instances using random rules, templates, and/or LLM completion.

        :param n: Number of instances to generate.
        :type n: int
        :param temperature: LLM temperature (if LLM used).
        :type temperature: float
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param id: Optional explicit ID for single-object generation (each object gets its own if multiple).
        :type id: int
        :param parent_id: Optional parent ID linkage.
        :type parent_id: int
        :param notes: Optional metadata notes.
        :type notes: str
        :param max_attempts: Maximum retries to fill missing attributes.
        :type max_attempts: int
        :return: A single instance if n == 1, else a list of instances.
        :rtype: Union[BaseAttributeModel, List[BaseAttributeModel]]
        :raises ValueError: On missing files referenced in template specifications.
        """
        seed = seed if seed is not None else random.getrandbits(32)
        random.seed(seed)

        output_object = None
        random_objects_dict = [{} for _ in range(n)]
        target_model_dict = self._attribute_model.__dict__

        for attempt in range(max_attempts):
            for random_object_dict in random_objects_dict:
                llm_attribute_instructions_txt = ""
                llm_attribute_instructions = {}

                for key, value in target_model_dict.items():
                    if value or value == 0:
                        random_object_dict[key] = value
                    elif key in self._rnd_attributes:
                        spec = self._rnd_attributes[key]
                        if callable(spec):
                            random_object_dict[key] = spec
                        elif isinstance(spec, list):
                            random_object_dict[key] = random.choice(spec)
                        elif isinstance(spec, str) and spec:
                            if spec == "*":
                                random_object_dict[key] = None
                            elif spec.startswith("{") and spec.endswith("}"):
                                spec_inner = spec.strip("{}")
                                m_range = re.match(r"(\d+)-(\d+)", spec_inner)
                                m_txt = re.match(r"txt:(.+)", spec_inner)
                                m_csv = re.match(r"csv:([^:]+):(.+)", spec_inner)
                                m_tsv = re.match(r"tsv:([^:]+):(.+)", spec_inner)
                                m_llm = re.match(r"llm(:.+)?", spec_inner)
                                if m_range:
                                    a, b = int(m_range.group(1)), int(m_range.group(2))
                                    random_object_dict[key] = random.randint(a, b)
                                elif m_txt:
                                    path = m_txt.group(1)
                                    try:
                                        with open(path) as f:
                                            lines = [ln for ln in f if ln.strip()]
                                        random_object_dict[key] = random.choice(lines).strip()
                                    except FileNotFoundError:
                                        raise ValueError(f"File '{path}' not found for template '{spec}'.")
                                elif m_csv or m_tsv:
                                    m_csv = m_csv or m_tsv
                                    col, path = m_csv.group(1), m_csv.group(2)
                                    col = int(col) if col.isdigit() else col
                                    try:
                                        with open(path, newline='', encoding="utf-8") as csvfile:
                                            delim = '\t' if m_tsv else ','
                                            if isinstance(col, int):
                                                reader = csv.reader(csvfile, delimiter=delim)
                                                values = [row[col] for row in reader if row[col]]
                                            else:
                                                reader = csv.DictReader(csvfile, delimiter=delim)
                                                if col not in reader.fieldnames:
                                                    raise ValueError(f"Column '{col}' not found in '{path}'.")
                                                values = [row[col] for row in reader if row[col]]
                                        random_object_dict[key] = random.choice(values)
                                    except FileNotFoundError:
                                        raise ValueError(f"File '{path}' not found for template '{spec}'.")
                                elif m_llm:
                                    random_object_dict[key] = None
                                    instr = m_llm.group(1)[1:] if m_llm.group(1) else None
                                    if instr:
                                        llm_attribute_instructions[key] = instr
                            else:
                                random_object_dict[key] = spec
                    elif self.generated_attributes and (self.generated_attributes == "all"
                                                        or key in self.generated_attributes):
                        random_object_dict[key] = None

                for key, value in list(random_object_dict.items()):
                    if callable(value):
                        try:
                            random_object_dict[key] = value(**random_object_dict)
                        except TypeError:
                            random_object_dict[key] = value()

            llm = None
            if any(v is None for v in random_objects_dict[0].values()):
                schema = self._attribute_model.model_json_schema()
                null_attrs = {k for k, v in random_objects_dict[0].items() if v is None}
                field_desc = self._extract_field_descriptions(schema, null_attrs)

                if field_desc or (n == 1 and null_attrs):
                    for k, v in field_desc.items():
                        if k not in llm_attribute_instructions:
                            llm_attribute_instructions[k] = v
                    if llm_attribute_instructions:
                        llm_attribute_instructions_txt = ("Consider the following instructions for filling "
                                                          "the following attributes:\n")
                        llm_attribute_instructions_txt += "\n".join(
                            f"* {k}: {v}." for k, v in llm_attribute_instructions.items()
                        )

                if n > 1:
                    template = Template(self.llm_prompt_n)
                    prompt = template.render(
                        objects=json.dumps(random_objects_dict, indent=2),
                        class_name=type(self._attribute_model).__name__,
                        attributes_instructions=llm_attribute_instructions_txt,
                        n=n
                    )
                else:
                    template = Template(self.llm_prompt)
                    prompt = template.render(
                        object=json.dumps(random_objects_dict[0], indent=2),
                        class_name=type(self._attribute_model).__name__,
                        attributes_instructions=llm_attribute_instructions_txt
                    )

                schema = self._attribute_model.model_json_schema()
                filtered_properties = schema
                if n > 1:
                    if is_ollama_model_name(self.llm_model):
                        schema["type"] = "array"
                    else:
                        filtered_properties = list(_objects_schema["$defs"].values())[0]
                        filtered_properties["properties"] = schema["properties"]
                        schema = _objects_schema
                filtered_properties["properties"] = {
                    k: v for k, v in filtered_properties["properties"].items()
                    if k in random_objects_dict[0]
                }

                llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
                llm_kwargs = {**llm_config_params, **self.llm_kwargs}
                llm_kwargs = set_ollama_model_defaults(self.llm_model, llm_kwargs)
                if temperature is not None:
                    llm_kwargs["temperature"] = temperature
                llm_kwargs["seed"] = seed + attempt

                llm = get_llm_model(model_name=self.llm_model,
                                    output_format=schema,
                                    **llm_kwargs)

                messages = [SystemMessage(self.system_prompt), HumanMessage(prompt)]

                if n > 1:
                    for ix in range(max_attempts):
                        llm_output = llm.invoke(messages)
                        if not is_ollama_model_name(self.llm_model):
                            llm_output = llm_output["objects"]
                        if isinstance(llm_output, list):
                            break
                        logger.warning("LLM output is not a list, retrying "
                                       f"((attempt {ix + 1} out of {max_attempts}))...")
                    if isinstance(llm_output, list):
                        llm_output = llm_output[:n]
                        for ix in range(len(llm_output)):
                            llm_output[ix] = {
                                k: llm_output[ix].get(k, None) if v is None else v
                                for k, v in random_objects_dict[ix].items()
                            }
                    else:
                        logging.error("LLM failed to generate a list; attributes left empty.")
                        llm_output = []
                else:
                    llm_output = llm.invoke(messages)
                    random_objects_dict[0].update({
                        k: v for k, v in llm_output.items() if random_objects_dict[0][k] is None
                    })

            if n > 1:
                instances = []
                for ix, object_dict in enumerate(random_objects_dict):
                    object_dict = llm_output[ix] if ix < len(llm_output) else object_dict
                    try:
                        instances.append(self._attribute_model.model_validate(object_dict))
                        instances[-1]._metadata = Metadata(
                            model=str(self.llm_model) if llm else None,
                            seed=seed,
                            id=id if id is not None else get_universal_id(),
                            parentId=parent_id,
                            className=type(self._attribute_model).__name__,
                            notes=notes
                        )
                    except ValidationError as e:
                        logger.warning(f"Validation error in generated object {ix + 1}: {e}")
                        object_dict = {
                            k: v if v or v == 0 else (
                                object_dict[k] if k in object_dict and object_dict[k] is not None else v
                            )
                            for k, v in self._attribute_model.model_dump().items()
                        }
                        instances.append(self._attribute_model.model_validate(object_dict))
                        instances[-1]._metadata = Metadata(
                            model=str(self.llm_model) if llm else None,
                            seed=seed,
                            id=id if id is not None else get_universal_id(),
                            parentId=parent_id,
                            className=type(self._attribute_model).__name__,
                            notes=notes
                        )
                if len(instances) != n:
                    logger.warning(f"Only {len(instances)} objects out of {n} were fully generated.")
                return instances
            else:
                try:
                    if any(v in [None, "", "null"] for v in random_objects_dict[0].values()):
                        raise ValidationError([], [])
                    output_object = self._attribute_model.model_validate(random_objects_dict[0])
                    break
                except ValidationError:
                    missing_attributes = {
                        k: v for k, v in self._attribute_model.model_dump().items()
                        if k not in random_objects_dict[0] or random_objects_dict[0][k] in [None, "", "null"]
                    }
                    logger.warning(
                        f"The following {len(missing_attributes)} attributes are missing: "
                        f"{', '.join(missing_attributes.keys())}. "
                        f"Retrying (attempt {attempt + 1} of {max_attempts})..."
                    )
                    target_model_dict = {
                        k: v if k in missing_attributes else random_objects_dict[0][k]
                        for k, v in target_model_dict.items()
                    }

        if output_object is None:
            logger.warning("Generated object still has missing attributes after max attempts.")
            random_objects_dict[0].update(missing_attributes)
            output_object = self._attribute_model.model_validate(random_objects_dict[0])

        output_object._metadata = Metadata(
            model=str(self.llm_model) if llm else None,
            seed=seed,
            id=id if id is not None else get_universal_id(),
            parentId=parent_id,
            className=type(self._attribute_model).__name__,
            notes=notes
        )
        return output_object


class PersonaGenerator(BaseAttributeModelGenerator):
    """
    Generates persona objects (Persona subclasses of BaseAttributeModel) with randomized or LLM-populated attributes.

    Example:
    ```python
        from sdialog.personas import Doctor
        from sdialog.generators import PersonaGenerator

        base_persona = Doctor(speciality="Cardiology")

        doctor_generator = PersonaGenerator(base_persona)

        doctor_generator.set_attribute_generators(
            years_of_experience="{4-10}",
            gender=["male", "female", "non-binary"]
        )

        doctor = doctor_generator.generate()
        doctor.print()
    ```
    """
    def __init__(self,
                 persona: BasePersona,
                 generated_attributes: str = "all",
                 model: str = None,
                 **llm_kwargs):
        """
        Initialize a PersonaGenerator.

        :param persona: Persona instance or class to generate.
        :type persona: BasePersona
        :param generated_attributes: Strategy specifying which attributes to fill ("all", list, or dict).
        :type generated_attributes: Union[str, list, dict]
        :param model: LLM model name (optional).
        :type model: str
        :param llm_kwargs: Extra LLM keyword arguments.
        :type llm_kwargs: dict
        """
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
                         model=model,
                         system_prompt=system_prompt,
                         llm_prompt=llm_prompt,
                         llm_prompt_n=llm_prompt_n,
                         **llm_kwargs)


class ContextGenerator(BaseAttributeModelGenerator):
    """
    Generates Context objects with randomized or LLM-populated attributes.

    Example:
    ```python
        from sdialog import Context
        from sdialog.generators import ContextGenerator

        base_context = Context(location="Mars Forward Base Alpha")

        ctx_generator = ContextGenerator(base_context)

        ctx_generator.set_attribute_generators(
            environment=["Pressurized dome", "Dusty lab", "Airlock staging zone"],
            topics=["terraforming", "resource logistics", "crew morale"]
        )

        ctx = ctx_generator.generate()
        ctx.print()
    ```
    """
    def __init__(self,
                 context: Context,
                 generated_attributes: str = "all",
                 model: str = None,
                 **llm_kwargs):
        """
        Initialize a ContextGenerator.

        :param context: Context instance or subclass to generate.
        :type context: Context
        :param generated_attributes: Attribute selection strategy ("all", list, or dict).
        :type generated_attributes: Union[str, list, dict]
        :param model: LLM model name (optional).
        :type model: str
        :param llm_kwargs: Extra LLM keyword arguments.
        :type llm_kwargs: dict
        :raises ValueError: If context is not a Context or subclass.
        """
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
    ```python
        from sdialog.generators import Paraphraser

        # Assume 'original_dialog' is an existing `Dialog` with one of the speaker being "Bot"
        paraphraser = Paraphraser(target_speaker="Bot")

        new_dialog = paraphraser(original_dialog)
        new_dialog.print()
    ```
    """
    def __init__(self,
                 extra_instructions: str = "Keep entities and values identical while making it sound more natural",
                 target_speaker: str = None,
                 turn_by_turn: bool = False,
                 model: Union[str, BaseLanguageModel] = None,
                 **llm_kwargs):
        """
        Initializes a Paraphraser that rewrites dialog turns while preserving entities and values.

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
        if model is None:
            model = config["llm"]["model"]

        self.model = model
        self.output_format = Turn if turn_by_turn else LLMDialogOutput
        self.llm = get_llm_model(model_name=model,
                                 output_format=self.output_format,
                                 **llm_kwargs)
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

        self.model_name = str(model)  # TODO: improve by adding llm params str(self.llm)
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
