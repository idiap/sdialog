"""
generators: Dialogue Generation Utilities for sdialog

This module provides classes for generating synthetic dialogues using LLMs, including support for persona-based
role-play and scenario-driven dialogue generation. Output can be structured using Pydantic models for downstream tasks.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import re
import csv
import json
import random
import logging

from jinja2 import Template
from typing import Union, List, Any
from pydantic import BaseModel, ValidationError
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from . import Dialog, Turn
from .config import config
from .personas import BasePersona, Persona, PersonaAgent, PersonaMetadata
from .util import get_llm_model, set_ollama_model_defaults, get_universal_id

logger = logging.getLogger(__name__)


class LLMDialogOutput(BaseModel):
    """
    Pydantic model for LLM-generated dialogue output.

    :ivar dialog: List of dialogue turns.
    :vartype dialog: List[Turn]
    """
    dialog: List[Turn]


# TODO: create a BaseDialogGenerator
class DialogGenerator:
    """
    Base class for generating synthetic dialogues using an LLM.
    """
    def __init__(self,
                 dialogue_details: str,
                 example_dialogs: List['Dialog'] = None,
                 model: Union[BaseLanguageModel, str] = None,
                 output_format: Union[dict, BaseModel] = LLMDialogOutput,
                 scenario: dict = None,
                 personas: dict[str, dict[str, Any]] = None,
                 llm_kwargs: dict = {}):
        """
        Initializes a DialogGenerator.

        :param dialogue_details: Instructions or details for the dialogue.
        :type dialogue_details: str
        :param example_dialogs: Optional list of example dialogues to guide the generation.
        :type example_dialogs: List[Dialog]
        :param model: The LLM or model name to use.
        :type model: Union[BaseLanguageModel, str]
        :param output_format: Output format schema or Pydantic model.
        :type output_format: Union[dict, BaseModel]
        :param scenario: Scenario metadata for the dialogue (if not provided, value set to `dialogue_details`).
        :type scenario: dict
        :param personas: Optional personas for role-playing in the dialogue (if any).
        :type personas: dict[str, dict[str, Any]]
        :param llm_kwargs: Additional keyword arguments for the LLM (overrides config).
        :type llm_kwargs: dict
        """
        if model is None:
            model = config["llm"]["model"]

        # Collect LLM parameters from config, only if not None
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}

        self.output_format = output_format

        if isinstance(model, str):
            self.llm = get_llm_model(model_name=model,
                                     output_format=self.output_format,
                                     llm_kwargs=llm_kwargs)
        else:
            self.llm = model
            if output_format:
                self.llm.format = self.output_format.model_json_schema()

        with open(config["prompts"]["dialog_generator"], encoding="utf-8") as f:
            self.system_prompt_template = Template(f.read())

        self._personas = personas
        self.example_dialogs = example_dialogs
        self.dialogue_details = dialogue_details
        self.model_name = str(self.llm)
        self.scenario = scenario
        self.messages = [SystemMessage(""), HumanMessage("")]

    def _set_prompt(self, dialogue_details: str, example_dialogs: List['Dialog'] = None):
        """
        Sets the dialogue details and scenario for generation.

        :param dialogue_details: Instructions or details for the dialogue.
        :type dialogue_details: str
        :param scenario: Scenario metadata.
        :type scenario: dict
        """
        # Load system message from prompt file
        system_message = self.system_prompt_template.render(example_dialogs=example_dialogs)

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
                 example_dialogs: List[Dialog] = None,
                 seed: int = None,
                 id: int = None,
                 parent_id: int = None,
                 notes: str = None):
        """
        Generates a synthetic dialogue using the LLM.

        :param dialogue_details: Instructions or details for the dialogue (to override the default).
        :type dialogue_details: str
        :param example_dialogs: Optional list of example dialogues to guide the generation (to override the default).
        :type example_dialogs: List[Dialog]
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param id: Dialogue ID.
        :type id: int
        :param parent_id: ID of the parent dialogue, if any.
        :type parent_id: int
        :param notes: Optional notes to include in the dialogue.
        :type notes: str
        :return: The generated dialogue or output object.
        :rtype: Union[Dialog, dict, BaseModel]
        """
        self._set_prompt(dialogue_details or self.dialogue_details, example_dialogs or self.example_dialogs)
        self.llm.seed = seed if seed is not None else random.getrandbits(32)
        logger.log(logging.DEBUG, f"Generating dialogue with seed {self.llm.seed}...")

        # hack to avoid seed bug in prompt cache
        # (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
        _ = self.llm.num_predict
        self.llm.num_predict = 1
        self.llm.invoke(self.messages)
        self.llm.num_predict = _

        dialogue = self.llm.invoke(self.messages).content

        logger.log(logging.INFO, f"Prompt used: {messages_to_dict(self.messages)}")

        if not self.output_format:
            return dialogue
        else:
            llm_output = self.output_format.model_validate(json.loads(dialogue))

            if self.output_format is LLMDialogOutput:
                return Dialog(id=id if id is not None else get_universal_id(),
                              parentId=parent_id,
                              model=self.model_name,
                              seed=self.llm.seed,
                              personas=self._personas,
                              scenario=self.scenario if self.scenario else self.dialogue_details,
                              notes=notes,
                              turns=llm_output.dialog)
            else:
                return llm_output

    __call__ = generate  # alias for generate method


class PersonaDialogGenerator(DialogGenerator):
    """
    Generates dialogues between two personas using an LLM.

    :ivar persona_a: The first persona.
    :vartype persona_a: Persona
    :ivar persona_b: The second persona.
    :vartype persona_b: Persona
    """
    _agent_a = None
    _agent_b = None

    def __init__(self,
                 persona_a: Union[Persona, PersonaAgent],
                 persona_b: Union[Persona, PersonaAgent],
                 example_dialogs: List['Dialog'] = None,
                 dialogue_details: str = "",
                 response_details: str = "",
                 model: Union[BaseLanguageModel, str] = None,
                 scenario: dict = None,
                 llm_kwargs: dict = {}):
        """
        Initializes a PersonaDialogGenerator.

        :param persona_a: The first persona.
        :type persona_a: Persona (or PersonaAgent)
        :param persona_b: The second persona.
        :type persona_b: Persona (or PersonaAgent)
        :example_dialogs: Optional list of example dialogues to guide the generation.
        :type example_dialogs: List[Dialog]
        :param dialogue_details: Additional dialogue instructions.
        :type dialogue_details: str
        :param response_details: Instructions for response style.
        :type response_details: str
        :param model: The LLM or model name to use.
        :type model: Union[BaseLanguageModel, str]
        :param scenario: Scenario metadata.
        :type scenario: dict
        :param llm_kwargs: Additional keyword arguments for the LLM (overrides config).
        :type llm_kwargs: dict
        """

        if isinstance(persona_a, PersonaAgent) and isinstance(persona_b, PersonaAgent):
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
            dialogue_details=dialogue_details,
            response_details=response_details
        )

        super().__init__(dialogue_details=dialogue_details,
                         example_dialogs=example_dialogs,
                         model=model,
                         scenario=scenario,
                         personas={
                             persona_a.name: persona_a.json(),
                             persona_b.name: persona_b.json()
                         },
                         llm_kwargs=llm_kwargs)

    def generate(self,
                 example_dialogs: List[Dialog] = None,
                 seed: int = None,
                 id: int = None,
                 parent_id: int = None,
                 max_turns: int = 200,
                 notes: str = None):
        """
        Generates a dialogue between two personas using the LLM or PersonaAgents.

        :param example_dialogs: Optional list of example dialogues to guide the generation.
        :type example_dialogs: List[Dialog]
        :param seed: Random seed for reproducibility.
        :type seed: int, optional
        :param id: Dialogue ID.
        :type id: int, optional
        :param parent_id: ID of the parent dialogue, if any.
        :type parent_id: int, optional
        :param max_turns: Maximum number of dialogue turns. Only used if both agents are PersonaAgent.
        :type max_turns: int, optional
        :param notes: Optional notes to include in the dialogue.
        :type notes: str, optional
        :return: The generated dialogue as a Dialog object, or the output format specified.
        :rtype: Dialog or output object
        """
        if self._agent_a and self._agent_b:
            return self._agent_a.dialog_with(self._agent_b,
                                             max_turns=max_turns,
                                             id=id,
                                             seed=seed,
                                             notes=notes,
                                             parent_id=parent_id)
        else:
            return super().generate(example_dialogs=example_dialogs,
                                    seed=seed,
                                    id=id,
                                    notes=notes,
                                    parent_id=parent_id)

    __call__ = generate  # alias for generate method


class PersonaGenerator:
    """
    Generates persona objects with randomized or LLM-populated attributes.

    :param persona: An instance of a subclass of `BasePersona` to generate personas from.
    :type persona: BasePersona
    :param generated_attributes: Specifies which attributes to fill by default. Can be "all", a list of attribute names, or None. Defaults to "all".
    :type generated_attributes: str, list, or dict, optional
    :param model: The default language model to use for attribute population via LLM.
    :type model: str, optional

    :raises ValueError: If specified attributes do not exist in the persona or if required files for templates are missing.

    :example:
        generator = PersonaGenerator(Doctor)
        persona_instance = generator.generate()
    """  # noqa: E501

    def __init__(self,
                 persona: BasePersona,
                 generated_attributes: str = "all",  # None
                 model: str = None,
                 llm_kwargs: dict = {}):
        if isinstance(persona, BasePersona):
            self._persona = persona
        elif isinstance(persona, type) and issubclass(persona, BasePersona):
            self._persona = persona()

        if isinstance(generated_attributes, (list, dict)):
            self._check_attributes(generated_attributes)

        self._persona_rnd_attributes = generated_attributes if isinstance(generated_attributes, dict) else {}

        self.generated_attributes = generated_attributes
        self.llm_model = model if model is not None else config["llm"]["model"]
        self.llm_kwargs = llm_kwargs

        # Load persona generation prompt template from file if not provided
        with open(config["prompts"]["persona_generator"], encoding="utf-8") as f:
            self.llm_prompt = f.read()
        with open(config["prompts"]["persona_generator_n"], encoding="utf-8") as f:
            self.llm_prompt_n = f.read()

    def _check_attributes(self, persona_attributes):
        """
        Validate that provided attribute keys exist in the persona.

        :param persona_attributes: Iterable of attribute keys to check.
        :raises ValueError: If any attribute is not found in the persona.
        """
        for key in persona_attributes:
            if key not in self._persona.__dict__:
                raise ValueError(f"Default attribute '{key}' not found in "
                                 f"persona class '{type(self._persona).__name__}'. "
                                 "Expected attributes are: "
                                 f"{list(self._persona.__dict__.keys())}.")

    def _extract_field_descriptions(self, schema, target_attributes=None):
        """
        Extract field descriptions from the persona's JSON schema.

        :param schema: The JSON schema dictionary
        :param target_attributes: Optional set of attribute names to filter descriptions
        :return: Dictionary mapping field names to their descriptions
        """
        descriptions = {}
        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            if target_attributes is None or field_name in target_attributes:
                description = field_schema.get("description")
                if description:
                    descriptions[field_name] = description

        return descriptions

    def prompt(self) -> str:
        """
        Returns the prompt used for generating personas.

        :return: The prompt string.
        :rtype: str
        """
        return self.llm_prompt

    def set_attribute_generators(self, **attributes):
        """
        Set attributes to be randomly generated for the persona.

        Each keyword argument specifies an attribute name and its randomization specification. The value for each attribute can be one of the following:

        - "*": The attribute will be filled by the default value (ie. by the LLM).
        - A function: The function will be called to generate the value.
        - A list: A random element will be selected from the list.
        - A fixed value: The attribute will always be set to this value.
        - A template string: Use double curly braces to specify a template, e.g., "{{VALUE}}". Supported template formats include:
            - "{{min-max}}": A random integer in the range [min, max] will be selected.
            - "{{txt:PATH}}": A random line will be selected from the text file at PATH.
            - "{{csv:COLUMN:PATH}}": A random value will be selected from the COLUMN column in the CSV file at PATH.
            - "{{llm}}: A random value will be generated by the LLM based on the persona context.
            - "{{llm:INSTRUCTION}}: A random value will be generated by the LLM based on the persona context by following the provided INSTRUCTION.

        :param attributes: Keyword arguments of attribute names and values to set.
        :raises ValueError: If any attribute is not found in the persona.
        """  # noqa: E501
        self._check_attributes(attributes)
        self._persona_rnd_attributes = attributes

    def generate(self,
                 n: int = 1,
                 temperature: float = None,
                 seed: int = None,
                 id: int = None,
                 parent_id: int = None,
                 notes: str = None,
                 max_attempts: int = 3) -> BasePersona:
        """
        Generate a persona instance with attributes filled by random selection, templates, or LLM as needed.

        :param n: Number of personas to generate (default: 1).
        :type n: int, optional
        :param temperature: Temperature for LLM generation (if applicable).
        :type temperature: float, optional
        :param seed: Optional random seed for reproducibility.
        :type seed: int, optional
        :param id: Optional unique identifier for the persona.
        :type id: int, optional
        :param parent_id: Optional parent persona ID (if any).
        :type parent_id: int, optional
        :param notes: Optional notes to include in the persona metadata.
        :type notes: str, optional
        :param max_attempts: Maximum number of attempts to generate all attributes (default: 3).
        :type max_attempts: int, optional
        :return: A validated persona instance with metadata.
        :rtype: BasePersona
        :raises ValueError: If required files for templates are missing.
        """
        seed = seed if seed is not None else random.getrandbits(32)
        random.seed(seed)

        output_persona = None
        random_personas_dict = [{} for _ in range(n)]
        target_persona_dict = self._persona.__dict__

        for attempt in range(max_attempts):
            for random_persona_dict in random_personas_dict:
                llm_attribute_instructions_txt = ""
                llm_attribute_instructions = {}

                for key, value in target_persona_dict.items():
                    if value or value == 0:
                        random_persona_dict[key] = value  # keep the default value
                    elif key in self._persona_rnd_attributes:
                        rnd_value = self._persona_rnd_attributes[key]
                        if callable(rnd_value) and key in self._persona_rnd_attributes:
                            random_persona_dict[key] = rnd_value  # a callable
                        elif isinstance(rnd_value, list) and key in self._persona_rnd_attributes:
                            random_persona_dict[key] = random.choice(rnd_value)
                        elif isinstance(rnd_value, str) and rnd_value and key in self._persona_rnd_attributes:
                            if rnd_value == "*":
                                random_persona_dict[key] = None  # to be filled by the LLM
                            elif rnd_value.startswith("{") and rnd_value.endswith("}"):  # templates
                                # TODO: Shall we also have pre-devined lists for name and other attributes
                                #       and then have temples like {{name}} to use them?
                                rnd_value = rnd_value.strip("{}")  # remove outer curly braces
                                m_range = re.match(r"(\d+)-(\d+)", rnd_value)  # match {{min-max}}
                                m_txt = re.match(r"txt:(.+)", rnd_value)  # path to txt file (one line per rnd_value)
                                m_csv = re.match(r"csv:([^:]+):(.+)", rnd_value)  # path to csv file (+column name)
                                m_tsv = re.match(r"tsv:([^:]+):(.+)", rnd_value)  # path to tsv file (+column name)
                                m_llm = re.match(r"llm(:.+)?", rnd_value)  # LLM template with optional instruction
                                if m_range:
                                    min_len, max_len = int(m_range.group(1)), int(m_range.group(2))
                                    random_persona_dict[key] = random.randint(min_len, max_len)
                                elif m_txt:
                                    txt_path = m_txt.group(1)
                                    try:
                                        with open(txt_path) as f:
                                            lines = [ln for ln in f.readlines() if ln.strip()]
                                        random_persona_dict[key] = random.choice(lines).strip()
                                    except FileNotFoundError:
                                        raise ValueError(f"File '{txt_path}' not found for '{rnd_value}' attribute.")
                                elif m_csv or m_tsv:
                                    m_csv = m_csv or m_tsv
                                    csv_column, csv_path = m_csv.group(1), m_csv.group(2)
                                    csv_column = int(csv_column) if csv_column.isdigit() else csv_column
                                    try:
                                        with open(csv_path, newline='', encoding="utf-8") as csvfile:
                                            if isinstance(csv_column, int):
                                                reader = csv.reader(csvfile, delimiter='\t' if m_tsv else ',')
                                                values = [row[csv_column] for row in reader if row[csv_column]]
                                            else:
                                                reader = csv.DictReader(csvfile, delimiter='\t' if m_tsv else ',')
                                                if csv_column not in reader.fieldnames:
                                                    raise ValueError(
                                                        f"Column '{csv_column}' not found in CSV file '{csv_path}'."
                                                    )
                                                values = [row[csv_column] for row in reader if row[csv_column]]
                                        random_persona_dict[key] = random.choice(values)
                                    except FileNotFoundError:
                                        raise ValueError(f"File '{csv_path}' not found for '{rnd_value}' attribute.")
                                elif m_llm:
                                    random_persona_dict[key] = None  # to be filled by the LLM

                                    instruction = m_llm.group(1)[1:] if m_llm.group(1) else None
                                    if instruction:
                                        llm_attribute_instructions[key] = instruction

                                # elif rnd_value == "{{name}}":
                                #     random_persona_dict[key] = get_name(seed=seed)  # get name from pre-defined list
                            else:
                                random_persona_dict[key] = rnd_value
                    elif self.generated_attributes and (
                        self.generated_attributes == "all" or key in self.generated_attributes
                    ):
                        random_persona_dict[key] = None  # to be filled by the LLM

                for key, value in random_persona_dict.items():
                    if callable(value):
                        try:
                            random_persona_dict[key] = value(**random_persona_dict)
                        except TypeError:
                            random_persona_dict[key] = value()  # in case user-proved function has no arguments

            llm = None
            # If there are None value, we need to fill them using the LLM
            if any(value is None for value in random_persona_dict.values()):
                schema = self._persona.model_json_schema()
                null_attributes = {k for k, v in random_persona_dict.items() if v is None}
                field_descriptions = self._extract_field_descriptions(schema, null_attributes)

                if llm_attribute_instructions or field_descriptions:
                    for k, v in field_descriptions.items():
                        if k not in llm_attribute_instructions:
                            llm_attribute_instructions[k] = v
                    llm_attribute_instructions_txt = ("Consider the following instructions for filling "
                                                      "the following attributes:\n")
                    llm_attribute_instructions_txt += "\n".join(
                        [f"* {k}: {v}." for k, v in llm_attribute_instructions.items()]
                    )

                if n > 1:
                    template = Template(self.llm_prompt_n)
                    prompt = template.render(
                        personas=json.dumps(random_personas_dict, indent=2),
                        persona_class_name=str(type(self._persona).__name__),
                        attributes_instructions=llm_attribute_instructions_txt,
                        n_personas=n
                    )
                else:
                    template = Template(self.llm_prompt)
                    prompt = template.render(
                        persona=json.dumps(random_persona_dict, indent=2),
                        persona_class_name=str(type(self._persona).__name__),
                        attributes_instructions=llm_attribute_instructions_txt
                    )

                if not isinstance(self.llm_model, str):
                    llm = self.llm_model
                else:
                    schema = self._persona.model_json_schema()
                    schema["properties"] = {k: v
                                            for k, v in schema["properties"].items()
                                            if k in random_persona_dict}
                    if n > 1:
                        schema["type"] = "array"
                    # Collect LLM parameters from config, only if not None
                    llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
                    llm_kwargs = {**llm_config_params, **self.llm_kwargs}
                    llm_kwargs = set_ollama_model_defaults(self.llm_model, llm_kwargs)
                    # temperature from function argument overrides config
                    if temperature is not None:
                        llm_kwargs["temperature"] = temperature
                    llm_kwargs["seed"] = seed + attempt  # to ensure different seed for each attempt
                    # llm_kwargs from __init__ override config

                    llm = get_llm_model(model_name=self.llm_model,
                                        output_format=schema,
                                        llm_kwargs=llm_kwargs)

                messages = [
                    SystemMessage("You are an expert at generating persona JSON objects "
                                  "for synthetic dialogue generation."),
                    HumanMessage(prompt)
                ]

                if n > 1:
                    for ix in range(max_attempts):
                        llm_output = json.loads(llm.invoke(messages).content)
                        if type(llm_output) is list:
                            break
                        else:
                            logger.warning(
                                f"LLM output is not a list, retrying ((attempt {ix + 1} out of {max_attempts}))..."
                            )

                    if type(llm_output) is list:
                        for ix in range(len(llm_output)):
                            llm_output[ix] = {
                                k: llm_output[ix].get(k, None) if v is None else v
                                for k, v in random_personas_dict[ix].items()
                            }
                    else:
                        logging.error("LLM failed to generate a list of personas, all attributes will be left empty.")
                        llm_output = []
                else:
                    llm_output = json.loads(llm.invoke(messages).content)
                    random_persona_dict.update({k: v
                                                for k, v in llm_output.items()
                                                if random_persona_dict[k] is None})

            if n > 1:
                personas = []
                for ix, persona_dict in enumerate(random_personas_dict):
                    persona_dict = llm_output[ix] if ix < len(llm_output) else persona_dict
                    try:
                        personas.append(self._persona.model_validate(persona_dict))
                        personas[-1]._metadata = PersonaMetadata(
                            model=str(llm) if llm else None,
                            seed=seed,
                            id=id if id is not None else get_universal_id(),
                            parentId=parent_id,
                            className=type(output_persona).__name__,
                            notes=notes
                        )
                    except ValidationError as e:
                        logger.warning(f"Validation error in generated persona {ix + 1}: {e}")
                        persona_dict = {k: persona_dict[k]
                                        if k in persona_dict and persona_dict[k] not in [None, "", "null"]
                                        else v
                                        for k, v in self._persona.__class__().model_dump().items()}
                        personas.append(self._persona.model_validate(persona_dict))
                        personas[-1]._metadata = PersonaMetadata(
                            model=str(llm) if llm else None,
                            seed=seed,
                            id=id if id is not None else get_universal_id(),
                            parentId=parent_id,
                            className=type(output_persona).__name__,
                            notes=notes
                        )
                if len(personas) != n:
                    logger.warning(f"Only {len(personas)} personas out of {n }were generated fully. "
                                   "Returning the generated personas anyway.")
                return personas
            else:
                try:
                    if any(value in [None, "", "null"] for value in random_persona_dict.values()):
                        raise ValidationError([], [])
                    output_persona = self._persona.model_validate(random_persona_dict)
                    break
                except ValidationError:
                    missing_attributes = {
                        k: v for k, v in self._persona.model_dump().items()
                        if k not in random_persona_dict or random_persona_dict[k] in [None, "", "null"]
                    }
                    logger.warning(
                        f"The following {len(missing_attributes)} attributes are missing in the "
                        f"generated persona: {', '.join(missing_attributes.keys())}. "
                        f"Trying to fill the missing attributes again (attempt {attempt + 1} out of {max_attempts})..."
                    )

                    target_persona_dict = {k: v if k in missing_attributes else random_persona_dict[k]
                                           for k, v in target_persona_dict.items()}

        # If we ran out of attempts and still have missing attributes...
        # we return a persona with missing attributes filled with default null values
        if output_persona is None:
            logger.warning(
                f"The generated persona is missing the following {len(missing_attributes)} attributes: "
                f"{', '.join(missing_attributes.keys())}."
            )
            random_persona_dict.update(missing_attributes)
            output_persona = self._persona.model_validate(random_persona_dict)

        # Adding metadata to the generated persona
        # TODO: shall we also add generator parameters? (e.g. self._persona_rnd_attributes, self.default_*)
        output_persona._metadata = PersonaMetadata(
            model=str(llm) if llm else None,
            seed=seed,
            id=id if id is not None else get_universal_id(),
            parentId=parent_id,
            className=type(output_persona).__name__,
            notes=notes
        )
        return output_persona
