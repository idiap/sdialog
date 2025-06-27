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

from jinja2 import Template
from typing import Union, List, Any
from pydantic import BaseModel, ValidationError
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from . import Dialog, Turn
from .personas import BasePersona, Persona, PersonaAgent


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

    :ivar model_name: The model or model name used for generation.
    :vartype model_name: str
    :ivar output_format: The output format (Pydantic model or dict).
    :vartype output_format: Union[dict, BaseModel]
    :ivar scenario: Scenario metadata for the dialogue.
    :vartype scenario: dict
    :ivar dialogue_details: Instructions or details for the dialogue.
    :vartype dialogue_details: str
    :ivar messages: List of system and human messages for the LLM.
    :vartype messages: list
    """
    def __init__(self,
                 model: Union[ChatOllama, str],
                 dialogue_details: str,
                 output_format: Union[dict, BaseModel] = LLMDialogOutput,
                 scenario: dict = None,
                 personas: dict[str, dict[str, Any]] = None):
        """
        Initializes a DialogGenerator.

        :param model: The LLM or model name to use.
        :type model: Union[ChatOllama, str]
        :param dialogue_details: Instructions or details for the dialogue.
        :type dialogue_details: str
        :param output_format: Output format schema or Pydantic model.
        :type output_format: Union[dict, BaseModel]
        :param scenario: Scenario metadata for the dialogue (if not provided, value set to `dialogue_details`).
        :type scenario: dict
        :param personas: Optional personas for role-playing in the dialogue (if any).
        :type personas: dict[str, dict[str, Any]]
        """

        if not output_format or type(output_format) is dict:
            output_format_schema = output_format
            self.output_format = None
        else:
            output_format_schema = output_format.model_json_schema()
            self.output_format = output_format

        if type(model) is str:
            self.llm = ChatOllama(model=model,
                                  format=output_format_schema,
                                  temperature=0.8,
                                  seed=13)
        else:
            self.llm = model
            if output_format:
                self.llm.format = output_format

        self._personas = personas
        self.model_name = model
        self.set(dialogue_details, scenario)

    def generate(self, seed: int = None, id: int = None, notes: str = None):
        """
        Generates a synthetic dialogue using the LLM.

        :param seed: Random seed for reproducibility.
        :type seed: int
        :param id: Dialogue ID.
        :type id: int
        :param notes: Optional notes to include in the dialogue.
        :type notes: str
        :return: The generated dialogue or output object.
        :rtype: Union[Dialog, dict, BaseModel]
        """
        self.llm.seed = seed if seed is not None else random.getrandbits(32)

        # hack to avoid seed bug in prompt cache
        # (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
        _ = self.llm.num_predict
        self.llm.num_predict = 1
        self.llm.invoke(self.messages)
        self.llm.num_predict = _

        dialogue = self.llm.invoke(self.messages).content

        if not self.output_format:
            return dialogue
        else:
            llm_output = self.output_format.model_validate(json.loads(dialogue))

            if self.output_format is LLMDialogOutput:
                return Dialog(dialogId=id if id else None,
                              model=self.model_name,
                              seed=self.llm.seed,
                              personas=self._personas,
                              scenario=self.scenario if self.scenario else self.dialogue_details,
                              notes=notes,
                              turns=llm_output.dialog)
            else:
                return llm_output

    def set(self, dialogue_details: str, scenario: dict = None):
        """
        Sets the dialogue details and scenario for generation.

        :param dialogue_details: Instructions or details for the dialogue.
        :type dialogue_details: str
        :param scenario: Scenario metadata.
        :type scenario: dict
        """
        self.scenario = scenario
        self.dialogue_details = dialogue_details
        self.messages = [
            SystemMessage(
                "You are a knowledgeable and useful AI assistant that can write natural conversations by role paying "
                "different speakers. The output should be a full dialogue, from begining (greetings) to end (bye bye "
                "messages)."
            ),
            HumanMessage(content=dialogue_details)
        ]

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
                 model: Union[ChatOllama, str],
                 persona_a: Union[Persona, PersonaAgent],
                 persona_b: Union[Persona, PersonaAgent],
                 dialogue_details: str = "",
                 response_details: str = "responses SHOULD NOT be too long and wordy, should be "
                                         "approximately one utterance long",
                 scenario: dict = None):
        """
        Initializes a PersonaDialogGenerator.

        :param model: The LLM or model name to use.
        :type model: Union[ChatOllama, str]
        :param persona_a: The first persona.
        :type persona_a: Persona (or PersonaAgent)
        :param persona_b: The second persona.
        :type persona_b: Persona (or PersonaAgent)
        :param dialogue_details: Additional dialogue instructions.
        :type dialogue_details: str
        :param response_details: Instructions for response style.
        :type response_details: str
        :param scenario: Scenario metadata.
        :type scenario: dict
        """

        if isinstance(persona_a, PersonaAgent) and isinstance(persona_b, PersonaAgent):
            self._agent_a = persona_a
            self._agent_b = persona_b
            persona_a = persona_a.persona
            persona_b = persona_b.persona

        dialogue_details = f"""Role play as the following two characters having a conversations. The characters are defined by the personas in the following lines. You always stay in character.
[[ ## BEGING FIRST PERSONA ## ]]
{persona_a}
[[ ## END FIRST PERSONA ## ]]

[[ ## BEGING SECOND PERSONA ## ]]
{persona_b}
[[ ## END SECOND PERSONA ## ]]
---
{"Details about the overall dialogue: " + dialogue_details if dialogue_details else ""}
{"Details about your responses: " + response_details if response_details else ""}
Finally, remember:
   1. You always stay on character. You are the characters described above.
   2. Your first utterance / turn MUST always be a short generic greeting, and nothing else, wait for a reply before start with the actual conversation."""  # noqa: E501
        super().__init__(model=model,
                         dialogue_details=dialogue_details,
                         scenario=scenario,
                         personas={
                             persona_a.name: persona_a.json(),
                             persona_b.name: persona_b.json()
                         })

    def generate(self, seed: int = None, id: int = None, max_iterations: int = 20, notes: str = None):
        if self._agent_a and self._agent_b:
            return self._agent_a.dialog_with(self._agent_b,
                                             max_iterations=max_iterations,
                                             id=id,
                                             seed=seed,
                                             notes=notes)
        else:
            return super().generate(seed=seed, id=id, notes=notes)

    __call__ = generate  # alias for generate method


class PersonaGenerator:
    """
    Generates persona objects with randomized or LLM-populated attributes.

    :param persona: An instance or class of BasePersona to generate personas from.
    :type persona: BasePersona
    :param default_attributes: Specifies which attributes to fill by default. Can be "all", a list of attribute names, or None. Defaults to "all".
    :type default_attributes: str, list, or dict, optional
    :param default_llm: The default language model to use for attribute population via LLM.
    :type default_llm: str, optional
    :param default_llm_prompt: The prompt template for the LLM to fill empty attributes.
    :type default_llm_prompt: str, optional

    :ivar _persona: The persona instance being generated.
    :vartype _persona: BasePersona
    :ivar _persona_rnd_attributes: Attributes to be randomly set or filled.
    :vartype _persona_rnd_attributes: dict
    :ivar default_attributes: Default attributes specification.
    :vartype default_attributes: str, list, or dict
    :ivar default_llm: Default LLM model name or instance.
    :vartype default_llm: str
    :ivar default_llm_prompt: Prompt template for LLM.
    :vartype default_llm_prompt: str

    :raises ValueError: If specified attributes do not exist in the persona or if required files for templates are missing.

    :example:
        generator = PersonaGenerator(MyPersonaClass)
        persona_instance = generator.generate()
    """  # noqa: E501

    def __init__(self,
                 persona: BasePersona,
                 default_attributes: str = "all",  # None
                 default_llm: str = "qwen2.5:14b",
                 default_llm_prompt: str = (
                     "You are an expert in creating realistic persona profiles for dialogue systems. "
                     "Given the following JSON object representing a `{{persona_class_name}}` persona, "
                     "fill in ALL attributes that are set to null with plausible, coherent, and diverse values. "
                     "Ensure all fields are completed in fluent English, and the persona is internally consistent. "
                     "Return ONLY the completed JSON object, with no extra commentary or explanation.\n"
                     "{{persona}}\n\n"
                     "{{attributes_instructions}}"
                 )):
        if isinstance(persona, BasePersona):
            self._persona = persona
        elif isinstance(persona, type) and issubclass(persona, BasePersona):
            self._persona = persona()

        if isinstance(default_attributes, (list, dict)):
            self._check_attributes(default_attributes)

        self._persona_rnd_attributes = default_attributes if isinstance(default_attributes, dict) else {}

        self.default_attributes = default_attributes
        self.default_llm = default_llm
        self.default_llm_prompt = default_llm_prompt

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

    def set_random_attributes(self, **persona_rnd_attributes):
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

        :param persona_rnd_attributes: Keyword arguments of attribute names and values to set.
        :raises ValueError: If any attribute is not found in the persona.
        """  # noqa: E501
        self._check_attributes(persona_rnd_attributes)
        self._persona_rnd_attributes = persona_rnd_attributes

    def generate(self, seed: int = None, temperature: float = 0.8):
        """
        Generate a persona instance with attributes filled by random selection, templates, or LLM as needed.

        :param seed: Optional random seed for reproducibility.
        :type seed: int, optional
        :param temperature: Temperature for LLM generation (if applicable).
        :type temperature: float, optional
        :return: A validated persona instance.
        :rtype: BasePersona
        :raises ValueError: If required files for templates are missing.
        """
        seed = seed if seed is not None else random.getrandbits(32)
        random.seed(seed)

        llm_attribute_instructions_txt = ""
        llm_attribute_instructions = {}
        random_persona_dict = {}
        target_persona_dict = self._persona.__dict__
        target_persona_dict.update(self._persona_rnd_attributes)
        for key, value in target_persona_dict.items():
            if callable(value):
                random_persona_dict[key] = value  # a callable
            elif isinstance(value, list):
                random_persona_dict[key] = random.choice(value)
            elif isinstance(value, str) and value:
                if value == "*":
                    random_persona_dict[key] = None  # to be filled by the LLM
                elif value.startswith("{{") and value.endswith("}}"):  # templates
                    # TODO: Shall we also have pre-devined lists for name and other attributes
                    #       and then have temples like {{name}} to use them?
                    m_range = re.match(r"{{(\d+)-(\d+)}}", value)  # match {{min-max}}
                    m_txt = re.match(r"{{txt:(.+)}}", value)  # path to txt file (one line per value)
                    m_csv = re.match(r"{{csv:(\w+):(.+)}}", value)  # path to csv file (column to sample from)
                    m_tsv = re.match(r"{{tsv:(\w+):(.+)}}", value)  # path to tsv file (column to sample from)
                    m_llm = re.match(r"{{llm(:.+)?}}", value)  # LLM template with optional instruction
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
                            raise ValueError(f"File '{txt_path}' not found for '{value}' attribute.")
                    elif m_csv or m_tsv:
                        m_csv = m_csv or m_tsv
                        csv_column, csv_path = m_csv.group(1), m_csv.group(2)
                        csv_column = int(csv_column) if csv_column.isdigit() else csv_column
                        try:
                            with open(csv_path, newline='') as csvfile:
                                if isinstance(csv_column, int):
                                    reader = csv.reader(csvfile, delimiter='\t' if m_tsv else ',')
                                    values = [row[csv_column] for row in reader if row[csv_column]]
                                else:
                                    reader = csv.DictReader(csvfile, delimiter='\t' if m_tsv else ',')
                                    if csv_column not in reader.fieldnames:
                                        raise ValueError(f"Column '{csv_column}' not found in CSV file '{csv_path}'.")
                                    values = [row[csv_column] for row in reader if row[csv_column]]
                            random_persona_dict[key] = random.choice(values)
                        except FileNotFoundError:
                            raise ValueError(f"File '{csv_path}' not found for '{value}' attribute.")
                    elif m_llm:
                        random_persona_dict[key] = None  # to be filled by the LLM

                        instruction = m_llm.group(1)[1:] if m_llm.group(1) else None  # get instruction if provided
                        if instruction:
                            llm_attribute_instructions[key] = instruction

                    # elif value == "{{name}}":
                    #     random_persona_dict[key] = get_name(seed=seed)  # get name from pre-defined list
                else:
                    random_persona_dict[key] = value
            elif self.default_attributes and (self.default_attributes == "all" or key in self.default_attributes):
                random_persona_dict[key] = None  # to be filled by the LLM

        for key, value in random_persona_dict.items():
            if callable(value):
                try:
                    random_persona_dict[key] = value(**random_persona_dict)
                except TypeError:
                    random_persona_dict[key] = value()  # in case user-proved function has no arguments

        # If there are None value, we need to fill them using the LLM
        if any(value is None for value in random_persona_dict.values()):
            if llm_attribute_instructions:
                llm_attribute_instructions_txt = ("Consider the following instructions for filling "
                                                  "the following attributes:\n")
                llm_attribute_instructions_txt += "\n".join(
                    [f"* {k}: {v}." for k, v in llm_attribute_instructions.items()]
                )
            template = Template(self.default_llm_prompt)
            prompt = template.render(
                persona=json.dumps(random_persona_dict, indent=2),
                persona_class_name=str(type(self._persona).__name__),
                attributes_instructions=llm_attribute_instructions_txt
            )

            if not isinstance(self.default_llm, str):
                llm = self.default_llm
            else:
                schema = self._persona.model_json_schema()
                schema["properties"] = {k: v
                                        for k, v in schema["properties"].items()
                                        if k in random_persona_dict}

                llm = ChatOllama(model=self.default_llm,
                                 format=schema,
                                 temperature=temperature,
                                 seed=seed)

            messages = [
                SystemMessage("You are an expert at generating persona JSON objects "
                              "for synthetic dialogue generation."),
                HumanMessage(prompt)
            ]
            persona_llm_dict = json.loads(llm.invoke(messages).content)
            random_persona_dict.update({k: v
                                        for k, v in persona_llm_dict.items()
                                        if random_persona_dict[k] is None})

        try:
            random_persona = self._persona.model_validate(random_persona_dict)
        except ValidationError:
            missing_attributes = {k: v for k, v in self._persona.items()
                                  if k not in random_persona_dict or random_persona_dict[k] is None}
            print(f"\n\nWARNING! There following {len(missing_attributes)} attributes are missing in the "
                  f"generated persona: {','.join(missing_attributes.keys())}.\n\n")
            random_persona_dict.update(missing_attributes)
            random_persona = self._persona.model_validate(random_persona_dict)

        return random_persona
