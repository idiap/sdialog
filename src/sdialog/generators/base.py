"""
Base and abstract classes for generators in sdialog.
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
from jinja2 import Template
from typing import List
from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage

from .. import Turn
from ..config import config
from ..base import Metadata, BaseAttributeModel
from ..util import get_universal_id, get_llm_model, get_llm_default_params, is_ollama_model_name, is_openai_model_name

logger = logging.getLogger(__name__)


class LLMDialogOutput(BaseModel):
    """
    Pydantic model for LLM-generated dialogue output.

    :param dialog: Ordered list of generated dialogue turns.
    :type dialog: List[Turn]

    :meta private:
    """
    dialog: List[Turn]


class AttributeObject(BaseAttributeModel):
    """
    Generic attribute object used for structured generation of placeholder-bearing entities.

    :param placeholder: placeholder attribute to be removed and replaced by real attributes at generation time.
    :type placeholder: str

    :meta private:
    """
    placeholder: str = None


class ListOfAttributeObjects(BaseModel):
    """
    Container model for validating a list of AttributeObject instances.

    :param objects: Collection of attribute objects.
    :type objects: List[AttributeObject]

    :meta private:
    """
    objects: List[AttributeObject]


_objects_schema = ListOfAttributeObjects.model_json_schema()


class BaseAttributeModelGenerator(ABC):
    """
    Abstract class to create subclasses for generators with randomized and/or LLM-populated attributes.

    Workflow:

      1. Provide a target attribute model instance or class.
      2. Configure attribute generation rules (e.g. ``.set(...)`` or ``generated_attributes='all'``).
      3. Call ``generate(n=...)`` to produce validated instances.

    :param attribute_model: Instance or subclass of BaseAttributeModel to generate.
    :type attribute_model: BaseAttributeModel
    :param generated_attributes: Attribute selection strategy ("all", iterable, or dict of rules).
    :type generated_attributes: Union[str, list, dict]
    :param extra_instructions: Additional instructions to include in the LLM prompt.
    :type extra_instructions: str
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
    def __init__(self,
                 attribute_model: BaseAttributeModel,
                 generated_attributes: str = "all",
                 extra_instructions: str = "",
                 model: str = None,
                 system_prompt: str = None,
                 llm_prompt: str = None,
                 llm_prompt_n: str = None,
                 **llm_kwargs):
        """Initializes a BaseAttributeModelGenerator."""
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

        if extra_instructions:
            lines = [ln.strip(" •-\t") for ln in str(extra_instructions).splitlines() if ln.strip()]
            if len(lines) == 1:
                extra_block = f"Additional instructions: {lines[0]}"
            else:
                extra_block = "Additional instructions:\n" + "\n".join(f"- {ln}" for ln in lines)
            self.system_prompt = f"{self.system_prompt}\n\n{extra_block}"

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

    def set(self, **attributes):
        """
        Define per-attribute randomization / generation specifications as ``attribute_name=<value>``.

        Where ``<value>`` can be:

          * "*": Defer to LLM.
          * A callable: Invoked (with current partial object as kwargs if compatible).
          * A list: Random element chosen.
          * A fixed scalar / str: Assigned directly.
          * A templated string ``"{...}"``:

            * ``"{min-max}"``: Random int in inclusive range.
            * ``"{txt:PATH}"``: Random non-empty line from file.
            * ``"{csv:COLUMN:PATH}"``: Random value from CSV column (name or index).
            * ``"{tsv:COLUMN:PATH}"``: Same for TSV.
            * ``"{llm}"``: Defer to LLM.
            * ``"{llm:INSTRUCTION}"``: Defer with custom instruction.

        Example:

            .. code-block:: python

                from sdialog.generators import ContextGenerator

                ctx_gen = ContextGenerator()

                ctx_gen.set(
                    location=["office", "home", "school"],
                    objects=get_objects_from_db,  # callable function
                    circumstances="{csv:circumstances:./data/circumstances.csv}",
                    goals="{llm:Suggest a realistic goal for the context}"
                )

                my_context = ctx_gen.generate()
                my_context.print()

        :param attributes: Mapping of attribute name -> generation rule.
        :raises ValueError: If any attribute is not defined on the target model.
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
                if is_openai_model_name(self.llm_model):
                    schema["description"] = schema["description"][:1024]  # To avoid API string maximum length error
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
                llm_kwargs = get_llm_default_params(self.llm_model, llm_kwargs)
                if temperature is not None:
                    llm_kwargs["temperature"] = temperature
                llm_kwargs["seed"] = seed + attempt

                llm = get_llm_model(model_name=self.llm_model,
                                    output_format=schema,
                                    **llm_kwargs)
                model_info = {"name": str(self.llm_model), **llm_kwargs}

                messages = [SystemMessage(self.system_prompt), HumanMessage(prompt)]

                if n > 1:
                    for ix in range(max_attempts):
                        llm_output = llm.invoke(messages)
                        if not is_ollama_model_name(self.llm_model):
                            llm_output = llm_output["objects"]
                        if isinstance(llm_output, list) and all(isinstance(obj, dict) for obj in llm_output):
                            break
                        logger.warning("LLM output is not a list of objects, retrying "
                                       f"((attempt {ix + 1} out of {max_attempts}))...")
                    if isinstance(llm_output, list) and all(isinstance(obj, dict) for obj in llm_output):
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
                            model=model_info if llm else None,
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
                            model=model_info if llm else None,
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
            model=model_info if llm else None,
            seed=seed,
            id=id if id is not None else get_universal_id(),
            parentId=parent_id,
            className=type(self._attribute_model).__name__,
            notes=notes
        )
        return output_object
