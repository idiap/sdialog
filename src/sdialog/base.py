"""
Model foundations for sdialog root module.

Provides:

  - Metadata: common provenance fields (version, timestamp, ids).
  - BaseAttributeModel: pydantic-based abstract base for persona/context-like objects
    with cloning, serialization, and dynamic subclass discovery utilities.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import sys
import json
import inspect

from abc import ABC

from print_color import print as cprint
from typing import Union, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict

from .util import _get_dynamic_version, get_timestamp, get_universal_id, camel_or_snake_to_words, remove_newlines

# Global cache for AttributeModel subclass lookup
_OBJECT_CLASS_MAP = None


def _build_attribute_model_class_map():
    """
    Build (once) and return a mapping {className: Class} for all BaseAttributeModel subclasses
    discovered in any already-imported module that references the same BaseAttributeModel object.

    :return: Mapping from subclass name to the subclass type.
    :rtype: dict
    """
    global _OBJECT_CLASS_MAP
    if _OBJECT_CLASS_MAP is not None:
        return _OBJECT_CLASS_MAP

    target_attrmodel = globals().get("BaseAttributeModel")
    modules = set()

    # Collect every loaded module whose AttributeModel symbol points to the same class object
    for m in list(sys.modules.values()):
        if not m:
            continue
        try:
            if getattr(m, "BaseAttributeModel", None) is target_attrmodel:
                modules.add(m)
        except Exception:
            continue

    cls_map = {}
    for _m in modules:
        try:
            for _, cls in inspect.getmembers(_m, inspect.isclass):
                if target_attrmodel and cls is not target_attrmodel and issubclass(cls, target_attrmodel):
                    cls_map[cls.__name__] = cls
        except Exception:
            continue

    _OBJECT_CLASS_MAP = cls_map
    return _OBJECT_CLASS_MAP


class Metadata(BaseModel):
    """
    Metadata class for object, context and other objects.

    :param version: Version of the object format (matches sdialog version).
    :type version: Optional[str]
    :param timestamp: Timestamp of when the object was generated.
    :type timestamp: Optional[str]
    :param model: The model used to generate the object.
    :type model: Optional[str]
    :param seed: The random seed used for object generation.
    :type seed: Optional[int]
    :param id: Unique identifier for the object.
    :type id: Optional[Union[int, str]]
    :param parentId: ID of the parent object, if any.
    :type parentId: Optional[Union[int, str]]
    :param notes: Free-text notes or comments about the generated object.
    :type notes: Optional[str]
    :param className: The class name of the object (a subclass of BaseAttributeModel).
    :type className: str
    """
    version: Optional[str] = Field(default_factory=_get_dynamic_version)
    timestamp: Optional[str] = Field(default_factory=get_timestamp)
    model: Optional[Union[str, Dict]] = None
    seed: Optional[int] = None
    id: Optional[Union[int, str]] = Field(default_factory=get_universal_id)
    parentId: Optional[Union[int, str]] = None
    className: str = None
    notes: Optional[str] = None


class BaseAttributeModel(BaseModel, ABC):
    """
    Base class for defining an attribute-based object.

    Features:

      - Strict field control.
      - Automatic static attributes() helper listing declared fields.
      - Metadata tracking (id, parentId, version, timestamp).
      - Clone with optional field overrides and proper lineage linkage.
      - JSON / prompt serialization helpers.
    """
    model_config = ConfigDict(extra='forbid')
    _metadata: Optional[Metadata] = None

    # Automatically inject a staticmethod attributes() into every subclass
    def __init_subclass__(cls, **kwargs):
        """Injects a static attributes(print=False) helper into every subclass."""
        super().__init_subclass__(**kwargs)

        def _attributes(_cls=cls, print=False):
            """
            List (or pretty-print) public attribute field names for this subclass.

            :param print: If True, pretty-prints instead of returning the list.
            :type print: bool
            :return: List of attribute names (if print=False).
            :rtype: List[str] | None
            """
            items = []
            for name, _ in _cls.model_fields.items():
                if name.startswith("_"):
                    continue
                items.append(name)

            if print:
                cprint(f"--- {_cls.__name__} Begins ---", color="magenta", format="bold")
                for attr in items:
                    cprint("", tag=attr, tag_color="red", color="white")
                cprint(f"--- {_cls.__name__} Ends ---", color="magenta", format="bold")
            else:
                return items

        cls.attributes = staticmethod(_attributes)

    def clone(self, new_id: int = None, **kwargs) -> "BaseAttributeModel":
        """
        Create a deep copy of this object with optional attribute overrides.

        Metadata handling:

          - parentId of clone = original id (if present).
          - id of clone = new_id if provided else a new universal id.
          - Other metadata fields are copied.

        :param new_id: Optional new unique id for the clone.
        :type new_id: Optional[int]
        :param kwargs: Field overrides applied to the cloned instance.
        :type kwargs: Any
        :return: Independent cloned instance.
        :rtype: BaseAttributeModel
        """
        data = self.json()
        data.update(kwargs)
        if "_metadata" in data:
            del data["_metadata"]  # to avoid model validation issues
        new_object = self.__class__(**data)
        if self._metadata:
            new_object._metadata = self._metadata.model_copy()
            new_object._metadata.parentId = self._metadata.id if self._metadata.id else None
            new_object._metadata.id = new_id if new_id is not None else get_universal_id()
        else:
            new_object._metadata = Metadata(className=self.__class__.__name__,
                                            id=new_id if new_id is not None else get_universal_id(),
                                            parentId=self._metadata.id if self._metadata else None)
        return new_object

    def description(self) -> str:
        """
        Returns a string description of the object's attributes.

        :return: Description of the object.
        :rtype: str
        """
        return "\n".join(f"* {camel_or_snake_to_words(key).capitalize()}: {value}"
                         for key, value in self.__dict__.items()
                         if value not in [None, ""])

    def __str__(self) -> str:
        """
        Returns the string representation of the object.

        :return: Description of the object.
        :rtype: str
        """
        return self.description()

    def print(self):
        """
        Pretty-prints the object, including its metadata information.
        """
        object_name = self.__class__.__name__
        if hasattr(self, "_metadata") and self._metadata is not None:
            for key, value in self._metadata.model_dump().items():
                if value not in [None, ""]:
                    cprint(remove_newlines(value), tag=key, tag_color="purple", color="magenta", format="bold")
        cprint(f"--- {object_name} Begins ---", color="magenta", format="bold")
        for key, value in self.__dict__.items():
            if key == "_metadata":
                continue
            cprint(remove_newlines(value),
                   tag=camel_or_snake_to_words(key).capitalize(),
                   tag_color="red",
                   color="grey")
        cprint(f"--- {object_name} Ends ---", color="magenta", format="bold")

    def json(self, string: bool = False, indent=2, output_metadata: bool = True):
        """
        Serializes the object to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :param output_metadata: Include the metadata in the serialization.
        :type output_metadata: bool
        :return: The serialized object.
        :rtype: Union[str, dict]
        """
        data = {key: value for key, value in self.__dict__.items() if value not in [None, ""]}
        if self._metadata and output_metadata:
            data["_metadata"] = self._metadata.model_dump()
        return json.dumps(data, indent=indent) if string else data

    def prompt(self) -> str:
        """
        Returns the textual representation of the object, used as part of the system prompt.

        :return: JSON string without metadata (intended for prompt inclusion).
        :rtype: str
        """
        return self.json(string=True, output_metadata=False)

    def to_file(self, path: str, makedir: bool = True):
        """
        Saves the object to a file in either JSON or plain text format.

        :param path: Output file path.
        :type path: str
        :param makedir: If True, creates parent directories as needed.
        :type makedir: bool
        """
        if makedir and os.path.split(path)[0]:
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        if self._metadata is None:
            self._metadata = Metadata(className=self.__class__.__name__)

        with open(path, "w") as writer:
            writer.write(self.json(string=True))

    @staticmethod
    def from_file(path: str, object_class: Optional["BaseAttributeModel"] = None):
        """
        Load an object from a JSON file.

        :param path: Path to file.
        :type path: str
        :param object_class: Optional explicit subclass to force (bypasses className dispatch).
        :type object_class: Optional[BaseAttributeModel]
        :return: Loaded instance.
        :rtype: BaseAttributeModel
        :raises ValueError: If metadata/className is missing or unknown.
        """
        return BaseAttributeModel.from_json(open(path, "r", encoding="utf-8").read(), object_class)

    @staticmethod
    def from_dict(data: dict, object_class: Optional["BaseAttributeModel"] = None):
        """
        Create an object instance from a dictionary.

        Dispatch rules:

          - If object_class is provided and is a BaseAttributeModel subclass, it is used directly.
          - Else uses _metadata.className to resolve a registered subclass.

        :param data: Source dictionary (must include _metadata.className).
        :type data: dict
        :param object_class: Optional explicit subclass.
        :type object_class: Optional[BaseAttributeModel]
        :return: Instantiated object.
        :rtype: BaseAttributeModel
        :raises ValueError: If className missing or cannot be resolved.
        """
        # Assign to "object" the instance of the right class using the ``className``
        if "_metadata" in data and "className" in data["_metadata"] and data["_metadata"]["className"]:
            object_class_name = data["_metadata"]["className"]
            metadata = Metadata(**data["_metadata"])
            del data["_metadata"]  # to avoid model_validate(data) issues
            if object_class and issubclass(object_class, BaseAttributeModel):
                # If the user provided a specific class, use it
                object = object_class.model_validate(data)
                object._metadata = metadata
                return object
            else:  # Assuming the class name is from one of the built-in classes
                object_class_map = _build_attribute_model_class_map()
                object_class = object_class_map.get(object_class_name)
                if object_class:
                    object = object_class.model_validate(data)
                    object._metadata = metadata
                    return object
                else:
                    raise ValueError(f"Unknown object class given in the `className` field: {object_class_name}.")
        else:
            raise ValueError("Metadata with `className` is required to create an object from a dict or json.")

    @staticmethod
    def from_json(json_str: str, object_class: Optional["BaseAttributeModel"] = None):
        """
        Create an object instance from a JSON string.

        :param json_str: JSON serialization including _metadata.className.
        :type json_str: str
        :param object_class: Optional explicit subclass override.
        :type object_class: Optional[BaseAttributeModel]
        :return: Instantiated object.
        :rtype: BaseAttributeModel
        """
        return BaseAttributeModel.from_dict(json.loads(json_str), object_class)
