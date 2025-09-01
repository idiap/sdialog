import os
import sys
import json
import inspect

from abc import ABC

from typing import Union, Optional
from print_color import print as cprint
from pydantic import BaseModel, Field, ConfigDict

from .util import _get_dynamic_version, get_timestamp, get_universal_id, camel_or_snake_to_words, remove_newlines

# Global cache for AttributeModel subclass lookup
_OBJECT_CLASS_MAP = None


def _build_attribute_model_class_map():
    """
    Build (once) and return a mapping {className: Class} for all AttributeModel subclasses
    discovered in any already-imported module that references the same AttributeModel object.
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

    :ivar version: Version of the object format (matches sdialog version).
    :vartype version: Optional[str]
    :ivar timestamp: Timestamp of when the object was generated.
    :vartype timestamp: Optional[str]
    :ivar model: The model used to generate the object.
    :vartype model: Optional[str]
    :ivar seed: The random seed used for object generation.
    :vartype seed: Optional[int]
    :ivar id: Unique identifier for the object.
    :vartype id: Optional[int]
    :ivar parentId: ID of the parent object, if any.
    :vartype parentId: Optional[int]
    :ivar notes: Free-text notes or comments about the generated object.
    :vartype notes: Optional[str]
    :ivar className: The class name of the object (a subclass of AttributeModel).
    :vartype className: str
    """
    version: Optional[str] = Field(default_factory=_get_dynamic_version)
    timestamp: Optional[str] = Field(default_factory=get_timestamp)
    model: Optional[str] = None
    seed: Optional[int] = None
    id: Optional[Union[int, str]] = Field(default_factory=get_universal_id)
    parentId: Optional[Union[int, str]] = None
    className: str = None
    notes: Optional[str] = None


class BaseAttributeModel(BaseModel, ABC):
    """
    Base class for defining a attribute-based objects.
    """
    model_config = ConfigDict(extra='forbid')
    _metadata: Optional[Metadata] = None

    # Automatically inject a staticmethod attributes() into every subclass
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def _attributes(_cls=cls):
            return [k for k in _cls.model_fields.keys() if not k.startswith("_")]
        cls.attributes = staticmethod(_attributes)

    @staticmethod
    def attributes():
        """
        Return the list of attribute (field) names for AttributeModel itself.
        Subclasses get their own version injected in __init_subclass__.
        """
        return [k for k in BaseAttributeModel.model_fields.keys() if not k.startswith("_")]

    def clone(self, new_id: int = None, **kwargs) -> "BaseAttributeModel":
        """
        Creates a deep copy of the object, with optional attribute overrides.

        The cloned object will have all attributes copied from the original, with any provided keyword arguments
        (`kwargs`) used to override or update specific fields. The clone receives a new metadata object:

        - The `parentId` field in the clone's metadata is set to the original object's `id` (if present).
        - The `id` field in the clone's metadata is set to `new_id` if provided, otherwise to the original's `id`.
        - All other metadata fields are copied from the original.

        This method is useful for generating variations of a object for ablation, branching, or scenario testing
        without modifying the original instance. The clone is a fully independent object.

        :param new_id: Optional new unique ID for the cloned object.
        :type new_id: int, optional
        :param kwargs: Attributes to override in the cloned object.
        :return: A new instance of the object with updated attributes and metadata.
        :rtype: AttributeModel
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

    def print(self, object_name: str = "Object"):
        """
        Pretty-prints the object, including its metadata information.
        """
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
                   color="white")
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
        Loads object from a file.

        :param path: Path to the object file.
        :type path: str
        :param object_class: Optional specific class to use for the object.
        :type object_class: Optional[AttributeModel]
        :return: The loaded object object.
        :rtype: MetaPersona
        """
        return BaseAttributeModel.from_json(open(path, "r", encoding="utf-8").read(), object_class)

    @staticmethod
    def from_dict(data: dict, object_class: Optional["BaseAttributeModel"] = None):
        """
        Creates a object object from a dictionary.

        :param data: The dictionary containing object data.
        :type data: dict
        :param object_class: Optional specific class to use for the object.
        :type object_class: Optional[AttributeModel]
        :return: The created object object.
        :rtype: MetaPersona
        """
        # Assign to "object" the instance of the right class using the `className`
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
            raise ValueError("Metadata with `className` is required to create a object from a dict or json.")

    @staticmethod
    def from_json(json_str: str, object_class: Optional["BaseAttributeModel"] = None):
        """
        Creates a object object from a JSON string.

        :param json_str: The JSON string containing object data.
        :type json_str: str
        :param object_class: Optional specific class to use for the object.
        :type object_class: Optional[AttributeModel]
        :return: The created object object.
        :rtype: MetaPersona
        """
        return BaseAttributeModel.from_dict(json.loads(json_str), object_class)
