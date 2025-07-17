# medical_dialogue_evaluator/evaluators/base.py
"""
Defines the abstract base class (ABC) for all evaluation indicators.
This modular design allows users to easily add new evaluators.
"""
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """Abstract base class for a single evaluation indicator."""
    @property
    @abstractmethod
    def indicator_id(self) -> str:
        """A short, unique, machine-readable identifier (e.g., 'med_knowledge')."""
        pass
        
    @property
    @abstractmethod
    def indicator_name(self) -> str:
        """The official name of the indicator."""
        pass

    @property
    @abstractmethod
    def definition(self) -> str:
        """The detailed definition of what this indicator measures."""
        pass

    @property
    @abstractmethod
    def scoring_rubric(self) -> dict:
        """A dictionary with 'low_example' and 'high_example' descriptions."""
        pass