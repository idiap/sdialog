# medical_dialogue_evaluator/evaluators/__init__.py
"""
Automates the discovery of all evaluator classes.
"""
import inspect
import pkgutil
from pathlib import Path
from typing import List
from .base import BaseEvaluator

def discover_evaluators() -> List[BaseEvaluator]:
    """Finds and instantiates all BaseEvaluator subclasses in this package."""
    evaluator_instances = []
    package_path = Path(__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        if module_name == 'base':
            continue
        module = __import__(f"{__name__}.{module_name}", fromlist=["*"])
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseEvaluator) and obj is not BaseEvaluator:
                evaluator_instances.append(obj())
                
    return sorted(evaluator_instances, key=lambda x: x.indicator_name)

ALL_EVALUATORS = discover_evaluators()

