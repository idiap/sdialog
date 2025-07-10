"""
evaluation: Evaluation components for dialogue generation and analysis.

This module provides abstract base classes for evaluating dialogues,
including LLM judges, metrics, and similarity scores.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json

from typing import Union, List
from abc import ABC, abstractmethod

from jinja2 import Template
from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from .. import Dialog
from ..config import config
from ..util import get_llm_model


class LLMJudgeYesNoOutput(BaseModel):
    """
    Pydantic model for LLM-generated dialogue output.
    """
    output: bool
    feedback: Optional[str] = None


class BaseEvaluator(ABC):
    def __init__(self, metrics=None):
        pass

    @abstractmethod
    def evaluate(self, input: Union[Dialog, List[Dialog]]) -> dict:
        """
        Evaluate the dialogs.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseMetric(ABC):
    """
    Base class for metrics.
    """
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, input: Union[Dialog, List[Dialog]]) -> Union[dict, float]:
        raise NotImplementedError("Subclasses should implement this method.")


class BaseLLMJudge(ABC):
    """
    Base class for LLM judges.
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 output_format: Union[dict, BaseModel] = None,
                 llm_kwargs: dict = {}):
        if model is None:
            model = config["llm"]["model"]

        # Collect LLM parameters from config, only if not None
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}

        if not output_format or type(output_format) is dict:
            self.output_format = None
        else:
            self.output_format = output_format

        if isinstance(model, str):
            self.llm = get_llm_model(model_name=model,
                                     output_format=self.output_format,
                                     llm_kwargs=llm_kwargs)
        else:
            self.llm = model
            if output_format:
                self.llm.format = self.output_format.model_json_schema()

        with open(config["prompts"]["evaluation"]["llm_as_judge"], encoding="utf-8") as f:
            self.messages = [SystemMessage(f.read()), HumanMessage("")]

    def __call__(self, prompt: str) -> Union[dict, BaseModel]:
        self.messages[1].content = prompt
        return self.llm.invoke(self.messages).content

    @abstractmethod
    def judge(self, input: Union[Dialog, List[Dialog]]) -> dict:
        """
        Judge the dialogs using the LLM.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class LLMJudgeYesNo(BaseLLMJudge):
    """LLM judge for classifying a dialogue as "yes or no" (boolean) output and feedback."""
    def __init__(self, prompt_template: str, model: Union[BaseLanguageModel, str] = None, llm_kwargs: dict = {}):
        super().__init__(output_format=LLMJudgeYesNoOutput, model=model, llm_kwargs=llm_kwargs)

        self.prompt_template = Template(prompt_template)

    def judge(self, input: Union[Dialog, List[Dialog]]) -> LLMJudgeYesNoOutput:
        if isinstance(input, list):
            dialog = input[0]  # Only support single dialog for now
        else:
            dialog = input
        prompt = self.prompt_template.render(dialog=dialog)
        output = super().__call__(prompt)
        return self.output_format.model_validate(json.loads(output))

    __call__ = judge  # Allow direct call to judge method


class LLMJudgeRealOrSynthetic(LLMJudgeYesNo):
    """
    LLM judge for classifying a dialogue as real (human) or synthetic (machine-generated), with boolean output and feedback.
    Returns an instance of LLMJudgeYesNoOutput.
    """  # noqa: E501
    def __init__(self, model: Union[BaseLanguageModel, str] = None, llm_kwargs: dict = {}):
        with open(config["prompts"]["evaluation"]["llm_as_judge_real_or_not"], encoding="utf-8") as f:
            prompt_template_real_or_not = f.read()
        super().__init__(prompt_template_real_or_not,
                         model=model,
                         llm_kwargs=llm_kwargs)


class LLMJudgePersonaAttributes(LLMJudgeYesNo):
    def __init__(self, attributes: dict, model: Union[BaseLanguageModel, str] = None, llm_kwargs: dict = {}):
        with open(config["prompts"]["evaluation"]["llm_as_judge_persona_attributes"], encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_template = prompt_template.render(attributes=attributes)

        super().__init__(prompt_template,
                         model=model,
                         llm_kwargs=llm_kwargs)

# class Comparator(ABC):
#     def __init__(self, metrics: List[BaseMetric]):
#         self.metrics = metrics

#     @abstractmethod
#     def compare(self, dialogs: Union[Dialog, List[Dialog]]) -> Union[dict, float]:
#         raise NotImplementedError("Subclasses should implement this method.")


class SimilarityScore(BaseMetric, ABC):
    def compute(self, dialog_a: Dialog, dialog_b: Dialog) -> float:
        """
        Compute the similarity score between two dialogs.

        :param dialog_a: The first dialog.
        :param dialog_b: The second dialog.
        :return: A float representing the similarity score.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class SentenceTransformerSimilarity(SimilarityScore):
    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        """
        Initialize the SentenceEmbeddingSimilarity with a model name.

        :param model_name: The name of the sentence embedding model to use.
        """
        self.model_name = model_name

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    @abstractmethod
    def compute(self, dialog_a: Dialog, dialog_b: Dialog) -> float:
        """
        Compute the similarity score between two dialogs using sentence embeddings.
        """
        embs = self.model.encode([dialog_a, dialog_b])
        return self.model.similarity(embs[0], embs[1])


# class BGEM3EmbeddingMetric(StringDistance):
#     """
#     StringDistance implementation using BGE-M3 embeddings.
#     """
#     model = None
#     model_name = None

#     def __init__(self, model_name="BAAI/bge-m3", mode: str = "dense", modes_weight: list[int] = [0.33, 0.33, 0.33]):
#         """
#         Args:
#             model_name (str): Model name.
#             mode (str): Embedding mode ("dense", "sparse", "colbert", "sparse+dense", or "colbert+sparse+dense").
#             modes_weight (list): Weights for each mode.
#         """
#         self.model = BGEM3FlagModel(model_name,  use_fp16=True)
#         self.mode = mode
#         self.model_name = model_name
#         self.modes_weight = modes_weight

#     def distance(self, sent1, sent2):
#         """
#         Compute distance between two sentences using BGE-M3 embeddings.
#         """
#         global bge_emb_model

#         bge_emb_model = self.model
#         scores = bge_compute_score(sent1, sent2, self.model_name, self.modes_weight)

#         # TODO: in combined mode score can be greater than 1, perhaps is better to use:
#         # return -scores[self.mode]
#         return 1 - scores[self.mode]
