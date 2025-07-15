"""
evaluation: Evaluation components for dialogue generation and analysis.

This module provides abstract base classes for evaluating dialogues,
including LLM judges, metrics, and similarity scores.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json
import numpy as np

from jinja2 import Template
from typing import Optional
from tqdm.auto import tqdm
from typing import Union, List
from pydantic import BaseModel
from math import exp, log, sqrt
from abc import ABC, abstractmethod
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from .. import Dialog
from ..config import config
from .dialog2flow import dialog2graph, DEFAULT_TOKEN_START
from ..util import KNNModel, softmax, get_llm_model


def cs_divergence(p1, p2, resolution=100, bw_method=1):
    """
    Calculates the Cauchy-Schwarz divergence between two probability distributions.

    :param p1: First sample (1D array or list)
    :type p1: array-like
    :param p2: Second sample (1D array or list)
    :type p2: array-like
    :param resolution: Number of points to evaluate the KDEs on (default: 100)
    :type resolution: int
    :param bw_method: Bandwidth for KDE (default: 1, i.e., standard bandwidth)
    :type bw_method: float or str
    :return: Cauchy-Schwarz divergence (0 means identical distributions)
    :rtype: float
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    r = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), resolution)
    p1_kernel = gaussian_kde(p1, bw_method=bw_method)
    p2_kernel = gaussian_kde(p2, bw_method=bw_method)
    p1_vals = p1_kernel(r)
    p2_vals = p2_kernel(r)
    numerator = np.sum(p1_vals * p2_vals)
    denominator = sqrt(np.sum(p1_vals ** 2) * np.sum(p2_vals ** 2))
    return -log(numerator / denominator)


def kl_divergence(p1, p2, resolution=100, bw_method=1e-1):
    """
    Estimates the Kullback-Leibler (KL) divergence KL(p1 || p2) between two distributions given samples, using KDE.

    KL divergence is not symmetric: KL(p1 || p2) != KL(p2 || p1).
    The result is >= 0, and 0 means the distributions are identical.

    :param p1: First sample (1D array or list) (the 'true' distribution)
    :type p1: array-like
    :param p2: Second sample (1D array or list) (the 'approximate' distribution)
    :type p2: array-like
    :param resolution: Number of points to evaluate the KDEs on (default: 100)
    :type resolution: int
    :param bw_method: Bandwidth for KDE (default: 0.1)
    :type bw_method: float or str
    :return: KL divergence KL(p1 || p2)
    :rtype: float
    """
    r = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), resolution)
    p1_kernel = gaussian_kde(p1, bw_method=bw_method)
    p2_kernel = gaussian_kde(p2, bw_method=bw_method)
    p1_vals = p1_kernel(r)
    p2_vals = p2_kernel(r)
    # Avoid division by zero and log(0) by adding a small epsilon
    eps = 1e-12
    p1_vals = np.clip(p1_vals, eps, None)
    p2_vals = np.clip(p2_vals, eps, None)

    return float(np.sum(p1_vals * np.log(p1_vals / p2_vals)) / np.sum(p1_vals))


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


class BaseDatasetMetric(ABC):
    """
    Base class for metrics.
    """
    @abstractmethod
    def __call__(self, dialogues: Union[str, List[Dialog]]) -> Union[dict, float]:
        """
        Call the score method to compute the metric.
        This allows the metric to be used as a callable.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class DatasetComparator(ABC):
    def __init__(self, metrics: List[BaseDatasetMetric]):
        self.metrics = metrics

    @abstractmethod
    def compare(
        self,
        reference: Union[str, List[Dialog]],
        candidates: Union[str, List[str], List[Dialog], List[List[Dialog]]],
        **kwargs
    ) -> dict:
        """
        Compare one reference dataset against one or more candidate datasets.

        :param reference: The reference dataset (either path containing the dialogues or list of Dialogs).
        :param candidates: The candidate datasets (either path containing the dialogues or list of Dialogs).
        :param kwargs: Additional keyword arguments for the comparison.
        :return: A dictionary containing the comparison results.
        """
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


class FlowDistanceMetric(BaseDatasetMetric):
    def __init__(self, reference_dialogues: Union[str, List[Dialog]], k_neighbors=64, verbose=False, **d2f_kwargs):
        d2f_kwargs = {"node_llm_labels_enabled": False,
                      "out_png": False,
                      "verbose": verbose,
                      **d2f_kwargs}
        self.verbose = verbose
        self.graph, self.nodes = dialog2graph(reference_dialogues, **d2f_kwargs)
        self.speakers = self.nodes["_metadata"]["speakers"]
        self.encoder = SentenceTransformer(self.nodes["_metadata"]["model"])
        self.knn_models = {
            "user": KNNModel([(node_id.lower(), info["centroid-embedding"])
                              for node_id, info in self.nodes.items() if node_id[0].lower() == "u"],
                             k=k_neighbors),
            "system": KNNModel([(node_id.lower(), info["centroid-embedding"])
                                for node_id, info in self.nodes.items() if node_id[0].lower() == "s"],
                               k=k_neighbors)
        }
        self.ref_scores = np.array([self.score_dialog(dialogue)
                                    for dialogue in tqdm(reference_dialogues,
                                                         desc="Computing reference scores",
                                                         leave=verbose)])

    def get_reference_scores(self) -> np.ndarray:
        return self.ref_scores

    def score_dialog(self, dialog, use_softmax=True, only_system=False):
        sum_log_p = 0
        sys_turns = 0
        prev_node = DEFAULT_TOKEN_START
        for turn in dialog.turns:
            speaker = turn.speaker.lower()
            if speaker in self.speakers:
                speaker = self.speakers[speaker]
            else:
                raise ValueError(f"WARNING: speaker '{turn.speaker}' not found in the graph metadata, expected one of "
                                 f"{list(self.speakers.keys())}")
            utt_emb = self.encoder.encode(turn.text, show_progress_bar=False)
            neighbors = self.knn_models[speaker](utt_emb, k=None if use_softmax else 1)
            nearest_id, _ = neighbors[0]
            prob_correct_node = softmax([1 - dist for _, dist in neighbors])[0] if use_softmax else 1

            prob_next_node = self.graph.get_edge_data(prev_node, nearest_id)
            if (not only_system or speaker == "system") and prob_next_node is not None:
                sum_log_p += log(prob_next_node["weight"] * prob_correct_node)
                sys_turns += 1
            prev_node = nearest_id

        return exp(-sum_log_p / sys_turns)

    def __call__(self,
                 dialogues: Union[str, List[Dialog]],
                 metric="all",
                 kde_bw=None,
                 return_dialog_scores=False) -> Union[dict, float]:
        scores = np.array([self.score_dialog(dialogue)
                           for dialogue in tqdm(dialogues, desc="Computing scores", leave=self.verbose)])

        if metric == "kl":
            result = kl_divergence(self.ref_scores, scores, bw_method=kde_bw)
        elif metric == "cs":
            result = cs_divergence(self.ref_scores, scores, bw_method=kde_bw)
        else:
            result = {
                "cs": cs_divergence(self.ref_scores, scores, bw_method=kde_bw),
                "kl": kl_divergence(self.ref_scores, scores, bw_method=kde_bw)
            }

        return (result, scores) if return_dialog_scores else result
