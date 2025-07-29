"""
evaluation: Evaluation components for dialogue generation and analysis.

This module provides abstract base classes for evaluating dialogues,
including LLM judges, metrics, and similarity scores.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import logging
import syllables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import linalg
from tqdm.auto import tqdm
from scipy.stats import norm
from math import exp, log, sqrt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from pydantic import BaseModel, Field
from typing import Optional, Annotated
from sklearn.cluster import MiniBatchKMeans
from typing import Union, List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from langchain_core.language_models.base import BaseLanguageModel

from .. import Dialog
from ..config import config
from ..personas import BasePersona
from ..util import SentencePairTransformer
from ..util import dict_to_table, upper_camel_to_dash, dialogs_to_utt_pairs
from .base import BaseDatasetEvaluator, BaseDatasetScoreEvaluator, BaseDatasetEmbeddingEvaluator
from .base import CacheDialogScore, BaseLLMJudge, BaseDialogEmbedder, BaseDialogScore, BaseDialogFlowScore

logger = logging.getLogger(__name__)


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
    if len(p1) == 0 or len(p2) == 0:
        logger.error("Both input distributions must have at least one sample. Returning None")
        return None
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
    if len(p1) == 0 or len(p2) == 0:
        logger.error("Both input distributions must have at least one sample. Returning None")
        return None
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
    yes: Union[bool, List[bool]]
    feedback: Optional[Union[str, List[str]]] = None


class DialogFlowPPL(BaseDialogFlowScore):
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 k_neighbors: int = 64,
                 use_softmax: bool = True,
                 use_only_known_edges: bool = True,
                 name: str = None,
                 verbose: bool = False,
                 **d2f_kwargs):
        self.use_only_known_edges = use_only_known_edges
        super().__init__(
            reference_dialogues,
            ai_speaker=ai_speaker,
            k_neighbors=k_neighbors,
            use_softmax=use_softmax,
            name=name if name else "dfppl" + ("" if use_softmax else "-hard") + ("-ai" if ai_speaker else ""),
            verbose=verbose,
            **d2f_kwargs
        )

    @CacheDialogScore.cache
    def score(self, dialog: Dialog) -> float:
        sum_log_p_known, n_turns_known, sum_log_p, n_turns = self.compute_dialog_log_likelihood(dialog)
        if n_turns <= 1:
            dialog_path = getattr(dialog, "_path", None)
            if dialog_path:
                logger.warning(f"Dialog at '{dialog_path}' has no known transitions or valid turns. Skipping.")
            else:
                logger.warning(f"Dialog (id={getattr(dialog, 'id', 'unknown')}) has no known transitions "
                               "or valid turns. Skipping.")
            return None
        if self.use_only_known_edges:
            return exp(-sum_log_p_known / n_turns_known)
        else:
            return exp(-sum_log_p / n_turns)


class DialogFlowScore(BaseDialogFlowScore):
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 k_neighbors: int = 64,
                 use_softmax: bool = True,
                 use_only_known_edges: bool = True,
                 name: str = None,
                 verbose: bool = False,
                 **d2f_kwargs):
        self.use_only_known_edges = use_only_known_edges
        if name is None:
            name = "dfs" + ("" if use_softmax else "-hard") + ("-ai" if ai_speaker else "")
            name += "-known" if use_only_known_edges else "-all"
        super().__init__(
            reference_dialogues,
            ai_speaker=ai_speaker,
            k_neighbors=k_neighbors,
            use_softmax=use_softmax,
            name=name,
            verbose=verbose,
            **d2f_kwargs
        )

    @CacheDialogScore.cache
    def score(self, dialog: Dialog) -> float:
        sum_log_p_known, n_turns_known, sum_log_p, n_turns = self.compute_dialog_log_likelihood(dialog)
        if n_turns <= 1:
            dialog_path = getattr(dialog, '_path', None)
            if dialog_path:
                logger.warning(f"Dialog at '{dialog_path}' has no known transitions or valid turns. Skipping.")
            else:
                logger.warning(f"Dialog (id={getattr(dialog, 'id', 'unknown')}) has no known transitions "
                               "or valid turns. Skipping.")
            return None
        if self.use_only_known_edges:
            return pow(exp(sum_log_p_known), 1 / n_turns_known)
        else:
            return pow(exp(sum_log_p), 1 / n_turns)


class LLMJudgeYesNo(BaseDialogScore, BaseLLMJudge):
    """LLM judge for classifying a dialogue as "yes or no" (boolean) output and feedback."""
    def __init__(self,
                 prompt_template: str,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 **llm_kwargs):
        BaseDialogScore.__init__(self,
                                 name=upper_camel_to_dash(self.__class__.__name__))
        BaseLLMJudge.__init__(self,
                              model=model,
                              output_format=LLMJudgeYesNoOutput,
                              prompt_template=prompt_template,
                              **llm_kwargs)

        self.feedback = feedback

    def judge(self, dialogs: Union[Dialog, List[Dialog]], feedback: bool = None) -> Union[LLMJudgeYesNoOutput, int]:
        if isinstance(dialogs, Dialog):
            dialogs = [dialogs]  # Wrap single dialog in a list

        prompt = self.prompt_template.render(dialogs=dialogs,
                                             dialog=dialogs[0],
                                             feedback=feedback if feedback is not None else self.feedback)
        output = self.output_format.model_validate(BaseLLMJudge.__call__(self, prompt))

        return output

    @CacheDialogScore.cache
    def score(self, dialog: Dialog) -> int:
        """
        Computes the score for the provided dialog, 1 if dialogues is judged as real, 0 otherwise.

        :param dialog: The dialog to score.
        :return: An int representing the score of the dialog.
        """
        output = self.judge(dialog)
        try:
            return int(output.yes[0]) if isinstance(output.yes, list) else int(output.yes)
        except TypeError:
            raise ValueError(f"LLMJudgeYesNo output '{output.yes}' is not a boolean or list of booleans, "
                             f"cannot convert to integer score.")


class LLMJudgeScore(BaseDialogScore, BaseLLMJudge):
    """LLM judge for scoring a dialogue with a numerical score and optional feedback."""
    def __init__(self,
                 prompt_template: str,
                 model: Union[BaseLanguageModel, str] = None,
                 min_score: float = 1,
                 max_score: float = 5,
                 score_type: type = int,
                 feedback: bool = False,
                 **llm_kwargs):

        if score_type not in [int, float]:
            raise ValueError(f"Invalid score_type: {score_type}. Must be int or float.")
        elif score_type is float:
            logger.warning(
                "Using float as `score_type` may cause boundary issues (min_score, max_score). "
                "Consider using int for discrete scales."
            )

        class LLMJudgeScoreOutput(BaseModel):
            score: Annotated[
                score_type,
                Field(ge=min_score, le=max_score)
            ]
            feedback: Optional[str] = None

        BaseDialogScore.__init__(self,
                                 name=upper_camel_to_dash(self.__class__.__name__))
        BaseLLMJudge.__init__(self,
                              model=model,
                              output_format=LLMJudgeScoreOutput,
                              prompt_template=prompt_template,
                              **llm_kwargs)

        self.score_type = score_type
        self.min_score = min_score
        self.max_score = max_score
        self.feedback = feedback

    def judge(self,
              dialogs: Union[Dialog, List[Dialog]],
              feedback: bool = None) -> Union[LLMJudgeYesNoOutput, int, float]:
        if isinstance(dialogs, Dialog):
            dialogs = [dialogs]  # Wrap single dialog in a list

        prompt = self.prompt_template.render(dialogs=dialogs,
                                             dialog=dialogs[0],
                                             min_score=self.min_score,
                                             max_score=self.max_score,
                                             feedback=feedback if feedback is not None else self.feedback)
        output = self.output_format.model_validate(BaseLLMJudge.__call__(self, prompt))

        return output

    @CacheDialogScore.cache
    def score(self, dialog: Dialog) -> Union[float, int]:
        """
        Computes the score for the provided dialog.

        :param dialog: The dialog to score.
        :return: A float representing the score of the dialog.
        """
        output = self.judge(dialog)
        try:
            score = output.score[0] if isinstance(output.score, list) else output.score
            # Clamp score to [min_score, max_score] if out of bounds
            if score < self.min_score or score > self.max_score:
                old_score = score
                score = max(self.score_min, min(score, self.max_score))
                logger.warning(
                    f"Generated score {old_score} is out of bounds [{self.score_min}, {self.max_score}]. "
                    f"Clamping to valid range: {score}."
                )
            return score
        except TypeError:
            raise ValueError(f"LLMJudgeScore output ({output.score}) is not a {self.score_type} or list of booleans, "
                             "cannot convert to integer score.")


class LLMJudgeRealDialog(LLMJudgeYesNo):
    """
    LLM judge for classifying a dialogue as real (human) or synthetic (machine-generated), with boolean output and feedback.
    Returns an instance of LLMJudgeYesNoOutput.
    """  # noqa: E501
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_real_dialog"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         feedback=feedback,
                         **llm_kwargs)


class LLMJudgeRealDialogLikertScore(LLMJudgeScore):
    """
    LLM judge for evaluating whether a dialogue appears real (human) or synthetic (machine-generated),
    providing a Likert score between 1 (definitely synthetic) and 5 (definitely real), with optional feedback.
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_real_dialog_likert_score"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         score_type=int,
                         score_min=1,
                         max_score=5,
                         feedback=feedback,
                         **llm_kwargs)


class LLMJudgeRealDialogScore(LLMJudgeScore):
    """
    LLM judge for evaluating how "real" (human-like) or "synthetic" (machine-generated) a dialogue appears,
    returning a numerical score (e.g., Likert scale or custom range) and optional feedback.
    Useful for fine-grained assessment of dialogue authenticity.
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 min_score: int = 0,
                 max_score: int = 10,
                 feedback: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_real_dialog_score"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         score_type=int,
                         min_score=min_score,
                         max_score=max_score,
                         feedback=feedback,
                         **llm_kwargs)


class LLMJudgeRefusal(LLMJudgeYesNo):
    """
    LLM judge for evaluating if a dialogue contains a refusal response.
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_refusal"], encoding="utf-8") as f:
            prompt_template = f.read()
        super().__init__(prompt_template,
                         model=model,
                         feedback=feedback,
                         **llm_kwargs)


class LLMJudgePersonaAttributes(LLMJudgeYesNo):
    """LLM judge for evaluating if a speaker follows the persona attributes in a dialogue."""
    def __init__(self,
                 persona: BasePersona,
                 speaker: str,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_persona_attributes"], encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_template = prompt_template.render(persona=persona, speaker=speaker)

        super().__init__(prompt_template,
                         model=model,
                         feedback=feedback,
                         **llm_kwargs)


class SentenceTransformerDialogEmbedder(BaseDialogEmbedder):
    """
    Dialog embedder using SentenceTransformer.
    Can embed a dialog as the mean of turn embeddings or as a single embedding of the whole dialog text.
    """
    def __init__(self,
                 model_name: str = "sentence-transformers/LaBSE",
                 mean: bool = True,
                 name: str = None,
                 verbose: bool = False):
        """
        :param model_name: SentenceTransformer model name.
        :param mean: If True, embed as mean of turn embeddings; if False, embed whole dialog as a single string.
        :param name: Optional name for the embedder.
        """
        mode_str = "mean-" if mean else ""
        super().__init__(name=name or f"{mode_str}{model_name.split('/')[-1]}")
        self.model = SentenceTransformer(model_name)
        self.mean = mean
        self.verbose = verbose

    def embed(self, dialog: Dialog) -> np.ndarray:
        if self.mean:
            texts = [turn.text for turn in dialog.turns if hasattr(turn, "text")]
            if not texts:
                return np.zeros(self.model.get_sentence_embedding_dimension())
            embs = self.model.encode(texts, show_progress_bar=self.verbose)
            return np.mean(embs, axis=0)
        else:
            dialog_text = "\n".join([turn.text for turn in dialog.turns if hasattr(turn, "text")])
            if not dialog_text:
                return np.zeros(self.model.get_sentence_embedding_dimension())
            emb = self.model.encode([dialog_text], show_progress_bar=self.verbose)[0]
            return emb


class ReferenceCentroidEmbeddingEvaluator(BaseDatasetEmbeddingEvaluator):
    """
    Evaluator that computes the centroid of reference dialog embeddings and compares
    the centroid of candidate dialog embeddings using cosine similarity.
    """
    def __init__(self,
                 dialog_embedder: BaseDialogEmbedder,
                 reference_dialogues: Union[str, List[Dialog]],
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False):
        name = name or f"centroid-similarity-{dialog_embedder.name}"
        super().__init__(dialog_embedder, name=name, enable_plotting=enable_plotting, verbose=verbose)
        # Compute reference centroid
        if isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        reference_embs = np.array([self.dialog_embedder(dialog)
                                   for dialog in tqdm(reference_dialogues,
                                                      desc="Computing reference embeddings",
                                                      leave=verbose)])
        self.reference_embs = reference_embs if enable_plotting else None
        self.reference_centroid = np.mean(reference_embs, axis=0)

    def __plot__(self, dialog_embs: Dict[str, np.ndarray]):
        """
        Plot the embeddings of the datasets.
        :param dialog_embs: A dictionary with dataset names as keys and embeddings as values.
        :param tsne_model: The t-SNE model used for dimensionality reduction.
        """
        # Concatenate all embeddings and keep track of dataset labels
        all_embs = [self.reference_centroid.reshape(1, -1)]
        all_labels = ["centroid-reference"]
        all_embs.append(self.reference_embs)
        all_labels.extend(["reference"] * len(self.reference_embs))
        for dataset_name, embs in dialog_embs.items():
            all_embs.append(embs)
            all_labels.extend([dataset_name] * len(embs))
            all_embs.append(np.mean(embs, axis=0).reshape(1, -1))
            all_labels.append("centroid-" + dataset_name)
        all_embs = np.vstack(all_embs)
        all_labels = np.array(all_labels)

        # Compute t-SNE (2D)
        logger.info("Computing t-SNE for embeddings...")
        tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30, metric="cosine")
        tsne_embs = tsne.fit_transform(all_embs)

        # Plot
        unique_labels = [label for label in np.unique(all_labels).tolist() if "centroid-" not in label]
        colors = plt.cm.tab10.colors if len(unique_labels) <= 10 else plt.cm.tab20.colors
        for i, label in enumerate(unique_labels):
            idx = all_labels == label
            plt.scatter(tsne_embs[idx, 0], tsne_embs[idx, 1],
                        label=label if label != "reference" else None,
                        alpha=0.15 if label == "reference" else 0.7,
                        color="black" if label == "reference" else colors[i % len(colors)])
        for label in ["reference"] + list(dialog_embs.keys()):
            idx = all_labels == f"centroid-{label}"
            plt.scatter(tsne_embs[idx, 0], tsne_embs[idx, 1],
                        label="Reference centroid" if label == "reference" else None,
                        linewidths=3 if label == "reference" else 2,
                        alpha=1,  # if label == "reference" else 0.7,
                        color="black" if label == "reference" else colors[unique_labels.index(label) % len(colors)],
                        s=100,
                        marker="x")

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"Dialog embeddings ({self.dialog_embedder.name}) with centroids")
        plt.legend()

    def eval(self, dialog_embs: List[np.ndarray]) -> float:
        """
        Compute the centroid of the given embeddings and return the cosine similarity
        with the reference centroid.
        """
        if isinstance(dialog_embs, list):
            dialog_embs = np.array(dialog_embs)
        if dialog_embs.ndim == 1:
            dialog_embs = dialog_embs.reshape(1, -1)
        centroid = np.mean(dialog_embs, axis=0)
        # Cosine similarity
        dot = np.dot(self.reference_centroid, centroid)
        norm_ref = np.linalg.norm(self.reference_centroid)
        norm_cand = np.linalg.norm(centroid)
        if norm_ref == 0 or norm_cand == 0:
            return 0.0
        return float(dot / (norm_ref * norm_cand))


class KDEDistanceEvaluator(BaseDatasetScoreEvaluator):
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 reference_dialogues: Union[str, List[Dialog]] = None,
                 metric: str = "all",
                 kde_bw: float = None,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 **evaluator_kwargs):
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose, **evaluator_kwargs)

        if reference_dialogues is None:
            if hasattr(dialog_score, "reference_dialogues"):
                reference_dialogues = dialog_score.reference_dialogues
            else:
                raise ValueError("Reference dialogues must be provided or "
                                 "the dialog_score must have a reference_dialogues attribute.")
        elif isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        elif not isinstance(reference_dialogues, list):
            raise ValueError("Reference dialogues must be provided as a list of Dialog objects or a file path.")

        if metric != "all":
            self.name += f"-{metric}"
        self.metric = metric
        self.kde_bw = kde_bw
        self.reference_scores = [self.dialog_score(dialogue)
                                 for dialogue in tqdm(reference_dialogues,
                                                      desc=f"Computing reference {self.name} scores",
                                                      leave=verbose)]
        self.reference_scores = np.array([s for s in self.reference_scores if s is not None])

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        if "reference" not in dialog_scores and self.reference_scores is not None:
            pd.Series(self.reference_scores, name="Reference").plot.kde(bw_method=self.kde_bw, lw=3, color="grey")
        for dataset_name, scores in dialog_scores.items():
            try:
                pd.Series(scores, name=dataset_name).plot.kde(bw_method=self.kde_bw)
            except ValueError as e:
                logger.error(f"Error plotting KDE for {dataset_name}: {e}")
        plot.xlabel(self.dialog_score.name)
        plot.legend()
        plot.title(f"KDE of {self.dialog_score.name} distributions")

    def eval(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        if self.metric == "kl":
            result = kl_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw)
        elif self.metric == "cs":
            result = cs_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw)
        else:
            result = {
                "cs": cs_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw),
                "kl": kl_divergence(self.reference_scores, dialog_scores, bw_method=self.kde_bw)
            }
        return result


class FrechetDistanceEvaluator(BaseDatasetScoreEvaluator):
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 reference_dialogues: Union[str, List[Dialog]] = None,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False,
                 **evaluator_kwargs):
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose, **evaluator_kwargs)

        if reference_dialogues is None:
            if hasattr(dialog_score, "reference_dialogues"):
                reference_dialogues = dialog_score.reference_dialogues
            else:
                raise ValueError("Reference dialogues must be provided or "
                                 "the dialog_score must have a reference_dialogues attribute.")

        reference_scores = np.array([self.dialog_score(dialogue)
                                     for dialogue in tqdm(reference_dialogues,
                                                          desc=f"Computing reference {self.name} scores",
                                                          leave=verbose)])
        self.reference_norm_dist = norm(loc=np.mean(reference_scores), scale=np.std(reference_scores))

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        if "reference" not in dialog_scores and self.reference_norm_dist is not None:
            x = np.linspace(self.reference_norm_dist.ppf(0.001), self.reference_norm_dist.ppf(0.999), 100)
            plot.plot(x, self.reference_norm_dist.pdf(x), color="grey", lw=3, label="Reference")
        for dataset_name, scores in dialog_scores.items():
            x = np.linspace(np.min(scores), np.max(scores), 100)
            plot.plot(x, norm.pdf(x, loc=np.mean(scores), scale=np.std(scores)), label=dataset_name)
        plot.xlabel(self.dialog_score.name)
        plot.legend()
        plot.title(f"Normal Distributions of {self.dialog_score.name}")

    def eval(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        # Compute the Frechet distance between the reference normal distribution and the one from dialog_scores
        if not isinstance(dialog_scores, np.ndarray):
            dialog_scores = np.array(dialog_scores)
        mu_src, sigma_src = self.reference_norm_dist.mean(), self.reference_norm_dist.std()
        mu_tgt, sigma_tgt = np.mean(dialog_scores), np.std(dialog_scores)
        # Frechet distance between two 1D Gaussians: sqrt((mu_src-mu_tgt)^2 + (sigma_src-sigma_tgt)^2)
        return np.sqrt((mu_src - mu_tgt) ** 2 + (sigma_src - sigma_tgt) ** 2)


class FrechetBERTDistanceEvaluator(BaseDatasetEvaluator):
    """
    Frechet distance evaluator based on BERT embeddings.
    See: https://aclanthology.org/2021.findings-acl.193/ where it was introduced
    """
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 name: str = None,
                 model_name: str = "roberta-base",
                 batch_size: int = 128,
                 device: str = None,
                 enable_plotting: bool = False,
                 verbose: bool = False):
        self.reference_embs = None
        self.datasets_embs = {}
        self.enable_plotting = enable_plotting
        self.verbose = verbose
        self.ai_speaker = ai_speaker
        self.name = name or "frechet-bert-distance" + ("-ai" if ai_speaker else "")
        self.batch_size = batch_size
        self.model = SentencePairTransformer(model_name=model_name,
                                             device=device,
                                             verbose=verbose)

        if isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        if not reference_dialogues or not isinstance(reference_dialogues, list):
            raise ValueError("Reference dialogues must be provided as a list of Dialog objects or a file path.")

        self.reference_mu, self.reference_sigma = self._get_multidim_gaussian_mu_sigma(reference_dialogues)

    def _get_multidim_gaussian_mu_sigma(self,
                                        dialogs: List[Dialog],
                                        dataset_name: str = "reference") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and covariance matrix of a set of embeddings.

        :param embeddings: A 2D numpy array of shape (n_samples, n_features).
        :param dataset_name: Name of the dataset for logging.
        :return: A tuple (mean, covariance_matrix).
        """
        utts, utts_next = dialogs_to_utt_pairs(dialogs, self.ai_speaker)

        embs = self.model.encode(utts, utts_next,
                                 batch_size=self.batch_size,
                                 progress_bar_desc=f"Computing embeddings for FrechetBERT on {dataset_name}")

        if self.enable_plotting and dataset_name:
            if dataset_name == "reference":
                self.reference_embs = embs
            else:
                self.datasets_embs[dataset_name] = embs

        mu = np.mean(embs, axis=0)
        sigma = np.cov(embs, rowvar=False)
        return mu, sigma

    def __call__(self, dialogues: Union[str, List[Dialog]], dataset_name: str = "candidate") -> float:
        """
        Compute the Frechet distance (a.k.a. Fréchet Inception Distance, FID) between two multivariate Gaussians.
        See: https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance#Application_in_machine_learning

        :param mu_src: Mean vector of the source distribution (np.ndarray, shape [d])
        :param sigma_src: Covariance matrix of the source distribution (np.ndarray, shape [d, d])
        :param mu_tgt: Mean vector of the target distribution (np.ndarray, shape [d])
        :param sigma_tgt: Covariance matrix of the target distribution (np.ndarray, shape [d, d])
        :return: The Frechet distance (float)
        """
        mu_src, sigma_src = np.atleast_1d(self.reference_mu), np.atleast_2d(self.reference_sigma)
        mu_tgt, sigma_tgt = self._get_multidim_gaussian_mu_sigma(dialogues, dataset_name=dataset_name)

        mu_tgt = np.atleast_1d(mu_tgt)
        sigma_tgt = np.atleast_2d(sigma_tgt)

        diff = mu_src - mu_tgt

        covmean, _ = linalg.sqrtm(sigma_src.dot(sigma_tgt), disp=False)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.imag(covmean), 0, atol=1e-6):
                logger.warning("linalg.sqrtm returned complex values; taking real part of result.")
            covmean = np.real(covmean)

        tr_covmean = np.trace(covmean)
        fid = float(diff.dot(diff) + np.trace(sigma_src) + np.trace(sigma_tgt) - 2 * tr_covmean)
        return max(fid, 0.0)

    def plot(self, show: bool = True, save_path: str = None):
        """
        Plot the sentence-pair embeddings of the datasets.
        """
        if not self.enable_plotting or not self.datasets_embs:
            return
        plt.figure(figsize=(8, 5))
        # Concatenate all embeddings and keep track of dataset labels
        all_embs = [self.reference_embs]
        all_labels = ["reference"] * len(self.reference_embs)
        for dataset_name, embs in self.datasets_embs.items():
            all_embs.append(embs)
            all_labels.extend([dataset_name] * len(embs))
        all_embs = np.vstack(all_embs)
        all_labels = np.array(all_labels)

        # Compute t-SNE (2D)
        logger.info("Computing t-SNE for embeddings...")
        tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30, metric="cosine")
        tsne_embs = tsne.fit_transform(all_embs)

        # Plot
        unique_labels = np.unique(all_labels).tolist()
        colors = plt.cm.tab10.colors if len(unique_labels) <= 10 else plt.cm.tab20.colors
        for i, label in enumerate(unique_labels):
            idx = all_labels == label
            plt.scatter(tsne_embs[idx, 0], tsne_embs[idx, 1],
                        label=label if label != "reference" else "Reference",
                        alpha=0.15 if label == "reference" else 0.7,
                        color="black" if label == "reference" else colors[i % len(colors)])

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"Sentence-pair embeddings for {self.name}")
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()


class PrecisionRecallDistanceEvaluator(BaseDatasetEvaluator):
    """
    Precision-Recall distance evaluator based on BERT embeddings.
    See: https://aclanthology.org/2021.findings-acl.193/
    """
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 num_clusters=20,
                 num_angles=1001,
                 num_runs=10,
                 name: str = None,
                 model_name: str = "roberta-base",
                 batch_size: int = 128,
                 device: str = None,
                 verbose: bool = False):
        if isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        if not reference_dialogues or not isinstance(reference_dialogues, list):
            raise ValueError("Reference dialogues must be provided as a list of Dialog objects or a file path.")

        self.name = name or f"pr-distance-{model_name.split('/')[-1]}" + ("-ai" if ai_speaker else "")
        self.verbose = verbose
        self.ai_speaker = ai_speaker
        self.num_clusters = num_clusters
        self.num_angles = num_angles
        self.num_runs = num_runs
        self.batch_size = batch_size
        self.model = SentencePairTransformer(model_name=model_name,
                                             device=device,
                                             verbose=verbose)
        self.reference_embs = self._encode_utterance_pairs(reference_dialogues)

    def _encode_utterance_pairs(self, dialogues: List[Dialog], dataset_name: str = "reference") -> np.ndarray:
        """
        Encode utterance pairs from dialogues using the model.

        :param dialogues: List of Dialog objects.
        :param dataset_name: Name of the dataset for logging.
        :return: Encoded utterance pairs as a 2D numpy array.
        """
        return self.model.encode(*dialogs_to_utt_pairs(dialogues, self.ai_speaker),
                                 batch_size=self.batch_size,
                                 progress_bar_desc=f"Computing embeddings for PRD on {dataset_name}")

    def _cluster_histograms(self, target_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cluster_data = np.vstack([target_embs, self.reference_embs])
        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, n_init=10)
        labels = kmeans.fit(cluster_data).labels_

        reference_labels = labels[len(target_embs):]
        target_labels = labels[:len(target_embs)]

        reference_histogram = np.histogram(reference_labels, bins=self.num_clusters,
                                           range=[0, self.num_clusters], density=True)[0]
        target_histogram = np.histogram(target_labels, bins=self.num_clusters,
                                        range=[0, self.num_clusters], density=True)[0]
        return reference_histogram, target_histogram

    def _precision_recall_distance(self,
                                   reference_histogram: np.ndarray,
                                   target_histogram: np.ndarray,
                                   epsilon=1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Precision-Recall Distributions (PRD) curve between two histograms, as described in
        "Improved Precision and Recall Metric for Assessing Generative Models" (arXiv:1904.06991).
        The PRD curve is calculated for a set of num_angles values, linearly spaced between [0, pi/2],
        providing a trade-off between precision and recall for the two distributions.
        """
        if not (epsilon > 0 and epsilon < 0.1):
            raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
        if not (self.num_angles >= 3 and self.num_angles <= 1e6):
            raise ValueError('num_angles must be in [3, 1e6] but is %d.' % self.num_angles)

        angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=self.num_angles)
        slopes = np.tan(angles)

        slopes_2d = np.expand_dims(slopes, 1)

        ref_dist_2d = np.expand_dims(reference_histogram, 0)
        eval_dist_2d = np.expand_dims(target_histogram, 0)

        precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
        recall = precision / slopes

        return np.clip(precision, 0, 1), np.clip(recall, 0, 1)

    def __call__(self, dialogues: Union[str, List[Dialog]], dataset_name: str = None) -> Union[dict, float]:
        target_embs = self._encode_utterance_pairs(dialogues, dataset_name)

        if len(target_embs) != len(self.reference_embs):
            logger.warning("The total number of utterance pairs in the reference dialogues "
                           f"({len(self.reference_embs)}) and those of the evaluation dialogues "
                           f"({len(target_embs)}) are not equal. "
                           "This may lead to misleading results since unbalanced distributions bias "
                           "the clustering towards the larger dataset.")

            precisions = []
            recalls = []
            for _ in range(self.num_runs):
                reference_histogram, target_histogram = self._cluster_histograms(target_embs)
                precision, recall = self._precision_recall_distance(reference_histogram, target_histogram)
                precisions.append(precision)
                recalls.append(recall)
            precision = np.mean(precisions, axis=0).tolist()
            recall = np.mean(recalls, axis=0).tolist()
            return max([2 * p * r / (p + r) if (p + r) > 0 else 0
                        for p, r in zip(precision, recall)])  # max F1 score


class StatsEvaluator(BaseDatasetScoreEvaluator):
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False):
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose)

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        # Plot box plots for each dataset
        plot.title(f"Boxplot of {self.dialog_score.name} scores")
        plot.boxplot(list(dialog_scores.values()),
                     labels=list(dialog_scores.keys()))
        plot.xlabel("datasets")
        plot.ylabel(self.dialog_score.name)

    def eval(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        return {
            "mean": np.mean(dialog_scores),
            "std": np.std(dialog_scores),
            "min": np.min(dialog_scores),
            "max": np.max(dialog_scores),
            "median": np.median(dialog_scores)
        }


class FrequencyEvaluator(BaseDatasetScoreEvaluator):
    """
    Evaluator for computing the frequency or percentage of dialogues matching a condition (e.g., refusal responses).
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False):
        super().__init__(dialog_score, name=name, enable_plotting=enable_plotting, verbose=verbose)

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        # Bar plot for frequency/percentage
        percentages = {k: np.mean(v) * 100 for k, v in dialog_scores.items()}
        bars = plot.bar(percentages.keys(), percentages.values(), color=plt.cm.tab10.colors[:len(percentages)])
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plot.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%", ha='center', va='bottom')
        plot.ylabel(f"Percentage of {self.dialog_score.name} (%)")
        plot.xlabel("datasets")
        plot.title(f"Percentage of {self.dialog_score.name} per dataset")

    def eval(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        # Assumes dialog_scores are binary (0/1 or True/False)
        total = len(dialog_scores)
        count = np.sum(dialog_scores)
        percentage = count / total if total > 0 else 0
        return percentage


class LinguisticFeaturesDatasetEvaluator(BaseDatasetScoreEvaluator):
    def __init__(self, features=None, name="linguistic_features", enable_plotting: bool = True):
        super().__init__(dialog_score=None, name=name, enable_plotting=enable_plotting)
        self.name = name
        self.features = features or [
            "mean_turn_length", "hesitation_rate", "gunning_fog", "flesch_reading_ease"
        ]
        self.all_results = []

    @staticmethod
    def clean_utterance(text):
        cleaned = re.sub(r'<[^>]*>', '', text)
        cleaned = re.sub(r'\*[^*]*\*', '', cleaned)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    @staticmethod
    def count_syllables(word):
        return max(1, syllables.estimate(word))

    @staticmethod
    def count_complex_words(text):
        words = text.split()
        return sum(1 for word in words if syllables.estimate(word) >= 3), len(words)

    @staticmethod
    def calculate_gunning_fog(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0
        complex_words, total_words = LinguisticFeaturesDatasetEvaluator.count_complex_words(text)
        avg_sentence_length = len(words) / len(sentences)
        complex_word_ratio = (complex_words / total_words) * 100 if total_words > 0 else 0
        fog_index = 0.4 * (avg_sentence_length + complex_word_ratio)
        return fog_index

    @staticmethod
    def calculate_flesch_reading_ease(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0
        total_syllables = sum(LinguisticFeaturesDatasetEvaluator.count_syllables(word) for word in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return flesch_score

    @staticmethod
    def count_hesitations(text):
        # Exclude the backchannel
        hesitation_patterns = [
            r'\buh+\b',     # uh, uhh, uhhh
            r'\bum+\b',     # um, umm, ummm
            r'\ber+\b',     # er, err, errr
            r'\bahh*\b',    # ah, ahh, ahhh
            r'\bohh*\b',    # oh, ohh, ohhh
            r'\bhmm+\b',    # hmm, hmmm
            r'\bhuh+\b',    # h uh
            r'\bmm+\b',     # mm, mmm
            r'\bmhm+\b',    # mhm, mhmm
            r'\buh\-huh\b',    # uh-huh (backchannel)
            r'\bum-hum+\b',    # um-hum (backchannel)
        ]
        total_hesitations = 0
        text_lower = text.lower()
        for pattern in hesitation_patterns:
            matches = re.findall(pattern, text_lower)
            total_hesitations += len(matches)
        return total_hesitations

    def evaluate(self, dialog, dataset_name=None):
        speaker_stats = {}
        for turn in dialog.turns:
            if not getattr(turn, 'speaker', None) or not getattr(turn, 'text', None):
                continue
            speaker = turn.speaker
            if speaker not in speaker_stats:
                speaker_stats[speaker] = []
            speaker_stats[speaker].append(self.clean_utterance(turn.text))
        results = {"dataset": dataset_name or "unknown"}
        for speaker, utts in speaker_stats.items():
            all_text = " ".join(utts)
            turn_lengths = [len(utt.split()) for utt in utts]
            hesitations = [self.count_hesitations(utt) for utt in utts]
            results[f"{speaker}_mean_turn_length"] = np.mean(turn_lengths)
            # results[f"{speaker}_hesitation_rate"] = sum(hesitations) / max(1, sum(turn_lengths))
            results[f"{speaker}_hesitation_rate"] = (sum(hesitations) / max(1, sum(turn_lengths)) * 100)
            results[f"{speaker}_gunning_fog"] = self.calculate_gunning_fog(all_text)
            results[f"{speaker}_flesch_reading_ease"] = self.calculate_flesch_reading_ease(all_text)
        self.all_results.append(results)
        return results

    def __call__(self, dialogs, dataset_name=None, **kwargs):
        if isinstance(dialogs, list):
            for dialog in dialogs:
                self.evaluate(dialog, dataset_name=dataset_name)
            keys = set(k for res in self.all_results for k in res.keys() if k != "dataset")
            dataset_results = {
                k: np.mean([
                    res[k]
                    for res in self.all_results
                    if (k in res and (dataset_name is None or res["dataset"] == dataset_name))
                ])
                for k in keys
            }
            return dataset_results
        else:
            return self.evaluate(dialogs, dataset_name=dataset_name)

    def plot(self, feature=None, kde_bw=0.3, show=True, save_dir=None, save_stats_csv=True):
        if not self.all_results:
            print("No results to plot. Please run evaluation first.")
            return
        df = pd.DataFrame(self.all_results)
        if feature is None:
            exclude_cols = {"dataset"}
            all_features = [col for col in df.columns if col not in exclude_cols]
            base_names = set("_".join(col.split("_")[1:]) for col in all_features)
        else:
            base_names = [feature]
        stats_all = []
        for base in base_names:
            feature_cols = [col for col in df.columns if base in col]
            if not feature_cols:
                continue
            for f in feature_cols:
                plt.figure(figsize=(8, 5))
                stats = {"feature": f}
                means = {}
                stds = {}
                ax = plt.gca()
                for dataset in df['dataset'].unique():
                    values = df[df['dataset'] == dataset][f].dropna()
                    if len(values) < 2:
                        continue
                    values.plot.kde(bw_method=kde_bw, label=f"{dataset}", ax=ax)
                for i, dataset in enumerate(df['dataset'].unique()):
                    values = df[df['dataset'] == dataset][f].dropna()
                    if len(values) < 2:
                        continue
                    mean = values.mean()
                    std = values.std()
                    color = ax.get_lines()[i].get_color()
                    plt.axvline(mean, linestyle="--", color=color, label=f"{dataset} mean ({mean:.2f})")
                    stats[f"{dataset}_mean"] = mean
                    stats[f"{dataset}_std"] = std
                    means[dataset] = mean
                    stds[dataset] = std
                # sds_away calculation
                if "primock" in means and "ours" in means and stds["primock"] > 0:
                    sds_away = (means["ours"] - means["primock"]) / stds["primock"]
                    stats["sds_away"] = sds_away
                    if sds_away > 0:
                        stats["sds_away_explanation"] = (
                            f"Our dataset is {abs(sds_away):.2f} standard deviations higher than Primock."
                        )
                    else:
                        stats["sds_away_explanation"] = (
                            f"Our dataset is {abs(sds_away):.2f} standard deviations lower than Primock."
                        )
                # plt.xlabel(f)
                plt.xlabel(f"{f} (%)" if "hesitation_rate" in f else f)
                plt.ylabel("Density")
                plt.title(f"KDE plot of {f} by dataset")
                plt.legend()
                plt.grid(alpha=0.3)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, f"{f}.png"), dpi=300)
                if show:
                    plt.show()
                plt.close()
                stats_all.append(stats)
        # Save all statistics as CSV
        if save_stats_csv and save_dir:
            stats_df = pd.DataFrame(stats_all)
            stats_csv_path = os.path.join(save_dir, "all_feature_stats.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            print(f"All feature statistics saved to {stats_csv_path}")
        if save_dir:
            print(f"All plots saved to {save_dir}")


class DatasetComparator:
    def __init__(self, evaluators: List[BaseDatasetEvaluator]):
        if not evaluators:
            raise ValueError("No evaluators provided for comparison.")
        for evaluator in evaluators:
            if not isinstance(evaluator, BaseDatasetEvaluator):
                raise TypeError(f"Evaluator {evaluator} is not an instance of `BaseDatasetEvaluator`")

        self.evaluators = evaluators

    def __call__(
        self,
        candidates: Union[str, List[Dialog], List[str], List[List[Dialog]], Dict[str, str], Dict[str, List[Dialog]]],
        digits: int = 2,
        output: Union[str, type] = "markdown",
    ) -> dict:
        if not candidates:
            raise ValueError("No candidates provided for comparison.")

        if isinstance(candidates, str) or isinstance(candidates, list) and isinstance(candidates[0], Dialog):
            candidates = [candidates]  # Ensure candidates is always a list of datasets (set of dialogues)

        results = {}
        dataset_iterator = candidates.items() if isinstance(candidates, dict) else enumerate(candidates)
        for dataset_name, dataset in tqdm(dataset_iterator, desc="Evaluating datasets", leave=False):
            if isinstance(dataset_name, int):
                dataset_name += 1
            results[dataset_name] = {}
            for evaluator in self.evaluators:
                evaluator_name = evaluator.name
                score = evaluator(dataset, dataset_name=dataset_name)
                if isinstance(score, dict):
                    for metric, value in score.items():
                        results[dataset_name][f"{evaluator_name}-{metric}"] = value
                else:
                    results[dataset_name][evaluator_name] = score

        if output == "dict" or output is dict:
            return results
        elif output in ["markdown", "table"]:
            dict_to_table(results, markdown=output == "markdown", format=f".{digits}f")  # sort_by="evaluator_name"
        else:
            raise ValueError(f"Unsupported output format: {output}. Supported formats are "
                             "'dict', 'markdown', and 'table'.")

    def plot(self, show: bool = True, save_folder_path: str = None):
        """
        Plot the results of the evaluators.
        """
        if not self.evaluators:
            logger.info("No evaluators to plot.")
            return

        for evaluator in self.evaluators:
            if hasattr(evaluator, "plot"):
                evaluator.plot(show=show,
                               save_path=os.path.join(save_folder_path,
                                                      f"{evaluator.name}.png") if save_folder_path else None)
