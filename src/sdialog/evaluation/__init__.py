"""
evaluation: Evaluation components for dialogue generation and analysis.

This module provides abstract base classes for evaluating dialogues,
including LLM judges, metrics, and similarity scores.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

# pyright: reportInvalidTypeForm=false

import os
import re
import logging
import syllables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from jinja2 import Template
from typing import Optional
from pydantic import BaseModel, conint
from math import exp, log, sqrt
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from .. import Dialog
from ..personas import BasePersona
from ..config import config
from .dialog2flow import dialog2graph, DEFAULT_TOKEN_START
from ..util import CacheDialogScore, KNNModel, softmax, get_llm_model, dict_to_table, upper_camel_to_dash

logger = logging.getLogger(__name__)

scores_cache = CacheDialogScore(config["cache"]["path"], enable_cache=config["cache"]["enabled"])


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
    yes: Union[bool, List[bool]]
    feedback: Optional[Union[str, List[str]]] = None


class LLMJudgeLikertOutput(BaseModel):
    """
    Pydantic model for LLM-generated dialogue output.
    """
    score: conint(ge=1, le=5)
    feedback: Optional[str] = None


class BaseMetric(ABC):
    """
    Base class for metrics.
    """
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, input: Union[Dialog, List[Dialog]]) -> Union[dict, float]:
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDialogScore(ABC):
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the dialog score with a name.
        :param name: Name of the dialog score.
        """
        self.name = name

    def __call__(self, dialog: Dialog):
        """
        Computes the score for the provided dialog.

        :param dialog: The dialog to score.
        :return: A float representing the score of the dialog.
        """
        return self.score(dialog)

    @abstractmethod
    def score(self, dialog: Dialog) -> float:
        """
        Computes the score for the provided dialog.

        :param dialog: The dialog to score.
        :return: A float representing the score of the dialog.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDatasetEvaluator(ABC):
    """
    Base class for dataset evaluators.
    """
    def __init__(self, dialog_score: BaseDialogScore, name: str = None, verbose: bool = False):
        self.dialog_score = dialog_score
        if not name:
            self.name = upper_camel_to_dash(self.__class__.__name__).replace("-evaluator", "") + f"-{dialog_score.name}"
        else:
            self.name = name
        self.datasets_scores = {}
        self.verbose = verbose

    def __call__(self,
                 dialogues: Union[str, List[Dialog]],
                 dataset_name: str = None,
                 return_dialog_scores: bool = False) -> Union[dict, float]:
        dataset_name = dataset_name or "candidate"
        if dataset_name == "candidate":
            desc = f"Computing {self.name} scores for candidate dataset"
        else:
            desc = f"Computing {self.name} scores for dataset "
            desc += dataset_name if isinstance(dataset_name, int) else f"'{dataset_name}'"
        try:
            scores = np.array([self.dialog_score(dialogue)
                               for dialogue in tqdm(dialogues, desc=desc, leave=self.verbose)])
        except KeyboardInterrupt:
            logger.warning(
                f"Evaluation interrupted by user. Partial results for dataset '{dataset_name}' "
                f"with evaluator '{self.name}' will be saved to disk."
            )
            scores_cache.save()  # Save the cache to disk after scoring
            raise KeyboardInterrupt
        scores_cache.save()

        self.datasets_scores[dataset_name] = scores  # Store the scores for later use
        results = self.eval(scores)
        return (results, scores) if return_dialog_scores else results

    @abstractmethod
    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        """
        Plot the scores of the datasets.
        :param dialog_scores: A dictionary with dataset names as keys and scores as values.
        :param plot: Optional matplotlib Axes object to plot on. If None, creates a new figure.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def clear_history(self):
        self.datasets_scores.clear()

    def plot(self,
             show: bool = True,
             save_path: str = None):
        if not self.datasets_scores:
            return

        # Plot box plots for each dataset
        plt.figure(figsize=(8, 5))
        self.__plot__(self.datasets_scores, plot=plt)
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

    @abstractmethod
    def eval(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        raise NotImplementedError("Subclasses should implement this method.")


class BaseLLMJudge(ABC):
    """
    Base class for LLM judges.
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 prompt_template: str = "",
                 output_format: Union[dict, BaseModel] = None,
                 **llm_kwargs):
        if model is None:
            model = config["llm"]["model"]

        # Collect LLM parameters from config, only if not None
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}

        self.output_format = output_format
        self.prompt_template = Template(prompt_template)

        self.llm = get_llm_model(model_name=model,
                                 output_format=self.output_format,
                                 **llm_kwargs)

        with open(config["prompts"]["evaluation"]["llm_as_judge"], encoding="utf-8") as f:
            self.messages = [SystemMessage(f.read()), HumanMessage("")]

    def __call__(self, prompt: str) -> Union[dict, BaseModel]:
        self.messages[1].content = prompt
        return self.llm.invoke(self.messages)

    @abstractmethod
    def judge(self, dialogs: Union[Dialog, List[Dialog]]) -> dict:
        """
        Judge the dialogs using the LLM.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def prompt(self, system: bool = False) -> str:
        """
        Returns the prompt template used by the LLM judge.
        """
        return self.messages[0].content if system else self.messages[1].content


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
                                             feedback=feedback if feedback is not None else self.feedback)
        output = self.output_format.model_validate(BaseLLMJudge.__call__(self, prompt))

        return output

    @scores_cache.cache
    def score(self, dialog: Dialog) -> float:
        """
        Computes the score for the provided dialog.

        :param dialog: The dialog to score.
        :return: A float representing the score of the dialog.
        """
        output = self.judge(dialog)
        try:
            return int(output.yes[0]) if isinstance(output.yes, list) else int(output.yes)
        except TypeError:
            raise ValueError(f"LLMJudgeYesNo output '{output.yes}' is not a boolean or list of booleans, "
                             f"cannot convert to integer score.")


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
            prompt_template_real_or_not = f.read()
        super().__init__(prompt_template_real_or_not,
                         model=model,
                         feedback=feedback,
                         **llm_kwargs)


class LLMJudgeLikert(BaseDialogScore, BaseLLMJudge):
    """LLM judge for classifying a dialogue as "yes or no" (boolean) output and feedback."""
    def __init__(self,
                 prompt_template: str,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 as_score_error_value: int = -1,
                 **llm_kwargs):
        BaseDialogScore.__init__(self,
                                 name=upper_camel_to_dash(self.__class__.__name__))
        BaseLLMJudge.__init__(self,
                              model=model,
                              output_format=LLMJudgeLikertOutput,
                              prompt_template=prompt_template,
                              **llm_kwargs)

        self.feedback = feedback
        self.as_score_error_value = as_score_error_value  # Default value to return if LLM output cannot be parsed

    def judge(self, dialogs: Union[Dialog, List[Dialog]], feedback: bool = None) -> Union[LLMJudgeLikertOutput,
                                                                                          List[LLMJudgeLikertOutput]]:
        if isinstance(dialogs, Dialog):
            dialogs = [dialogs]  # Wrap single dialog in a list

        outputs = []
        for dialog in dialogs:
            prompt = self.prompt_template.render(dialog=dialog,
                                                 feedback=feedback if feedback is not None else self.feedback)
            output = self.output_format.model_validate(BaseLLMJudge.__call__(self, prompt))
            outputs.append(output)

        if isinstance(dialogs, Dialog):
            outputs = outputs[0]  # Take single output in a list

        return outputs

    @scores_cache.cache
    def score(self, dialog: Dialog) -> float:
        """
        Computes the score for the provided dialog.

        :param dialog: The dialog to score.
        :return: A float representing the score of the dialog.
        """
        output = self.judge(dialog)
        try:
            return int(output.score[0]) if isinstance(output.score, list) else int(output.score)
        except TypeError:
            raise ValueError(f"LLMJudgeLikert output '{output.score}' is not a boolean or list of booleans, "
                             f"cannot convert to integer score.")


class LLMJudgeRefusal(LLMJudgeYesNo):
    """
    LLM judge for evaluating if a dialogue contains a refusal response.
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_refusal"], encoding="utf-8") as f:
            prompt_template_real_or_not = f.read()
        super().__init__(prompt_template_real_or_not,
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


class KDEDivergenceEvaluator(BaseDatasetEvaluator):
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 reference_dialogues: Union[str, List[Dialog]] = None,
                 metric: str = "all",
                 kde_bw: float = None,
                 name: str = None,
                 verbose: bool = False,
                 **evaluator_kwargs):
        super().__init__(dialog_score, name=name, **evaluator_kwargs)

        if reference_dialogues is None:
            if hasattr(dialog_score, "reference_dialogues"):
                reference_dialogues = dialog_score.reference_dialogues
            else:
                raise ValueError("Reference dialogues must be provided or "
                                 "the dialog_score must have a reference_dialogues attribute.")

        if metric != "all":
            self.name += f"-{metric}"
        self.metric = metric
        self.kde_bw = kde_bw
        self.reference_scores = [self.dialog_score(dialogue)
                                 for dialogue in tqdm(reference_dialogues,
                                                      desc=f"Computing reference {self.name} scores",
                                                      leave=verbose)]
        self.reference_scores = np.array(self.reference_scores)

    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        if "reference" not in dialog_scores and self.reference_scores is not None:
            pd.Series(self.reference_scores, name="reference").plot.kde(bw_method=self.kde_bw, lw=3, color="grey")
        for dataset_name, scores in dialog_scores.items():
            pd.Series(scores, name=dataset_name).plot.kde(bw_method=self.kde_bw)
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


class StatsEvaluator(BaseDatasetEvaluator):
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


class FrequencyEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for computing the frequency or percentage of dialogues matching a condition (e.g., refusal responses).
    """
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


class LinguisticFeaturesDatasetEvaluator(BaseDatasetEvaluator):
    def __init__(self, features=None, name="linguistic_features"):
        super().__init__()
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


class DialogFlowPPL(BaseDialogScore):
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 k_neighbors=64,
                 use_softmax=True,
                 only_system=False,
                 name=None,
                 verbose=False,
                 **d2f_kwargs):
        super().__init__(name=name if name else "fppl" + ("+sm" if use_softmax else ""))

        d2f_kwargs = {"node_llm_labels_enabled": False,
                      "out_png": False,
                      #  "node_embedding_model": embedding_model,
                      "verbose": verbose,
                      **d2f_kwargs}

        if isinstance(reference_dialogues, str):
            reference_dialogues = Dialog.from_file(reference_dialogues)
        if not reference_dialogues or not isinstance(reference_dialogues, list):
            raise ValueError("Reference dialogues must be provided as a list of Dialog objects or a file path.")

        self.reference_dialogues_ids = [d.id for d in reference_dialogues]  # for the key cache
        self.d2f_kwargs = d2f_kwargs  # for the key cache

        self.reference_dialogues = reference_dialogues
        self.use_softmax = use_softmax
        self.only_system = only_system
        self.graph, self.nodes = dialog2graph(reference_dialogues, **self.d2f_kwargs)
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

    @scores_cache.cache
    def score(self, dialog: Dialog) -> float:
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
            neighbors = self.knn_models[speaker](utt_emb, k=None if self.use_softmax else 1)
            nearest_id, _ = neighbors[0]
            prob_correct_node = softmax([1 - dist for _, dist in neighbors])[0] if self.use_softmax else 1

            prob_next_node = self.graph.get_edge_data(prev_node, nearest_id)
            if (not self.only_system or speaker == "system") and prob_next_node is not None:
                sum_log_p += log(prob_next_node["weight"] * prob_correct_node)
                sys_turns += 1
            prev_node = nearest_id

        return exp(-sum_log_p / sys_turns)


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
        output: str = "table",
    ) -> dict:
        if not candidates:
            raise ValueError("No candidates provided for comparison.")

        if isinstance(candidates, str) or isinstance(candidates, list) and isinstance(candidates[0], Dialog):
            candidates = [candidates]  # Ensure candidates is always a list of datasets (set of dialogues)

        results = {}
        dataset_iterator = candidates.items() if isinstance(candidates, dict) else enumerate(candidates)
        for dataset_name, dataset in dataset_iterator:
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

        if output == "dict":
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
            raise ValueError("No evaluators to plot.")

        for evaluator in self.evaluators:
            evaluator.plot(show=show,
                           save_path=os.path.join(save_folder_path,
                                                  f"{evaluator.name}.png") if save_folder_path else None)

    compare = __call__  # Allow direct call to compare method
