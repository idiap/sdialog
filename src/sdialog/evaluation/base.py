"""
Base and abstract evaluation components.

Provides abstract interfaces and utilities to:

  * Embed a dialog (BaseDialogEmbedder)
  * Score a single dialog (BaseDialogScore / BaseDialogFlowScore)
  * Aggregate dialog scores across datasets (BaseDatasetScoreEvaluator)
  * Aggregate dialog embeddings-based scores across datasets (BaseDatasetEmbeddingEvaluator)
  * Judge dialogs with an LLM (BaseLLMJudge)

These abstractions standardize evaluation pipelines for synthetic dialog generation.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import logging
import numpy as np
import matplotlib.pyplot as plt

from math import log
from tqdm.auto import tqdm
from jinja2 import Template
from pydantic import BaseModel
from sklearn.manifold import TSNE
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from sentence_transformers import SentenceTransformer
from langchain_core.language_models.base import BaseLanguageModel

from .. import Dialog
from ..config import config
from .dialog2flow import dialog2graph, DEFAULT_TOKEN_START
from langchain_core.messages import HumanMessage, SystemMessage
from ..util import CacheDialogScore, KNNModel, get_llm_model, upper_camel_to_dash, softmax


CacheDialogScore.init(config["cache"]["path"], enable_cache=config["cache"]["enabled"])
logger = logging.getLogger(__name__)


class LLMJudgeYesNoOutput(BaseModel):
    """
    Structured output used by yes/no LLM judgments.

    :param positive: Boolean (or list of booleans) indicating classification outcome(s).
    :type positive: Union[bool, List[bool]]
    :param reason: Optional explanatory reason (string or list).
    :type reason: Optional[Union[str, List[str]]]
    """
    reason: Optional[Union[str, List[str]]] = None
    positive: Union[bool, List[bool]]


class LLMJudgeScoreOutput(BaseModel):
    """
    Structured output used by numeric score LLM judgments.

    :param score: Numeric score (int or float).
    :type score: Union[int, float]
    :param reason: Optional explanatory reason.
    :type reason: Optional[str]
    """
    reason: Optional[str] = None
    score: Union[int, float] = None


class BaseDialogEmbedder(ABC):
    """
    Base class for dialog embedding models.

    Subclasses must implement the abstract method:
    ``embed(dialog: Dialog) -> np.ndarray``

    Example:

        .. code-block:: python

            from sdialog.evaluation.base import BaseDialogEmbedder
            import numpy as np

            # Custom embedder to embed dialogues as random N-d embeddings
            class RndEmbedder(BaseDialogEmbedder):
                def __init__(self, n=256):
                    self.n = n

                def embed(self, dialog):
                    return np.random.rand(self.n)

            # Create a new embedder for 128-d embeddings and embed some dialogues
            rnd_embedder = RndEmbedder(n=128)
            for d in dialogues:
                emb = rnd_embedder(d)
                print(emb.shape)  # (128,)

    :param name: Optional name identifier for the embedder.
    :type name: Optional[str]
    """
    def __init__(self, name: Optional[str] = None):
        """Initialize the dialog embedder."""
        self.name = name

    def __call__(self, dialog: Dialog) -> np.ndarray:
        """
        Embed a dialog into a vector representation (delegates to embed()).

        :param dialog: The dialog instance to embed.
        :type dialog: Dialog
        :return: Vector representation of the dialog.
        :rtype: np.ndarray
        """
        return self.embed(dialog)

    @abstractmethod
    def embed(self, dialog: Dialog) -> np.ndarray:
        """
        Produce an embedding vector for the given dialog.

        :param dialog: The dialog instance to embed.
        :type dialog: Dialog
        :return: Embedding vector.
        :rtype: np.ndarray
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDialogScore(ABC):
    """
    Base class for computing a scalar score for a single dialog.

    Subclasses must implement the abstract method:
    ``score(dialog: Dialog) -> float``

    Example:

        .. code-block:: python

            from sdialog.evaluation.base import BaseDialogScore
            from sdialog import Dialog, Turn

            # Custom score class to count the number of turns in a dialogue
            class TurnCountScore(BaseDialogScore):
                def score(self, dialog):
                    return len(dialog.turns)

            # Create a new instance of our score
            turn_counter = TurnCountScore()

            d = Dialog(turns=[Turn(speaker="u", text="Hi"),
                              Turn(speaker="s", text="Hello")])

            print(turn_counter(d))  # Outputs: 2

    :param name: Name of the score (used in reporting).
    :type name: Optional[str]
    :param ai_speaker: If provided, restrict scoring to turns spoken by this AI speaker (case-insensitive).
    :type ai_speaker: Optional[str]
    """
    def __init__(self, name: Optional[str] = None, ai_speaker: str = None):
        """Initialize the dialog score object."""
        self.name = name
        self.ai_speaker = ai_speaker

    def __call__(self, dialog: Dialog, **kwargs):
        """
        Compute the score for a given dialog (delegates to score()).

        :param dialog: The dialog to score.
        :type dialog: Dialog
        :param kwargs: Additional keyword arguments for scoring.
        :type kwargs: dict
        :return: Scalar score value.
        :rtype: float
        """
        return self.score(dialog, **kwargs)

    @abstractmethod
    def score(self, dialog: Dialog) -> float:
        """
        Compute the score for the provided dialog.

        :param dialog: The dialog to score.
        :type dialog: Dialog
        :return: Scalar score value.
        :rtype: float
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDialogFlowScore(BaseDialogScore):
    """
    Base class for flow‑based dialog scores using a reference dialog graph.

    Builds (or reuses) a flow graph from reference dialogs, encodes turns,
    retrieves nearest nodes, and derives transition probabilities.
    Serves as the foundation for perplexity / likelihood style scores
    (e.g., DialogFlowPPL, DialogFlowScore).

    This is an abstract class (extends ``BaseDialogScore``) and cannot be instantiated directly.
    Subclasses must implement the abstract method: ``score(dialog: Dialog) -> float``

    :param reference_dialogues: List of Dialog objects or path to a serialized dialog file.
    :param ai_speaker: If provided, only system/AI speaker turns are considered in scoring.
    :param k_neighbors: Number of neighbors for softmax aggregation.
    :param use_softmax: If True, weight neighbor probabilities via softmax, else pick top-1.
    :param graph: Optional precomputed graph object to reuse (bypasses construction).
    :param nodes: Optional precomputed node metadata dictionary.
    :param name: Optional score name override (auto if None).
    :param verbose: Verbosity flag forwarded to graph construction.
    :param d2f_kwargs: Additional dialog2graph customization parameters.
    :raises ValueError: If reference_dialogues is invalid.
    """
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 k_neighbors: int = 64,
                 use_softmax: bool = True,
                 graph=None,
                 nodes=None,
                 name: str = None,
                 verbose: bool = False,
                 **d2f_kwargs):
        """Initialize the flow score model by Building or loading a dialog flow graph from reference dialogs."""
        super().__init__(name=name if name else "dfs" + ("" if use_softmax else "-hard"), ai_speaker=ai_speaker)

        d2f_kwargs = {"node_llm_labels_enabled": False,
                      "out_png": False,
                      "edges_prune_threshold": 0.001,
                      "nodes_prune_threshold": 0.001,
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
        self.k_neighbors = k_neighbors
        self.only_system = bool(ai_speaker)
        if graph is not None and nodes is not None:
            # If graph and nodes are provided, use them directly
            self.graph, self.nodes = graph, nodes
        else:
            self.graph, self.nodes = dialog2graph(reference_dialogues,
                                                  system_speaker_name=ai_speaker,
                                                  **self.d2f_kwargs)
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

    def get_node_sequence(self, dialog: Dialog, probs: bool = False) -> List[str]:
        """
        Map each turn to its nearest node and optionally return transition probabilities.

        :param dialog: Dialog to map.
        :type dialog: Dialog
        :param probs: If True, also return per-transition probability estimates.
        :type probs: bool
        :return: List of node IDs or (node_sequence, probability_sequence) if probs=True.
        :rtype: Union[List[str], Tuple[List[str], List[Optional[float]]]]
        :raises ValueError: If a dialog speaker is not found in graph metadata.
        """
        node_sequence = []
        prob_sequence = []
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
            current_node, _ = neighbors[0]
            node_sequence.append(current_node)

            if probs:
                prob_correct_node = softmax([1 - dist for _, dist in neighbors])[0] if self.use_softmax else 1
                prob_current_node = self.graph.get_edge_data(prev_node, current_node)
                prob_sequence.append(prob_current_node["weight"] * prob_correct_node
                                     if prob_current_node is not None else None)
                prev_node = current_node
        return (node_sequence, prob_sequence) if probs else node_sequence

    def compute_dialog_log_likelihood(self, dialog: Dialog) -> Tuple[float, int]:
        """
        Compute cumulative log-probability statistics for a dialog.

        Returns four values:
          sum_log_p_known: Sum of log probabilities only over known edges.
          n_turns_known: Count of contributing turns with known edges (includes initial offset).
          sum_log_p: Sum over all considered turns (unknown edges use uniform fallback).
          n_turns: Total counted turns (includes initial offset; respects ai_speaker filtering).

        :param dialog: Dialog to evaluate.
        :type dialog: Dialog
        :return: Tuple (sum_log_p_known, n_turns_known, sum_log_p, n_turns).
        :rtype: Tuple[float, int, float, int]
        :raises ValueError: If a speaker is missing from metadata.
        """
        sum_log_p, sum_log_p_known = 0, 0
        n_turns, n_turns_known = 1, 1  # start with 1 to account for the first turn and avoid division by zero
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
            current_node, _ = neighbors[0]
            prob_correct_node = softmax([1 - dist for _, dist in neighbors])[0] if self.use_softmax else 1

            prob_current_node = self.graph.get_edge_data(prev_node, current_node)
            if (not self.only_system or speaker == "system"):
                if prob_current_node is not None:
                    log_p = log(prob_current_node["weight"] * prob_correct_node)
                    sum_log_p += log_p
                    sum_log_p_known += log_p
                    n_turns_known += 1
                else:
                    sum_log_p += log(1 / len(self.graph.nodes))  # Uniform distribution if no edge exists
                n_turns += 1
            prev_node = current_node

        return sum_log_p_known, n_turns_known, sum_log_p, n_turns

    @abstractmethod
    def score(self, dialog: Dialog) -> float:
        """
        Compute a flow-based perplexity / likelihood derived score for the dialog.

        :param dialog: Dialog to score.
        :type dialog: Dialog
        :return: Scalar score value.
        :rtype: float
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDatasetEvaluator(ABC):
    """ Base class for dataset evaluators.

    Dataset evaluators take a set of dialogs and return an evaluation.
    Typically, Dataset evaluator subclasses will take a dialogue score (BaseDialogScore object) when
    created and will return an aggregate of the per-dialog scores.

    Subclasses must implement the abstract method:
    ``__call__(dialogues, dataset_name: Optional[str] = None, **kwargs) -> Union[dict, float]``

    Example:

        .. code-block:: python

            from sdialog.evaluation.base import BaseDatasetEvaluator
            from sdialog import Dialog, Turn

            class CountDialogsEvaluator(BaseDatasetEvaluator):
                def __call__(self, dialogues, dataset_name=None):
                    return len(dialogues)

            dialog_counter = CountDialogsEvaluator()

            dialogs = [Dialog(turns=[Turn(speaker="u", text="Hi")]) for _ in range(3)]

            print(dialog_counter(dialogs))  # Outputs: 3
    """
    @abstractmethod
    def __call__(self,
                 dialogues: Union[str, List[Dialog]],
                 dataset_name: str = None,
                 **kwargs) -> Union[dict, float]:
        """
        Evaluate a collection of dialogues.

        :param dialogues: List of Dialog objects or a path to a serialized file.
        :type dialogues: Union[str, List[Dialog]]
        :param dataset_name: Optional label for the dataset.
        :type dataset_name: Optional[str]
        :param kwargs: Additional evaluator-specific parameters.
        :type kwargs: dict
        :return: Evaluation results (scalar or dict).
        :rtype: Union[float, dict]
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError


class BaseDatasetScoreEvaluator(BaseDatasetEvaluator):
    """
    Base class for dataset-level aggregation of per-dialog scores.

    Dataset score evaluators take a dialog score (BaseDialogScore object) when created
    and given a collection of dialogs, aggregate their individual scores to return a single
    value for the collection.

    Subclasses must implement the abstract methods:

        - ``__eval__(dialog_scores: List[Union[float, int]]) -> Union[dict, float]``
        - (Optional) ``__plot__(dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None) -> None``

    Example:

        .. code-block:: python

            import numpy as np
            from sdialog.evaluation import LinguisticFeatureScore
            from sdialog.evaluation.base import BaseDatasetScoreEvaluator

            # Let's create our average score evaluator
            # We need to implement only the __eval__ method (and optionally the __plot__ method)
            # In practice, don't use this example class for average, but better use the built-in `MeanEvaluator`!
            class AverageEvaluator(BaseDatasetScoreEvaluator):
                def __plot__(self, dialog_scores, plot=None): pass  # no-op

                def __eval__(self, dialog_scores):
                    return np.mean(dialog_scores)

            avg_evaluator = AverageEvaluator(
                dialog_score=LinguisticFeatureScore(name="hesitation-rate")
            )
            print(avg_evaluator(dialogs))  # Outputs average hesitation rate over the dialogs

    :param dialog_score: Dialog-level scoring component.
    :type dialog_score: BaseDialogScore
    :param name: Optional evaluator name (auto-derived if None).
    :type name: Optional[str]
    :param enable_plotting: Whether to keep per-dataset scores for plotting.
    :type enable_plotting: bool
    :param verbose: Whether to keep tqdm bars visible.
    :type verbose: bool
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False):
        """Initialize the score evaluator."""
        self.dialog_score = dialog_score
        if not name:
            self.name = upper_camel_to_dash(self.__class__.__name__).replace("-evaluator", "")
            if dialog_score.name:
                self.name += f"-{dialog_score.name}"
        else:
            self.name = name
        self.datasets_scores = {}
        self.enable_plotting = enable_plotting
        self.verbose = verbose

    def __call__(self,
                 dialogues: Union[str, List[Dialog]],
                 dataset_name: str = None,
                 return_scores: bool = False) -> Union[dict, float]:
        """
        Compute per-dialog scores then aggregate.

        :param dialogues: Iterable of Dialog objects or path.
        :type dialogues: Union[str, List[Dialog]]
        :param dataset_name: Label for the dataset (default 'candidate').
        :type dataset_name: Optional[str]
        :param return_scores: If True also return raw score array(s).
        :type return_scores: bool
        :return: Aggregated results or (results, raw_scores) if return_scores=True.
        :rtype: Union[dict, float, Tuple[Union[dict, float], np.ndarray]]
        :raises KeyboardInterrupt: If user interrupts (partial results saved).
        """
        dataset_name = dataset_name or "candidate"
        if dataset_name == "candidate":
            desc = f"Computing {self.name} scores for candidate dataset"
        else:
            desc = f"Computing {self.name} scores for dataset "
            desc += dataset_name if isinstance(dataset_name, int) else f"'{dataset_name}'"
        try:
            scores = [self.dialog_score(dialogue)
                      for dialogue in tqdm(dialogues, desc=desc, leave=self.verbose)]
        except KeyboardInterrupt:
            logger.warning(
                f"Evaluation interrupted by user. Partial results for dataset '{dataset_name}' "
                f"with evaluator '{self.name}' will be saved to disk."
            )
            CacheDialogScore.save()  # Save the cache to disk after scoring
            raise KeyboardInterrupt
        CacheDialogScore.save()

        if scores and isinstance(scores[0], dict):
            metrics = scores[0].keys()
            scores = {metric: np.array([s[metric] for s in scores if s[metric] is not None])
                      for metric in metrics}
            results = {metric: self.__eval__(scores[metric]) for metric in metrics}
            for metric in metrics:
                if metric not in self.datasets_scores:
                    self.datasets_scores[metric] = {}
                self.datasets_scores[metric][dataset_name] = scores[metric]
        else:
            scores = np.array([s for s in scores if s is not None])  # Filter out None scores
            results = self.__eval__(scores)
            self.datasets_scores[dataset_name] = scores  # Store the scores for later use
        return (results, scores) if return_scores else results

    @abstractmethod
    def __plot__(self, dialog_scores: Dict[str, np.ndarray], plot: Optional[plt.Axes] = None):
        """
        Plot stored per-dataset scores.

        :param dialog_scores: Mapping dataset_name -> array of scores.
        :type dialog_scores: Dict[str, np.ndarray]
        :param plot: Optional matplotlib object to use (plt or Axes).
        :type plot: Optional[plt.Axes]
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def __eval__(self, dialog_scores: List[Union[float, int]]) -> Union[dict, float]:
        """
        Aggregate a collection of per-dialog scores.

        :param dialog_scores: List or array of numeric scores.
        :type dialog_scores: List[Union[float, int]]
        :return: Aggregated scalar or dict of metrics.
        :rtype: Union[float, dict]
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    def clear(self):
        """
        Clear stored per-dataset raw scores history.
        """
        self.datasets_scores.clear()

    def plot(self,
             show: bool = True,
             save_path: str = None):
        """
        Generate plots for stored dataset scores.

        :param show: Whether to display the plot(s).
        :type show: bool
        :param save_path: If provided, save figure(s) to this path (metric name appended when multi-metric).
        :type save_path: Optional[str]
        :return: None
        :rtype: None
        """
        if not self.enable_plotting or not self.datasets_scores:
            return

        # Plot box plots for each dataset
        if self.datasets_scores and isinstance(next(iter(self.datasets_scores.values())), dict):
            for metric in self.datasets_scores:
                plt.figure(figsize=(8, 5))
                self.__plot__(self.datasets_scores[metric], plot=plt, metric=metric)
                if save_path:
                    # Append metric name to filename before saving
                    if "." in save_path.split("/")[-1]:
                        base, ext = save_path.rsplit(".", 1)
                        metric_save_path = f"{base}-{metric}.{ext}"
                    else:
                        metric_save_path = f"{save_path}-{metric}"
                    plt.savefig(metric_save_path, dpi=300)
                if show:
                    plt.show()
        else:
            plt.figure(figsize=(8, 5))
            self.__plot__(self.datasets_scores, plot=plt)
            if save_path:
                plt.savefig(save_path, dpi=300)
            if show:
                plt.show()


class BaseDatasetEmbeddingEvaluator(BaseDatasetEvaluator):
    """
    Base class for dataset-level evaluation using dialog embeddings.

    It takes a dialog embedder (BaseDialogEmbedder object) when created
    and given a collection of dialogs, computes their embeddings and returns a single value for the collection.

    Subclasses must implement: ``__eval__(dialog_embs: List[np.ndarray]) -> Union[dict, float]```
      and ``__plot__(dialog_embs: Dict[str, np.ndarray], tsne_model: TSNE, plot: Optional[plt.Axes]) -> None``

    Example:

        .. code-block:: python

            from sdialog import SentenceTransformerDialogEmbedder
            from sdialog.evaluation.base import BaseDatasetEmbeddingEvaluator

            # Evaluator that computes the average centroid cosine distance to a reference centroid
            # We need to implement only the __eval__ method (and optionally the __plot__ method)
            # (in practice, use the built-in ReferenceCentroidEmbeddingEvaluator!)
            class ReferenceCentroidEmbeddingEvaluator(BaseDatasetEmbeddingEvaluator):
                def __plot__(self, dialog_embs): pass  # no-op

                def __init__(self, dialog_embedder, reference_dialogs):
                    self.reference_centroid = np.mean(
                        [dialog_embedder(dialog) for dialog in reference_dialogs], axis=0
                    )

                def __eval__(self, dialog_embs):
                    centroid = np.mean(dialog_embs, axis=0)
                    return cosine(centroid, self.reference_centroid)

            dialog_embedder = SentenceTransformerDialogEmbedder(model_name="sentence-transformers/LaBSE")
            centroid_evaluator = ReferenceCentroidEmbeddingEvaluator(dialog_embedder=dialog_embedder,
                                                                     reference_dialogs=reference_dialogs)

            print(centroid_evaluator(dialogs))  # distance between candidate and reference dialogues
                                                # (cosine distance between their centroids)

    :param dialog_embedder: Dialog embedding component.
    :type dialog_embedder: BaseDialogEmbedder
    :param name: Optional evaluator name (auto-derived if None).
    :type name: Optional[str]
    :param enable_plotting: Whether to store embeddings for plotting.
    :type enable_plotting: bool
    :param verbose: Verbosity flag for progress bars.
    :type verbose: bool
    """
    def __init__(self,
                 dialog_embedder: BaseDialogEmbedder,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False):
        """Initialize the embedding evaluator."""
        self.dialog_embedder = dialog_embedder
        if not name:
            self.name = upper_camel_to_dash(self.__class__.__name__).replace("-evaluator", "")
        else:
            self.name = name
        self.datasets_embs = {}
        self.enable_plotting = enable_plotting
        self.verbose = verbose

    def __call__(self,
                 dialogues: Union[str, List[Dialog]],
                 dataset_name: str = None,
                 return_embs: bool = False) -> Union[dict, float]:
        """
        Embed dialogs and aggregate their embeddings to a single score.

        :param dialogues: Iterable of Dialogs or path.
        :type dialogues: Union[str, List[Dialog]]
        :param dataset_name: Dataset label (default 'candidate').
        :type dataset_name: Optional[str]
        :param return_embs: If True return (results, embeddings_array).
        :type return_embs: bool
        :return: Aggregated evaluation or (results, embeddings) if return_embs.
        :rtype: Union[dict, float, Tuple[Union[dict, float], np.ndarray]]
        """
        dataset_name = dataset_name or "candidate"
        if dataset_name == "candidate":
            desc = f"Computing {self.name} embeddings for candidate dataset"
        else:
            desc = f"Computing {self.name} embeddings for dataset "
            desc += dataset_name if isinstance(dataset_name, int) else f"'{dataset_name}'"

        # Skipping "reference" dataset name
        if dataset_name == "reference" and hasattr(self, "reference_embs"):
            embs = self.reference_embs
        else:
            embs = np.array([self.dialog_embedder(dialogue)
                            for dialogue in tqdm(dialogues, desc=desc, leave=self.verbose)])

            if self.enable_plotting:
                self.datasets_embs[dataset_name] = embs  # Store the embeddings for later use

        results = self.__eval__(embs)
        return (results, embs) if return_embs else results

    @abstractmethod
    def __plot__(self, dialog_embs: Dict[str, np.ndarray], tsne_model: TSNE, plot: Optional[plt.Axes]):
        """
        Plot embeddings from multiple datasets.

        :param dialog_embs: Mapping dataset_name -> embeddings array.
        :type dialog_embs: Dict[str, np.ndarray]
        :param tsne_model: t-SNE model used for dimensionality reduction.
        :type tsne_model: TSNE
        :param plot: Matplotlib handle (plt module or Axes).
        :type plot: Optional[plt.Axes]
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def __eval__(self, dialog_embs: List[np.ndarray]) -> Union[dict, float]:
        """
        Evaluate a collection of dialog embeddings.

        :param dialog_embs: List or array of per-dialog embedding vectors.
        :type dialog_embs: List[np.ndarray]
        :return: Aggregated evaluation (scalar or dict).
        :rtype: Union[dict, float]
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def clear(self):
        """
        Clear stored per-dataset embeddings.
        """
        self.datasets_embs.clear()

    def plot(self,
             show: bool = True,
             save_path: str = None):
        """
        Plot embeddings (e.g., via subclass t-SNE projection) for stored datasets.

        :param show: Whether to display the plot.
        :type show: bool
        :param save_path: If provided, save plot to this path.
        :type save_path: Optional[str]
        :return: None
        :rtype: None
        """
        if not self.enable_plotting or not self.datasets_embs:
            return

        # Plot box plots for each dataset
        plt.figure(figsize=(8, 5))
        self.__plot__(self.datasets_embs)
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()


class BaseLLMJudge(ABC):
    """
    Base class for all LLM-based evaluation judges that render a prompt and return an output.
    This is the base class of built-in base judges like ``LLMJudgeYesNo`` or ``LLMJudgeScore``.

    Subclasses must implement the abstract method:
    ``judge(dialogs: Union[Dialog, List[Dialog]]) -> dict``

    Example:

        .. code-block:: python

            from sdialog.evaluation.base import BaseLLMJudge
            from sdialog import Dialog, Turn

            class MagicJudge(BaseLLMJudge):
                def judge(self, dialog):
                    # Render prompt (dialog -> text)
                    prompt = self.prompt_template.render(dialog=dialog)
                    raw = self(prompt)  # call underlying LLM
                    return raw  # normally you'd parse into structured output

            magic_judge = MagicJudge(prompt_template="How magical is the following dialogue? Dialogue:\\n{{ dialog }}")
            print(magic_judge.judge(dialog))  # Outputs raw LLM response

    :param model: Model instance or model name (falls back to config if None).
    :type model: Union[BaseLanguageModel, str]
    :param prompt_template: Jinja2 template string used to build the human prompt.
    :type prompt_template: str
    :param output_format: Optional Pydantic schema or JSON schema dict for structured output.
    :type output_format: Union[dict, BaseModel]
    :param llm_kwargs: Additional model instantiation parameters overriding config.
    :type llm_kwargs: dict
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 prompt_template: str = "",
                 output_format: Union[dict, BaseModel] = None,
                 **llm_kwargs):
        """Initialize the LLM judge."""
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

        with open(config["prompts"]["evaluation"]["llm_judge"]["base"], encoding="utf-8") as f:
            self.messages = [SystemMessage(f.read()), HumanMessage("")]

    def __call__(self, prompt: str) -> Union[dict, BaseModel]:
        """
        Invoke the underlying LLM with the given rendered prompt.

        :param prompt: Fully rendered human prompt content.
        :type prompt: str
        :return: Raw model response or structured output (depending on output_format).
        :rtype: Union[dict, BaseModel]
        """
        self.messages[1].content = prompt
        return self.llm.invoke(self.messages)

    @abstractmethod
    def judge(self, dialogs: Union[Dialog, List[Dialog]]) -> dict:
        """
        Judge one or many dialogs using the LLM.

        :param dialogs: A single Dialog or list of Dialog objects.
        :type dialogs: Union[Dialog, List[Dialog]]
        :return: Dictionary of judged metrics / fields extracted.
        :rtype: dict
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def prompt(self, system: bool = False) -> str:
        """
        Return the current system or human prompt text.

        :param system: If True return system prompt; else return last human prompt.
        :type system: bool
        :return: Prompt text.
        :rtype: str
        """
        return self.messages[0].content if system else self.messages[1].content
