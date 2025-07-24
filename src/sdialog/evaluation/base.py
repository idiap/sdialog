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


scores_cache = CacheDialogScore(config["cache"]["path"], enable_cache=config["cache"]["enabled"])
logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """
    Base class for metrics.
    """
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, input: Union[Dialog, List[Dialog]]) -> Union[dict, float]:
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDialogEmbedder(ABC):
    """
    Base class for dialog embedding models.
    """
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the dialog embedder with a name.
        :param name: Name of the dialog embedder.
        """
        self.name = name

    def __call__(self, dialog: Dialog) -> np.ndarray:
        """
        Embed a dialog into a vector representation.

        :param dialog: The dialog to embed.
        :return: A numpy array representing the embedded dialog.
        """
        return self.embed(dialog)

    @abstractmethod
    def embed(self, dialog: Dialog) -> np.ndarray:
        """
        Embed a dialog into a vector representation.

        :param dialog: The dialog to embed.
        :return: A numpy array representing the embedded dialog.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDialogScore(ABC):
    def __init__(self, name: Optional[str] = None, ai_speaker: str = None):
        """
        Initialize the dialog score with a name.

        :param name: Name of the dialog score.
        :param ai_speaker: If specified, the score will only consider the AI responses.
                           This is useful when we want to study only the AI responses quality.
        """
        self.name = name
        self.ai_speaker = ai_speaker

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


class BaseDialogFlowScore(BaseDialogScore):
    def __init__(self,
                 reference_dialogues: Union[str, List[Dialog]],
                 ai_speaker: str = None,
                 k_neighbors: int = 64,
                 use_softmax: bool = True,
                 name: str = None,
                 verbose: bool = False,
                 **d2f_kwargs):
        super().__init__(name=name if name else "fppl" + ("+sm" if use_softmax else ""), ai_speaker=ai_speaker)

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
        self.only_system = bool(ai_speaker)
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

    def compute_dialog_log_likelihood(self, dialog: Dialog) -> Tuple[float, int]:
        sum_log_p = 0
        n_turns = 0
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
            if (not self.only_system or speaker == "system") and prob_current_node is not None:
                sum_log_p += log(prob_current_node["weight"] * prob_correct_node)
                n_turns += 1
            prev_node = current_node

        return sum_log_p, n_turns

    @abstractmethod
    def score(self, dialog: Dialog) -> float:
        """
        Computes the flow PPL score for the provided dialog.

        :param dialog: The dialog to score.
        :return: A float representing the flow PPL score of the dialog.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDatasetEvaluator(ABC):
    """ Base class for dataset evaluators."""
    @abstractmethod
    def __call__(self,
                 dialogues: Union[str, List[Dialog]],
                 dataset_name: str = None,
                 **kwargs) -> Union[dict, float]:
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDatasetScoreEvaluator(BaseDatasetEvaluator):
    """
    Base class for score-based dataset evaluators.
    """
    def __init__(self,
                 dialog_score: BaseDialogScore,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False):
        self.dialog_score = dialog_score
        if not name:
            self.name = upper_camel_to_dash(self.__class__.__name__).replace("-evaluator", "") + f"-{dialog_score.name}"
        else:
            self.name = name
        self.datasets_scores = {}
        self.enable_plotting = enable_plotting
        self.verbose = verbose

    def __call__(self,
                 dialogues: Union[str, List[Dialog]],
                 dataset_name: str = None,
                 return_scores: bool = False) -> Union[dict, float]:
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
        return (results, scores) if return_scores else results

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
        if not self.enable_plotting or not self.datasets_scores:
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


class BaseDatasetEmbeddingEvaluator(BaseDatasetEvaluator):
    """
    Base class for dataset evaluators.
    """
    def __init__(self,
                 dialog_embedder: BaseDialogEmbedder,
                 name: str = None,
                 enable_plotting: bool = True,
                 verbose: bool = False):
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
        dataset_name = dataset_name or "candidate"
        if dataset_name == "candidate":
            desc = f"Computing {self.name} embeddings for candidate dataset"
        else:
            desc = f"Computing {self.name} embeddings for dataset "
            desc += dataset_name if isinstance(dataset_name, int) else f"'{dataset_name}'"
        embs = np.array([self.dialog_embedder(dialogue)
                         for dialogue in tqdm(dialogues, desc=desc, leave=self.verbose)])

        if self.enable_plotting:
            self.datasets_embs[dataset_name] = embs  # Store the embeddings for later use
        results = self.eval(embs)
        return (results, embs) if return_embs else results

    def clear_history(self):
        self.datasets_embs.clear()

    def plot(self,
             show: bool = True,
             save_path: str = None):
        if not self.enable_plotting or not self.datasets_embs:
            return

        # Plot box plots for each dataset
        plt.figure(figsize=(8, 5))
        self.__plot__(self.datasets_embs)
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

    @abstractmethod
    def __plot__(self, dialog_embs: Dict[str, np.ndarray], tsne_model: TSNE, plot: Optional[plt.Axes]):
        """
        Plot the embeddings of the datasets.
        :param dialog_embs: A dictionary with dataset names as keys and embeddings as values.
        :param tsne_model: The t-SNE model used for dimensionality reduction.
        :param plot: Optional matplotlib Axes object to plot on. If None, creates a new figure.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def eval(self, dialog_embs: List[np.ndarray]) -> Union[dict, float]:
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
