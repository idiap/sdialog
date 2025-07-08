from typing import Union, List
from abc import ABC, abstractmethod

from sdialog import Dialog


class BaseEvaluator(ABC):
    def __init__(self, metrics=None):
        pass

    @abstractmethod
    def evaluate(self, dialogs: Union[Dialog, List[Dialog]]) -> dict:
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
    def compute(self, dialogs: Union[Dialog, List[Dialog]]) -> Union[dict, float]:
        raise NotImplementedError("Subclasses should implement this method.")


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
