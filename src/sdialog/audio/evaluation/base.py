"""
Base and abstract audio evaluation components.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from sdialog.audio.dialog import AudioDialog


class Result:
    """
    Result class for audio evaluation.
    """
    def __init__(
        self,
        metrics: float,
        data: Dict[str, Any],
        plots: list
    ):
        self.metrics = metrics
        self.data = data
        self.plots = plots

    def display(self):
        """
        Display the result.
        """
        from IPython.display import display

        print(self.metrics)

        if self.plots is not None:
            for img in self.plots:
                display(img)


class BaseAudioDialogScore(ABC):
    """
    Base class for computing a scalar score for a single audio dialog.
    Subclasses must implement the abstract method:
    ``score(dialog: AudioDialog) -> float``
    Example:
        .. code-block:: python
            from sdialog.evaluation.base import BaseAudioDialogScore
            from sdialog.audio.dialog import AudioDialog
            # Custom score class to count the number of turns in an audio dialogue
            class TurnCountScore(BaseAudioDialogScore):
                def score(self, dialog):
                    return len(dialog.turns)
            # Create a new instance of our score
            turn_counter = TurnCountScore()
            d = AudioDialog() # create your AudioDialog
            print(turn_counter(d))
    :param name: Name of the score (used in reporting).
    :type name: Optional[str]
    :param ai_speaker: If provided, restrict scoring to turns spoken by this AI speaker (case-insensitive).
    :type ai_speaker: Optional[str]
    """
    def __init__(self, name: Optional[str] = None):
        """Initialize the dialog score object."""
        self.name = name

    def __call__(self, dialog: AudioDialog, **kwargs):
        """
        Compute the score for a given dialog (delegates to score()).
        :param dialog: The dialog to score.
        :type dialog: AudioDialog
        :param kwargs: Additional keyword arguments for scoring.
        :type kwargs: dict
        :return: Scalar score value.
        :rtype: float
        """
        return self.score(dialog, **kwargs)

    def __str__(self):
        return self.name

    @abstractmethod
    def results2result(self, results: Dict[str, Any], compute_plots: bool = False) -> Result:
        """
        Compute the overall result from the results of the evaluator on all dialogs.
        :param results: The results of the evaluator.
        :type results: Dict[str, Any]
        :return: The overall result of the evaluator on all dialogs.
        :rtype: Result
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def plot(self, data: Dict[str, Any]) -> list:
        """
        Plot the results of the evaluator.
        :param data: The data to plot.
        :type data: Dict[str, Any]
        :return: A list of plots.
        :rtype: list
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def score(self, dialog: AudioDialog) -> Result:
        """
        Compute the score for the provided dialog.
        :param dialog: The dialog to score.
        :type dialog: AudioDialog
        :return: A dictionary containing the score and any additional information.
        :rtype: Dict[str, Any]
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")
