"""
Base and abstract audio evaluation components.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from sdialog.audio.dialog import AudioDialog, RoomAcousticsConfig


class Result:
    """
    Result class for audio evaluation.
    :param metrics: The metrics of the result.
    :type metrics: Dict[str, Any]
    :param data: The data of the result.
    :type data: Dict[str, Any]
    :param plots: The plots of the result.
    :type plots: Optional[list]
    """
    def __init__(
        self,
        metrics: Dict[str, Any],
        data: Dict[str, Any],
        plots: Optional[list] = None,
        dialog_id: Optional[str] = None,
        evaluator_name: Optional[str] = None,
        room_name: Optional[str] = None,
        room_config: Optional[RoomAcousticsConfig] = None,
    ):
        self.metrics = metrics
        self.data = data
        self.plots = plots
        self.dialog_id = dialog_id
        self.evaluator_name = evaluator_name
        self.room_name = room_name
        self.room_config = room_config

    def to_dict(self):
        """Converts the Result object to a dictionary."""
        plot_paths = []
        if self.plots:
            for plot in self.plots:
                if isinstance(plot, str):
                    plot_paths.append(plot)

        return {
            "metrics": self.metrics,
            "data": self.data,
            "dialog_id": self.dialog_id,
            "evaluator_name": self.evaluator_name,
            "room_name": self.room_name,
            "room_config": self.room_config.model_dump(exclude_none=True) if self.room_config else None,
            "plots": plot_paths,
        }

    def display(self, mode: str = "raw", show_plots: bool = True):
        """
        Display the result.
        :param mode: The display mode. Can be 'raw', 'table', or 'structured_text'.
        :type mode: str
        :param show_plots: Whether to display the plots.
        :type show_plots: bool
        """
        from IPython.display import display

        if mode == "raw":
            print("Metrics:", self.metrics)
        elif mode == "table":
            self._display_table()
        elif mode == "structured_text":
            self._display_structured_text()
        else:
            raise ValueError("Invalid display mode. Choose from 'raw', 'table', or 'structured_text'.")

        if show_plots and self.plots is not None:
            for img in self.plots:
                display(img)

    def _display_table(self):
        """
        Display the metrics as a table.
        """

        try:
            from tabulate import tabulate
        except ImportError:
            raise ImportError("tabulate is not installed. Please install it with `pip install tabulate`.")

        rows = []
        is_nested = any(isinstance(v, dict) for v in self.metrics.values())

        if not is_nested:
            headers = ["Metric", "Value"]
            for k, v in self.metrics.items():
                rows.append([k, f"{v:.4f}" if isinstance(v, float) else v])
        else:
            headers = ["Group", "Metric", "Value"]
            groups = list(self.metrics.items())
            for i, (group, metrics_dict) in enumerate(groups):
                if isinstance(metrics_dict, dict):
                    for metric, value in metrics_dict.items():
                        if isinstance(value, dict):
                            if 'mean' in value and 'std' in value:
                                val_str = f"{value['mean']:.4f} (±{value['std']:.4f})"
                                if 'min' in value and 'max' in value:
                                    val_str += (
                                        f" [{value['min']:.4f} - {value['max']:.4f}]"
                                    )
                                rows.append([group, metric, val_str])
                            else:
                                for sub_k, sub_v in value.items():
                                    formatted_sub_v = f"{sub_v:.4f}" if isinstance(sub_v, float) else sub_v
                                    rows.append([group, f"{metric}/{sub_k}", formatted_sub_v])
                        else:
                            formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                            rows.append([group, metric, formatted_value])
                else:
                    formatted_metrics = f"{metrics_dict:.4f}" if isinstance(metrics_dict, float) else metrics_dict
                    rows.append(["Overall", group, formatted_metrics])

                if i < len(groups) - 1:
                    rows.append([''] * len(headers))

        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def _display_structured_text(self):
        """
        Display the metrics as a structured text.
        """

        print("Metrics:")
        is_nested = any(isinstance(v, dict) for v in self.metrics.values())
        if not is_nested:
            for metric, value in self.metrics.items():
                print(f"  - {metric}: {value:.4f}" if isinstance(value, float) else f"  - {metric}: {value}")
            return

        for group, metrics_dict in self.metrics.items():
            if isinstance(metrics_dict, dict):
                print(f"  {group}:")
                for metric, value in metrics_dict.items():
                    if isinstance(value, dict):
                        if 'mean' in value and 'std' in value:
                            val_str = f"{value['mean']:.4f} (±{value['std']:.4f})"
                            if 'min' in value and 'max' in value:
                                val_str += (
                                    f" [{value['min']:.4f} - {value['max']:.4f}]"
                                )
                            print(f"    - {metric}: {val_str}")
                        else:
                            print(f"    - {metric}:")
                            for sub_k, sub_v in value.items():
                                formatted_sub_v = f"{sub_v:.4f}" if isinstance(sub_v, float) else sub_v
                                print(f"      - {sub_k}: {formatted_sub_v}")
                    else:
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else f"{value}"
                        print(f"    - {metric}: {formatted_value}")
            else:
                formatted_metrics = f"{metrics_dict:.4f}" if isinstance(metrics_dict, float) else f"{metrics_dict}"
                print(f"  - {group}: {formatted_metrics}")


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
