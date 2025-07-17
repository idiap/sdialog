# medical_dialogue_evaluator/data_models.py
"""
Data models for dialogues and evaluation results, using Pydantic for validation.
This module also includes the plotting logic for evaluation reports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Dialogue(BaseModel):
    """Represents a single medical dialogue with optional metadata."""
    id: str = Field(..., description="Unique dialogue identifier.")
    content: str = Field(..., min_length=1, description="Full text of the conversation.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context (e.g., patient age, gender).")

class EvaluationResult(BaseModel):
    """Stores the structured outcome for a single evaluation indicator."""
    indicator_id: str = Field(..., description="Short, machine-readable ID of the indicator.")
    indicator_name: str = Field(..., description="Full name of the evaluated indicator.")
    not_applicable: bool = Field(..., description="True if the indicator does not apply to the dialogue.")
    score: Optional[int] = Field(default=None, ge=1, le=5, description="The score from 1-5, or null if not applicable.")
    justification: str = Field(..., min_length=3, description="Detailed explanation for the score or reason for being not applicable.")

class FullEvaluationReport(BaseModel):
    """A comprehensive report containing all evaluation results for a single dialogue."""
    dialogue_id: str
    evaluation_results: List[EvaluationResult]

    def _get_plot_data(self) -> pd.DataFrame:
        """Helper to create a pandas DataFrame from applicable results."""
        applicable_results = [r.dict() for r in self.evaluation_results if not r.not_applicable]
        if not applicable_results:
            return pd.DataFrame()
        return pd.DataFrame(applicable_results)

    def plot(self, plot_type: str = 'bar', save_path: Optional[str] = None):
        """
        Generates and saves a plot of the evaluation scores.

        Args:
            plot_type: The type of plot to generate ('bar' or 'radar').
            save_path: The file path to save the plot image. If None, displays the plot.
        """
        df = self._get_plot_data()
        if df.empty:
            print(f"Skipping plot for {self.dialogue_id}: No applicable indicators to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        
        if plot_type == 'bar':
            self._plot_bar(df, save_path)
        elif plot_type == 'radar':
            self._plot_radar(df, save_path)
        else:
            raise ValueError("Invalid plot_type. Choose 'bar' or 'radar'.")

    def _plot_bar(self, df: pd.DataFrame, save_path: Optional[str]):
        """Generates a bar chart."""
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='score', y='indicator_name', data=df.sort_values('score'), palette='viridis', hue='indicator_name', dodge=False, legend=False)
        ax.set_title(f"Evaluation Scores for Dialogue: {self.dialogue_id}", fontsize=16, weight='bold')
        ax.set_xlabel("Score (1-5)", fontsize=12)
        ax.set_ylabel("Indicator", fontsize=12)
        ax.set_xlim(0, 5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_radar(self, df: pd.DataFrame, save_path: Optional[str]):
        """Generates a radar chart."""
        labels = df['indicator_name'].values
        stats = df['score'].values
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        stats = np.concatenate((stats, [stats[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, stats, color='blue', linewidth=2, linestyle='solid')
        ax.fill(angles, stats, color='blue', alpha=0.25)
        
        ax.set_rlim(0, 5)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=12)
        ax.set_title(f"Evaluation Profile for Dialogue: {self.dialogue_id}", size=16, weight='bold', y=1.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

        