# medical_dialogue_evaluator/formatters.py
"""
Provides functions to format evaluation reports into various output formats.
"""
import json
import csv
import io
from typing import List, Dict
from data_models import FullEvaluationReport

def to_json(reports: List[FullEvaluationReport]) -> str:
    """Formats a list of reports into a JSON string."""
    return json.dumps([r.dict() for r in reports], indent=2, ensure_ascii=False)

def to_csv(reports: List[FullEvaluationReport]) -> str:
    """Formats reports into a single CSV string, ideal for spreadsheets."""
    output = io.StringIO()
    headers = ['dialogue_id', 'indicator_id', 'indicator_name', 'not_applicable', 'score', 'justification']
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    
    for report in reports:
        for result in report.evaluation_results:
            row = {
                'dialogue_id': report.dialogue_id,
                'indicator_id': result.indicator_id,
                'indicator_name': result.indicator_name,
                'not_applicable': result.not_applicable,
                'score': 'N/A' if result.not_applicable else result.score,
                'justification': result.justification,
            }
            writer.writerow(row)
            
    return output.getvalue()

def to_markdown(reports: List[FullEvaluationReport]) -> str:
    """Formats reports into a human-readable Markdown string."""
    lines = ["# Clinical Dialogue Evaluation Reports\n"]
    for report in reports:
        lines.append(f"## Report for Dialogue: `{report.dialogue_id}`\n")
        lines.append("| Indicator | Score | Justification |")
        lines.append("|:---|:---:|:---|")
        for result in report.evaluation_results:
            score_display = "N/A" if result.not_applicable else f"{result.score}/5"
            justification = result.justification.replace("\n", " ")
            lines.append(f"| **{result.indicator_name}**<br>(`{result.indicator_id}`) | {score_display} | {justification} |")
        lines.append("\n---\n")
    return "\n".join(lines)

FORMATTERS: Dict[str, callable] = {
    "json": to_json,
    "csv": to_csv,
    "md": to_markdown,
}

