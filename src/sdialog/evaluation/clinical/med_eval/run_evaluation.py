# run_evaluation.py
"""
Command-Line Interface (CLI) to run the medical dialogue evaluation framework.
"""
import asyncio
import argparse
import os
from typing import List
from data_models import Dialogue, FullEvaluationReport
from main_evaluator import DialogueEvaluator
from evaluators import discover_evaluators
from logger import logger
from formatters import FORMATTERS


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run asynchronous medical dialogue evaluations.")
    parser.add_argument(
        '--dialogue_file',
        type=str,
        required=True,
        help="Path to the input dialogue file (must be in JSONL format)."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="Path to the output report file (e.g., 'report.json', 'report.csv')."
    )
    parser.add_argument(
        '--format',
        type=str,
        default="json",
        choices=FORMATTERS.keys(),
        help="The output format for the report. Defaults to 'json'."
    )
    parser.add_argument(
        '--plot',
        type=str,
        nargs='?',
        const='plots',
        default=None,
        help="Generate plots for each dialogue. Optionally specify a directory to save them (default: 'plots')."
    )
    return parser.parse_args()


def load_dialogues(dialogue_file: str) -> List[Dialogue]:
    """Loads dialogues from a JSON Lines (.jsonl) file."""
    dialogues = []
    try:
        with open(dialogue_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        dialogues.append(Dialogue.parse_raw(line))
                    except Exception as e:
                        logger.error(f"Failed to parse line {i+1} in {dialogue_file}: {e}")
    except FileNotFoundError:
        logger.error(f"FATAL: Dialogue file not found at: {dialogue_file}")
    return dialogues


async def main():
    """Main asynchronous function to orchestrate the evaluation process."""
    args = parse_args()
    
    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
        logger.info(f"Plots will be saved to '{args.plot}' directory.")
    
    logger.info("Initializing evaluation framework...")
    all_evaluators = discover_evaluators()
    if not all_evaluators:
        logger.error("FATAL: No evaluators found. Please check the 'evaluators' directory.")
        return
    logger.info(f"Successfully discovered {len(all_evaluators)} evaluators.")
    
    dialogues = load_dialogues(args.dialogue_file)
    if not dialogues:
        logger.warning("No dialogues loaded from file. Exiting.")
        return
    
    evaluator = DialogueEvaluator(evaluators=all_evaluators)
    
    all_reports: List[FullEvaluationReport] = []
    for dialogue in dialogues:
        report = await evaluator.evaluate(dialogue)
        all_reports.append(report)

        if args.plot:
            logger.info(f"Generating plots for dialogue: {dialogue.id}")
            report.plot(plot_type='bar', save_path=os.path.join(args.plot, f"{dialogue.id}_bar.png"))
            report.plot(plot_type='radar', save_path=os.path.join(args.plot, f"{dialogue.id}_radar.png"))

    formatter = FORMATTERS[args.format]
    output_content = formatter(all_reports)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
        
    logger.info(f"Evaluation complete. Report saved to '{args.output_file}' in {args.format.upper()} format.")

if __name__ == "__main__":
    asyncio.run(main())
