# run_evaluation.py
import asyncio
import argparse
import os
from typing import List
from medical_dialogue_evaluator.data_models import Dialogue, FullEvaluationReport
from medical_dialogue_evaluator.main_evaluator import DialogueEvaluator
from medical_dialogue_evaluator.evaluators import discover_evaluators
from medical_dialogue_evaluator.logger import logger
from medical_dialogue_evaluator.formatters import FORMATTERS

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run asynchronous medical dialogue evaluations.")
    parser.add_argument('--dialogue_file', type=str, required=True, help="Path to the input dialogue file (JSONL).")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output report file (e.g., 'report.json').")
    parser.add_argument('--format', type=str, default="json", choices=FORMATTERS.keys(), help="The output format for the report.")
    parser.add_argument('--plot', type=str, nargs='?', const='plots', default=None, help="Generate plots for each dialogue. Optionally specify a directory to save them.")
    return parser.parse_args()

def load_dialogues(dialogue_file: str) -> List[Dialogue]:
    # ... (function remains the same)
    import json
    with open(dialogue_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return [Dialogue.parse_raw(line) for line in lines]

async def main():
    """Main asynchronous function to orchestrate the evaluation process."""
    args = parse_args()
    
    # Create plot directory if it doesn't exist
    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
        logger.info(f"Plots will be saved to '{args.plot}' directory.")
    
    logger.info("Initializing evaluation framework...")
    all_evaluators = discover_evaluators()
    logger.info(f"Successfully discovered {len(all_evaluators)} evaluators.")
    
    dialogues = load_dialogues(args.dialogue_file)
    if not dialogues:
        logger.warning("No dialogues loaded. Exiting.")
        return
    
    evaluator = DialogueEvaluator(evaluators=all_evaluators)
    
    all_reports: List[FullEvaluationReport] = []
    for dialogue in dialogues:
        report = await evaluator.evaluate(dialogue)
        all_reports.append(report)

        # Generate and save plots if the flag is set
        if args.plot:
            logger.info(f"Generating plots for dialogue: {dialogue.id}")
            report.plot(plot_type='bar', save_path=os.path.join(args.plot, f"{dialogue.id}_bar.png"))
            report.plot(plot_type='radar', save_path=os.path.join(args.plot, f"{dialogue.id}_radar.png"))

    # Format and save the text-based report
    formatter = FORMATTERS[args.format]
    output_content = formatter(all_reports)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
        
    logger.info(f"Evaluation complete. Report saved to '{args.output_file}'.")

if __name__ == "__main__":
    asyncio.run(main())
