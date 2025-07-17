# medical_dialogue_evaluator/main_evaluator.py
"""
The main asynchronous evaluation orchestrator.
"""
import asyncio
from typing import List
from jinja2 import Template

from data_models import Dialogue, EvaluationResult, FullEvaluationReport
from evaluators.base import BaseEvaluator
from logger import logger
from prompts import PROMPT_TEMPLATE
from utils import get_llm_client, EvaluationOutput

class DialogueEvaluator:
    """Orchestrates the asynchronous evaluation of a medical dialogue."""
    def __init__(self, evaluators: List[BaseEvaluator]):
        if not evaluators:
            raise ValueError("At least one evaluator must be provided.")
        self.evaluators = evaluators
        self.llm_client = get_llm_client()
        self.prompt_template = Template(PROMPT_TEMPLATE)

    async def _evaluate_single_indicator(self, evaluator: BaseEvaluator, dialogue: Dialogue) -> EvaluationResult:
        """Asynchronously evaluates one indicator for a dialogue."""
        try:
            prompt = self.prompt_template.render(
                indicator_id=evaluator.indicator_id,
                indicator_name=evaluator.indicator_name,
                indicator_definition=evaluator.definition,
                low_example=evaluator.scoring_rubric["low_example"],
                high_example=evaluator.scoring_rubric["high_example"],
                dialogue_content=dialogue.content
            )
            llm_response: EvaluationOutput = await self.llm_client.ainvoke(prompt)
            
            return EvaluationResult(
                indicator_id=evaluator.indicator_id,
                indicator_name=evaluator.indicator_name,
                not_applicable=llm_response.not_applicable,
                score=llm_response.score,
                justification=llm_response.justification
            )
        except Exception as e:
            logger.error(f"Evaluator '{evaluator.indicator_name}' failed for dialogue '{dialogue.id}': {e}")
            return EvaluationResult(
                indicator_id=evaluator.indicator_id,
                indicator_name=evaluator.indicator_name,
                not_applicable=True,
                score=None,
                justification=f"Automatic failure due to exception: {e}"
            )

    async def evaluate(self, dialogue: Dialogue) -> FullEvaluationReport:
        """Runs evaluation for a dialogue across all indicators concurrently."""
        logger.info(f"Starting evaluation for dialogue: {dialogue.id}...")
        
        tasks = [self._evaluate_single_indicator(ev, dialogue) for ev in self.evaluators]
        evaluation_results = await asyncio.gather(*tasks)
        
        logger.info(f"Finished evaluation for dialogue: {dialogue.id}.")
        return FullEvaluationReport(
            dialogue_id=dialogue.id,
            evaluation_results=evaluation_results
        )
    
    