# medical_dialogue_evaluator/main_evaluator.py
# ... (imports)
# The only change is in the `_evaluate_single_indicator` method.

class DialogueEvaluator:
    # ... (__init__ remains the same)

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
                not_applicable=llm_response.not_applicable, # Pass the flag
                score=llm_response.score,                    # Pass the optional score
                justification=llm_response.justification
            )
        except Exception as e:
            logger.error(f"Evaluator '{evaluator.indicator_name}' failed for dialogue '{dialogue.id}': {e}")
            return EvaluationResult(
                indicator_id=evaluator.indicator_id,
                indicator_name=evaluator.indicator_name,
                not_applicable=True, # Fail safe to not applicable
                score=None,
                justification=f"Automatic failure due to exception: {e}"
            )
    
    # ... (the rest of the class remains the same)
    async def evaluate(self, dialogue: Dialogue) -> FullEvaluationReport:
        logger.info(f"Starting evaluation for dialogue: {dialogue.id}...")
        
        tasks = [self._evaluate_single_indicator(ev, dialogue) for ev in self.evaluators]
        evaluation_results = await asyncio.gather(*tasks)
        
        logger.info(f"Finished evaluation for dialogue: {dialogue.id}.")
        return FullEvaluationReport(
            dialogue_id=dialogue.id,
            evaluation_results=evaluation_results
        )