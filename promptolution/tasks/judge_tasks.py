"""Module for judge tasks."""

import pandas as pd

from typing import TYPE_CHECKING, List, Optional

from promptolution.tasks.base_task import BaseTask
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.tasks.base_task import EvalStrategy
    from promptolution.utils.config import ExperimentConfig

logger = get_logger(__name__)

JUDGE_PROMPT_WITH_GROUND_TRUTH = """You are an expert evaluator. Judge how well the prediction matches the ground truth, for the given task.

Task:
{task}

Input:
{input}

Ground Truth:
{ground_truth}

Prediction:
{prediction}

Evaluate how closely the prediction aligns with the ground truth. Consider correctness, completeness, and accuracy of the match.

Provide a score from -5 to +5 where:
- -5: Completely incorrect/opposite
- 0: Partially correct
- +5: Perfect match

Return your answer encompased by <final_score></final_score>"""

JUDGE_PROMPT_WITHOUT_GROUND_TRUTH = """You are an expert evaluator. Judge the quality of the prediction, for the given task.

Task:
{task}

Input:
{input}

Prediction:
{prediction}

Evaluate how well the prediction addresses the input for the given task. Consider correctness, quality, relevance, completeness, and excellence of execution.

Provide a score from -5 to +5 where:
- -5: Completely wrong/inappropriate
- 0: Partially addresses the task with mixed quality
- +5: Exceptional response that brilliantly solves the task with creativity, insight, or outstanding execution that goes beyond basic correctness

Return your answer encompased by <final_score></final_score>"""


class JudgeTask(BaseTask):
    """Task that evaluates a predictor using an LLM as a judge, optionally accepting a ground truth."""

    def __init__(
        self,
        df: pd.DataFrame,
        judge_llm: "BaseLLM",
        x_column: str = "x",
        y_column: Optional[str] = None,
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: "EvalStrategy" = "full",
        seed: int = 42,
        judge_prompt: Optional[str] = None,
        min_score: float = -5.0,
        max_score: float = 5.0,
        config: "ExperimentConfig" = None,
    ):
        """Initialize the JudgeTask.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            judge_llm (BaseLLM): The LLM judging the predictions.
            x_column (str): Name of the column containing input texts.
            y_column (Optional[str]): Name of the column containing labels/ground truth (if applicable).
            task_description (Optional[str]): Description of the task, parsed to the Judge-LLM and Meta-LLM.
            n_subsamples (int): Number of subsamples to use for evaluation.
            eval_strategy (EvalStrategy): Subsampling strategy to use for evaluation.
            seed (int): Random seed for reproducibility.
            judge_prompt (Optional[str]): Custom prompt for the judge. Note: The score of the Judge will be extracted inside <final_score> tags.
            min_score (float): Minimum score for evaluation.
            max_score (float): Maximum score for evaluation.
            config (ExperimentConfig, optional): Configuration for the task, overriding defaults.
        """
        if judge_prompt is None:
            judge_prompt = JUDGE_PROMPT_WITH_GROUND_TRUTH if y_column else JUDGE_PROMPT_WITHOUT_GROUND_TRUTH
        self.judge_prompt = judge_prompt
        self.min_score = min_score
        self.max_score = max_score

        super().__init__(
            df=df,
            x_column=x_column,
            y_column=y_column,
            task_description=task_description,
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
            config=config,
        )
        self.judge_llm = judge_llm

    def _construct_judge_prompt(self, x: str, pred: str, y: Optional[str] = None) -> str:
        """Construct the judge prompt based on whether ground truth is available."""
        if y is not None:
            prompt = self.judge_prompt.replace("{ground_truth}", str(y))
        else:
            prompt = self.judge_prompt

        task_description = self.task_description or ""
        prompt = prompt.replace("{task}", task_description).replace("{input}", x).replace("{prediction}", pred)
        return prompt

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> List[float]:
        """Calculate the score for a single prediction using the LLM judge."""
        prompts: List[str] = []
        for x, y, pred in zip(xs, ys, preds):
            judge_prompt = self._construct_judge_prompt(x, pred, y)
            prompts.append(judge_prompt)
        judge_responses = self.judge_llm.get_response(prompts)
        scores_str = extract_from_tag(judge_responses, "<final_score>", "</final_score>")
        scores = []
        for score_str in scores_str:
            try:
                # only numeric chars, - or . are allowed
                score_str = "".join(filter(lambda c: c.isdigit() or c in "-.", score_str))
                score = float(score_str)
                # normalize from [min_score, max_score] to [0, 1]
                score = (score - self.min_score) / (self.max_score - self.min_score)
                score = max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Failed to parse score '{score_str}' as float. Defaulting to a score 0.0.")
                score = 0.0

            scores.append(score)

        return scores
