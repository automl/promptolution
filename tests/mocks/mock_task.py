"""Mock task for testing purposes."""

import math
from unittest.mock import MagicMock

import pandas as pd

from typing import List

from promptolution.tasks.base_task import BaseTask


class MockTask(BaseTask):
    """Mock task for testing optimizers.

    This class simulates a classification task without requiring
    actual data or model inference.
    """

    def __init__(
        self,
        predetermined_scores=None,
        *,
        df: pd.DataFrame | None = None,
        n_subsamples: int = 1,
        eval_strategy: str = "full",
        n_blocks: int | None = None,
        block_idx: int | list[int] = 0,
        eval_blocks: dict[str, set[int]] | None = None,
        task_description: str = "Mock classification task",
        evaluate_fn=None,
        config=None,
    ):
        """Initialize the MockTask with optional overrides for task settings.

        Args:
            predetermined_scores: Dict/list/callable for score generation used by _evaluate.
            config: Optional ExperimentConfig applied to the base class.
            df: Optional dataframe override to seed the task.
            n_subsamples: Number of subsamples to expose through BaseTask.
            eval_strategy: Eval strategy to expose (defaults to "full").
            n_blocks: Number of blocks to report.
            block_idx: Current block index (int or list).
            eval_blocks: Mapping prompt->set of evaluated blocks for selection logic.
            task_description: Description to attach to the task.
            evaluate_fn: Optional callable to replace evaluate entirely for tests.
        """
        base_df = (
            df
            if df is not None
            else pd.DataFrame(
                {"x": ["Sample text 1", "Sample text 2", "Sample text 3"], "y": ["positive", "negative", "neutral"]}
            )
        )

        super().__init__(
            df=base_df,
            x_column="x",
            y_column="y",
            eval_strategy=eval_strategy,
            n_subsamples=n_subsamples,
            config=config,
        )
        self.predetermined_scores = predetermined_scores or {}
        self.call_history = []
        self.score_index = 0
        self.eval_blocks: dict[str, set[int]] = eval_blocks or {}

        self.task_description = task_description
        self.classes = ["positive", "neutral", "negative"]
        self.initial_prompts = ["Classify:", "Determine:"]

        # Allow tests to control block metadata
        self.n_blocks = n_blocks if n_blocks is not None else max(1, math.ceil(len(self.xs) / self.n_subsamples))
        self.block_idx = block_idx

        # Track block operations for assertions while keeping original behavior
        self._reset_block_idx_impl = super().reset_block_idx
        self.reset_block_idx = MagicMock(side_effect=self._reset_block_idx_impl)
        self._increment_block_idx_impl = super().increment_block_idx
        self.increment_block_idx = MagicMock(side_effect=self._increment_block_idx_impl)

        if evaluate_fn is not None:
            # Replace evaluate for bespoke test logic
            self.evaluate = evaluate_fn  # type: ignore[assignment]

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str], **kwargs) -> List[float]:
        """Calculate the score for a single prediction.

        Args:
            xs: Input data (not used in mock)
            ys: Ground truth labels (not used in mock)
            preds: Predicted labels

        Returns:
            Score based on predetermined scores or a default logic.
        """
        if isinstance(self.predetermined_scores, dict):
            return [self.predetermined_scores.get(pred, 0) for pred in preds]
        elif isinstance(self.predetermined_scores, list):
            if not self.predetermined_scores:
                return [0 for _ in preds]

            scores = [
                self.predetermined_scores[(self.score_index + i) % len(self.predetermined_scores)]
                for i in range(len(preds))
            ]
            self.score_index += len(preds)
            return scores
        elif callable(self.predetermined_scores):
            return self.predetermined_scores(xs)
        else:
            return [len(pred) for pred in preds]

    def get_evaluated_blocks(self, prompts):
        """Return per-prompt evaluated block sets for testing selection logic."""
        return {str(p): set(self.prompt_evaluated_blocks.get(str(p), set())) for p in prompts}
