"""Mock task for testing purposes."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from typing import List

from promptolution.tasks.base_task import BaseTask


class MockTask(BaseTask):
    """Mock task for testing optimizers.

    This class simulates a classification task without requiring
    actual data or model inference.
    """

    def __init__(self, predetermined_scores=None):
        """Initialize the MockTask with optional predetermined scores.

        Args:
            predetermined_scores: Dictionary mapping prompts to scores,
                or a list of scores to return in sequence, or a function
                that generates scores based on prompts.
        """
        super().__init__(
            df=pd.DataFrame(
                {"x": ["Sample text 1", "Sample text 2", "Sample text 3"], "y": ["positive", "negative", "neutral"]}
            ),
            x_column="x",
            y_column="y",
        )
        self.predetermined_scores = predetermined_scores or {}
        self.call_history = []
        self.score_index = 0

        self.x_column = "x"
        self.y_column = "y"
        # Default attributes similar to ClassificationTask
        self.task_description = "Mock classification task"
        self.classes = ["positive", "neutral", "negative"]
        self.initial_prompts = ["Classify:", "Determine:"]
        self.n_blocks = 10

        self.increment_block_idx = MagicMock()
        self.reset_block_idx = MagicMock()

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
            self.score_index += 1
            return self.predetermined_scores
        elif callable(self.predetermined_scores):
            return self.predetermined_scores(xs)
        else:
            return [len(pred) for pred in preds]
