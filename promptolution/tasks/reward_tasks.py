"""Module for Reward tasks."""


import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Callable, List, Optional

from promptolution.tasks.base_task import BaseTask

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.tasks.base_task import EvalStrategy
    from promptolution.utils.config import ExperimentConfig


class RewardTask(BaseTask):
    """A task that evaluates a predictor using a reward function.

    This task takes a DataFrame, a column name for input data, and a reward function.
    The reward function takes in a prediction as input and returns a scalar reward.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        reward_function: Callable[[str], float],
        x_column: str = "x",
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: "EvalStrategy" = "full",
        seed: int = 42,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the RewardTask.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
            reward_function (Callable): Function that takes a prediction and returns a reward score. Note: The optimizers aim to maximize.
            x_column (str, optional): Name of the column containing input texts. Defaults to "x".
            task_description (str, optional): Description of the task.
            n_subsamples (int, optional): Number of subsamples to use. Defaults to 30.
            eval_strategy (str, optional): Subsampling strategy to use. Defaults to "full".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            config (ExperimentConfig, optional): Configuration for the task, overriding defaults.
        """
        self.reward_function = reward_function
        super().__init__(
            df=df,
            x_column=x_column,
            task_description=task_description,
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
            config=config,
        )

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> np.ndarray:
        """Calculate the score for a single reward prediction using the reward function."""
        rewards = [self.reward_function(pred) for pred in preds]
        return np.asarray(rewards, dtype=float)
