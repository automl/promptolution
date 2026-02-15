"""Module for Reward tasks."""


from collections import defaultdict

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
        y_column: Optional[str] = None,
        reward_columns: Optional[List[str]] = None,
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: "EvalStrategy" = "full",
        seed: int = 42,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the RewardTask.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
            reward_function (Callable): Function that takes a prediction, potential keyword arguments from the dataframe, and returns a reward score. Note: The optimizers aim to maximize.
            x_column (str, optional): Name of the column containing input texts. Defaults to "x".
            y_column (str, optional): Name of the column containing target texts if available. Defaults to None.
            reward_columns (List[str], optional): Additional dataframe columns to pass as keyword args to reward_function.
            task_description (str, optional): Description of the task.
            n_subsamples (int, optional): Number of subsamples to use. Defaults to 30.
            eval_strategy (str, optional): Subsampling strategy to use. Defaults to "full".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            config (ExperimentConfig, optional): Configuration for the task, overriding defaults.
        """
        self.reward_function = reward_function
        self.reward_columns = reward_columns or []
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
        self.task_type = "reward"
        # x -> kwargs to reward function
        km = self.df.set_index(x_column)[self.reward_columns].to_dict("index")
        self.kwargs_map = defaultdict(dict, km)

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> np.ndarray:
        """Calculate reward for each prediction, passing configured columns as kwargs."""
        kwargs_list = [self.kwargs_map[x] for x in xs]
        rewards = [self.reward_function(pred, **kwargs) for pred, kwargs in zip(preds, kwargs_list)]
        return np.asarray(rewards, dtype=float)
