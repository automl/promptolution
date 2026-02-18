"""Module for Reward tasks."""
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

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> List[float]:
        """Calculate the score for a single reward prediction using the reward function."""
        rewards = [self.reward_function(pred) for pred in preds]
        return rewards


class DatasetRewardTask(BaseTask):
    """
    Computes ONE reward over the whole evaluated set (xs/preds),
    then broadcasts it back as per-example rewards.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_reward_fn,  # (xs: List[str], preds: List[str], ys: Optional[List[str]], y2s: Optional[List[str]]) -> float
        x_column: str = "x",
        y_column: Optional[str] = None,   # optional, if you have references
        y2_column: Optional[str] = None,    # optional, if you have initial answers and want to compare consistency
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: str = "full",
        seed: int = 42,
        config: Optional["ExperimentConfig"] = None,
    ):
        super().__init__(
            df=df,
            x_column=x_column,
            y_column=y_column or "y",  # BaseTask wants a y_column name; we can still ignore ys if none
            task_description=task_description,
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
            config=config,
        )
        self.dataset_reward_fn = dataset_reward_fn
        self.y_column = y_column
        self.y2_column = y2_column
        if y2_column is not None:
            self.y2s = self.df[y2_column].values.astype(str).tolist()
        else:
            self.y2s = [""] * len(self.xs)
        

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> List[float]:
        # If user didn't provide y_column, ignore ys (it will still be passed around by BaseTask)
        use_ys = ys if self.y_column is not None else None
        use_y2s = self.y2s if self.y2_column is not None else None

        R = float(self.dataset_reward_fn(xs, preds, use_ys, use_y2s))
        # Broadcast so Promptolution’s aggregation equals R
        return [R] * len(preds)