"""Callback classes for logging, saving, and tracking optimization progress."""


import os
from abc import ABC
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

if TYPE_CHECKING:
    from logging import Logger

    from promptolution.optimizers.base_optimizer import BaseOptimizer


class BaseCallback(ABC):
    """Base class for optimization callbacks.

    Callbacks can be used to monitor the optimization process, save checkpoints,
    log metrics, or implement early stopping criteria.

    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the callback with a configuration.

        Args:
            config: Configuration for the callback.
            **kwargs: Additional keyword arguments.
        """
        pass

    def on_step_end(self, optimizer: "BaseOptimizer") -> bool:
        """Called at the end of each optimization step.

        Args:
            optimizer: The optimizer object that called the callback.

        Returns:
            Bool: True if the optimization should continue, False if it should stop.
        """
        return True

    def on_epoch_end(self, optimizer: "BaseOptimizer") -> bool:
        """Called at the end of each optimization epoch.

        Args:
            optimizer: The optimizer object that called the callback.

        Returns:
            Bool: True if the optimization should continue, False if it should stop.
        """
        return True

    def on_train_end(self, optimizer: "BaseOptimizer") -> bool:
        """Called at the end of the entire optimization process.

        Args:
            optimizer: The optimizer object that called the callback.

        Returns:
            Bool: True if the optimization should continue, False if it should stop.
        """
        return True


class LoggerCallback(BaseCallback):
    """Callback for logging optimization progress.

    This callback logs information about each step, epoch, and the end of training.

    Attributes:
        logger: The logger object to use for logging.
        step (int): The current step number.
    """

    def __init__(self, logger: "Logger") -> None:
        """Initialize the LoggerCallback."""
        self.logger = logger
        self.step = 0

    def on_step_end(self, optimizer: "BaseOptimizer") -> bool:
        """Log information about the current step."""
        self.step += 1
        time = datetime.now().strftime("%d-%m-%y %H:%M:%S:%f")
        self.logger.critical(f"{time} - ✨ Step {self.step} ended ✨")
        for i, (prompt, score) in enumerate(zip(optimizer.prompts, optimizer.scores)):
            self.logger.critical(f"📝 Prompt {i}: Score: {score}")
            self.logger.critical(f"💬 {prompt}")

        return True

    def on_train_end(self, optimizer: "BaseOptimizer", logs: Optional[Any] = None) -> bool:
        """Log information at the end of training.

        Args:
        optimizer: The optimizer object that called the callback.
        logs: Additional information to log.
        """
        time = datetime.now().strftime("%d-%m-%y %H:%M:%S:%f")
        if logs is None:
            self.logger.critical(f"{time} - 🏁 Training ended")
        else:
            self.logger.critical(f"{time} - 🏁 Training ended - {logs}")

        return True


class FileOutputCallback(BaseCallback):
    """Callback for saving optimization progress to a specified file type.

    This callback saves information about each step to a file.

    Attributes:
        dir (str): Directory the file is saved to.
        step (int): The current step number.
        file_type (str): The type of file to save the output to.
    """

    def __init__(self, dir: str, file_type: Literal["parquet", "csv"] = "parquet") -> None:
        """Initialize the FileOutputCallback.

        Args:
        dir (str): Directory the CSV file is saved to.
        file_type (str): The type of file to save the output to.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.file_type = file_type

        if file_type == "parquet":
            self.path = os.path.join(dir, "step_results.parquet")
        elif file_type == "csv":
            self.path = os.path.join(dir, "step_results.csv")
        else:
            raise ValueError(f"File type {file_type} not supported.")

        self.step = 0

    def on_step_end(self, optimizer: "BaseOptimizer") -> bool:
        """Save prompts and scores to csv.

        Args:
        optimizer: The optimizer object that called the callback
        """
        self.step += 1
        df = pd.DataFrame(
            {
                "step": [self.step] * len(optimizer.prompts),
                "input_tokens": [optimizer.predictor.llm.input_token_count] * len(optimizer.prompts),
                "output_tokens": [optimizer.predictor.llm.output_token_count] * len(optimizer.prompts),
                "time": [datetime.now().timestamp()] * len(optimizer.prompts),
                "score": optimizer.scores,
                "prompt": [str(p) for p in optimizer.prompts],
            }
        )

        if self.file_type == "parquet":
            if self.step == 1:
                df.to_parquet(self.path, index=False)
            else:
                df.to_parquet(self.path, mode="a", index=False)
        elif self.file_type == "csv":
            if self.step == 1:
                df.to_csv(self.path, index=False)
            else:
                df.to_csv(self.path, mode="a", header=False, index=False)

        return True


class BestPromptCallback(BaseCallback):
    """Callback for tracking the best prompt during optimization.

    This callback keeps track of the prompt with the highest score.

    Attributes:
        best_prompt (str): The prompt with the highest score so far.
        best_score (float): The highest score achieved so far.
    """

    def __init__(self) -> None:
        """Initialize the BestPromptCallback."""
        self.best_prompt = ""
        self.best_score = -99999.0

    def on_step_end(self, optimizer: "BaseOptimizer") -> bool:
        """Update the best prompt and score if a new high score is achieved.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        if optimizer.scores[0] > self.best_score:
            self.best_score = optimizer.scores[0]
            self.best_prompt = optimizer.prompts[0]

        return True

    def get_best_prompt(self) -> Tuple[str, float]:
        """Get the best prompt and score achieved during optimization.

        Returns:
        Tuple[str, float]: The best prompt and score.
        """
        return self.best_prompt, self.best_score


class ProgressBarCallback(BaseCallback):
    """Callback for displaying a progress bar during optimization.

    This callback uses tqdm to display a progress bar that updates at each step.

    Attributes:
        pbar (tqdm): The tqdm progress bar object.
    """

    def __init__(self, total_steps: int) -> None:
        """Initialize the ProgressBarCallback.

        Args:
        total_steps (int): The total number of steps in the optimization process.
        """
        self.pbar = tqdm(total=total_steps)

    def on_step_end(self, optimizer: "BaseOptimizer") -> bool:
        """Update the progress bar at the end of each step.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        self.pbar.update(1)

        return True

    def on_train_end(self, optimizer: "BaseOptimizer") -> bool:
        """Close the progress bar at the end of training.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        self.pbar.close()

        return True


class TokenCountCallback(BaseCallback):
    """Callback for stopping optimization based on the total token count."""

    def __init__(
        self,
        max_tokens_for_termination: int,
        token_type_for_termination: Literal["input_tokens", "output_tokens", "total_tokens"],
    ) -> None:
        """Initialize the TokenCountCallback.

        Args:
        max_tokens_for_termination (int): Maximum number of tokens which is allowed befor the algorithm is stopped.
        token_type_for_termination (str): Can be one of either "input_tokens", "output_tokens" or "total_tokens".
        """
        self.max_tokens_for_termination = max_tokens_for_termination
        self.token_type_for_termination = token_type_for_termination

    def on_step_end(self, optimizer: "BaseOptimizer") -> bool:
        """Check if the total token count exceeds the maximum allowed. If so, stop the optimization."""
        token_counts = optimizer.predictor.llm.get_token_count()

        if token_counts[self.token_type_for_termination] > self.max_tokens_for_termination:
            return False

        return True
