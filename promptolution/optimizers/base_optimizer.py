"""Base module for optimizers in the promptolution library."""


from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, List, Literal, Optional

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.tasks.base_task import BaseTask
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.utils.config import ExperimentConfig
    from promptolution.utils.callbacks import BaseCallback

from promptolution.utils.logging import get_logger

logger = get_logger(__name__)

OptimizerType = Literal["evopromptde", "evopromptga", "opro", "capo"]


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    This class defines the basic structure and interface for prompt optimization algorithms.

    Attributes:
        config (ExperimentConfig, optional): Configuration for the optimizer, overriding defaults.
        prompts (List[str]): List of current prompts being optimized.
        task (BaseTask): The task object for evaluating prompts.
        callbacks (List[Callable]): List of callback functions to be called during optimization.
        predictor: The predictor used for prompt evaluation (if applicable).
    """

    def __init__(
        self,
        predictor: "BasePredictor",
        task: "BaseTask",
        initial_prompts: Optional[List[str]] = None,
        callbacks: Optional[List["BaseCallback"]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the optimizer with a configuration and/or direct parameters.

        Args:
            task: Task object for prompt evaluation.
            predictor: Predictor for prompt evaluation.
            initial_prompts: Initial set of prompts to start optimization with.
            callbacks: List of callback functions.
            config (ExperimentConfig, optional): Configuration for the optimizer, overriding defaults.
        """
        # Set up optimizer state
        self.prompts: List[str] = initial_prompts or []
        self.task = task
        self.callbacks: List["BaseCallback"] = callbacks or []
        self.predictor = predictor
        self.scores: List[float] = []

        if config is not None:
            config.apply_to(self)

        self.config = config

    def optimize(self, n_steps: int) -> List[str]:
        """Perform the optimization process.

        This method should be implemented by concrete optimizer classes to define
        the specific optimization algorithm.

        Args:
            n_steps (int): Number of optimization steps to perform.

        Returns:
            The optimized list of prompts after all steps.
        """
        # validate config
        if self.config is not None:
            self.config.validate()
        self._pre_optimization_loop()

        for _ in range(n_steps):
            try:
                self.prompts = self._step()
            except Exception as e:
                # exit training loop and gracefully fail
                logger.error("⛔ Error during optimization step! ⚠️ Exiting optimization loop.", exc_info=e)
                break

            # Callbacks at the end of each step
            continue_optimization = self._on_step_end()
            if not continue_optimization:
                break

        self._on_train_end()

        return self.prompts

    @abstractmethod
    def _pre_optimization_loop(self) -> None:
        """Prepare for the optimization loop.

        This method should be implemented by concrete optimizer classes to define
        any setup required before the optimization loop starts.
        """
        pass

    @abstractmethod
    def _step(self) -> List[str]:
        """Perform a single optimization step.

        This method should be implemented by concrete optimizer classes to define
        the specific optimization step.

        Returns:
            The optimized list of prompts after the step.
        """
        pass

    def _on_step_end(self) -> bool:
        """Call all registered callbacks at the end of each optimization step."""
        continue_optimization = True
        for callback in self.callbacks:
            if not callback.on_step_end(self):
                continue_optimization = False

        return continue_optimization

    def _on_train_end(self) -> None:
        """Call all registered callbacks at the end of the entire optimization process."""
        for callback in self.callbacks:
            callback.on_train_end(self)
