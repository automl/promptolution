"""Base class for exemplar selectors."""


from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Optional, Union

from promptolution.utils.prompt import Prompt

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig


class BaseExemplarSelector(ABC):
    """An abstract base class for exemplar selectors.

    This class defines the basic interface and common functionality
    that all exemplar selectors should implement.
    """

    def __init__(self, task: "BaseTask", predictor: "BasePredictor", config: Optional["ExperimentConfig"] = None):
        """Initialize the BaseExemplarSelector.

        Args:
            task (BaseTask): An object representing the task to be performed.
            predictor (BasePredictor): An object capable of making predictions based on prompts.
            config (ExperimentConfig, optional): "ExperimentConfig" overwriting the defaults
        """
        self.task = task
        self.predictor = predictor

        if config is not None:
            config.apply_to(self)

    @abstractmethod
    def select_exemplars(self, prompt: Union[str, Prompt], n_examples: int = 5) -> Prompt:
        """Select exemplars based on the given prompt.

        Args:
            prompt (Union[str, Prompt]): The input prompt to base the exemplar selection on. A raw string is coerced to a Prompt.
            n_examples (int, optional): The number of exemplars to select. Defaults to 5.

        Returns:
            str: A new prompt that includes the original prompt and the selected exemplars.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
