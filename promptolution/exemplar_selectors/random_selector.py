"""Random exemplar selector."""

import numpy as np

from typing import TYPE_CHECKING, List, Optional

from promptolution.exemplar_selectors.base_exemplar_selector import BaseExemplarSelector
from promptolution.utils.prompt import Prompt

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig


class RandomSelector(BaseExemplarSelector):
    """A selector that randomly selects correct exemplars.

    This class implements a strategy that generates random examples and selects
    those that are evaluated as correct until the desired number of exemplars is reached.
    """

    def __init__(
        self,
        task: "BaseTask",
        predictor: "BasePredictor",
        desired_score: int = 1,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the RandomSelector.

        Args:
            task (BaseTask): An object representing the task to be performed.
            predictor (BasePredictor): An object capable of making predictions based on prompts.
            desired_score (int, optional): The desired score for the exemplars. Defaults to 1.
            config (ExperimentConfig, optional): Configuration for the selector, overriding defaults.
        """
        self.desired_score = desired_score
        super().__init__(task, predictor, config)

    def select_exemplars(self, prompt: Prompt, n_examples: int = 5) -> Prompt:
        """Select exemplars using a random selection strategy.

        This method generates random examples and selects those that are evaluated as correct
        (score == self.desired_score) until the desired number of exemplars is reached.

        Args:
            prompt (Prompt): The input prompt to base the exemplar selection on.
            n_examples (int, optional): The number of exemplars to select. Defaults to 5.

        Returns:
            Prompt: A new prompt that includes the original prompt and the selected exemplars.
        """
        examples: List[str] = []
        while len(examples) < n_examples:
            result = self.task.evaluate(prompt, self.predictor, eval_strategy="subsample")
            scores = result.scores
            seqs = result.sequences
            score = float(np.mean(scores))
            seq = seqs[0][0]
            if score == self.desired_score:
                examples.append(seq)
        return Prompt(prompt.instruction, examples)
