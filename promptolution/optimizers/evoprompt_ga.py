"""Module for EvoPromptGA optimizer."""


import numpy as np

from typing import TYPE_CHECKING, List, Optional

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.prompt import Prompt, sort_prompts_by_scores
from promptolution.utils.templates import EVOPROMPT_GA_TEMPLATE_TD

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.utils.config import ExperimentConfig

from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.logging import get_logger

logger = get_logger(__name__)


class EvoPromptGA(BaseOptimizer):
    """EvoPromptGA: Genetic Algorithm-based Prompt Optimizer.

    This class implements a genetic algorithm for optimizing prompts in large language models.
    It is adapted from the paper "Connecting Large Language Models with Evolutionary Algorithms
    Yields Powerful Prompt Optimizers" by Guo et al., 2023.

    The optimizer uses crossover operations to generate new prompts from existing ones,
    with different selection methods available for choosing parent prompts.

    Attributes:
        prompt_template (str): Template for generating meta-prompts during crossover.
        meta_llm: Language model used for generating child prompts from meta-prompts.
        selection_mode (str): Method for selecting parent prompts ('random', 'wheel', or 'tour').

    Args:
        prompt_template (str): Template for meta-prompts.
        meta_llm: Language model for child prompt generation.
        selection_mode (str, optional): Parent selection method. Defaults to "wheel".

    Raises:
        AssertionError: If an invalid selection mode is provided.
    """

    def __init__(
        self,
        predictor: "BasePredictor",
        task: "BaseTask",
        meta_llm: "BaseLLM",
        initial_prompts: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
        selection_mode: str = "wheel",
        callbacks: Optional[List["BaseCallback"]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the EvoPromptGA optimizer."""
        self.meta_llm = meta_llm
        self.selection_mode = selection_mode
        super().__init__(
            predictor=predictor, initial_prompts=initial_prompts, task=task, callbacks=callbacks, config=config
        )
        self.prompt_template = self._initialize_meta_template(prompt_template or EVOPROMPT_GA_TEMPLATE_TD)

        assert self.selection_mode in ["random", "wheel", "tour"], "Invalid selection mode."

    def _pre_optimization_loop(self) -> None:
        self.scores = self.task.evaluate(self.prompts, self.predictor, return_agg_scores=True)
        self.prompts, self.scores = sort_prompts_by_scores(self.prompts, self.scores)

    def _step(self) -> List[Prompt]:
        new_prompts = self._crossover(self.prompts, self.scores)
        new_scores = self.task.evaluate(new_prompts, self.predictor, return_agg_scores=True)

        prompts = self.prompts + new_prompts
        scores = self.scores + new_scores

        self.prompts, self.scores = sort_prompts_by_scores(prompts, scores, top_k=len(self.prompts))

        return self.prompts

    def _crossover(self, prompts: List[Prompt], scores: List[float]) -> List[Prompt]:
        """Perform crossover operation to generate new child prompts.

        This method selects parent prompts based on the chosen selection mode,
        creates meta-prompts using the prompt template, and generates new child
        prompts using the meta language model.

        Args:
            prompts (List[str]): List of current prompts.
            scores (List[float]): Corresponding scores for the prompts.

        Returns:
            List[str]: Newly generated child prompts.
        """
        # parent selection
        if self.selection_mode == "wheel":
            wheel_idx = np.random.choice(
                np.arange(0, len(prompts)),
                size=len(prompts),
                replace=True,
                p=np.array(scores) / np.sum(scores) if np.sum(scores) > 0 else np.ones(len(scores)) / len(scores),
            ).tolist()
            parent_pop = [self.prompts[idx] for idx in wheel_idx]

        elif self.selection_mode in ["random", "tour"]:
            parent_pop = self.prompts

        # crossover
        meta_prompts = []
        for _ in self.prompts:
            if self.selection_mode in ["random", "wheel"]:
                parent_1, parent_2 = np.random.choice(parent_pop, size=2, replace=False)
            elif self.selection_mode == "tour":
                group_1 = np.random.choice(parent_pop, size=2, replace=False)
                group_2 = np.random.choice(parent_pop, size=2, replace=False)
                # use the best of each group based on scores
                parent_1 = group_1[np.argmax([self.scores[self.prompts.index(p)] for p in group_1])]
                parent_2 = group_2[np.argmax([self.scores[self.prompts.index(p)] for p in group_2])]

            parent_1, parent_2 = parent_1.construct_prompt(), parent_2.construct_prompt()
            meta_prompt = self.prompt_template.replace("<prompt1>", parent_1).replace("<prompt2>", parent_2)
            meta_prompts.append(meta_prompt)

        child_instructions = self.meta_llm.get_response(meta_prompts)
        child_instructions = extract_from_tag(child_instructions, "<prompt>", "</prompt>")
        child_prompts = [Prompt(p) for p in child_instructions]

        return child_prompts
