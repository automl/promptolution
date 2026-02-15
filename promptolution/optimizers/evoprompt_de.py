"""Module for EvoPromptDE optimizer."""


import random

from typing import TYPE_CHECKING, List, Optional

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt, sort_prompts_by_scores
from promptolution.utils.templates import EVOPROMPT_DE_TEMPLATE_TD

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.utils.config import ExperimentConfig


class EvoPromptDE(BaseOptimizer):
    """EvoPromptDE: Differential Evolution-based Prompt Optimizer.

    This class implements a differential evolution algorithm for optimizing prompts in large language models.
    It is adapted from the paper "Connecting Large Language Models with Evolutionary Algorithms
    Yields Powerful Prompt Optimizers" by Guo et al., 2023.

    The optimizer uses a differential evolution strategy to generate new prompts from existing ones,
    with an option to use the current best prompt as a donor.

    Attributes:
        prompt_template (str): Template for generating meta-prompts during evolution.
        donor_random (bool): If False, uses the current best prompt as a donor; if True, uses a random prompt.
        meta_llm: Language model used for generating child prompts from meta-prompts.

    Args:
        prompt_template (str): Template for meta-prompts.
        meta_llm: Language model for child prompt generation.
        donor_random (bool, optional): Whether to use a random donor. Defaults to False.
        config (ExperimentConfig, optional): Configuration for the optimizer, overriding defaults.
    """

    def __init__(
        self,
        predictor: "BasePredictor",
        task: "BaseTask",
        meta_llm: "BaseLLM",
        initial_prompts: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
        donor_random: bool = False,
        callbacks: Optional[List["BaseCallback"]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the EvoPromptDE optimizer."""
        self.donor_random = donor_random
        self.meta_llm = meta_llm
        super().__init__(
            predictor=predictor, task=task, initial_prompts=initial_prompts, callbacks=callbacks, config=config
        )
        self.prompt_template = self._initialize_meta_template(prompt_template or EVOPROMPT_DE_TEMPLATE_TD)

    def _pre_optimization_loop(self) -> None:
        result = self.task.evaluate(self.prompts, self.predictor)
        self.scores = result.agg_scores.tolist()
        self.prompts, self.scores = sort_prompts_by_scores(self.prompts, self.scores)

    def _step(self) -> List[Prompt]:
        """Perform the optimization process for a specified number of steps.

        This method iteratively improves the prompts using a differential evolution strategy.
        It evaluates prompts, generates new prompts using the DE algorithm, and replaces
        prompts if the new ones perform better.


        Returns:
            List[Prompt]: The optimized list of prompts after all steps.
        """
        cur_best = self.prompts[0]
        meta_prompts = []
        for i in range(len(self.prompts)):
            # create meta prompts
            old_prompt = self.prompts[i]

            candidates = [prompt for prompt in self.prompts if prompt != old_prompt]
            a, b, c = random.sample(candidates, k=3)

            if not self.donor_random:
                c = cur_best

            meta_prompt = (
                self.prompt_template.replace("<prompt0>", old_prompt.construct_prompt())
                .replace("<prompt1>", a.construct_prompt())
                .replace("<prompt2>", b.construct_prompt())
                .replace("<prompt3>", c.construct_prompt())
            )

            meta_prompts.append(meta_prompt)

        child_instructions = self.meta_llm.get_response(meta_prompts)
        child_instructions = extract_from_tag(child_instructions, "<prompt>", "</prompt>")
        child_prompts = [Prompt(p) for p in child_instructions]

        child_result = self.task.evaluate(child_prompts, self.predictor)
        child_scores = child_result.agg_scores.tolist()

        for i in range(len(self.prompts)):
            if child_scores[i] > self.scores[i]:
                self.prompts[i] = child_prompts[i]
                self.scores[i] = child_scores[i]

        self.prompts, self.scores = sort_prompts_by_scores(self.prompts, self.scores)

        return self.prompts
