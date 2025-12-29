"""Module implementing the OPRO (Optimization by PROmpting) algorithm."""


import numpy as np

from typing import TYPE_CHECKING, List, Optional

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt, sort_prompts_by_scores
from promptolution.utils.templates import OPRO_TEMPLATE

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.utils.config import ExperimentConfig


class OPRO(BaseOptimizer):
    """OPRO: Optimization by PROmpting.

    Implementation of the technique proposed in "Large Language Models as Optimizers"
    (Yang et al., 2023: https://arxiv.org/abs/2309.03409).

    OPRO works by providing a meta-LLM with task descriptions and previous
    prompt-score pairs to generate improved prompts for a downstream LLM.
    """

    def __init__(
        self,
        predictor: "BasePredictor",
        task: "BaseTask",
        meta_llm: "BaseLLM",
        initial_prompts: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
        max_num_instructions: int = 20,
        num_instructions_per_step: int = 8,
        num_few_shots: int = 3,
        callbacks: Optional[List["BaseCallback"]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the OPRO optimizer.

        Args:
            predictor: Predictor for prompt evaluation
            task: Task object for prompt evaluation
            meta_llm: LLM that generates improved prompts
            initial_prompts: Initial set of prompts to start optimization with
            prompt_template: Custom meta prompt template (uses OPRO_TEMPLATE if None)
            max_num_instructions: Maximum previous instructions to include in meta prompt
            num_instructions_per_step: Number of prompts to generate in each step
            num_few_shots: Number of few-shot examples to include (0 for none)
            callbacks: List of callback functions
            config: "ExperimentConfig" overwriting default parameters
        """
        self.meta_llm = meta_llm
        self.max_num_instructions = max_num_instructions
        self.num_instructions_per_step = num_instructions_per_step
        self.num_few_shots = num_few_shots
        super().__init__(
            predictor=predictor, task=task, initial_prompts=initial_prompts, callbacks=callbacks, config=config
        )
        self.meta_prompt_template = self._initialize_meta_template(prompt_template or OPRO_TEMPLATE)

    def _sample_examples(self) -> str:
        """Sample few-shot examples from the dataset.

        Returns:
            Formatted string of few-shot examples with inputs and expected outputs
        """
        idx = np.random.choice(len(self.task.xs), self.num_few_shots)
        sample_x = [self.task.xs[i] for i in idx]
        sample_y = [self.task.ys[i] for i in idx]

        return "\n".join([f"Input: {x}\nOutput: {y}" for x, y in zip(sample_x, sample_y)])

    def _format_instructions(self) -> str:
        """Format previous prompts and their scores for the meta prompt.

        Returns:
            Formatted string of previous prompts and their scores,
            sorted by ascending score (worse to better)
        """
        prompt_score_pairs = list(zip(self.prompts, self.scores))
        sorted_pairs = sorted(prompt_score_pairs, key=lambda x: x[1])

        return "".join([f"text:\n{prompt}\nscore: {int(100 * round(score, 2))}\n\n" for prompt, score in sorted_pairs])

    def _add_prompt_and_score(self, prompt: Prompt, score: float) -> None:
        """Add a prompt and its score to the lists, maintaining max length.

        Args:
            prompt: The prompt to add
            score: The corresponding score for the prompt
        """
        if prompt in self.prompts:
            return

        self.prompts.append(prompt)
        self.scores.append(score)

        # Keep only the top-performing prompts if we exceed the maximum number of instructions
        self.prompts, self.scores = sort_prompts_by_scores(self.prompts, self.scores, top_k=self.max_num_instructions)

    def _pre_optimization_loop(self):
        result = self.task.evaluate(self.prompts, self.predictor)
        self.scores = result.agg_scores.tolist()
        self.meta_prompt = self.meta_prompt_template.replace("<instructions>", self._format_instructions()).replace(
            "<examples>", self._sample_examples()
        )

    def _step(self) -> List[Prompt]:
        duplicate_prompts = 0
        for _ in range(self.num_instructions_per_step):
            generation_seed = np.random.randint(0, int(1e9))
            self.meta_llm.set_generation_seed(generation_seed)

            response = self.meta_llm.get_response([self.meta_prompt])[0]

            instruction = extract_from_tag(response, "<prompt>", "</prompt>")
            prompt = Prompt(instruction)

            if prompt in self.prompts:
                duplicate_prompts += 1
                continue

            prompt_result = self.task.evaluate([prompt], self.predictor)
            score = prompt_result.agg_scores.tolist()[0]

            self._add_prompt_and_score(prompt, score)

        # Update meta prompt
        self.meta_prompt = self.meta_prompt_template.replace("<instructions>", self._format_instructions()).replace(
            "<examples>", self._sample_examples()
        )

        return self.prompts
