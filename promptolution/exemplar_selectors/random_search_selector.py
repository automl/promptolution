"""Random search exemplar selector."""

import random

from promptolution.exemplar_selectors.base_exemplar_selector import BaseExemplarSelector
from promptolution.utils.prompt import Prompt


class RandomSearchSelector(BaseExemplarSelector):
    """A selector that uses random search to find the best set of exemplars.

    This class implements a strategy that generates multiple sets of random examples,
    evaluates their performance, and selects the best performing set.
    """

    def select_exemplars(self, prompt: Prompt, n_examples: int = 5, n_trials: int = 5) -> Prompt:
        """Select exemplars using a random search strategy.

        This method generates multiple sets of random examples, evaluates their performance
        when combined with the original prompt, and returns the best performing set.

        Args:
            prompt (Prompt): The input prompt to base the exemplar selection on.
            n_examples (int, optional): The number of exemplars to select. Defaults to 5.
            n_trials (int, optional): The number of random trials to perform. Defaults to 5.

        Returns:
            Prompt: The best performing prompt, which includes the original prompt and the selected exemplars.
        """
        best_score = 0.0
        best_prompt = prompt

        for _ in range(n_trials):
            result = self.task.evaluate(prompt, self.predictor, eval_strategy="subsample")
            seq = result.sequences[0]
            examples = random.sample(list(seq), n_examples)
            prompt_with_examples = Prompt(prompt.instruction, examples)
            # evaluate prompts as few shot prompt
            result = self.task.evaluate(prompt_with_examples, self.predictor, eval_strategy="subsample")
            score = float(result.agg_scores[0])
            if score > best_score:
                best_score = score
                best_prompt = prompt_with_examples

        return best_prompt
