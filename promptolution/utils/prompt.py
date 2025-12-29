"""Module defining the Prompt class and related utilities."""

import numpy as np

from typing import List, Optional, Sequence, Tuple, Union

from promptolution.utils.templates import DOWNSTREAM_TEMPLATE, DOWNSTREAM_TEMPLATE_W_FEWSHOTS


class Prompt:
    """Represent a prompt consisting of an instruction and few-shot examples."""

    def __init__(
        self, instruction: str, few_shots: Optional[List[str]] = None, downstream_template: Optional[str] = None
    ) -> None:
        """Initialize the Prompt with an instruction and associated examples.

        Args:
            instruction (str): The instruction or prompt text.
            few_shots (List[str]): List of examples as string.
            downstream_template (str, optional): Template for formatting the full prompt.
        """
        self.instruction = instruction.strip()
        self.few_shots = few_shots or []
        if downstream_template is None:
            if self.few_shots:
                downstream_template = DOWNSTREAM_TEMPLATE_W_FEWSHOTS
            else:
                downstream_template = DOWNSTREAM_TEMPLATE
        self.downstream_template = downstream_template

    def construct_prompt(self) -> str:
        """Construct the full prompt string by replacing placeholders in the template with the instruction and formatted examples.

        Returns:
            str: The constructed prompt string.
        """
        few_shot_str = "\n\n".join(self.few_shots).strip()
        prompt = (
            self.downstream_template.replace("<instruction>", self.instruction)
            .replace("<few_shots>", few_shot_str)
            .replace("\n\n\n\n", "\n\n")  # replace extra newlines if no few shots are provided
            .strip()
        )
        return prompt

    def __str__(self) -> str:
        """Return the string representation of the prompt."""
        return self.construct_prompt()


def sort_prompts_by_scores(
    prompts: List[Prompt], scores: Union[Sequence[float], np.ndarray], top_k: Optional[int] = None
) -> Tuple[List[Prompt], List[float]]:
    """Sort prompts by score, accepting scalar, 1D, or multi-dimensional scores.

    Scores can be provided as Python lists or NumPy arrays. If scores are multi-
    dimensional (e.g., per-subsample results), they are aggregated with a
    ``nanmean`` across all non-leading axes before sorting.

    Args:
        prompts (List[Prompt]): Prompt objects to sort.
        scores (Sequence[float] | np.ndarray): Corresponding scores; can be nested lists or arrays.
        top_k (Optional[int]): Limit the result to the top_k prompts.

    Returns:
        Tuple[List[Prompt], List[float]]: Prompts and their aggregated scores,
        sorted in descending order.
    """
    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.ndim == 0:
        scores_arr = scores_arr.reshape(1)

    assert scores_arr.shape[0] == len(prompts), "Prompts and scores must have the same length."

    if scores_arr.ndim > 1:
        axes_to_reduce = tuple(range(1, scores_arr.ndim))
        scores_arr = np.nanmean(scores_arr, axis=axes_to_reduce)

    prompt_score_pairs = list(zip(prompts, scores_arr.tolist()))
    prompt_score_pairs.sort(key=lambda pair: pair[1], reverse=True)

    if top_k is not None:
        prompt_score_pairs = prompt_score_pairs[:top_k]

    sorted_prompts = [p for p, _ in prompt_score_pairs]
    sorted_scores = [s for _, s in prompt_score_pairs]

    return sorted_prompts, sorted_scores
