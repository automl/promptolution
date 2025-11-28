"""Module defining the Prompt class and related utilities."""

from typing import List, Optional, Tuple

from promptolution.utils.templates import DOWNSTREAM_TEMPLATE, DOWNSTREAM_TEMPLATE_W_FEWSHOTS


class Prompt:
    """Represents a prompt consisting of an instruction and few-shot examples."""

    def __init__(
        self, instruction: str, few_shots: Optional[List[str]] = None, downstream_template: Optional[str] = None
    ) -> None:
        """Initializes the Prompt with an instruction and associated examples.

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
        """Constructs the full prompt string by replacing placeholders in the template with the instruction and formatted examples.

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
        """Returns the string representation of the prompt."""
        return self.construct_prompt()


def sort_prompts_by_scores(
    prompts: List[Prompt], scores: List[float], top_k: Optional[int] = None
) -> Tuple[List[Prompt], List[float]]:
    """Sorts prompts based on their associated scores in descending order.

    Args:
        prompts (List[Prompt]): List of Prompt objects.
        scores (List[float]): Corresponding list of scores.
        top_k (Optional[int]): If provided, limits the result to the top_k prompts. Defaults to None (returns all).

    Returns:
        Tuple[List[Prompt], List[float]]: A tuple containing prompts sorted by scores in descending order and their corresponding sorted scores.
    """
    assert len(prompts) == len(scores), "Prompts and scores must have the same length."

    sorted_prompts = [prompt for score, prompt in sorted(zip(scores, prompts), reverse=True, key=lambda x: x[0])]
    sorted_scores = sorted(scores, reverse=True)

    if top_k is not None:
        sorted_prompts = sorted_prompts[:top_k]
        sorted_scores = sorted_scores[:top_k]

    return sorted_prompts, sorted_scores
