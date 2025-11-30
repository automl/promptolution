"""Utility functions for prompt creation."""

import json

import numpy as np

from typing import TYPE_CHECKING, List, Optional, Union

from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.tasks.base_task import BaseTask

from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.templates import (
    PROMPT_CREATION_TEMPLATE,
    PROMPT_CREATION_TEMPLATE_FROM_TASK_DESCRIPTION,
    PROMPT_CREATION_TEMPLATE_TD,
    PROMPT_VARIATION_TEMPLATE,
    default_prompts,
)

logger = get_logger(__name__)


def create_prompt_variation(
    prompt: Union[List[str], str], llm: "BaseLLM", meta_prompt: Optional[str] = None
) -> List[str]:
    """Generate a variation of the given prompt(s) while keeping the semantic meaning.

    Idea taken from the paper Zhou et al. (2021) https://arxiv.org/pdf/2211.01910

    Args:
        prompt (Union[List[str], str]): The prompt(s) to generate variations of.
        llm (BaseLLM): The language model to use for generating the variations.
        meta_prompt (str): The meta prompt to use for generating the variations.
        If None, a default meta prompt is used. Should contain <prev_prompt> tag.

    Returns:
        List[str]: A list of generated variations of the input prompt(s).
    """
    meta_prompt = PROMPT_VARIATION_TEMPLATE if meta_prompt is None else meta_prompt

    if isinstance(prompt, str):
        prompt = [prompt]
    varied_prompts = llm.get_response([meta_prompt.replace("<prev_prompt>", p) for p in prompt])
    varied_prompts = extract_from_tag(varied_prompts, "<prompt>", "</prompt>")

    return varied_prompts


def create_prompts_from_samples(
    task: "BaseTask",
    llm: "BaseLLM",
    meta_prompt: Optional[str] = None,
    n_samples: int = 3,
    task_description: Optional[str] = None,
    n_prompts: int = 1,
    get_uniform_labels: bool = False,
) -> List[str]:
    """Generate a set of prompts from dataset examples sampled from a given task.

    Idea taken from the paper Zhou et al. (2021) https://arxiv.org/pdf/2211.01910
    Samples are selected, such that
    (1) all possible classes are represented
    (2) the samples are as representative as possible

    Args:
        task (BaseTask): The task to generate prompts for.
        Xs and Ys from this object are used to generate the prompts.
        llm (BaseLLM): The language model to use for generating the prompts.
        meta_prompt (str): The meta prompt to use for generating the prompts.
        If None, a default meta prompt is used.
        n_samples (int): The number of samples to use for generating prompts.
        task_description (str): The description of the task to include in the prompt.
        n_prompts (int): The number of prompts to generate.
        get_uniform_labels (bool): If True, samples are selected such that all classes are represented.

    Returns:
        List[str]: A list of generated prompts.
    """
    meta_prompt_template: str
    if meta_prompt is None:
        if task_description is None:
            meta_prompt_template = PROMPT_CREATION_TEMPLATE
        else:
            meta_prompt_template = PROMPT_CREATION_TEMPLATE_TD.replace("<task_desc>", task_description)
    else:
        if task_description is None:
            meta_prompt_template = meta_prompt
        else:
            meta_prompt_template = meta_prompt.replace("<task_desc>", task_description)

    meta_prompts = []
    for _ in range(n_prompts):
        if isinstance(task, ClassificationTask) and get_uniform_labels:
            # if classification task sample such that all classes are represented
            unique_labels, counts = np.unique(task.ys, return_counts=True)
            proportions = counts / len(task.ys)
            samples_per_class = np.round(proportions * n_samples).astype(int)
            samples_per_class = np.maximum(samples_per_class, 1)

            # sample
            xs: List[str] = []
            ys: List[str] = []
            for label, n_per_class in zip(unique_labels, samples_per_class):
                indices = np.where(task.ys == label)[0]
                indices = np.random.choice(indices, n_per_class, replace=False)
                xs.extend(task.xs[indices])
                ys.extend(task.ys[indices])

        else:
            # if not classification task, sample randomly
            indices = np.random.choice(len(task.xs), n_samples, replace=False)
            xs = [task.xs[i] for i in indices]
            ys = [task.ys[i] for i in indices]

        examples = "\n\n".join([f"Input: {x}\nOutput: {y}" for x, y in zip(xs, ys)])
        meta_prompt = meta_prompt_template.replace("<input_output_pairs>", examples)
        meta_prompts.append(meta_prompt)

    prompts = llm.get_response(meta_prompts)
    prompts = extract_from_tag(prompts, "<prompt>", "</prompt>")

    return prompts


def create_prompts_from_task_description(
    task_description: str,
    llm: "BaseLLM",
    meta_prompt: Optional[str] = None,
    n_prompts: int = 10,
    n_retries: int = 3,
) -> List[str]:
    """Generate a set of prompts from a given task description.

    Args:
        task_description (str): The description of the task to generate prompts for.
        llm (BaseLLM): The language model to use for generating the prompts.
        meta_prompt (str): The meta prompt to use for generating the prompts.
            If None, a default meta prompt is used.
        n_prompts (int): The number of prompts to generate.
        n_retries (int): The number of retries to attempt if prompt generation fails.
    """
    if meta_prompt is None:
        meta_prompt = PROMPT_CREATION_TEMPLATE_FROM_TASK_DESCRIPTION

    meta_prompt = meta_prompt.replace("<task_desc>", task_description).replace("<n_prompts>", str(n_prompts))
    final_prompts = None
    for _ in range(n_retries):
        prompts_str = llm.get_response(meta_prompt)[0]
        try:
            prompts = json.loads(prompts_str)
            assert isinstance(prompts, list) and all(isinstance(p, str) for p in prompts) and len(prompts) == n_prompts
            final_prompts = prompts
            break
        except (json.JSONDecodeError, AssertionError):
            logger.warning("Failed to parse prompts JSON, retrying...")

    if final_prompts is None:
        logger.error(
            f"Failed to generate prompts from task description after {n_retries} retries, returning default prompts."
        )
        final_prompts = default_prompts[:n_prompts]

    return final_prompts
