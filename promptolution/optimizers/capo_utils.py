"""Shared utilities for CAPO-style optimizers."""

from __future__ import annotations

import random
from typing import Callable, List

import pandas as pd

from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt


def build_few_shot_examples(
    instruction: str,
    num_examples: int,
    df_few_shots: pd.DataFrame,
    task,
    predictor,
    fewshot_template: str,
    target_begin_marker: str,
    target_end_marker: str,
    check_fs_accuracy: bool,
    create_fs_reasoning: bool,
) -> List[str]:
    """Create few-shot examples with optional reasoning replacement."""
    if num_examples == 0:
        return []

    few_shot_samples = df_few_shots.sample(num_examples, replace=False)
    sample_inputs = few_shot_samples[task.x_column].values.astype(str)
    sample_targets = few_shot_samples[task.y_column].values
    few_shots = [
        fewshot_template.replace("<input>", i).replace("<output>", f"{target_begin_marker}{t}{target_end_marker}")
        for i, t in zip(sample_inputs, sample_targets)
    ]

    if not create_fs_reasoning:
        return few_shots

    preds, seqs = predictor.predict(
        [instruction] * num_examples,
        list(sample_inputs),
        return_seq=True,
    )
    if isinstance(seqs, str):
        seqs = [seqs]
    if isinstance(preds, str):
        preds = [preds]

    for j in range(num_examples):
        seqs[j] = seqs[j].replace(sample_inputs[j], "", 1).strip()
        if preds[j] == sample_targets[j] or not check_fs_accuracy:
            few_shots[j] = fewshot_template.replace("<input>", sample_inputs[j]).replace("<output>", seqs[j])

    return few_shots


def perform_crossover(
    parents: List[Prompt],
    crossovers_per_iter: int,
    template: str,
    meta_llm,
) -> List[Prompt]:
    """Generate crossover offspring prompts."""
    crossover_prompts: List[str] = []
    offspring_few_shots: List[List[str]] = []
    for _ in range(crossovers_per_iter):
        mother, father = (parents if len(parents) == 2 else random.sample(parents, 2))
        crossover_prompt = template.replace("<mother>", mother.instruction).replace("<father>", father.instruction).strip()
        crossover_prompts.append(crossover_prompt)
        combined_few_shots = mother.few_shots + father.few_shots
        num_few_shots = (len(mother.few_shots) + len(father.few_shots)) // 2
        offspring_few_shot = random.sample(combined_few_shots, num_few_shots) if combined_few_shots else []
        offspring_few_shots.append(offspring_few_shot)

    child_instructions = meta_llm.get_response(crossover_prompts)
    return [
        Prompt(extract_from_tag(instr, "<prompt>", "</prompt>"), examples)
        for instr, examples in zip(child_instructions, offspring_few_shots)
    ]


def perform_mutation(
    offsprings: List[Prompt],
    mutation_template: str,
    create_few_shots: Callable[[str, int], List[str]],
    upper_shots: int,
    meta_llm,
) -> List[Prompt]:
    """Mutate offspring prompts."""
    mutation_prompts = [mutation_template.replace("<instruction>", prompt.instruction) for prompt in offsprings]
    new_instructions = meta_llm.get_response(mutation_prompts)

    mutated: List[Prompt] = []
    for new_instruction, prompt in zip(new_instructions, offsprings):
        new_instruction = extract_from_tag(new_instruction, "<prompt>", "</prompt>")
        p = random.random()

        if p < 1 / 3 and len(prompt.few_shots) < upper_shots:
            new_few_shot = create_few_shots(new_instruction, 1)
            new_few_shots = prompt.few_shots + new_few_shot
        elif 1 / 3 <= p < 2 / 3 and len(prompt.few_shots) > 0:
            new_few_shots = random.sample(prompt.few_shots, len(prompt.few_shots) - 1)
        else:
            new_few_shots = prompt.few_shots

        random.shuffle(new_few_shots)
        mutated.append(Prompt(new_instruction, new_few_shots))

    return mutated
