"""Implementation of the CAPO (Cost-Aware Prompt Optimization) algorithm."""

import random
from itertools import compress

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from promptolution.utils.formatting import extract_from_tag

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig
    from promptolution.utils.test_statistics import TestStatistics

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt, sort_prompts_by_scores
from promptolution.utils.templates import CAPO_CROSSOVER_TEMPLATE, CAPO_FEWSHOT_TEMPLATE, CAPO_MUTATION_TEMPLATE
from promptolution.utils.test_statistics import get_test_statistic_func
from promptolution.utils.token_counter import get_token_counter

logger = get_logger(__name__)


class CAPO(BaseOptimizer):
    """CAPO: Cost-Aware Prompt Optimization.

    This class implements an evolutionary algorithm for optimizing prompts in large language models
    by incorporating racing techniques and multi-objective optimization. It uses crossover, mutation,
    and racing based on evaluation scores and statistical tests to improve efficiency while balancing
    performance with prompt length. It is adapted from the paper "CAPO: Cost-Aware Prompt Optimization" by Zehle et al., 2025.
    """

    def __init__(
        self,
        predictor: "BasePredictor",
        task: "BaseTask",
        meta_llm: "BaseLLM",
        initial_prompts: Optional[List[str]] = None,
        crossover_template: Optional[str] = None,
        mutation_template: Optional[str] = None,
        crossovers_per_iter: int = 4,
        upper_shots: int = 5,
        max_n_blocks_eval: int = 10,
        test_statistic: "TestStatistics" = "paired_t_test",
        alpha: float = 0.2,
        length_penalty: float = 0.05,
        check_fs_accuracy: bool = True,
        create_fs_reasoning: bool = True,
        df_few_shots: Optional[pd.DataFrame] = None,
        callbacks: Optional[List["BaseCallback"]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initializes the CAPOptimizer with various parameters for prompt evolution.

        Args:
            predictor (BasePredictor): The predictor for evaluating prompt performance.
            task (BaseTask): The task instance containing dataset and description.
            meta_llm (BaseLLM): The meta language model for crossover/mutation.
            initial_prompts (List[str]): Initial prompt instructions.
            crossover_template (str, optional): Template for crossover instructions.
            mutation_template (str, optional): Template for mutation instructions.
            crossovers_per_iter (int): Number of crossover operations per iteration.
            upper_shots (int): Maximum number of few-shot examples per prompt.
            p_few_shot_reasoning (float): Probability of generating llm-reasoning for few-shot examples, instead of simply using input-output pairs.
            max_n_blocks_eval (int): Maximum number of evaluation blocks.
            test_statistic (TestStatistics): Statistical test to compare prompt performance. Default is "paired_t_test".
            alpha (float): Significance level for the statistical test.
            length_penalty (float): Penalty factor for prompt length.
            check_fs_accuracy (bool): Whether to check the accuracy of few-shot examples before appending them to the prompt.
                In cases such as reward tasks, this can be set to False, as no ground truth is available. Default is True.
            create_fs_reasoning (bool): Whether to create reasoning for few-shot examples using the downstream model,
                instead of simply using input-output pairs from the few shots DataFrame. Default is True.
            df_few_shots (pd.DataFrame): DataFrame containing few-shot examples. If None, will pop 10% of datapoints from task.
            callbacks (List[Callable], optional): Callbacks for optimizer events.
            config (ExperimentConfig, optional): Configuration for the optimizer.
        """
        self.meta_llm = meta_llm
        self.downstream_llm = predictor.llm

        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.max_n_blocks_eval = max_n_blocks_eval
        self.test_statistic = get_test_statistic_func(test_statistic)
        self.alpha = alpha

        self.length_penalty = length_penalty
        self.token_counter = get_token_counter(self.downstream_llm)

        self.check_fs_accuracy = check_fs_accuracy
        self.create_fs_reasoning = create_fs_reasoning

        super().__init__(predictor, task, initial_prompts, callbacks, config)

        self.crossover_template = self._initialize_meta_template(crossover_template or CAPO_CROSSOVER_TEMPLATE)
        self.mutation_template = self._initialize_meta_template(mutation_template or CAPO_MUTATION_TEMPLATE)

        self.df_few_shots = df_few_shots if df_few_shots is not None else task.pop_datapoints(frac=0.1)
        if self.max_n_blocks_eval > self.task.n_blocks:
            logger.warning(
                f"ℹ️ max_n_blocks_eval ({self.max_n_blocks_eval}) is larger than the number of blocks ({self.task.n_blocks})."
                f" Setting max_n_blocks_eval to {self.task.n_blocks}."
            )
            self.max_n_blocks_eval = self.task.n_blocks
        self.population_size = len(self.prompts)

        if hasattr(self.predictor, "begin_marker") and hasattr(self.predictor, "end_marker"):
            self.target_begin_marker = self.predictor.begin_marker
            self.target_end_marker = self.predictor.end_marker
        else:
            self.target_begin_marker = ""
            self.target_end_marker = ""

    def _initialize_population(self, initial_prompts: List[Prompt]) -> List[Prompt]:
        """Initializes the population of Prompt objects from initial instructions.

        Args:
            initial_prompts (List[str]): List of initial prompt instructions.

        Returns:
            List[Prompt]: Initialized population of prompts with few-shot examples.
        """
        population = []
        for prompt in initial_prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = self._create_few_shot_examples(prompt.instruction, num_examples)
            population.append(Prompt(prompt.instruction, few_shots))

        return population

    def _create_few_shot_examples(self, instruction: str, num_examples: int) -> List[str]:
        if num_examples == 0:
            return []

        few_shot_samples = self.df_few_shots.sample(num_examples, replace=False)
        sample_inputs = few_shot_samples[self.task.x_column].values.astype(str)
        sample_targets = few_shot_samples[self.task.y_column].values
        few_shots = [
            CAPO_FEWSHOT_TEMPLATE.replace("<input>", i).replace(
                "<output>", f"{self.target_begin_marker}{t}{self.target_end_marker}"
            )
            for i, t in zip(sample_inputs, sample_targets)
        ]

        if not self.create_fs_reasoning:
            # If we do not create reasoning, return the few-shot examples directly
            return few_shots

        preds, seqs = self.predictor.predict(
            [instruction] * num_examples,
            list(sample_inputs),
            return_seq=True,
        )
        if isinstance(seqs, str):
            seqs = [seqs]
        if isinstance(preds, str):
            preds = [preds]

        # Check which predictions are correct and get a single one per example
        for j in range(num_examples):
            # Process and clean up the generated sequences
            seqs[j] = seqs[j].replace(sample_inputs[j], "", 1).strip()
            # Check if the prediction is correct and add reasoning if so
            if preds[j] == sample_targets[j] or not self.check_fs_accuracy:
                few_shots[j] = CAPO_FEWSHOT_TEMPLATE.replace("<input>", sample_inputs[j]).replace("<output>", seqs[j])

        return few_shots

    def _crossover(self, parents: List[Prompt]) -> List[Prompt]:
        """Performs crossover among parent prompts to generate offsprings.

        Args:
            parents (List[Prompt]): List of parent prompts.

        Returns:
            List[Prompt]: List of new offsprings after crossover.
        """
        crossover_prompts = []
        offspring_few_shots = []
        for _ in range(self.crossovers_per_iter):
            mother, father = random.sample(parents, 2)
            crossover_prompt = (
                self.crossover_template.replace("<mother>", mother.instruction)
                .replace("<father>", father.instruction)
                .strip()
            )
            # collect all crossover prompts then pass them bundled to the meta llm (speedup)
            crossover_prompts.append(crossover_prompt)
            combined_few_shots = mother.few_shots + father.few_shots
            num_few_shots = (len(mother.few_shots) + len(father.few_shots)) // 2
            offspring_few_shot = random.sample(combined_few_shots, num_few_shots) if combined_few_shots else []
            offspring_few_shots.append(offspring_few_shot)

        child_instructions = self.meta_llm.get_response(crossover_prompts)

        offsprings = []
        for instruction, examples in zip(child_instructions, offspring_few_shots):
            instruction = extract_from_tag(instruction, "<prompt>", "</prompt>")
            offsprings.append(Prompt(instruction, examples))

        return offsprings

    def _mutate(self, offsprings: List[Prompt]) -> List[Prompt]:
        """Apply mutation to offsprings to generate new candidate prompts.

        Args:
            offsprings (List[Prompt]): List of offsprings to mutate.

        Returns:
            List[Prompt]: List of mutated prompts.
        """
        # collect all mutation prompts then pass them bundled to the meta llm (speedup)
        mutation_prompts = [
            self.mutation_template.replace("<instruction>", prompt.instruction) for prompt in offsprings
        ]
        new_instructions = self.meta_llm.get_response(mutation_prompts)

        mutated = []
        for new_instruction, prompt in zip(new_instructions, offsprings):
            new_instruction = extract_from_tag(new_instruction, "<prompt>", "</prompt>")
            p = random.random()

            new_few_shots: List[str]
            if p < 1 / 3 and len(prompt.few_shots) < self.upper_shots:  # add a random few shot
                new_few_shot = self._create_few_shot_examples(new_instruction, 1)
                new_few_shots = prompt.few_shots + new_few_shot
            elif 1 / 3 <= p < 2 / 3 and len(prompt.few_shots) > 0:  # remove a random few shot
                new_few_shots = random.sample(prompt.few_shots, len(prompt.few_shots) - 1)
            else:  # do not change few shots, but shuffle
                new_few_shots = prompt.few_shots

            random.shuffle(new_few_shots)
            mutated.append(Prompt(new_instruction, new_few_shots))

        return mutated

    def _do_racing(self, candidates: List[Prompt], k: int) -> Tuple[List[Prompt], List[float]]:
        """Perform the racing (selection) phase by comparing candidates based on their evaluation scores using the provided test statistic.

        Args:
            candidates (List[Prompt]): List of candidate prompts.
            k (int): Number of survivors to retain.

        Returns:
            List[Prompt]: List of surviving prompts after racing.
        """
        self.task.reset_block_idx()
        block_scores: List[List[float]] = []
        i = 0
        while len(candidates) > k and i < self.max_n_blocks_eval:
            # new_scores shape: (n_candidates, n_samples)
            new_scores: List[float] = self.task.evaluate(candidates, self.predictor, return_agg_scores=False)

            # subtract length penalty
            prompt_lengths = np.array([self.token_counter(c.construct_prompt()) for c in candidates])
            rel_prompt_lengths = prompt_lengths / self.max_prompt_length

            penalized_new_scores = np.array(new_scores) - self.length_penalty * rel_prompt_lengths[:, None]

            new_scores = penalized_new_scores.tolist()

            block_scores.append(new_scores)
            scores = np.concatenate(block_scores, axis=1)

            # boolean matrix C_ij indicating if candidate j is better than candidate i
            comparison_matrix = np.array(
                [[self.test_statistic(other_score, score, self.alpha) for other_score in scores] for score in scores]
            )

            # Sum along rows to get number of better scores for each candidate
            n_better = np.sum(comparison_matrix, axis=1)

            candidates, block_scores = filter_survivors(candidates, block_scores, mask=n_better < k)

            i += 1
            self.task.increment_block_idx()

        avg_scores = self.task.evaluate(candidates, self.predictor, eval_strategy="evaluated")
        prompts, avg_scores = sort_prompts_by_scores(candidates, avg_scores, top_k=k)

        return prompts, avg_scores

    def _pre_optimization_loop(self) -> None:
        self.prompts = self._initialize_population(self.prompts)
        self.max_prompt_length = (
            max(self.token_counter(p.construct_prompt()) for p in self.prompts) if self.prompts else 1
        )
        self.task.reset_block_idx()

    def _step(self) -> List[Prompt]:
        """Perform a single optimization step.

        Returns:
            List[Prompt]: The optimized list of prompts after the step.
        """
        offsprings = self._crossover(self.prompts)
        mutated = self._mutate(offsprings)
        combined = self.prompts + mutated

        self.prompts, self.scores = self._do_racing(combined, self.population_size)

        return self.prompts


def filter_survivors(
    candidates: List[Prompt], scores: List[List[float]], mask: Any
) -> Tuple[List[Prompt], List[List[float]]]:
    """Filter candidates and scores based on a boolean mask.

    Args:
        candidates (List[Prompt]): List of candidate prompts.
        scores (List[List[float]]): Corresponding scores for the candidates.
        mask (Any): Boolean mask indicating which candidates to keep.

    Returns:
        Tuple[List[Prompt], List[List[float]]]: Filtered candidates and their scores.
    """
    filtered_candidates = list(compress(candidates, mask))
    filtered_scores = list(compress(scores, mask))
    return filtered_candidates, filtered_scores
