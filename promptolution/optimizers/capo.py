"""Implementation of the CAPO (Cost-Aware Prompt Optimization) algorithm."""

import random

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig
    from promptolution.utils.test_statistics import TestStatistics

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.capo_utils import build_few_shot_examples, perform_crossover, perform_mutation
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
        """Initialize the CAPOptimizer with various parameters for prompt evolution.

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
        if "block" not in self.task.eval_strategy:
            logger.warning(
                f"ℹ️ CAPO requires 'block' in the eval_strategy, but got {self.task.eval_strategy}. Setting eval_strategy to 'sequential_block'."
            )
            self.task.eval_strategy = "sequential_block"
        self.population_size = len(self.prompts)

        if hasattr(self.predictor, "begin_marker") and hasattr(self.predictor, "end_marker"):
            self.target_begin_marker = self.predictor.begin_marker  # type: ignore
            self.target_end_marker = self.predictor.end_marker  # type: ignore
        else:
            self.target_begin_marker = ""
            self.target_end_marker = ""

    def _initialize_population(self, initial_prompts: List[Prompt]) -> List[Prompt]:
        """Initialize the population of Prompt objects from initial instructions."""
        population = []
        for prompt in initial_prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = build_few_shot_examples(
                instruction=prompt.instruction,
                num_examples=num_examples,
                df_few_shots=self.df_few_shots,
                x_column=self.task.x_column,
                y_column=self.task.y_column,
                predictor=self.predictor,
                fewshot_template=CAPO_FEWSHOT_TEMPLATE,
                target_begin_marker=self.target_begin_marker,
                target_end_marker=self.target_end_marker,
                check_fs_accuracy=self.check_fs_accuracy,
                create_fs_reasoning=self.create_fs_reasoning,
            )
            population.append(Prompt(prompt.instruction, few_shots))

        return population

    def _do_racing(self, candidates: List[Prompt], k: int) -> Tuple[List[Prompt], List[float]]:
        """Perform the racing (selection) phase by comparing candidates based on their evaluation scores using the provided test statistic.

        Args:
            candidates (List[Prompt]): List of candidate prompts.
            k (int): Number of survivors to retain.

        Returns:
            List[Prompt]: List of surviving prompts after racing.
        """
        self.task.reset_block_idx()
        block_scores: List[np.ndarray] = []
        i = 0
        while len(candidates) > k and i < self.max_n_blocks_eval:
            # new_scores shape: (n_candidates, n_samples)
            results = self.task.evaluate(candidates, self.predictor)
            new_scores = results.scores

            # subtract length penalty
            prompt_lengths = np.array([self.token_counter(c.construct_prompt()) for c in candidates])
            rel_prompt_lengths = prompt_lengths / self.max_prompt_length

            penalized_new_scores = new_scores - self.length_penalty * rel_prompt_lengths[:, None]

            block_scores.append(penalized_new_scores)
            scores = np.concatenate(block_scores, axis=1)

            # boolean matrix C_ij indicating if candidate j is better than candidate i
            comparison_matrix = np.array(
                [[self.test_statistic(other_score, score, self.alpha) for other_score in scores] for score in scores]
            )

            # Sum along rows to get number of better scores for each candidate
            n_better = np.sum(comparison_matrix, axis=1)

            candidates, block_scores = self.filter_survivors(candidates, block_scores, mask=n_better < k)

            i += 1
            self.task.increment_block_idx()

        final_result = self.task.evaluate(candidates, self.predictor, eval_strategy="evaluated")
        avg_scores = final_result.scores.tolist()
        prompts, avg_scores = sort_prompts_by_scores(candidates, avg_scores, top_k=k)

        return prompts, avg_scores

    def _pre_optimization_loop(self) -> None:
        self.prompts = self._initialize_population(self.prompts)
        self.max_prompt_length = (
            max(self.token_counter(p.construct_prompt()) for p in self.prompts) if self.prompts else 1
        )
        self.task.reset_block_idx()

    def _step(self) -> List[Prompt]:
        """Perform a single optimization step."""
        offsprings = perform_crossover(self.prompts, self.crossovers_per_iter, self.crossover_template, self.meta_llm)
        mutated = perform_mutation(
            offsprings=offsprings,
            mutation_template=self.mutation_template,
            upper_shots=self.upper_shots,
            meta_llm=self.meta_llm,
            few_shot_kwargs=dict(
                df_few_shots=self.df_few_shots,
                x_column=self.task.x_column,
                y_column=self.task.y_column,
                predictor=self.predictor,
                fewshot_template=CAPO_FEWSHOT_TEMPLATE,
                target_begin_marker=self.target_begin_marker,
                target_end_marker=self.target_end_marker,
                check_fs_accuracy=self.check_fs_accuracy,
                create_fs_reasoning=self.create_fs_reasoning,
            ),
        )
        combined = self.prompts + mutated

        self.prompts, self.scores = self._do_racing(combined, self.population_size)

        return self.prompts

    @staticmethod
    def filter_survivors(
        candidates: List[Prompt], scores: List[np.ndarray], mask: Any
    ) -> Tuple[List[Prompt], List[np.ndarray]]:
        """Filter candidates and scores based on a boolean mask.

        Args:
            candidates (List[Prompt]): List of candidate prompts.
            scores (List[List[float]]): Corresponding scores for the candidates.
            mask (Any): Boolean mask indicating which candidates to keep.

        Returns:
            Tuple[List[Prompt], List[List[float]]]: Filtered candidates and their scores.
        """
        assert len(candidates) == len(mask), "Length of candidates, and mask must be the same."
        assert all(
            len(candidates) == len(score) for score in scores
        ), "Each score list must have the same length as candidates."

        filtered_candidates = [c for c, m in zip(candidates, mask) if m]
        filtered_scores = [np.asarray(score)[mask] for score in scores]

        return filtered_candidates, filtered_scores
