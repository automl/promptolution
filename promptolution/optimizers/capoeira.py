"""Implementation of the Capoeira (Multi-Objective CAPO) optimizer."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.capo_utils import build_few_shot_examples, perform_crossover, perform_mutation
from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt
from promptolution.utils.templates import CAPO_CROSSOVER_TEMPLATE, CAPO_FEWSHOT_TEMPLATE, CAPO_MUTATION_TEMPLATE
from promptolution.utils.token_counter import get_token_counter

logger = get_logger(__name__)


class Capoeira(BaseOptimizer):
    """Multi-objective variant of CAPO with Pareto-based selection."""

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
        cost_per_input_token: float = 1.0,
        cost_per_output_token: float = 0.0,
        check_fs_accuracy: bool = True,
        create_fs_reasoning: bool = True,
        df_few_shots: Optional[pd.DataFrame] = None,
        callbacks: Optional[List["BaseCallback"]] = None,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the Capoeira optimizer.

        Args:
            predictor: The predictor used to evaluate prompt performance.
            task: The task instance containing data and evaluation settings.
            meta_llm: Meta language model for crossover and mutation generation.
            initial_prompts: Starting prompt strings to seed the population.
            crossover_template: Optional meta-prompt template for crossover.
            mutation_template: Optional meta-prompt template for mutation.
            crossovers_per_iter: Number of crossover operations per iteration.
            upper_shots: Maximum number of few-shot examples to attach.
            cost_per_input_token: Weight applied to input token cost for the cost objective.
            cost_per_output_token: Weight applied to output token cost for the cost objective.
            check_fs_accuracy: Whether to verify few-shot correctness before use.
            create_fs_reasoning: Whether to replace few-shots with model reasoning.
            df_few_shots: Optional dataframe providing few-shot examples. If None, will pop 10% of datapoints from task.
            callbacks: Optional list of optimization callbacks.
            config: Optional experiment configuration object.
        """
        self.meta_llm = meta_llm
        self.downstream_llm = predictor.llm
        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots

        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.check_fs_accuracy = check_fs_accuracy
        self.create_fs_reasoning = create_fs_reasoning

        super().__init__(predictor, task, initial_prompts, callbacks, config)

        self.crossover_template = self._initialize_meta_template(crossover_template or CAPO_CROSSOVER_TEMPLATE)
        self.mutation_template = self._initialize_meta_template(mutation_template or CAPO_MUTATION_TEMPLATE)
        self.token_counter = get_token_counter(self.downstream_llm)
        self.df_few_shots = df_few_shots if df_few_shots is not None else task.pop_datapoints(frac=0.1)
        self.population_size = len(self.prompts)

        if hasattr(self.predictor, "begin_marker") and hasattr(self.predictor, "end_marker"):
            self.target_begin_marker = self.predictor.begin_marker  # type: ignore
            self.target_end_marker = self.predictor.end_marker  # type: ignore
        else:
            self.target_begin_marker = ""
            self.target_end_marker = ""

    def _pre_optimization_loop(self) -> None:
        population: List[Prompt] = []
        for prompt in self.prompts:
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

        self.prompts = population
        # TODO: align placement of the logic with capo
        self.max_prompt_length = (
            max(self.token_counter(p.construct_prompt()) for p in self.prompts) if self.prompts else 1
        )
        initial_vectors = self._evaluate_candidates(self.prompts)
        self.prompts, selected_vectors = self._select_population(self.prompts, initial_vectors)
        self.scores = (-selected_vectors[:, 0]).tolist()

    def _evaluate_candidates(self, candidates: List[Prompt]) -> np.ndarray:
        scores, input_tokens, output_tokens = self.task.evaluate(
            candidates,
            self.predictor,
            eval_strategy=self.task.eval_strategy,
            return_costs=True,
            return_seq=False,
            return_agg_scores=True,
        )

        # TODO move to evaluate method!
        input_tokens_array = np.array(input_tokens, dtype=float)
        output_tokens_array = np.array(output_tokens, dtype=float)
        scores_array = np.array(scores, dtype=float)

        score_vectors = np.column_stack(
            [
                -scores_array,
                self.cost_per_input_token * input_tokens_array + self.cost_per_output_token * output_tokens_array,
            ]
        )
        return score_vectors

    def _select_population(
        self, candidates: List[Prompt], score_vectors: np.ndarray
    ) -> Tuple[List[Prompt], np.ndarray]:
        selected_indices: List[int] = []
        fronts = self.fast_non_dominated_sort(score_vectors)
        for front in fronts:
            if len(selected_indices) + len(front) <= self.population_size:
                selected_indices.extend(front)
            else:
                remaining = self.population_size - len(selected_indices)
                front_vectors = score_vectors[front]
                distances = self.calculate_crowding_distance(front_vectors)
                sorted_front = [i for _, i in sorted(zip(distances, front), reverse=True)]
                selected_indices.extend(sorted_front[:remaining])
                break

        selected_prompts = [candidates[i] for i in selected_indices]
        selected_vectors = score_vectors[selected_indices]
        return selected_prompts, selected_vectors

    def _step(self) -> List[Prompt]:
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

        score_vectors = self._evaluate_candidates(combined)
        self.prompts, selected_vectors = self._select_population(combined, score_vectors)
        self.scores = (-selected_vectors[:, 0]).tolist()
        return self.prompts

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Return the current Pareto front with objective values."""
        score_vectors = self._evaluate_candidates(self.prompts)
        return [
            {
                "prompt": prompt.construct_prompt(),
                "score": float(score_vectors[i][0] * -1),
                "cost": float(score_vectors[i][1]),
            }
            for i, prompt in enumerate(self.prompts)
        ]

    @staticmethod
    def fast_non_dominated_sort(obj_vectors: np.ndarray) -> List[List[int]]:
        """Perform fast non-dominated sorting (NSGA-II) in a vectorized manner."""
        num_solutions = obj_vectors.shape[0]
        if num_solutions == 0:
            return []

        less = obj_vectors[:, None, :] < obj_vectors[None, :, :]
        less_equal = obj_vectors[:, None, :] <= obj_vectors[None, :, :]
        dominates = np.all(less_equal, axis=2) & np.any(less, axis=2)

        domination_counts = dominates.sum(axis=0)
        dominated_solutions = [list(np.where(dominates[i])[0]) for i in range(num_solutions)]

        fronts: List[List[int]] = [list(np.where(domination_counts == 0)[0])]
        current_front = 0

        while current_front < len(fronts) and fronts[current_front]:
            next_front: List[int] = []
            for i in fronts[current_front]:
                for dominated in dominated_solutions[i]:
                    domination_counts[dominated] -= 1
                    if domination_counts[dominated] == 0:
                        next_front.append(dominated)
            if next_front:
                fronts.append(next_front)
            current_front += 1

        return fronts

    @staticmethod
    def calculate_crowding_distance(obj_vectors: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for a set of solutions."""
        num_solutions, num_obj = obj_vectors.shape
        if num_solutions <= 2:
            return np.full(num_solutions, float("inf"))

        distances = np.zeros(num_solutions)
        for i in range(num_obj):
            sorted_indices = np.argsort(obj_vectors[:, i])
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            f_min = obj_vectors[sorted_indices[0], i]
            f_max = obj_vectors[sorted_indices[-1], i]
            if f_max == f_min:
                continue

            slice_indices = sorted_indices[1:-1]
            next_vals = obj_vectors[sorted_indices[2:], i]
            prev_vals = obj_vectors[sorted_indices[:-2], i]
            distances[slice_indices] += (next_vals - prev_vals) / (f_max - f_min)
        return distances
