"""Implementation of the Capoeira (Multi-Objective CAPO) optimizer."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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
from promptolution.utils.templates import CAPO_CROSSOVER_TEMPLATE, CAPO_MUTATION_TEMPLATE
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

        self.incumbents: List[Prompt] = []
        self.challengers: List[Prompt] = []

        self.crossover_template = self._initialize_meta_template(crossover_template or CAPO_CROSSOVER_TEMPLATE)
        self.mutation_template = self._initialize_meta_template(mutation_template or CAPO_MUTATION_TEMPLATE)
        self.token_counter = get_token_counter(self.downstream_llm)
        self.df_few_shots = df_few_shots if df_few_shots is not None else task.pop_datapoints(frac=0.1)
        self.population_size = len(self.prompts)

        if "block" not in self.task.eval_strategy:
            logger.warning(
                f"ℹ️ CAPO requires 'block' in the eval_strategy, but got {self.task.eval_strategy}. Setting eval_strategy to 'sequential_block'."
            )
            self.task.eval_strategy = "sequential_block"

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
                optimizer=self,
            )
            population.append(Prompt(prompt.instruction, few_shots))

        self.prompts = population
        # TODO: align placement of the logic with capo
        self.max_prompt_length = (
            max(self.token_counter(p.construct_prompt()) for p in self.prompts) if self.prompts else 1
        )
        initial_vectors = self._calculate_objective_vector(self.prompts) #TODO rename
        fronts = self.fast_non_dominated_sort(initial_vectors)
        self.incumbents = [self.prompts[i] for i in fronts[0]]
        self.challengers = [self.prompts[i] for front in fronts[1:] for i in front]

        # keep self.prompts as a "view" if base class expects it
        self.prompts = self.incumbents + self.challengers
        self.scores = initial_vectors[:, 0].tolist()

    def _do_intensification(self, challenger: Prompt) -> None:
        """
        Default MO-CAPO intensification (closest-incumbent comparison):
        - evaluate challenger + incumbents on sequential blocks
        - maintain running averages (challenger and incumbents)
        - early reject if closest incumbent dominates challenger average
        - if challenger survives all blocks: promote to incumbents and update front
        """
        if not self.incumbents:
            self.incumbents.append(challenger)
            return

        # Start race from a consistent block index
        self.task.reset_block_idx() # TODO this might need to change

        chal_hist: List[np.ndarray] = []
        inc_hist: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(self.incumbents))}

        for _ in range(self.task.n_blocks):
            joint_result = self.task.evaluate(
                self.incumbents + [challenger],
                self.predictor,
                eval_strategy="sequential_block",
            )
            joint_vecs = self._objective_vectors_from_result(joint_result)

            inc_vecs = joint_vecs[:-1]
            chal_vec = joint_vecs[-1]

            chal_hist.append(chal_vec)
            for i, v in enumerate(inc_vecs):
                inc_hist[i].append(v)

            chal_avg = np.mean(chal_hist, axis=0)

            # Default: compare only against closest incumbent (in normalized objective space)
            closest = self._get_closest_incumbent(chal_avg)
            closest_idx = self.incumbents.index(closest)
            closest_avg = np.mean(inc_hist[closest_idx], axis=0)

            if self._is_dominated(chal_avg, closest_avg):
                # challenger loses -> goes to population
                self.challengers.append(challenger)
                self.task.reset_block_idx()
                return

            self.task.increment_block_idx()

        # Survived full race -> promote and update incumbent front
        self.incumbents.append(challenger)
        self._update_incumbent_front()
        self.task.reset_block_idx()


    def _get_closest_incumbent(self, challenger_vec: np.ndarray):
        """Finds the geometrically closest incumbent."""
        inc_vecs = self._calculate_objective_vector(self.incumbents, eval_strategy="sequential_block")
        all_vecs = np.vstack([inc_vecs, challenger_vec[None, :]])
        min_b = np.min(all_vecs, axis=0)
        max_b = np.max(all_vecs, axis=0)
        rng = max_b - min_b
        rng[rng == 0] = 1.0  # Avoid div/0

        norm_chal = (challenger_vec - min_b) / rng
        norm_incs = (inc_vecs - min_b) / rng

        dists = np.linalg.norm(norm_incs - norm_chal, axis=1)
        return self.incumbents[np.argmin(dists)]


    def _update_incumbent_front(self) -> None:
        """
        After adding a challenger that survived a full race, recompute the incumbent Pareto front.
        Default behavior: incumbents become front-0 (on current evaluation state),
        all other incumbents are demoted to challengers.
        """
        if not self.incumbents:
            return

        vecs = self._calculate_objective_vector(self.incumbents, eval_strategy="sequential_block")
        fronts = self.fast_non_dominated_sort(vecs)

        new_incumbents = [self.incumbents[i] for i in fronts[0]]
        demoted = [self.incumbents[i] for front in fronts[1:] for i in front]

        self.incumbents = new_incumbents
        self.challengers.extend(demoted)


    def _calculate_objective_vector(self, prompts: List[Prompt], eval_strategy=None) -> np.ndarray:
        eval_result = self.task.evaluate(
            prompts=prompts,
            predictor=self.predictor,
            eval_strategy=eval_strategy,
        )
        return self._objective_vectors_from_result(eval_result)

    def _objective_vectors_from_result(self, result) -> np.ndarray:
        agg_scores = result.agg_scores
        agg_input_tokens = result.costs.agg_input_tokens
        agg_output_tokens = result.costs.agg_output_tokens
        cost_scalar = self.cost_per_input_token * agg_input_tokens + self.cost_per_output_token * agg_output_tokens
        return np.column_stack([agg_scores, -cost_scalar])

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


    def _advance_one_incumbent(self) -> None:
        """
        Default MO-CAPO step after processing a challenger:
        evaluate one incumbent on one additional sequential block.
        (With your current task API, this is the closest equivalent to the
        "catch-up / new block" logic.)
        """
        if not self.incumbents:
            return

        chosen = random.choice(self.incumbents)

        _ = self.task.evaluate( # TODO might need to change
            prompts=[chosen],
            predictor=self.predictor,
            eval_strategy="sequential_block",
        )
        self.task.increment_block_idx()

    def _prune_population(self) -> None:
        """
        Enforce |incumbents| + |challengers| <= population_size.
        Default behavior: prune challengers first; if none, prune incumbents by crowding distance.
        """
        while len(self.incumbents) + len(self.challengers) > self.population_size:
            if self.challengers:
                # simplest default: remove a random challenger
                self.challengers.pop(random.randrange(len(self.challengers)))
            else:
                if len(self.incumbents) <= 1:
                    break
                vecs = self._calculate_objective_vector(self.incumbents, eval_strategy="sequential_block")
                dists = self.calculate_crowding_distance(vecs)
                worst = int(np.argmin(dists))
                self.incumbents.pop(worst)


    def _step(self) -> List[Prompt]:
        # 1) generate challengers (random parent selection happens inside perform_crossover)
        offsprings = perform_crossover(self.prompts, optimizer=self)
        new_challengers = perform_mutation(offsprings=offsprings, optimizer=self)

        # 2) intensify each challenger; after each, advance incumbents + prune
        for chal in new_challengers:
            self._do_intensification(chal)
            self._advance_one_incumbent()
            self._prune_population()

        # 3) update "view" for base class / callbacks
        self.prompts = self.incumbents + self.challengers

        # 4) logging scores: incumbents only (optional)
        if self.incumbents:
            vecs_inc = self._calculate_objective_vector(self.incumbents, eval_strategy="sequential_block")
            self.scores = vecs_inc[:, 0].tolist()
        else:
            self.scores = []

        return self.prompts



    @staticmethod
    def _is_dominated(vec1, vec2):
        """Returns True if vec2 dominates vec1 in a maximize-all setting."""
        return np.all(vec2 >= vec1) and np.any(vec2 > vec1)
    
    @staticmethod
    def fast_non_dominated_sort(obj_vectors: np.ndarray) -> List[List[int]]:
        """Perform fast non-dominated sorting (NSGA-II) in a vectorized manner."""
        num_solutions = obj_vectors.shape[0]
        if num_solutions == 0:
            return []

        greater = obj_vectors[:, None, :] > obj_vectors[None, :, :]
        greater_equal = obj_vectors[:, None, :] >= obj_vectors[None, :, :]
        dominates = np.all(greater_equal, axis=2) & np.any(greater, axis=2)

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
