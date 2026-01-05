"""Implementation of the Capoeira (Multi-Objective CAPO) optimizer."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.callbacks import BaseCallback
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask
    from promptolution.utils.config import ExperimentConfig

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.tasks.multi_objective_task import MultiObjectiveTask

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
        init_result = self.task.evaluate(prompts=self.prompts, predictor=self.predictor)
        initial_vectors = self._get_objective_vectors(init_result) #TODO rename
        fronts = self._non_dominated_sort(initial_vectors)
        self.incumbents = [self.prompts[i] for i in fronts[0]]
        self.challengers = [self.prompts[i] for front in fronts[1:] for i in front]

        # keep self.prompts as a "view" if base class expects it
        self.prompts = self.incumbents + self.challengers
        self.scores = initial_vectors[:, 0].tolist()

    
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
            inc_result = self.task.evaluate(prompts=self.incumbents, predictor=self.predictor, eval_strategy="evaluated")
            vecs_inc = self._get_objective_vectors(inc_result)
            self.scores = vecs_inc[:, 0].tolist()
        else:
            self.scores = []

        return self.prompts

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


        common_block_idx = 0
        while common_block_idx is not None:
            common_block_idx = self._sample_common_block(self.incumbents)
            self.task.set_block_idx(common_block_idx)  # type: ignore

            joint_result = self.task.evaluate(
                prompts=self.incumbents + [challenger],
                predictor=self.predictor
            )

            objective_vectors = self._get_objective_vectors(joint_result)
            challenger_vec = objective_vectors[-1]
            incumbent_vecs = objective_vectors[:-1]

            closest_inc_vec = self._get_closest_incumbent(challenger_vec, incumbent_vecs)

            if self._is_dominated(challenger_vec, closest_inc_vec):
                # challenger loses -> goes to population
                self.challengers.append(challenger)
                return

        self.incumbents.append(challenger)
        self._update_incumbent_front()

    def _sample_common_block(self, prompts: List[Prompt]) -> Optional[int]:
        """Sample a block index that has been evaluated by all given prompts.
        Returns None if no such block exists."""
        per_prompt = self.task.get_evaluated_blocks(prompts)  # Dict[prompt -> Set[int]]
        block_sets = list(per_prompt.values())

        if not block_sets:
            return random.randrange(self.task.n_blocks)

        common = set.intersection(*block_sets)
        if not common:
            return None

        return random.choice(tuple(common))

    def _get_closest_incumbent(
        self, challenger_vec: np.ndarray, incumbent_vecs: np.ndarray
    ) -> np.ndarray:
        """Return the vector of the geometrically closest incumbent."""
        all_vecs = np.vstack([incumbent_vecs, challenger_vec[None, :]])
        min_b = np.min(all_vecs, axis=0)
        max_b = np.max(all_vecs, axis=0)
        rng = max_b - min_b
        rng[rng == 0] = 1.0  # Avoid div/0

        norm_chal = (challenger_vec - min_b) / rng
        norm_incs = (incumbent_vecs - min_b) / rng

        dists = np.linalg.norm(norm_incs - norm_chal, axis=1)
        idx = int(np.argmin(dists))
        return incumbent_vecs[idx]


    def _update_incumbent_front(self) -> None:
        """
        After adding a challenger that survived a full race, recompute the incumbent Pareto front.
        Default behavior: incumbents become front-0 (on current evaluation state),
        all other incumbents are demoted to challengers.
        """
        if not self.incumbents:
            return

        vecs_result = self.task.evaluate(prompts=self.incumbents, predictor=self.predictor, eval_strategy="evaluated")
        vecs = self._get_objective_vectors(vecs_result)
        fronts = self._non_dominated_sort(vecs)

        new_incumbents = [self.incumbents[i] for i in fronts[0]]
        demoted = [self.incumbents[i] for front in fronts[1:] for i in front]

        self.incumbents = new_incumbents
        self.challengers.extend(demoted)


    def _get_objective_vectors(self, result) -> np.ndarray:

        # If the task is multi-objective, include all objective dimensions, else single objective.
        if isinstance(self.task, MultiObjectiveTask):
            agg_scores = np.stack(result.agg_scores, axis=1)  # shape: (n_prompts, n_objectives)
        else:
            agg_scores = np.atleast_2d(result.agg_scores).T  # shape: (n_prompts, 1)

        agg_input_tokens = np.asarray(result.agg_input_tokens)
        agg_output_tokens = np.asarray(result.agg_output_tokens)
        cost_scalar = self.cost_per_input_token * agg_input_tokens + self.cost_per_output_token * agg_output_tokens
        cost_scalar = cost_scalar.reshape(-1, 1)

        return np.hstack([agg_scores, -cost_scalar])

    def _advance_one_incumbent(self) -> None:
        """
        Default MO-CAPO step after processing a challenger:
        evaluate one incumbent on one additional sequential block.
        """
        # choose least evaluated incumbent
        eval_counts = [
            len(self.task.get_evaluated_blocks([inc])) for inc in self.incumbents
        ]
        min_count = min(eval_counts)
        candidates = [inc for inc, count in zip(self.incumbents, eval_counts) if count == min_count]
        chosen = random.sample(candidates, k=1)
        self.task.evaluate(prompts=chosen, predictor=self.predictor)


    def _prune_population(self) -> None:
        """
        Enforce |incumbents| + |challengers| <= population_size using Pareto logic.
        
        Logic:
        1. Prune from Challengers first (they are less optimal than incumbents).
        - If challengers have DIFFERENT evaluation blocks (Heterogeneous):
            We cannot fairly compare their scores. Prune the one with the FEWEST evaluations
            (least information/newest).
        - If challengers have the SAME evaluation blocks (Homogeneous):
            Perform Non-Dominated Sorting (NDS). Identify the worst front.
            Use Crowding Distance to prune the most crowded (least unique) individual from that front.
        
        2. If no Challengers, prune from Incumbents.
        - Use Crowding Distance to remove the least unique incumbent.
        """
        while len(self.incumbents) + len(self.challengers) > self.population_size:
            if self.challengers:
                # 1. Check Heterogeneity (Fairness Check)
                chal_blocks_map = self.task.get_evaluated_blocks(self.challengers)
                block_sets = list(chal_blocks_map.values())
                
                # Ensure we have data to compare
                if not block_sets:
                    self.challengers.pop(random.randrange(len(self.challengers)))
                    continue

                first_set = block_sets[0]
                # Are all challengers evaluated on the exact same set of blocks?
                is_homogeneous = all(s == first_set for s in block_sets)

                if not is_homogeneous:
                    # CASE A: Heterogeneous (Unfair comparison).
                    # Prune the prompt with the FEWEST evaluations (least reliable/least invested).
                    counts = [len(s) for s in block_sets]
                    min_count = min(counts)
                    
                    # Find all indices with the minimum count (handle ties randomly)
                    candidates = [i for i, c in enumerate(counts) if c == min_count]
                    victim_idx = random.choice(candidates)
                    
                    self.challengers.pop(victim_idx)
                
                else:
                    # CASE B: Homogeneous (Fair comparison).
                    # Use NDS + Crowding Distance.
                    
                    # Get objective vectors for all challengers (safe because blocks are identical)
                    res = self.task.evaluate(
                        self.challengers, 
                        self.predictor, 
                        eval_strategy="evaluated"
                    )
                    vecs = self._get_objective_vectors(res)
                    
                    # Perform Non-Dominated Sort
                    fronts = self._non_dominated_sort(vecs)
                    
                    # Select the worst front (the last one)
                    worst_front_indices = fronts[-1]
                    
                    if len(worst_front_indices) == 1:
                        # Only one candidate in the worst front -> prune it
                        victim_idx = worst_front_indices[0]
                    else:
                        # Multiple candidates in worst front -> Prune by Crowding Distance
                        # We want to keep diversity (high CD), so we remove low CD.
                        worst_front_vecs = vecs[worst_front_indices]
                        dists = self._calculate_crowding_distance(worst_front_vecs)
                        
                        # Find index relative to the worst front list
                        local_worst_idx = int(np.argmin(dists))
                        # Map back to the main challenger list index
                        victim_idx = worst_front_indices[local_worst_idx]
                    
                    self.challengers.pop(victim_idx)

            else:
                # --- PRUNE FROM INCUMBENTS ---
                # Fallback: If we only have incumbents, remove the least unique one.
                if len(self.incumbents) <= 1:
                    break
                
                res = self.task.evaluate(
                    self.incumbents, 
                    self.predictor, 
                    eval_strategy="evaluated"
                )
                vecs = self._get_objective_vectors(res)
                dists = self._calculate_crowding_distance(vecs)
                
                # Remove the one with the smallest crowding distance
                victim_idx = int(np.argmin(dists))
                self.incumbents.pop(victim_idx)

        self.prompts = self.incumbents + self.challengers


    def _non_dominated_sort(self, obj_vectors: np.ndarray) -> List[List[int]]:
        """Perform fast non-dominated sorting (NSGA-II) in a vectorized manner."""
        n_solutions = obj_vectors.shape[0]

        greater = obj_vectors[:, None, :] > obj_vectors[None, :, :]
        greater_equal = obj_vectors[:, None, :] >= obj_vectors[None, :, :]
        dominates = np.all(greater_equal, axis=2) & np.any(greater, axis=2)

        domination_counts = dominates.sum(axis=0)
        dominated_solutions = [list(np.where(dominates[i])[0]) for i in range(n_solutions)]

        fronts: List[List[int]] = [list(np.where(domination_counts == 0)[0])]

        current_front = 0
        while current_front < len(fronts) and len(fronts[current_front]) > 0:
            next_front: List[int] = []
            for i in fronts[current_front]:
                for dominated in dominated_solutions[i]:
                    domination_counts[dominated] -= 1
                    if domination_counts[dominated] == 0:
                        next_front.append(dominated)
            if len(next_front) > 0:
                fronts.append(next_front)
            current_front += 1

        return fronts

    @staticmethod
    def _is_dominated(vec1, vec2):
        """Returns True if vec2 dominates vec1 in a maximize-all setting."""
        return np.all(vec2 >= vec1) and np.any(vec2 > vec1)
    
    @staticmethod
    def _calculate_crowding_distance(obj_vectors: np.ndarray) -> np.ndarray:
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
