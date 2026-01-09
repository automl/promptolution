"""Implementation of the Capoeira (Multi-Objective CAPO) optimizer."""

import random

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, List, Optional, Tuple

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

    supports_multi_objective = True

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
        cost_per_output_token: float = 1.0,
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

        self.incumbents: List[Prompt] = self.prompts
        self.non_incumbents: List[Prompt] = []
        self.population_size = len(self.prompts)
        
        if self.task.task_type == "multi":
            self.n_objectives = len(self.task.tasks) + 1  # +1 for cost objective
        else:
            self.n_objectives = 2  # single objective + cost objective
            
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

        init_result = self.task.evaluate(population, self.predictor)
        initial_vectors = self._get_objective_vectors(init_result)
        fronts = self._non_dominated_sort(initial_vectors)
        self.incumbents = [population[i] for i in fronts[0]]
        self.non_incumbents = [population[i] for front in fronts[1:] for i in front]

        # keep self.prompts as a "view" if base class expects it
        self.scores = initial_vectors[:, 0].tolist()

    def _step(self) -> List[Prompt]:
        # 1) generate challengers
        offsprings = perform_crossover(self.prompts, self, self._tournament_selection)
        new_challengers = perform_mutation(offsprings, self)

        # 2) intensify each challenger; after each, advance incumbents + prune
        for challenger in new_challengers:
            self._do_intensification(challenger)
            self._select_survivors()
            self._advance_one_incumbent()

        inc_result = self.task.evaluate(
            prompts=self.incumbents, predictor=self.predictor, eval_strategy="evaluated"
        )
        vecs_inc = self._get_objective_vectors(inc_result)
        self.scores = vecs_inc[:, 0].tolist()
        self.prompts = self.incumbents

        return self.prompts

    def _do_intensification(self, challenger: Prompt) -> None:
        common_blocks = self._get_common_blocks(self.incumbents)

        # bootstrap if no common blocks yet
        if not common_blocks:
            b = random.randrange(self.task.n_blocks)
            self.task.set_block_idx(b)
            self.task.evaluate(self.incumbents + [challenger], self.predictor)
            self.incumbents.append(challenger)
            self._update_incumbent_front(blocks={b})
            return

        remaining_blocks = set(common_blocks)

        challenger_mean: Optional[np.ndarray] = None
        incumbents_mean: Optional[np.ndarray] = None
        t = 0

        fold_vec = np.full((self.n_objectives,), -np.inf)

        while remaining_blocks:
            b = random.choice(tuple(remaining_blocks))
            remaining_blocks.remove(b)

            # evaluate all incumbents + challenger on THIS block (cache will avoid recompute)
            self.task.set_block_idx(b)
            res = self.task.evaluate(self.incumbents + [challenger], self.predictor)
            vecs = self._get_objective_vectors(res)  # per-block vectors, shape (n_inc+1, n_obj)
            incumbent_block = vecs[:-1]
            challenger_block = vecs[-1]

            # running means
            t += 1
            if challenger_mean is None:
                challenger_mean = challenger_block.copy()
                incumbents_mean = incumbent_block.copy()
            else:
                challenger_mean += (challenger_block - challenger_mean) / t
                incumbents_mean += (incumbent_block - incumbents_mean) / t  # type: ignore

            if self._is_dominated(fold_vec, challenger_mean):
                continue

            fold_vec = challenger_mean.copy() # TODO RENAME

            closest_incumbent = self._get_closest_incumbent(challenger)  # type: ignore
            if self._is_dominated(challenger_mean, closest_incumbent):
                self.non_incumbents.append(challenger)
                return

        # survived all common blocks -> admit and update front restricted to common_blocks
        self.incumbents.append(challenger)
        self._update_incumbent_front(blocks=common_blocks)

    def _get_closest_incumbent(self, challenger) -> np.ndarray:
        """Return the vector of the geometrically closest incumbent."""
        challenger_res = self.task.evaluate(challenger, self.predictor, eval_strategy="evaluated")
        challenger_vec = self._get_objective_vectors(challenger_res)
        
        incumbent_res = self.task.evaluate(self.incumbents, self.predictor, eval_strategy="evaluated")
        incumbent_vecs = self._get_objective_vectors(incumbent_res)
        
        all_vecs = np.vstack([incumbent_vecs, challenger_vec])
        min_b = np.min(all_vecs, axis=0)
        max_b = np.max(all_vecs, axis=0)
        rng = max_b - min_b
        rng[rng == 0] = 1.0  # Avoid div/0

        challenger_norm = (challenger_vec - min_b) / rng
        incumbents_norm = (incumbent_vecs - min_b) / rng

        dists = np.linalg.norm(incumbents_norm - challenger_norm, axis=1)
        idx = int(np.argmin(dists))
        return incumbent_vecs[idx]

    def _update_incumbent_front(self, blocks: Optional[set[int]] = None) -> None:
        if blocks is None:
            res = self.task.evaluate(self.incumbents, self.predictor, eval_strategy="evaluated")
        else:
            self.task.set_block_idx(list(sorted(blocks)))  # sorted for deterministic behaviour
            res = self.task.evaluate(self.incumbents, self.predictor)

        vecs = self._get_objective_vectors(res)

        fronts = self._non_dominated_sort(vecs)

        new_incumbents = [self.incumbents[i] for i in fronts[0]]
        demoted = [self.incumbents[i] for front in fronts[1:] for i in front]

        self.incumbents = new_incumbents
        self.non_incumbents.extend(demoted)

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
        if not self.incumbents:
            return

        blocks_map = self.task.get_evaluated_blocks(self.incumbents)  # Dict[str -> Set[int]]
        inc_keys = [str(inc) for inc in self.incumbents]

        # least evaluated incumbents
        eval_counts = [len(blocks_map[k]) for k in inc_keys]
        min_count = min(eval_counts)
        least = [inc for inc, c in zip(self.incumbents, eval_counts) if c == min_count]
        chosen_inc = random.choice(least)

        # union over incumbents
        union_blocks: set[int] = set()
        for inc in self.incumbents:
            union_blocks |= set(blocks_map[str(inc)])

        chosen_blocks = set(blocks_map[str(chosen_inc)])

        # gap-first, else brand-new
        gap_blocks = union_blocks - chosen_blocks
        if gap_blocks:
            b = random.choice(tuple(gap_blocks))
        else:
            all_blocks = set(range(self.task.n_blocks))
            new_blocks = all_blocks - union_blocks
            if not new_blocks:
                return
            b = random.choice(tuple(new_blocks))

        self.task.set_block_idx(b)
        self.task.evaluate(prompts=[chosen_inc], predictor=self.predictor)

    def _select_survivors(self) -> None:
        """Prune population via Pareto logic to enforce size constraints."""
        while len(self.incumbents) + len(self.non_incumbents) > self.population_size:
            if len(self.non_incumbents) > 0:
                # 1. Check Heterogeneity (Fairness Check)
                chal_blocks_map = self.task.get_evaluated_blocks(self.non_incumbents)
                block_sets = list(chal_blocks_map.values())

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

                    self.non_incumbents.pop(victim_idx)
                    continue

                # CASE B: Homogeneous (Fair comparison).
                # Use NDS + Crowding Distance.

                # Get objective vectors for all challengers (safe because blocks are identical)
                res = self.task.evaluate(self.non_incumbents, self.predictor, eval_strategy="evaluated")
                vecs = self._get_objective_vectors(res)

                # Perform Non-Dominated Sort
                fronts = self._non_dominated_sort(vecs)

                # Select the worst front (the last one)
                worst_front_indices = fronts[-1]

                # Multiple candidates in worst front -> Prune by Crowding Distance
                # We want to keep diversity (high CD), so we remove low CD.
                worst_front_vecs = vecs[worst_front_indices]
                dists = self._calculate_crowding_distance(worst_front_vecs)

                # Find index relative to the worst front list
                local_worst_idx = int(np.argmin(dists))
                # Map back to the main challenger list index
                victim_idx = worst_front_indices[local_worst_idx]

                self.non_incumbents.pop(victim_idx)
                continue

            # --- PRUNE FROM INCUMBENTS ---
            # Fallback: If we only have incumbents, remove the least unique one.
            res = self.task.evaluate(self.incumbents, self.predictor, eval_strategy="evaluated")
            vecs = self._get_objective_vectors(res)
            dists = self._calculate_crowding_distance(vecs)

            # Remove the one with the smallest crowding distance
            victim_idx = int(np.argmin(dists))
            self.incumbents.pop(victim_idx)

    def _get_common_blocks(self, prompts: List[Prompt]) -> set:
        """Get the set of block indices that have been evaluated by all given prompts."""
        per_prompt = self.task.get_evaluated_blocks(prompts)  # Dict[prompt -> Set[int]]
        block_sets = list(per_prompt.values())

        if not block_sets:
            return set()

        common = set.intersection(*block_sets)
        return common
    
    def _select_parent_from_pool(self, selection_pool: List[Prompt]) -> Prompt:
        """Tournament-pick a parent, preferring incumbents and using crowding for ties."""
        p1, p2 = random.sample(selection_pool, 2)

        if p1 in self.incumbents and p2 in self.incumbents:
            return self._pick_incumbent_by_crowding(p1, p2)
        if p1 in self.incumbents:
            return p1
        if p2 in self.incumbents:
            return p2

        return random.choice((p1, p2))


    def _pick_incumbent_by_crowding(self, p1: Prompt, p2: Prompt) -> Prompt:
        """Break incumbent ties using crowding distance over common evaluated blocks."""
        res = self.task.evaluate(self.incumbents, self.predictor, eval_strategy="evaluated")
        inc_vectors = self._get_objective_vectors(res)
        inc_distances = self._calculate_crowding_distance(inc_vectors)

        p1_idx = self.incumbents.index(p1)
        p2_idx = self.incumbents.index(p2)
        if inc_distances[p1_idx] > inc_distances[p2_idx]:
            return p1
        if inc_distances[p2_idx] > inc_distances[p1_idx]:
            return p2
        return random.choice((p1, p2))


    def _tournament_selection(self) -> Tuple[Prompt, Prompt]:
        """Pick two distinct parents via tournament selection."""
        selection_pool = self.incumbents + self.non_incumbents
        parent1 = self._select_parent_from_pool(selection_pool)

        parent2 = self._select_parent_from_pool(selection_pool)
        while parent2 == parent1:
            parent2 = self._select_parent_from_pool(selection_pool)

        return parent1, parent2

    @staticmethod
    def _non_dominated_sort(obj_vectors: np.ndarray) -> List[List[int]]:
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
