"""
Implementation of the MO-CAPO (Multi-Objective Cost-Aware Prompt Optimization) algorithm.
Contains the MOCAPOptimizer class, which manages the prompt optimization process using
intensification techniques for multi-objective optimization.

This is the new multi-objective version. The original single-objective CAPO remains
in capo.py for backward compatibility and comparative experiments.
"""

import random
from logging import getLogger
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from promptolution.llms.base_llm import BaseLLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask
from promptolution.utils.prompt import Prompt

from capo.mo_task import MOCAPOClassificationTask
from capo.runhistory import RunHistory
from capo.templates import CROSSOVER_TEMPLATE, FEWSHOT_TEMPLATE, MUTATION_TEMPLATE
from capo.utils import seed_everything

# ### HELPER FUNCTIONS FOR MULTI-OBJECTIVE OPTIMIZATION ###


def fast_non_dominated_sort(obj_vectors: np.ndarray) -> list[list[int]]:
    """
    Performs a fast non-dominated sort on a set of objective vectors.
    This is a standard algorithm from NSGA-II.

    Args:
        obj_vectors: A numpy array of shape (n_solutions, n_objectives).

    Returns:
        A list of fronts, where each front is a list of indices corresponding
        to the input obj_vectors. The first front is the Pareto-optimal set.
    """
    num_solutions = obj_vectors.shape[0]
    if num_solutions == 0:
        return []

    domination_counts = np.zeros(num_solutions, dtype=int)
    dominated_solutions = [[] for _ in range(num_solutions)]
    fronts = [[]]  # The first front

    for i in range(num_solutions):
        for j in range(i + 1, num_solutions):
            # Assumes minimization for all objectives
            is_i_dom_j = np.all(obj_vectors[i] <= obj_vectors[j]) and np.any(
                obj_vectors[i] < obj_vectors[j]
            )
            is_j_dom_i = np.all(obj_vectors[j] <= obj_vectors[i]) and np.any(
                obj_vectors[j] < obj_vectors[i]
            )

            if is_i_dom_j:
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif is_j_dom_i:
                dominated_solutions[j].append(i)
                domination_counts[i] += 1

    # Identify the first front (solutions with domination_count == 0)
    for i in range(num_solutions):
        if domination_counts[i] == 0:
            fronts[0].append(i)

    # Build subsequent fronts
    front_idx = 0
    while fronts and front_idx < len(fronts) and fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        front_idx += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def _calculate_crowding_distance(obj_vectors: np.ndarray) -> np.ndarray:
    """
    Calculates the crowding distance for each solution in a set.
    Used as a tie-breaker for selection and pruning.

    Args:
        obj_vectors: A numpy array of shape (n_solutions, n_objectives).

    Returns:
        A numpy array of shape (n_solutions,) with the crowding distance for each.
    """
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

        for j in range(1, num_solutions - 1):
            distances[sorted_indices[j]] += (
                obj_vectors[sorted_indices[j + 1], i] - obj_vectors[sorted_indices[j - 1], i]
            ) / (f_max - f_min)

    return distances


class CAPOEIRA(BaseOptimizer):
    """
    Multi-Objective Cost-Aware Prompt Optimizer that evolves prompt instructions
    using crossover, mutation, and intensification based on Pareto-optimal fronts
    and dominance relationships.
    """

    def __init__(
        self,
        initial_prompts: List[str],
        task: BaseTask,
        df_few_shots: pd.DataFrame,
        meta_llm: BaseLLM,
        downstream_llm: BaseLLM,
        crossovers_per_iter: int,
        population_size: int,
        upper_shots: int,
        freeze_p_pop: bool = False,
        w_in: float = 1.0,
        w_out: float = 1.0,
        crossover_meta_prompt: str = None,
        mutation_meta_prompt: str = None,
        callbacks: List[Callable] = [],
        predictor: BasePredictor = None,
        verbosity: int = 0,
        logger=getLogger(__name__),
        intensify_vs_all_incumbents: bool = False,
        init_with_intensification: bool = False,
        init_on_all_blocks: bool = False,
        no_weak_dominance: bool = False,
        random_parent_selection: bool = False,
        random_pruning: bool = False,
        random_seed: int = 42,
    ):
        """
        Initializes the MO-CAPO optimizer with parameters for multi-objective prompt evolution.

        Parameters:
            initial_prompts (List[str]): Initial prompt instructions.
            task (BaseTask): The task instance containing dataset and description.
            df_few_shots (pd.DataFrame): DataFrame containing few-shot examples.
            meta_llm (BaseLLM): The meta language model for crossover/mutation.
            downstream_llm (BaseLLM): The downstream language model used for responses.
            crossovers_per_iter (int): Number of crossover operations per iteration.
            population_size (int): Maximum population size for pruning.
            upper_shots (int): Maximum number of few-shot examples per prompt.
            w_in (float): Weight for input tokens in cost calculation.
            w_out (float): Weight for output tokens in cost calculation.
            crossover_meta_prompt (str, optional): Template for crossover instructions.
            mutation_meta_prompt (str, optional): Template for mutation instructions.
            callbacks (List[Callable], optional): Callbacks for optimizer events.
            predictor (BasePredictor, optional): Predictor to evaluate prompt performance.
            verbosity (int, optional): Verbosity level for logging. Defaults to 0.
            logger: Logger instance for debugging and information output.
        """
        assert isinstance(task, MOCAPOClassificationTask), "MOCAPOptimizer requires a MO-CAPO task."

        super().__init__(initial_prompts, task, callbacks, predictor)
        self.df_few_shots = df_few_shots
        self.meta_llm = meta_llm
        self.downstream_llm = downstream_llm

        self.crossover_meta_prompt = crossover_meta_prompt or CROSSOVER_TEMPLATE
        self.mutation_meta_prompt = mutation_meta_prompt or MUTATION_TEMPLATE

        self.population_size = population_size
        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.freeze_p_pop = freeze_p_pop
        self.w_in = w_in
        self.w_out = w_out
        self.verbosity = verbosity
        self.logger = logger
        self.intensify_vs_all_incumbents = intensify_vs_all_incumbents
        self.init_with_intensification = init_with_intensification
        self.init_on_all_blocks = init_on_all_blocks
        self.no_weak_dominance = no_weak_dominance
        self.random_parent_selection = random_parent_selection
        self.random_pruning = random_pruning
        self.random_seed = random_seed
        seed_everything(self.random_seed)

        self.P_inc: List[Prompt] = []
        self.P_pop: List[Prompt] = []
        self.runhistory = RunHistory()

        # Buffers for minimally invasive lineage tracking
        self._selection_details_buffer: List[Dict[str, Any]] = []
        self._crossover_lineage_buffer: List[Dict[str, Any]] = []

        initial_prompt_objects = self._create_initial_prompts(initial_prompts)

        if self.init_with_intensification:
            # Shuffle and intensify each initial prompt
            random.shuffle(initial_prompt_objects)
            for prompt in initial_prompt_objects:
                self._do_intensification(prompt)
        else:
            # Original initialization
            self._initialize_population_and_fronts(initial_prompt_objects)

    def _create_initial_prompts(self, initial_prompts: List[str]) -> List[Prompt]:
        """
        Initializes the population of Prompt objects from initial instructions.

        Parameters:
            initial_prompts (List[str]): List of initial prompt instructions.

        Returns:
            List[Prompt]: Initialized population of prompts with few-shot examples.
        """
        population = []

        for instruction_text in initial_prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = self._create_few_shot_examples(instruction_text, num_examples)
            population.append(Prompt(instruction_text, few_shots))

        if self.verbosity > 0:
            self.logger.warning(
                f"🍿Initialized population with {len(population)} prompts: \n {[p.construct_prompt() for p in population]}"
            )
        return population

    def _initialize_population_and_fronts(self, initial_prompts: List[Prompt]):
        if not initial_prompts:
            return

        prompt_strings = [p.construct_prompt() for p in initial_prompts]

        # Determine which blocks to evaluate on during initialization
        if self.init_on_all_blocks:
            # Evaluate on all blocks
            blocks_to_evaluate = list(range(len(self.task.blocks)))
        else:
            # Default: Evaluate only on a single random block
            block_id = random.randrange(len(self.task.blocks))
            blocks_to_evaluate = [block_id]

        # Evaluate initial prompts on selected blocks
        for block_id in blocks_to_evaluate:
            self.task.evaluate_on_block(
                prompt_strings,
                block_id,
                self.predictor,
                self.runhistory,
            )

        # Compute current vectors (will aggregate across all evaluated blocks)
        all_vectors = np.array(
            [self.runhistory.compute_current_vector(prompt_str) for prompt_str in prompt_strings]
        )
        all_fronts = fast_non_dominated_sort(all_vectors)

        inc_indices = all_fronts[0]
        self.P_inc = [initial_prompts[i] for i in inc_indices]

        for front in all_fronts[1:]:
            self.P_pop.extend([initial_prompts[i] for i in front])

        if self.verbosity > 0:
            self.logger.info(
                f"🌱 Population Initialized. Incumbents: {len(self.P_inc)}, Population: {len(self.P_pop)} "
                f"(evaluated on {len(blocks_to_evaluate)} block(s))"
            )

    def _create_few_shot_examples(
        self, instruction: str, num_examples: int
    ) -> List[Tuple[str, str]]:
        if num_examples == 0:
            return []
        few_shot_samples = self.df_few_shots.sample(num_examples, replace=False)
        sample_inputs = few_shot_samples["input"].values
        sample_targets = few_shot_samples["target"].values
        few_shots = [
            FEWSHOT_TEMPLATE.replace("<input>", i).replace(
                "<output>",
                f"{self.predictor.begin_marker}{t}{self.predictor.end_marker}",
            )
            for i, t in zip(sample_inputs, sample_targets)
        ]
        # Select partition of the examples to generate reasoning from downstream model
        preds, seqs = self.predictor.predict(
            instruction,
            sample_inputs,
            return_seq=True,
        )
        preds, seqs = preds.reshape(num_examples), seqs.reshape(num_examples)

        # Check which predictions are correct and get a single one per example
        for j in range(num_examples):
            # Process and clean up the generated sequences
            seqs[j] = seqs[j].replace(sample_inputs[j], "").strip()
            # Check if the prediction is correct and add reasoning if so
            if preds[j] == sample_targets[j]:
                few_shots[j] = FEWSHOT_TEMPLATE.replace("<input>", sample_inputs[j]).replace(
                    "<output>", seqs[j]
                )

        if self.verbosity > 1:
            self.logger.warning(f"🔫Few-shot examples: {few_shots}")
            self.logger.warning(f"💆‍♂️Generated reasoning: {seqs}")

        return few_shots

    def _is_dominated(self, vec1: np.ndarray, vec2: np.ndarray) -> bool:
        return np.all(vec2 <= vec1) and np.any(vec2 < vec1)

    def _is_weakly_dominated(
        self,
        prompt_a: Prompt,
        prompt_b: Prompt,
        str_a: str | None = None,
        str_b: str | None = None,
    ) -> Tuple[Prompt | None, str]:
        """
        Check weak dominance relationship between two prompts according to ParamILS weak dominance.

        Args:
            prompt_a: First prompt
            prompt_b: Second prompt
            str_a: Optional pre-constructed prompt string for prompt_a (optimization)
            str_b: Optional pre-constructed prompt string for prompt_b (optimization)

        Returns:
            Tuple of (winner, reason) where:
            - winner: The selected prompt or None if no clear winner
            - reason: One of:
                * "better_rank": winner is on a better (lower) front than loser (same blocks)
                * "crowding_distance": same blocks, same front, CD used
                * "weak_dominance": winner weakly dominates loser (different blocks)
                * "no_weak_dominance": no subset relationship exists
        """
        # Use pre-constructed strings if provided, otherwise construct them
        if str_a is None:
            str_a = prompt_a.construct_prompt()
        if str_b is None:
            str_b = prompt_b.construct_prompt()

        # Get evaluated blocks for both prompts (returns Set[int] for single prompt)
        blocks_a = self.runhistory.get_evaluated_blocks(str_a)
        blocks_b = self.runhistory.get_evaluated_blocks(str_b)

        # Ensure we have Set[int] type (type narrowing)
        assert isinstance(blocks_a, set), f"Expected set, got {type(blocks_a)}"
        assert isinstance(blocks_b, set), f"Expected set, got {type(blocks_b)}"

        # CASE 1: Same blocks - use rank (front) then crowding distance
        if blocks_a == blocks_b and len(blocks_a) > 0:
            # Get all P_pop prompts with the same block set
            same_block_prompts = [
                p
                for p in self.P_pop
                if self.runhistory.get_evaluated_blocks(p.construct_prompt()) == blocks_a
            ]

            # Perform NDS to identify fronts
            vectors = np.array(
                [
                    self.runhistory.compute_current_vector(p.construct_prompt(), blocks_a)
                    for p in same_block_prompts
                ]
            )
            fronts = fast_non_dominated_sort(vectors)

            # Find indices of prompt_a and prompt_b in same_block_prompts
            idx_a = same_block_prompts.index(prompt_a)
            idx_b = same_block_prompts.index(prompt_b)

            # Find which front each prompt is on
            rank_a = None
            rank_b = None
            for front_idx, front in enumerate(fronts):
                if idx_a in front:
                    rank_a = front_idx
                if idx_b in front:
                    rank_b = front_idx
                if rank_a is not None and rank_b is not None:
                    break

            # Both prompts must be found in some front
            assert rank_a is not None, f"prompt_a (idx={idx_a}) missing from all fronts"
            assert rank_b is not None, f"prompt_b (idx={idx_b}) missing from all fronts"

            # Select by rank first (lower rank = better front = better prompt)
            if rank_a < rank_b:
                return (prompt_a, "better_rank")
            elif rank_b < rank_a:
                return (prompt_b, "better_rank")
            else:
                # Same rank - use crowding distance
                front = fronts[rank_a]
                front_vectors = vectors[front]
                distances = _calculate_crowding_distance(front_vectors)

                # Find positions of our prompts in the front
                pos_a = front.index(idx_a)
                pos_b = front.index(idx_b)

                # Select based on crowding distance (higher is better for diversity)
                if distances[pos_a] > distances[pos_b]:
                    return (prompt_a, "crowding_distance")
                elif distances[pos_b] > distances[pos_a]:
                    return (prompt_b, "crowding_distance")
                else:
                    # Legitimate tie - both have same crowding distance
                    return (random.choice([prompt_a, prompt_b]), "crowding_distance")

        # CASE 2: Different blocks - check weak dominance (information + performance)
        # Check Information Rule: blocks_b ⊆ blocks_a (a weakly dominates b)
        a_weakly_dominates_b = blocks_b.issubset(blocks_a)
        # Check Information Rule: blocks_a ⊆ blocks_b (b weakly dominates a)
        b_weakly_dominates_a = blocks_a.issubset(blocks_b)

        # Early return if no subset relationship exists
        if not a_weakly_dominates_b and not b_weakly_dominates_a:
            return (None, "no_weak_dominance")

        if a_weakly_dominates_b and len(blocks_b) > 0:
            # Check Performance Rule on shared blocks (which is blocks_b)
            vec_a = self.runhistory.compute_current_vector(str_a, blocks_b)
            vec_b = self.runhistory.compute_current_vector(str_b, blocks_b)

            if self._is_dominated(vec_b, vec_a):  # a weakly dominates b
                return (prompt_a, "weak_dominance")

        if b_weakly_dominates_a and len(blocks_a) > 0:
            # Check Performance Rule on shared blocks (which is blocks_a)
            vec_a = self.runhistory.compute_current_vector(str_a, blocks_a)
            vec_b = self.runhistory.compute_current_vector(str_b, blocks_a)

            if self._is_dominated(vec_a, vec_b):  # b weakly dominates a
                return (prompt_b, "weak_dominance")

        return (None, "no_weak_dominance")

    def _select_parent_from_pool(self, selection_pool: List[Prompt]) -> Prompt:
        """Select one parent from the selection pool using tournament rules."""
        p1, p2 = random.sample(selection_pool, 2)
        details: Dict[str, Any] = {
            "sampled_candidates": [p1.construct_prompt(), p2.construct_prompt()],
            "reason": "",
        }

        p1_is_inc = p1 in self.P_inc
        p2_is_inc = p2 in self.P_inc

        winner = None
        if p1_is_inc and p2_is_inc:
            # Case 1: Both prompts are from incumbent set -> use crowding distance to break ties
            # Use B_common (intersection of all incumbent blocks) for fair comparison
            inc_strings = [inc.construct_prompt() for inc in self.P_inc]
            all_inc_blocks = self.runhistory.get_evaluated_blocks(inc_strings)

            # Ensure we have List[Set[int]] type
            assert isinstance(all_inc_blocks, list), f"Expected list, got {type(all_inc_blocks)}"
            B_common = set.intersection(*all_inc_blocks) if all_inc_blocks else set()

            inc_vectors = np.array(
                [
                    self.runhistory.compute_current_vector(inc_str, B_common)
                    for inc_str in inc_strings
                ]
            )
            inc_distances = _calculate_crowding_distance(inc_vectors)

            # Find the indices of p1 and p2 within the incumbent set
            p1_idx = self.P_inc.index(p1)
            p2_idx = self.P_inc.index(p2)

            if inc_distances[p1_idx] > inc_distances[p2_idx]:
                winner = p1
            elif inc_distances[p2_idx] > inc_distances[p1_idx]:
                winner = p2
            else:
                winner = random.choice([p1, p2])
                details["reason"] = "crowding_distance_tie_inc"

        elif p1_is_inc:
            # Case 2: One is from pop and one from inc -> use incumbent
            winner = p1
            details["reason"] = "incumbent"
        elif p2_is_inc:
            # Case 2: One is from pop and one from inc -> use incumbent
            winner = p2
            details["reason"] = "incumbent"
        else:
            # Case 3: Both are from population
            if self.no_weak_dominance:
                # Ablation: Use random sampling directly
                winner = random.choice([p1, p2])
                details["reason"] = "random"
            else:
                # Use weak dominance with crowding distance tie-breaking
                weak_dom_winner, reason = self._is_weakly_dominated(p1, p2)
                if weak_dom_winner is not None:
                    winner = weak_dom_winner
                    details["reason"] = reason
                else:
                    # No relationship - use random sampling
                    winner = random.choice([p1, p2])
                    details["reason"] = "random"

        details["winner"] = winner.construct_prompt()
        self._selection_details_buffer.append(details)
        return winner

    def _tournament_selection(self) -> Tuple[Prompt, Prompt]:
        selection_pool = self.P_inc + self.P_pop

        parent1 = self._select_parent_from_pool(selection_pool)
        parent2 = self._select_parent_from_pool(selection_pool)

        # Ensure we don't select the same parent twice
        while parent1 == parent2:
            # Remove the last selection details and try again
            self._selection_details_buffer.pop()
            parent2 = self._select_parent_from_pool(selection_pool)

        return parent1, parent2

    def _crossover(self) -> List[Prompt]:
        """
        Performs crossover among parent prompts to generate offsprings.
        """
        self._selection_details_buffer = []  # Clear buffer for this generation
        crossover_prompts = []
        offspring_few_shots = []
        parent_pairs = []
        for _ in range(self.crossovers_per_iter):
            if self.random_parent_selection:
                mother, father = random.sample(self.P_inc + self.P_pop, 2)
                # For random selection, we don't have detailed tournament data
                mother_details = {
                    "sampled_candidates": [mother.construct_prompt()],  # List with single element
                    "reason": "random_selection",
                    "winner": mother.construct_prompt(),
                }
                father_details = {
                    "sampled_candidates": [father.construct_prompt()],
                    "reason": "random_selection",
                    "winner": father.construct_prompt(),
                }
                self._selection_details_buffer.extend([mother_details, father_details])
            else:
                mother, father = self._tournament_selection()
            parent_pairs.append((mother, father))
            crossover_prompt = (
                self.crossover_meta_prompt.replace("<mother>", mother.instruction_text)
                .replace("<father>", father.instruction_text)
                .replace("<task_desc>", self.task.description)
                .strip()
            )
            crossover_prompts.append(crossover_prompt)
            combined_few_shots = mother.few_shots + father.few_shots
            if combined_few_shots:
                num_few_shots = (len(mother.few_shots) + len(father.few_shots)) // 2
                offspring_few_shot = random.sample(combined_few_shots, num_few_shots)
            else:
                offspring_few_shot = []
            offspring_few_shots.append(offspring_few_shot)

        child_instructions = self.meta_llm.get_response(crossover_prompts)
        if self.verbosity > 1:
            self.logger.warning(f"🥐Generated crossover prompts: \n{child_instructions}")

        offsprings = []
        self._crossover_lineage_buffer = []  # Clear/prepare for mutation step
        for i, (instruction, examples) in enumerate(zip(child_instructions, offspring_few_shots)):
            instruction = instruction.split("<prompt>")[-1].split("</prompt>")[0].strip()
            offspring = Prompt(instruction, examples)
            offsprings.append(offspring)

            mother_details = self._selection_details_buffer[i * 2]
            father_details = self._selection_details_buffer[i * 2 + 1]

            lineage = {
                "step": self.runhistory.current_step,
                "mother_selection": mother_details,
                "father_selection": father_details,
            }
            self._crossover_lineage_buffer.append(lineage)

        return offsprings

    def _mutate(self, offsprings: List[Prompt]) -> List[Prompt]:
        """
        Applies mutation to offsprings to generate new candidate prompts.
        """
        mutation_prompts = [
            self.mutation_meta_prompt.replace("<instruction>", prompt.instruction_text).replace(
                "<task_desc>", self.task.description
            )
            for prompt in offsprings
        ]
        new_instructions = self.meta_llm.get_response(mutation_prompts)

        mutated = []
        for i, (new_instruction, original_offspring) in enumerate(
            zip(new_instructions, offsprings)
        ):
            new_instruction = new_instruction.split("<prompt>")[-1].split("</prompt>")[0].strip()
            p = random.random()

            if (
                p < 1 / 3 and len(original_offspring.few_shots) < self.upper_shots
            ):  # add a random few shot
                new_few_shot = self._create_few_shot_examples(new_instruction, 1)
                new_few_shots = original_offspring.few_shots + new_few_shot
            elif (
                1 / 3 <= p < 2 / 3 and len(original_offspring.few_shots) > 0
            ):  # remove a random few shot
                new_few_shots = random.sample(
                    original_offspring.few_shots, len(original_offspring.few_shots) - 1
                )
            else:  # do not change few shots, but shuffle
                new_few_shots = original_offspring.few_shots

            random.shuffle(new_few_shots)
            mutated_prompt = Prompt(new_instruction, new_few_shots)
            mutated.append(mutated_prompt)

            # Log full lineage from the buffer
            lineage_data = self._crossover_lineage_buffer[i]
            lineage_data["offspring"] = original_offspring.construct_prompt()
            self.runhistory.add_lineage(mutated_prompt.construct_prompt(), lineage_data.copy())

        if self.verbosity > 0:
            self.logger.warning(f"🧟Generated {len(mutated)} mutated prompts.")
            self.logger.warning(f"😶Generated Prompts: {[p.construct_prompt() for p in mutated]}")

        return mutated

    def _get_closest_incumbent(self, challenger_vec: np.ndarray):
        """
        Finds the incumbent prompt closest to a challenger in a normalized
        multi-objective space using Euclidean distance. This implementation is
        vectorized for efficiency.

        Normalization bounds (min and max for each objective) are calculated on-demand
        from all evaluations stored in the runhistory to ensure they are up-to-date.

        Args:
            challenger_vec: The challenger's objective vector.
            challenger: The challenger prompt object.

        Returns:
            The incumbent prompt object that is closest to the challenger, or None
            if no incumbents exist.
        """
        if not self.P_inc:
            return None

        # Step 1: Calculate Global Objective Bounds (On-Demand)
        all_obj_vectors = self.runhistory.get_all_objective_vectors()

        if all_obj_vectors.shape[0] < 2:
            return random.choice(self.P_inc)

        min_bounds = np.min(all_obj_vectors, axis=0)
        max_bounds = np.max(all_obj_vectors, axis=0)

        # Step 2: Normalize Vectors
        range_val = max_bounds - min_bounds
        range_val[range_val == 0] = 1.0  # Avoid division by zero

        norm_chal_vec = (challenger_vec - min_bounds) / range_val

        # Vectorized normalization of all incumbent vectors
        inc_vectors = np.array(
            [self.runhistory.compute_current_vector(p.construct_prompt()) for p in self.P_inc]
        )
        norm_inc_vectors = (inc_vectors - min_bounds) / range_val

        # Step 3: Find Closest Incumbent (Vectorized)
        # Calculate Euclidean distance for all incumbents at once
        distances = np.linalg.norm(norm_inc_vectors - norm_chal_vec, axis=1)

        # Find the index of the incumbent with the minimum distance
        closest_inc_idx = np.argmin(distances)
        closest_incumbent = self.P_inc[closest_inc_idx]

        return closest_incumbent

    def _do_intensification(self, challenger: Prompt):
        """
        Implements the MO-CAPO intensification algorithm as defined in Algorithm 3.

        Args:
            challenger: The challenger prompt to intensify
        """

        # Handle the edge case for initialization with intensification when P_inc is empty
        if not self.P_inc:
            # Evaluate the first challenger on a single random block
            challenger_str = challenger.construct_prompt()
            random_block_id = random.choice(range(len(self.task.blocks)))
            self.task.evaluate_on_block(
                [challenger_str], random_block_id, self.predictor, self.runhistory
            )

            # This first prompt becomes the first incumbent
            self.P_inc.append(challenger)
            if self.verbosity > 0:
                self.logger.info(f"🐣 Initializing with first incumbent: {challenger_str[:30]}...")
            return

        # Step 1: Get intersection of all incumbent evaluated blocks (common blocks)
        inc_strings = [inc.construct_prompt() for inc in self.P_inc]
        all_inc_blocks = self.runhistory.get_evaluated_blocks(inc_strings)

        # Ensure we have List[Set[int]] type (since we passed a list of strings)
        assert isinstance(all_inc_blocks, list), f"Expected list, got {type(all_inc_blocks)}"

        # Get intersection of all blocks that ALL incumbents have been evaluated on
        B_common = set.intersection(*all_inc_blocks)

        # Step 2: Initialize challenger evaluation
        challenger_str = challenger.construct_prompt()
        B_eval_challenger = set()  # Blocks challenger has been evaluated on
        new_cost_vector = np.full(2, np.inf)
        # Step 3: Main while loop
        while True:
            # Step 3.1: Set old_cost_vector = new_cost_vector
            old_cost_vector = new_cost_vector.copy()

            # Step 3.2: Repeat-until loop (check conditions at END)
            while True:
                # Step 3.2.2: Sample random block from B_common \ B_eval_challenger
                available_blocks = B_common - B_eval_challenger

                sampled_block = random.choice(list(available_blocks))
                B_eval_challenger.add(sampled_block)

                # Step 3.2.3: Evaluate challenger on sampled block
                eval_results = self.task.evaluate_on_block(
                    [challenger_str], sampled_block, self.predictor, self.runhistory
                )
                obj_vec, _, _ = eval_results[0]

                # Update cost vector incrementally (running average)
                n_evals = len(B_eval_challenger)
                if n_evals == 1:
                    new_cost_vector = obj_vec.copy()
                else:
                    # Correct incremental average: new_avg = (old_avg * (n-1) + new_val) / n
                    new_cost_vector = (old_cost_vector * (n_evals - 1) + obj_vec) / n_evals

                # Step 3.2.5: Check exit conditions
                condition_1 = self._is_dominated(
                    old_cost_vector, new_cost_vector
                )  # new dominates old
                condition_2 = B_eval_challenger == B_common  # evaluated on all common blocks

                if condition_1 or condition_2:
                    break

            # Step 3.2: Check if challenger evaluated on ALL common blocks
            if B_eval_challenger == B_common:
                # Step 3.2.1: Create temporary list with all prompts (don't modify P_inc yet)
                all_prompts = self.P_inc + [challenger]  # Temporary combined list
                all_inc_strings = inc_strings + [challenger_str]  # Reuse inc_strings

                # Step 3.2.2: Perform NDS on all prompts
                inc_vectors = np.array(
                    [
                        self.runhistory.compute_current_vector(prompt_str, B_common)
                        for prompt_str in all_inc_strings
                    ]
                )
                fronts = fast_non_dominated_sort(inc_vectors)

                # Step 3.2.3: Assign fronts correctly (indices now match all_prompts)
                self.P_inc = [all_prompts[i] for i in fronts[0]]  # First front becomes new P_inc
                for front_idx in range(1, len(fronts)):
                    for i in fronts[front_idx]:
                        self._add_to_population(all_prompts[i])  # Dominated fronts to P_pop

                # Prune immediately after updating sets
                self._prune_population()

                # Step 3.2.3: Stop the while loop
                break

            if self.intensify_vs_all_incumbents:
                any_inc_dominates = False
                for inc in self.P_inc:
                    inc_vector = self.runhistory.compute_current_vector(
                        inc.construct_prompt(), B_eval_challenger
                    )
                    if self._is_dominated(new_cost_vector, inc_vector):
                        any_inc_dominates = True
                        break

                if any_inc_dominates:
                    self._add_to_population(challenger)
                    self._prune_population()
                    break
            else:
                # Step 3.3: Get closest incumbent in normalized objective space
                closest_incumbent = self._get_closest_incumbent(new_cost_vector)
                assert closest_incumbent is not None, "There should always be incumbents"

                # Step 3.4: Get closest incumbent's cost vector on challenger's evaluated blocks
                closest_inc_str = closest_incumbent.construct_prompt()
                closest_inc_vector = self.runhistory.compute_current_vector(
                    closest_inc_str, B_eval_challenger
                )

                # Step 3.5: Check if inc domiantes the challenger (on same block subset)
                inc_dominates = self._is_dominated(new_cost_vector, closest_inc_vector)

                # Step 3.6: Decision based on dominance
                if inc_dominates:  # Incumbent dominates or no dominance
                    # Add challenger to P_pop and stop
                    self._add_to_population(challenger)
                    self._prune_population()
                    break

            # Continue while loop if challenger is not dominated (update old_cost_vector at beginning of next iteration)

        # Step 5-6: Incumbent evaluation on additional block
        # Step 5: Get incumbent with least evaluations
        inc_strings = [inc.construct_prompt() for inc in self.P_inc]
        least_evaluated_results = self.runhistory.get_least_evaluated_prompts(inc_strings)

        # Get union of all blocks that ANY incumbent has been evaluated on
        all_inc_blocks = self.runhistory.get_evaluated_blocks(inc_strings)
        assert isinstance(all_inc_blocks, list), f"Expected list, got {type(all_inc_blocks)}"
        union_all_inc_blocks = set().union(*all_inc_blocks)

        # Randomly select from least evaluated incumbents
        chosen_inc_str, chosen_inc_blocks = random.choice(least_evaluated_results)

        # Calculate gap: blocks that other incumbents have evaluated but this one hasn't
        gap_blocks = union_all_inc_blocks - chosen_inc_blocks

        if gap_blocks:
            # Case 1: Catch up - evaluate on a gap block (should be exactly one block)
            new_block = random.choice(list(gap_blocks))

            if self.verbosity > 1:
                self.logger.info(f"📈 Catching up incumbent: evaluating on gap block {new_block}")
        else:
            # Case 2: All incumbents at same level - evaluate on a completely new block
            all_available_blocks = set(range(len(self.task.blocks)))  # All possible blocks
            unevaluated_blocks = all_available_blocks - union_all_inc_blocks

            if not unevaluated_blocks:
                # End case: All incumbents have been evaluated on all available blocks
                if self.verbosity > 0:
                    self.logger.info(
                        "🏁 All incumbents evaluated on all blocks - no further incumbent evaluation needed"
                    )
                return
                # Continue optimization - we can still find better solutions through evolution

            new_block = random.choice(list(unevaluated_blocks))

            if self.verbosity > 0:
                self.logger.info(
                    f"🆕 All incumbents at same level: evaluating on new block {new_block}"
                )

        # Evaluate the chosen incumbent on the selected block
        _ = self.task.evaluate_on_block(
            [chosen_inc_str], new_block, self.predictor, self.runhistory
        )

    def _add_to_population(self, prompt_to_add: Prompt):
        """
        Adds a prompt to P_pop only if it's not already present.
        """
        # Create a set of existing prompt strings in P_pop for efficient checking
        population_strings = {p.construct_prompt() for p in self.P_pop}
        prompt_str = prompt_to_add.construct_prompt()
        if prompt_str not in population_strings:
            self.P_pop.append(prompt_to_add)

    def _prune_population(self):
        while len(self.P_inc) + len(self.P_pop) > self.population_size:
            if self.P_pop:
                if self.random_pruning:
                    worst_idx = random.randrange(len(self.P_pop))
                else:
                    # Compute prompts once per iteration to avoid redundancy
                    prompts = [p.construct_prompt() for p in self.P_pop]

                    # Get eval counts and actual block sets for uniformity check
                    pop_eval_counts = [
                        len(self.runhistory.get_evaluated_blocks(prompt)) for prompt in prompts
                    ]
                    pop_block_sets = [
                        set(
                            self.runhistory.get_evaluated_blocks(prompt)
                        )  # Actual sets for comparison
                        for prompt in prompts
                    ]

                    # Check if all have the SAME block set (not just count)
                    all_same_block_set = (
                        all(bs == pop_block_sets[0] for bs in pop_block_sets)
                        if pop_block_sets
                        else False
                    )

                    if all_same_block_set:
                        # Case: Uniform block sets → Full NDS + CD on entire P_pop
                        all_pop_vectors = np.array(
                            [self.runhistory.compute_current_vector(prompt) for prompt in prompts]
                        )
                        fronts = fast_non_dominated_sort(all_pop_vectors)

                        if fronts:
                            # If only one front (all non-dominated), treat full set as "worst front" for CD pruning
                            if len(fronts) == 1:
                                worst_front = list(range(len(self.P_pop)))  # All indices
                            else:
                                worst_front = fronts[-1]  # Standard: most-dominated front

                            # Now prune from worst_front via CD
                            worst_front_vectors = all_pop_vectors[worst_front]
                            distances = _calculate_crowding_distance(worst_front_vectors)

                            min_distance = np.min(distances)
                            tied_indices = np.where(distances == min_distance)[0]

                            if len(tied_indices) == 1:
                                local_worst = tied_indices[0]
                            else:
                                local_worst = random.choice(tied_indices)

                            worst_idx = worst_front[local_worst]
                        else:
                            # Fail explicitly if no fronts at all (empty P_pop or NDS bug)
                            raise AssertionError(
                                f"No fronts found in P_pop during pruning (empty or NDS failure). "
                                f"P_pop size: {len(self.P_pop)}, vectors shape: {all_pop_vectors.shape if len(all_pop_vectors) > 0 else 'empty'}"
                            )

                    else:
                        # Case: Non-uniform block sets → Random from least evaluated (by count)
                        min_eval_count = min(pop_eval_counts)
                        least_evaluated_indices = [
                            i for i, count in enumerate(pop_eval_counts) if count == min_eval_count
                        ]
                        worst_idx = random.choice(least_evaluated_indices)

                if self.verbosity > 1:
                    self.logger.info(
                        f"Pruning {self.P_pop[worst_idx].instruction_text[:30]}... from P_pop."
                    )

                self.P_pop.pop(worst_idx)

            elif self.P_inc:
                # Prune from P_inc based on worst crowding distance (unchanged)
                prompts = [p.construct_prompt() for p in self.P_inc]
                all_inc_blocks = self.runhistory.get_evaluated_blocks(prompts)

                # Ensure we have List[Set[int]] type
                assert isinstance(
                    all_inc_blocks, list
                ), f"Expected list, got {type(all_inc_blocks)}"
                B_common = set.intersection(*all_inc_blocks) if all_inc_blocks else set()

                obj_vectors = np.array(
                    [self.runhistory.compute_current_vector(prompt, B_common) for prompt in prompts]
                )
                distances = _calculate_crowding_distance(obj_vectors)

                min_distance = np.min(distances)
                tied_indices = np.where(distances == min_distance)[0]

                if len(tied_indices) == 1:
                    worst_idx = tied_indices[0]
                else:
                    worst_idx = random.choice(tied_indices)

                if self.verbosity > 0:
                    self.logger.info(
                        f"Pruning {self.P_inc[worst_idx].instruction_text[:30]}... from P_inc."
                    )

                self.P_inc.pop(worst_idx)
            else:
                break

    def _intensify_challengers(self, challengers: List[Prompt]):
        """
        Routes challengers to intensification based on the optimizer's strategy.
        This version now uses the self.prompts_P_inc and self.prompts_P_pop sets
        which are maintained by the main optimize loop.
        """
        if self.freeze_p_pop:
            # Original behavior: filter for only brand-new prompts
            for challenger in challengers:
                prompt_str = challenger.construct_prompt()
                if not self.runhistory.get_evaluated_blocks(prompt_str):
                    self._do_intensification(challenger)
            return

        # --- Default MO-SMAC-like Logic (freeze_population=False) ---
        # Use the live P_inc to determine incumbency rather than the
        # stale snapshot `self.prompts_P_inc` created at the start of the
        # optimization step. Build a set of current incumbent strings once
        # for efficient membership checks, and also allow direct object
        # identity checks to short-circuit when the exact instance was
        # already added to P_inc.
        current_inc_strings = {p.construct_prompt() for p in self.P_inc}

        for challenger in challengers:
            challenger_str = challenger.construct_prompt()

            # 1) If challenger is already an incumbent (live view), skip
            if challenger_str in current_inc_strings:
                if self.verbosity > 0:
                    self.logger.info(
                        f"Skipping challenger (already incumbent): {challenger_str[:30]}..."
                    )
                continue

            # 2) If challenger string already exists in P_pop, remove all old
            #    objects with the same constructed string and replace them with
            #    the challenger as the canonical representative. This prevents
            #    duplicates across P_inc and P_pop and ensures the challenger
            #    (freshly generated object) is the one that gets intensified.
            if self.P_pop:
                # Filter out any P_pop entries that match the challenger string. We remove
                # stale duplicates so the challenger will not coexist with old objects
                # that have the same constructed prompt. Do NOT append the challenger
                # here — intensification (_do_intensification) will decide whether the
                # challenger becomes an incumbent or a population member. This keeps the
                # evaluation flow faithful to ParamILS/SMAC-like intensification.
                new_pop = [p for p in self.P_pop if p.construct_prompt() != challenger_str]
                if len(new_pop) != len(self.P_pop):
                    if self.verbosity > 1:
                        self.logger.info(
                            f"🔁 Removed {len(self.P_pop) - len(new_pop)} old P_pop object(s) matching challenger: {challenger_str[:30]}..."
                        )
                    # Replace the population with the filtered list. Do NOT append challenger.
                    self.P_pop = new_pop

            # Intensify the challenger (now canonical in P_pop or new)
            if self.verbosity > 0:
                self.logger.info(f"Intensifying challenger: {challenger_str[:30]}...")

            self._do_intensification(challenger)

    def optimize(self, n_steps: int) -> List[str]:
        """
        Main optimization loop that evolves the prompt population.

        Parameters:
            n_steps (int): Number of optimization steps to perform.

        Returns:
            List[str]: The final population of prompts after optimization.
        """

        self.prompts_P_inc = [p.construct_prompt() for p in self.P_inc]
        self.prompts_P_pop = [p.construct_prompt() for p in self.P_pop]

        self._on_step_end()
        self.runhistory.set_current_step()

        for step in range(n_steps):
            if self.verbosity > 0:
                self.logger.info(f"--- Starting Step {step + 1}/{n_steps} ---")

            offsprings = self._crossover()
            mutated_challengers = self._mutate(offsprings)

            self._intensify_challengers(mutated_challengers)

            # Update attributes for callbacks
            self.prompts_P_inc = [p.construct_prompt() for p in self.P_inc]
            self.prompts_P_pop = [p.construct_prompt() for p in self.P_pop]

            continue_optimization = self._on_step_end()
            self.runhistory.set_current_step()
            if not continue_optimization:
                break

        self._on_train_end()

        # Return final Pareto front with detailed information
        final_pareto_front = []
        prompt_strings = []
        for p in self.P_inc:
            prompt_str = p.construct_prompt()
            prompt_strings.append(prompt_str)
            obj_vector = self.runhistory.compute_current_vector(prompt_str)
            total_in, total_out = self.runhistory.get_total_token_counts(prompt_str)
            final_pareto_front.append(
                {
                    "prompt": prompt_str,
                    "objectives": obj_vector.tolist(),
                    "total_input_tokens": total_in,
                    "total_output_tokens": total_out,
                }
            )

        # Return just the prompt strings for compatibility with base class
        return prompt_strings

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        state.pop("predictor", None)
        state.pop("logger", None)
        state.pop("meta_llm", None)
        state.pop("downstream_llm", None)

        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        self.predictor = None
        self.logger = getLogger(__name__)

    def _on_step_end(self) -> bool:
        """
        Override base _on_step_end to only call MO-CAPO compatible callbacks.
        The base implementation expects 'scores' attribute which doesn't exist in MO-CAPO.
        """
        continue_optimization = True
        for callback in self.callbacks:
            # Only call callbacks that are MO-CAPO aware (have on_step_end method)
            if hasattr(callback, "on_step_end"):
                continue_optimization &= callback.on_step_end(self)
        return continue_optimization