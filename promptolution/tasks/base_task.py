"""Base module for tasks."""


from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union, cast, overload

from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt
from promptolution.utils.token_counter import get_token_counter

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.utils.config import ExperimentConfig


TaskType = Literal["classification", "reward", "judge"]
EvalStrategy = Literal["full", "subsample", "sequential_block", "random_block", "evaluated"]

logger = get_logger(__name__)


class BaseTask(ABC):
    """Abstract base class for tasks in the promptolution library."""

    def __init__(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: Optional[str] = None,
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: "EvalStrategy" = "full",
        seed: int = 42,
        config: Optional["ExperimentConfig"] = None,
    ) -> None:
        """Initialize the BaseTask.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            x_column (str): Name of the column containing input texts.
            y_column (Optional[str]): Name of the column containing labels/ground truth (if applicable).
            task_description (str): Description of the task.
            n_subsamples (int): Number of subsamples to use for evaluation.
            eval_strategy (Literal): Subsampling strategy ("full", "subsample", "sequential_block", "random_block", "evaluated").
            seed (int): Random seed for reproducibility.
            config (ExperimentConfig, optional): Configuration for the task, overriding defaults.
        """
        self.df: pd.DataFrame = df
        self.x_column: str = x_column
        self.y_column: Optional[str] = y_column
        self.task_description: Optional[str] = task_description
        self.n_subsamples: int = n_subsamples
        self.eval_strategy: EvalStrategy = eval_strategy
        self.seed: int = seed

        super().__init__()
        if config is not None:
            config.apply_to(self)

        self.xs: List[str] = df[self.x_column].values.astype(str).tolist()
        self.has_y: bool = y_column is not None
        if self.has_y and y_column is not None:
            self.ys: List[str] = df[y_column].values.astype(str).tolist()
        else:
            # If no y_column is provided, create a dummy y array
            self.ys = [""] * len(self.xs)

        self.block_idx: int = 0
        self.n_blocks: int = len(self.xs) // self.n_subsamples if self.n_subsamples > 0 else 1
        self.rng = np.random.default_rng(seed)

        self.eval_cache: Dict[Tuple[str, str, str], float] = {}  # (prompt, x, y): scores per datapoint
        self.seq_cache: Dict[Tuple[str, str, str], str] = {}  # (prompt, x, y): generating sequence per datapoint

    def subsample(self, eval_strategy: Optional["EvalStrategy"] = None) -> Tuple[List[str], List[str]]:
        """Subsample the dataset based on the specified parameters.

        Args:
            eval_strategy (EvalStrategy, optional): Subsampling strategy to use instead of self.eval_strategy. Defaults to None.

        Returns:
            Tuple[List[str], List[str]]: Subsampled input data and labels.
        """
        if eval_strategy is None:
            eval_strategy = self.eval_strategy

        if eval_strategy in ["full", "evaluated"]:
            return self.xs, self.ys
        elif eval_strategy == "subsample":
            indices = self.rng.choice(len(self.xs), min(self.n_subsamples, len(self.xs)), replace=False)
            return [self.xs[i] for i in indices], [self.ys[i] for i in indices]
        elif eval_strategy == "random_block":
            block_id = self.rng.integers(0, self.n_blocks)
            start_idx = block_id * self.n_subsamples
            end_idx = min((block_id + 1) * self.n_subsamples, len(self.xs))
            indices = np.arange(start_idx, end_idx)
            return [self.xs[i] for i in indices], [self.ys[i] for i in indices]
        elif eval_strategy == "sequential_block":
            start_idx = self.block_idx * self.n_subsamples
            end_idx = min((self.block_idx + 1) * self.n_subsamples, len(self.xs))
            indices = np.arange(start_idx, end_idx)
            return [self.xs[i] for i in indices], [self.ys[i] for i in indices]
        else:
            raise ValueError(f"Unknown subsampling strategy: '{eval_strategy}'")

    def _prepare_batch(
        self,
        prompts: List[Prompt],
        xs: List[str],
        ys: List[str],
        eval_strategy: Literal["full", "subsample", "sequential_block", "random_block", "evaluated"] = "full",
    ) -> List[Tuple[str, str, str]]:
        """Generate (prompt, x, y) keys that require prediction.

        Returns keys not found in eval_cache.
        """
        if eval_strategy == "evaluated":
            return []
        keys_to_predict = []
        for prompt in prompts:
            for x, y in zip(xs, ys):
                cache_key = (str(prompt), x, str(y))
                if cache_key not in self.eval_cache:
                    keys_to_predict.append(cache_key)
        return keys_to_predict

    def _collect_results_from_cache(
        self,
        prompts: List[Prompt],
        xs: List[str],
        ys: List[str],
        return_agg_scores: bool,
        return_seq: bool,
    ) -> Union[List[float], List[List[float]], Tuple[List[List[float]], List[List[str]]]]:
        """Collect all results for the current batch from the cache and format them."""
        assert not (return_agg_scores and return_seq), "Cannot return both aggregated scores and sequences"

        scores = []
        seqs = []

        for prompt in prompts:
            datapoint_scores = []
            datapoint_seqs = []
            for x, y in zip(xs, ys):
                cache_key = (prompt.construct_prompt(), x, y)
                datapoint_score = self.eval_cache.get(cache_key)
                if datapoint_score is None:
                    continue
                datapoint_scores.append(datapoint_score)
                if return_seq:
                    datapoint_seqs.append(self.seq_cache.get(cache_key, ""))
            scores.append(datapoint_scores)
            if return_seq:
                seqs.append(datapoint_seqs)

        if return_agg_scores:
            agg_scores = [np.nanmean(s).item() for s in scores]
            return agg_scores

        return scores if not return_seq else (scores, seqs)

    @abstractmethod
    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> List[float]:
        """Abstract method to calculate the score for a predictions.

        This method should be implemented by subclasses based on their specific evaluation logic.
        """
        raise NotImplementedError

    @overload
    def evaluate(
        self,
        prompts: List[Prompt],
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[True] = True,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[False] = False,
    ) -> List[float]:
        ...

    @overload
    def evaluate(
        self,
        prompts: List[Prompt],
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[False] = False,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[False] = False,
    ) -> List[List[float]]:
        ...

    @overload
    def evaluate(
        self,
        prompts: List[Prompt],
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[False] = False,
        return_seq: Literal[True] = True,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[False] = False,
    ) -> Tuple[List[List[float]], List[List[str]]]:
        ...

    @overload
    def evaluate(
        self,
        prompts: Prompt,
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[True] = True,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[False] = False,
    ) -> List[float]:
        ...

    @overload
    def evaluate(
        self,
        prompts: Prompt,
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[False] = False,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[False] = False,
    ) -> List[List[float]]:
        ...

    @overload
    def evaluate(
        self,
        prompts: Prompt,
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[False] = False,
        return_seq: Literal[True] = True,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[False] = False,
    ) -> Tuple[List[List[float]], List[List[str]]]:
        ...

    @overload
    def evaluate(
        self,
        prompts: List[Prompt],
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[True] = True,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[True] = True,
    ) -> Tuple[List[float], List[float], List[float]]:
        ...

    @overload
    def evaluate(
        self,
        prompts: List[Prompt],
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[False] = False,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[True] = True,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        ...

    @overload
    def evaluate(
        self,
        prompts: Prompt,
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[True] = True,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[True] = True,
    ) -> Tuple[List[float], List[float], List[float]]:
        ...

    @overload
    def evaluate(
        self,
        prompts: Prompt,
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: Literal[False] = False,
        return_seq: Literal[False] = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: Literal[True] = True,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        ...

    def evaluate(
        self,
        prompts: Union[Prompt, List[Prompt]],
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        return_agg_scores: bool = True,
        return_seq: bool = False,
        eval_strategy: Optional["EvalStrategy"] = None,
        return_costs: bool = False,
    ) -> Union[
        List[float],
        List[List[float]],
        Tuple[List[List[float]], List[List[str]]],
        Tuple[List[float], List[float], List[float]],
        Tuple[List[List[float]], List[List[float]], List[List[float]]],
    ]:
        """Evaluate a set of prompts using a given predictor.

        This method orchestrates subsampling, prediction, caching, and result collection.

        Note: Cannot return both aggregated scores and sequences (assertion will fail).
        """
        assert not (return_agg_scores and return_seq), "Cannot return both aggregated scores and sequences"
        assert not return_seq or not return_costs, "Token cost reporting is not supported together with sequences."

        prompts = [prompts] if isinstance(prompts, Prompt) else prompts
        eval_strategy = eval_strategy or self.eval_strategy
        xs, ys = self.subsample(eval_strategy=eval_strategy)
        batches = self._prepare_batch(prompts, xs, ys, eval_strategy=eval_strategy)
        (prompts_to_evaluate, xs_to_evaluate, ys_to_evaluate) = ([], [], []) if not batches else zip(*batches)

        if prompts_to_evaluate:
            preds_seqs = predictor.predict(
                prompts=list(prompts_to_evaluate),
                xs=list(xs_to_evaluate),
                system_prompts=system_prompts,
                return_seq=return_seq,
            )
        else:
            preds_seqs = ([], []) if return_seq else []

        seqs: List[str] = []
        if return_seq:
            preds, seqs = preds_seqs if isinstance(preds_seqs, tuple) else (preds_seqs, [])
        else:
            preds = cast(List[str], preds_seqs)

        scores: List[float] = self._evaluate(list(xs_to_evaluate), list(ys_to_evaluate), preds)
        for i, cache_key in enumerate(batches):
            self.eval_cache[cache_key] = scores[i]
            if return_seq:
                self.seq_cache[cache_key] = seqs[i]

        agg_scores = self._collect_results_from_cache(
            prompts,
            xs,
            ys,
            return_agg_scores,
            return_seq,
        )

        if not return_costs:
            return agg_scores

        token_counter = get_token_counter(predictor.llm)

        per_prompt_inputs: List[List[float]] = []
        per_prompt_outputs: List[List[float]] = []

        input_token_counts = [float(token_counter(x)) for x in xs]

        for idx, prompt in enumerate(prompts):
            prompt_tokens = float(token_counter(prompt.construct_prompt()))
            start = idx * len(xs)
            end = (idx + 1) * len(xs)
            preds_for_prompt = preds[start:end]
            output_token_counts = [float(token_counter(p)) for p in preds_for_prompt]

            prompt_input_tokens = [prompt_tokens + input_toks for input_toks in input_token_counts]
            per_prompt_inputs.append(prompt_input_tokens)
            per_prompt_outputs.append(output_token_counts)

        if return_agg_scores:
            agg_scores_list = cast(List[float], agg_scores)
            per_prompt_inputs_mean = [float(np.mean(tokens)) for tokens in per_prompt_inputs]
            per_prompt_outputs_mean = [float(np.mean(tokens)) for tokens in per_prompt_outputs]
            return agg_scores_list, per_prompt_inputs_mean, per_prompt_outputs_mean

        score_matrix = cast(List[List[float]], agg_scores)
        return score_matrix, per_prompt_inputs, per_prompt_outputs

    def pop_datapoints(self, n: Optional[int] = None, frac: Optional[float] = None) -> pd.DataFrame:
        """Pop a number of datapoints from the dataset.

        Args:
            n (int, optional): Number of datapoints to pop. Defaults to None.
            frac (float, optional): Fraction of datapoints to pop. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the popped datapoints.
        """
        assert n is None or frac is None, "Only one of n or frac can be specified."
        if n is not None:
            indices = self.rng.choice(len(self.xs), n, replace=False)
        elif frac is not None:
            indices = self.rng.choice(len(self.xs), int(len(self.xs) * frac), replace=False)
        else:
            raise ValueError("Either n or frac must be specified.")

        popped_xs = [self.xs[i] for i in indices]
        popped_ys = [self.ys[i] for i in indices]
        df_popped = pd.DataFrame({self.x_column: popped_xs, self.y_column: popped_ys})

        self.xs = [x for i, x in enumerate(self.xs) if i not in indices]
        self.ys = [y for i, y in enumerate(self.ys) if i not in indices]

        # Update n_blocks and block_idx based on the new dataset size
        self.n_blocks = len(self.xs) // self.n_subsamples if self.n_subsamples > 0 else 1
        self.block_idx = min(self.block_idx, self.n_blocks - 1) if self.n_blocks > 0 else 0

        # Clear cache for popped items (optional, but good practice if memory is a concern)
        keys_to_remove = []
        for key in self.eval_cache:
            if key[1] in popped_xs and key[2] in popped_ys:  # Check if the x and y correspond to popped data
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.eval_cache.pop(key, None)
            self.seq_cache.pop(key, None)

        return df_popped

    def increment_block_idx(self) -> None:
        """Increment the block index for subsampling.

        Raises:
            ValueError: If the eval_strategy does not contain "block".
        """
        if "block" not in self.eval_strategy:
            raise ValueError("Block increment is only valid for block subsampling.")
        self.block_idx += 1
        if self.n_blocks > 0:  # Ensure n_blocks is not zero to avoid division by zero
            self.block_idx %= self.n_blocks
        else:
            self.block_idx = 0  # If no blocks, reset to 0

    def reset_block_idx(self) -> None:
        """Reset the block index for subsampling.

        Raises:
            ValueError: If the eval_strategy does not contain "block".
        """
        if "block" not in self.eval_strategy:
            raise ValueError("Block reset is only valid for block subsampling.")
        self.block_idx = 0
