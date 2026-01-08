"""Base module for tasks."""


from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt
from promptolution.utils.token_counter import get_token_counter

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.utils.config import ExperimentConfig


TaskType = Literal["classification", "reward", "judge", "multi"]
EvalStrategy = Literal["full", "subsample", "sequential_block", "random_block", "evaluated"]

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Evaluation outputs including scores, sequences, and costs."""

    scores: np.ndarray  # shape: (n_prompts, n_datapoints)
    agg_scores: np.ndarray  # shape: (n_prompts,) - mean over datapoints
    sequences: np.ndarray  # shape: (n_prompts, n_datapoints)
    input_tokens: np.ndarray  # shape: (n_prompts, n_datapoints)
    output_tokens: np.ndarray  # shape: (n_prompts, n_datapoints)
    agg_input_tokens: np.ndarray  # shape: (n_prompts,) - mean over datapoints
    agg_output_tokens: np.ndarray  # shape: (n_prompts,) - mean over datapoints


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
        self.df = df.drop_duplicates(subset=[x_column])
        if len(self.df) != len(df):
            logger.warning(
                f"Duplicate entries detected for x_column '{x_column}' - dropped {len(df) - len(self.df)} rows to enforce uniqueness."
            )
        self.x_column: str = x_column
        self.y_column: Optional[str] = y_column
        self.task_type: TaskType | None = None
        self.task_description: Optional[str] = task_description
        self.n_subsamples: int = n_subsamples
        self.eval_strategy: EvalStrategy = eval_strategy
        self.seed: int = seed

        super().__init__()
        if config is not None:
            config.apply_to(self)

        self.xs: List[str] = self.df[self.x_column].values.astype(str).tolist()
        self.has_y: bool = y_column is not None
        if self.has_y and y_column is not None:
            self.ys: List[str] = self.df[y_column].values.astype(str).tolist()
        else:
            # If no y_column is provided, create a dummy y array
            self.ys = [""] * len(self.xs)

        self.block_idx: int | list[int] = 0
        self.n_blocks: int = len(self.xs) // self.n_subsamples if self.n_subsamples > 0 else 1
        self.rng = np.random.default_rng(seed)

        self.eval_cache: Dict[Tuple[str, str, str], float] = {}  # (prompt, x, y): scores per datapoint
        self.seq_cache: Dict[Tuple[str, str, str], str] = {}  # (prompt, x, y): raw model output per datapoint

        self.prompt_evaluated_blocks: Dict[str, set[int]] = {}  # prompt_str: set of evaluated block indices

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
            if isinstance(self.block_idx, list):
                block_indices: List[int] = []
                for block_id in self.block_idx:
                    start_idx = block_id * self.n_subsamples
                    end_idx = min((block_id + 1) * self.n_subsamples, len(self.xs))
                    block_indices.extend(range(start_idx, end_idx))
                indices = np.array(sorted(set(block_indices)), dtype=int)
            else:
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
    ) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str, str]]]:
        """Return uncached prompt/x/y triples for prediction and their cache keys."""
        if eval_strategy == "evaluated":
            return [], [], [], []

        prompts_to_predict: List[str] = []
        xs_to_predict: List[str] = []
        ys_to_predict: List[str] = []
        keys_to_predict: List[Tuple[str, str, str]] = []

        for prompt in prompts:
            for x, y in zip(xs, ys):
                cache_key = (str(prompt), x, str(y))
                if cache_key in self.eval_cache:
                    continue
                prompts_to_predict.append(str(prompt))
                xs_to_predict.append(x)
                ys_to_predict.append(str(y))
                keys_to_predict.append(cache_key)

        return prompts_to_predict, xs_to_predict, ys_to_predict, keys_to_predict

    @staticmethod
    def _cache_key(prompt: Prompt, x: str, y: str) -> Tuple[str, str, str]:
        return (prompt.construct_prompt(), x, y)

    def _collect_results_from_cache(
        self, prompts: List[Prompt], xs: List[str], ys: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect cached scores and sequences for provided prompts/xs/ys."""
        score_rows: List[List[float]] = []
        seq_rows: List[List[str]] = []

        for prompt in prompts:
            datapoint_scores: List[float] = []
            datapoint_seqs: List[str] = []
            for x, y in zip(xs, ys):
                cache_key = self._cache_key(prompt, x, str(y))
                datapoint_score = self.eval_cache.get(cache_key, np.nan)
                datapoint_scores.append(datapoint_score)
                datapoint_seqs.append(self.seq_cache.get(cache_key, ""))
            score_rows.append(datapoint_scores)
            seq_rows.append(datapoint_seqs)

        scores_array = np.array(score_rows, dtype=float)
        agg_scores = np.nanmean(scores_array, axis=1) if scores_array.size else np.array([])
        seqs_array = np.array(seq_rows, dtype=object)
        return scores_array, agg_scores, seqs_array

    def _compute_costs(
        self,
        prompts: List[Prompt],
        xs: List[str],
        ys: List[str],
        seq_cache: Dict[Tuple[str, str, str], str],
        predictor: "BasePredictor",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        token_counter = get_token_counter(predictor.llm)

        per_prompt_inputs: List[np.ndarray] = []
        per_prompt_outputs: List[np.ndarray] = []

        input_token_counts = np.array([token_counter(x) for x in xs], dtype=float)

        for prompt in prompts:
            prompt_tokens = token_counter(prompt.construct_prompt())
            seq_token_counts: List[float] = []
            for x, y in zip(xs, ys):
                cache_key = self._cache_key(prompt, x, str(y))
                seq_text = seq_cache.get(cache_key, "")
                seq_token_counts.append(token_counter(seq_text))

            prompt_input_tokens = prompt_tokens + input_token_counts
            output_token_counts = np.maximum(np.array(seq_token_counts, dtype=float) - input_token_counts, 0.0)

            per_prompt_inputs.append(np.asarray(prompt_input_tokens, dtype=float))
            per_prompt_outputs.append(output_token_counts)

        inputs_array = np.vstack(per_prompt_inputs)
        outputs_array = np.vstack(per_prompt_outputs)

        agg_input_tokens = inputs_array.mean(axis=1) if inputs_array.size else np.array([])
        agg_output_tokens = outputs_array.mean(axis=1) if outputs_array.size else np.array([])

        return inputs_array, outputs_array, agg_input_tokens, agg_output_tokens

    @abstractmethod
    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> np.ndarray:
        """Abstract method to calculate the score for a predictions.

        This method should be implemented by subclasses based on their specific evaluation logic.
        """
        raise NotImplementedError

    def activate_scalarized_objective(self) -> None:
        """Activate scalarized objective for multi-objective tasks."""
        raise NotImplementedError

    def evaluate(
        self,
        prompts: Union[Prompt, List[Prompt]],
        predictor: "BasePredictor",
        system_prompts: Optional[Union[str, List[str]]] = None,
        eval_strategy: Optional["EvalStrategy"] = None,
    ) -> EvalResult:
        """Evaluate a set of prompts using a given predictor.

        This method orchestrates subsampling, prediction, caching, and result collection.
        Sequences, token costs, raw scores, and aggregated scores are always returned.
        """
        prompts_list: List[Prompt] = [prompts] if isinstance(prompts, Prompt) else list(prompts)
        eval_strategy = eval_strategy or self.eval_strategy
        xs, ys = self.subsample(eval_strategy=eval_strategy)
        (
            prompts_to_evaluate,
            xs_to_evaluate,
            ys_to_evaluate,
            cache_keys,
        ) = self._prepare_batch(prompts_list, xs, ys, eval_strategy=eval_strategy)

        preds, pred_seqs = predictor.predict(
            prompts=prompts_to_evaluate,
            xs=xs_to_evaluate,
            system_prompts=system_prompts,
        )

        scores = self._evaluate(xs_to_evaluate, ys_to_evaluate, preds)
        for i, cache_key in enumerate(cache_keys):
            self.eval_cache[cache_key] = scores[i]
            self.seq_cache[cache_key] = str(pred_seqs[i])

        scores, agg_scores, seqs = self._collect_results_from_cache(
            prompts_list,
            xs,
            ys,
        )

        # Record evaluated block for block strategies
        for prompt in prompts_list:
            if isinstance(self.block_idx, list):
                self.prompt_evaluated_blocks.setdefault(str(prompt), set()).update(self.block_idx)
            else:
                self.prompt_evaluated_blocks.setdefault(str(prompt), set()).add(self.block_idx)

        input_tokens, output_tokens, agg_input_tokens, agg_output_tokens = self._compute_costs(
            prompts_list, xs, ys, self.seq_cache, predictor
        )

        return EvalResult(
            scores=scores,
            agg_scores=agg_scores,
            sequences=seqs,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agg_input_tokens=agg_input_tokens,
            agg_output_tokens=agg_output_tokens,
        )

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
        assert isinstance(self.block_idx, int), "Block index must be an integer to increment."
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

    def set_block_idx(self, idx: Union[int, List[int]]) -> None:
        """Set the block index (or indices) for block subsampling strategies."""
        if "block" not in self.eval_strategy:
            raise ValueError("Block assignment is only valid for block subsampling.")

        if isinstance(idx, list):
            assert all(0 <= i < self.n_blocks for i in idx), "Block indices must be integers within valid range"
        else:
            assert isinstance(idx, int), "Block index must be an integer"

        self.block_idx = idx

    def get_evaluated_blocks(self, prompts: List[Prompt]) -> Dict[str, set[int]]:
        """Return mapping of prompt string to evaluated block indices."""
        return {str(p): set(self.prompt_evaluated_blocks.get(str(p), set())) for p in prompts}
