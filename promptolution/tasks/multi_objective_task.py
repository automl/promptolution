"""Multi-objective task wrapper that evaluates prompts across multiple tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import Dict, List, Optional, Tuple

from promptolution.tasks.base_task import BaseTask, EvalResult, EvalStrategy
from promptolution.utils.prompt import Prompt


@dataclass
class MultiObjectiveEvalResult:
    """Container for per-task evaluation outputs in multi-objective runs."""

    scores: List[np.ndarray]
    agg_scores: List[np.ndarray]
    sequences: np.ndarray
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    agg_input_tokens: np.ndarray
    agg_output_tokens: np.ndarray


class MultiObjectiveTask(BaseTask):
    """A task that aggregates evaluations across multiple underlying tasks."""

    def __init__(
        self,
        tasks: List[BaseTask],
        eval_strategy: Optional[EvalStrategy] = None,
    ) -> None:
        """Initialize with a list of tasks sharing subsampling and seed settings."""
        if not tasks:
            raise ValueError("tasks must be a non-empty list")

        primary = tasks[0]
        for t in tasks[1:]:
            assert t.n_subsamples == primary.n_subsamples, "All tasks must share n_subsamples"
            assert t.seed == primary.seed, "All tasks must share seed"
            assert t.eval_strategy == primary.eval_strategy, "All tasks must share eval_strategy"

        combined_description = "This task is a combination of the following tasks:\n" + "\n".join(
            [f"Task: {t.task_description}" for t in tasks if t.task_description]
        )

        super().__init__(
            df=primary.df,
            x_column=primary.x_column,
            y_column=primary.y_column,
            task_description=combined_description,
            n_subsamples=primary.n_subsamples,
            eval_strategy=eval_strategy or primary.eval_strategy,
            seed=primary.seed,
            config=None,
        )
        self.task_type = "multi"
        self.tasks = tasks
        self._scalarized_objective: bool = False

    def activate_scalarized_objective(self) -> None:
        """Force single-objective behavior by equally averaging task scores."""
        self._scalarized_objective = True

    def evaluate(  # type: ignore
        self,
        prompts: Prompt | List[Prompt],
        predictor,
        system_prompts: Optional[str | List[str]] = None,
        eval_strategy: Optional[EvalStrategy] = None,
    ) -> MultiObjectiveEvalResult | EvalResult:
        """Run prediction once, then score via each task's _evaluate."""
        prompts_list: List[Prompt] = [prompts] if isinstance(prompts, Prompt) else list(prompts)
        strategy = eval_strategy or self.eval_strategy

        # Keep block alignment across tasks so block-based strategies stay in sync.
        for task in self.tasks:
            task.block_idx = self.block_idx

        xs, ys = self.subsample(eval_strategy=strategy)

        # Collect all uncached prompt/x/y triples across tasks to predict only once.
        prompts_to_evaluate: List[str] = []
        xs_to_evaluate: List[str] = []
        ys_to_evaluate: List[str] = []
        key_to_index: Dict[Tuple[str, str, str], int] = {}
        cache_keys: List[Tuple[str, str, str]] = []

        for task in self.tasks:
            t_prompts, t_xs, t_ys, t_keys = task._prepare_batch(prompts_list, xs, ys, eval_strategy=strategy)
            for prompt_str, x_val, y_val, key in zip(t_prompts, t_xs, t_ys, t_keys):
                if key in key_to_index:
                    continue
                key_to_index[key] = len(prompts_to_evaluate)
                prompts_to_evaluate.append(prompt_str)
                xs_to_evaluate.append(x_val)
                ys_to_evaluate.append(y_val)
                cache_keys.append(key)

        preds: List[str] = []
        pred_seqs: List[str] = []
        if prompts_to_evaluate:
            preds, pred_seqs = predictor.predict(
                prompts=prompts_to_evaluate,
                xs=xs_to_evaluate,
                system_prompts=system_prompts,
            )

        # Map predictions back to each task and populate caches via _evaluate.
        key_to_pred: Dict[Tuple[str, str, str], Tuple[str, str]] = {
            key: (preds[idx], pred_seqs[idx]) for key, idx in key_to_index.items()
        }

        per_task_results: List[EvalResult] = []
        for task in self.tasks:
            if cache_keys:
                xs_eval = [k[1] for k in cache_keys]
                ys_eval = [k[2] for k in cache_keys]
                preds_eval = [key_to_pred[k][0] for k in cache_keys]
                scores = task._evaluate(xs_eval, ys_eval, preds_eval)
                for score, cache_key in zip(scores, cache_keys):
                    task.eval_cache[cache_key] = score
                    task.seq_cache[cache_key] = key_to_pred[cache_key][1]

            scores_array, agg_scores, seqs = task._collect_results_from_cache(prompts_list, xs, ys)
            input_tokens, output_tokens, agg_input_tokens, agg_output_tokens = task._compute_costs(
                prompts_list, xs, ys, predictor
            )

            per_task_results.append(
                EvalResult(
                    scores=scores_array,
                    agg_scores=agg_scores,
                    sequences=seqs,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    agg_input_tokens=agg_input_tokens,
                    agg_output_tokens=agg_output_tokens,
                )
            )

        stacked_scores = [r.scores for r in per_task_results]
        stacked_agg_scores = [r.agg_scores for r in per_task_results]

        # Mirror evaluated block bookkeeping using the first task for parity with BaseTask.
        first_task = self.tasks[0]
        first_result = per_task_results[0]
        self.prompt_evaluated_blocks = {p: first_task.prompt_evaluated_blocks[p] for p in prompts_list}

        if self._scalarized_objective:
            return EvalResult(
                scores=np.mean(stacked_scores, axis=0),
                agg_scores=np.mean(stacked_agg_scores, axis=0),
                sequences=first_result.sequences,
                input_tokens=first_result.input_tokens,
                output_tokens=first_result.output_tokens,
                agg_input_tokens=first_result.agg_input_tokens,
                agg_output_tokens=first_result.agg_output_tokens,
            )

        return MultiObjectiveEvalResult(
            scores=stacked_scores,
            agg_scores=stacked_agg_scores,
            sequences=first_result.sequences,
            input_tokens=first_result.input_tokens,
            output_tokens=first_result.output_tokens,
            agg_input_tokens=first_result.agg_input_tokens,
            agg_output_tokens=first_result.agg_output_tokens,
        )

    def _evaluate(self, xs, ys, preds):  # pragma: no cover
        raise NotImplementedError("MultiObjectiveTask overrides evaluate directly")
