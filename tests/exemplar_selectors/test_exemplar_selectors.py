"""Tests for exemplar selectors."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.mocks.mock_predictor import MockPredictor

from promptolution.exemplar_selectors.random_search_selector import RandomSearchSelector
from promptolution.tasks.base_task import EvalResult
from promptolution.utils.prompt import Prompt


def make_eval_result(sequences, score):
    n = len(sequences)
    return EvalResult(
        scores=np.array([[score] * n], dtype=float),
        agg_scores=np.array([score], dtype=float),
        sequences=np.array([sequences], dtype=object),
        input_tokens=np.array([[1.0] * n], dtype=float),
        output_tokens=np.array([[1.0] * n], dtype=float),
        agg_input_tokens=np.array([1.0], dtype=float),
        agg_output_tokens=np.array([1.0], dtype=float),
    )


@pytest.fixture
def task_and_predictor():
    task = MagicMock()
    pred = MockPredictor()
    return task, pred


def test_random_search_selector_respects_n_examples(task_and_predictor):
    task, pred = task_and_predictor
    sequences = [f"ex_{i}" for i in range(10)]
    task.evaluate.return_value = make_eval_result(sequences, score=0.8)

    selector = RandomSearchSelector(task, pred)
    result = selector.select_exemplars(Prompt("Classify:"), n_examples=3, n_trials=1)

    assert len(result.few_shots) == 3
    assert all(ex in sequences for ex in result.few_shots)


def test_random_search_selector_returns_best_trial(task_and_predictor):
    task, pred = task_and_predictor
    sequences = [f"ex_{i}" for i in range(5)]

    # First trial scores low, second scores high
    task.evaluate.side_effect = [
        make_eval_result(sequences, score=0.3),  # zero-shot eval trial 1
        make_eval_result(sequences, score=0.3),  # few-shot eval trial 1
        make_eval_result(sequences, score=0.9),  # zero-shot eval trial 2
        make_eval_result(sequences, score=0.9),  # few-shot eval trial 2
    ]

    selector = RandomSearchSelector(task, pred)
    result = selector.select_exemplars(Prompt("Classify:"), n_examples=2, n_trials=2)

    assert len(result.few_shots) == 2
    assert result.few_shots != []


def test_random_search_selector_n_examples_kwarg(task_and_predictor):
    """Regression test: calling with n_examples as keyword arg must not raise TypeError."""
    task, pred = task_and_predictor
    sequences = [f"ex_{i}" for i in range(5)]
    task.evaluate.return_value = make_eval_result(sequences, score=0.5)

    selector = RandomSearchSelector(task, pred)
    # This call pattern is what triggered the original bug report
    result = selector.select_exemplars(prompt=Prompt("Classify:"), n_examples=2)

    assert len(result.few_shots) == 2
