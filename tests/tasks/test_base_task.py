import numpy as np
import pandas as pd
import pytest

from tests.mocks.dummy_config import DummyConfig
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_predictor import MockPredictor
from tests.mocks.mock_task import MockTask

from promptolution.tasks.base_task import BaseTask
from promptolution.utils.prompt import Prompt


@pytest.fixture
def predictor():
    return MockPredictor(llm=MockLLM())


@pytest.fixture
def small_task():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["1", "0", "1"]})
    return MockTask(df=df, eval_strategy="sequential_block", n_subsamples=1)


@pytest.fixture
def cost_task():
    df = pd.DataFrame({"x": ["m", "n", "o"], "y": ["1", "0", "1"]})
    return MockTask(df=df, eval_strategy="full", n_subsamples=3)


def test_subsample_and_block_controls(small_task):
    task = small_task

    xs, ys = task.subsample()
    assert len(xs) == 1

    task.increment_block_idx()
    assert task.block_idx == 1 % task.n_blocks if task.n_blocks else 0

    task.set_block_idx([0, 1, 2])
    xs2, _ = task.subsample()
    assert set(xs2) == set(task.xs)

    task.set_block_idx(0)
    popped = task.pop_datapoints(n=1)
    assert len(popped) == 1
    assert len(task.xs) == 2

    task.reset_block_idx()
    assert task.block_idx == 0

    task.eval_strategy = "full"
    with pytest.raises(ValueError):
        task.increment_block_idx()
    with pytest.raises(ValueError):
        task.reset_block_idx()


def test_prepare_batch_and_evaluated_strategy(small_task):
    task = small_task
    prompts = [Prompt("p1"), Prompt("p2")]
    xs, ys = task.subsample()

    to_eval = task._prepare_batch(prompts, xs, ys, eval_strategy="evaluated")
    assert to_eval == ([], [], [], [])

    normal = task._prepare_batch(prompts, xs, ys)
    assert len(normal[0]) == len(prompts) * len(xs)


def test_pop_datapoints_clears_cache_and_frac(small_task):
    task = small_task
    p = Prompt("p")
    key = (str(p), task.xs[0], task.ys[0])
    task.eval_cache[key] = 0.5
    task.seq_cache[key] = "seq"

    popped = task.pop_datapoints(frac=0.5)
    assert len(popped) > 0
    assert not task.eval_cache
    assert not task.seq_cache


def test_unknown_strategy_raises(small_task):
    task = small_task
    task.eval_strategy = "unknown"
    with pytest.raises(ValueError):
        task.subsample()


def test_set_block_idx_validation(small_task):
    task = small_task
    with pytest.raises(AssertionError):
        task.set_block_idx("bad")  # type: ignore


def test_pop_datapoints_requires_arg(small_task):
    task = small_task
    with pytest.raises(AssertionError):
        task.pop_datapoints(n=1, frac=0.1)


def test_get_evaluated_blocks_mapping(small_task):
    task = small_task
    prompt = Prompt("p")
    task.prompt_evaluated_blocks[str(prompt)] = {0, 1}
    mapping = task.get_evaluated_blocks([prompt])
    assert mapping[str(prompt)] == {0, 1}


def test_compute_costs_shapes(predictor, cost_task):
    task = cost_task
    prompts = [Prompt("inst"), Prompt("inst2")]
    result = task.evaluate(prompts, predictor)

    assert result.input_tokens.shape[0] == len(prompts)
    assert result.output_tokens.shape[0] == len(prompts)


def test_evaluate_with_block_list_updates_blocks(predictor, small_task):
    task = small_task
    task.block_idx = [0, 1]
    prompts = [Prompt("p1"), Prompt("p2")]
    task.evaluate(prompts, predictor)
    for p in prompts:
        assert task.prompt_evaluated_blocks[str(p)] == {0, 1}


def test_task_config_applied():
    cfg = DummyConfig()
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["1", "0", "1"]})
    task = MockTask(df=df, eval_strategy="sequential_block", n_subsamples=1, config=cfg)
    assert cfg.applied is True
    assert hasattr(task, "config_applied")


def test_block_wraparound_and_get_cache_keys():
    df = pd.DataFrame({"x": ["a", "b"], "y": ["1", "0"]})
    task = MockTask(df=df, eval_strategy="sequential_block", n_subsamples=1)
    task.block_idx = task.n_blocks - 1
    task.increment_block_idx()
    assert task.block_idx == 0

    prompt = Prompt("hi")
    key = task._cache_key(prompt, "x", "y")
    assert key[0].startswith(prompt.instruction)