import numpy as np
import pandas as pd
import pytest

from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_predictor import MockPredictor
from tests.mocks.mock_task import MockTask

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.tasks.base_task import BaseTask, EvalResult
from promptolution.tasks.multi_objective_task import MultiObjectiveTask
from promptolution.utils.prompt import Prompt


def test_multi_objective_single_prediction_flow():
    task1 = MockTask()
    task2 = MockTask()
    predictor = MockPredictor(llm=MockLLM())

    prompt = Prompt("classify")
    result = MultiObjectiveTask([task1, task2]).evaluate([prompt], predictor=predictor)

    assert len(result.agg_scores) == 2
    assert result.agg_scores[0].shape == (1,)
    assert result.sequences.shape[0] == 1
    assert MultiObjectiveTask([task1, task2]).tasks[0].n_subsamples == task1.n_subsamples


def test_multi_objective_shares_block_and_caches():
    df = pd.DataFrame({"x": ["u", "v"], "y": ["1", "0"]})
    t1 = MockTask(df=df, eval_strategy="sequential_block", n_subsamples=1, n_blocks=len(df), block_idx=0)
    t2 = MockTask(df=df, eval_strategy="sequential_block", n_subsamples=1, n_blocks=len(df), block_idx=0)

    predictor = MockPredictor(llm=MockLLM())
    prompt = Prompt("judge")

    multi = MultiObjectiveTask([t1, t2])
    multi.block_idx = 1
    res = multi.evaluate(prompt, predictor=predictor)

    assert len(t1.eval_cache) == len(t2.eval_cache)
    assert res.input_tokens.shape[0] == 1
    assert multi.prompt_evaluated_blocks[str(prompt)] == {1}


def test_multi_objective_requires_tasks():
    with pytest.raises(ValueError):
        MultiObjectiveTask([])


def test_multi_objective_matches_individual_results():
    df = pd.DataFrame({"x": ["u", "v"], "y": ["1", "0"]})

    def make_task():
        return MockTask(df=df, eval_strategy="sequential_block", n_subsamples=1, n_blocks=len(df), block_idx=0)

    t1 = make_task()
    t2 = make_task()
    predictor = MockPredictor(llm=MockLLM())

    prompt = Prompt("judge")
    multi = MultiObjectiveTask([t1, t2])
    multi.block_idx = 1
    multi_res = multi.evaluate([prompt], predictor=predictor)

    # Fresh tasks/predictor to mirror a single-task call
    s1 = make_task()
    s2 = make_task()
    single_pred = MockPredictor(llm=MockLLM())
    res1 = s1.evaluate([prompt], predictor=single_pred)
    res2 = s2.evaluate([prompt], predictor=single_pred)

    assert np.allclose(multi_res.agg_scores[0], res1.agg_scores)
    assert np.allclose(multi_res.agg_scores[1], res2.agg_scores)
    assert multi_res.sequences.shape == res1.sequences.shape
    assert multi.prompt_evaluated_blocks[str(prompt)] == {1}


class ConstantTask(BaseTask):
    """Simple task that returns a constant score for all predictions."""

    def __init__(self, df: pd.DataFrame, value: float) -> None:
        self._value = value
        super().__init__(
            df=df,
            x_column="x",
            y_column=None,
            n_subsamples=len(df),
            eval_strategy="full",
            seed=0,
            task_description="constant",
            config=None,
        )

    def _evaluate(self, xs, ys, preds):
        return np.full(len(preds), self._value, dtype=float)


class DummyOptimizer(BaseOptimizer):
    """Non-multi-objective optimizer used to trigger fallback logic."""

    def _pre_optimization_loop(self) -> None:
        pass

    def _step(self):
        return self.prompts


def test_multi_objective_fallback_warns_and_averages(caplog):
    df = pd.DataFrame({"x": ["a", "b"]})
    t1 = ConstantTask(df.copy(), value=1.0)
    t2 = ConstantTask(df.copy(), value=3.0)
    mo_task = MultiObjectiveTask([t1, t2])

    predictor = MockPredictor(llm=MockLLM(predetermined_responses=["p1", "p2"]))

    with caplog.at_level("WARNING"):
        DummyOptimizer(predictor=predictor, task=mo_task)

    assert mo_task._scalarized_objective is True
    assert any("averaged equally" in message for message in caplog.messages)

    result = mo_task.evaluate(prompts=[Prompt("p")], predictor=predictor)

    assert isinstance(result, EvalResult)
    assert np.allclose(result.scores, np.array([[2.0, 2.0]]))
    assert np.allclose(result.agg_scores, np.array([2.0]))
