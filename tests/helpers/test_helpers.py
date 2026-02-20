from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_predictor import MockPredictor
from tests.mocks.mock_task import MockTask

from promptolution.exemplar_selectors.random_search_selector import RandomSearchSelector
from promptolution.exemplar_selectors.random_selector import RandomSelector
from promptolution.helpers import (
    get_exemplar_selector,
    get_llm,
    get_optimizer,
    get_predictor,
    get_task,
    run_evaluation,
    run_experiment,
    run_optimization,
)
from promptolution.optimizers.capo import CAPO
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.opro import OPRO
from promptolution.predictors.first_occurrence_predictor import FirstOccurrencePredictor
from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.tasks.base_task import EvalResult
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.tasks.judge_tasks import JudgeTask
from promptolution.tasks.reward_tasks import RewardTask
from promptolution.utils import ExperimentConfig
from promptolution.utils.prompt import Prompt


@pytest.fixture
def sample_df():
    """Fixture providing a sample DataFrame for testing."""
    data = {
        "x": [
            "This product is amazing!",
            "I'm disappointed with this purchase.",
            "The quality is average, nothing special.",
            "Worst product ever, avoid at all costs!",
            "Decent product, does what it's supposed to.",
        ],
        "y": ["positive", "negative", "neutral", "negative", "positive"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def experiment_config():
    """Fixture providing a configuration for experiments."""
    return ExperimentConfig(
        optimizer_name="evoprompt_ga",
        task_name="classification",
        task_description="Classify sentiment.",
        llm_name="mock",
        predictor_name="first_occurrence",
        classes=["positive", "neutral", "negative"],
        n_steps=2,
        posthoc_exemplar_selection=False,
    )


@pytest.fixture
def experiment_config_with_exemplars():
    """Fixture providing a configuration with exemplars enabled."""
    return ExperimentConfig(
        optimizer_name="evoprompt_ga",
        task_name="classification",
        task_description="Classify sentiment.",
        llm_name="mock",
        predictor_name="first_occurrence",
        classes=["positive", "neutral", "negative"],
        n_steps=2,
        posthoc_exemplar_selection=True,
        exemplar_selector="random",
        n_exemplars=2,
    )


@patch("promptolution.helpers.get_llm")
@patch("promptolution.helpers.get_predictor")
@patch("promptolution.helpers.get_task")
@patch("promptolution.helpers.get_optimizer")
def test_run_optimization(
    mock_get_optimizer, mock_get_task, mock_get_predictor, mock_get_llm, sample_df, experiment_config
):
    """Test the run_optimization function."""
    # Set up mocks
    mock_llm = MockLLM()
    mock_predictor = MockPredictor(classes=experiment_config.classes)
    mock_predictor.extraction_description = "Extraction description."
    mock_task = MockTask()
    mock_optimizer = MagicMock()

    # Configure mocks to return our test objects
    mock_get_llm.return_value = mock_llm
    mock_get_predictor.return_value = mock_predictor
    mock_get_task.return_value = mock_task
    mock_get_optimizer.return_value = mock_optimizer

    # Set up optimizer to return some prompts
    optimized_prompts = [
        "Classify this as positive or negative:",
        "Determine the sentiment (positive/negative/neutral):",
        "Is this text positive, negative, or neutral?",
    ]
    mock_optimizer.optimize.return_value = optimized_prompts

    # Run the function
    result = run_optimization(sample_df, experiment_config)

    # Verify the results
    assert result == optimized_prompts

    # Verify mocks were called
    mock_get_llm.assert_called_once_with(config=experiment_config)
    mock_get_predictor.assert_called_once_with(mock_llm, config=experiment_config)
    mock_get_task.assert_called_once_with(sample_df, experiment_config, judge_llm=mock_llm)
    mock_get_optimizer.assert_called_once_with(
        predictor=mock_predictor, meta_llm=mock_llm, task=mock_task, config=experiment_config
    )
    mock_optimizer.optimize.assert_called_once_with(n_steps=experiment_config.n_steps)


@patch("promptolution.helpers.get_llm")
@patch("promptolution.helpers.get_predictor")
@patch("promptolution.helpers.get_task")
@patch("promptolution.helpers.get_optimizer")
@patch("promptolution.helpers.get_exemplar_selector")
def test_run_optimization_with_exemplars(
    mock_get_exemplar_selector,
    mock_get_optimizer,
    mock_get_task,
    mock_get_predictor,
    mock_get_llm,
    sample_df,
    experiment_config_with_exemplars,
):
    """Test run_optimization with exemplar selection enabled."""
    # Set up mocks
    mock_llm = MockLLM()
    mock_predictor = MockPredictor(classes=experiment_config_with_exemplars.classes)
    mock_predictor.extraction_description = "Extraction description."
    mock_task = MockTask()
    mock_optimizer = MagicMock()
    mock_selector = MagicMock()

    # Configure mocks to return our test objects
    mock_get_llm.return_value = mock_llm
    mock_get_predictor.return_value = mock_predictor
    mock_get_task.return_value = mock_task
    mock_get_optimizer.return_value = mock_optimizer
    mock_get_exemplar_selector.return_value = mock_selector

    # Set up optimizer to return some prompts
    optimized_prompts = [
        "Classify this as positive or negative:",
        "Determine the sentiment (positive/negative/neutral):",
    ]
    mock_optimizer.optimize.return_value = optimized_prompts

    # Set up exemplar selector
    exemplar_prompts = [
        "Example 1: 'Great product!' - positive\nExample 2: 'Terrible!' - negative\nClassify this as positive or negative:",
        "Example 1: 'Great product!' - positive\nExample 2: 'Terrible!' - negative\nDetermine the sentiment (positive/negative/neutral):",
    ]
    mock_selector.select_exemplars.side_effect = exemplar_prompts

    # Run the function
    result = run_optimization(sample_df, experiment_config_with_exemplars)

    # Verify the results
    assert result == exemplar_prompts

    # Verify mocks were called
    mock_get_llm.assert_called_once_with(config=experiment_config_with_exemplars)
    mock_get_predictor.assert_called_once_with(mock_llm, config=experiment_config_with_exemplars)
    mock_get_task.assert_called_once_with(sample_df, experiment_config_with_exemplars, judge_llm=mock_llm)
    mock_get_optimizer.assert_called_once_with(
        predictor=mock_predictor, meta_llm=mock_llm, task=mock_task, config=experiment_config_with_exemplars
    )
    mock_optimizer.optimize.assert_called_once_with(n_steps=experiment_config_with_exemplars.n_steps)

    # Verify exemplar selector was called
    mock_get_exemplar_selector.assert_called_once_with(
        experiment_config_with_exemplars.exemplar_selector, mock_task, mock_predictor
    )
    assert mock_selector.select_exemplars.call_count == 2


@patch("promptolution.helpers.get_llm")
@patch("promptolution.helpers.get_predictor")
@patch("promptolution.helpers.get_task")
def test_run_evaluation(mock_get_task, mock_get_predictor, mock_get_llm, sample_df, experiment_config):
    """Test the run_evaluation function."""
    # Set up mocks
    mock_llm = MockLLM()
    mock_predictor = MockPredictor()

    # Use MagicMock instead of MockTask
    mock_task = MagicMock()
    mock_task.classes = ["positive", "neutral", "negative"]

    # Configure mocks to return our test objects
    mock_get_llm.return_value = mock_llm
    mock_get_predictor.return_value = mock_predictor
    mock_get_task.return_value = mock_task

    # Set up task to return scores
    prompts = [
        "Classify this as positive or negative:",
        "Determine the sentiment (positive/negative/neutral):",
        "Is this text positive, negative, or neutral?",
    ]

    prompts = [Prompt(p) for p in prompts]

    # Now this will work because mock_task is a MagicMock
    mock_task.evaluate.return_value = EvalResult(
        scores=np.array([[0.9], [0.8], [0.7]], dtype=float),
        agg_scores=np.array([0.9, 0.8, 0.7], dtype=float),
        sequences=np.array([["s1"], ["s2"], ["s3"]], dtype=object),
        input_tokens=np.array([[10.0], [10.0], [10.0]], dtype=float),
        output_tokens=np.array([[5.0], [5.0], [5.0]], dtype=float),
        agg_input_tokens=np.array([10.0, 10.0, 10.0], dtype=float),
        agg_output_tokens=np.array([5.0, 5.0, 5.0], dtype=float),
    )

    # Run the function
    result = run_evaluation(sample_df, experiment_config, prompts)

    # Verify the results
    assert isinstance(result, pd.DataFrame)
    assert "prompt" in result.columns
    assert "score" in result.columns
    assert len(result) == 3

    # Verify the DataFrame is sorted by score (descending)
    assert result.iloc[0]["score"] == 0.9
    assert result.iloc[1]["score"] == 0.8
    assert result.iloc[2]["score"] == 0.7

    # Verify mocks were called
    mock_get_llm.assert_called_once_with(config=experiment_config)
    mock_get_predictor.assert_called_once_with(mock_llm, config=experiment_config)
    mock_get_task.assert_called_once_with(sample_df, experiment_config, judge_llm=mock_llm)
    mock_task.evaluate.assert_called_once_with(prompts, mock_predictor, eval_strategy="full")


@patch("promptolution.helpers.run_optimization")
@patch("promptolution.helpers.run_evaluation")
def test_run_experiment(mock_run_evaluation, mock_run_optimization, sample_df, experiment_config):
    """Test the run_experiment function."""
    # Set up mocks
    optimized_prompts_strs = [
        "Classify this as positive or negative:",
        "Determine the sentiment (positive/negative/neutral):",
    ]
    optimized_prompts = [Prompt(p) for p in optimized_prompts_strs]
    mock_run_optimization.return_value = optimized_prompts

    # Create a sample results DataFrame
    eval_results = pd.DataFrame({"prompt": optimized_prompts_strs, "score": [0.8, 0.7]})
    mock_run_evaluation.return_value = eval_results

    # Run the function
    result = run_experiment(sample_df, experiment_config)

    # Verify results
    assert result is eval_results

    # Verify the train-test split
    mock_run_optimization_args = mock_run_optimization.call_args[0]
    mock_run_evaluation_args = mock_run_evaluation.call_args[0]

    train_df = mock_run_optimization_args[0]
    test_df = mock_run_evaluation_args[0]

    # Check that we have a 80-20 split
    assert len(train_df) == 4  # 80% of 5 rows
    assert len(test_df) == 1  # 20% of 5 rows

    # Check that no data is lost
    assert len(train_df) + len(test_df) == len(sample_df)

    # Verify the prompts were passed to evaluation
    assert mock_run_evaluation.call_args[0][2] == optimized_prompts_strs


def test_helpers_integration(sample_df, experiment_config):
    """Integration test for helper functions - this tests the full experiment flow."""
    # This test will use the actual functions but with mocked components
    with patch("promptolution.helpers.get_llm") as mock_get_llm, patch(
        "promptolution.helpers.get_predictor"
    ) as mock_get_predictor, patch("promptolution.helpers.get_task") as mock_get_task, patch(
        "promptolution.helpers.get_optimizer"
    ) as mock_get_optimizer:
        # Set up mocks
        mock_llm = MockLLM()
        mock_predictor = MockPredictor(classes=experiment_config.classes)
        mock_predictor.extraction_description = "Extract the sentiment."

        # Use a MagicMock instead of MockTask
        mock_task = MagicMock()
        mock_task.classes = ["positive", "neutral", "negative"]
        mock_task.evaluate = MagicMock(
            return_value=EvalResult(
                scores=np.array([[0.9], [0.8]], dtype=float),
                agg_scores=np.array([0.9, 0.8], dtype=float),
                sequences=np.array([["s1"], ["s2"]], dtype=object),
                input_tokens=np.array([[10.0], [10.0]], dtype=float),
                output_tokens=np.array([[5.0], [5.0]], dtype=float),
                agg_input_tokens=np.array([10.0, 10.0], dtype=float),
                agg_output_tokens=np.array([5.0, 5.0], dtype=float),
            )
        )

        mock_optimizer = MagicMock()

        # Configure mocks
        mock_get_llm.return_value = mock_llm
        mock_get_predictor.return_value = mock_predictor
        mock_get_task.return_value = mock_task
        mock_get_optimizer.return_value = mock_optimizer

        # Set up optimizer to return prompts
        optimized_prompts_str = ["Classify sentiment:", "Determine if positive/negative:"]
        optimized_prompts = [Prompt(p) for p in optimized_prompts_str]
        mock_optimizer.optimize.return_value = optimized_prompts

        # Run the experiment
        result = run_experiment(sample_df, experiment_config)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        print([p in result["prompt"].values for p in optimized_prompts_str])
        assert all(p in result["prompt"].values for p in optimized_prompts_str)

        # Verify optimization was called
        mock_optimizer.optimize.assert_called_once()

        # Verify evaluation was called
        mock_task.evaluate.assert_called()


def test_get_llm_variants(monkeypatch):
    def factory(model_name=None, config=None, **kwargs):
        created["name"] = model_name or kwargs.get("model_id")
        created["config"] = config
        return MockLLM()

    created = {}

    monkeypatch.setattr("promptolution.helpers.LocalLLM", factory)
    monkeypatch.setattr("promptolution.helpers.VLLM", factory)
    monkeypatch.setattr("promptolution.helpers.APILLM", factory)

    cfg = ExperimentConfig()
    cfg.model_id = "local-foo"
    res = get_llm(config=cfg)
    assert isinstance(res, MockLLM)
    assert created["name"] == "foo"

    cfg.model_id = "vllm-bar"
    res = get_llm(config=cfg)
    assert created["name"] == "bar"

    cfg.model_id = "api-model"
    res = get_llm(config=cfg)
    assert created["name"] == "api-model"

    with pytest.raises(ValueError):
        get_llm()


def test_get_task_variants(sample_df):
    cfg = ExperimentConfig()
    cfg.task_type = "reward"
    task = get_task(sample_df, cfg, reward_function=lambda _: 1.0)

    assert isinstance(task, RewardTask)

    cfg.task_type = "judge"
    judge_task = get_task(sample_df, cfg, judge_llm=MockLLM())

    assert isinstance(judge_task, JudgeTask)

    cfg.task_type = "classification"
    cls_task = get_task(sample_df, cfg)

    assert isinstance(cls_task, ClassificationTask)


def test_get_optimizer_variants():
    pred = MockPredictor(llm=MockLLM())
    task = MockTask()
    cfg = ExperimentConfig()

    opt = get_optimizer(pred, MockLLM(), task, optimizer="capo", config=cfg)

    assert isinstance(opt, CAPO)

    opt3 = get_optimizer(pred, MockLLM(), task, optimizer="evopromptde", config=cfg)

    assert isinstance(opt3, EvoPromptDE)

    opt4 = get_optimizer(pred, MockLLM(), task, optimizer="evopromptga", config=cfg)

    assert isinstance(opt4, EvoPromptGA)

    opt5 = get_optimizer(pred, MockLLM(), task, optimizer="opro", config=cfg)

    assert isinstance(opt5, OPRO)

    with pytest.raises(ValueError):
        get_optimizer(pred, MockLLM(), task, optimizer="unknown", config=cfg)


def test_get_exemplar_selector_variants():
    task = MockTask()
    pred = MockPredictor()

    sel = get_exemplar_selector("random", task, pred)
    assert isinstance(sel, RandomSelector)

    sel2 = get_exemplar_selector("random_search", task, pred)
    assert isinstance(sel2, RandomSearchSelector)

    with pytest.raises(ValueError):
        get_exemplar_selector("nope", task, pred)


def test_get_predictor_variants():
    llm = MockLLM()

    p1 = get_predictor(llm, type="first_occurrence", classes=["a", "b"])
    assert isinstance(p1, FirstOccurrencePredictor)

    p2 = get_predictor(llm, type="marker")
    assert isinstance(p2, MarkerBasedPredictor)

    with pytest.raises(ValueError):
        get_predictor(llm, type="bad")
