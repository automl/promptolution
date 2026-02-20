"""Fixtures for testing."""

import pandas as pd
import pytest
from mocks.mock_llm import MockLLM
from mocks.mock_predictor import MockPredictor
from mocks.mock_task import MockTask

from promptolution.tasks import ClassificationTask
from promptolution.tasks.judge_tasks import JudgeTask
from promptolution.tasks.reward_tasks import RewardTask
from promptolution.utils import ExperimentConfig


@pytest.fixture
def initial_prompts():
    """Fixture providing a set of initial prompts for testing."""
    return [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
        "Analyze the sentiment of this text and categorize as positive or negative.",
    ]


@pytest.fixture
def experiment_config():
    """Fixture providing a basic experiment configuration."""
    return ExperimentConfig(optimizer_name="test_optimizer", n_steps=3, population_size=3, random_seed=42)


@pytest.fixture
def mock_task():
    """Fixture providing a MockTask with predetermined scoring behavior."""

    def score_function(preds):
        # Prefer longer prompts for testing purposes
        return [len(pred) for pred in preds]

    return MockTask(predetermined_scores=score_function)


@pytest.fixture
def mock_meta_llm():
    """Fixture providing a MockLLM configured for meta-prompt responses."""
    # Create a list of responses that already include prompt tags
    responses = [
        "<prompt>Meta-generated prompt for input 0</prompt>",
        "<prompt>Meta-generated prompt for input 1</prompt>",
        "<prompt>Combined prompt</prompt>",
        "<prompt>Improved version of prompt</prompt>",
        "<prompt>Default meta-response</prompt>",
    ]

    # Simply initialize the MockLLM with the predetermined responses
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_downstream_llm():
    """Fixture providing a MockLLM configured for downstream task responses."""
    # Create a list of responses that already include prompt tags
    responses = [
        "This review is not negative, so my answer is <final_answer>positive</final_answer>",
        "This review is not positive, so my answer is <final_answer>negative</final_answer>",
        "This review is neither positive nor negative, so my answer is <final_answer>neutral</final_answer>",
        "Pfff hard to say, <final_answer>I dont know</final_answer>",
    ]

    # Simply initialize the MockLLM with the predetermined responses
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_predictor(mock_downstream_llm):
    """Fixture providing a MockPredictor for testing."""
    mock_predictor = MockPredictor(llm=mock_downstream_llm)
    return mock_predictor


@pytest.fixture
def mock_df():
    """Fixture providing a DataFrame with few-shot examples for CAPO."""
    return pd.DataFrame(
        {
            "x": ["This is excellent!", "I hate this product.", "Average experience."],
            "y": ["positive", "negative", "neutral"],
        }
    )


@pytest.fixture
def mock_classification_task_with_subsampling(mock_df):
    """Fixture providing a ClassificationTask instance with subsampling."""
    return ClassificationTask(
        df=mock_df,
        task_description="Sentiment classification task",
        x_column="x",
        y_column="y",
        eval_strategy="subsample",
        n_subsamples=2,
    )


@pytest.fixture
def simple_reward_function():
    """Define a simple reward function for testing RewardTask."""

    def reward_func(prediction: str) -> float:
        if "great" in prediction.lower() or "perfect" in prediction.lower():
            return 1.0
        elif "ok" in prediction.lower():
            return 0.5
        else:
            return 0.0

    return reward_func


@pytest.fixture
def mock_reward_task(mock_df, simple_reward_function):
    """Fixture providing a RewardTask instance."""
    return RewardTask(
        df=mock_df,
        reward_function=simple_reward_function,
        x_column="x",
        task_description="Evaluate text quality",
        n_subsamples=2,
        eval_strategy="full",  # Using "full" for initial clarity, can be changed in specific tests
        seed=42,
    )


@pytest.fixture
def mock_reward_task_no_x_column(simple_reward_function):
    """Fixture providing a RewardTask instance without a meaningful x_column."""
    # Create a DataFrame where 'x' is just a placeholder, not used for prompt construction directly
    df_no_x_data = {
        "id_col": list(range(5)),
        "dummy_input": ["", "", "", "", ""],  # Or just 0, 1, 2, 3, 4
        "some_attribute": ["A", "B", "C", "D", "E"],
    }
    df_no_x = pd.DataFrame(df_no_x_data)
    return RewardTask(
        df=df_no_x,
        reward_function=simple_reward_function,
        x_column="dummy_input",  # The x_column is still technically provided but contains empty strings or Nones
        task_description="Generate and evaluate jokes without explicit input text.",
        n_subsamples=3,
        eval_strategy="subsample",
        seed=42,
    )


@pytest.fixture
def mock_judge_llm():
    """Fixture providing a MockLLM configured for judge responses."""
    # Responses containing the final_score tag
    responses = [
        "<final_score>5.0</final_score>",  # Perfect match
        "<final_score>-5.0</final_score>",  # Completely incorrect
        "<final_score>0.0</final_score>",  # Partially correct
        "<final_score>1.0</final_score>",  # Default/Other
        "<final_score>2.0</final_score>",  # Another specific score
        "This response does not contain a score tag.",  # For parsing error test
    ]
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_judge_task_with_y(mock_df, mock_judge_llm):
    """Fixture providing a JudgeTask instance with y_column."""
    return JudgeTask(
        df=mock_df,
        x_column="x",
        y_column="y",
        judge_llm=mock_judge_llm,
        task_description="Evaluate sentiment prediction quality.",
        n_subsamples=2,
        eval_strategy="full",
        seed=42,
    )


@pytest.fixture
def mock_judge_task_no_y(mock_df, mock_judge_llm):
    """Fixture providing a JudgeTask instance without y_column."""
    # Use mock_df, but ensure y_column is explicitly None for this task instance
    return JudgeTask(
        df=mock_df,
        x_column="x",
        judge_llm=mock_judge_llm,
        task_description="Evaluate joke quality (no ground truth).",
        n_subsamples=2,
        eval_strategy="subsample",  # Test with subsampling here
        seed=42,
    )
