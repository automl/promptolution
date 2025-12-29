from promptolution.utils.prompt import Prompt


def test_reward_task_initialization(mock_reward_task, simple_reward_function):
    """Test that RewardTask initializes correctly."""
    assert mock_reward_task.task_description == "Evaluate text quality"
    assert mock_reward_task.reward_function == simple_reward_function
    assert mock_reward_task.x_column == "x"
    assert not mock_reward_task.has_y
    assert len(mock_reward_task.xs) == len(mock_reward_task.df)
    assert all(y == "" for y in mock_reward_task.ys)  # noqa: E711


def test_reward_task_initialization_no_x_column(mock_reward_task_no_x_column, simple_reward_function):
    """Test RewardTask initialization when a dummy x_column is provided (no semantic input)."""
    assert mock_reward_task_no_x_column.x_column == "dummy_input"
    assert not mock_reward_task_no_x_column.has_y
    assert len(mock_reward_task_no_x_column.xs) == len(mock_reward_task_no_x_column.df)
    assert all(x == "" for x in mock_reward_task_no_x_column.xs)
    assert all([y == "" for y in mock_reward_task_no_x_column.ys])  # noqa: E711


def test_reward_task_evaluate_with_return_seq(mock_reward_task, mock_predictor):
    """Test the evaluate method with return_seq=True for RewardTask."""
    prompts = [Prompt("Generate a short text:")]

    result = mock_reward_task.evaluate(prompts, mock_predictor)

    assert result.scores.shape[0] == 1
    assert result.sequences is not None
    assert result.sequences.shape[0] == 1
    assert result.costs is not None
