import numpy as np

from promptolution.utils.prompt import Prompt


def test_judge_task_initialization(mock_judge_task_with_y, mock_judge_llm):
    """Test that JudgeTask initializes correctly with ground truth."""
    assert mock_judge_task_with_y.task_description == "Evaluate sentiment prediction quality."
    assert mock_judge_task_with_y.x_column == "x"
    assert mock_judge_task_with_y.y_column == "y"
    assert mock_judge_task_with_y.judge_llm == mock_judge_llm
    assert mock_judge_task_with_y.has_y is True
    assert len(mock_judge_task_with_y.xs) == len(mock_judge_task_with_y.df)
    assert len(mock_judge_task_with_y.ys) == len(mock_judge_task_with_y.df)


def test_judge_task_initialization_no_y(mock_judge_task_no_y):
    """Test JudgeTask initialization when no y_column is provided."""
    assert mock_judge_task_no_y.y_column is None
    assert mock_judge_task_no_y.has_y is False
    assert len(mock_judge_task_no_y.xs) == len(mock_judge_task_no_y.df)
    assert all(y == "" for y in mock_judge_task_no_y.ys)  # noqa: E711


def test_judge_task_construct_judge_prompt_with_ground_truth(mock_judge_task_with_y):
    """Test _construct_judge_prompt generates correct prompt when ground truth is available."""
    x_val = "This movie was great!"
    pred_val = "positive"
    y_val = "positive"
    prompt = mock_judge_task_with_y._construct_judge_prompt(x_val, pred_val, y_val)

    assert mock_judge_task_with_y.task_description in prompt
    assert f"Input:\n{x_val}" in prompt
    assert f"Ground Truth:\n{y_val}" in prompt
    assert f"Prediction:\n{pred_val}" in prompt
    assert "Response:" not in prompt
    assert "<final_score>" in prompt


def test_judge_task_construct_judge_prompt_without_ground_truth(mock_judge_task_no_y):
    """Test _construct_judge_prompt generates correct prompt when no ground truth."""
    x_val = "Tell me a joke."
    pred_val = "Why did the scarecrow win an award? Because he was outstanding in his field!"
    prompt = mock_judge_task_no_y._construct_judge_prompt(x_val, pred_val, None)

    assert mock_judge_task_no_y.task_description in prompt
    assert f"Input:\n{x_val}" in prompt
    assert pred_val in prompt
    assert "<final_score>" in prompt


def test_judge_task_evaluate_with_ground_truth(mock_judge_task_with_y, mock_predictor, mock_judge_llm):
    """Test the evaluate method of JudgeTask with ground truth and full evaluation."""
    prompts = ["Rate the sentiment:", "What is the sentiment?", "How would you classify this?"]
    prompts = [Prompt(p) for p in prompts]

    mock_predictor.call_history = []
    mock_judge_llm.call_history = []

    result = mock_judge_task_with_y.evaluate(prompts, mock_predictor)
    scores_per_datapoint = result.scores

    assert scores_per_datapoint.shape[0] == len(prompts)
    expected_scores = [1.0, 0, 0.5]
    np.testing.assert_allclose(scores_per_datapoint[0], expected_scores)

    mock_predictor.call_history = []
    mock_judge_llm.call_history = []

    aggregated_scores = result.agg_scores
    assert aggregated_scores.shape[0] == len(prompts)
    expected_scores = [0.5, 0.4333333, 0.0]
    np.testing.assert_allclose(aggregated_scores, expected_scores)


def test_judge_task_evaluate_no_ground_truth(mock_judge_task_no_y, mock_predictor, mock_judge_llm):
    """Test the evaluate method of JudgeTask without a y_column (no ground truth)."""
    prompts = ["Tell a funny joke:", "Make me laugh:", "What's a good joke?"]
    prompts = [Prompt(p) for p in prompts]

    mock_predictor.call_history = []
    mock_judge_llm.call_history = []

    aggregated_result = mock_judge_task_no_y.evaluate(prompts, mock_predictor)
    aggregated_scores = aggregated_result.agg_scores

    assert aggregated_scores.shape[0] == len(prompts)
    expected_scores = [0.5, 0.55, 0.35]
    np.testing.assert_allclose(aggregated_scores, expected_scores)


def test_judge_task_evaluate_with_return_seq(mock_judge_task_with_y, mock_predictor):
    """Test the evaluate method with return_seq=True for JudgeTask."""
    prompts = ["Evaluate this text:", "What is the sentiment?", "How would you classify this?"]
    prompts = [Prompt(p) for p in prompts]

    seq_result = mock_judge_task_with_y.evaluate(prompts, mock_predictor)

    assert seq_result.scores.shape == (3, len(mock_judge_task_with_y.xs))
    assert seq_result.sequences is not None
    assert seq_result.sequences.shape == (3, len(mock_judge_task_with_y.xs))
    assert seq_result.costs is not None
