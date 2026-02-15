import numpy as np

from promptolution.tasks.base_task import BaseTask
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.prompt_creation import create_prompt_variation, create_prompts_from_samples


def test_create_prompt_variation_single_prompt(mock_meta_llm):
    """Test create_prompt_variation with a single string prompt and default meta-prompt."""
    original_prompt = "Analyze the sentiment of the following text."

    mock_meta_llm.call_history = []

    varied_prompts = create_prompt_variation(original_prompt, mock_meta_llm)

    assert isinstance(varied_prompts, list)
    assert len(varied_prompts) == 1
    assert varied_prompts[0] == "Meta-generated prompt for input 0"

    assert len(mock_meta_llm.call_history) == 1


def test_create_prompt_variation_list_of_prompts(mock_meta_llm):
    """Test create_prompt_variation with a list of prompts and custom meta-prompt."""
    original_prompts = ["Prompt A.", "Prompt B."]
    custom_meta_prompt = "Vary the following: <prev_prompt>"

    mock_meta_llm.call_history = []

    varied_prompts = create_prompt_variation(original_prompts, mock_meta_llm, meta_prompt=custom_meta_prompt)

    assert isinstance(varied_prompts, list)
    assert len(varied_prompts) == 2
    assert varied_prompts[0] == "Meta-generated prompt for input 0"
    assert varied_prompts[1] == "Meta-generated prompt for input 1"

    assert len(mock_meta_llm.call_history) == 1


def test_create_prompts_from_samples_default_meta_prompt(mock_df, mock_meta_llm):
    """Test create_prompts_from_samples with default meta_prompt (no task_description)."""
    task = ClassificationTask(df=mock_df, x_column="x", y_column="y", task_description="Dummy task")
    n_samples = 2
    n_prompts = 1

    mock_meta_llm.call_history = []

    generated_prompts = create_prompts_from_samples(task, mock_meta_llm, n_samples=n_samples, n_prompts=n_prompts)

    assert isinstance(generated_prompts, list)
    assert len(generated_prompts) == n_prompts
    assert generated_prompts[0] == "Meta-generated prompt for input 0"

    assert len(mock_meta_llm.call_history) == n_prompts


def test_create_prompts_from_samples_with_task_description_only(mock_df, mock_meta_llm):
    """Test create_prompts_from_samples with task_description and no meta_prompt."""
    task = ClassificationTask(df=mock_df, x_column="x", y_column="y")
    test_task_description = "Classify customer reviews into positive, negative, or neutral."
    n_samples = 2
    n_prompts = 1

    mock_meta_llm.call_history = []

    generated_prompts = create_prompts_from_samples(
        task, mock_meta_llm, n_samples=n_samples, task_description=test_task_description, n_prompts=n_prompts
    )

    assert len(generated_prompts) == n_prompts
    assert generated_prompts[0] == "Meta-generated prompt for input 0"


def test_create_prompts_from_samples_with_custom_meta_prompt_only(mock_df, mock_meta_llm):
    """Test create_prompts_from_samples with custom meta_prompt and no task_description."""
    task = ClassificationTask(df=mock_df, x_column="x", y_column="y")
    custom_meta_prompt = "Generate a prompt based on these examples: <input_output_pairs>"
    n_samples = 2
    n_prompts = 1

    mock_meta_llm.call_history = []

    generated_prompts = create_prompts_from_samples(
        task, mock_meta_llm, meta_prompt=custom_meta_prompt, n_samples=n_samples, n_prompts=n_prompts
    )

    assert len(generated_prompts) == n_prompts
    assert generated_prompts[0] == "Meta-generated prompt for input 0"


def test_create_prompts_from_samples_with_both_meta_prompt_and_task_description(mock_df, mock_meta_llm):
    """Test create_prompts_from_samples with both custom meta_prompt and task_description."""
    task = ClassificationTask(df=mock_df, x_column="x", y_column="y")
    custom_meta_prompt = "For <task_desc>, create a prompt using: <input_output_pairs>"
    test_task_description = "Identify categories."
    n_samples = 2
    n_prompts = 1

    mock_meta_llm.call_history = []

    generated_prompts = create_prompts_from_samples(
        task,
        mock_meta_llm,
        meta_prompt=custom_meta_prompt,
        n_samples=n_samples,
        task_description=test_task_description,
        n_prompts=n_prompts,
    )

    assert len(generated_prompts) == n_prompts
    assert generated_prompts[0] == "Meta-generated prompt for input 0"


def test_create_prompts_from_samples_random_sampling(mock_df, mock_meta_llm):
    """Test create_prompts_from_samples with random sampling (not ClassificationTask or get_uniform_labels=False)."""

    class DummyTask(BaseTask):
        def _evaluate(self, x, y, pred):
            return [1.0] * len(x)

    task = DummyTask(df=mock_df, x_column="x", y_column="y", task_description="Dummy task for random sampling")
    n_samples = 2
    n_prompts = 1

    mock_meta_llm.call_history = []

    generated_prompts = create_prompts_from_samples(
        task, mock_meta_llm, n_samples=n_samples, get_uniform_labels=False, n_prompts=n_prompts
    )

    assert len(generated_prompts) == n_prompts


def test_create_prompts_from_samples_multiple_prompts(mock_df, mock_meta_llm):
    """Test create_prompts_from_samples generates multiple prompts."""
    task = ClassificationTask(df=mock_df, x_column="x", y_column="y")
    n_samples = 2
    n_prompts = 3

    mock_meta_llm.call_history = []

    generated_prompts = create_prompts_from_samples(task, mock_meta_llm, n_samples=n_samples, n_prompts=n_prompts)

    assert isinstance(generated_prompts, list)
    assert len(generated_prompts) == n_prompts

    assert len(mock_meta_llm.call_history) == 1


def test_create_prompts_from_samples_uniform_labels(mock_df, mock_meta_llm):
    """Ensure uniform-label sampling includes every class in the meta-prompt examples."""
    task = ClassificationTask(df=mock_df, x_column="x", y_column="y")
    task.xs = np.asarray(task.xs)
    task.ys = np.asarray(task.ys)

    mock_meta_llm.reset()

    prompts = create_prompts_from_samples(task, mock_meta_llm, n_samples=2, n_prompts=1, get_uniform_labels=True)

    assert len(prompts) == 1
    # The constructed meta-prompt should include at least one example per label
    sent_prompt = mock_meta_llm.call_history[0]["prompts"][0]
    for label in ["positive", "negative", "neutral"]:
        assert f"Output: {label}" in sent_prompt
