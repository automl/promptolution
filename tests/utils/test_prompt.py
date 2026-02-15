import numpy as np

from promptolution.utils.prompt import Prompt, sort_prompts_by_scores


def test_prompt_initialization():
    """Test that Prompt initializes correctly."""
    instruction = "Classify the sentiment of the text."
    few_shots = ["Example 1: Positive", "Example 2: Negative"]
    prompt = Prompt(instruction, few_shots)

    # Verify attributes
    assert prompt.instruction == instruction
    assert prompt.few_shots == few_shots


def test_prompt_construct_prompt():
    """Test the construct_prompt method of Prompt."""
    instruction = "Classify the sentiment of the text."
    few_shots = ["Example 1: Positive", "Example 2: Negative"]
    prompt = Prompt(instruction, few_shots)

    # Get the constructed prompt
    constructed = prompt.construct_prompt()

    # Verify the prompt contains the instruction
    assert instruction in constructed


def test_sort_prompts_by_scores():
    """Test the sort_prompts_by_scores function."""
    prompt1 = Prompt("Instruction 1", ["Example A"])
    prompt2 = Prompt("Instruction 2", ["Example B"])
    prompt3 = Prompt("Instruction 3", ["Example C"])

    prompts = [prompt1, prompt2, prompt3]
    scores = [0.75, 0.90, 0.60]

    sorted_prompts, sorted_scores = sort_prompts_by_scores(prompts, scores)

    # Verify sorting
    assert sorted_prompts == [prompt2, prompt1, prompt3]
    assert sorted_scores == [0.90, 0.75, 0.60]


def test_sort_prompts_by_scores_with_array():
    """Ensure sorting works when scores are numpy arrays (aggregated via mean)."""
    prompts = [Prompt("p1"), Prompt("p2"), Prompt("p3")]
    scores = np.array([[0.5, 0.7], [0.8, 0.9], [0.4, 0.6]])

    sorted_prompts, sorted_scores = sort_prompts_by_scores(prompts, scores)

    assert sorted_prompts == [prompts[1], prompts[0], prompts[2]]
    np.testing.assert_allclose(sorted_scores, [0.85, 0.6, 0.5])
