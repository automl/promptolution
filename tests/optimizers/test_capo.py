from unittest.mock import MagicMock, patch

import pandas as pd

from promptolution.optimizers.capo import CAPO
from promptolution.utils import CAPO_CROSSOVER_TEMPLATE
from promptolution.utils.capo_utils import build_few_shot_examples, perform_crossover, perform_mutation
from promptolution.utils.prompt import Prompt


def test_capo_initialization(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test that CAPO initializes correctly."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
        crossovers_per_iter=3,
        upper_shots=4,
    )

    # Verify essential properties
    assert optimizer.crossovers_per_iter == 3
    assert optimizer.upper_shots == 4
    assert isinstance(optimizer.df_few_shots, pd.DataFrame)


def test_capo_initialize_population(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test initializing the population using pre-optimization loop."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    with patch("random.randint", return_value=2):
        optimizer._pre_optimization_loop()
        population = optimizer.prompts

    # Verify population was created
    assert len(population) == len(initial_prompts)
    assert all(isinstance(p, Prompt) for p in population)


def test_capo_step(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test the _step method."""
    # Use a smaller population size for the test
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    # Create mock prompt objects
    mock_prompts = [Prompt("Instruction 1", ["Example 1"]), Prompt("Instruction 2", ["Example 2"])]
    optimizer.prompt_objects = mock_prompts

    # Mock the internal methods to avoid complexity
    mock_offspring = [Prompt("Offspring", ["Example"])]
    mock_mutated = [Prompt("Mutated", ["Example"])]
    with patch("promptolution.optimizers.capo.perform_crossover", return_value=mock_offspring), patch(
        "promptolution.optimizers.capo.perform_mutation", return_value=mock_mutated
    ):
        mock_survivors = [Prompt("Survivor 1", ["Example"]), Prompt("Survivor 2", ["Example"])]
        mock_scores = [0.9, 0.8]
        optimizer._do_racing = lambda x, k: (mock_survivors, mock_scores)

        # Call _step
        result = optimizer._step()

    # Verify results
    assert len(result) == 2  # Should match population_size
    assert all(isinstance(p, Prompt) for p in result)


def test_capo_optimize(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test the optimize method."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    # Mock the internal methods to avoid complexity
    optimizer._pre_optimization_loop = MagicMock()

    def mock_step():
        optimizer.prompts = ["Optimized prompt 1", "Optimized prompt 2"]
        return optimizer.prompts

    optimizer._step = mock_step

    # Call optimize
    optimized_prompts = optimizer.optimize(2)

    # Verify results
    assert len(optimized_prompts) == 2
    assert all(isinstance(p, str) for p in optimized_prompts)

    # Verify method calls
    optimizer._pre_optimization_loop.assert_called_once()


def test_create_few_shots(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test the few-shot example builder."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    few_shot_examples = build_few_shot_examples(
        instruction="Classify the sentiment of the text.",
        num_examples=2,
        optimizer=optimizer,
    )

    # Verify results
    assert len(few_shot_examples) == 2
    assert all(isinstance(example, str) for example in few_shot_examples)

    few_shot_examples = build_few_shot_examples(
        instruction="Classify the sentiment of the text.",
        num_examples=0,
        optimizer=optimizer,
    )

    assert len(few_shot_examples) == 0


def test_crossover(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
        crossovers_per_iter=5,
    )

    offsprings = perform_crossover(
        [Prompt("Instruction 1", ["Example 1"]), Prompt("Instruction 2", ["Example 2"])],
        optimizer=optimizer,
    )
    assert len(offsprings) == 5


def test_mutate(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    mutated = perform_mutation(
        offsprings=[Prompt("Instruction 1", ["Example 1"]), Prompt("Instruction 2", ["Example 2"])],
        optimizer=optimizer,
    )
    assert len(mutated) == 2


def test_capo_crossover_prompt(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test that when _crossover is called, the mock_meta_llm received a call with the correct meta prompt."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
        crossovers_per_iter=1,  # Only perform one crossover so we can test the exact prompt
    )

    import random

    random.seed(42)
    mother = Prompt("Classify the sentiment of the text.", ["Input: I love this! Output: Positive"])
    father = Prompt("Determine if the review is positive or negative.", ["Input: This is terrible. Output: Negative"])
    perform_crossover([mother, father], optimizer=optimizer)

    expected_meta_prompt = (
        optimizer.crossover_template.replace("<mother>", mother.instruction)
        .replace("<father>", father.instruction)
        .strip()
    )
    alt_meta_prompt = (
        CAPO_CROSSOVER_TEMPLATE.replace("<mother>", father.instruction)
        .replace("<father>", mother.instruction)
        .replace("<task_desc>", mock_task.task_description)
    )

    assert str(mock_meta_llm.call_history[0]["prompts"][0]) in {expected_meta_prompt, alt_meta_prompt}


def test_capo_mutate_prompt(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test that when _mutate is called, the mock_meta_llm received a call with the correct meta prompt."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    parent = Prompt("Classify the sentiment of the text.", ["Input: I love this! Output: Positive"])
    perform_mutation(
        offsprings=[parent],
        optimizer=optimizer,
    )

    expected_meta_prompt = optimizer.mutation_template.replace("<instruction>", parent.instruction)

    assert mock_meta_llm.call_history[0]["prompts"][0] == expected_meta_prompt
