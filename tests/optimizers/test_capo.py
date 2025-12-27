from unittest.mock import MagicMock, patch

import pandas as pd

from tests.mocks.mock_task import MockTask

from promptolution.optimizers.capo import CAPO
from promptolution.utils.capo_utils import build_few_shot_examples, perform_crossover, perform_mutation
from promptolution.utils.prompt import Prompt
from promptolution.utils.templates import CAPO_CROSSOVER_TEMPLATE, CAPO_FEWSHOT_TEMPLATE, CAPO_MUTATION_TEMPLATE


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
        df_few_shots=mock_df,
        x_column=mock_task.x_column,
        y_column=mock_task.y_column,
        predictor=mock_predictor,
        fewshot_template=CAPO_FEWSHOT_TEMPLATE,
        target_begin_marker=optimizer.target_begin_marker,
        target_end_marker=optimizer.target_end_marker,
        check_fs_accuracy=True,
        create_fs_reasoning=True,
    )

    # Verify results
    assert len(few_shot_examples) == 2
    assert all(isinstance(example, str) for example in few_shot_examples)

    few_shot_examples = build_few_shot_examples(
        instruction="Classify the sentiment of the text.",
        num_examples=0,
        df_few_shots=mock_df,
        x_column=mock_task.x_column,
        y_column=mock_task.y_column,
        predictor=mock_predictor,
        fewshot_template=CAPO_FEWSHOT_TEMPLATE,
        target_begin_marker=optimizer.target_begin_marker,
        target_end_marker=optimizer.target_end_marker,
        check_fs_accuracy=True,
        create_fs_reasoning=True,
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
        optimizer.crossovers_per_iter,
        optimizer.crossover_template,
        optimizer.meta_llm,
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
        mutation_template=optimizer.mutation_template,
        upper_shots=optimizer.upper_shots,
        meta_llm=optimizer.meta_llm,
        few_shot_kwargs=dict(
            df_few_shots=mock_df,
            x_column=mock_task.x_column,
            y_column=mock_task.y_column,
            predictor=mock_predictor,
            fewshot_template=CAPO_FEWSHOT_TEMPLATE,
            target_begin_marker=optimizer.target_begin_marker,
            target_end_marker=optimizer.target_end_marker,
            check_fs_accuracy=True,
            create_fs_reasoning=True,
        ),
    )
    assert len(mutated) == 2


def test_do_racing(mock_meta_llm, mock_predictor, initial_prompts, mock_df):
    mock_task = MockTask(predetermined_scores=[0.89, 0.9] * 3)
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=pd.concat([mock_df] * 5, ignore_index=True),
    )
    optimizer._pre_optimization_loop()
    survivors, scores = optimizer._do_racing(
        [Prompt("good instruction", ["Example 1"]), Prompt("better instruction", ["Example 2"])], 1
    )
    assert len(survivors) == 1
    assert len(scores) == 1

    assert "better instruction" in survivors[0].instruction

    assert mock_task.reset_block_idx.call_count == 2
    assert mock_task.increment_block_idx.call_count == 3


def test_capo_crossover_prompt(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    """Test that when _crossover is called, the mock_meta_llm received a call with the correct meta prompt."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    mother = Prompt("Classify the sentiment of the text.", ["Input: I love this! Output: Positive"])
    father = Prompt("Determine if the review is positive or negative.", ["Input: This is terrible. Output: Negative"])
    perform_crossover([mother, father], optimizer.crossovers_per_iter, optimizer.crossover_template, optimizer.meta_llm)

    full_task_desc = mock_task.task_description + "\n" + optimizer.predictor.extraction_description

    expected_meta_prompt = (
        CAPO_CROSSOVER_TEMPLATE.replace("<mother>", mother.instruction)
        .replace("<father>", father.instruction)
        .replace("<task_desc>", full_task_desc)
    )
    alt_meta_prompt = (
        CAPO_CROSSOVER_TEMPLATE.replace("<mother>", father.instruction)
        .replace("<father>", mother.instruction)
        .replace("<task_desc>", full_task_desc)
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
    full_task_desc = mock_task.task_description + "\n" + optimizer.predictor.extraction_description

    parent = Prompt("Classify the sentiment of the text.", ["Input: I love this! Output: Positive"])
    perform_mutation(
        offsprings=[parent],
        mutation_template=optimizer.mutation_template,
        upper_shots=optimizer.upper_shots,
        meta_llm=optimizer.meta_llm,
        few_shot_kwargs=dict(
            df_few_shots=mock_df,
            x_column=mock_task.x_column,
            y_column=mock_task.y_column,
            predictor=mock_predictor,
            fewshot_template=CAPO_FEWSHOT_TEMPLATE,
            target_begin_marker=optimizer.target_begin_marker,
            target_end_marker=optimizer.target_end_marker,
            check_fs_accuracy=True,
            create_fs_reasoning=True,
        ),
    )

    expected_meta_prompt = CAPO_MUTATION_TEMPLATE.replace("<instruction>", parent.instruction).replace(
        "<task_desc>", full_task_desc
    )

    assert mock_meta_llm.call_history[0]["prompts"][0] == expected_meta_prompt
