from unittest.mock import MagicMock, patch

import pandas as pd

from promptolution.optimizers.capoeira import Capoeira
from promptolution.utils.capo_utils import build_few_shot_examples, perform_crossover, perform_mutation
from promptolution.utils.prompt import Prompt
from promptolution.utils.templates import CAPO_CROSSOVER_TEMPLATE, CAPO_FEWSHOT_TEMPLATE, CAPO_MUTATION_TEMPLATE


def test_capoeira_initialization(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    optimizer = Capoeira(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
        population_size=None,
    )

    assert optimizer.crossovers_per_iter == 4
    assert optimizer.population_size == len(initial_prompts)
    assert isinstance(optimizer.df_few_shots, pd.DataFrame)


def test_capoeira_initialize_population(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    optimizer = Capoeira(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
    )

    with patch("random.randint", return_value=1):
        optimizer._pre_optimization_loop()
        population = optimizer.prompts

    assert len(population) == len(initial_prompts)
    assert all(isinstance(p, Prompt) for p in population)


def test_capoeira_selection_prefers_better_score(mock_meta_llm, mock_predictor, mock_task, mock_df):
    optimizer = Capoeira(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=["short", "longer prompt"],
        df_few_shots=mock_df,
        population_size=1,
    )
    optimizer.token_counter = lambda _: 1
    candidates = [Prompt("short"), Prompt("longer prompt")]
    optimizer.task.evaluate = MagicMock(return_value=[0.4, 0.9])

    objectives = optimizer._evaluate_candidates(candidates)
    selected, _ = optimizer._select_population(candidates, objectives)

    assert len(selected) == 1
    assert selected[0].instruction == "longer prompt"


def test_capoeira_meta_prompts(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    optimizer = Capoeira(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
        crossovers_per_iter=2,
    )

    mother = Prompt("Instruction 1", ["Example 1"])
    father = Prompt("Instruction 2", ["Example 2"])
    perform_crossover([mother, father], optimizer.crossovers_per_iter, optimizer.crossover_template, optimizer.meta_llm)

    full_task_desc = mock_task.task_description + "\n" + optimizer.predictor.extraction_description
    expected_crossover = (
        CAPO_CROSSOVER_TEMPLATE.replace("<mother>", mother.instruction)
        .replace("<father>", father.instruction)
        .replace("<task_desc>", full_task_desc)
    )
    assert expected_crossover in mock_meta_llm.call_history[0]["prompts"]

    mock_meta_llm.reset()
    parent = Prompt("Instruction 3", ["Example 3"])
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
    expected_mutation = CAPO_MUTATION_TEMPLATE.replace("<instruction>", parent.instruction).replace(
        "<task_desc>", full_task_desc
    )
    assert expected_mutation in mock_meta_llm.call_history[0]["prompts"]
