from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.mocks.mock_task import MockTask

from promptolution.optimizers.capoeira import Capoeira
from promptolution.tasks.base_task import EvalResult
from promptolution.tasks.multi_objective_task import MultiObjectiveEvalResult, MultiObjectiveTask
from promptolution.utils.capo_utils import perform_crossover, perform_mutation
from promptolution.utils.prompt import Prompt
from promptolution.utils.templates import CAPO_CROSSOVER_TEMPLATE, CAPO_MUTATION_TEMPLATE


def test_capoeira_initialization(mock_meta_llm, mock_predictor, initial_prompts, mock_task, mock_df):
    optimizer = Capoeira(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=mock_df,
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


def test_capoeira_objective_vectors_and_sort(mock_meta_llm, mock_predictor, mock_task, mock_df):
    optimizer = Capoeira(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=mock_meta_llm,
        initial_prompts=["short"],
        df_few_shots=mock_df,
    )

    result = EvalResult(
        scores=np.array([[0.4], [0.9]], dtype=float),
        agg_scores=np.array([0.4, 0.9], dtype=float),
        sequences=np.array([["s1"], ["s2"]], dtype=object),
        input_tokens=np.array([[1.0], [1.0]], dtype=float),
        output_tokens=np.array([[0.0], [0.0]], dtype=float),
        agg_input_tokens=np.array([10.0, 8.0], dtype=float),
        agg_output_tokens=np.array([0.0, 0.0], dtype=float),
    )

    vecs = optimizer._get_objective_vectors(result)

    assert vecs.shape == (2, 2)
    assert np.allclose(vecs[:, 0], np.array([0.4, 0.9]))
    assert np.allclose(vecs[:, 1], -np.array([10.0, 8.0]))

    fronts = optimizer._non_dominated_sort(vecs)

    assert fronts[0] == [1]
    assert 0 in fronts[1]


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
    perform_crossover([mother, father], optimizer=optimizer)

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
        optimizer=optimizer,
    )
    expected_mutation = CAPO_MUTATION_TEMPLATE.replace("<instruction>", parent.instruction).replace(
        "<task_desc>", full_task_desc
    )
    assert expected_mutation in mock_meta_llm.call_history[0]["prompts"]


def test_capoeira_crowding_distance_edges():
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]])
    dists = Capoeira._calculate_crowding_distance(vecs)
    assert np.isinf(dists).all()


def test_capoeira_select_survivors_handles_heterogeneous_blocks(mock_meta_llm, mock_predictor):
    def fake_evaluate(*_, **__):
        return EvalResult(
            scores=np.array([[0.5]], dtype=float),
            agg_scores=np.array([0.5], dtype=float),
            sequences=np.array([[""]], dtype=object),
            input_tokens=np.array([[0.0]], dtype=float),
            output_tokens=np.array([[0.0]], dtype=float),
            agg_input_tokens=np.array([0.0], dtype=float),
            agg_output_tokens=np.array([0.0], dtype=float),
        )

    task = MockTask(
        eval_strategy="sequential_block",
        n_blocks=2,
        block_idx=0,
        eval_blocks={},
        evaluate_fn=fake_evaluate,
    )

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["inc1", "inc2"],
        df_few_shots=task.pop_datapoints(n=1),
    )

    c1, c2 = Prompt("c1"), Prompt("c2")
    task.eval_blocks = {str(c1): {0}, str(c2): {0, 1}}
    optimizer.incumbents = [Prompt("i1"), Prompt("i2")]
    optimizer.challengers = [c1, c2]
    optimizer.population_size = 3

    optimizer._select_survivors()

    assert len(optimizer.challengers) == 1
    assert optimizer.challengers[0].instruction == "c2"


def test_capoeira_select_survivors_homogeneous_prunes_lowest(mock_meta_llm, mock_predictor):
    next_result: dict[str, EvalResult | None] = {"value": None}

    def fake_evaluate(prompts, *_, **__):
        return next_result["value"]  # type: ignore[return-value]

    task = MockTask(
        eval_strategy="sequential_block",
        n_blocks=2,
        block_idx=0,
        eval_blocks={},
        evaluate_fn=fake_evaluate,
    )

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["inc"],
        df_few_shots=task.pop_datapoints(n=1),
    )

    c1, c2 = Prompt("c1"), Prompt("c2")
    task.eval_blocks = {str(c1): {0}, str(c2): {0}}

    next_result["value"] = EvalResult(
        scores=np.array([[0.1], [0.2]], dtype=float),
        agg_scores=np.array([0.1, 0.2], dtype=float),
        sequences=np.array([["s1"], ["s2"]], dtype=object),
        input_tokens=np.array([[0.0], [0.0]], dtype=float),
        output_tokens=np.array([[0.0], [0.0]], dtype=float),
        agg_input_tokens=np.array([0.0, 0.0], dtype=float),
        agg_output_tokens=np.array([0.0, 0.0], dtype=float),
    )

    optimizer.incumbents = [Prompt("inc")]  # keeps population pressure
    optimizer.challengers = [c1, c2]
    optimizer.population_size = 2

    optimizer._select_survivors()

    assert len(optimizer.challengers) == 1
    assert optimizer.challengers[0].instruction == "c2"


def test_capoeira_select_survivors_prefers_lower_cost(mock_meta_llm, mock_predictor):
    def fake_evaluate(prompts, *_, **__):
        costs = np.array([1.0 if "cheap" in p.instruction else 5.0 for p in prompts], dtype=float)
        return EvalResult(
            scores=np.array([[0.4], [0.4]], dtype=float),
            agg_scores=np.array([0.4, 0.4], dtype=float),
            sequences=np.array([["s1"], ["s2"]], dtype=object),
            input_tokens=costs.reshape(-1, 1),
            output_tokens=np.zeros((len(prompts), 1)),
            agg_input_tokens=costs,
            agg_output_tokens=np.zeros(len(prompts)),
        )

    task = MockTask(
        eval_strategy="sequential_block",
        n_blocks=1,
        block_idx=0,
        eval_blocks={"cheap": {0}, "expensive": {0}},
        evaluate_fn=fake_evaluate,
    )

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["cheap", "expensive"],
        df_few_shots=task.pop_datapoints(n=1),
    )

    optimizer.incumbents = []
    optimizer.challengers = [Prompt("cheap"), Prompt("expensive")]
    optimizer.population_size = 1

    optimizer._select_survivors()

    assert len(optimizer.challengers) == 1
    assert optimizer.challengers[0].instruction == "cheap"


def test_capoeira_step_invokes_hooks(mock_meta_llm, mock_predictor, mock_df):
    task = MockTask()
    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p1", "p2"],
        df_few_shots=mock_df,
    )

    def fake_eval(prompts, *_, **__):
        n = len(prompts)
        return EvalResult(
            scores=np.zeros((n, 1), dtype=float),
            agg_scores=np.arange(n, dtype=float),
            sequences=np.array([[""] for _ in range(n)], dtype=object),
            input_tokens=np.ones((n, 1)),
            output_tokens=np.zeros((n, 1)),
            agg_input_tokens=np.ones(n),
            agg_output_tokens=np.zeros(n),
        )

    optimizer.task.evaluate = fake_eval  # type: ignore[assignment]
    optimizer.incumbents = [Prompt("inc")]
    optimizer.prompts = [Prompt("p1"), Prompt("p2")]

    with patch("promptolution.optimizers.capoeira.perform_crossover", return_value=[Prompt("c1")]), patch(
        "promptolution.optimizers.capoeira.perform_mutation", return_value=[Prompt("m1")]
    ), patch.object(optimizer, "_do_intensification") as do_int, patch.object(
        optimizer, "_advance_one_incumbent"
    ) as adv_inc, patch.object(
        optimizer, "_select_survivors"
    ) as sel:
        optimizer._step()

    assert do_int.call_count == 1
    assert adv_inc.call_count == 1
    assert sel.call_count == 1


def test_capoeira_do_intensification_updates_incumbents(mock_meta_llm, mock_predictor):
    def fake_eval(prompts, *_, **__):
        n = len(prompts)
        scores = np.arange(1, n + 1, dtype=float).reshape(n, 1)
        return EvalResult(
            scores=scores,
            agg_scores=scores.flatten(),
            sequences=np.array([[""] for _ in range(n)], dtype=object),
            input_tokens=np.ones((n, 1)),
            output_tokens=np.zeros((n, 1)),
            agg_input_tokens=np.ones(n),
            agg_output_tokens=np.zeros(n),
        )

    task = MockTask(eval_strategy="sequential_block", n_blocks=2, block_idx=0, evaluate_fn=fake_eval)
    challenger = Prompt("chal")
    inc1, inc2 = Prompt("i1"), Prompt("i2")
    task.prompt_evaluated_blocks = {str(inc1): {0}, str(inc2): {0}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [inc1, inc2]
    update_mock = patch.object(optimizer, "_update_incumbent_front", autospec=True).start()

    with patch("random.choice", side_effect=lambda seq: seq[0]):
        optimizer._do_intensification(challenger)

    patch.stopall()
    assert challenger in optimizer.incumbents
    update_mock.assert_called_once()


def test_capoeira_do_intensification_bootstrap_no_common_blocks(mock_meta_llm, mock_predictor):
    def fake_eval(prompts, *_, **__):
        n = len(prompts)
        return EvalResult(
            scores=np.zeros((n, 1)),
            agg_scores=np.zeros(n),
            sequences=np.array([[""] for _ in range(n)], dtype=object),
            input_tokens=np.zeros((n, 1)),
            output_tokens=np.zeros((n, 1)),
            agg_input_tokens=np.zeros(n),
            agg_output_tokens=np.zeros(n),
        )

    task = MockTask(eval_strategy="sequential_block", n_blocks=3, block_idx=0, evaluate_fn=fake_eval)
    inc1, inc2, challenger = Prompt("i1"), Prompt("i2"), Prompt("chal")
    task.prompt_evaluated_blocks = {str(inc1): {0}, str(inc2): {1}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [inc1, inc2]
    with patch("random.randrange", return_value=2), patch.object(
        optimizer, "_update_incumbent_front", autospec=True
    ) as upd:
        optimizer._do_intensification(challenger)

    assert task.block_idx == 2
    assert challenger in optimizer.incumbents
    upd.assert_called_once_with(blocks={2})


def test_capoeira_do_intensification_running_mean_path(monkeypatch, mock_meta_llm, mock_predictor):
    task = MockTask(eval_strategy="sequential_block", n_blocks=2, block_idx=0)
    inc1, inc2, challenger = Prompt("i1"), Prompt("i2"), Prompt("chal")
    task.prompt_evaluated_blocks = {str(inc1): {0, 1}, str(inc2): {0, 1}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [inc1, inc2]

    vec1 = np.array([[0.1, -0.1], [0.2, -0.2], [0.15, -0.15]])
    vec2 = np.array([[0.2, -0.2], [0.3, -0.3], [0.25, -0.25]])

    calls: list[tuple] = []

    def fake_is_dom(_self, v1, v2):
        calls.append((v1.copy(), v2.copy()))
        return False

    monkeypatch.setattr(Capoeira, "_is_dominated", fake_is_dom)

    with patch.object(Capoeira, "_get_objective_vectors", side_effect=[vec1, vec2]), patch(
        "random.choice", side_effect=lambda seq: list(seq)[0]
    ), patch.object(optimizer, "_update_incumbent_front", autospec=True) as upd:
        optimizer._do_intensification(challenger)

    # fold_vec path should call dominance check at least once
    assert calls, "_is_dominated should be invoked when challenger_mean already set"
    assert challenger in optimizer.incumbents
    upd.assert_called_once()


def test_capoeira_do_intensification_dominated_challenger(monkeypatch, mock_meta_llm, mock_predictor):
    task = MockTask(eval_strategy="sequential_block", n_blocks=1, block_idx=0)
    inc1, inc2, challenger = Prompt("i1"), Prompt("i2"), Prompt("chal")
    task.prompt_evaluated_blocks = {str(inc1): {0}, str(inc2): {0}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [inc1, inc2]

    dominated_vecs = np.array([[0.9, -0.1], [0.8, -0.1], [0.1, -0.1]])

    with patch.object(Capoeira, "_get_objective_vectors", return_value=dominated_vecs), patch(
        "random.choice", side_effect=lambda seq: list(seq)[0]
    ):
        optimizer._do_intensification(challenger)

    assert challenger in optimizer.challengers
    assert challenger not in optimizer.incumbents


def test_capoeira_update_incumbent_front_demotes(mock_meta_llm, mock_predictor):
    def fake_eval(prompts, *_, **__):
        scores = np.array([0.3, 0.1], dtype=float)
        return EvalResult(
            scores=scores.reshape(-1, 1),
            agg_scores=scores,
            sequences=np.array([["s1"], ["s2"]], dtype=object),
            input_tokens=np.zeros((2, 1)),
            output_tokens=np.zeros((2, 1)),
            agg_input_tokens=np.zeros(2),
            agg_output_tokens=np.zeros(2),
        )

    task = MockTask(eval_strategy="sequential_block", n_blocks=1, evaluate_fn=fake_eval)
    inc1, inc2 = Prompt("best"), Prompt("worst")
    task.prompt_evaluated_blocks = {str(inc1): {0}, str(inc2): {0}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [inc1, inc2]

    optimizer._update_incumbent_front()

    assert optimizer.incumbents == [inc1]
    assert inc2 in optimizer.challengers


def test_capoeira_advance_one_incumbent_no_gapblocks(mock_meta_llm, mock_predictor):
    task = MockTask(eval_strategy="sequential_block", n_blocks=2, block_idx=0)
    inc = Prompt("p1")
    task.prompt_evaluated_blocks = {str(inc): {0, 1}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p1"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [inc]

    called = {"evaluate": 0}

    def no_call(*args, **kwargs):
        called["evaluate"] += 1
        raise AssertionError("evaluate should not be called when no new blocks")

    task.evaluate = no_call  # type: ignore[assignment]

    optimizer._advance_one_incumbent()

    assert called["evaluate"] == 0


def test_capoeira_get_closest_incumbent_returns_nearest():
    challenger = np.array([0.5, 0.5])
    incumbents = np.array([[0.0, 0.0], [0.6, 0.6]])
    res = Capoeira._get_closest_incumbent(None, challenger, incumbents)
    assert np.allclose(res, incumbents[1])


def test_capoeira_objective_vectors_multiobjective(mock_meta_llm, mock_predictor, mock_df):
    t1 = MockTask(df=mock_df, n_subsamples=1, n_blocks=1)
    t2 = MockTask(df=mock_df, n_subsamples=1, n_blocks=1)
    multi_task = MultiObjectiveTask(tasks=[t1, t2])

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=multi_task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p"],
        df_few_shots=mock_df,
    )

    result = MultiObjectiveEvalResult(
        scores=[np.array([[0.1], [0.2]]), np.array([[0.3], [0.4]])],
        agg_scores=[np.array([0.1, 0.2]), np.array([0.3, 0.4])],
        sequences=np.array([["s1"], ["s2"]], dtype=object),
        input_tokens=np.array([[1.0], [2.0]]),
        output_tokens=np.array([[0.0], [0.0]]),
        agg_input_tokens=np.array([1.0, 2.0]),
        agg_output_tokens=np.array([0.0, 0.0]),
    )

    vecs = optimizer._get_objective_vectors(result)
    assert vecs.shape == (2, 3)
    assert np.allclose(vecs[:, 0], [0.1, 0.2])
    assert np.allclose(vecs[:, 1], [0.3, 0.4])
    assert np.allclose(vecs[:, 2], -np.array([1.0, 2.0]))


def test_capoeira_advance_one_incumbent_chooses_gap(mock_meta_llm, mock_predictor):
    def fake_eval(*_, **__):
        return EvalResult(
            scores=np.array([[0.0]]),
            agg_scores=np.array([0.0]),
            sequences=np.array([[""]], dtype=object),
            input_tokens=np.array([[0.0]]),
            output_tokens=np.array([[0.0]]),
            agg_input_tokens=np.array([0.0]),
            agg_output_tokens=np.array([0.0]),
        )

    task = MockTask(eval_strategy="sequential_block", n_blocks=3, block_idx=0, evaluate_fn=fake_eval)
    p1, p2 = Prompt("p1"), Prompt("p2")
    task.prompt_evaluated_blocks = {str(p1): {0}, str(p2): {0, 2}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p1", "p2"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [p1, p2]

    with patch("random.choice", side_effect=lambda seq: list(seq)[0]):
        optimizer._advance_one_incumbent()

    assert task.block_idx == 2


def test_capoeira_select_survivors_heterogeneous_removes_lowest(mock_meta_llm, mock_predictor):
    task = MockTask(eval_strategy="sequential_block", n_blocks=3)
    c1, c2 = Prompt("c1"), Prompt("c2")
    task.prompt_evaluated_blocks = {str(c1): {0}, str(c2): {1}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["inc"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = []
    optimizer.challengers = [c1, c2]
    optimizer.population_size = 1

    with patch("random.choice", side_effect=lambda seq: list(seq)[0]):
        optimizer._select_survivors()

    assert len(optimizer.challengers) == 1


def test_capoeira_select_survivors_incumbent_only(mock_meta_llm, mock_predictor):
    def fake_eval(prompts, *_, **__):
        n = len(prompts)
        vals = np.linspace(0.1, 0.2, n)
        return EvalResult(
            scores=np.tile(vals.reshape(n, 1), (1, 1)),
            agg_scores=vals,
            sequences=np.array([[""] for _ in range(n)], dtype=object),
            input_tokens=np.ones((n, 1)),
            output_tokens=np.zeros((n, 1)),
            agg_input_tokens=np.ones(n),
            agg_output_tokens=np.zeros(n),
        )

    task = MockTask(eval_strategy="sequential_block", n_blocks=2, evaluate_fn=fake_eval)
    inc1, inc2 = Prompt("i1"), Prompt("i2")
    task.prompt_evaluated_blocks = {str(inc1): {0}, str(inc2): {0}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["i1", "i2"],
        df_few_shots=task.pop_datapoints(n=1),
    )
    optimizer.incumbents = [inc1, inc2]
    optimizer.challengers = []
    optimizer.population_size = 1

    optimizer._select_survivors()

    assert len(optimizer.incumbents) == 1


def test_capoeira_get_common_blocks(mock_meta_llm, mock_predictor):
    task = MockTask(eval_strategy="sequential_block", n_blocks=2)
    p1, p2 = Prompt("p1"), Prompt("p2")
    task.prompt_evaluated_blocks = {str(p1): {0, 1}, str(p2): {1}}

    optimizer = Capoeira(
        predictor=mock_predictor,
        task=task,
        meta_llm=mock_meta_llm,
        initial_prompts=["p1", "p2"],
        df_few_shots=task.pop_datapoints(n=1),
    )

    common = optimizer._get_common_blocks([p1, p2])
    assert common == {1}


def test_capoeira_is_dominated_logic():
    assert Capoeira._is_dominated(np.array([0.1, 0.1]), np.array([0.2, 0.2]))
    assert not Capoeira._is_dominated(np.array([0.3, 0.2]), np.array([0.3, 0.2]))
    assert not Capoeira._is_dominated(np.array([0.4, 0.5]), np.array([0.3, 0.6]))


def test_capoeira_calculate_crowding_distance_three_points():
    vecs = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    dists = Capoeira._calculate_crowding_distance(vecs)
    assert np.isinf(dists[[0, -1]]).all()
    assert dists[1] > 0
