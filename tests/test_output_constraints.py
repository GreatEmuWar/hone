"""Constraints on outputs: limits judged once an evaluation reports what they need, written like any other constraint."""
import math
import pickle

import pytest
from pydantic import ValidationError

from hone import (
    CMAES,
    NSGA2,
    BayesianConfig,
    BayesianOptimizer,
    ChoiceParameter,
    CMAESConfig,
    Constraint,
    EvaluationFailed,
    GeneticOptimizer,
    GeneticOptimizerConfig,
    MultiObjectiveBayesian,
    MultiObjectiveBayesianConfig,
    MultiObjectiveRandomSearch,
    MultiObjectiveConfig,
    NSGA2Config,
    OptimizerConfig,
    Output,
    ParetoResult,
    RandomSearch,
    RangeParameter,
    Result,
    SearchSpace,
    UnsupportedSearchSpace,
    sensitivity,
)

X = RangeParameter("x", 0, 10)
Y = RangeParameter("y", 0, 10)
TOTAL = Output("total")
SPACE = SearchSpace([X, Y], [TOTAL <= 4])  # the optimum, nearest (5, 5) with x + y <= 4, is (2, 2), at 18


def bowl(x, y):
    """The squared distance from (5, 5), which the output limit keeps the search away from."""
    return {"objective": (x - 5) ** 2 + (y - 5) ** 2, "total": x + y}


# --- writing them --------------------------------------------------------------------


def test_an_output_is_a_term_like_a_parameter():
    constraint = TOTAL * 2 <= X + 3
    assert str(constraint) == "total * 2 <= x + 3"
    assert constraint.names() == {"x"} and constraint.output_names() == {"total"}
    assert constraint.satisfied({"x": 1}, {"total": 2}) and not constraint.satisfied({"x": 1}, {"total": 2.5})
    assert constraint.margin({"x": 1}, {"total": 2.5}) == 1.0


def test_a_search_space_sorts_its_constraints_by_when_they_can_be_judged():
    before, after = X + Y <= 15, TOTAL >= 1
    space = SearchSpace([X, Y], [before, after])
    assert space.constraints == [before, after]
    assert space.parameter_constraints == [before] and space.output_constraints == [after]
    assert space.outputs == ["total"]


def test_output_constraints_are_not_checked_before_evaluating():
    space = SearchSpace([X, Y], [X + Y <= 15, TOTAL >= 1])
    assert space.feasible({"x": 1, "y": 1}) and space.broken({"x": 9, "y": 9}) == [space.constraints[0]]


def test_the_violation_is_relative_to_each_limit_so_units_compare():
    space = SearchSpace([X, Y], [Output("stress") <= 250, Output("deflection") <= 0.5])
    # 10% over each limit counts the same, however different their sizes.
    assert space.output_violation({"x": 1, "y": 1}, {"stress": 275, "deflection": 0.5}) == pytest.approx(0.1)
    assert space.output_violation({"x": 1, "y": 1}, {"stress": 250, "deflection": 0.55}) == pytest.approx(0.1)
    assert space.output_violation({"x": 1, "y": 1}, {"stress": 200, "deflection": 0.4}) == 0


def test_a_limit_of_zero_counts_its_violation_as_it_is():
    space = SearchSpace([X, Y], [Output("gap") <= 0])
    assert space.output_violation({"x": 1, "y": 1}, {"gap": 0.3}) == pytest.approx(0.3)


def test_a_function_constraint_can_take_outputs():
    def stress_ratio(x, stress):
        return stress / (100 * x) - 1

    constraint = Constraint.function(stress_ratio, outputs=["stress"])
    space = SearchSpace([X, Y], [constraint])
    assert space.output_constraints == [constraint] and space.outputs == ["stress"]
    assert constraint.names() == {"x"}
    assert constraint.satisfied({"x": 2}, {"stress": 150}) and not constraint.satisfied({"x": 1}, {"stress": 150})


def test_a_function_constraint_on_outputs_is_saved_by_name_and_rebuilt():
    def ratio(stress):
        return stress / 250 - 1

    space = SearchSpace([X, Y], [Constraint.function(ratio, outputs=["stress"])])
    loaded = SearchSpace.model_validate_json(space.model_dump_json())
    assert loaded.outputs == ["stress"]
    rebuilt = loaded.with_functions({"ratio": ratio})
    assert rebuilt.output_constraints[0].satisfied({}, {"stress": 200})


def test_an_output_cant_share_a_parameters_name():
    with pytest.raises(ValidationError, match="output named 'x', which is also a parameter"):
        SearchSpace([X, Y], [Output("x") <= 1])


def test_an_output_cant_be_tested_on_the_left_of_a_condition():
    with pytest.raises(ValidationError, match="tests the output 'total'"):
        SearchSpace([X, Y], conditions=[(TOTAL >= 3) >> Y])
    with pytest.raises(ValidationError, match="tests the output 'total'"):
        SearchSpace([X, Y], [(TOTAL >= 3) >> (X <= 2)])


def test_a_conditional_output_constraint_applies_only_while_its_test_holds():
    material = ChoiceParameter("material", ["steel", "aluminium"])
    stress = Output("stress")
    space = SearchSpace([X, material], [material["steel"] >> (stress <= 400), material["aluminium"] >> (stress <= 200)])
    assert space.output_violation({"x": 1, "material": "steel"}, {"stress": 300}) == 0
    assert space.output_violation({"x": 1, "material": "aluminium"}, {"stress": 300}) == pytest.approx(0.5)
    assert space.output_margins({"x": 1, "material": "steel"}, {"stress": 300}) == [-0.25, None]


def test_a_chained_comparison_with_an_output_still_raises():
    with pytest.raises(TypeError, match="write two constraints"):
        bool(0 <= TOTAL <= 4)


def test_output_constraints_save_and_load_with_the_search_space():
    space = SearchSpace([X, Y], [TOTAL <= 2 * X + 1, abs(Output("tilt")) <= 0.1])
    assert SearchSpace.model_validate_json(space.model_dump_json()) == space


# --- what the objective returns ----------------------------------------------------


def test_the_objective_returns_its_value_and_outputs_in_a_dict():
    result = RandomSearch(SPACE, OptimizerConfig(max_evaluations=200, seed=0)).optimize(bowl)
    assert result.feasible and result.best_outputs is not None
    assert result.best_outputs["total"] <= 4
    assert result.best_fitness == bowl(**result.best_params)["objective"]


@pytest.mark.parametrize(
    "value, message",
    [
        (3.0, "must be a dict"),
        ({"objective": 1.0}, "missing total"),
        ({"objective": 1.0, "total": 2.0, "other": 3.0}, "unknown other"),
        ({"objective": 1.0, "total": "two"}, "value of total must be a number"),
    ],
)
def test_a_dict_that_doesnt_fit_raises_type_error(value, message):
    optimizer = RandomSearch(SPACE, OptimizerConfig(max_evaluations=10, seed=0))
    trial = optimizer.ask(1)[0]
    with pytest.raises(TypeError, match=message):
        optimizer.tell(trial, value)


def test_nan_in_an_output_marks_the_evaluation_failed():
    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=2, max_evaluations=2, seed=0))
    first, second = optimizer.ask()
    optimizer.tell(first, {"objective": 1.0, "total": math.nan})
    optimizer.tell(second, {"objective": 5.0, "total": 1.0})
    result = optimizer.result()
    assert result.failures == 1 and result.best_fitness == 5.0


def test_evaluation_failed_still_works():
    def flaky(x, y):
        if x > 5:
            raise EvaluationFailed
        return bowl(x, y)

    result = RandomSearch(SPACE, OptimizerConfig(max_evaluations=100, seed=0)).optimize(flaky)
    assert result.failures > 0 and result.feasible


def test_noise_is_refused():
    with pytest.raises(UnsupportedSearchSpace, match="noise"):
        RandomSearch(SPACE, OptimizerConfig(noise=True))


# --- ranking ----------------------------------------------------------------------------


def test_a_feasible_point_beats_any_infeasible_one_however_small_its_value():
    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=3, max_evaluations=3, seed=0))
    trials = optimizer.ask()
    optimizer.tell(trials[0], {"objective": -1000.0, "total": 5.0})  # far better value, breaks the limit
    optimizer.tell(trials[1], {"objective": 30.0, "total": 3.0})
    optimizer.tell(trials[2], {"objective": 20.0, "total": 4.0})
    result = optimizer.result()
    assert result.feasible and result.best_fitness == 20.0 and result.infeasible == 1
    stats = result.generations[0]
    assert stats.infeasible == 1 and stats.best_fitness == 20.0 and stats.worst_fitness == 30.0


def test_with_nothing_feasible_the_least_violating_point_is_the_result():
    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=2, max_evaluations=2, seed=0))
    first, second = optimizer.ask()
    optimizer.tell(first, {"objective": 1.0, "total": 8.0})
    optimizer.tell(second, {"objective": 50.0, "total": 5.0})
    result = optimizer.result()
    assert not result.feasible and result.best_fitness == 50.0 and result.best_outputs == {"total": 5.0}
    assert result.generations[0].best_fitness is None and result.generations[0].infeasible == 2


def test_only_a_feasible_point_reaches_the_threshold():
    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=2, generations=3, threshold=0.0, seed=0))
    for trial in optimizer.ask():
        optimizer.tell(trial, {"objective": -1.0, "total": 9.0})
    assert not optimizer.done
    first, second = optimizer.ask()
    optimizer.tell(first, {"objective": 5.0, "total": 9.0})
    optimizer.tell(second, {"objective": -1.0, "total": 1.0})
    assert optimizer.done and optimizer.result().reached_threshold


def test_maximizing_ranks_feasible_points_by_the_larger_value():
    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=3, max_evaluations=3, maximize=True, seed=0))
    trials = optimizer.ask()
    for trial, value in zip(trials, [(10.0, 1.0), (99.0, 9.0), (5.0, 2.0)]):
        optimizer.tell(trial, {"objective": value[0], "total": value[1]})
    assert optimizer.result().best_fitness == 10.0


# --- every optimizer keeps to them ----------------------------------------------------------


@pytest.mark.parametrize(
    "optimizer, budget, tolerance",
    [
        (lambda: GeneticOptimizer(SPACE, GeneticOptimizerConfig(max_evaluations=600, seed=0)), 600, 0.3),
        (lambda: GeneticOptimizer(SPACE, GeneticOptimizerConfig(max_evaluations=600, seed=0, selection_method="roulette")), 600, 0.5),
        (lambda: CMAES(SPACE, CMAESConfig(max_evaluations=600, seed=0)), 600, 0.1),
        (lambda: BayesianOptimizer(SPACE, BayesianConfig(max_evaluations=40, seed=0)), 40, 0.05),
    ],
)
def test_each_optimizer_finds_the_optimum_on_the_boundary(optimizer, budget, tolerance):
    result = optimizer().optimize(bowl)
    assert result.feasible and result.best_outputs["total"] <= 4 + 1e-9
    assert result.best_fitness == pytest.approx(18, abs=tolerance)
    assert result.evaluations == budget


def test_bayesian_optimization_learns_where_the_limit_is():
    # Random search spends most of its budget beyond the limit; the constraint's model steers away from it.
    bayesian = BayesianOptimizer(SPACE, BayesianConfig(max_evaluations=40, seed=0)).optimize(bowl)
    random = RandomSearch(SPACE, OptimizerConfig(max_evaluations=40, seed=0)).optimize(bowl)
    assert bayesian.best_fitness < random.best_fitness


def test_bayesian_optimization_searches_for_feasibility_first():
    # Only a corner is feasible, and the start is spread over the whole square.
    corner = SearchSpace([X, Y], [Output("distance") <= 1])

    def far_corner(x, y):
        return {"objective": x + y, "distance": math.hypot(x - 9, y - 9)}

    result = BayesianOptimizer(corner, BayesianConfig(max_evaluations=30, initial_samples=6, seed=1)).optimize(far_corner)
    assert result.feasible


def test_bayesian_optimization_handles_a_conditional_output_constraint():
    material = ChoiceParameter("material", ["steel", "aluminium"])
    stress = Output("stress")
    space = SearchSpace([X, material], [material["aluminium"] >> (stress <= 2)])

    def beam(x, material):
        # Aluminium is cheaper, but its stress limit rules out thin designs.
        return {"objective": (10 - x) * (1 if material == "aluminium" else 3), "stress": 10 / (1 + x)}

    result = BayesianOptimizer(space, BayesianConfig(max_evaluations=40, seed=0)).optimize(beam)
    assert result.feasible
    assert result.best_params["material"] == "steel" or result.best_outputs["stress"] <= 2


def test_ask_and_tell_with_bayesian_optimization():
    optimizer = BayesianOptimizer(SPACE, BayesianConfig(max_evaluations=20, seed=0))
    while not optimizer.done:
        for trial in optimizer.ask():
            optimizer.tell(trial, bowl(**trial.params))
    assert optimizer.result().feasible


def test_a_mixed_limit_of_outputs_and_parameters():
    space = SearchSpace([X, Y], [TOTAL <= X + 1])  # x + y <= x + 1, so y <= 1, and the optimum is (5, 1), at 16

    result = CMAES(space, CMAESConfig(max_evaluations=400, seed=0)).optimize(bowl)
    assert result.feasible and result.best_fitness == pytest.approx(16, abs=0.05)


def test_workers_return_the_dict_from_other_processes():
    result = RandomSearch(SPACE, OptimizerConfig(population_size=4, max_evaluations=8, seed=0)).optimize(bowl, workers=2)
    assert result.evaluations == 8 and result.best_outputs is not None


# --- results, checkpoints, rebuilding ------------------------------------------------------


def test_a_result_saves_and_loads_with_its_outputs(tmp_path):
    result = RandomSearch(SPACE, OptimizerConfig(max_evaluations=50, seed=0)).optimize(bowl)
    path = tmp_path / "result.json"
    result.save(path)
    loaded = Result.load(path)
    assert loaded == result and loaded.best_outputs == result.best_outputs and loaded.run.search_space == SPACE
    again = loaded.rebuild_optimizer().optimize(bowl)
    assert again == result


def test_a_checkpoint_carries_on_the_run(tmp_path):
    path = tmp_path / "run.checkpoint"
    whole = GeneticOptimizer(SPACE, GeneticOptimizerConfig(max_evaluations=200, seed=0)).optimize(bowl)

    def killed_at_generation_3(progress):
        if progress.generation.generation == 3:
            raise KeyboardInterrupt  # as if the run were killed here

    stopped = GeneticOptimizer(SPACE, GeneticOptimizerConfig(max_evaluations=200, seed=0))
    assert stopped.optimize(bowl, checkpoint=path, callback=killed_at_generation_3).stopped
    resumed = GeneticOptimizer(SPACE, GeneticOptimizerConfig(max_evaluations=200, seed=0)).optimize(bowl, checkpoint=path)
    assert resumed == whole


def test_a_checkpoint_from_the_previous_format_is_refused(tmp_path):
    path = tmp_path / "old.checkpoint"
    path.write_bytes(pickle.dumps({"format": 1}))
    with pytest.raises(ValueError, match="not a checkpoint this version of hone can read"):
        RandomSearch(SPACE).load_checkpoint(path)


def test_progress_says_whether_the_best_is_feasible(capsys):
    from hone import print_progress

    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=2, generations=1, seed=0))
    seen = []
    optimizer.optimize(lambda x, y: {"objective": 1.0, "total": 100.0}, callback=lambda p: seen.append(p) or print_progress(p))
    assert not seen[0].feasible
    assert "breaks an output constraint" in capsys.readouterr().out


# --- several objectives -------------------------------------------------------------------


def front_problem(x, y):
    return {"a": x, "b": (10 - x) * (1 + y), "total": x + y, "cost": y}


FRONT_SPACE = SearchSpace([X, Y], [TOTAL >= 2, Output("cost") <= 5])
OBJECTIVES = {"a": "minimize", "b": "minimize"}


@pytest.mark.parametrize(
    "build",
    [
        lambda: MultiObjectiveRandomSearch(FRONT_SPACE, MultiObjectiveConfig(objectives=OBJECTIVES, max_evaluations=300, seed=0)),
        lambda: NSGA2(FRONT_SPACE, NSGA2Config(objectives=OBJECTIVES, max_evaluations=600, seed=0)),
        lambda: MultiObjectiveBayesian(FRONT_SPACE, MultiObjectiveBayesianConfig(objectives=OBJECTIVES, max_evaluations=30, seed=0)),
        lambda: MultiObjectiveBayesian(
            FRONT_SPACE, MultiObjectiveBayesianConfig(objectives=OBJECTIVES, max_evaluations=30, seed=0, method="models")
        ),
    ],
)
def test_the_front_holds_only_feasible_points(build):
    result = build().optimize(front_problem)
    assert result.feasible and len(result.front) > 0
    for point in result.front:
        assert point.outputs is not None
        assert point.outputs["total"] >= 2 - 1e-9 and point.outputs["cost"] <= 5 + 1e-9


def test_an_objective_can_be_an_output_too():
    space = SearchSpace([X, Y], [Output("a") >= 3])

    def problem(x, y):
        return {"a": x, "b": 10 - x + y}

    result = NSGA2(space, NSGA2Config(objectives=OBJECTIVES, max_evaluations=300, seed=0)).optimize(problem)
    assert all(point.values["a"] >= 3 - 1e-9 for point in result.front)
    assert result.front[0].outputs == {"a": result.front[0].values["a"]}


def test_nsga2_keeps_feasible_points_ahead_of_infeasible_ones():
    optimizer = NSGA2(FRONT_SPACE, NSGA2Config(objectives=OBJECTIVES, population_size=4, max_evaluations=8, seed=0))
    pool = []
    for trial, total in zip(optimizer.ask(), [1.0, 1.5, 3.0, 4.0]):
        optimizer.tell(trial, {"a": 0.0, "b": 0.0, "total": total, "cost": 0.0})
        pool.append(total)
    population = optimizer._population
    survivors = optimizer._survivors([ind for ind in population if ind.evaluated], 3)
    assert [ind.violation == 0 for ind in survivors] == [True, True, False]
    assert survivors[2].outputs["total"] == 1.5  # the smaller violation of the two that break the limit


def test_a_front_with_nothing_feasible_is_empty():
    optimizer = MultiObjectiveRandomSearch(FRONT_SPACE, MultiObjectiveConfig(objectives=OBJECTIVES, population_size=2, max_evaluations=2))
    for trial in optimizer.ask():
        optimizer.tell(trial, {"a": 1.0, "b": 1.0, "total": 0.0, "cost": 0.0})
    result = optimizer.result()
    assert not result.feasible and len(result.front) == 0 and result.infeasible == 2
    loaded = ParetoResult.model_validate_json(result.model_dump_json())
    assert loaded == result


def test_a_multi_objective_dict_needs_the_outputs():
    optimizer = MultiObjectiveRandomSearch(FRONT_SPACE, MultiObjectiveConfig(objectives=OBJECTIVES, population_size=2, max_evaluations=2))
    with pytest.raises(TypeError, match="missing cost, total"):
        optimizer.tell(optimizer.ask()[0], {"a": 1.0, "b": 1.0})


# --- sensitivity -------------------------------------------------------------------------


def test_sensitivity_measures_the_objective_and_ignores_the_outputs():
    report = sensitivity(lambda x, y: {"objective": 3 * x + 0.1 * y, "total": x + y}, SPACE, max_evaluations=60, seed=0)
    assert report.ranking == ["x", "y"]


def test_sensitivity_still_refuses_constraints_on_the_parameters():
    with pytest.raises(UnsupportedSearchSpace, match="on the parameters"):
        sensitivity(lambda x, y: x, SearchSpace([X, Y], [X <= Y]), max_evaluations=60)


# --- the example and the benchmark's problems ---------------------------------------------


def test_the_welded_beam_example_comes_within_two_percent_of_the_published_design():
    import subprocess
    import sys
    from pathlib import Path

    example = Path(__file__).parent.parent / "examples" / "welded_beam.py"
    output = subprocess.run([sys.executable, str(example)], capture_output=True, text=True, check=True).stdout
    cost = float(output.split()[1].rstrip(","))
    assert cost < 1.724852 * 1.02


@pytest.mark.parametrize("name", ["pressure_vessel", "tension_spring", "welded_beam"])
def test_each_output_problem_agrees_with_its_parameter_version(name):
    import numpy as np

    from hone.benchmarks import constrained

    before, after = getattr(constrained, name)(), getattr(constrained, f"{name}_outputs")()
    assert after.optimum == before.optimum and after.search_space.output_constraints
    rng = np.random.default_rng(0)
    agreed = 0
    for _ in range(300):
        params = before.search_space.parameters
        point = {p.name: p.sample(rng) for p in params}
        reported = after.function(**point)
        assert reported["objective"] == pytest.approx(before.function(**point))
        feasible_after = after.search_space.feasible(point) and after.search_space.output_violation(point, reported) == 0
        agreed += feasible_after == before.search_space.feasible(point)
    assert agreed == 300
