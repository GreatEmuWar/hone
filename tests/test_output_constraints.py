"""Constraints on outputs: limits judged once an evaluation reports what they need, written like any other constraint."""
import math
import pickle
from statistics import NormalDist

import numpy as np

import pytest
from pydantic import ValidationError

from hone.optimizers.base import MARGIN_FREEDOM
from hone.optimizers.core import Individual
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


def test_only_a_limit_that_is_a_number_scales_the_violation():
    # A limit that varies with the point, or one of zero, counts in its own units, so the violation changes smoothly.
    assert (Output("s") <= 250).relative_violation({}, {"s": 275}) == pytest.approx(0.1)
    assert (250 >= Output("s")).relative_violation({}, {"s": 275}) == pytest.approx(0.1)
    assert (Output("s") - 250 <= 0).relative_violation({}, {"s": 275}) == pytest.approx(25)
    varying = Output("s") <= X - 0.5
    assert [varying.relative_violation({"x": x}, {"s": 0.01}) for x in (0.5001, 0.5, 0.4999)] == pytest.approx([0.0099, 0.01, 0.0101])


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


# --- noise ------------------------------------------------------------------------------


def fixed_point(**config):
    """Random search over a space with one point, so every evaluation is of the same design."""
    config = {"generations": 1, "noise": True, "final_evaluations": 0, **config}
    space = SearchSpace([RangeParameter("n", 0, 0, integer=True)], [TOTAL <= 4])
    return RandomSearch(space, OptimizerConfig(**config))


def tell_all(optimizer, outcomes):
    for trial, outcome in zip(optimizer.ask(), outcomes):
        optimizer.tell(trial, outcome)


def test_noise_is_refused_for_several_objectives():
    space = SearchSpace([X, Y], [TOTAL <= 4])
    with pytest.raises(UnsupportedSearchSpace, match="several objectives"):
        MultiObjectiveRandomSearch(space, MultiObjectiveConfig(objectives={"a": "minimize", "b": "minimize"}, noise=True))


def test_outputs_are_averaged_and_a_limit_is_judged_on_the_average():
    optimizer = fixed_point(population_size=3)
    tell_all(optimizer, [{"objective": 1.0, "total": 5.0}, {"objective": 2.0, "total": 3.0}, {"objective": 6.0, "total": 3.0}])
    result = optimizer.result()
    assert result.best_fitness == 3.0 and result.best_outputs == {"total": pytest.approx(11 / 3)}
    assert result.feasible, "the average total, 3.67, is within the limit though one draw broke it"
    assert result.infeasible == 1, "the draws that broke it are still counted"


def test_a_design_that_passes_a_limit_by_luck_is_infeasible_once_its_average_breaks_it():
    optimizer = fixed_point(population_size=3)
    tell_all(optimizer, [{"objective": 1.0, "total": 3.0}, {"objective": 1.0, "total": 5.0}, {"objective": 1.0, "total": 5.0}])
    result = optimizer.result()
    assert not result.feasible and result.best_outputs == {"total": pytest.approx(13 / 3)}
    first, _, last = optimizer._population
    assert first.violation == 0 and first.outputs == {"total": 3.0}, "recorded before the other draws arrived"
    assert last.violation == pytest.approx((13 / 3 - 4) / 4) and last.outputs == {"total": pytest.approx(13 / 3)}, "recorded with every draw in"


def test_designs_rank_by_the_violation_of_their_averages_then_their_average_value():
    """A lucky design that passes once beats a steady one on value, but not once its average breaks the limit."""
    space = SearchSpace([ChoiceParameter("design", ["lucky", "steady"])], [TOTAL <= 4])
    optimizer = RandomSearch(space, OptimizerConfig(population_size=4, generations=1, noise=True, final_evaluations=0, seed=0))
    designs = iter(["lucky", "lucky", "steady", "steady"])
    optimizer._random_individual = lambda generation: Individual(params={"design": next(designs)}, generation=generation)
    totals = {"lucky": iter([3.0, 7.0]), "steady": iter([3.0, 3.0])}
    for trial in optimizer.ask():
        design = trial.params["design"]
        optimizer.tell(trial, {"objective": 0.0 if design == "lucky" else 1.0, "total": next(totals[design])})
    result = optimizer.result()
    assert result.best_params == {"design": "steady"} and result.best_fitness == 1.0 and result.feasible


def test_progress_and_the_threshold_follow_the_average_outputs():
    optimizer = fixed_point(population_size=2, threshold=10.0)
    seen = []
    first, second = optimizer.ask()
    optimizer.tell(first, {"objective": 1.0, "total": 3.0})
    optimizer.tell(second, {"objective": 1.0, "total": 7.0})
    optimizer._report(seen.append, 0)
    assert not optimizer.result().reached_threshold and not optimizer.result().feasible
    assert not seen[0].feasible, "the average total, 5, breaks the limit, though the first draw passed it"


def test_the_genetic_optimizer_refreshes_violations_when_it_breeds():
    space = SearchSpace([RangeParameter("n", 0, 0, integer=True)], [TOTAL <= 4])
    config = GeneticOptimizerConfig(
        population_size=4, tournament_size=2, generations=3, noise=True, final_evaluations=0, confidence=None, seed=0
    )
    optimizer = GeneticOptimizer(space, config)
    tell_all(optimizer, [{"objective": 0.0, "total": 3.0}, {"objective": 1.0, "total": 7.0}, {"objective": 1.0, "total": 7.0}, {"objective": 1.0, "total": 7.0}])
    first = optimizer._population[0]
    assert first.violation == 0 and first.outputs == {"total": 3.0}, "its own draw, before the others arrived"
    optimizer.ask()
    assert first.violation == pytest.approx(0.5) and first.outputs == {"total": 6.0} and first.fitness == 0.75


@pytest.mark.parametrize(
    "config, reserve",
    [
        ({"noise": True, "max_evaluations": 60}, 10),
        ({"noise": True, "max_evaluations": 300}, 30),
        ({"noise": True, "max_evaluations": 8}, 4),
        ({"noise": True}, 10),
        ({"noise": True, "max_evaluations": 60, "final_evaluations": 0}, 0),
        ({"noise": True, "max_evaluations": 60, "final_evaluations": 3}, 3),
        ({"max_evaluations": 60}, 0),
    ],
)
def test_the_final_round_is_on_by_default_with_noise_and_output_constraints(config, reserve):
    assert RandomSearch(SPACE, OptimizerConfig(population_size=10, **config))._reserve == reserve
    assert RandomSearch([X, Y], OptimizerConfig(population_size=10, **config))._reserve == (config.get("final_evaluations") or 0) * ("noise" in config)


def test_without_the_margin_the_final_round_re_evaluates_the_finalists_and_judges_them_on_their_averages():
    rng = np.random.default_rng(0)

    def noisy_bowl(x, y):
        value = bowl(x, y)
        return {"objective": value["objective"] + rng.normal(), "total": value["total"] + 0.5 * rng.normal()}

    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=10, max_evaluations=60, noise=True, confidence=None, seed=0))
    result = optimizer.optimize(noisy_bowl)
    assert result.evaluations == 60 and len(optimizer._finalists) == 5
    key = optimizer._key(optimizer.search_space.check(result.best_params))
    assert key in optimizer._finalists and len(optimizer._observations[key]) >= 3
    assert result.feasible == (optimizer._violation(key) == 0)
    assert result.best_outputs == optimizer._average_outputs(key) and result.best_fitness == optimizer._average(key)


def _one_point(totals, **config):
    """A genetic optimizer whose only point has been evaluated once per total, its violation then refreshed."""
    space = SearchSpace([RangeParameter("n", 0, 0, integer=True)], [TOTAL <= 4])
    config = {"population_size": len(totals), "tournament_size": 2, "generations": 3, "noise": True, "final_evaluations": 0, "seed": 0, **config}
    optimizer = GeneticOptimizer(space, GeneticOptimizerConfig(**config))
    tell_all(optimizer, [{"objective": 0.0, "total": t} for t in totals])
    return optimizer, optimizer._key(optimizer._population[0].params)


def test_the_margin_widens_a_limit_by_k_standard_errors_of_the_pooled_noise():
    optimizer, key = _one_point([3.0, 7.0, 7.0, 7.0])
    # relative margins -0.25, 0.75, 0.75, 0.75: mean 0.5, pooled sd 0.5 on 3 degrees of freedom
    assert optimizer._limit_noise() == [(pytest.approx(0.5), 3)]
    k = NormalDist().inv_cdf(0.95)
    assert optimizer._violation(key) == pytest.approx(0.5 + k * 0.5 / math.sqrt(4))


def test_a_design_just_inside_a_noisy_limit_fails_the_margin_and_one_clear_of_it_passes():
    inside = [3.9, 3.8, 4.1, 3.6]  # average 3.85, under 4 by less than its noise
    assert _one_point(inside, confidence=None)[0]._violation(_one_point(inside, confidence=None)[1]) == 0
    optimizer, key = _one_point(inside)
    assert optimizer._violation(key) > 0
    optimizer, key = _one_point([t - 1 for t in inside])  # average 2.85
    assert optimizer._violation(key) == 0


def test_without_enough_repeats_a_limit_is_judged_on_the_average_alone():
    optimizer, key = _one_point([3.9, 3.8, 4.1])  # 2 degrees of freedom
    assert optimizer._limit_noise()[0][1] == 2 and optimizer._violation(key) == 0


def test_with_the_margin_the_final_round_comes_in_two_halves():
    rng = np.random.default_rng(0)

    def noisy_bowl(x, y):
        value = bowl(x, y)
        return {"objective": value["objective"] + rng.normal(), "total": value["total"] + 0.5 * rng.normal()}

    optimizer = RandomSearch(SPACE, OptimizerConfig(population_size=10, max_evaluations=60, noise=True, seed=0))
    result = optimizer.optimize(noisy_bowl)
    assert result.evaluations == 60 and optimizer._final_half == 2
    assert [g.evaluations for g in result.generations][-3:] == [50, 55, 60], "the search, then two halves of 5"
    assert optimizer._limit_noise()[0][1] >= MARGIN_FREEDOM
    key = optimizer._key(optimizer.search_space.check(result.best_params))
    assert key in optimizer._finalists and result.feasible == (optimizer._violation(key) == 0)


@pytest.mark.parametrize("noise, space", [(False, SPACE), (True, [X, Y])])
def test_without_noise_or_without_output_constraints_the_margin_changes_nothing(noise, space):
    rng = np.random.default_rng(0)

    def objective(x, y):
        value = bowl(x, y)["objective"] + (rng.normal() if noise else 0.0)
        return {"objective": value, "total": x + y} if space is SPACE else value

    def run(confidence):
        rng.bit_generator.state = np.random.default_rng(0).bit_generator.state
        config = GeneticOptimizerConfig(population_size=10, max_evaluations=60, noise=noise, confidence=confidence, seed=0)
        return GeneticOptimizer(space, config).optimize(objective)

    with_margin, without = run(0.95), run(None)
    assert (with_margin.best_params, with_margin.best_fitness, with_margin.evaluations) == (without.best_params, without.best_fitness, without.evaluations)
    assert [g.evaluations for g in with_margin.generations] == [g.evaluations for g in without.generations]


def test_every_optimizer_runs_noise_with_output_constraints():
    rng = np.random.default_rng(1)

    def noisy_bowl(x, y):
        value = bowl(x, y)
        return {"objective": value["objective"] + rng.normal(), "total": value["total"] + 0.1 * rng.normal()}

    for optimizer in [
        GeneticOptimizer(SPACE, GeneticOptimizerConfig(population_size=10, max_evaluations=100, noise=True, seed=0)),
        CMAES(SPACE, CMAESConfig(max_evaluations=100, noise=True, seed=0)),
        BayesianOptimizer(SPACE, BayesianConfig(max_evaluations=25, noise=True, seed=0)),
    ]:
        result = optimizer.optimize(noisy_bowl)
        assert result.evaluations == optimizer.config.max_evaluations and result.feasible
        assert result.best_params["x"] + result.best_params["y"] < 4.5, type(optimizer).__name__


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

    result = CMAES(space, CMAESConfig(max_evaluations=800, seed=0)).optimize(bowl)
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


def test_a_checkpoint_of_an_unknown_format_is_refused(tmp_path):
    path = tmp_path / "old.checkpoint"
    path.write_bytes(pickle.dumps({"format": 99}))
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


def test_a_margin_that_cant_be_computed_is_modelled_as_infeasible():
    space = SearchSpace([X], [Output("a") / Output("b") <= 1])
    optimizer = BayesianOptimizer(space, BayesianConfig(max_evaluations=10, seed=0))
    for x, b in [(1, 1.0), (2, 1.0), (8, 0.0), (9, 0.0)]:  # b = 0 divides by zero: the margin is infinite
        individual = Individual(params={"x": float(x)}, generation=0, fitness=0.0, outputs={"a": 0.5, "b": b})
        individual.violation = space.output_violation(individual.params, individual.outputs)
        optimizer._X.append(optimizer._encoding.encode(individual.params))
        optimizer._absorb_outputs(individual)
    models = optimizer._constraint_models(np.array(optimizer._X), [])
    probability = np.exp(optimizer._log_feasibility(models)(optimizer._encoding.encode({"x": 8.5})[None, :]))
    assert probability[0] < 0.5


def test_roulette_with_one_feasible_point_picks_it():
    from hone.optimizers.genetic import selection

    rng = np.random.default_rng(0)
    population = [Individual(params={}, generation=0, fitness=f, violation=v) for f, v in [(1.0, 0.0), (0.0, 0.5), (0.0, 3.0), (0.0, 10.0)]]
    picks = {id(selection.roulette(population, rng, maximize=False)) for _ in range(200)}
    assert picks == {id(population[0])}


def test_roulette_with_nothing_feasible_favours_the_smaller_violation_even_beside_an_infinite_one():
    from hone.optimizers.genetic import selection

    rng = np.random.default_rng(0)
    population = [Individual(params={}, generation=0, fitness=0.0, violation=v) for v in [0.01, 0.02, math.inf]]
    picks = [selection.roulette(population, rng, maximize=False).violation for _ in range(400)]
    assert picks.count(0.01) == 400  # the worst finite violation has weight zero, as the worst fitness does


# --- found in review ---------------------------------------------------------------------


def test_a_limit_and_its_mirror_image_agree():
    w = RangeParameter("w", 1, 10)
    s = Output("s")
    for first, second in [(s <= 250, 250 >= s), (s <= 50 * w, 50 * w >= s), (2 * s >= 7, 7 <= 2 * s)]:
        point, outputs = {"w": 2.0}, {"s": 150.0}
        assert first.relative_violation(point, outputs) == pytest.approx(second.relative_violation(point, outputs))
        assert first.margin(point, outputs) == pytest.approx(second.margin(point, outputs))


def test_outputs_must_be_a_list_of_names():
    with pytest.raises(TypeError, match=r"outputs=\['stress'\]"):
        Constraint.function(lambda stress: stress - 1, outputs="stress")


def test_a_function_must_take_the_outputs_it_is_passed():
    with pytest.raises(ValidationError, match="doesn't take them"):
        SearchSpace([X, Y], [Constraint.function(lambda x: x - 5, name="limit", outputs=["s"])])


def test_a_function_taking_an_output_without_saying_so_gets_a_hint():
    with pytest.raises(ValidationError, match=r"outputs=\['stress'\]"):
        SearchSpace([X, Y], [Constraint.function(lambda x, stress: stress - x, name="limit")])


def test_a_function_constraint_without_outputs_saves_as_before():
    space = SearchSpace([X, Y], [Constraint.function(lambda x: x - 5, name="limit")])
    assert "outputs" not in space.model_dump_json()


def test_the_reference_point_counts_infeasible_points_so_hypervolume_grows():
    # Only a thin strip is feasible, so the first front is a single point; the reference mustn't sit on it.
    space = SearchSpace([RangeParameter("x", 0, 1), RangeParameter("y", 0, 1)], [Output("g") <= 0.03])

    def problem(x, y):
        return {"f1": x, "f2": 1 - x + y, "g": y}

    config = MultiObjectiveConfig(objectives={"f1": "minimize", "f2": "minimize"}, max_evaluations=300, seed=1)
    result = MultiObjectiveRandomSearch(space, config).optimize(problem)
    volumes = [stats.hypervolume for stats in result.generations if stats.hypervolume is not None]
    assert volumes[-1] > 0.1 and volumes[-1] > volumes[0]


def test_pareto_progress_says_whether_anything_is_feasible():
    seen = []
    config = MultiObjectiveConfig(objectives=OBJECTIVES, population_size=2, generations=1, seed=0)
    MultiObjectiveRandomSearch(FRONT_SPACE, config).optimize(
        lambda x, y: {"a": x, "b": y, "total": 0.0, "cost": 0.0}, callback=seen.append
    )
    assert not seen[0].feasible and len(seen[0].front) == 0


def test_a_checkpoint_from_before_output_constraints_carries_on(tmp_path, monkeypatch):
    from hone.optimizers import core
    from hone.result import GenerationStats

    def make():
        return GeneticOptimizer([X, Y], GeneticOptimizerConfig(max_evaluations=300, seed=0))

    def sphere(x, y):
        return x * x + y * y

    def killed_at_generation_3(progress):
        if progress.generation.generation == 3:
            raise KeyboardInterrupt

    whole = make().optimize(sphere)
    # Save as hone 0.13.0 did: format 1, with none of the fields added since.
    with monkeypatch.context() as patch:
        patch.setattr(core, "CHECKPOINT_FORMAT", 1)
        for cls, added in [(Individual, ("outputs", "violation")), (GenerationStats, ("infeasible",))]:
            original = cls.__getstate__

            def without(self, original=original, added=added):
                state = original(self)
                state["__dict__"] = {k: v for k, v in state["__dict__"].items() if k not in added}
                return state

            patch.setattr(cls, "__getstate__", without)
        make().optimize(sphere, checkpoint=tmp_path / "old.checkpoint", callback=killed_at_generation_3)
    assert pickle.loads((tmp_path / "old.checkpoint").read_bytes())["format"] == 1
    assert make().optimize(sphere, checkpoint=tmp_path / "old.checkpoint") == whole


def test_sensitivity_holds_reduced_parameters_at_a_feasible_point():
    space = SearchSpace([X, Y], [TOTAL <= 4])
    report = sensitivity(lambda x, y: {"objective": -(x + y), "total": x + y}, space, max_evaluations=60, seed=0)
    assert report.best_params["x"] + report.best_params["y"] <= 4
