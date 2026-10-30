"""CMA-ES: what it accepts, how it samples, and that it converges where it should."""
import numpy as np
import pytest

from hone import CMAES, CMAESConfig, UnsupportedSearchSpace
from hone.benchmarks import standard_problems
from hone.optimizers import cmaes
from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter

from conftest import bounded_parameters, sphere

PROBLEMS = {problem.name: problem for problem in standard_problems()}


def make(search_space=None, **config):
    config.setdefault("seed", 0)
    return CMAES(search_space if search_space is not None else bounded_parameters(), CMAESConfig(**config))


def solve(name, seed=0, per_dimension=1000, **config):
    problem = PROBLEMS[name]
    budget = per_dimension * problem.dimensions
    return make(problem.search_space, max_evaluations=budget, seed=seed, **config).optimize(problem.function)


# --- what it accepts ------------------------------------------------------


def test_rejects_choice_parameters():
    with pytest.raises(UnsupportedSearchSpace, match="use GeneticOptimizer for choice parameters \\(c\\)"):
        make([RangeParameter("x", 0, 1), ChoiceParameter("c", ["a", "b"])])


def test_rejects_a_space_with_nothing_to_search():
    with pytest.raises(UnsupportedSearchSpace, match="at least one range parameter"):
        make([FixedParameter("x", 1.0), RangeParameter("y", 2, 2)])


def test_unsupported_search_space_is_a_value_error():
    assert issubclass(UnsupportedSearchSpace, ValueError)


def test_fixed_and_degenerate_parameters_pass_through():
    space = [RangeParameter("x", -5, 5), FixedParameter("mode", "fast"), RangeParameter("y", 2, 2), RangeParameter("n", 3, 3, integer=True)]
    trial = make(space).ask(1)[0]
    assert trial.params["mode"] == "fast" and trial.params["y"] == 2.0 and trial.params["n"] == 3
    assert list(trial.params) == ["x", "mode", "y", "n"]


def test_default_population_size_grows_with_dimensions():
    assert make([RangeParameter(f"x{i}", 0, 1) for i in range(2)]).population_size == 6
    assert make([RangeParameter(f"x{i}", 0, 1) for i in range(10)]).population_size == 10
    assert make(population_size=20).population_size == 20


# --- sampling -------------------------------------------------------------


def test_samples_stay_within_bounds_and_types():
    space = [RangeParameter("x", -1, 1), RangeParameter("n", 0, 3, integer=True)]
    optimizer = make(space, initial_step=1.0, population_size=200)
    trials = optimizer.ask()
    assert all(-1 <= t.params["x"] <= 1 for t in trials)
    assert all(type(t.params["n"]) is int and 0 <= t.params["n"] <= 3 for t in trials)
    assert {t.params["n"] for t in trials} == {0, 1, 2, 3}


def test_mirror_reflects_into_the_unit_interval():
    assert np.allclose(cmaes._mirror(np.array([-0.25, 0.5, 1.25, 2.5, -1.75])), [0.25, 0.5, 0.75, 0.5, 0.25])


def test_integer_variables_keep_moving_after_the_continuous_ones_converge():
    """Without the minimum spread, the search would lock onto one integer value."""
    space = [RangeParameter("x", -5, 5), RangeParameter("n", -5, 5, integer=True)]
    optimizer = make(space, max_evaluations=600, restarts=False)
    optimizer.optimize(lambda x, n: x**2 + 0.001 * n**2)
    late = [optimizer._sample(0).params for _ in range(200)]
    assert np.std([p["x"] for p in late]) < 1e-3, "the continuous variable has converged"
    assert len({p["n"] for p in late}) > 1, "the integer variable still tries neighbouring values"


# --- the run --------------------------------------------------------------


def test_uses_exactly_the_evaluation_budget():
    result = make(max_evaluations=1234).optimize(sphere)
    assert result.evaluations == 1234
    assert result.generations[-1].evaluations == 1234


def test_is_reproducible():
    assert make(max_evaluations=300, seed=3).optimize(sphere) == make(max_evaluations=300, seed=3).optimize(sphere)


def test_maximizes():
    result = make(max_evaluations=600, maximize=True).optimize(lambda x, y: -((x - 1) ** 2) - (y + 2) ** 2)
    assert result.best_fitness == pytest.approx(0, abs=1e-8)
    assert result.best_params["x"] == pytest.approx(1, abs=1e-4)


def test_ignores_failed_evaluations():
    def unstable(x, y):
        return float("nan") if x > 2 else (x - 1) ** 2 + y**2

    result = make(max_evaluations=1000).optimize(unstable)
    assert result.failures > 0
    assert result.best_fitness < 1e-8


def test_restarts_double_the_population():
    optimizer = make(max_evaluations=4000)
    optimizer.optimize(sphere)
    assert optimizer.population_size > 6, "sphere converges fast, so the run should have restarted"
    assert optimizer.population_size in {6 * 2**k for k in range(1, 10)}


def test_no_restarts_keeps_the_population():
    optimizer = make(max_evaluations=4000, restarts=False)
    optimizer.optimize(sphere)
    assert optimizer.population_size == 6


def test_reset_restores_the_starting_population():
    optimizer = make(max_evaluations=4000)
    first = optimizer.optimize(sphere)
    assert optimizer.population_size > 6
    assert optimizer.optimize(sphere) == first


@pytest.mark.parametrize("name", ["sphere-10d", "ellipsoid-10d", "rosenbrock-10d", "integer-sphere-10d"])
def test_solves_problems_the_genetic_optimizer_cannot(name):
    """The benchmark's 10-dimensional unimodal problems, solved to 1e-8 on every seed tried."""
    for seed in range(3):
        assert solve(name, seed=seed).best_fitness < 1e-8


def test_ellipsoid_is_solved_with_far_fewer_evaluations_than_the_genetic_optimizer_gets():
    assert solve("ellipsoid-2d", per_dimension=500).best_fitness < 1e-8


@pytest.mark.parametrize(
    "state, restarts",
    [
        ({}, False),
        ({"_sigma": 1e-14}, True),
        ({"_D": np.array([1.0, 1e-8])}, True),
    ],
    ids=["fresh", "step-collapsed", "badly-conditioned"],
)
def test_restart_conditions(state, restarts):
    optimizer = make()
    for name, value in state.items():
        setattr(optimizer, name, value)
    assert optimizer._should_restart() is restarts


def test_stalled_runs_restart():
    optimizer = make()
    optimizer._recent_bests = [1.0] * 30
    assert optimizer._should_restart()
