"""Starting from known points: checking them against the search space, and how each optimizer uses them."""
import math

import pytest

from hone import CMAES, CMAESConfig, GeneticOptimizer, GeneticOptimizerConfig, RandomSearch
from hone.config import OptimizerConfig
from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter, SearchSpace

SPACE = SearchSpace(
    [
        RangeParameter("x", -5, 5),
        RangeParameter("n", 0, 10, integer=True),
        ChoiceParameter("material", ["steel", "aluminium"]),
        ChoiceParameter("tags", ["a", "b", "c"], choose_n=2),
        FixedParameter("mode", "fast"),
    ]
)
POINT = {"x": 1.5, "n": 3, "material": "steel", "tags": ["a", "c"]}
CONTINUOUS = [RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)]


def shifted_sphere(x, y):
    return (x - 4) ** 2 + (y - 4) ** 2


# --- checking a point -----------------------------------------------------


def test_check_completes_and_orders_a_point():
    checked = SPACE.check({"tags": ("c", "a"), "n": 3.0, "x": 2, "material": "steel"})
    assert SPACE.check({**POINT, "mode": "fast"}) == SPACE.check(POINT)
    assert checked == {"x": 2.0, "n": 3, "material": "steel", "tags": ["c", "a"], "mode": "fast"}
    assert list(checked) == SPACE.names
    assert type(checked["x"]) is float and type(checked["n"]) is int


@pytest.mark.parametrize(
    "change, error",
    [
        ({"size": 1}, "unknown parameters: size"),
        ({"x": None}, "x: None is not a number"),
        ({"x": True}, "x: True is not a number"),
        ({"x": 5.5}, "x: 5.5 is outside"),
        ({"n": 2.5}, "n: 2.5 is not an integer"),
        ({"material": "wood"}, "material: 'wood' is not one of"),
        ({"tags": ["a"]}, "tags: expected a list of 2 choices"),
        ({"tags": "ab"}, "tags: expected a list of 2 choices"),
        ({"tags": ["a", "a"]}, "must be 2 different values"),
        ({"tags": ["a", "z"]}, "must be 2 different values"),
        ({"mode": "slow"}, "mode: 'slow' is not its fixed value 'fast'"),
    ],
)
def test_check_rejects_values_outside_the_space(change, error):
    with pytest.raises(ValueError, match=error):
        SPACE.check({**POINT, **change})


def test_check_needs_every_parameter_but_the_fixed_ones():
    with pytest.raises(ValueError, match="missing values for: x, material"):
        SPACE.check({"n": 3, "tags": ["a", "b"]})


def test_optimizers_check_initial_points():
    with pytest.raises(ValueError, match="outside"):
        GeneticOptimizer(SPACE, initial_points={**POINT, "x": 9})


# --- the genetic optimizer and random search ------------------------------


@pytest.mark.parametrize("make", [GeneticOptimizer, RandomSearch], ids=["genetic", "random"])
def test_initial_points_lead_the_first_generation(make):
    second = {**POINT, "x": -2.0, "material": "aluminium"}
    optimizer = make(SPACE, initial_points=[POINT, second])
    trials = optimizer.ask()
    assert [t.params for t in trials[:2]] == [SPACE.check(POINT), SPACE.check(second)]
    assert len(trials) == optimizer.config.population_size


def test_a_single_point_can_be_passed_as_a_dict():
    assert GeneticOptimizer(SPACE, initial_points=POINT).initial_points == [SPACE.check(POINT)]


def test_more_initial_points_than_the_population_are_all_evaluated():
    points = [{"x": float(x), "y": 0.0} for x in range(-5, 6)]
    config = GeneticOptimizerConfig(population_size=4, tournament_size=2, generations=2, seed=0)
    optimizer = GeneticOptimizer(CONTINUOUS, config, initial_points=points)
    assert [t.params for t in optimizer.ask()] == points


@pytest.mark.parametrize(
    "make",
    [
        lambda points: GeneticOptimizer(CONTINUOUS, GeneticOptimizerConfig(max_evaluations=200, seed=0), initial_points=points),
        lambda points: RandomSearch(CONTINUOUS, OptimizerConfig(max_evaluations=200, seed=0), initial_points=points),
        lambda points: CMAES(CONTINUOUS, CMAESConfig(max_evaluations=200, seed=0), initial_points=points),
    ],
    ids=["genetic", "random", "cma-es"],
)
def test_a_known_optimum_is_never_lost(make):
    result = make({"x": 4.0, "y": 4.0}).optimize(shifted_sphere)
    assert result.best_fitness == 0 and result.best_params == {"x": 4.0, "y": 4.0}


def test_runs_with_initial_points_are_reproducible():
    def make():
        return GeneticOptimizer(CONTINUOUS, GeneticOptimizerConfig(max_evaluations=200, seed=0), initial_points={"x": 1.0, "y": 1.0})

    optimizer = make()
    first = optimizer.optimize(shifted_sphere)
    assert optimizer.optimize(shifted_sphere) == first == make().optimize(shifted_sphere)


def test_without_initial_points_nothing_changes():
    config = GeneticOptimizerConfig(max_evaluations=200, seed=0)
    assert GeneticOptimizer(CONTINUOUS, config).optimize(shifted_sphere) == GeneticOptimizer(
        CONTINUOUS, config, initial_points=[]
    ).optimize(shifted_sphere)


# --- CMA-ES ---------------------------------------------------------------


def cmaes(points, **config):
    return CMAES(CONTINUOUS, CMAESConfig(seed=0, **config), initial_points=points)


def tell_all(optimizer, trials, objective=shifted_sphere):
    for trial in trials:
        optimizer.tell(trial, objective(**trial.params))


def test_cmaes_evaluates_the_initial_points_as_its_first_generation():
    points = [{"x": 4.0, "y": 3.0}, {"x": -3.0, "y": 2.0}]
    optimizer = cmaes(points)
    trials = optimizer.ask()
    assert [t.params for t in trials] == points


def test_cmaes_searches_around_the_best_initial_point():
    optimizer = cmaes([{"x": -3.0, "y": 2.0}, {"x": 3.5, "y": 3.0}], initial_step=0.01)
    tell_all(optimizer, optimizer.ask())
    samples = optimizer.ask()
    assert all(abs(t.params["x"] - 3.5) < 0.5 and abs(t.params["y"] - 3.0) < 0.5 for t in samples)


def test_cmaes_starts_at_the_centre_when_every_initial_point_fails():
    optimizer = cmaes([{"x": 4.0, "y": 4.0}], initial_step=0.01)
    optimizer.tell(optimizer.ask()[0], failed=True)
    assert all(abs(t.params["x"]) < 0.5 and abs(t.params["y"]) < 0.5 for t in optimizer.ask())


def test_cmaes_starting_near_the_optimum_saves_evaluations():
    def error(points):
        return cmaes(points, max_evaluations=60, initial_step=0.05, restarts=False).optimize(shifted_sphere).best_fitness

    assert error({"x": 3.9, "y": 4.1}) < 1e-3 < error(None)


def test_cmaes_integer_initial_points_round_trip():
    space = [RangeParameter("x", -5, 5), RangeParameter("n", -5, 5, integer=True)]
    optimizer = CMAES(space, CMAESConfig(seed=0, initial_step=0.01), initial_points={"x": 1.0, "n": 3})
    tell_all(optimizer, optimizer.ask(), lambda x, n: (x - 1) ** 2 + (n - 3) ** 2)
    assert {t.params["n"] for t in optimizer.ask()} <= {2, 3, 4}


# --- infinity counts as failed ---------------------------------------------


@pytest.mark.parametrize("value", [math.inf, -math.inf])
@pytest.mark.parametrize("maximize", [False, True])
def test_infinite_values_count_as_failed(value, maximize):
    config = OptimizerConfig(population_size=2, generations=1, maximize=maximize, seed=0)
    optimizer = RandomSearch(CONTINUOUS, config)
    first, second = optimizer.ask()
    optimizer.tell(first, value)
    optimizer.tell(second, 4.0)
    result = optimizer.result()
    assert result.failures == 1 and result.best_fitness == 4.0


def test_an_objective_returning_infinity_is_never_best():
    def overflowing(x, y):
        return -math.inf if x > 0 else x**2 + y**2

    result = RandomSearch(CONTINUOUS, OptimizerConfig(population_size=20, generations=3, seed=0)).optimize(overflowing)
    assert result.failures > 0 and math.isfinite(result.best_fitness) and result.best_params["x"] <= 0


def test_tell_rejects_booleans():
    optimizer = RandomSearch(CONTINUOUS, OptimizerConfig(population_size=2, generations=1))
    with pytest.raises(TypeError, match="must be a number"):
        optimizer.tell(optimizer.ask(1)[0], True)
