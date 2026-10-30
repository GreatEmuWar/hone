"""Behavior every optimizer shares through the Optimizer base class."""
import pytest

from hone import GeneticOptimizer, OptimizerConfig, RandomSearch
from hone.config import GeneticOptimizerConfig
from hone.optimizers import Individual, Optimizer
from hone.parameters import ChoiceParameter, RangeParameter, SearchSpace

from conftest import bounded_parameters, make_optimizer, sphere


def make_random_search(objective_function=sphere, **config):
    config.setdefault("seed", 0)
    return RandomSearch(objective_function, bounded_parameters(), OptimizerConfig(**config))


OPTIMIZERS = {"genetic": make_optimizer, "random": make_random_search}


@pytest.fixture(params=OPTIMIZERS)
def make(request):
    return OPTIMIZERS[request.param]


# --- construction ---------------------------------------------------------


def test_accepts_search_space_or_list():
    params = [RangeParameter("x", 0, 1), RangeParameter("y", 0, 1)]
    assert make_optimizer(search_space=params).search_space == make_optimizer(search_space=SearchSpace(params)).search_space


def test_config_defaults_when_omitted():
    assert GeneticOptimizer(sphere, bounded_parameters()).config == GeneticOptimizerConfig()
    assert RandomSearch(sphere, bounded_parameters()).config == OptimizerConfig()


def test_rejects_the_wrong_config_class():
    with pytest.raises(TypeError, match="needs a GeneticOptimizerConfig"):
        GeneticOptimizer(sphere, bounded_parameters(), OptimizerConfig())


def test_rejects_function_that_does_not_match_search_space():
    with pytest.raises(ValueError, match="missing from the search space: y"):
        make_optimizer(search_space=[RangeParameter("x", 0, 1)])


def test_rejects_function_that_returns_a_non_number():
    optimizer = make_optimizer(objective_function=lambda x, y: "close", generations=1, population_size=4)
    with pytest.raises(TypeError, match="must return a number"):
        optimizer.optimize()


# --- the generation loop --------------------------------------------------


def test_runs_every_generation_without_a_threshold(make):
    result = make(generations=6, population_size=10).optimize()
    assert [g.generation for g in result.generations] == list(range(6))
    assert result.reached_threshold is False


def test_threshold_stops_early(make):
    result = make(generations=50, population_size=20, threshold=5.0).optimize()
    assert len(result.generations) < 50
    assert result.reached_threshold is True
    assert result.best_fitness <= 5.0


@pytest.mark.parametrize("maximize", [False, True])
def test_reports_the_best_of_every_generation(make, maximize):
    result = make(generations=5, population_size=20, maximize=maximize).optimize()
    bests = [g.best_fitness for g in result.generations]
    assert result.best_fitness == (max(bests) if maximize else min(bests))
    assert sphere(**result.best_params) == result.best_fitness


def test_generation_stats(make):
    result = make(generations=3, population_size=10).optimize()
    for g in result.generations:
        assert g.best_fitness <= g.median <= g.worst_fitness
        assert g.best_fitness <= g.mean <= g.worst_fitness
        assert g.std >= 0
        assert sphere(**g.best_params) == g.best_fitness


def test_same_seed_gives_same_result(make):
    assert make(generations=4, population_size=10, seed=7).optimize() == make(generations=4, population_size=10, seed=7).optimize()


def test_optimize_twice_gives_the_same_result(make):
    optimizer = make(generations=3, population_size=10)
    assert optimizer.optimize() == optimizer.optimize()


def test_different_seeds_give_different_results(make):
    assert make(generations=2, population_size=10, seed=1).optimize() != make(generations=2, population_size=10, seed=2).optimize()


def test_counts_evaluations(make):
    calls = []

    def counted(x, y):
        calls.append(1)
        return x + y

    result = make(objective_function=counted, generations=3, population_size=10).optimize()
    assert result.evaluations == len(calls) > 0


def test_values_stay_within_the_search_space():
    space = [RangeParameter("x", -5, 5, integer=True), ChoiceParameter("y", [-1, 0.5, 2])]
    result = make_optimizer(search_space=space, generations=4, population_size=10).optimize()
    for g in result.generations:
        assert type(g.best_params["x"]) is int and -5 <= g.best_params["x"] <= 5
        assert g.best_params["y"] in (-1, 0.5, 2)


# --- writing a new optimizer ----------------------------------------------


class GridWalk(Optimizer):
    """A deliberately simple optimizer: every individual sits on the diagonal x == y."""

    def _next_population(self, previous, generation):
        step = 1 / (generation + 1)
        return [
            Individual(params={"x": step * i, "y": step * i}, generation=generation)
            for i in range(self.config.population_size)
        ]


def test_a_subclass_only_needs_next_population():
    result = GridWalk(sphere, bounded_parameters(), OptimizerConfig(generations=3, population_size=4)).optimize()
    assert result.best_params == {"x": 0.0, "y": 0.0}
    assert result.evaluations == 12


def test_optimizer_cannot_be_used_directly():
    with pytest.raises(TypeError):
        Optimizer(sphere, bounded_parameters())


def test_genetic_beats_random_search_for_the_same_evaluations():
    """A sanity check that the genetic operators do better than chance."""
    genetic = [make_optimizer(seed=s).optimize().best_fitness for s in range(5)]
    random = [make_random_search(seed=s).optimize().best_fitness for s in range(5)]
    assert sorted(genetic)[2] < sorted(random)[2]
