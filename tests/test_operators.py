"""The genetic optimizer's selection, crossover and mutation functions, on their own."""
import numpy as np
import pytest

from hone.optimizers.genetic import selection
from hone.optimizers.genetic.variation import mutate, uniform_crossover
from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter, SearchSpace

from conftest import individual


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def population(*fitnesses):
    return [individual(f) for f in fitnesses]


# --- selection ------------------------------------------------------------


@pytest.mark.parametrize("maximize, expected", [(True, 9), (False, 1)])
def test_tournament_picks_best_when_p_is_one(rng, maximize, expected):
    picked = selection.tournament(population(1, 5, 9), rng, size=50, p=1.0, maximize=maximize)
    assert picked.fitness == expected


@pytest.mark.parametrize("maximize, expected", [(True, 1), (False, 9)])
def test_tournament_picks_worst_when_p_is_zero(rng, maximize, expected):
    picked = selection.tournament(population(1, 5, 9), rng, size=50, p=0.0, maximize=maximize)
    assert picked.fitness == expected


@pytest.mark.parametrize("maximize, favoured", [(True, 99), (False, 1)])
def test_roulette_favours_fitter_individuals(rng, maximize, favoured):
    picks = [selection.roulette(population(1, 50, 99), rng, maximize=maximize).fitness for _ in range(300)]
    assert picks.count(favoured) > 150


@pytest.mark.parametrize("maximize, never", [(True, -5), (False, 3)])
def test_roulette_never_picks_the_worst(rng, maximize, never):
    picks = {selection.roulette(population(-5, 1, 3), rng, maximize=maximize).fitness for _ in range(100)}
    assert never not in picks


def test_roulette_with_equal_fitness_picks_anyone(rng):
    candidates = population(4, 4, 4)
    picked = {id(selection.roulette(candidates, rng, maximize=True)) for _ in range(50)}
    assert picked == {id(c) for c in candidates}


# --- crossover ------------------------------------------------------------


def test_crossover_takes_each_gene_from_a_parent(rng):
    first, second = {"x": 1, "y": 2}, {"x": 10, "y": 20}
    children = [uniform_crossover(first, second, rng, rate=0.5) for _ in range(50)]
    assert all(c["x"] in (1, 10) and c["y"] in (2, 20) for c in children)
    assert {c["x"] for c in children} == {1, 10}


@pytest.mark.parametrize("rate, expected", [(1.0, {"x": 1, "y": 2}), (0.0, {"x": 10, "y": 20})])
def test_crossover_rate_extremes(rng, rate, expected):
    assert uniform_crossover({"x": 1, "y": 2}, {"x": 10, "y": 20}, rng, rate=rate) == expected


def test_crossover_does_not_change_parents(rng):
    first, second = {"x": 1}, {"x": 10}
    uniform_crossover(first, second, rng, rate=0.5)
    assert first == {"x": 1} and second == {"x": 10}


# --- mutation -------------------------------------------------------------


SPACE = SearchSpace([RangeParameter("x", -5, 5), ChoiceParameter("c", ["a", "b"]), FixedParameter("f", 3)])


def test_mutation_rate_one_changes_every_mutable_gene(rng):
    child = mutate({"x": 1.0, "c": "a", "f": 3}, SPACE, rng, rate=1.0, scale=0.1)
    assert child["x"] != 1.0
    assert child["c"] == "b"
    assert child["f"] == 3


def test_mutation_rate_zero_changes_nothing(rng):
    params = {"x": 1.0, "c": "a", "f": 3}
    assert mutate(params, SPACE, rng, rate=0.0, scale=0.1) == params


def test_mutation_returns_new_parameters(rng):
    params = {"x": 1.0, "c": "a", "f": 3}
    mutate(params, SPACE, rng, rate=1.0, scale=0.1)
    assert params == {"x": 1.0, "c": "a", "f": 3}


def test_mutation_scale_sets_step_size(rng):
    small = [mutate({"x": 0.0, "c": "a", "f": 3}, SPACE, rng, rate=1.0, scale=0.01)["x"] for _ in range(300)]
    large = [mutate({"x": 0.0, "c": "a", "f": 3}, SPACE, rng, rate=1.0, scale=0.2)["x"] for _ in range(300)]
    assert np.std(small) < np.std(large) / 5
