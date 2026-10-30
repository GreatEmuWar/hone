"""Range parameters on a log scale, and every optimizer searching them that way."""
import math

import numpy as np
import pytest
from pydantic import ValidationError

from hone import BayesianConfig, BayesianOptimizer, CMAES, CMAESConfig, GeneticOptimizer, GeneticOptimizerConfig, RandomSearch
from hone.config import OptimizerConfig
from hone.parameters import RangeParameter

RATE = RangeParameter("rate", 1e-4, 1.0, log=True)
TREES = RangeParameter("trees", 10, 1000, integer=True, log=True)


def test_log_range_needs_a_positive_minimum():
    with pytest.raises(ValidationError, match="positive minimum"):
        RangeParameter("rate", 0, 1, log=True)


@pytest.mark.parametrize("parameter", [RATE, TREES, RangeParameter("x", -5, 5), RangeParameter("n", 0, 10, integer=True)])
def test_unit_mapping_round_trips(parameter):
    for unit in (0.0, 0.25, 0.5, 1.0):
        value = parameter.from_unit(unit)
        assert parameter.minimum <= value <= parameter.maximum
        assert parameter.from_unit(parameter.to_unit(value)) == value
    assert type(parameter.from_unit(0.3)) is (int if parameter.integer else float)


def test_log_unit_mapping_spaces_orders_of_magnitude_evenly():
    assert [RATE.from_unit(u) for u in (0, 0.25, 0.5, 0.75, 1)] == pytest.approx([1e-4, 1e-3, 1e-2, 1e-1, 1])
    assert TREES.from_unit(0.5) == 100


def test_unit_mapping_works_on_arrays():
    assert np.allclose(RATE.from_unit(np.array([0.0, 0.5])), [1e-4, 1e-2])
    assert np.allclose(RATE.to_unit(np.array([1e-4, 1e-2])), [0.0, 0.5])


def test_degenerate_range_maps_to_its_value():
    same = RangeParameter("same", 3, 3, log=True)
    assert same.to_unit(3) == 0.0 and same.from_unit(0.7) == 3.0


def test_samples_are_even_in_the_logarithm():
    rng = np.random.default_rng(0)
    exponents = np.log10([RATE.sample(rng) for _ in range(4000)])
    assert np.histogram(exponents, bins=4, range=(-4, 0))[0].min() > 900, "each decade gets about a quarter"


def test_integer_log_samples_are_integers_in_range():
    rng = np.random.default_rng(0)
    values = [TREES.sample(rng) for _ in range(500)]
    assert all(type(v) is int and 10 <= v <= 1000 for v in values)
    assert sum(v < 100 for v in values) > 200, "half the draws fall in the lower decade"


def test_log_mutation_is_proportional_to_the_value():
    rng = np.random.default_rng(0)
    small = [RATE.mutate(1e-3, rng, 0.05) for _ in range(500)]
    large = [RATE.mutate(1e-1, rng, 0.05) for _ in range(500)]
    assert np.std(np.log(small)) == pytest.approx(np.std(np.log(large)), rel=0.2)
    assert all(RATE.minimum <= v <= RATE.maximum for v in small + large)


def test_log_integer_mutation_always_moves():
    rng = np.random.default_rng(0)
    assert all(TREES.mutate(500, rng, 1e-6) != 500 for _ in range(50))
    assert all(TREES.mutate(10, rng, 1e-6) in (10, 11) for _ in range(50)), "clipped at the minimum"


def objective(rate, trees):
    """Best at rate 0.003 and 150 trees, a valley that is narrow in absolute terms."""
    return (math.log10(rate) - math.log10(3e-3)) ** 2 + (math.log10(trees) - math.log10(150)) ** 2


@pytest.mark.parametrize(
    "make",
    [
        lambda: CMAES([RATE, TREES], CMAESConfig(max_evaluations=300, seed=0)),
        lambda: BayesianOptimizer([RATE, TREES], BayesianConfig(max_evaluations=40, seed=0)),
        lambda: GeneticOptimizer([RATE, TREES], GeneticOptimizerConfig(max_evaluations=600, seed=0)),
    ],
    ids=["cma-es", "bayesian", "genetic"],
)
def test_optimizers_find_an_optimum_near_the_bottom_of_a_log_range(make):
    result = make().optimize(objective)
    assert result.best_params["rate"] == pytest.approx(3e-3, rel=0.25)
    assert 120 <= result.best_params["trees"] <= 190


def test_log_scale_beats_a_linear_scale_for_small_values():
    """Uniform sampling of [0.0001, 1] almost never tries values below 0.01."""
    linear = RandomSearch([RangeParameter("rate", 1e-4, 1.0), TREES], OptimizerConfig(population_size=20, max_evaluations=60, seed=0))
    logged = RandomSearch([RATE, TREES], OptimizerConfig(population_size=20, max_evaluations=60, seed=0))
    assert logged.optimize(objective).best_fitness < linear.optimize(objective).best_fitness
