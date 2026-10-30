import numpy as np
import pytest

from hone.config import GeneticOptimizerConfig
from hone.constraint import Constraint
from hone.optimizers import GeneticOptimizer, Individual


@pytest.fixture(autouse=True)
def seeded_rng():
    """The optimizer draws from numpy's global RNG, so seed it for repeatable tests."""
    np.random.seed(0)


def sphere(x, y):
    return x**2 + y**2


def bounded_constraints(low=-5, high=5):
    return [
        Constraint(param_name="x", minimum=low, maximum=high),
        Constraint(param_name="y", minimum=low, maximum=high),
    ]


def make_optimizer(constraints=None, seed_parameters=None, **config):
    return GeneticOptimizer(
        objective_function=sphere,
        seed_parameters=seed_parameters if seed_parameters is not None else {"x": 1.0, "y": 1.0},
        constraints=constraints if constraints is not None else bounded_constraints(),
        config=GeneticOptimizerConfig(**config),
    )


def individual(fitness, **params):
    ind = Individual(**params)
    ind.fitness = fitness
    return ind
