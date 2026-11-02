import pytest

from hone.config import GeneticOptimizerConfig
from hone.optimizers import GeneticOptimizer, Individual
from hone.parameters import RangeParameter


def sphere(x, y):
    return x**2 + y**2


def bounded_parameters(low=-5, high=5):
    return [RangeParameter("x", low, high), RangeParameter("y", low, high)]


def make_optimizer(search_space=None, **config):
    config.setdefault("seed", 0)
    return GeneticOptimizer(
        search_space=search_space if search_space is not None else bounded_parameters(),
        config=GeneticOptimizerConfig(**config),
    )


def run(objective_function=sphere, search_space=None, **config):
    """Build a genetic optimizer and run it on objective_function."""
    return make_optimizer(search_space, **config).optimize(objective_function)


def individual(fitness, generation=0, **params):
    return Individual(params=params or {"x": fitness}, generation=generation, fitness=fitness)


def pytest_collection_modifyitems(config, items):
    """On a pytest-xdist worker, skip the tests that measure wall-clock time: beside every other test
    on a loaded machine their timings don't hold. They run on their own with -n 0 -m timing."""
    if getattr(config, "workerinput", None) is None:
        return
    skip = pytest.mark.skip(reason="measures wall-clock time: run with pytest -n 0 -m timing")
    for item in items:
        if "timing" in item.keywords:
            item.add_marker(skip)
