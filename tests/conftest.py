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
