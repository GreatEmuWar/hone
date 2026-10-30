"""Ways to make a child's parameters from its parents'."""
from typing import Any

import numpy as np

from hone.parameters import SearchSpace


def uniform_crossover(
    first: dict[str, Any], second: dict[str, Any], rng: np.random.Generator, *, rate: float
) -> dict[str, Any]:
    """Build a child gene by gene, taking each from the first parent with probability rate."""
    return {name: first[name] if rng.uniform(0, 1) < rate else second[name] for name in first}


def mutate(
    params: dict[str, Any], search_space: SearchSpace, rng: np.random.Generator, *, rate: float, scale: float
) -> dict[str, Any]:
    """Mutate each gene with probability rate, returning new parameters.

    How a gene changes depends on its parameter: ranges are nudged by scale times their width,
    choices switch to another value, and fixed parameters never change.
    """
    return {
        name: search_space[name].mutate(value, rng, scale) if rng.uniform(0, 1) < rate else value
        for name, value in params.items()
    }
