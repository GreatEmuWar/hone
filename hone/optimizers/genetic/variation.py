"""Ways to make a child's parameters from its parents'."""
from typing import Any

import numpy as np

from hone.parameters import RangeParameter, SearchSpace


def uniform_crossover(
    first: dict[str, Any], second: dict[str, Any], rng: np.random.Generator, *, rate: float
) -> dict[str, Any]:
    """Build a child gene by gene, taking each from the first parent with probability rate."""
    return {name: first[name] if rng.uniform(0, 1) < rate else second[name] for name in first}


def blend_crossover(
    first: dict[str, Any], second: dict[str, Any], search_space: SearchSpace, rng: np.random.Generator, *, rate: float, alpha: float
) -> dict[str, Any]:
    """Uniform crossover, then each range gene drawn between its parents' values (BLX-alpha).

    A range value is drawn uniformly from the span between the parents' values, widened by alpha
    times that span on each side, measured on the range's own scale and clipped to it. Parents
    with the same value pass it on. Other genes come from one parent, as in uniform_crossover.
    """
    params = uniform_crossover(first, second, rng, rate=rate)
    for parameter in search_space.parameters:
        if isinstance(parameter, RangeParameter) and parameter.maximum > parameter.minimum:
            a, b = float(parameter.to_unit(first[parameter.name])), float(parameter.to_unit(second[parameter.name]))
            low, high = min(a, b), max(a, b)
            if high > low:
                value = parameter.from_unit(rng.uniform(low - alpha * (high - low), high + alpha * (high - low)))
                params[parameter.name] = int(value) if parameter.integer else float(value)
    return params


def mutate(
    params: dict[str, Any],
    search_space: SearchSpace,
    rng: np.random.Generator,
    *,
    rate: float,
    scale: float,
    only: set[str] | None = None,
) -> dict[str, Any]:
    """Mutate each gene with probability rate, returning new parameters.

    How a gene changes depends on its parameter: ranges are nudged by scale times their width,
    choices switch to another value, and fixed parameters never change. With only, genes not
    in it are left as they are, such as those a condition switches off.
    """
    return {
        name: search_space[name].mutate(value, rng, scale) if (only is None or name in only) and rng.uniform(0, 1) < rate else value
        for name, value in params.items()
    }
