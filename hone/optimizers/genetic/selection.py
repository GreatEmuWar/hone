"""Ways to pick a parent from an evaluated population."""
import numpy as np

from hone.optimizers.base import Individual


def tournament(
    individuals: list[Individual], rng: np.random.Generator, *, size: int, p: float, maximize: bool
) -> Individual:
    """Draw size individuals, with replacement, and return the fittest with probability p, else the least fit."""
    indices = rng.choice(len(individuals), size=size)
    contenders = sorted((individuals[i] for i in indices), key=lambda x: x.fitness, reverse=maximize)
    return contenders[0] if rng.uniform(0, 1) < p else contenders[-1]


def roulette(individuals: list[Individual], rng: np.random.Generator, *, maximize: bool) -> Individual:
    """Pick an individual with probability proportional to its fitness.

    Fitnesses are shifted so the worst individual has weight zero, which makes the
    selection work for negative fitnesses and for minimization alike.
    """
    fitnesses = np.array([individual.fitness for individual in individuals], dtype=float)
    weights = fitnesses - fitnesses.min() if maximize else fitnesses.max() - fitnesses
    total = weights.sum()
    if total == 0:
        # Every individual is equally fit.
        return individuals[rng.integers(len(individuals))]
    return individuals[rng.choice(len(individuals), p=weights / total)]
