"""Ways to pick a parent from an evaluated population.

Both follow Deb's feasibility rule when there are constraints on outputs: a point that satisfies
them beats one that doesn't, and between two that don't, the smaller violation wins.
"""
import numpy as np

from hone.optimizers.base import Individual


def _rank(individual: Individual, maximize: bool) -> tuple[float, float]:
    """A sort key putting the fittest first: smaller violation, then better fitness."""
    assert individual.fitness is not None
    return individual.violation, -individual.fitness if maximize else individual.fitness


def tournament(
    individuals: list[Individual], rng: np.random.Generator, *, size: int, p: float, maximize: bool
) -> Individual:
    """Draw size individuals, with replacement, and return the fittest with probability p, else the least fit."""
    indices = rng.choice(len(individuals), size=size)
    contenders = sorted((individuals[i] for i in indices), key=lambda x: _rank(x, maximize))
    return contenders[0] if rng.uniform(0, 1) < p else contenders[-1]


def roulette(individuals: list[Individual], rng: np.random.Generator, *, maximize: bool) -> Individual:
    """Pick an individual with probability proportional to its fitness.

    Fitnesses are shifted so the worst individual has weight zero, which makes the
    selection work for negative fitnesses and for minimization alike. Points that break an
    output constraint get weight zero while any point satisfies them all, since the worst
    feasible point already does; when none does, weights fall with the violation instead, so
    the point closest to satisfying them is likeliest.
    """
    violations = np.array([individual.violation for individual in individuals], dtype=float)
    feasible = violations == 0
    if feasible.any():
        fitnesses = np.array([individual.fitness for individual in individuals], dtype=float)
        shown = fitnesses[feasible]
        weights = np.where(feasible, fitnesses - shown.min() if maximize else shown.max() - fitnesses, 0.0)
        tied = feasible  # with every feasible point equally fit, pick among them
    else:
        finite = np.isfinite(violations)
        worst = violations[finite].max() if finite.any() else 0.0
        weights = np.where(finite, worst - violations, 0.0)
        tied = finite if finite.any() else np.ones(len(individuals), dtype=bool)
    total = weights.sum()
    if total == 0:
        candidates = np.flatnonzero(tied)
        return individuals[int(candidates[rng.integers(len(candidates))])]
    return individuals[rng.choice(len(individuals), p=weights / total)]
