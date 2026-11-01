"""NSGA-II: the genetic optimizer for several objectives (Deb, Pratap, Agarwal and Meyarivan, 2002)."""
import numpy as np

from hone.config.multi_objective_config import NSGA2Config
from hone.optimizers.core import Individual
from hone.optimizers.multi import MultiObjectiveOptimizer
from hone.pareto import _minimizing, non_dominated

from .breeding import Breeding


class NSGA2(MultiObjectiveOptimizer, Breeding):
    """The genetic optimizer for several objectives, returning the front of best trade-offs.

    Each generation, the previous generation and its children are sorted into fronts: the
    points nothing dominates, then those only the first front dominates, and so on. The best
    fronts survive, and where one doesn't fit whole, its most isolated points do, by crowding
    distance, so the front stays spread out. Parents are chosen from the survivors by
    tournaments of two, won by the better front and then the more isolated point. Children are
    bred as in the genetic optimizer, with its crossover, mutation, repeat avoidance, repair
    for constraints, conditions and niches.

    The result's front is that of every point evaluated, not only the survivors, so a
    trade-off found early is never lost when crowding drops it from the population.
    Ties are broken by the order points were created, so a seeded run repeats exactly, and
    its result doesn't depend on the number of workers.
    """
    config_class = NSGA2Config
    supports_constraints = True
    supports_conditions = True
    config: NSGA2Config

    def reset(self) -> None:
        super().reset()
        self._setup_breeding()
        self._standing: dict[Individual, tuple[int, float]] = {}  # each survivor's front, and its crowding distance

    # --- selection ---------------------------------------------------------

    def _order(self, individual: Individual) -> tuple[int, float, int]:
        rank, crowding = self._standing.get(individual, (len(self._standing), 0.0))
        return rank, -crowding, individual.id

    def _select(self, individuals: list[Individual]) -> Individual:
        first, second = (individuals[i] for i in self.rng.integers(len(individuals), size=2))
        return min(first, second, key=self._order)

    def _fitter(self, first: Individual, second: Individual) -> Individual:
        return min(first, second, key=self._order)

    def _ranked_niches(self, niches: list[list[Individual]]) -> list[list[Individual]]:
        return sorted(niches, key=lambda members: min(self._order(m) for m in members))

    # --- survival ----------------------------------------------------------

    def _survivors(self, pool: list[Individual], size: int) -> list[Individual]:
        """The best size individuals of pool, by front and then crowding distance, with their standing recorded."""
        assert all(ind.values is not None for ind in pool)
        vectors = np.array([_minimizing(ind.values, self._objectives) for ind in pool])  # type: ignore[arg-type]
        remaining = list(range(len(pool)))
        survivors: list[Individual] = []
        self._standing = {}
        rank = 0
        while remaining and len(survivors) < size:
            front = [remaining[i] for i in non_dominated(vectors[remaining])]
            crowding = _crowding(vectors[front])
            for i, distance in zip(front, crowding):
                self._standing[pool[i]] = (rank, float(distance))
            front.sort(key=lambda i: self._order(pool[i]))
            survivors += [pool[i] for i in front[: size - len(survivors)]]
            chosen = set(front)
            remaining = [i for i in remaining if i not in chosen]
            rank += 1
        return survivors

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        size = self.config.population_size
        if previous is None:
            initial = self._initial_individuals(generation)
            population = initial + [self._random_individual(generation) for _ in range(size - len(initial))]
            self._asked.update(self._key(individual.params) for individual in population)
            return population
        pool = [individual for individual in previous if not individual.failed]
        if not pool:
            # Every evaluation so far failed: start afresh.
            population = [self._random_individual(generation) for _ in range(size)]
            self._asked.update(self._key(individual.params) for individual in population)
            return population
        if self.config.noise:
            # A point evaluated again since has a new average; rank by it.
            for individual in pool:
                individual.values = self._values(self._observed[self._key(individual.params)].mean)
        survivors = self._survivors(pool, size)
        niched = self._niche_children(survivors, int(self._niche_share * size), generation)
        children = niched + [self._child(survivors, generation) for _ in range(size - len(niched))]
        self._keep_steps(set(survivors) | set(children) | set(self._outstanding()))
        return survivors + children


def _crowding(vectors: np.ndarray) -> np.ndarray:
    """Each point's crowding distance within its front: the sum over objectives of the gap between its neighbours, scaled by the front's range.

    The points at either end of any objective get infinity, so the front's extremes always survive.
    """
    n, m = vectors.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(vectors[:, j], kind="stable")
        low, high = vectors[order[0], j], vectors[order[-1], j]
        distance[order[0]] = distance[order[-1]] = np.inf
        if high > low:
            distance[order[1:-1]] += (vectors[order[2:], j] - vectors[order[:-2], j]) / (high - low)
    return distance
