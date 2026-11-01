"""Optimizers of several objectives at once, which return the front of best trade-offs rather than one best point."""
import math
import numbers
from typing import Any, Callable

import numpy as np

from hone.config.multi_objective_config import MultiObjectiveConfig
from hone.objective_function.objective_function import MultiObjectiveFunction
from hone.pareto import (
    Front,
    ParetoGenerationStats,
    ParetoPoint,
    ParetoProgress,
    ParetoResult,
    _minimizing,
    _joining,
    hypervolume,
    non_dominated,
)

from .core import Individual, OptimizerCore

REFERENCE_MARGIN = 0.1  # the default reference point lies this share of each objective's range past its worst value


class _Observed:
    """Every evaluation of one point: the sums for its averages, and where it was first seen."""

    def __init__(self, params: dict[str, Any], first: int) -> None:
        self.params = params
        self.first = first
        self.count = 0
        self.total: np.ndarray | None = None
        self.squares: np.ndarray | None = None

    def add(self, vector: np.ndarray) -> None:
        self.count += 1
        self.total = vector.copy() if self.total is None else self.total + vector
        self.squares = vector**2 if self.squares is None else self.squares + vector**2

    @property
    def mean(self) -> np.ndarray:
        assert self.total is not None
        return self.total / self.count

    @property
    def standard_error(self) -> np.ndarray | None:
        if self.count < 2:
            return None
        assert self.squares is not None
        variance = np.maximum(self.squares / self.count - self.mean**2, 0.0) * self.count / (self.count - 1)
        return np.sqrt(variance / self.count)


class MultiObjectiveOptimizer(OptimizerCore[ParetoResult]):
    """Base class for optimizers of several objectives, which return a ParetoResult.

    The objective function returns a dict with one value per objective named in the config's
    objectives, such as {"accuracy": 0.97, "cost": 12.0}, and tell() takes the same. A dict with
    missing or extra keys, or a value that isn't a number, raises TypeError: it is a bug in the
    objective. Raise EvaluationFailed, or return NaN or infinity for any objective, to record an
    evaluation as failed; failed evaluations are never on the front.

    The front is that of every point evaluated, not only the latest generation, so a trade-off
    found early is never lost. With noise, each point's values are the averages of all its
    evaluations, and the front comes with their standard errors.
    """
    config_class = MultiObjectiveConfig
    config: MultiObjectiveConfig

    def reset(self) -> None:
        """Start over: forget every evaluation and reseed the random number generator."""
        super().reset()
        self._objectives = dict(self.config.objectives)
        self._observed: dict[tuple, _Observed] = {}  # every point evaluated successfully, in the order first seen
        self._means: list[np.ndarray] = []  # each observed point's values to minimize, in the same order
        self._front: list[int] = []  # the observed points on the front, by position in that order
        self._front_stale = False  # with noise, an average changed, so the front must be worked out again
        self._reference: np.ndarray | None = None if self.config.reference is None else _minimizing(self.config.reference, self._objectives)

    # --- what a value is ---------------------------------------------------

    def _check_outcome(self, value: Any, failed: bool) -> bool:
        if failed:
            return True
        if value is None:
            raise ValueError("pass the trial's values, or failed=True")
        if not isinstance(value, dict):
            raise TypeError(f"the value must be a dict with a number for each objective ({', '.join(self._objectives)}), got {value!r}")
        missing, extra = sorted(set(self._objectives) - set(value)), sorted(set(value) - set(self._objectives))
        if missing or extra:
            problems = [f"missing {', '.join(missing)}"] * bool(missing) + [f"unknown {', '.join(extra)}"] * bool(extra)
            raise TypeError(f"the values must have exactly the objectives {', '.join(self._objectives)}: {'; '.join(problems)}")
        for name, number in value.items():
            if isinstance(number, bool) or not isinstance(number, numbers.Real):
                raise TypeError(f"the value of {name} must be a number, got {number!r}")
        return not all(math.isfinite(number) for number in value.values())

    def _record_failure(self, individual: Individual) -> None:
        individual.values = None

    def _record(self, individual: Individual, value: Any) -> None:
        values = {name: float(value[name]) for name in self._objectives}
        key = self._key(individual.params)
        observed = self._observed.get(key)
        new = observed is None
        if observed is None:
            observed = self._observed[key] = _Observed(self.search_space.active(individual.params), individual.id)
        observed.add(_minimizing(values, self._objectives))
        if self.config.noise:
            values = self._values(observed.mean)
            if new:
                self._means.append(observed.mean)
            self._front_stale = True
        elif new:
            # Without noise a point's values never change, so the front only needs this point added.
            self._means.append(observed.mean)
            joins, stays = _joining(np.array([self._means[j] for j in self._front]), observed.mean)
            if joins:
                self._front = [j for j, kept in zip(self._front, stays) if kept] + [len(self._means) - 1]
        individual.values = values

    def _has_result(self) -> bool:
        return bool(self._observed)

    def _values(self, vector: np.ndarray) -> dict[str, float]:
        """A vector to minimize as values by objective, with maximized objectives the right way round."""
        return {name: float(v if direction == "minimize" else -v) for (name, direction), v in zip(self._objectives.items(), vector)}

    # --- the front ---------------------------------------------------------

    def _front_indices(self) -> tuple[list[_Observed], list[int]]:
        observed = list(self._observed.values())
        if self._front_stale:
            self._means = [o.mean for o in observed]
            self._front = non_dominated(np.array(self._means)) if observed else []
            self._front_stale = False
        means = np.array(self._means) if self._means else np.zeros((0, len(self._objectives)))
        return observed, sorted(self._front, key=lambda j: (tuple(means[j]), j))

    def front(self) -> Front:
        """The front of every point evaluated so far."""
        observed, indices = self._front_indices()
        points = []
        for i in indices:
            error = observed[i].standard_error if self.config.noise else None
            points.append(
                ParetoPoint(
                    params=dict(observed[i].params),
                    values=self._values(observed[i].mean),
                    evaluations=observed[i].count,
                    standard_error=None if error is None else {name: float(e) for name, e in zip(self._objectives, error)},
                )
            )
        return Front(objectives=self._objectives, points=points)

    def _hypervolume(self) -> float | None:
        observed, indices = self._front_indices()
        if not indices:
            return None
        if self._reference is None:
            # Frozen from the first generation, so hypervolume is comparable through the run.
            values = np.array([o.mean for o in observed])
            worst, best = values.max(axis=0), values.min(axis=0)
            self._reference = worst + REFERENCE_MARGIN * np.maximum(worst - best, np.maximum(np.abs(worst), 1.0) * 1e-9)
        return hypervolume(np.array([observed[i].mean for i in indices]), self._reference)

    # --- the result --------------------------------------------------------

    def result(self) -> ParetoResult:
        """The front of everything evaluated so far, with a summary of every completed generation."""
        if not self._observed:
            raise RuntimeError("no successful evaluations yet")
        return ParetoResult(
            front=self.front(),
            generations=self._history,
            evaluations=self._evaluations,
            failures=self._failures,
            stopped=self._stopped,
            run=self._run_info(),
        )

    def _summarize(self, generation: int, evaluations: int, population: list[Individual]) -> ParetoGenerationStats:
        evaluated = [ind for ind in population if ind.evaluated]
        return ParetoGenerationStats(
            generation=generation,
            evaluations=evaluations,
            failures=sum(ind.failed for ind in evaluated),
            front_size=len(self._front_indices()[1]),
            hypervolume=self._hypervolume(),
        )

    def _report(self, callback: Callable[[ParetoProgress], bool | None] | None, reported: int) -> int:
        if callback is None:
            return len(self._history)
        front = self.front()
        for stats in self._history[reported:]:
            progress = ParetoProgress(generation=stats, evaluations=stats.evaluations, front=front, elapsed=self._elapsed())
            if callback(progress) and not self._done:
                self._finish(stopped=True)
                break
        return len(self._history)

    def _objective(self, objective_function: Callable) -> Callable[[dict[str, Any]], Any]:
        return MultiObjectiveFunction(
            name=getattr(objective_function, "__name__", "objective"), function=objective_function, search_space=self.search_space
        )


class MultiObjectiveRandomSearch(MultiObjectiveOptimizer):
    """Evaluates population_size random points per generation and keeps the front of them all.

    A baseline: a multi-objective optimizer should find a better front for the same number of
    evaluations. Any initial points take the place of random points in the first generation.
    """
    supports_constraints = True
    supports_conditions = True

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        initial = self._initial_individuals(generation) if previous is None else []
        return initial + [self._random_individual(generation) for _ in range(self.config.population_size - len(initial))]


def builtin() -> list[type[MultiObjectiveOptimizer]]:
    """hone's own multi-objective optimizers, which ParetoResult.rebuild_optimizer finds by name."""
    from .genetic import NSGA2
    from .multi_bayesian import MultiObjectiveBayesian

    return [NSGA2, MultiObjectiveBayesian, MultiObjectiveRandomSearch]
