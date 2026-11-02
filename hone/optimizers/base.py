"""The base class of every single-objective optimizer: one number per evaluation, to minimize or maximize."""
import math
import numbers
from typing import Any, Callable

import numpy as np

from hone.result import GenerationStats, Progress, Result

from .core import (
    CHECKPOINT_FORMAT,
    OBJECTIVE,
    EvaluationFailed,
    Individual,
    OptimizerCore,
    Trial,
    UnsupportedSearchSpace,
    _evaluate,
    _hone_version,
    _key,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "OBJECTIVE",
    "EvaluationFailed",
    "Individual",
    "Optimizer",
    "Trial",
    "UnsupportedSearchSpace",
    "_evaluate",
    "_fitness",
    "_hone_version",
    "_key",
]


class Optimizer(OptimizerCore[Result]):
    """Base class for population-based optimizers of a single objective.

    Use it either way round:

    - optimize(objective) calls the objective for every candidate and returns the Result.
    - ask() and tell() let you evaluate candidates yourself, anywhere and in any order;
      result() returns the Result so far.

    The base class runs the generation loop: it hands out each generation's candidates,
    records their outcomes, tracks the best so far, stops at the threshold, and builds the
    Result. A subclass only decides how each generation's population is produced, by
    implementing _next_population.

    Args:
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: The optimizer's configuration. Defaults are used if omitted.
        initial_points: Points to evaluate first, such as a known good design or the best
            parameters of an earlier run: one dict of parameter values, or a list of them.
            Fixed parameters may be left out, and so must parameters a condition switches off.

    Every individual an optimizer breeds holds a value for every parameter, including those
    the conditions switch off at its point, so an optimizer can keep them to use again. The
    base class leaves them out of what it hands out and reports: trials, results and progress.

    The objective returns a number. To record an evaluation as failed instead, it can raise
    EvaluationFailed or return NaN or infinity.

    With constraints on outputs, the objective returns a dict instead: its value under
    "objective", and a number for each output the constraints use. Evaluations are then ranked
    by Deb's feasibility rule: a point that satisfies every output constraint beats one that
    doesn't, feasible points rank by value, and the others by their total violation, so no
    penalty weight is needed. Failed evaluations rank below them all.
    """

    def reset(self) -> None:
        """Start over: forget every evaluation and reseed the random number generator."""
        super().reset()
        self._best: Individual | None = None
        # With noise: every successful value seen at each point, and the final re-evaluation's state.
        self._observations: dict[tuple, list[float]] = {}
        self._points: dict[tuple, dict[str, Any]] = {}
        self._finalists: list[tuple] = []
        self._final_round = False
        self._infeasible = 0  # evaluations that broke a constraint on the outputs

    # --- what a value is ---------------------------------------------------

    def _check_outcome(self, value: Any, failed: bool) -> bool:
        if failed:
            return True
        if value is None:
            raise ValueError("pass the trial's value, or failed=True")
        outputs = self.search_space.outputs
        if outputs:
            return not _check_outputs(
                value, [OBJECTIVE, *outputs], f"the value, since the search space has constraints on the outputs {', '.join(outputs)},"
            )
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"the value must be a number, got {value!r}")
        return not math.isfinite(value)

    def _record_failure(self, individual: Individual) -> None:
        individual.fitness = -math.inf if self.config.maximize else math.inf

    def _record(self, individual: Individual, value: Any) -> None:
        if isinstance(value, dict):
            individual.outputs = {name: float(value[name]) for name in self.search_space.outputs}
            individual.violation = self.search_space.output_violation(individual.params, individual.outputs)
            self._infeasible += individual.violation > 0
            value = value[OBJECTIVE]
        individual.fitness = float(value)
        if self.config.noise:
            # Average every evaluation of the same point, so a lucky value regresses toward its mean.
            key = self._key(individual.params)
            self._observations.setdefault(key, []).append(float(value))
            self._points.setdefault(key, dict(individual.params))
            individual.fitness = float(np.mean(self._observations[key]))
        if self._best is None or self._rank(individual) < self._rank(self._best):
            self._best = individual

    def _has_result(self) -> bool:
        return self._best is not None

    # --- the result --------------------------------------------------------

    def result(self) -> Result:
        """The best result found so far, with a summary of every completed generation."""
        best = self._best_point()
        if best is None:
            raise RuntimeError("no successful evaluations yet")
        best_fitness, best_params = best
        assert self._best is not None
        return Result(
            best_fitness=best_fitness,
            best_params=dict(best_params),
            best_outputs=dict(self._best.outputs) if self._best.outputs is not None else None,
            feasible=self._best.violation == 0,
            generations=self._history,
            evaluations=self._evaluations,
            failures=self._failures,
            infeasible=self._infeasible,
            reached_threshold=self._reached_threshold,
            stopped=self._stopped,
            run=self._run_info(),
        )

    def _best_point(self) -> tuple[float, dict[str, Any]] | None:
        """The best value so far and its parameters, or None before any evaluation has succeeded."""
        if self._best is None:
            return None
        if self.config.noise and self._observations:
            # Once the finalists have been re-evaluated, the one with the best average; before
            # that, the most promising point by the optimizer's best estimate.
            key = self._ranked_keys(self._finalists)[0] if self._finalists else self._finalist_order()[0]
            return self._average(key), self.search_space.active(self._points[key])
        return _fitness(self._best), self.search_space.active(self._best.params)

    def _report(self, callback: Callable[[Progress], bool | None] | None, reported: int) -> int:
        """Pass callback each generation finished since the first reported ones, and stop the run if it asks."""
        if callback is None:
            return len(self._history)
        best = self._best_point()
        for stats in self._history[reported:]:
            progress = Progress(
                generation=stats,
                evaluations=stats.evaluations,
                best_fitness=best[0] if best is not None else None,
                best_params=dict(best[1]) if best is not None else None,
                feasible=self._best is None or self._best.violation == 0,
                elapsed=self._elapsed(),
            )
            if callback(progress) and not self._done:
                self._finish(stopped=True)
                break
        return len(self._history)

    # --- stopping, and the final round with noise ---------------------------

    def _goal_reached(self) -> bool:
        best = self._best_point()
        # Only a point that satisfies every constraint on the outputs can reach the threshold.
        return best is not None and self._best is not None and self._best.violation == 0 and self._reached(best[0])

    def _begin_final_round(self) -> bool:
        if self.config.reserved_evaluations and not self._final_round and self._observations:
            self._final_round = True
            self._needs_generation = True
            return True
        return False

    def _kept_back(self) -> int:
        return 0 if self._final_round else self.config.reserved_evaluations

    def _make_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        if self._final_round:
            return self._final_population(generation)
        return self._next_population(previous, generation)

    # --- noise -----------------------------------------------------------------

    def _average(self, key: tuple) -> float:
        return float(np.mean(self._observations[key]))

    def _ranked_keys(self, keys: list[tuple]) -> list[tuple]:
        """Evaluated points, by key, best average first."""
        return sorted(keys, key=self._average, reverse=self.config.maximize)

    def _finalist_order(self) -> list[tuple]:
        """Every successfully evaluated point, by key, most promising first. Optimizers with a
        better estimate of each point's true value than its average, such as a model, override this."""
        return self._ranked_keys(list(self._observations))

    def _refresh_averages(self, individuals: list[Individual]) -> None:
        """Set each successful individual's fitness to the current average of every evaluation of its point."""
        for individual in individuals:
            if not individual.failed and individual.fitness is not None:
                individual.fitness = self._average(self._key(individual.params))

    def _final_population(self, generation: int) -> list[Individual]:
        """The final round with noise: the most promising points, each evaluated again.

        Up to 5 finalists share the final_evaluations, each getting at least two when there are
        enough, and the result is the finalist with the best average.
        """
        reserve = self.config.reserved_evaluations
        if self.config.max_evaluations is not None:
            reserve = min(reserve, self.config.max_evaluations - self._handed_out_total)
        order = self._finalist_order()
        self._finalists = order[: max(1, min(5, reserve // 2, len(order)))]
        return [
            Individual(params=dict(self._points[self._finalists[i % len(self._finalists)]]), generation=generation)
            for i in range(reserve)
        ]

    # --- ranking ---------------------------------------------------------------

    def _rank(self, individual: Individual) -> tuple[bool, float, float]:
        """A sort key putting the best individual first, by Deb's feasibility rule.

        Successful before failed; among those, smaller violation of the output constraints
        first, which puts every feasible point (violation 0) ahead; then better fitness. Without
        output constraints every violation is 0, so this ranks by fitness alone.
        """
        fitness = _fitness(individual)
        return individual.failed, individual.violation, -fitness if self.config.maximize else fitness

    def _ranked(self, individuals: list[Individual]) -> list[Individual]:
        """The individuals, best first: by feasibility, then fitness. Failed individuals sort last."""
        return sorted(individuals, key=self._rank)

    def _reached(self, fitness: float) -> bool:
        """Whether fitness is good enough to stop early. Never true without a threshold."""
        if self.config.threshold is None:
            return False
        return fitness >= self.config.threshold if self.config.maximize else fitness <= self.config.threshold

    def _summarize(self, generation: int, evaluations: int, population: list[Individual]) -> GenerationStats:
        evaluated = [ind for ind in population if ind.fitness is not None]
        ranked = self._ranked([ind for ind in evaluated if not ind.failed])
        failures = len(evaluated) - len(ranked)
        # The fitness statistics cover only points that satisfy the output constraints.
        succeeded = [ind for ind in ranked if ind.violation == 0]
        infeasible = len(ranked) - len(succeeded)
        if not succeeded:
            return GenerationStats(generation=generation, evaluations=evaluations, failures=failures, infeasible=infeasible)
        fitnesses = [_fitness(ind) for ind in succeeded]
        return GenerationStats(
            generation=generation,
            evaluations=evaluations,
            failures=failures,
            infeasible=infeasible,
            best_fitness=succeeded[0].fitness,
            best_params=self.search_space.active(succeeded[0].params),
            worst_fitness=succeeded[-1].fitness,
            mean=float(np.mean(fitnesses)),
            median=float(np.median(fitnesses)),
            std=float(np.std(fitnesses)),
        )


def _check_outputs(value: Any, names: list[str], what: str) -> bool:
    """Check value is a dict with a number for exactly the given names, raising TypeError if not. Returns whether every number is finite."""
    if not isinstance(value, dict):
        raise TypeError(f"{what} must be a dict with a number for each of {', '.join(names)}, got {value!r}")
    missing, extra = [n for n in names if n not in value], sorted(set(value) - set(names))
    if missing or extra:
        problems = [f"missing {', '.join(missing)}"] * bool(missing) + [f"unknown {', '.join(map(str, extra))}"] * bool(extra)
        raise TypeError(f"{what} must have exactly {', '.join(names)}: {'; '.join(problems)}")
    for name in names:
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, numbers.Real):
            raise TypeError(f"the value of {name} must be a number, got {number!r}")
    return all(math.isfinite(value[name]) for name in names)


def _fitness(individual: Individual) -> float:
    """The fitness of an individual that has been evaluated."""
    assert individual.fitness is not None, "an individual was ranked before it was evaluated"
    return individual.fitness
