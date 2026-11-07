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

DEFAULT_FINAL_EVALUATIONS = 10  # kept back by default with noise and constraints on outputs, when a tenth of the budget is less, or there is none


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

    With noise, a point's outputs are averaged like its value, and whether it satisfies a limit
    is judged on the averages: the limit is on the true output, which the average estimates.
    By default a tenth of the budget is then kept back for the final round, so the result's
    feasibility rests on more than one draw; see final_evaluations.
    """
    supports_noisy_outputs = True

    def reset(self) -> None:
        """Start over: forget every evaluation and reseed the random number generator."""
        super().reset()
        self._best: Individual | None = None
        # With noise: every successful value seen at each point, and the final re-evaluation's state.
        self._observations: dict[tuple, list[float]] = {}
        self._output_observations: dict[tuple, list[dict[str, float]]] = {}  # every set of outputs seen at each point
        self._points: dict[tuple, dict[str, Any]] = {}
        self._finalists: list[tuple] = []
        self._final_round = False
        self._reserve = self._reserved()
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
        outputs = None
        if isinstance(value, dict):
            outputs = {name: float(value[name]) for name in self.search_space.outputs}
            self._infeasible += self.search_space.output_violation(individual.params, outputs) > 0
            value = value[OBJECTIVE]
        if self.config.noise:
            # Average every evaluation of the same point, outputs included, so a lucky value, or a
            # limit passed by luck, regresses toward its mean.
            key = self._key(individual.params)
            self._observations.setdefault(key, []).append(float(value))
            self._points.setdefault(key, dict(individual.params))
            if outputs is not None:
                self._output_observations.setdefault(key, []).append(outputs)
            self._refresh(individual)
        else:
            individual.fitness = float(value)
            if outputs is not None:
                individual.outputs = outputs
                individual.violation = self.search_space.output_violation(individual.params, outputs)
        if self._best is None or self._rank(individual) < self._rank(self._best):
            self._best = individual

    def _has_result(self) -> bool:
        return self._best is not None

    # --- the result --------------------------------------------------------

    def result(self) -> Result:
        """The best result found so far, with a summary of every completed generation."""
        chosen = self._chosen()
        if chosen is None:
            raise RuntimeError("no successful evaluations yet")
        best_fitness, best_params, best_outputs, feasible = chosen
        return Result(
            best_fitness=best_fitness,
            best_params=dict(best_params),
            best_outputs=best_outputs,
            feasible=feasible,
            generations=self._history,
            evaluations=self._evaluations,
            failures=self._failures,
            infeasible=self._infeasible,
            reached_threshold=self._reached_threshold,
            stopped=self._stopped,
            run=self._run_info(),
        )

    def _chosen(self) -> tuple[float, dict[str, Any], dict[str, float] | None, bool] | None:
        """The result so far: the best value, its parameters, its outputs and whether they satisfy every output constraint.

        None before any evaluation has succeeded. With noise, the value and outputs are averages:
        once the finalists have been re-evaluated, of the one with the best average; before
        that, of the most promising point by the optimizer's best estimate.
        """
        if self._best is None:
            return None
        if self.config.noise and self._observations:
            key = self._ranked_keys(self._finalists)[0] if self._finalists else self._finalist_order()[0]
            return self._average(key), self.search_space.active(self._points[key]), self._average_outputs(key), self._violation(key) == 0
        best = self._best
        outputs = dict(best.outputs) if best.outputs is not None else None
        return _fitness(best), self.search_space.active(best.params), outputs, best.violation == 0

    def _best_point(self) -> tuple[float, dict[str, Any]] | None:
        """The best value so far and its parameters, or None before any evaluation has succeeded."""
        chosen = self._chosen()
        return None if chosen is None else chosen[:2]

    def _report(self, callback: Callable[[Progress], bool | None] | None, reported: int) -> int:
        """Pass callback each generation finished since the first reported ones, and stop the run if it asks."""
        if callback is None:
            return len(self._history)
        chosen = self._chosen()
        for stats in self._history[reported:]:
            progress = Progress(
                generation=stats,
                evaluations=stats.evaluations,
                best_fitness=chosen[0] if chosen is not None else None,
                best_params=dict(chosen[1]) if chosen is not None else None,
                feasible=chosen is None or chosen[3],
                elapsed=self._elapsed(),
            )
            if callback(progress) and not self._done:
                self._finish(stopped=True)
                break
        return len(self._history)

    # --- stopping, and the final round with noise ---------------------------

    def _goal_reached(self) -> bool:
        if self.config.threshold is None:
            return False  # and with noise, ranking every point to find the best is work for nothing
        chosen = self._chosen()
        # Only a point that satisfies every constraint on the outputs can reach the threshold.
        return chosen is not None and chosen[3] and self._reached(chosen[0])

    def _reserved(self) -> int:
        """Evaluations kept back for the final round with noise: final_evaluations, or its default.

        The default is none, except with constraints on outputs, where it is a tenth of
        max_evaluations, at least DEFAULT_FINAL_EVALUATIONS and at most half, or
        DEFAULT_FINAL_EVALUATIONS without max_evaluations: judged on one draw, a design on the
        boundary of a limit passes it half the time.
        """
        config = self.config
        if not config.noise:
            return 0
        if config.final_evaluations is not None:
            return config.final_evaluations
        if not self.search_space.output_constraints:
            return 0
        if config.max_evaluations is None:
            return DEFAULT_FINAL_EVALUATIONS
        return min(max(DEFAULT_FINAL_EVALUATIONS, config.max_evaluations // 10), config.max_evaluations // 2)

    def _begin_final_round(self) -> bool:
        if self._reserve and not self._final_round and self._observations:
            self._final_round = True
            self._needs_generation = True
            return True
        return False

    def _kept_back(self) -> int:
        return 0 if self._final_round else self._reserve

    def _make_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        if self._final_round:
            return self._final_population(generation)
        return self._next_population(previous, generation)

    # --- noise -----------------------------------------------------------------

    def _average(self, key: tuple) -> float:
        return float(np.mean(self._observations[key]))

    def _average_outputs(self, key: tuple) -> dict[str, float] | None:
        """The average of every set of outputs seen at the point, by output; None without constraints on outputs."""
        seen = self._output_observations.get(key)
        if not seen:
            return None
        return {name: float(np.mean([outputs[name] for outputs in seen])) for name in self.search_space.outputs}

    def _violation(self, key: tuple) -> float:
        """The total violation of the output constraints by the point's average outputs: 0 without any."""
        outputs = self._average_outputs(key)
        return 0.0 if outputs is None else self.search_space.output_violation(self._points[key], outputs)

    def _ranked_keys(self, keys: list[tuple]) -> list[tuple]:
        """Evaluated points, by key, best first: by Deb's rule on the averages, so those whose average
        outputs satisfy every constraint come first, by average value, then the rest by violation."""
        sign = -1.0 if self.config.maximize else 1.0
        return sorted(keys, key=lambda key: (self._violation(key), sign * self._average(key)))

    def _finalist_order(self) -> list[tuple]:
        """Every successfully evaluated point, by key, most promising first. Optimizers with a
        better estimate of each point's true value than its average, such as a model, override this."""
        return self._ranked_keys(list(self._observations))

    def _refresh_averages(self, individuals: list[Individual]) -> None:
        """Set each successful individual's fitness, outputs and violation to the averages of every evaluation of its point."""
        for individual in individuals:
            if not individual.failed and individual.fitness is not None:
                self._refresh(individual)

    def _refresh(self, individual: Individual) -> None:
        key = self._key(individual.params)
        individual.fitness = self._average(key)
        if self.search_space.output_constraints:
            individual.outputs = self._average_outputs(key)
            individual.violation = self._violation(key)

    def _final_population(self, generation: int) -> list[Individual]:
        """The final round with noise: the most promising points, each evaluated again.

        Up to 5 finalists share the final_evaluations, each getting at least two when there are
        enough, and the result is the finalist with the best average. With constraints on
        outputs, a finalist whose average outputs satisfy them all beats one whose don't.
        """
        reserve = self._reserve
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
