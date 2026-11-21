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

from .core import MARGIN_FREEDOM, MARGIN_HALF_AT_LEAST, Individual, OptimizerCore, _OutputNoise

REFERENCE_MARGIN = 0.1  # the default reference point lies this share of each objective's range past its worst value
# The final round's second half re-evaluates this many points of the front, each several times: three repeats
# halve a one-draw standard error, and four points keep more of a front than its ends.
SECOND_HALF_FRONT_POINTS = 4


class _Observed:
    """Every evaluation of one point: the sums for its averages, and where it was first seen."""

    def __init__(self, params: dict[str, Any], first: int, outputs: dict[str, float] | None = None) -> None:
        self.params = params
        self.first = first
        self.outputs = outputs
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


class MultiObjectiveOptimizer(_OutputNoise, OptimizerCore[ParetoResult]):
    """Base class for optimizers of several objectives, which return a ParetoResult.

    The objective function returns a dict with one value per objective named in the config's
    objectives, such as {"accuracy": 0.97, "cost": 12.0}, and tell() takes the same. A dict with
    missing or extra keys, or a value that isn't a number, raises TypeError: it is a bug in the
    objective. Raise EvaluationFailed, or return NaN or infinity for any objective, to record an
    evaluation as failed; failed evaluations are never on the front.

    The front is that of every point evaluated, not only the latest generation, so a trade-off
    found early is never lost. With noise, each point's values are the averages of all its
    evaluations, and the front comes with their standard errors.

    With constraints on outputs, the dict also holds a number for each output they use; a name
    can be both an objective and an output. Only points that satisfy every one join the front,
    and the optimizers rank the others behind them by their total violation.

    With noise and constraints on outputs, a point's outputs are averaged like its values, and
    whether it satisfies a limit is judged on the averages, with a margin of safety set by
    feasibility_confidence, as for one objective. Every successful evaluation of a point counts
    toward its averages, including those that break a limit, so a point isn't biased toward
    passing. The front is then that of the points that pass, and each point's outputs are its
    averages. By default a tenth of max_evaluations is kept back for a final round that
    re-evaluates points of the front (see final_evaluations): in two halves, the first spreading
    over the front to measure the noise, the second on the points whose pass is least certain.
    """
    config_class = MultiObjectiveConfig
    config: MultiObjectiveConfig
    supports_noisy_outputs = True

    def reset(self) -> None:
        """Start over: forget every evaluation and reseed the random number generator."""
        super().reset()
        self._objectives = dict(self.config.objectives)
        self._observed: dict[tuple, _Observed] = {}  # every point evaluated successfully, in the order first seen
        self._means: list[np.ndarray] = []  # each observed point's values to minimize, in the same order
        self._front: list[int] = []  # the observed points on the front, by position in that order
        self._front_stale = False  # with noise, an average changed, so the front must be worked out again
        self._succeeded = 0  # successful evaluations, feasible or not
        self._infeasible = 0  # evaluations that broke a constraint on the outputs
        self._infeasible_values: list[np.ndarray] = []  # their values to minimize, for the reference point
        self._reference: np.ndarray | None = None if self.config.reference is None else _minimizing(self.config.reference, self._objectives)
        self._reset_output_noise()
        # The final round for a front, only with noise and constraints on outputs.
        self._finalists: list[tuple] = []
        self._final_round = False
        self._final_half = 0  # with the margin of safety: which half of the final round is under way, 1 or 2
        self._second_half = 0  # evaluations kept back for the final round's second half
        self._second_reserve = 0  # the second half's size, once the first has begun
        self._reserve = self._reserved() if self._averaged() else 0

    def _averaged(self) -> bool:
        """Whether a point's outputs are averaged and judged against the limits: noise with constraints on outputs."""
        return bool(self.config.noise and self.search_space.output_constraints)

    # --- what a value is ---------------------------------------------------

    def _check_outcome(self, value: Any, failed: bool) -> bool:
        if failed:
            return True
        if value is None:
            raise ValueError("pass the trial's values, or failed=True")
        outputs = [name for name in self.search_space.outputs if name not in self._objectives]
        if not isinstance(value, dict):
            raise TypeError(f"the value must be a dict with a number for each objective ({', '.join(self._objectives)}), got {value!r}")
        expected = [*self._objectives, *outputs]
        missing, extra = [n for n in expected if n not in value], sorted(set(value) - set(expected))
        if missing or extra:
            problems = [f"missing {', '.join(missing)}"] * bool(missing) + [f"unknown {', '.join(map(str, extra))}"] * bool(extra)
            what = "the objectives" + (f" and the outputs {', '.join(outputs)}" if outputs else "")
            raise TypeError(f"the values must have exactly {what} ({', '.join(expected)}): {'; '.join(problems)}")
        for name, number in value.items():
            if isinstance(number, bool) or not isinstance(number, numbers.Real):
                raise TypeError(f"the value of {name} must be a number, got {number!r}")
        return not all(math.isfinite(number) for number in value.values())

    def _record_failure(self, individual: Individual) -> None:
        individual.values = None

    def _record_averaged(self, individual: Individual, value: Any) -> None:
        """Record a successful evaluation with noise and constraints on outputs.

        Every draw joins its point's averages, those that break a limit too, so the averages aren't
        biased toward passing; the draw counts as infeasible if it broke a limit. The individual then
        holds its point's averages and the verdict on them.
        """
        values = {name: float(value[name]) for name in self._objectives}
        self._succeeded += 1
        outputs = {name: float(value[name]) for name in self.search_space.outputs}
        if self.search_space.output_violation(individual.params, outputs) > 0:
            self._infeasible += 1
            self._infeasible_values.append(_minimizing(values, self._objectives))
        key = self._key(individual.params)
        observed = self._observed.get(key)
        if observed is None:
            observed = self._observed[key] = _Observed(self.search_space.active(individual.params), individual.id)
            observed.add(_minimizing(values, self._objectives))
            self._means.append(observed.mean)
        else:
            observed.add(_minimizing(values, self._objectives))
        self._points.setdefault(key, dict(individual.params))
        self._output_observations.setdefault(key, []).append(outputs)
        self._front_stale = True
        individual.values = self._values(observed.mean)
        individual.outputs = self._average_outputs(key)
        individual.violation = self._violation(key)

    def _record(self, individual: Individual, value: Any) -> None:
        if self._averaged():
            return self._record_averaged(individual, value)
        values = {name: float(value[name]) for name in self._objectives}
        self._succeeded += 1
        if self.search_space.output_constraints:
            individual.outputs = {name: float(value[name]) for name in self.search_space.outputs}
            individual.violation = self.search_space.output_violation(individual.params, individual.outputs)
            if individual.violation > 0:
                # A point that breaks a limit never joins the front, but its values still teach a model.
                self._infeasible += 1
                self._infeasible_values.append(_minimizing(values, self._objectives))
                individual.values = values
                return
        key = self._key(individual.params)
        observed = self._observed.get(key)
        new = observed is None
        if observed is None:
            observed = self._observed[key] = _Observed(self.search_space.active(individual.params), individual.id, individual.outputs)
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
        return self._succeeded > 0

    def _values(self, vector: np.ndarray) -> dict[str, float]:
        """A vector to minimize as values by objective, with maximized objectives the right way round."""
        return {name: float(v if direction == "minimize" else -v) for (name, direction), v in zip(self._objectives.items(), vector)}

    # --- the front ---------------------------------------------------------

    def _front_indices(self) -> tuple[list[_Observed], list[int]]:
        observed = list(self._observed.values())
        if self._front_stale:
            self._means = [o.mean for o in observed]
            if self._averaged():
                # Only the points that pass on their averages, margin included, can be on the front.
                passing = [i for i, key in enumerate(self._observed) if self._violation(key) == 0]
                self._front = [passing[j] for j in non_dominated(np.array([self._means[i] for i in passing]))] if passing else []
            else:
                self._front = non_dominated(np.array(self._means)) if observed else []
            self._front_stale = False
        means = np.array(self._means) if self._means else np.zeros((0, len(self._objectives)))
        return observed, sorted(self._front, key=lambda j: (tuple(means[j]), j))

    def front(self) -> Front:
        """The front of every point evaluated so far."""
        observed, indices = self._front_indices()
        keys = list(self._observed) if self._averaged() else []
        points = []
        for i in indices:
            error = observed[i].standard_error if self.config.noise else None
            points.append(
                ParetoPoint(
                    params=dict(observed[i].params),
                    values=self._values(observed[i].mean),
                    evaluations=observed[i].count,
                    standard_error=None if error is None else {name: float(e) for name, e in zip(self._objectives, error)},
                    outputs=self._average_outputs(keys[i]) if keys else _copy(observed[i].outputs),
                )
            )
        return Front(objectives=self._objectives, points=points)

    def _hypervolume(self) -> float | None:
        observed, indices = self._front_indices()
        if not indices:
            return None
        if self._reference is None:
            # Frozen from the first generation, so hypervolume is comparable through the run.
            # With output constraints, infeasible points count too, so a front of one early point doesn't pin it there.
            values = np.array([o.mean for o in observed] + self._infeasible_values)
            worst, best = values.max(axis=0), values.min(axis=0)
            self._reference = worst + REFERENCE_MARGIN * np.maximum(worst - best, np.maximum(np.abs(worst), 1.0) * 1e-9)
        return hypervolume(np.array([observed[i].mean for i in indices]), self._reference)

    # --- the result --------------------------------------------------------

    def result(self) -> ParetoResult:
        """The front of everything evaluated so far, with a summary of every completed generation."""
        if not self._succeeded:
            raise RuntimeError("no successful evaluations yet")
        return ParetoResult(
            front=self.front(),
            generations=self._history,
            evaluations=self._evaluations,
            failures=self._failures,
            infeasible=self._infeasible,
            feasible=self._any_feasible(),
            stopped=self._stopped,
            run=self._run_info(),
        )

    def _any_feasible(self) -> bool:
        """Whether any point satisfies the limits: with noise and constraints on outputs, on its averages with the margin."""
        return bool(self._front_indices()[1]) if self._averaged() else bool(self._observed)

    # --- the final round for a front, with noise and constraints on outputs -------

    def _begin_final_round(self) -> bool:
        if self._reserve and not self._final_round and self._observed:
            self._final_round = True
            self._needs_generation = True
            return True
        if self._final_half == 1 and self._second_half:
            # The second half of the final round, now that the finalists' repeats give the noise.
            self._final_half, self._second_half = 2, 0
            self._needs_generation = True
            return True
        return False

    def _kept_back(self) -> int:
        return self._second_half if self._final_round else self._reserve

    def _make_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        if self._final_round:
            return self._final_population(generation)
        return self._next_population(previous, generation)

    def _final_population(self, generation: int) -> list[Individual]:
        """The final round for a front: points to evaluate again, so that what passes rests on more than one draw.

        The first half (the whole round without the margin) spreads over the front's points; the
        second goes to the points whose pass is least certain. The result is the front of the
        points that pass on their averages, each judged at the number of draws it has.
        """
        if self._final_half == 2:
            reserve = self._second_reserve
            chosen = self._least_certain(reserve)
            self._finalists += [key for key in chosen if key not in self._finalists]
        else:
            reserve = self._reserve
            if self.config.max_evaluations is not None:
                reserve = min(reserve, self.config.max_evaluations - self._handed_out_total)
            if self._margin_on() and reserve >= 2 * MARGIN_HALF_AT_LEAST:
                self._final_half, self._second_half = 1, reserve // 2
                self._second_reserve = reserve // 2
                reserve -= reserve // 2
            chosen = self._spread_finalists(reserve)
            self._finalists = list(chosen)
        return [Individual(params=dict(self._points[chosen[i % len(chosen)]]), generation=generation) for i in range(reserve)]

    def _unreached(self, keys: list[tuple], more: int = 0) -> list[tuple]:
        """Points by key, least violating first, ties in the order first seen: the order for points that fail."""
        order = {key: i for i, key in enumerate(self._observed)}
        return sorted(keys, key=lambda key: (self._violation(key, more), order[key]))

    def _spread_finalists(self, reserve: int) -> list[tuple]:
        """The first half's points: the front's, chosen to spread along it, and enough to give every limit degrees of freedom."""
        keys = list(self._observed)
        _, indices = self._front_indices()
        front = self._by_contribution(indices)
        on_front = set(indices)
        rest = self._unreached([key for i, key in enumerate(keys) if i not in on_front])
        if front:
            ordered = [keys[i] for i in front] + rest
            count = max(1, min(len(front), reserve // 2))
        else:
            ordered = self._unreached(keys)
            count = max(1, min(len(ordered), reserve // 2))
        chosen = ordered[:count]
        if self._margin_on():
            constraints = self.search_space.output_constraints
            known = self._limit_noise()

            def applies(j: int, key: tuple) -> bool:
                return constraints[j].applies(self.search_space.active(self._points[key]))

            def short() -> list[int]:
                """The limits that, after this round, would still have too few degrees of freedom."""
                given = [reserve // len(chosen) + (i < reserve % len(chosen)) for i in range(len(chosen))]
                return [
                    j
                    for j in range(len(constraints))
                    if known[j][1] + sum(extra for key, extra in zip(chosen, given) if applies(j, key)) < MARGIN_FREEDOM
                ]

            while len(chosen) < reserve // 2:
                added = False
                for j in short():
                    nxt = next((k for k in ordered if k not in chosen and applies(j, k)), None)
                    if nxt is not None:
                        chosen.append(nxt)
                        added = True
                        break
                if not added:
                    break
        return chosen

    def _by_contribution(self, indices: list[int]) -> list[int]:
        """The front's points, as positions among the observed, greedily by hypervolume contribution: each next adds the most to those chosen.

        Ties, and points beyond the reference point, which add nothing, keep the front's order.
        """
        if not indices:
            return []
        if self._reference is None:
            self._hypervolume()
        assert self._reference is not None
        remaining: list[int] = list(indices)
        chosen: list[int] = []
        while remaining:
            best, best_volume = remaining[0], -1.0
            for i in remaining:
                volume = hypervolume(np.array([self._means[j] for j in [*chosen, i]]), self._reference)
                if volume > best_volume:
                    best, best_volume = i, volume
            chosen.append(best)
            remaining.remove(best)
        return chosen

    def _least_certain(self, reserve: int) -> list[tuple]:
        """The second half's points: those of the front whose pass, after `more` more evaluations, is least certain.

        Every point is re-judged with the standard error it would have after its share of the half, and
        the front taken from those that pass. Certainty is how many standard errors the point's tightest limit is cleared by.
        """
        self._front_indices()  # brings every point's average up to date
        keys = list(self._observed)
        count = max(1, min(SECOND_HALF_FRONT_POINTS, reserve // 3))
        more = reserve // count
        passing = [i for i, key in enumerate(keys) if self._violation(key, more) == 0]
        if not passing:
            return self._unreached(keys, more)[:count]
        front = sorted(
            (passing[j] for j in non_dominated(np.array([self._means[i] for i in passing]))), key=lambda i: (tuple(self._means[i]), i)
        )
        certainty = [self._certainty(keys[i], more) for i in front]
        order = sorted(range(len(front)), key=lambda r: certainty[r])  # stable: ties keep the front's order
        return [keys[front[r]] for r in order[:count]]

    def _certainty(self, key: tuple, more: int) -> float:
        """How many standard errors the point's average clears its tightest limit by, after `more` more evaluations.

        Only limits that apply, with pooled noise known and above zero, count; infinity when there is none.
        """
        margins = self.search_space.output_margins(self._points[key], self._average_outputs(key) or {})
        n = len(self._output_observations[key]) + more
        z = math.inf
        for margin, (spread, freedom) in zip(margins, self._limit_noise()):
            if margin is None or not math.isfinite(margin) or freedom < MARGIN_FREEDOM or spread <= 0:
                continue
            z = min(z, -margin / (spread / math.sqrt(n)))
        return z

    def _summarize(self, generation: int, evaluations: int, population: list[Individual]) -> ParetoGenerationStats:
        evaluated = [ind for ind in population if ind.evaluated]
        return ParetoGenerationStats(
            generation=generation,
            evaluations=evaluations,
            failures=sum(ind.failed for ind in evaluated),
            infeasible=sum(not ind.failed and ind.violation > 0 for ind in evaluated),
            front_size=len(self._front_indices()[1]),
            hypervolume=self._hypervolume(),
        )

    def _report(self, callback: Callable[[ParetoProgress], bool | None] | None, reported: int) -> int:
        if callback is None:
            return len(self._history)
        front = self.front()
        for stats in self._history[reported:]:
            progress = ParetoProgress(
                generation=stats, evaluations=stats.evaluations, front=front, elapsed=self._elapsed(), feasible=self._any_feasible()
            )
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


def _copy(outputs: dict[str, float] | None) -> dict[str, float] | None:
    return None if outputs is None else dict(outputs)
