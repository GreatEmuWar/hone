"""Which parameters matter: global sensitivity analysis, before or instead of optimizing.

sensitivity() evaluates the objective at points chosen to measure each parameter's effect,
through the same workers, time limits and failure handling as optimize(), and reports each
parameter's importance. On small budgets it uses Morris screening, which ranks the
parameters; on larger ones Sobol indices, which say what share of the objective's variance
each parameter accounts for, alone and with its interactions.

Every parameter is measured on the scale the optimizers search it on: a log range evenly in
its logarithm, an integer range rounded, and a choice parameter by picking among its choices,
so an effect is the change across the parameter's whole range. Fixed parameters aren't
measured. A parameter a condition switches off isn't passed to the objective at that point,
so it has no effect there, which its indices include.
"""
import math
from functools import partial
from typing import Any, Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import qmc

from hone.objective_function import ObjectiveFunction
from hone.optimizers.base import _check_outputs
from hone.optimizers.core import OBJECTIVE, UnsupportedSearchSpace, _evaluate
from hone.optimizers._pool import EvaluationPool
from hone.parameters import ChoiceParameter, FixedParameter, Parameter, RangeParameter, SearchSpace

type Method = Literal["morris", "sobol"]

MORRIS_LEVELS = 4  # grid levels per parameter; the step is half the grid, so each level pairs with another
MORRIS_STEP = MORRIS_LEVELS / (2 * (MORRIS_LEVELS - 1))  # 2/3 of the range
MIN_TRAJECTORIES = 2  # Morris needs at least two trajectories to estimate sigma
MIN_SOBOL_SAMPLES = 16  # Sobol rows below which its indices are too noisy to report
AUTO_SOBOL_SAMPLES = 64  # rows at which sensitivity() picks Sobol over Morris by itself
BOOTSTRAP_RESAMPLES = 1000
DEFAULT_SOBOL_THRESHOLD = 0.05  # a total index below which reduced() fixes a parameter
DEFAULT_MORRIS_THRESHOLD = 0.1  # a share of the largest mu_star below which reduced() fixes a parameter


class Interval(BaseModel):
    """A confidence interval."""
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    low: float
    """The lower end."""
    high: float
    """The upper end."""


class ParameterSensitivity(BaseModel):
    """How much one parameter matters.

    importance ranks the parameters: mu_star for Morris, the total index for Sobol. The
    other fields are None for the method that wasn't used.
    """
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    name: str
    """The parameter's name."""
    importance: float
    """mu_star for Morris, the total index for Sobol."""
    interval: Interval | None = None
    """The confidence interval of importance; None when too few evaluations succeeded to estimate it."""
    mu_star: float | None = None
    """Morris: the mean size of the parameter's effect, the change in the objective across its whole
    range, whichever way it goes. Small means the parameter barely matters."""
    mu: float | None = None
    """Morris: the mean effect with its sign. Well below mu_star means the effect changes direction."""
    sigma: float | None = None
    """Morris: the standard deviation of the effect. Large against mu_star means the effect depends on
    the other parameters, through interactions or a curved response."""
    first: float | None = None
    """Sobol: the share of the objective's variance the parameter accounts for alone."""
    first_interval: Interval | None = None
    """Sobol: the confidence interval of first."""
    total: float | None = None
    """Sobol: the share of the variance the parameter accounts for with all its interactions; near 0
    means it can be fixed at any value with little effect. Estimates can fall slightly below 0."""
    total_interval: Interval | None = None
    """Sobol: the confidence interval of total, the same as interval."""


class SensitivityReport(BaseModel):
    """What sensitivity() measured: each parameter's importance, most important first."""
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    method: Method
    """How the parameters were measured: "morris" or "sobol"."""
    parameters: list[ParameterSensitivity]
    """Each parameter measured, most important first."""
    evaluations: int
    """Evaluations made, failed ones included."""
    failures: int
    """Evaluations that failed, timed out or crashed their worker."""
    confidence: float
    """The confidence level of the intervals."""
    best_params: dict[str, Any] | None
    """The best point evaluated, as the objective was called with it; None if every evaluation failed."""
    best_value: float | None
    """The objective's value at best_params; None if every evaluation failed."""
    maximize: bool
    """Whether higher values were counted as better in choosing best_params."""

    @property
    def ranking(self) -> list[str]:
        """The parameters' names, most important first."""
        return [parameter.name for parameter in self.parameters]

    def __getitem__(self, name: str) -> ParameterSensitivity:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def unimportant(self, threshold: float | None = None) -> list[str]:
        """The parameters that can be fixed with little effect: those whose whole interval is below threshold.

        For Sobol, threshold is a total index, 0.05 unless given: the parameter accounts for
        under 5% of the objective's variance, interactions included. For Morris it is a share of
        the largest mu_star, 0.1 unless given. Only the upper end of each interval is compared,
        so a parameter whose importance is uncertain is kept. A parameter without an interval,
        because too few of its evaluations succeeded, is kept too.
        """
        if threshold is None:
            threshold = DEFAULT_SOBOL_THRESHOLD if self.method == "sobol" else DEFAULT_MORRIS_THRESHOLD
        if threshold < 0:
            raise ValueError(f"threshold must be at least 0, got {threshold}")
        if self.method == "morris":
            largest = max((p.importance for p in self.parameters if math.isfinite(p.importance)), default=0.0)
            threshold *= largest
        return [p.name for p in self.parameters if p.interval is not None and p.interval.high < threshold]

    def reduced(
        self,
        search_space: SearchSpace | Sequence[Parameter],
        *,
        threshold: float | None = None,
        keep: int | None = None,
        at: dict[str, Any] | None = None,
    ) -> SearchSpace:
        """The search space with its unimportant parameters fixed, ready for choose_optimizer.

        The parameters unimportant() names are fixed, or with keep, every parameter but the
        keep most important. Each is fixed at its value in at, if given, and otherwise at the best
        point evaluated; one the conditions switched off there, or every parameter if every
        evaluation failed, is fixed at the middle of its range, or its middle choice.

        A fixed parameter is still passed to the objective, so the objective doesn't change.
        Conditions and constraints are kept: a parameter one of them tests stays a range or
        choice parameter, narrowed to its one value, so they still apply.

        Args:
            search_space: The search space the report measured.
            threshold: Passed to unimportant(). Not with keep.
            keep: Keep this many of the most important parameters, and fix the rest.
            at: Values to fix parameters at, by name.
        """
        space = search_space if isinstance(search_space, SearchSpace) else SearchSpace(search_space)
        measured = [p.name for p in space.parameters if not isinstance(p, FixedParameter)]
        if sorted(measured) != sorted(self.ranking):
            raise ValueError(f"this report measured {', '.join(self.ranking)}, but the search space's parameters are {', '.join(measured)}")
        if keep is not None:
            if threshold is not None:
                raise ValueError("give threshold or keep, not both")
            if keep < 1:
                raise ValueError(f"keep must be at least 1, got {keep}")
            dropped = self.ranking[keep:]
        else:
            dropped = self.unimportant(threshold)
        unknown = sorted(set(at or {}) - set(space.names))
        if unknown:
            raise ValueError(f"at names parameters the search space doesn't have: {', '.join(unknown)}")
        tested = {name for c in space.conditions for name in c.when.names()} | {name for c in space.constraints for name in c.names()}
        parameters = []
        for parameter in space.parameters:
            if parameter.name not in dropped:
                parameters.append(parameter)
                continue
            if at is not None and parameter.name in at:
                value = parameter.check(at[parameter.name])
            elif self.best_params is not None and parameter.name in self.best_params:
                value = self.best_params[parameter.name]
            else:
                value = _from_unit(parameter, 0.5)
            parameters.append(_fixed(parameter, value, keep_kind=parameter.name in tested))
        return SearchSpace(parameters, space.constraints, space.conditions)

    def __str__(self) -> str:
        if self.method == "morris":
            header = ("parameter", "mu_star", "interval", "mu", "sigma")
            rows = [
                (p.name, _number(p.mu_star), _interval(p.interval), _number(p.mu), _number(p.sigma))
                for p in self.parameters
            ]
        else:
            header = ("parameter", "total", "interval", "first", "interval")
            rows = [
                (p.name, _number(p.total), _interval(p.total_interval), _number(p.first), _interval(p.first_interval))
                for p in self.parameters
            ]
        widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
        lines = [
            f"{'Morris screening' if self.method == 'morris' else 'Sobol indices'}: {self.evaluations} evaluations, "
            f"{self.failures} failed; {self.confidence:.0%} intervals"
        ]
        for row in [header, *rows]:
            lines.append("  ".join(cell.ljust(width) if i == 0 else cell.rjust(width) for i, (cell, width) in enumerate(zip(row, widths))))
        return "\n".join(lines)


def sensitivity(
    objective_function: Callable,
    search_space: SearchSpace | Sequence[Parameter],
    *,
    max_evaluations: int = 500,
    method: Method | None = None,
    maximize: bool = False,
    seed: int | None = None,
    workers: int = 1,
    timeout: float | None = None,
    confidence: float = 0.95,
) -> SensitivityReport:
    """Measure how much each parameter of the search space matters to the objective.

    Morris screening needs about 10 × (parameters + 1) evaluations to rank the parameters.
    Sobol indices need about 64 × (parameters + 2) or more, and say what share of the
    objective's variance each parameter accounts for. Without method, Sobol is used when
    max_evaluations allows it, and Morris otherwise. Neither uses more than max_evaluations.

    The objective is called as optimize() calls it: with one keyword argument per parameter
    switched on, raising EvaluationFailed or returning NaN or infinity to record a failed
    evaluation. Failed evaluations are left out of the indices.

    Args:
        objective_function: The function to measure.
        search_space: The parameters, with any conditions. Constraints on outputs are allowed and
            ignored: the objective returns its dict, and only its "objective" is measured.
            Constraints on the parameters aren't supported yet.
        max_evaluations: The most evaluations to make.
        method: "morris" or "sobol", rather than choosing from max_evaluations.
        maximize: Whether higher values are better, which only decides best_params.
        seed: Fixes the points evaluated and the intervals, so a run can be repeated.
        workers: Evaluate this many points at once in background processes.
        timeout: The longest, in seconds, one evaluation may take, as in optimize().
        confidence: The confidence level of the intervals, from bootstrapping.

    With workers > 1 or a timeout, each evaluation runs in a separate process, so the
    function must be defined at the top level of a module.
    """
    space = search_space if isinstance(search_space, SearchSpace) else SearchSpace(search_space)
    if space.parameter_constraints:
        raise UnsupportedSearchSpace(
            "sensitivity() doesn't support constraints on the parameters yet: Morris and Sobol assume each parameter can "
            "take any of its values whatever the others are"
        )
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    measured = [parameter for parameter in space.parameters if not isinstance(parameter, FixedParameter)]
    if not measured:
        raise ValueError("every parameter is fixed, so there is nothing to measure")
    k = len(measured)
    method = method or ("sobol" if _sobol_samples(max_evaluations, k) >= AUTO_SOBOL_SAMPLES else "morris")
    if method not in ("morris", "sobol"):
        raise ValueError(f"method must be 'morris' or 'sobol', got {method!r}")
    rng = np.random.default_rng(seed)

    if method == "morris":
        trajectories = max_evaluations // (k + 1)
        if trajectories < MIN_TRAJECTORIES:
            raise ValueError(
                f"Morris screening of {k} parameters needs at least {MIN_TRAJECTORIES * (k + 1)} evaluations, "
                f"and about {10 * (k + 1)} to rank them reliably; max_evaluations is {max_evaluations}"
            )
        units, steps = _morris_design(trajectories, k, rng)
    else:
        n = _sobol_samples(max_evaluations, k)
        if n < MIN_SOBOL_SAMPLES:
            raise ValueError(
                f"Sobol indices of {k} parameters need at least {MIN_SOBOL_SAMPLES * (k + 2)} evaluations, "
                f"and about {AUTO_SOBOL_SAMPLES * (k + 2)} to be reliable; max_evaluations is {max_evaluations}. "
                "method='morris' ranks the parameters with fewer"
            )
        units = _sobol_design(n, k, rng)

    points = [_point(space, measured, row) for row in units]
    values = _evaluate_all(objective_function, space, points, workers, timeout)
    ok = np.isfinite(values)

    if method == "morris":
        results = _morris_indices(measured, values, steps, confidence, rng)
    else:
        results = _sobol_indices(measured, values, n, confidence, rng)
    results.sort(key=lambda parameter: -parameter.importance if math.isfinite(parameter.importance) else math.inf)

    best_params, best_value = None, None
    if ok.any():
        best = int(np.nanargmax(values) if maximize else np.nanargmin(values))
        best_params, best_value = space.active(points[best]), float(values[best])
    return SensitivityReport(
        method=method,
        parameters=results,
        evaluations=len(points),
        failures=int((~ok).sum()),
        confidence=confidence,
        best_params=best_params,
        best_value=best_value,
        maximize=maximize,
    )


# --- from the unit cube to points ----------------------------------------------


def _point(space: SearchSpace, measured: Sequence[Parameter], unit: np.ndarray) -> dict[str, Any]:
    """The point of space at position unit, one coordinate in [0, 1] per measured parameter."""
    values = {parameter.name: _from_unit(parameter, float(u)) for parameter, u in zip(measured, unit)}
    return {
        parameter.name: parameter.value if isinstance(parameter, FixedParameter) else values[parameter.name]
        for parameter in space.parameters
    }


def _from_unit(parameter: Parameter, u: float) -> Any:
    """The parameter's value at u in [0, 1], on the scale the optimizers search it on."""
    if isinstance(parameter, RangeParameter):
        return parameter.from_unit(u)
    assert isinstance(parameter, ChoiceParameter)
    if parameter.choose_n is None:
        return parameter.choices[min(int(u * len(parameter.choices)), len(parameter.choices) - 1)]
    count = math.comb(len(parameter.choices), parameter.choose_n)
    indices = _unrank_combination(min(int(u * count), count - 1), len(parameter.choices), parameter.choose_n)
    return [parameter.choices[i] for i in indices]


def _fixed(parameter: Parameter, value: Any, keep_kind: bool) -> Parameter:
    """parameter held at value: a FixedParameter, or with keep_kind, a range or choice of that one value."""
    if isinstance(parameter, RangeParameter):
        if keep_kind:
            return RangeParameter(parameter.name, value, value, integer=parameter.integer, log=parameter.log)
        return FixedParameter(parameter.name, value)
    assert isinstance(parameter, ChoiceParameter)
    if parameter.choose_n is not None:
        return ChoiceParameter(parameter.name, list(value), choose_n=parameter.choose_n)
    return ChoiceParameter(parameter.name, [value]) if keep_kind else FixedParameter(parameter.name, value)


def _unrank_combination(rank: int, n: int, k: int) -> list[int]:
    """The rank-th k-element subset of range(n), in lexicographic order."""
    chosen: list[int] = []
    candidate = 0
    while len(chosen) < k:
        with_candidate = math.comb(n - candidate - 1, k - len(chosen) - 1)
        if rank < with_candidate:
            chosen.append(candidate)
        else:
            rank -= with_candidate
        candidate += 1
    return chosen


def _evaluate_all(
    objective_function: Callable, space: SearchSpace, points: list[dict[str, Any]], workers: int, timeout: float | None
) -> np.ndarray:
    """Each point's value, NaN where the evaluation failed, as optimize() evaluates them."""
    objective = ObjectiveFunction(
        name=getattr(objective_function, "__name__", "objective"), function=objective_function, search_space=space
    )
    evaluate = partial(_evaluate, objective)
    called = [space.active(point) for point in points]
    if workers == 1 and timeout is None:
        outcomes = [evaluate(point) for point in called]
    else:
        with EvaluationPool(evaluate, workers, timeout) as pool:
            outcomes = pool.map(called)
    if space.output_constraints:
        # The objective returns a dict with its outputs, which don't change where the points are.
        for outcome in outcomes:
            if outcome is not None:
                _check_outputs(outcome, [OBJECTIVE, *space.outputs], f"{objective.name}()'s value")
        outcomes = [None if outcome is None else outcome[OBJECTIVE] for outcome in outcomes]
    values = np.array([math.nan if outcome is None else outcome for outcome in outcomes], dtype=float)
    values[~np.isfinite(values)] = math.nan
    return values


# --- Morris screening ------------------------------------------------------------


def _morris_design(trajectories: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, list[list[tuple[int, int, float]]]]:
    """Morris's one-at-a-time trajectories through a grid of MORRIS_LEVELS levels.

    Returns the points, trajectories × (k + 1) rows of k unit coordinates, and for each
    trajectory its steps: (parameter, row it steps from, signed step size).
    """
    levels = np.arange(MORRIS_LEVELS) / (MORRIS_LEVELS - 1)
    rows: list[np.ndarray] = []
    steps: list[list[tuple[int, int, float]]] = []
    for _ in range(trajectories):
        x = rng.choice(levels, size=k)
        start = len(rows)
        rows.append(x.copy())
        trajectory = []
        for position, i in enumerate(rng.permutation(k)):
            step = MORRIS_STEP if x[i] + MORRIS_STEP <= 1 + 1e-12 else -MORRIS_STEP
            x[i] = round(x[i] + step, 12)
            rows.append(x.copy())
            trajectory.append((int(i), start + position, step))
        steps.append(trajectory)
    return np.array(rows), steps


def _morris_indices(
    measured: Sequence[Parameter], values: np.ndarray, steps: list[list[tuple[int, int, float]]], confidence: float, rng: np.random.Generator
) -> list[ParameterSensitivity]:
    effects: list[list[float]] = [[] for _ in measured]
    for trajectory in steps:
        for i, row, step in trajectory:
            before, after = values[row], values[row + 1]
            if math.isfinite(before) and math.isfinite(after):
                effects[i].append((after - before) / step)
    results = []
    for parameter, found in zip(measured, effects):
        e = np.array(found)
        if not len(e):
            results.append(ParameterSensitivity(name=parameter.name, importance=math.nan))
            continue
        mu_star = float(np.mean(np.abs(e)))
        interval = None
        if len(e) >= 2:
            resampled = np.abs(e[rng.integers(len(e), size=(BOOTSTRAP_RESAMPLES, len(e)))]).mean(axis=1)
            interval = _percentiles(resampled, confidence)
        results.append(
            ParameterSensitivity(
                name=parameter.name,
                importance=mu_star,
                interval=interval,
                mu_star=mu_star,
                mu=float(np.mean(e)),
                sigma=float(np.std(e, ddof=1)) if len(e) >= 2 else None,
            )
        )
    return results


# --- Sobol indices -----------------------------------------------------------------


def _sobol_samples(max_evaluations: int, k: int) -> int:
    """The most rows of Sobol's design that fit in max_evaluations: a power of 2, as its sequence needs."""
    rows = max_evaluations // (k + 2)
    return 2 ** int(math.log2(rows)) if rows >= 1 else 0


def _sobol_design(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Saltelli's design: matrices A and B, then for each parameter A with that column from B.

    A and B are the two halves of one scrambled Sobol sequence of 2k dimensions, as scipy's
    sobol_indices draws them. Returns the (k + 2) n rows in that order.
    """
    a_b = qmc.Sobol(d=2 * k, scramble=True, seed=rng).random(n)
    a, b = a_b[:, :k], a_b[:, k:]
    blocks = [a, b]
    for i in range(k):
        ab = a.copy()
        ab[:, i] = b[:, i]
        blocks.append(ab)
    return np.vstack(blocks)


def _saltelli(f_a: np.ndarray, f_b: np.ndarray, f_ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and total indices by Saltelli et al. (2010), as scipy's sobol_indices computes them.

    The last axis is the rows; f_ab has one more leading axis, the parameter.
    """
    mean = np.mean([f_a, f_b], axis=(0, -1), keepdims=True)[0]
    f_a, f_b, f_ab = f_a - mean, f_b - mean, f_ab - mean
    var = np.var([f_a, f_b], axis=(0, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.mean(f_b * (f_ab - f_a), axis=-1) / var
        total = 0.5 * np.mean((f_a - f_ab) ** 2, axis=-1) / var
    # An objective that doesn't vary has no variance to share out.
    return np.where(np.isfinite(first), first, 0.0), np.where(np.isfinite(total), total, 0.0)


def _sobol_indices(measured: Sequence[Parameter], values: np.ndarray, n: int, confidence: float, rng: np.random.Generator) -> list[ParameterSensitivity]:
    k = len(measured)
    f_a, f_b = values[:n], values[n:2 * n]
    f_ab = values[2 * n:].reshape(k, n)
    # A failed evaluation spoils its row: keep the rows where every evaluation succeeded.
    rows = np.flatnonzero(np.isfinite(f_a) & np.isfinite(f_b) & np.isfinite(f_ab).all(axis=0))
    if len(rows) < 2:
        return [ParameterSensitivity(name=parameter.name, importance=math.nan) for parameter in measured]
    first, total = _saltelli(f_a[rows], f_b[rows], f_ab[:, rows])
    firsts, totals = [], []
    for start in range(0, BOOTSTRAP_RESAMPLES, 100):
        picks = rows[rng.integers(len(rows), size=(min(100, BOOTSTRAP_RESAMPLES - start), len(rows)))]
        s, t = _saltelli(f_a[picks], f_b[picks], f_ab[:, picks])  # (k, resamples) each
        firsts.append(s)
        totals.append(t)
    first_samples, total_samples = np.concatenate(firsts, axis=-1), np.concatenate(totals, axis=-1)
    return [
        ParameterSensitivity(
            name=parameter.name,
            importance=float(total[i]),
            interval=(total_interval := _percentiles(total_samples[i], confidence)),
            first=float(first[i]),
            first_interval=_percentiles(first_samples[i], confidence),
            total=float(total[i]),
            total_interval=total_interval,
        )
        for i, parameter in enumerate(measured)
    ]


def _percentiles(samples: np.ndarray, confidence: float) -> Interval:
    """The percentile bootstrap interval of samples."""
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return Interval(low=float(low), high=float(high))


def _number(value: float | None) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.4g}"


def _interval(interval: Interval | None) -> str:
    return "-" if interval is None else f"[{interval.low:.3g}, {interval.high:.3g}]"
