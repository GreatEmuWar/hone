"""Building optimizers on generated problems, and objectives that record every call, for the property tests."""
import math
from typing import Any

from hypothesis import strategies as st

from hone import (
    CMAES,
    BayesianConfig,
    BayesianOptimizer,
    CMAESConfig,
    ChoiceParameter,
    EvaluationFailed,
    GeneticOptimizer,
    GeneticOptimizerConfig,
    OptimizerConfig,
    RandomSearch,
    RangeParameter,
)
from pbt_strategies import search_spaces

OPTIMIZERS = {
    "genetic": (GeneticOptimizer, GeneticOptimizerConfig, ("range", "choice", "fixed", "choose_n")),
    "random": (RandomSearch, OptimizerConfig, ("range", "choice", "fixed", "choose_n")),
    "cmaes": (CMAES, CMAESConfig, ("range", "fixed")),
    "bayesian": (BayesianOptimizer, BayesianConfig, ("range", "choice", "fixed")),
}
# Bayesian optimization costs a model fit per candidate, so it gets smaller budgets.
BUDGETS = {"genetic": (10, 120), "random": (10, 120), "cmaes": (10, 120), "bayesian": (4, 14)}

seeds = st.integers(min_value=0, max_value=2**32 - 1)


def build(kind: str, space, initial_points=None, **config: Any):
    cls, config_cls, _ = OPTIMIZERS[kind]
    if kind == "random":
        config.setdefault("population_size", 10)
    return cls(space, config_cls(**config), initial_points=initial_points)


@st.composite
def problems(draw, kinds=tuple(OPTIMIZERS), max_size=4):
    """(kind, search space, seed, budget) for an optimizer kind that supports the space."""
    kind = draw(st.sampled_from(kinds))
    space = draw(search_spaces(kinds=OPTIMIZERS[kind][2], max_size=max_size, need_range=kind in ("cmaes", "bayesian")))
    low, high = BUDGETS[kind]
    return kind, space, draw(seeds), draw(st.integers(min_value=low, max_value=high))


def value_of(space, params: dict[str, Any]) -> float:
    """A deterministic objective over any search space: smooth in ranges, ordered by choice index."""
    total = 0.0
    for p in space.parameters:
        value = params[p.name]
        if isinstance(p, RangeParameter):
            total += (float(p.to_unit(value)) - 0.37) ** 2
        elif isinstance(p, ChoiceParameter):
            chosen = value if isinstance(value, list) else [value]
            total += sum(0.1 * (i + 1) * p.choices.index(v) for i, v in enumerate(chosen))
    return total


class Recorder:
    """An objective that records each call. fail_when(params) chooses failures: 'raise', 'nan', 'inf' or None."""

    def __init__(self, space, fail_when=None, sign=1.0, noise=None):
        self.space = space
        self.fail_when = fail_when
        self.sign = sign
        self.noise = noise  # a random.Random adding noise, for noisy objectives
        self.calls: list[tuple[dict[str, Any], float | None]] = []
        self.__name__ = "recorded"

    def __call__(self, **params):
        failure = self.fail_when(params) if self.fail_when else None
        if failure is not None:
            self.calls.append((dict(params), None))
            if failure == "raise":
                raise EvaluationFailed("simulated failure")
            return math.nan if failure == "nan" else math.inf
        value = self.sign * value_of(self.space, params)
        if self.noise is not None:
            value += self.noise.gauss(0, 0.05)
        self.calls.append((dict(params), value))
        return value

    @property
    def values(self) -> list[float]:
        return [value for _, value in self.calls if value is not None]

    @property
    def failures(self) -> int:
        return sum(value is None for _, value in self.calls)


def drive(optimizer, objective, order=None) -> None:
    """Run optimizer by ask and tell, as optimize() does with one worker; order permutes each batch's tells."""
    while not optimizer.done:
        trials = optimizer.ask()
        assert trials, "ask() returned nothing although the run isn't done and nothing is outstanding"
        outcomes = []
        for trial in trials:
            try:
                outcomes.append((trial, objective(**trial.params)))
            except EvaluationFailed:
                outcomes.append((trial, None))
        if order is not None:
            outcomes = order(outcomes)
        for trial, value in outcomes:
            if value is None:
                optimizer.tell(trial, failed=True)
            else:
                optimizer.tell(trial, value)
