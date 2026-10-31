"""Property tests for hone.optimizers.cmaes, from the module and class docstrings, CMAESConfig and the README."""
import math
import os

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from hone import CMAES, CMAESConfig, RangeParameter, UnsupportedSearchSpace
from pbt_runs import Recorder, seeds
from pbt_strategies import choice_parameters, search_spaces

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
spaces = search_spaces(kinds=("range", "fixed"), need_range=True)


def resolvable(p):
    """A continuous range wide enough, against float resolution, for positions in it to be compared."""
    return not p.integer and p.maximum - p.minimum > 1e-6 * max(1.0, abs(p.maximum))


def searched(space):
    return [p for p in space.parameters if isinstance(p, RangeParameter) and p.maximum > p.minimum]


# "Choice parameters are not supported: use GeneticOptimizer for those."
@given(spaces, choice_parameters(name="a_choice_"))
def test_choice_parameters_refused(space, choice):
    with pytest.raises(UnsupportedSearchSpace):
        CMAES(list(space.parameters) + [choice])


# CMAESConfig: "None: 4 + 3 ln(n) for n variables"; with restarts off every generation has that size.
@run_settings
@given(spaces, seeds)
def test_default_population_size(space, seed):
    n = len(searched(space))
    optimizer = CMAES(space, CMAESConfig(seed=seed, generations=5, restarts=False))
    sizes = []
    while not optimizer.done:
        trials = optimizer.ask()
        sizes.append(len(trials))
        for trial in trials:
            optimizer.tell(trial, sum(float(v) for v in trial.params.values() if isinstance(v, (int, float))))
    assert sizes == [4 + int(3 * math.log(n))] * 5


# "With restarts on, a run that converges or stalls starts again from a random point with twice the population":
# every generation's size is the starting population doubled some number of times, and never shrinks.
@run_settings
@given(spaces, seeds, st.integers(min_value=2, max_value=8))
def test_restart_population_doubles(space, seed, population):
    optimizer = CMAES(space, CMAESConfig(seed=seed, population_size=population, max_evaluations=600))
    sizes = []
    while not optimizer.done:
        trials = optimizer.ask()
        sizes.append(len(trials))
        for trial in trials:
            optimizer.tell(trial, 1.0)  # flat: it stalls and restarts
    assert sizes[:-1] == sorted(sizes[:-1])
    assert all(size in {population * 2**k for k in range(12)} for size in sizes[:-1])  # the last may be cut by the budget


# "The search starts at the centre of every range": with a tiny initial step the first generation sits there.
@run_settings
@given(search_spaces(kinds=("range", "fixed"), need_range=True), seeds)
def test_search_starts_at_the_centre(space, seed):
    continuous = [p for p in searched(space) if resolvable(p)]
    assume(continuous)
    optimizer = CMAES(space, CMAESConfig(seed=seed, initial_step=1e-9))
    for trial in optimizer.ask():
        for p in continuous:
            assert p.to_unit(trial.params[p.name]) == pytest.approx(0.5, abs=1e-6)


# "With initial points, it evaluates them first and starts at the best one instead; a smaller initial_step then keeps
# the search close to it."
@run_settings
@given(spaces, seeds, st.integers(min_value=1, max_value=4))
def test_search_starts_at_the_best_initial_point(space, seed, count):
    continuous = [p for p in searched(space) if resolvable(p)]
    assume(continuous)
    rng = np.random.default_rng(seed)
    points = [space.sample(rng) for _ in range(count)]
    optimizer = CMAES(space, CMAESConfig(seed=seed, initial_step=1e-9), initial_points=points)
    objective = Recorder(space)
    first = optimizer.ask()
    assert [t.params for t in first] == points
    values = []
    for trial in first:
        values.append(objective(**trial.params))
        optimizer.tell(trial, values[-1])
    best = points[int(np.argmin(values))]
    for trial in optimizer.ask():
        for p in continuous:
            assert p.to_unit(trial.params[p.name]) == pytest.approx(p.to_unit(best[p.name]), abs=1e-6)


# "fixed parameters are passed through", and so are ranges with a single value.
@run_settings
@given(spaces, seeds)
def test_constants_pass_through(space, seed):
    objective = Recorder(space)
    CMAES(space, CMAESConfig(seed=seed, max_evaluations=40)).optimize(objective)
    constant = [p for p in space.parameters if p not in searched(space)]
    for params, _ in objective.calls:
        for p in constant:
            assert params[p.name] == space.check({**params})[p.name]
            assert params[p.name] == (p.value if hasattr(p, "value") else p.from_unit(0.0))


# Integer parameters: "Integer parameters keep a minimum spread ... so the search can always move to a neighbouring
# value". On a one-dimensional integer problem whose optimum is away from the centre, it reaches the optimum.
@settings(max_examples=EXAMPLES, deadline=None)
@given(st.integers(min_value=3, max_value=60), st.data(), seeds)
def test_integer_search_reaches_optimum(width, data, seed):
    target = data.draw(st.integers(min_value=0, max_value=width))
    result = CMAES(
        [RangeParameter("k", 0, width, integer=True)], CMAESConfig(seed=seed, max_evaluations=300)
    ).optimize(lambda k: abs(k - target))
    assert result.best_fitness == 0 and result.best_params == {"k": target}


# CMAESConfig: "None: ... or twice the workers if more"; README: "it uses at least twice as many candidates per
# generation as there are workers".
@settings(max_examples=2, deadline=None)
@given(st.integers(min_value=3, max_value=4))
def test_population_grows_with_workers(workers):
    result = CMAES([RangeParameter("x", -1, 1)], CMAESConfig(seed=0, max_evaluations=40)).optimize(square, workers=workers)
    assert result.generations[0].evaluations == 2 * workers


def square(x):
    return x * x
