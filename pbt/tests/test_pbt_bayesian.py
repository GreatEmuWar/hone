"""Property tests for hone.optimizers.bayesian, from its docstrings, BayesianConfig, the README and CHANGELOG."""
import os

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from hone import BayesianConfig, BayesianOptimizer, ChoiceParameter, RangeParameter, UnsupportedSearchSpace
from pbt_runs import Recorder, seeds
from pbt_strategies import choice_parameters, search_spaces

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=max(EXAMPLES // 2, 8), deadline=None, suppress_health_check=[HealthCheck.too_slow])
spaces = search_spaces(kinds=("range", "choice", "fixed"), max_size=4, need_range=True)
continuous_spaces = search_spaces(kinds=("range", "fixed"), max_size=4, need_range=True, integer=False)


def variables(space):
    """Parameters with more than one possible value."""
    return [
        p for p in space.parameters
        if (isinstance(p, RangeParameter) and p.maximum > p.minimum) or (isinstance(p, ChoiceParameter) and len(p.choices) > 1)
    ]


# "Range parameters ... and choice parameters are supported, but not choice parameters with choose_n."
@given(spaces, choice_parameters(name="a_choice_", choose_n=True))
def test_choose_n_refused(space, choice):
    with pytest.raises(UnsupportedSearchSpace):
        BayesianOptimizer(list(space.parameters) + [choice])


# README: "Its first 2 × (variables + 1) candidates are spread over the search space; after that it proposes one
# candidate at a time ... and population_size sets that batch size directly." Initial points join the first.
@run_settings
@given(spaces, seeds, st.none() | st.integers(min_value=1, max_value=3), st.none() | st.integers(min_value=2, max_value=8),
       st.integers(min_value=0, max_value=3))
def test_batch_sizes(space, seed, batch, initial_samples, initial_count):
    rng = np.random.default_rng(seed)
    points = [space.sample(rng) for _ in range(initial_count)]
    optimizer = BayesianOptimizer(
        space, BayesianConfig(seed=seed, population_size=batch, initial_samples=initial_samples, generations=4),
        initial_points=points,
    )
    objective = Recorder(space)
    sizes = []
    while not optimizer.done:
        trials = optimizer.ask()
        if not sizes:
            assert [t.params for t in trials[: len(points)]] == points
        sizes.append(len(trials))
        for trial in trials:
            optimizer.tell(trial, objective(**trial.params))
    first = initial_samples or 2 * (len(variables(space)) + 1)
    assert sizes == [max(first, len(points))] + [batch or 1] * 3


# The code and CHANGELOG 0.8.0: candidates "still being evaluated are assumed to come out as predicted, so none is
# proposed again", and each candidate of a batch "goes elsewhere". On a continuous space, no point is proposed twice.
@run_settings
@given(continuous_spaces, seeds, st.integers(min_value=1, max_value=3))
def test_no_point_is_proposed_twice(space, seed, batch):
    objective = Recorder(space)
    BayesianOptimizer(space, BayesianConfig(seed=seed, population_size=batch, max_evaluations=14)).optimize(objective)
    points = [repr(sorted(params.items())) for params, _ in objective.calls]
    assert len(points) == len(set(points))


# The same with min_finished below 1: late results are outstanding while later batches are proposed.
@run_settings
@given(continuous_spaces, seeds, st.randoms(use_true_random=False))
def test_outstanding_points_are_not_proposed_again(space, seed, rnd):
    optimizer = BayesianOptimizer(space, BayesianConfig(seed=seed, population_size=3, max_evaluations=16, min_finished=0.6))
    objective = Recorder(space)
    held, seen = [], []
    while not optimizer.done:
        trials = optimizer.ask()
        seen += [repr(sorted(t.params.items())) for t in trials]
        for trial in trials:
            if rnd.random() < 0.3:
                held.append(trial)
            else:
                optimizer.tell(trial, objective(**trial.params))
        if held and (not trials or rnd.random() < 0.5):
            trial = held.pop(0)
            optimizer.tell(trial, objective(**trial.params))
    assert len(seen) == len(set(seen))
    result = optimizer.result()
    assert result.evaluations == 16 and result.best_fitness == min(objective.values)


# Docstring: "Failed evaluations are modelled as the worst value seen, which steers the search away from them":
# a run where some evaluations fail still completes and returns the best successful value.
@run_settings
@given(spaces, seeds)
def test_failures_are_survived(space, seed):
    objective = Recorder(space, fail_when=lambda params: "nan" if hash(repr(sorted(params.items()))) % 3 == 0 else None)
    try:
        result = BayesianOptimizer(space, BayesianConfig(seed=seed, max_evaluations=12)).optimize(objective)
    except RuntimeError:
        assume(False)
    assert result.best_fitness == min(objective.values)
    assert result.failures == objective.failures


# On a one-dimensional smooth problem the model finds the optimum to within a small error in a few dozen evaluations
# (README: "The first choice when evaluations are expensive and the budget is a few hundred evaluations or fewer").
@settings(max_examples=5, deadline=None)
@given(st.floats(min_value=-4, max_value=4), seeds)
def test_finds_a_one_dimensional_optimum(target, seed):
    result = BayesianOptimizer(
        [RangeParameter("x", -5, 5)], BayesianConfig(seed=seed, max_evaluations=20)
    ).optimize(lambda x: (x - target) ** 2)
    assert result.best_fitness < 1e-2
