"""Property tests for hone.optimizers.choice (choose_optimizer), from its docstring, the README table and CHANGELOG."""
import numpy as np
from hypothesis import assume, given, settings, strategies as st

from hone import CMAES, BayesianOptimizer, ChoiceParameter, GeneticOptimizer, RangeParameter, UnsupportedSearchSpace, choose_optimizer
from pbt_runs import Recorder, seeds
from pbt_strategies import search_spaces

spaces = search_spaces(kinds=("range", "choice", "fixed", "choose_n"))
budgets = st.integers(min_value=1, max_value=1000)


def expected(space, budget):
    choices = [p for p in space.parameters if isinstance(p, ChoiceParameter)]
    if budget <= 200 and not any(p.choose_n is not None for p in choices):
        return BayesianOptimizer
    integers = [p for p in space.parameters if isinstance(p, RangeParameter) and p.integer and p.maximum > p.minimum]
    return GeneticOptimizer if choices or integers else CMAES


# README table: 200 evaluations or fewer, unless a choice parameter has choose_n: BayesianOptimizer; more than 200,
# any choice or integer parameter: GeneticOptimizer; continuous numbers only: CMAES.
@given(spaces, budgets)
def test_follows_the_table(space, budget):
    try:
        optimizer = choose_optimizer(space, budget)
    except UnsupportedSearchSpace:
        assume(False)  # a space with nothing to search
    assert type(optimizer) is expected(space, budget)
    assert optimizer.config.max_evaluations == budget
    assert optimizer.search_space == space


# "Other settings, such as seed, maximize, noise or threshold, are passed on to the chosen optimizer's config."
@given(spaces, budgets, seeds, st.booleans(), st.booleans(), st.none() | st.floats(min_value=-10, max_value=10))
def test_settings_are_passed_on(space, budget, seed, maximize, noise, threshold):
    try:
        optimizer = choose_optimizer(space, budget, seed=seed, maximize=maximize, noise=noise, threshold=threshold)
    except UnsupportedSearchSpace:
        assume(False)
    config = optimizer.config
    assert (config.seed, config.maximize, config.noise, config.threshold) == (seed, maximize, noise, threshold)


# Initial points are passed on too, checked against the search space.
@given(spaces, budgets, seeds)
def test_initial_points_are_passed_on(space, budget, seed):
    point = space.sample(np.random.default_rng(seed))
    try:
        optimizer = choose_optimizer(space, budget, initial_points=point)
    except UnsupportedSearchSpace:
        assume(False)
    assert optimizer.initial_points == [space.check(point)]


# Every pick runs: the chosen optimizer can optimize the space it was chosen for, within its budget.
@settings(max_examples=25, deadline=None)
@given(search_spaces(kinds=("range", "choice", "fixed"), max_size=3, need_range=True), st.integers(min_value=5, max_value=12), seeds)
def test_chosen_optimizer_runs(space, budget, seed):
    result = choose_optimizer(space, budget, seed=seed).optimize(Recorder(space))
    assert result.evaluations == budget


def test_boundary():
    space = [RangeParameter("x", 0, 1)]
    assert type(choose_optimizer(space, 200)) is BayesianOptimizer
    assert type(choose_optimizer(space, 201)) is CMAES
    assert type(choose_optimizer(space + [ChoiceParameter("c", ["a", "b"])], 201)) is GeneticOptimizer
    assert type(choose_optimizer(space + [ChoiceParameter("c", ["a", "b"], choose_n=2)], 10)) is GeneticOptimizer
