"""Property tests for conditional parameters, from the SearchSpace docstrings and docs/design/conditions.md."""
import os

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from hone import ChoiceParameter, OptimizerConfig, RandomSearch, RangeParameter, SearchSpace
from hone.parameters.conditions import holds

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def conditional_spaces(draw):
    """A search space in which each parameter may be switched on by a test of an earlier one, listed in any order."""
    parameters = []
    conditions = []
    for i in range(draw(st.integers(1, 6))):
        if draw(st.booleans()):
            parameter = ChoiceParameter(f"p{i}", list(range(draw(st.integers(1, 4)))))
        else:
            parameter = RangeParameter(f"p{i}", 0, 10, integer=draw(st.booleans()))
        if parameters and draw(st.booleans()):
            parent = draw(st.sampled_from(parameters))
            if isinstance(parent, ChoiceParameter):
                values = draw(st.lists(st.sampled_from(parent.choices), min_size=1, unique=True))
                conditions.append(parent[tuple(values)] >> parameter)
            else:
                conditions.append((parent >= draw(st.integers(0, 10))) >> parameter)
        parameters.append(parameter)
    order = draw(st.permutations(parameters))
    return SearchSpace(order, conditions=conditions)


# A point is the sample without what the conditions switch off, and check accepts it as it is.
@run_settings
@given(conditional_spaces(), st.integers(0, 2**32 - 1))
def test_the_active_point_of_a_sample_checks_as_is(space, seed):
    point = space.active(space.sample(np.random.default_rng(seed)))
    assert space.check(point) == point
    assert space.active(point) == point
    assert list(point) == [name for name in space.names if name in point]


# A parameter is switched on exactly when it has no condition, or its test holds at the point.
@run_settings
@given(conditional_spaces(), st.integers(0, 2**32 - 1))
def test_a_parameter_is_on_exactly_when_its_test_holds(space, seed):
    point = space.active(space.sample(np.random.default_rng(seed)))
    for condition in space.conditions:
        for name in condition.parameters:
            assert (name in point) == holds(condition.when, point)
    unconditional = [name for name in space.names if not space.is_conditional(name)]
    assert set(unconditional) <= set(point)


# A value for a parameter that is switched off is refused, and so is a missing value for one switched on.
@run_settings
@given(conditional_spaces(), st.integers(0, 2**32 - 1))
def test_check_refuses_switched_off_values_and_missing_ones(space, seed):
    full = space.sample(np.random.default_rng(seed))
    point = space.active(full)
    for name in set(full) - set(point):
        try:
            space.check({**point, name: full[name]})
        except ValueError as error:
            assert "isn't used at this point" in str(error)
        else:
            raise AssertionError(f"{name} was accepted although it is switched off")
    for name in point:
        try:
            space.check({k: v for k, v in point.items() if k != name})
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} was left out of {point}, yet the point was accepted")


# Saving and loading keeps the conditions, and so which parameters each point switches on.
@run_settings
@given(conditional_spaces(), st.integers(0, 2**32 - 1))
def test_a_conditional_space_saves_and_loads(space, seed):
    loaded = SearchSpace.model_validate_json(space.model_dump_json())
    assert loaded == space
    full = space.sample(np.random.default_rng(seed))
    assert loaded.active(full) == space.active(full)


# Random search only hands out points without switched-off parameters, and its result is one too.
@run_settings
@given(conditional_spaces(), st.integers(0, 2**32 - 1))
def test_random_search_hands_out_active_points(space, seed):
    optimizer = RandomSearch(space, OptimizerConfig(seed=seed, max_evaluations=30, population_size=10))
    while not optimizer.done:
        for trial in optimizer.ask():
            assert space.check(trial.params) == trial.params
            optimizer.tell(trial, float(sum(v for v in trial.params.values())))
    best = optimizer.result().best_params
    assert space.check(best) == best
