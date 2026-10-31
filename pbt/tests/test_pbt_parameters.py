"""Property tests for hone.parameters, from its docstrings and the README's Parameters section."""

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from hone.parameters import FixedParameter, RangeParameter, SearchSpace
from pbt_strategies import choice_parameters, fixed_parameters, range_parameters, search_spaces

seeds = st.integers(min_value=0, max_value=2**32 - 1)


def in_range(p: RangeParameter, value) -> bool:
    return p.minimum <= value <= p.maximum


# sample() draws a value the parameter accepts: check() takes it back unchanged.
@given(range_parameters(), seeds)
def test_range_sample_is_a_value_of_the_range(p, seed):
    value = p.sample(np.random.default_rng(seed))
    assert in_range(p, value)
    assert isinstance(value, int) if p.integer else isinstance(value, float)
    assert p.check(value) == value


# from_unit: "The value at position unit in the range, clipped to it, and rounded for an integer range."
@given(range_parameters(), st.floats(min_value=-10, max_value=10, allow_nan=False))
def test_range_from_unit_is_clipped_and_rounded(p, unit):
    value = p.from_unit(unit)
    assert in_range(p, value)
    if p.integer:
        assert isinstance(value, int)
    assert p.check(value) == value


# from_unit is "the inverse of to_unit".
@given(range_parameters(integer=False), st.floats(min_value=0, max_value=1))
def test_range_to_unit_inverts_from_unit(p, unit):
    assume(p.maximum > p.minimum)
    value = p.from_unit(unit)
    assert p.to_unit(value) == pytest.approx(unit, abs=1e-6)


@given(range_parameters(), st.floats(min_value=0, max_value=1))
def test_range_from_unit_inverts_to_unit(p, unit):
    value = p.from_unit(unit)  # any value of the range
    assert p.from_unit(p.to_unit(value)) == pytest.approx(value, rel=1e-9, abs=1e-9)
    if p.integer:
        assert p.from_unit(p.to_unit(value)) == value


# to_unit and from_unit accept "a number or an array of them": both give the same values.
@given(range_parameters(), st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=5))
def test_range_unit_conversion_vectorizes(p, units):
    values = p.from_unit(np.array(units))
    assert list(values) == pytest.approx([p.from_unit(u) for u in units])
    assert list(p.to_unit(values)) == pytest.approx([p.to_unit(v) for v in values])


# to_unit: 0 at the minimum, 1 at the maximum.
@given(range_parameters())
def test_range_to_unit_endpoints(p):
    assume(p.maximum > p.minimum)
    assert p.to_unit(p.minimum) == pytest.approx(0.0, abs=1e-12)
    assert p.to_unit(p.maximum) == pytest.approx(1.0)


# mutate: "The result is clipped to the bounds"; integer ranges stay integers.
@given(range_parameters(), seeds, st.floats(min_value=1e-3, max_value=2.0))
def test_range_mutate_stays_in_the_range(p, seed, scale):
    rng = np.random.default_rng(seed)
    value = p.sample(rng)
    mutated = p.mutate(value, rng, scale)
    assert in_range(p, mutated)
    assert p.check(mutated) == mutated


# check rejects values outside the range, and non-numbers (booleans included).
@given(range_parameters(), st.floats(min_value=1e-3, max_value=1e3))
def test_range_check_rejects_out_of_range(p, beyond):
    with pytest.raises(ValueError):
        p.check(p.maximum + beyond + abs(p.maximum) * 1e-9)
    with pytest.raises(ValueError):
        p.check(p.minimum - beyond - abs(p.minimum) * 1e-9)
    with pytest.raises(ValueError):
        p.check(True)
    with pytest.raises(ValueError):
        p.check("1")


# The validator: minimum above maximum, an integer range with no integer, a log range without a positive minimum.
@given(st.floats(min_value=-1e6, max_value=1e6), st.floats(min_value=1e-3, max_value=1e6))
def test_range_rejects_reversed_bounds(low, gap):
    with pytest.raises(ValueError):
        RangeParameter("x", low + gap, low)


@given(st.floats(min_value=-1e6, max_value=0))
def test_log_range_needs_positive_minimum(low):
    with pytest.raises(ValueError):
        RangeParameter("x", low, 1e7, log=True)


# ChoiceParameter: a sample is one of the choices, or choose_n distinct ones.
@given(choice_parameters(choose_n=True), seeds)
def test_choose_n_sample_is_distinct_choices(p, seed):
    value = p.sample(np.random.default_rng(seed))
    assert len(value) == p.choose_n
    assert all(v in p.choices for v in value)
    assert len({repr(v) for v in value}) == p.choose_n
    assert p.check(value) == value


@given(choice_parameters(), seeds)
def test_choice_sample_is_a_choice(p, seed):
    value = p.sample(np.random.default_rng(seed))
    assert value in p.choices
    assert p.check(value) == value


# mutate: "A single choice switches to a different one when there is one to switch to."
@given(choice_parameters(), seeds)
def test_choice_mutate_switches(p, seed):
    rng = np.random.default_rng(seed)
    value = p.sample(rng)
    mutated = p.mutate(value, rng, 0.1)
    assert mutated in p.choices
    if len(p.choices) > 1:
        assert mutated != value


@given(choice_parameters(choose_n=True), seeds)
def test_choose_n_mutate_is_valid(p, seed):
    rng = np.random.default_rng(seed)
    mutated = p.mutate(p.sample(rng), rng, 0.1)
    assert p.check(mutated) == mutated


@given(choice_parameters(choose_n=True), st.data())
def test_choose_n_check_rejects_repeats(p, data):
    assume(p.choose_n >= 2)
    first = data.draw(st.sampled_from(p.choices))
    with pytest.raises(ValueError):
        p.check([first] * p.choose_n)


# FixedParameter: "A parameter held at a single value"; mutate never changes it.
@given(fixed_parameters(), seeds)
def test_fixed_never_changes(p, seed):
    rng = np.random.default_rng(seed)
    assert p.sample(rng) == p.value
    assert p.mutate(p.value, rng, 1.0) == p.value
    assert p.check(p.value) == p.value


# SearchSpace.check: "params as a complete point of this space, in parameter order"; fixed may be left out.
@given(search_spaces(), seeds)
def test_search_space_check_accepts_samples_in_order(space, seed):
    point = space.sample(np.random.default_rng(seed))
    checked = space.check(point)
    assert list(checked) == space.names
    assert checked == point
    assert space.check(checked) == checked  # idempotent
    without_fixed = {k: v for k, v in point.items() if not isinstance(space[k], FixedParameter)}
    assert space.check(without_fixed) == checked
    reordered = dict(reversed(list(point.items())))
    assert list(space.check(reordered)) == space.names


@given(search_spaces(), seeds, st.data())
def test_search_space_check_rejects_missing_and_unknown(space, seed, data):
    point = space.sample(np.random.default_rng(seed))
    with pytest.raises(ValueError):
        space.check({**point, "not_a_parameter_name_": 1})
    required = [p.name for p in space.parameters if not isinstance(p, FixedParameter)]
    assume(required)
    name = data.draw(st.sampled_from(required))
    with pytest.raises(ValueError):
        space.check({k: v for k, v in point.items() if k != name})


# Unique names: "each parameter needs a unique name".
@given(range_parameters(name="x"), choice_parameters(name="x"))
def test_search_space_rejects_repeated_names(a, b):
    with pytest.raises(ValueError):
        SearchSpace([a, b])


# SearchSpace accepts any sequence (CHANGELOG 0.7.0), and round-trips through JSON (Result.run holds one).
@given(search_spaces())
def test_search_space_json_round_trip(space):
    assert SearchSpace(tuple(space.parameters)) == space
    assert SearchSpace.model_validate_json(space.model_dump_json()) == space
    assert SearchSpace.model_validate(space.model_dump(mode="json")) == space


@given(search_spaces(), st.data())
def test_search_space_getitem(space, data):
    name = data.draw(st.sampled_from(space.names))
    assert space[name].name == name
    with pytest.raises(KeyError):
        space["not_a_parameter_name_"]


# README: "A log range ... works with integer=True": every value drawn or mutated is an integer in range.
@settings(max_examples=50)
@given(range_parameters(integer=True, log=True, degenerate=False), seeds)
def test_log_integer_values(p, seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        value = p.sample(rng)
        assert isinstance(value, int) and in_range(p, value)
        mutated = p.mutate(value, rng, 0.1)
        assert isinstance(mutated, int) and in_range(p, mutated)
