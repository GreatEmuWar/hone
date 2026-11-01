"""Property tests for sensitivity analysis, from the sensitivity() docstrings and docs/design/sensitivity.md."""
import math
import os

import numpy as np
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from hone import FixedParameter, RangeParameter, SearchSpace, sensitivity

from pbt_strategies import range_parameters, search_spaces

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
methods = st.sampled_from(["morris", "sobol"])


def budget_for(method, space):
    measured = sum(not isinstance(p, FixedParameter) for p in space.parameters)
    return 4 * (measured + 1) if method == "morris" else 16 * (measured + 2)


# Every point is a valid point of the space, within the budget, and each parameter that isn't
# fixed is measured exactly once.
@run_settings
@given(search_spaces(), methods, st.integers(0, 3), st.integers(0, 2**32 - 1))
def test_points_are_valid_and_within_budget(space, method, extra, seed):
    assume(any(not isinstance(p, FixedParameter) for p in space.parameters))
    calls = []

    def objective(**params):
        calls.append(params)
        return float(len(repr(params)))

    budget = budget_for(method, space) + extra
    report = sensitivity(objective, space, max_evaluations=budget, method=method, seed=seed)
    assert len(calls) == report.evaluations <= budget
    assert all(space.check(point) == point for point in calls)
    assert sorted(report.ranking) == sorted(p.name for p in space.parameters if not isinstance(p, FixedParameter))


# A parameter the objective ignores has no effect: mu_star, and both Sobol indices, are exactly 0.
@run_settings
@given(search_spaces(min_size=2), methods, st.integers(0, 2**32 - 1))
def test_an_ignored_parameter_has_no_effect(space, method, seed):
    measured = [p for p in space.parameters if not isinstance(p, FixedParameter)]
    assume(len(measured) >= 1)
    ignored = measured[-1].name

    def objective(**params):
        return float(sum(len(repr(v)) * (i + 1) for i, (name, v) in enumerate(params.items()) if name != ignored))

    report = sensitivity(objective, space, max_evaluations=budget_for(method, space), method=method, seed=seed)
    if method == "morris":
        assert report[ignored].mu_star == 0
    else:
        assert report[ignored].total == 0 and report[ignored].first == 0


# Morris effects are the change across the whole range on the search scale: for an objective that
# is linear in each range's scale, mu_star is exactly the coefficient times the range's span.
@run_settings
@given(
    st.lists(st.tuples(range_parameters(integer=False, degenerate=False), st.floats(-100, 100)), min_size=1, max_size=4, unique_by=lambda t: t[0].name),
    st.integers(0, 2**32 - 1),
)
def test_morris_measures_the_whole_range_on_its_scale(terms, seed):
    space = SearchSpace([parameter for parameter, _ in terms])

    def scaled(parameter, value):
        return math.log(value) if parameter.log else value

    def objective(**params):
        return sum(c * scaled(p, params[p.name]) for p, c in terms)

    report = sensitivity(objective, space, max_evaluations=4 * (len(terms) + 1), method="morris", seed=seed)
    for parameter, c in terms:
        span = scaled(parameter, parameter.maximum) - scaled(parameter, parameter.minimum)
        expected = abs(c) * span
        assert np.isclose(report[parameter.name].mu_star, expected, rtol=1e-6, atol=1e-6 * (1 + max(abs(c) * span for _, c in terms)))


# With a seed, the report repeats.
@run_settings
@given(search_spaces(), methods, st.integers(0, 2**32 - 1))
def test_a_seed_repeats_the_report(space, method, seed):
    assume(any(not isinstance(p, FixedParameter) for p in space.parameters))

    def objective(**params):
        return float(len(repr(params)))

    budget = budget_for(method, space)
    assert sensitivity(objective, space, max_evaluations=budget, method=method, seed=seed) == sensitivity(
        objective, space, max_evaluations=budget, method=method, seed=seed
    )


def test_integer_ranges_are_measured_too():
    report = sensitivity(lambda n: float(n), [RangeParameter("n", 0, 3, integer=True)], max_evaluations=40, method="morris", seed=0)
    assert report["n"].mu_star > 0
