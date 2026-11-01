"""Property tests for fronts and hypervolume, from the hone.pareto docstrings and docs/design/multiple-objectives.md."""
import os

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hone import (
    NSGA2,
    MultiObjectiveBayesian,
    MultiObjectiveBayesianConfig,
    MultiObjectiveConfig,
    MultiObjectiveOptimizer,
    MultiObjectiveRandomSearch,
    NSGA2Config,
    RangeParameter,
)
from hone.pareto import hypervolume, non_dominated

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])

finite = st.floats(-100, 100, allow_nan=False, allow_infinity=False)


def value_sets(dimensions=st.integers(2, 3)):
    return dimensions.flatmap(lambda m: arrays(np.float64, st.tuples(st.integers(1, 40), st.just(m)), elements=finite))


def strictly_dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


# No point on the front is beaten on every objective by another point on it.
@run_settings
@given(value_sets())
def test_the_front_is_mutually_non_dominated(values):
    front = values[non_dominated(values)]
    for i, a in enumerate(front):
        for j, b in enumerate(front):
            assert i == j or not strictly_dominates(a, b)


# Every point is on the front, tied with a point on it, or beaten by one.
@run_settings
@given(value_sets())
def test_every_point_is_on_the_front_or_covered_by_it(values):
    front = values[non_dominated(values)]
    for point in values:
        assert any(np.all(f <= point + 1e-9 * np.maximum(1, np.abs(point))) for f in front)


# Domination doesn't change under an increasing transformation of any objective, so neither does the front.
@run_settings
@given(st.integers(2, 3).flatmap(lambda m: arrays(np.float64, st.tuples(st.integers(1, 40), st.just(m)), elements=st.integers(-50, 50).map(float))), st.integers(0, 2))
def test_the_front_doesnt_change_under_an_increasing_rescaling(values, column):
    # Whole numbers, so no two values are near-ties that a rescaling could split or join.
    column = column % values.shape[1]
    rescaled = values.copy()
    rescaled[:, column] = values[:, column] ** 3 + 5 * values[:, column]
    assert non_dominated(values) == non_dominated(rescaled)


# A better front always has a larger hypervolume: adding a point never lowers it, and adding a dominated one doesn't change it.
@run_settings
@given(value_sets(), st.data())
def test_hypervolume_never_falls_when_a_point_is_added(values, data):
    reference = values.max(axis=0) + 1
    extra = data.draw(arrays(np.float64, values.shape[1], elements=finite))
    before = hypervolume(values, reference)
    after = hypervolume(np.vstack([values, extra]), reference)
    assert after >= before - 1e-9 * max(1.0, before)
    if any(np.all(v <= extra) for v in values):
        assert after == np.float64(before) or abs(after - before) <= 1e-9 * max(1.0, before)


# Hypervolume is the same whether computed from all points or from the front alone.
@run_settings
@given(value_sets())
def test_hypervolume_depends_only_on_the_front(values):
    reference = values.max(axis=0) + 1
    whole, front = hypervolume(values, reference), hypervolume(values[non_dominated(values)], reference)
    assert abs(whole - front) <= 1e-6 * max(1.0, whole)


# Every multi-objective optimizer's front is that of every point it evaluated, and its hypervolume never falls.
@run_settings
@given(
    st.integers(0, 2**32 - 1),
    st.sampled_from([("minimize", "minimize"), ("maximize", "minimize"), ("maximize", "maximize")]),
    st.sampled_from(["random", "nsga2", "bayesian"]),
)
def test_a_runs_front_is_that_of_everything_evaluated(seed, directions, kind):
    objectives = dict(zip(("f1", "f2"), directions))
    seen = []

    def objective(x, y):
        values = {"f1": x + 0.3 * np.sin(5 * y), "f2": (1 - x) ** 2 + y}
        seen.append(values)
        return values

    space = [RangeParameter("x", 0, 1), RangeParameter("y", 0, 1)]
    if kind == "random":
        optimizer: MultiObjectiveOptimizer = MultiObjectiveRandomSearch(
            space, MultiObjectiveConfig(objectives=objectives, max_evaluations=60, population_size=10, seed=seed)
        )
    elif kind == "nsga2":
        optimizer = NSGA2(space, NSGA2Config(objectives=objectives, max_evaluations=60, seed=seed))
    else:
        optimizer = MultiObjectiveBayesian(space, MultiObjectiveBayesianConfig(objectives=objectives, max_evaluations=12, seed=seed))
    result = optimizer.optimize(objective)
    signs = np.array([1 if d == "minimize" else -1 for d in directions])
    vectors = np.array([[v["f1"], v["f2"]] for v in seen]) * signs
    expected = sorted(tuple(vectors[i] * signs) for i in non_dominated(vectors))
    assert sorted((p.values["f1"], p.values["f2"]) for p in result.front) == expected
    volumes = [stats.hypervolume for stats in result.generations]
    assert all(b >= a for a, b in zip(volumes, volumes[1:]))
