"""Property tests for constraints, from the Constraint and SearchSpace docstrings and docs/design/constraints.md."""
import math
import operator
import os

from hypothesis import HealthCheck, given, settings, strategies as st

from hone import OptimizerConfig, RandomSearch, RangeParameter, SearchSpace
from pbt_runs import seeds

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])

PARAMETERS = [RangeParameter("a", -5, 5), RangeParameter("b", 0.5, 20, log=True), RangeParameter("c", -10, 10, integer=True)]
OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv]
values = st.fixed_dictionaries(
    {"a": st.floats(-5, 5), "b": st.floats(0.5, 20), "c": st.integers(-10, 10)}
)


@st.composite
def expressions(draw, depth=3):
    """An expression of the parameters, paired with a function computing the same thing on plain numbers."""
    if depth == 0 or draw(st.booleans()):
        if draw(st.booleans()):
            parameter = draw(st.sampled_from(PARAMETERS))
            return parameter, lambda point, name=parameter.name: point[name]
        number = draw(st.floats(-100, 100, allow_nan=False))
        return number, lambda point, number=number: number
    left, left_fn = draw(expressions(depth=depth - 1))
    right, right_fn = draw(expressions(depth=depth - 1))
    op = draw(st.sampled_from(OPERATORS))
    if not hasattr(left, "evaluate") and not hasattr(right, "evaluate"):
        right, right_fn = PARAMETERS[0], lambda point: point["a"]  # an expression needs a parameter somewhere
    return op(left, right), lambda point: op(left_fn(point), right_fn(point))


def plain(fn, point):
    try:
        return fn(point)
    except ZeroDivisionError:
        return math.nan


# Expressions use ordinary arithmetic: an expression's value is what the same arithmetic gives on the values.
@run_settings
@given(expressions(), values)
def test_an_expression_evaluates_like_python(expression, point):
    built, fn = expression
    expected = plain(fn, point)
    if not hasattr(built, "evaluate") or math.isnan(expected) or math.isinf(expected):
        return
    try:
        value = built.evaluate(point)
    except ZeroDivisionError:
        return
    assert math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9)


# "Expressions are pydantic models, so a search space with constraints still saves to JSON": a saved and loaded
# constraint is equal to the original and judges every point the same way.
@run_settings
@given(expressions(), st.sampled_from(["<=", ">="]), st.floats(-100, 100), values)
def test_constraints_save_and_load_unchanged(expression, op, limit, point):
    built, _ = expression
    if not hasattr(built, "evaluate"):
        return
    constraint = built <= limit if op == "<=" else built >= limit
    space = SearchSpace(PARAMETERS, [constraint])
    loaded = SearchSpace.model_validate_json(space.model_dump_json())
    assert loaded == space
    assert loaded.constraints[0].violation(point) == constraint.violation(point)


# A violation is never negative, and a point satisfies a constraint exactly when its violation is 0.
@run_settings
@given(expressions(), st.floats(-100, 100), values)
def test_violation_is_zero_exactly_when_satisfied(expression, limit, point):
    built, _ = expression
    if not hasattr(built, "evaluate"):
        return
    constraint = built <= limit
    violation = constraint.violation(point)
    assert violation >= 0 and constraint.satisfied(point) == (violation == 0)


@st.composite
def linear_constraints(draw):
    """Linear constraints that the centre of the space satisfies, so the space is never empty."""
    centre = {"a": 0.0, "b": math.sqrt(0.5 * 20), "c": 0}
    constraints = []
    for _ in range(draw(st.integers(1, 3))):
        weights = draw(st.lists(st.floats(-3, 3), min_size=3, max_size=3))
        expression = weights[0] * PARAMETERS[0] + weights[1] * PARAMETERS[1] + weights[2] * PARAMETERS[2]
        slack = draw(st.floats(0.5, 20))
        constraints.append(expression <= expression.evaluate(centre) + slack)
    return constraints


# "Every optimizer then only ever asks for points that satisfy every constraint": random search, for now.
@run_settings
@given(linear_constraints(), seeds)
def test_random_search_only_asks_for_feasible_points(constraints, seed):
    space = SearchSpace(PARAMETERS, constraints)
    optimizer = RandomSearch(space, OptimizerConfig(max_evaluations=60, seed=seed))
    while not optimizer.done:
        for trial in optimizer.ask():
            assert space.feasible(trial.params)
            optimizer.tell(trial, sum(float(v) for v in trial.params.values()))
