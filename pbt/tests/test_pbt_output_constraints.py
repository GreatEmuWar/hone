"""Property tests for constraints on outputs, from the Output and Optimizer docstrings and docs/design/output-constraints.md."""
import math
import operator
import os

from hypothesis import HealthCheck, given, settings, strategies as st

from hone import (
    NSGA2,
    CMAES,
    BayesianConfig,
    BayesianOptimizer,
    CMAESConfig,
    GeneticOptimizer,
    GeneticOptimizerConfig,
    NSGA2Config,
    OptimizerConfig,
    Output,
    RandomSearch,
    RangeParameter,
    SearchSpace,
)
from pbt_runs import seeds

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])

A = RangeParameter("a", -5, 5)
OUTPUTS = [Output("p"), Output("q")]
OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv]
reported = st.fixed_dictionaries({"p": st.floats(-50, 50), "q": st.floats(-50, 50)})


@st.composite
def expressions(draw, depth=3):
    """An expression of outputs, a parameter and numbers, with a function computing the same on plain numbers."""
    if depth == 0 or draw(st.booleans()):
        kind = draw(st.sampled_from(["output", "parameter", "number"]))
        if kind == "output":
            output = draw(st.sampled_from(OUTPUTS))
            return output, lambda point, outputs, name=output.name: outputs[name]
        if kind == "parameter":
            return A, lambda point, outputs: point["a"]
        number = draw(st.floats(-100, 100, allow_nan=False))
        return number, lambda point, outputs, number=number: number
    op = draw(st.sampled_from(OPERATORS))
    (left, f), (right, g) = draw(expressions(depth - 1)), draw(expressions(depth - 1))
    return op(left, right), lambda point, outputs: op(f(point, outputs), g(point, outputs))


@run_settings
@given(expressions(), expressions(), st.floats(-5, 5), reported)
def test_an_output_expression_judges_a_point_as_its_arithmetic_says(left, right, a, outputs):
    (left_expression, f), (right_expression, g) = left, right
    if isinstance(left_expression, float) and isinstance(right_expression, float):
        return  # two numbers compare in Python, not as a constraint
    constraint = left_expression <= right_expression
    try:
        lhs, rhs = f({"a": a}, outputs), g({"a": a}, outputs)
    except ZeroDivisionError:
        assert constraint.violation({"a": a}, outputs) == math.inf
        return
    if isinstance(lhs, complex) or isinstance(rhs, complex) or math.isnan(lhs - rhs):
        return
    satisfied = lhs - rhs <= 1e-9 * max(1.0, abs(rhs))
    assert constraint.satisfied({"a": a}, outputs) == satisfied
    assert (constraint.relative_violation({"a": a}, outputs) == 0) == satisfied


@run_settings
@given(st.lists(st.tuples(st.floats(-100, 100), st.floats(0, 10)), min_size=2, max_size=12), st.booleans())
def test_the_result_follows_debs_rule(outcomes, maximize):
    """Feasible beats infeasible; feasible points by value; infeasible ones by violation."""
    space = SearchSpace([A], [Output("p") <= 5])
    optimizer = RandomSearch(space, OptimizerConfig(population_size=len(outcomes), max_evaluations=len(outcomes), maximize=maximize, seed=0))
    for trial, (value, p) in zip(optimizer.ask(), outcomes):
        optimizer.tell(trial, {"objective": value, "p": p})
    result = optimizer.result()
    feasible = [value for value, p in outcomes if p <= 5]
    if feasible:
        assert result.feasible and result.best_fitness == (max(feasible) if maximize else min(feasible))
    else:
        assert not result.feasible and result.best_outputs == {"p": min(p for _, p in outcomes)}
    assert result.infeasible == len(outcomes) - len(feasible)


def _feasible_box(x, y):
    return {"objective": (x - 3) ** 2 + (y + 2) ** 2, "r": x * x + y * y}


BOX = SearchSpace([RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)], [Output("r") <= 4])


@run_settings
@given(seeds, st.sampled_from(["genetic", "cmaes", "random", "bayesian"]))
def test_a_feasible_point_is_kept_once_found(seed, kind):
    """Once any evaluation is feasible, the result is feasible: no infeasible point can displace it."""
    builders = {
        "genetic": lambda: GeneticOptimizer(BOX, GeneticOptimizerConfig(max_evaluations=60, seed=seed)),
        "cmaes": lambda: CMAES(BOX, CMAESConfig(max_evaluations=60, seed=seed)),
        "random": lambda: RandomSearch(BOX, OptimizerConfig(max_evaluations=60, seed=seed)),
        "bayesian": lambda: BayesianOptimizer(BOX, BayesianConfig(max_evaluations=10, seed=seed)),
    }
    optimizer = builders[kind]()
    seen = []
    while not optimizer.done:
        for trial in optimizer.ask():
            value = _feasible_box(**trial.params)
            seen.append(value)
            optimizer.tell(trial, value)
    feasible = [v["objective"] for v in seen if v["r"] <= 4 + 1e-9 * 4]
    result = optimizer.result()
    assert result.feasible == bool(feasible)
    if feasible:
        assert result.best_fitness == min(feasible)


@run_settings
@given(seeds)
def test_nsga2s_front_holds_only_feasible_points(seed):
    config = NSGA2Config(objectives={"f": "minimize", "g": "minimize"}, max_evaluations=80, seed=seed)

    def problem(x, y):
        return {"f": x, "g": y, "r": x * x + y * y}

    result = NSGA2(BOX, config).optimize(problem)
    assert all(point.outputs["r"] <= 4 + 1e-8 for point in result.front)
