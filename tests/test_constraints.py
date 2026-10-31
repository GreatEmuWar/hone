"""Constraints: building them from parameters, checking points against them, and keeping optimizers to them."""
import math

import numpy as np
import pytest
from pydantic import ValidationError

from hone import (
    CMAES,
    BayesianOptimizer,
    ChoiceParameter,
    Constraint,
    GeneticOptimizer,
    InfeasibleSearchSpace,
    OptimizerConfig,
    RandomSearch,
    RangeParameter,
    Result,
    SearchSpace,
    UnsupportedSearchSpace,
)

WIDTH = RangeParameter("width", 1, 10)
HEIGHT = RangeParameter("height", 1, 10)


def area(width, height):
    return -width * height  # maximize the area by minimizing its negative


def random_search(space, **config):
    return RandomSearch(space, OptimizerConfig(**{"max_evaluations": 200, "seed": 0, **config}))


# --- building constraints ----------------------------------------------------


def test_comparing_an_expression_makes_a_constraint():
    constraint = WIDTH * HEIGHT <= 40
    assert str(constraint) == "width * height <= 40"
    assert constraint.satisfied({"width": 5, "height": 8}) and not constraint.satisfied({"width": 5, "height": 8.1})
    assert constraint.violation({"width": 5, "height": 10}) == 10


@pytest.mark.parametrize(
    "build, expected, value",
    [
        (lambda w, h: w - (h - 1), "width - (height - 1)", 5.0),
        (lambda w, h: w - h - 1, "width - height - 1", 3.0),
        (lambda w, h: w / (h * 2), "width / (height * 2)", 1.5),
        (lambda w, h: (w**2) ** h, "(width ** 2) ** height", 1296.0),
        (lambda w, h: -(w + h), "-(width + height)", -8.0),
        (lambda w, h: abs(h - w) * 3, "abs(height - width) * 3", 12.0),
        (lambda w, h: 2 * w + 3, "2 * width + 3", 15.0),
        (lambda w, h: 1 / w, "1 / width", 1 / 6),
    ],
)
def test_expressions_print_and_evaluate_like_python(build, expected, value):
    expression = build(WIDTH, HEIGHT)
    assert str(expression) == expected
    assert expression.evaluate({"width": 6, "height": 2}) == pytest.approx(value)
    assert expression.evaluate({"width": 6, "height": 2}) == pytest.approx(build(6, 2))


def test_less_than_means_at_most():
    assert (WIDTH < 5).satisfied({"width": 5}) and (WIDTH > 5).satisfied({"width": 5})
    assert (5 >= WIDTH).satisfied({"width": 5}) and not (5 >= WIDTH).satisfied({"width": 6})


def test_a_chained_comparison_is_refused():
    with pytest.raises(TypeError, match="write two constraints"):
        0 <= WIDTH + HEIGHT <= 10  # noqa: B015


def test_choice_parameters_and_other_values_are_refused():
    with pytest.raises(TypeError, match="Constraint.function"):
        WIDTH + ChoiceParameter("material", ["steel", "titanium"])  # type: ignore[operator]
    with pytest.raises(TypeError):
        WIDTH <= "ten"


def test_a_point_that_cant_be_judged_is_infeasible():
    assert math.isinf((1 / (WIDTH - 5) <= 1).violation({"width": 5}))
    assert math.isinf(((WIDTH - 5) ** 0.5 <= 1).violation({"width": 4}))


def test_points_on_the_limit_satisfy_it_despite_rounding():
    assert (WIDTH * 0.1 * 3 <= 0.3 * WIDTH).satisfied({"width": 7})


# --- the search space ----------------------------------------------------------


def test_a_search_space_reports_the_constraints_a_point_breaks():
    space = SearchSpace([WIDTH, HEIGHT], [WIDTH * HEIGHT <= 40, WIDTH >= 2 * HEIGHT])
    assert space.feasible({"width": 8, "height": 4})
    assert [str(c) for c in space.broken({"width": 9, "height": 5})] == ["width * height <= 40", "width >= 2 * height"]


def test_constraints_must_use_the_search_spaces_range_parameters():
    depth = RangeParameter("depth", 0, 1)
    with pytest.raises(ValidationError, match="'depth', which isn't a parameter"):
        SearchSpace([WIDTH], [WIDTH + depth <= 5])
    with pytest.raises(ValidationError, match="isn't a range parameter"):
        SearchSpace([WIDTH, ChoiceParameter("height", [1, 2])], [WIDTH + HEIGHT <= 5])


def test_samples_satisfy_every_constraint():
    space = SearchSpace([WIDTH, HEIGHT], [WIDTH * HEIGHT <= 20, WIDTH >= 2 * HEIGHT])
    rng = np.random.default_rng(0)
    assert all(space.feasible(space.sample(rng)) for _ in range(200))


def test_without_constraints_sampling_is_unchanged():
    plain = SearchSpace([WIDTH, HEIGHT]).sample(np.random.default_rng(3))
    rng = np.random.default_rng(3)
    assert plain == {"width": WIDTH.sample(rng), "height": HEIGHT.sample(rng)}


def test_constraints_save_and_load_with_the_search_space():
    space = SearchSpace([WIDTH, HEIGHT], [WIDTH * HEIGHT <= 40, abs(WIDTH - HEIGHT) >= 1])
    loaded = SearchSpace.model_validate_json(space.model_dump_json())
    assert loaded == space
    assert loaded.broken({"width": 9, "height": 5}) == space.broken({"width": 9, "height": 5})


# --- function constraints ------------------------------------------------------


def test_a_function_constraint_takes_the_parameters_it_names():
    space = SearchSpace(
        [WIDTH, HEIGHT, ChoiceParameter("material", ["steel", "titanium"])],
        [Constraint.function(lambda width, material: width - (8 if material == "steel" else 4), name="stress")],
    )
    assert space.feasible({"width": 7, "height": 1, "material": "steel"})
    assert not space.feasible({"width": 7, "height": 1, "material": "titanium"})


def test_a_function_constraint_must_return_a_number():
    space = SearchSpace([WIDTH], [Constraint.function(lambda width: "fine", name="bad")])
    with pytest.raises(TypeError, match="must return a number"):
        space.feasible({"width": 2})


def test_a_function_constraint_needs_its_parameters_in_the_search_space():
    with pytest.raises(ValidationError, match="needs parameters missing from the search space: depth"):
        SearchSpace([WIDTH], [Constraint.function(lambda width, depth: width - depth, name="gap")])


def test_a_function_constraint_saves_only_its_name():
    space = SearchSpace([WIDTH], [Constraint.function(lambda width: width - 5, name="limit")])
    loaded = SearchSpace.model_validate_json(space.model_dump_json())
    with pytest.raises(ValueError, match="loaded without its function"):
        loaded.feasible({"width": 2})
    assert loaded.with_functions({"limit": lambda width: width - 5}).feasible({"width": 2})
    with pytest.raises(ValueError, match="needs its function"):
        loaded.with_functions({})


# --- optimizers ----------------------------------------------------------------


def test_random_search_only_asks_for_feasible_points():
    space = SearchSpace([WIDTH, HEIGHT], [WIDTH * HEIGHT <= 40, WIDTH >= 2 * HEIGHT])
    optimizer = random_search(space)
    while not optimizer.done:
        for trial in optimizer.ask():
            assert space.feasible(trial.params)
            optimizer.tell(trial, area(**trial.params))
    best = optimizer.result().best_params
    assert best["width"] * best["height"] <= 40 and best["width"] * best["height"] > 35


def test_initial_points_must_be_feasible():
    space = SearchSpace([WIDTH, HEIGHT], [WIDTH * HEIGHT <= 40])
    with pytest.raises(ValueError, match="breaks the constraint `width \\* height <= 40`"):
        RandomSearch(space, initial_points={"width": 10, "height": 10})


def test_constraints_that_leave_nothing_fail_before_any_evaluation():
    space = SearchSpace([WIDTH, HEIGHT], [WIDTH + HEIGHT <= 3, WIDTH >= 2 * HEIGHT])
    with pytest.raises(InfeasibleSearchSpace, match="contradict"):
        RandomSearch(space)


@pytest.mark.parametrize("optimizer_class", [GeneticOptimizer, CMAES, BayesianOptimizer])
def test_optimizers_without_support_refuse_constraints(optimizer_class):
    with pytest.raises(UnsupportedSearchSpace, match="doesn't support constraints yet"):
        optimizer_class(SearchSpace([WIDTH, HEIGHT], [WIDTH <= HEIGHT]))


def test_a_rebuilt_optimizer_repeats_a_constrained_run(tmp_path):
    space = SearchSpace([WIDTH, HEIGHT], [WIDTH * HEIGHT <= 40])
    result = random_search(space).optimize(area)
    result.save(tmp_path / "result.json")
    loaded = Result.load(tmp_path / "result.json")
    assert loaded.rebuild_optimizer().optimize(area) == result


def test_a_rebuilt_optimizer_needs_function_constraints_passed_again(tmp_path):
    def stress(width, height):
        return width * height - 40

    result = random_search(SearchSpace([WIDTH, HEIGHT], [Constraint.function(stress)])).optimize(area)
    result.save(tmp_path / "result.json")
    loaded = Result.load(tmp_path / "result.json")
    with pytest.raises(ValueError, match="needs its function"):
        loaded.rebuild_optimizer()
    assert loaded.rebuild_optimizer(functions={"stress": stress}).optimize(area) == result


def test_a_checkpoint_resumes_a_run_with_a_function_constraint(tmp_path):
    space = SearchSpace([WIDTH, HEIGHT], [Constraint.function(lambda width, height: width * height - 40, name="area")])
    whole = random_search(space, population_size=10).optimize(area)

    partial = random_search(space, population_size=10)
    for _ in range(5):
        for trial in partial.ask():
            partial.tell(trial, area(**trial.params))
    partial.save_checkpoint(tmp_path / "run.checkpoint")
    resumed = random_search(space, population_size=10)
    resumed.load_checkpoint(tmp_path / "run.checkpoint")
    while not resumed.done:
        for trial in resumed.ask():
            resumed.tell(trial, area(**trial.params))
    assert resumed.result() == whole
