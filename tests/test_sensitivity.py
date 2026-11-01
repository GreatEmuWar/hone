"""Sensitivity analysis: Morris screening and Sobol indices, checked against published values."""
import itertools
import math

import numpy as np
import pytest
from scipy.stats import sobol_indices

from hone import (
    ChoiceParameter,
    EvaluationFailed,
    FixedParameter,
    RangeParameter,
    SearchSpace,
    SensitivityReport,
    UnsupportedSearchSpace,
    sensitivity,
)
from hone.sensitivity_analysis import _point, _sobol_design, _unrank_combination

import objectives

PI = math.pi
ISHIGAMI_SPACE = [RangeParameter(f"x{i}", -PI, PI) for i in (1, 2, 3)]
# Ishigami and Homma (1990), with a = 7 and b = 0.1; the indices are known exactly.
ISHIGAMI_FIRST = {"x1": 0.3139, "x2": 0.4424, "x3": 0.0}
ISHIGAMI_TOTAL = {"x1": 0.5576, "x2": 0.4424, "x3": 0.2437}

G_A = [0, 1, 4.5, 9, 99, 99, 99, 99]  # Saltelli and Sobol (1995): the smaller a, the more x matters
G_SPACE = [RangeParameter(f"x{i}", 0, 1) for i in range(len(G_A))]


def ishigami(x1, x2, x3):
    return math.sin(x1) + 7 * math.sin(x2) ** 2 + 0.1 * x3**4 * math.sin(x1)


def g_function(**x):
    return math.prod((abs(4 * x[f"x{i}"] - 2) + a) / (1 + a) for i, a in enumerate(G_A))


def g_indices():
    """The G-function's exact first and total indices."""
    partial = [1 / (3 * (1 + a) ** 2) for a in G_A]
    variance = math.prod(1 + v for v in partial) - 1
    first = [v / variance for v in partial]
    total = [v * math.prod(1 + w for j, w in enumerate(partial) if j != i) / variance for i, v in enumerate(partial)]
    return first, total


def linear(x, y, z):
    return 3 * x - 2 * y


LINEAR_SPACE = [RangeParameter("x", 0, 1), RangeParameter("y", 0, 10), RangeParameter("z", -1, 1)]


# --- against published values ------------------------------------------------------


def test_sobol_indices_of_the_ishigami_function():
    report = sensitivity(ishigami, ISHIGAMI_SPACE, max_evaluations=1024 * 5, seed=0)
    assert report.method == "sobol" and report.evaluations == 1024 * 5
    for name in ISHIGAMI_TOTAL:
        assert report[name].first == pytest.approx(ISHIGAMI_FIRST[name], abs=0.05)
        assert report[name].total == pytest.approx(ISHIGAMI_TOTAL[name], abs=0.05)
        interval = report[name].total_interval
        assert interval.low <= ISHIGAMI_TOTAL[name] <= interval.high
    assert report.ranking == ["x1", "x2", "x3"]


def test_sobol_indices_of_the_g_function():
    first, total = g_indices()
    report = sensitivity(g_function, G_SPACE, max_evaluations=1024 * 10, seed=0)
    for i in range(len(G_A)):
        assert report[f"x{i}"].first == pytest.approx(first[i], abs=0.05)
        assert report[f"x{i}"].total == pytest.approx(total[i], abs=0.05)
    assert report.ranking[:4] == ["x0", "x1", "x2", "x3"]


def test_morris_ranks_the_g_function_parameters():
    report = sensitivity(g_function, G_SPACE, max_evaluations=10 * 9, method="morris", seed=0)
    assert report.method == "morris"
    assert report.ranking[:4] == ["x0", "x1", "x2", "x3"]
    # The parameters with a = 99 barely matter: under a tenth of the most important's effect.
    assert all(report[f"x{i}"].mu_star < 0.1 * report["x0"].mu_star for i in range(4, 8))


def test_sobol_matches_scipy_on_the_same_evaluations():
    n, k = 256, 3
    report = sensitivity(ishigami, ISHIGAMI_SPACE, max_evaluations=n * (k + 2), seed=4)
    # The same points, drawn as sensitivity() draws them, then given to scipy's estimator.
    units = _sobol_design(n, k, np.random.default_rng(4))
    space = SearchSpace(ISHIGAMI_SPACE)
    values = np.array([ishigami(**_point(space, ISHIGAMI_SPACE, row)) for row in units])
    expected = sobol_indices(func={"f_A": values[:n], "f_B": values[n:2 * n], "f_AB": values[2 * n:].reshape(k, 1, n)}, n=n)
    for i, name in enumerate(["x1", "x2", "x3"]):
        assert report[name].first == pytest.approx(expected.first_order[i], rel=1e-9, abs=1e-12)
        assert report[name].total == pytest.approx(expected.total_order[i], rel=1e-9, abs=1e-12)


# --- what an effect means ----------------------------------------------------------


def test_morris_effects_are_changes_across_the_whole_range():
    report = sensitivity(linear, LINEAR_SPACE, max_evaluations=40, seed=0)
    assert report.method == "morris"
    # x spans 1, so 3x changes by 3 across it; y spans 10, so -2y changes by 20.
    assert report["x"].mu_star == pytest.approx(3) and report["x"].mu == pytest.approx(3)
    assert report["y"].mu_star == pytest.approx(20) and report["y"].mu == pytest.approx(-20)
    assert report["z"].mu_star == pytest.approx(0, abs=1e-9)
    assert report["x"].sigma == pytest.approx(0, abs=1e-9)
    assert report.ranking == ["y", "x", "z"]


def test_a_log_range_is_measured_in_its_logarithm():
    def decades(rate):
        return math.log10(rate)

    report = sensitivity(decades, [RangeParameter("rate", 1e-3, 1e3, log=True)], max_evaluations=20, method="morris", seed=0)
    assert report["rate"].mu_star == pytest.approx(6) and report["rate"].sigma == pytest.approx(0, abs=1e-9)


def test_integer_and_choice_parameters():
    def objective(n, material, width):
        return {"steel": 10.0, "aluminium": 4.0, "carbon": 1.0}[material] + 0 * n + 0 * width

    space = [RangeParameter("n", 1, 5, integer=True), ChoiceParameter("material", ["steel", "aluminium", "carbon"]), RangeParameter("width", 0, 1)]
    seen = []

    def recording(n, material, width):
        seen.append((n, material))
        return objective(n, material, width)

    report = sensitivity(recording, space, max_evaluations=1000, seed=0)
    assert report.ranking[0] == "material"
    assert report["material"].total == pytest.approx(1, abs=0.05)
    assert report["n"].total == pytest.approx(0, abs=1e-9) and report["width"].total == pytest.approx(0, abs=1e-9)
    assert all(isinstance(n, int) and 1 <= n <= 5 for n, _ in seen)
    assert {material for _, material in seen} == {"steel", "aluminium", "carbon"}


def test_choosing_several_choices_at_once():
    seen = []

    def objective(tools, speed):
        seen.append(tools)
        return len(set(tools) & {"drill", "saw"}) + 0 * speed

    space = [ChoiceParameter("tools", ["drill", "saw", "file", "rasp"], choose_n=2), RangeParameter("speed", 0, 1)]
    report = sensitivity(objective, space, max_evaluations=400, seed=0)
    assert report.ranking == ["tools", "speed"]
    assert all(len(tools) == 2 and len(set(tools)) == 2 for tools in seen)
    assert len({tuple(tools) for tools in seen}) == math.comb(4, 2)


def test_combinations_are_unranked_in_order():
    for n, k in [(5, 2), (6, 3), (4, 4), (3, 1)]:
        found = [tuple(_unrank_combination(rank, n, k)) for rank in range(math.comb(n, k))]
        assert found == list(itertools.combinations(range(n), k))


def test_fixed_parameters_are_passed_but_not_measured():
    def objective(x, mode):
        assert mode == "fast"
        return x

    report = sensitivity(objective, [RangeParameter("x", 0, 1), FixedParameter("mode", "fast")], max_evaluations=20, seed=0)
    assert report.ranking == ["x"]


def test_conditional_parameters_are_passed_only_while_switched_on():
    model = ChoiceParameter("model", ["svm", "forest"])
    C = RangeParameter("C", 0, 1)
    trees = RangeParameter("trees", 10, 500, integer=True)
    space = SearchSpace([model, C, trees], conditions=[model["svm"] >> C, model["forest"] >> trees])

    def objective(model, C=None, trees=None):
        assert (C is None) != (trees is None)
        assert (model == "svm") == (C is not None)
        return 10 * C if model == "svm" else 0.0

    report = sensitivity(objective, space, max_evaluations=1000, seed=0)
    assert report["trees"].total == pytest.approx(0, abs=1e-9)
    assert report["C"].total > 0.3 and report["model"].total > 0.3
    assert report.best_params is not None and "C" in report.best_params or "trees" in report.best_params


def test_constraints_are_not_supported_yet():
    x, y = RangeParameter("x", 0, 1), RangeParameter("y", 0, 1)
    with pytest.raises(UnsupportedSearchSpace, match="constraints"):
        sensitivity(lambda x, y: x, SearchSpace([x, y], constraints=[x + y <= 1]))


# --- the budget ------------------------------------------------------------------------


@pytest.mark.parametrize("budget, method", [(30, "morris"), (319, "morris"), (320, "sobol"), (1000, "sobol")])
def test_the_method_is_picked_from_the_budget(budget, method):
    report = sensitivity(linear, LINEAR_SPACE, max_evaluations=budget, seed=0)
    assert report.method == method
    assert report.evaluations <= budget


def test_morris_uses_whole_trajectories_within_the_budget():
    report = sensitivity(linear, LINEAR_SPACE, max_evaluations=43, method="morris", seed=0)
    assert report.evaluations == 40  # ten trajectories of four points


def test_sobol_uses_a_power_of_two_rows_within_the_budget():
    report = sensitivity(linear, LINEAR_SPACE, max_evaluations=1000, seed=0)
    assert report.evaluations == 128 * 5


@pytest.mark.parametrize("method, budget, message", [("morris", 7, "at least 8"), ("sobol", 79, "at least 80")])
def test_too_small_a_budget_says_how_much_is_needed(method, budget, message):
    with pytest.raises(ValueError, match=message):
        sensitivity(linear, LINEAR_SPACE, max_evaluations=budget, method=method)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"method": "fast"}, "method"),
        ({"workers": 0}, "workers"),
        ({"timeout": 0}, "timeout"),
        ({"confidence": 1}, "confidence"),
    ],
)
def test_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        sensitivity(linear, LINEAR_SPACE, **settings)


def test_every_parameter_fixed_leaves_nothing_to_measure():
    with pytest.raises(ValueError, match="fixed"):
        sensitivity(lambda mode: 0, [FixedParameter("mode", "fast")])


def test_a_seed_repeats_the_report():
    assert sensitivity(ishigami, ISHIGAMI_SPACE, seed=3) == sensitivity(ishigami, ISHIGAMI_SPACE, seed=3)
    assert sensitivity(ishigami, ISHIGAMI_SPACE, seed=3) != sensitivity(ishigami, ISHIGAMI_SPACE, seed=4)


# --- failures, workers and time limits ---------------------------------------------


@pytest.mark.parametrize("method, budget", [("morris", 200), ("sobol", 640)])
def test_failed_evaluations_are_left_out(method, budget):
    def fails_sometimes(x, y, z):
        if z > 0.5:
            raise EvaluationFailed("no convergence")
        if z < -0.5:
            return math.nan
        return linear(x, y, z)

    report = sensitivity(fails_sometimes, LINEAR_SPACE, max_evaluations=budget, method=method, seed=0)
    assert 0 < report.failures < report.evaluations
    assert report.ranking == ["y", "x", "z"]
    if method == "morris":
        assert report["x"].mu_star == pytest.approx(3) and report["y"].mu_star == pytest.approx(20)


def test_every_evaluation_failing():
    def always_fails(x, y, z):
        raise EvaluationFailed

    report = sensitivity(always_fails, LINEAR_SPACE, max_evaluations=100, seed=0)
    assert report.failures == report.evaluations
    assert report.best_params is None and report.best_value is None
    assert all(math.isnan(parameter.importance) for parameter in report.parameters)


def test_other_exceptions_stop_the_analysis():
    def buggy(x, y, z):
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        sensitivity(buggy, LINEAR_SPACE)


def test_workers_give_the_same_report():
    serial = sensitivity(objectives.linear_sensitivity, LINEAR_SPACE, max_evaluations=40, seed=0)
    assert sensitivity(objectives.linear_sensitivity, LINEAR_SPACE, max_evaluations=40, seed=0, workers=3) == serial


def test_evaluations_past_the_time_limit_count_as_failed():
    timed = sensitivity(objectives.linear_hangs_right_of_zero, LINEAR_SPACE, max_evaluations=20, seed=0, workers=4, timeout=0.5)
    failing = sensitivity(objectives.linear_fails_right_of_zero, LINEAR_SPACE, max_evaluations=20, seed=0)
    assert timed == failing and timed.failures > 0


# --- the report ------------------------------------------------------------------------


@pytest.mark.parametrize("maximize", [False, True])
def test_the_best_point_evaluated(maximize):
    report = sensitivity(linear, LINEAR_SPACE, max_evaluations=640, seed=0, maximize=maximize)
    x, y = report.best_params["x"], report.best_params["y"]
    assert report.best_value == pytest.approx(linear(x, y, 0))
    assert (report.best_value > 2.5) if maximize else (report.best_value < -19)


@pytest.mark.parametrize("budget, columns", [(40, "mu_star"), (640, "total")])
def test_the_report_prints_as_a_table(budget, columns):
    report = sensitivity(linear, LINEAR_SPACE, max_evaluations=budget, seed=0)
    text = str(report)
    assert columns in text and all(name in text for name in ["x", "y", "z"])
    assert text.splitlines()[2].startswith("y")


def test_the_report_round_trips_through_json():
    report = sensitivity(ishigami, ISHIGAMI_SPACE, seed=0)
    assert SensitivityReport.model_validate_json(report.model_dump_json()) == report


def test_unknown_names_raise_key_error():
    with pytest.raises(KeyError):
        sensitivity(linear, LINEAR_SPACE, max_evaluations=40)["w"]
