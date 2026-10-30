import numpy as np
import pytest

from hone.benchmarks import (
    BUDGETS,
    Run,
    Summary,
    load_summaries,
    quick_problems,
    regressions,
    run_benchmark,
    save_summaries,
    standard_problems,
    summarize,
    to_markdown,
    wins,
)
from hone.benchmarks import problems as p
from hone.benchmarks.__main__ import main
from hone.optimizers import RandomSearch
from hone.config import OptimizerConfig


ALL = standard_problems()


def test_suite_covers_every_function_and_dimension():
    names = {problem.name for problem in ALL}
    assert len(ALL) == len(names) == 36
    assert {"sphere-2d", "ellipsoid-10d", "rastrigin-5d", "integer-sphere-10d", "material-selection-2d"} <= names
    assert {f"rotated-{f}-{d}d" for f in ("ellipsoid", "rosenbrock", "rastrigin", "ackley") for d in (2, 5, 10)} <= names
    assert "rotated-sphere-2d" not in names


def test_quick_problems_match_the_standard_ones():
    by_name = {problem.name: problem for problem in ALL}
    for problem in quick_problems():
        params = problem.search_space.sample(np.random.default_rng(0))
        assert problem.function(**params) == by_name[problem.name].function(**params)


@pytest.mark.parametrize("base", p.CONTINUOUS.values(), ids=p.CONTINUOUS.keys())
def test_base_functions_are_zero_at_the_origin_and_positive_elsewhere(base):
    assert base(np.zeros(4)) == pytest.approx(0, abs=1e-12)
    assert base(np.full(4, 0.5)) > 0


def optimum_of(problem):
    """The parameters at the problem's optimum, recovered from its shift."""
    names = problem.search_space.names
    if problem.name.startswith("material-selection"):
        rng = np.random.default_rng(p.zlib.crc32(problem.name.encode()))
        targets = {m: rng.uniform(-0.6 * p.BOUND, 0.6 * p.BOUND, size=problem.dimensions - 1) for m in p.MATERIALS}
        return {"material": "titanium", **{f"x{i}": v for i, v in enumerate(targets["titanium"])}}
    shift = p._shift(problem.name, problem.dimensions, integer=problem.name.startswith("integer"))
    if problem.name.startswith("mixed"):
        n_int = problem.dimensions // 2
        shift[problem.dimensions - n_int :] = np.round(shift[problem.dimensions - n_int :])
    return dict(zip(names, shift))


@pytest.mark.parametrize("problem", ALL, ids=lambda problem: problem.name)
def test_optimum_is_where_the_problem_says(problem):
    params = optimum_of(problem)
    assert problem.function(**params) == pytest.approx(problem.optimum, abs=1e-9)
    nudged = {k: (v + 0.3 if isinstance(v, float) else v) for k, v in params.items()}
    assert problem.function(**nudged) > problem.optimum


@pytest.mark.parametrize("problem", ALL, ids=lambda problem: problem.name)
def test_optimum_is_inside_the_bounds_and_off_centre(problem):
    values = [v for v in optimum_of(problem).values() if not isinstance(v, str)]
    assert all(-p.BOUND <= v <= p.BOUND for v in values)
    assert np.linalg.norm(values) > 0.5


def test_integer_problems_have_integer_optima():
    for problem in ALL:
        if problem.name.startswith("integer"):
            assert all(float(v).is_integer() for v in optimum_of(problem).values())


# --- runner ---------------------------------------------------------------


@pytest.fixture(scope="module")
def runs():
    return run_benchmark(quick_problems()[:2], seeds=range(3), budgets_per_dimension=[10, 40])


OPTIMIZERS = ("genetic", "cma-es", "random")


def test_runs_every_optimizer_problem_budget_and_seed(runs):
    assert len(runs) == 2 * 3 * 2 * 3
    assert {(r.problem, r.optimizer, r.budget_per_dimension) for r in runs} == {
        (problem, optimizer, budget)
        for problem in ("sphere-5d", "ellipsoid-5d")
        for optimizer in OPTIMIZERS
        for budget in (10, 40)
    }


def test_every_optimizer_gets_exactly_the_same_budget(runs):
    for r in runs:
        assert r.evaluations == r.budget_per_dimension * 5


def test_default_budgets_include_small_ones():
    assert BUDGETS[0] == 10 and max(BUDGETS) == 1000 and list(BUDGETS) == sorted(BUDGETS)


def test_each_budget_is_a_separate_run():
    """An optimizer can choose its settings from the budget it's given."""
    given = []

    def recording(problem, budget, seed):
        given.append(budget)
        return RandomSearch(problem.search_space, OptimizerConfig(population_size=5, max_evaluations=budget, seed=seed))

    run_benchmark(quick_problems()[:1], optimizers={"recording": recording}, seeds=range(1), budgets_per_dimension=[10, 20])
    assert given == [50, 100]


def test_more_evaluations_never_hurt_random_search(runs):
    """Random search with a larger budget sees the same first candidates, then more."""
    by_key = {(r.problem, r.seed, r.budget_per_dimension): r.error for r in runs if r.optimizer == "random"}
    for (problem, seed, budget), error in by_key.items():
        if budget == 40:
            assert error <= by_key[(problem, seed, 10)]


def test_optimizers_skip_problems_they_cannot_search():
    material = [problem for problem in quick_problems() if problem.name.startswith("material")]
    runs = run_benchmark(material, seeds=range(1), budgets_per_dimension=[20])
    assert {r.optimizer for r in runs} == {"genetic", "random"}
    assert "| material-selection-5d |" in to_markdown(summarize(runs))


def test_trace_is_non_increasing_and_ends_at_the_error(runs):
    for r in runs:
        errors = [error for _, error in r.trace]
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] == r.error
        assert r.trace[-1][0] == r.evaluations


def test_benchmark_is_reproducible():
    args = dict(problems=quick_problems()[:1], seeds=range(2), budgets_per_dimension=[20])
    assert run_benchmark(**args) == run_benchmark(**args)


def test_custom_optimizer_factory():
    def tiny(problem, budget, seed):
        return RandomSearch(problem.search_space, OptimizerConfig(population_size=5, generations=1, seed=seed))

    runs = run_benchmark(quick_problems()[:1], optimizers={"tiny": tiny}, seeds=range(2), budgets_per_dimension=[10])
    assert {r.optimizer for r in runs} == {"tiny"}
    assert all(r.evaluations == 5 for r in runs)


# --- summaries ------------------------------------------------------------


def run(error, seed=0, problem="sphere-2d", optimizer="genetic", budget=10):
    return Run(problem, optimizer, budget, seed, error, 2 * budget, [(2 * budget, error)])


def test_summary_statistics():
    summary = Summary.of([run(e, seed=i) for i, e in enumerate([0.0, 1e-6, 0.1, 1.0, 10.0])], tolerance=1e-4)
    assert summary.runs == 5
    assert summary.median_error == 0.1
    assert summary.lower_quartile == 1e-6
    assert summary.upper_quartile == 1.0
    assert summary.solved == 0.4


def test_summarize_groups_by_problem_optimizer_and_budget(runs):
    summaries = summarize(runs)
    assert [(s.problem, s.optimizer, s.budget_per_dimension, s.runs) for s in summaries] == [
        (problem, optimizer, budget, 3)
        for problem in ("sphere-5d", "ellipsoid-5d")
        for optimizer in OPTIMIZERS
        for budget in (10, 40)
    ]


def summary(error, **kwargs):
    return Summary.of([run(error, **kwargs)], 1e-4)


def test_wins_count_the_lowest_median_at_each_budget():
    summaries = [
        summary(1.0, optimizer="genetic"), summary(2.0, optimizer="random"),
        summary(5.0, optimizer="genetic", problem="ackley-2d"), summary(3.0, optimizer="random", problem="ackley-2d"),
        summary(1.0, optimizer="genetic", budget=100), summary(0.5, optimizer="random", budget=100),
    ]
    assert wins(summaries) == {10: {"genetic": 1, "random": 1}, 100: {"genetic": 0, "random": 1}}


def test_wins_count_ties_for_everyone_in_them():
    summaries = [summary(1e-12, optimizer="genetic"), summary(1e-10, optimizer="random"), summary(1.0, optimizer="cma-es")]
    assert wins(summaries) == {10: {"genetic": 1, "random": 1, "cma-es": 0}}


def test_wins_skip_problems_only_one_optimizer_ran_on():
    summaries = [summary(1.0, optimizer="genetic", problem="material-2d"), summary(1.0), summary(2.0, optimizer="random")]
    assert wins(summaries) == {10: {"genetic": 1, "random": 0}}


def test_markdown_table_marks_unsupported_problems():
    summaries = [Summary.of([run(0.5)], 1e-4), Summary.of([run(0.1, problem="material-2d")], 1e-4),
                 Summary.of([run(0.2, optimizer="cma-es")], 1e-4)]
    rows = {line.split(" | ")[0]: line for line in to_markdown(summaries).splitlines() if line.startswith("| ")}
    assert "n/a" in rows["| material-2d"]


def test_markdown_has_an_overview_then_a_table_per_budget(runs):
    lines = to_markdown(summarize(runs)).splitlines()
    headings = [line for line in lines if line.startswith("## ")]
    assert headings == ["## Overview", "## 10 evaluations per dimension", "## 40 evaluations per dimension"]
    assert "| Evaluations per dimension | genetic | cma-es | random |" in lines
    assert lines.count(
        "| Problem | genetic median | genetic solved | cma-es median | cma-es solved | random median | random solved |"
    ) == 2
    assert sum(line.startswith("| sphere-5d |") for line in lines) == 2


def test_summaries_round_trip(tmp_path, runs):
    summaries = summarize(runs)
    save_summaries(summaries, tmp_path / "baseline.json")
    assert load_summaries(tmp_path / "baseline.json") == summaries


def test_regressions_flag_large_increases_only():
    baseline = [Summary.of([run(0.01)], 1e-4), Summary.of([run(0.01, problem="rastrigin-2d")], 1e-4)]
    current = [Summary.of([run(0.05)], 1e-4), Summary.of([run(0.5, problem="rastrigin-2d")], 1e-4)]
    found = regressions(current, baseline)
    assert len(found) == 1 and "rastrigin-2d" in found[0]


def test_regressions_ignore_noise_near_zero():
    assert regressions([Summary.of([run(1e-9)], 1e-4)], [Summary.of([run(1e-14)], 1e-4)]) == []


def test_regressions_compare_like_budgets():
    baseline = [summary(1.0), summary(0.001, budget=100)]
    current = [summary(1.0), summary(0.5, budget=100)]
    found = regressions(current, baseline)
    assert found == ["genetic on sphere-2d at 100 evaluations per dimension: median error 0.5, baseline 0.001"]


# --- command line ---------------------------------------------------------


def test_cli_checks_against_a_baseline(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    args = ["--quick", "--seeds", "1", "--budgets", "20", "10"]
    assert main([*args, "--save", str(baseline), "--markdown", str(tmp_path / "results.md")]) == 0
    results = (tmp_path / "results.md").read_text()
    assert results.startswith("# Benchmark results")
    assert "6 problems, 1 seeds, budgets of 10 and 20 evaluations per dimension" in results
    assert "| Problem |" in results
    assert main([*args, "--baseline", str(baseline)]) == 0
    assert "No regressions" in capsys.readouterr().out


def test_cli_fails_on_a_regression(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    runs = run_benchmark(quick_problems(), seeds=range(1), budgets_per_dimension=[20])
    good = [Summary(s.problem, s.optimizer, 20, 1, 0.0, 0.0, 0.0, 1.0) for s in summarize(runs)]
    save_summaries(good, baseline)
    assert main(["--quick", "--seeds", "1", "--budgets", "20", "--baseline", str(baseline)]) == 1
    assert "Regressions against the baseline" in capsys.readouterr().out


# --- rotated problems -----------------------------------------------------


@pytest.mark.parametrize("dimensions", [2, 5, 10])
def test_rotations_are_proper_and_fixed(dimensions):
    rotation = p._rotation(f"rotated-rastrigin-{dimensions}d", dimensions)
    assert np.allclose(rotation @ rotation.T, np.eye(dimensions))
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.array_equal(rotation, p._rotation(f"rotated-rastrigin-{dimensions}d", dimensions))
    assert not np.allclose(rotation, p._rotation(f"rotated-ackley-{dimensions}d", dimensions))


def test_rotated_problems_are_not_separable():
    """Changing one variable changes how much another variable matters, unlike the plain version."""
    by_name = {problem.name: problem for problem in ALL}

    def interaction(problem):
        base = optimum_of(problem)
        def f(**changes):
            return problem.function(**{**base, **changes})
        x0, x1 = base["x0"], base["x1"]
        return f(x0=x0 + 1, x1=x1 + 1) - f(x0=x0 + 1) - f(x1=x1 + 1) + f()

    assert interaction(by_name["ellipsoid-5d"]) == pytest.approx(0, abs=1e-6)
    assert abs(interaction(by_name["rotated-ellipsoid-5d"])) > 1


def test_rotation_keeps_the_function_value_at_the_same_distance():
    """Rotating a sphere-like function changes nothing: a check that rotation preserves distances."""
    plain = p._vector_problem("check-5d", p.sphere, 5)
    turned = p._vector_problem("check-5d", p.sphere, 5, rotate=True)
    params = {f"x{i}": 0.1 * i for i in range(5)}
    assert turned.function(**params) == pytest.approx(plain.function(**params))
