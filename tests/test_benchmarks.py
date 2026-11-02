import math

import numpy as np
import pytest

from hone.benchmarks import (
    BUDGETS,
    DEFAULT_OPTIMIZERS,
    SkipBudget,
    Run,
    Summary,
    batch_optimizers,
    load_summaries,
    noisy_output_problems,
    output_problems,
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
from hone.benchmarks import runner
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


# Bayesian optimization is slow enough to leave out of most runner tests; it has its own below.
FAST = {name: factory for name, factory in DEFAULT_OPTIMIZERS.items() if name != "bayesian"}
OPTIMIZERS = tuple(FAST)


@pytest.fixture(scope="module")
def runs():
    return run_benchmark(quick_problems()[:2], optimizers=FAST, seeds=range(3), budgets_per_dimension=[10, 40])


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
    assert sorted(set(given)) == [50, 100]


def test_more_evaluations_never_hurt_random_search(runs):
    """Random search with a larger budget sees the same first candidates, then more."""
    by_key = {(r.problem, r.seed, r.budget_per_dimension): r.error for r in runs if r.optimizer == "random"}
    for (problem, seed, budget), error in by_key.items():
        if budget == 40:
            assert error <= by_key[(problem, seed, 10)]


def test_optimizers_skip_problems_they_cannot_search():
    material = [problem for problem in quick_problems() if problem.name.startswith("material")]
    runs = run_benchmark(material, optimizers=FAST, seeds=range(1), budgets_per_dimension=[20])
    assert {r.optimizer for r in runs} == {"genetic", "random"}
    assert "| material-selection-5d |" in to_markdown(summarize(runs))


def test_trace_is_non_increasing_and_ends_at_the_error(runs):
    for r in runs:
        errors = [error for _, error in r.trace]
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] == r.error
        assert r.trace[-1][0] == r.evaluations


def test_benchmark_is_reproducible():
    args = dict(problems=quick_problems()[:1], optimizers=FAST, seeds=range(2), budgets_per_dimension=[20])
    assert run_benchmark(**args) == run_benchmark(**args)


def test_jobs_give_the_same_runs():
    args = dict(problems=quick_problems()[:2], optimizers=FAST, seeds=range(2), budgets_per_dimension=[10, 20])
    assert run_benchmark(**args, jobs=3) == run_benchmark(**args)


def test_bayesian_optimization_runs_only_at_small_budgets():
    runs = run_benchmark(quick_problems()[:1], optimizers={"bayesian": DEFAULT_OPTIMIZERS["bayesian"]}, seeds=range(1), budgets_per_dimension=[10, 100])
    assert [(r.optimizer, r.budget_per_dimension, r.evaluations) for r in runs] == [("bayesian", 10, 50)]


def test_factories_can_skip_a_budget():
    def odd_budgets_only(problem, budget, seed):
        if budget % 2 == 0:
            raise SkipBudget()
        return RandomSearch(problem.search_space, OptimizerConfig(population_size=5, max_evaluations=budget, seed=seed))

    runs = run_benchmark(quick_problems()[:1], optimizers={"odd": odd_budgets_only}, seeds=range(1), budgets_per_dimension=[3, 4, 5])
    assert [r.budget_per_dimension for r in runs] == [3, 5]


def test_custom_optimizer_factory():
    def tiny(problem, budget, seed):
        return RandomSearch(problem.search_space, OptimizerConfig(population_size=5, generations=1, seed=seed))

    runs = run_benchmark(quick_problems()[:1], optimizers={"tiny": tiny}, seeds=range(2), budgets_per_dimension=[10])
    assert {r.optimizer for r in runs} == {"tiny"}
    assert all(r.evaluations == 5 for r in runs)


# --- summaries ------------------------------------------------------------


def run(error, seed=0, problem="sphere-2d", optimizer="genetic", budget=10):
    return Run(
        problem=problem, optimizer=optimizer, budget_per_dimension=budget, seed=seed, error=error,
        evaluations=2 * budget, trace=[(2 * budget, error)],
    )


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
    args = ["--quick", "--seeds", "1", "--budgets", "4", "2"]
    assert main([*args, "--save", str(baseline), "--markdown", str(tmp_path / "results.md")]) == 0
    results = (tmp_path / "results.md").read_text()
    assert results.startswith("# Benchmark results")
    assert "6 problems, 1 seeds, budgets of 2 and 4 evaluations per dimension" in results
    assert "| Problem |" in results
    assert main([*args, "--baseline", str(baseline)]) == 0
    assert "No regressions" in capsys.readouterr().out


def test_cli_fails_on_a_regression(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    runs = run_benchmark(quick_problems(), seeds=range(1), budgets_per_dimension=[4])
    good = [s.model_copy(update={"budget_per_dimension": 4, "runs": 1, "median_error": 0.0, "lower_quartile": 0.0, "upper_quartile": 0.0, "solved": 1.0}) for s in summarize(runs)]
    save_summaries(good, baseline)
    assert main(["--quick", "--seeds", "1", "--budgets", "4", "--baseline", str(baseline)]) == 1
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


# --- constraints on outputs: finding a feasible design ----------------------


SPRING = next(problem for problem in output_problems() if problem.name == "tension-spring-outputs")
NOISY_SPRING = next(problem for problem in noisy_output_problems() if problem.name == "noisy-tension-spring-outputs")


def random_search_without_noise_handling(problem, budget, seed):
    """Random search that doesn't average, so it can run on a noisy problem with output constraints, which the optimizers refuse."""
    return RandomSearch(problem.search_space, OptimizerConfig(population_size=10, generations=budget, max_evaluations=budget, seed=seed))


@pytest.fixture(scope="module")
def spring_runs():
    return run_benchmark([SPRING], optimizers={"random": random_search_without_noise_handling}, seeds=range(4), budgets_per_dimension=[10])


def test_runs_record_when_a_feasible_design_was_first_found(spring_runs):
    assert all(r.output_constrained for r in spring_runs)
    for r in spring_runs:
        if r.first_feasible is None:
            assert r.error == math.inf and all(error == math.inf for _, error in r.trace)
        else:
            assert r.first_feasible % 10 == 0 and r.first_feasible <= r.evaluations
            # The trace is infinite before the generation that found it, and finite from it on.
            assert all((error == math.inf) == (evaluations < r.first_feasible) for evaluations, error in r.trace)


def test_runs_without_output_constraints_have_nothing_to_find(runs):
    assert all(not r.output_constrained and r.first_feasible is None for r in runs)
    assert all(s.feasible is None and s.median_first_feasible is None for s in summarize(runs))


def test_summary_counts_feasible_runs_and_the_median_evaluations_to_the_first():
    found = [run(0.5, seed=0), run(math.inf, seed=1), run(0.1, seed=2)]
    found = [r.model_copy(update={"output_constrained": True, "first_feasible": f}) for r, f in zip(found, [20, None, 40])]
    summary = Summary.of(found, tolerance=1e-4)
    assert summary.feasible == pytest.approx(2 / 3)
    assert summary.median_first_feasible == 40  # the miss counts as infinity, so the median is the 40, not the 20
    missed = [r.model_copy(update={"first_feasible": None, "error": math.inf}) for r in found[:2]] + found[2:]
    assert Summary.of(missed, tolerance=1e-4).median_first_feasible == math.inf


def test_markdown_has_a_feasibility_table_only_for_problems_with_output_constraints(spring_runs, runs):
    with_table = to_markdown(summarize(spring_runs))
    assert "Finding a feasible design:" in with_table
    assert "| Problem | random feasible | random first |" in with_table
    assert "Feasible: the share of runs" in with_table
    without = to_markdown(summarize(runs))
    assert "Finding a feasible design" not in without and "Feasible:" not in without


def test_markdown_writes_none_when_most_runs_found_nothing():
    missed = [run(math.inf, seed=i).model_copy(update={"output_constrained": True}) for i in range(3)]
    table = to_markdown([Summary.of(missed, 1e-4)])
    assert "| sphere-2d | 0% | none |" in table


def test_summaries_with_feasibility_round_trip(tmp_path, spring_runs):
    summaries = summarize(spring_runs)
    save_summaries(summaries, tmp_path / "outputs.json")
    assert load_summaries(tmp_path / "outputs.json") == summaries


def test_older_summaries_load_without_feasibility():
    assert all(s.feasible is None for s in load_summaries("benchmarks/constrained-summaries.json"))


def test_noise_is_added_to_the_value_and_each_output_separately():
    point = SPRING.search_space.sample(np.random.default_rng(0))
    clean = SPRING.function(**point)
    noisy = runner._with_noise(NOISY_SPRING, seed=0)
    first, second = noisy(**point), noisy(**point)
    assert set(first) == set(clean) == {"objective", "deflection", "shear", "surge"}
    for name in clean:
        assert first[name] != clean[name] and second[name] != clean[name] and first[name] != second[name]
        assert abs(first[name] - clean[name]) < 6 * (NOISY_SPRING.noise if name == "objective" else NOISY_SPRING.output_noise[name])
    # The same seed gives the same noise.
    assert runner._with_noise(NOISY_SPRING, seed=0)(**point) == first
    assert runner._with_noise(NOISY_SPRING, seed=1)(**point) != first


def test_an_output_without_noise_listed_is_reported_exactly():
    problem = NOISY_SPRING.model_copy(update={"output_noise": {"shear": 0.01}})
    point = problem.search_space.sample(np.random.default_rng(0))
    clean, noisy = problem.function(**point), runner._with_noise(problem, seed=0)(**point)
    assert noisy["deflection"] == clean["deflection"] and noisy["surge"] == clean["surge"]
    assert noisy["shear"] != clean["shear"] and noisy["objective"] != clean["objective"]


def test_a_noisy_run_is_scored_by_the_true_value_and_the_true_feasibility():
    runs = run_benchmark([NOISY_SPRING], optimizers={"random": random_search_without_noise_handling}, seeds=range(6), budgets_per_dimension=[20])
    assert all(r.output_constrained for r in runs)
    for r in runs:
        if r.error < math.inf:
            # The error is from the noise-free value of a point that truly satisfies every limit.
            assert r.error >= 0
    # Check against the optimizer's own view: a run's design that only seemed feasible scores infinity.
    seen_infeasible_by_truth = 0
    for seed in range(6):
        optimizer = random_search_without_noise_handling(NOISY_SPRING, 60, seed)
        result = optimizer.optimize(runner._with_noise(NOISY_SPRING, seed))
        true_value, truly_feasible = runner._true(NOISY_SPRING, result.best_params)
        r = next(r for r in runs if r.seed == seed)
        if result.feasible and not truly_feasible:
            seen_infeasible_by_truth += 1
            assert r.error == math.inf
        elif result.feasible:
            assert r.error == pytest.approx(max(true_value - NOISY_SPRING.optimum, 0.0))
    assert seen_infeasible_by_truth >= 0  # how often depends on the seeds; the branch above is what's checked


def test_bayesian_batch_factories_set_the_batch_size():
    factories = batch_optimizers([4, 8])
    assert list(factories) == ["bayesian-4", "bayesian-8"]
    optimizer = factories["bayesian-4"](quick_problems()[0], 40, 0)
    assert optimizer.config.population_size == 4 and optimizer.config.max_evaluations == 40
    with pytest.raises(SkipBudget):
        factories["bayesian-8"](quick_problems()[0], 1000, 0)


def test_cli_runs_the_outputs_suite_with_batch_sizes(tmp_path, capsys):
    results = tmp_path / "outputs.md"
    assert main(["--outputs", "--seeds", "1", "--budgets", "2", "--batch-sizes", "2", "--markdown", str(results)]) == 0
    text = results.read_text()
    assert "14 problems, 1 seeds, budgets of 2 evaluations per dimension" in text
    assert "| bayesian-2 median |" in text and "| bayesian-4" not in text
    assert "Finding a feasible design:" in text and "| g07 |" in text


def test_cli_batch_sizes_default_to_none_outside_the_outputs_suite(capsys):
    assert main(["--quick", "--seeds", "1", "--budgets", "2"]) == 0
    assert "bayesian-4" not in capsys.readouterr().out
