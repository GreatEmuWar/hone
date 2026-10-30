import numpy as np
import pytest

from hone.benchmarks import (
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
)
from hone.benchmarks import problems as p
from hone.benchmarks.__main__ import main
from hone.optimizers import RandomSearch
from hone.config import OptimizerConfig


ALL = standard_problems()


def test_suite_covers_every_function_and_dimension():
    names = {problem.name for problem in ALL}
    assert len(ALL) == len(names) == 24
    assert {"sphere-2d", "ellipsoid-10d", "rastrigin-5d", "integer-sphere-10d", "material-selection-2d"} <= names


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
    return run_benchmark(quick_problems()[:2], seeds=range(3), budget_per_dimension=40)


OPTIMIZERS = ("genetic", "cma-es", "random")


def test_runs_every_optimizer_problem_and_seed(runs):
    assert len(runs) == 2 * 3 * 3
    assert {(r.problem, r.optimizer) for r in runs} == {
        (problem, optimizer) for problem in ("sphere-5d", "ellipsoid-5d") for optimizer in OPTIMIZERS
    }


def test_every_optimizer_gets_exactly_the_same_budget(runs):
    assert {r.evaluations for r in runs} == {40 * 5}


def test_optimizers_skip_problems_they_cannot_search():
    material = [problem for problem in quick_problems() if problem.name.startswith("material")]
    runs = run_benchmark(material, seeds=range(1), budget_per_dimension=20)
    assert {r.optimizer for r in runs} == {"genetic", "random"}
    assert "| material-selection-5d |" in to_markdown(summarize(runs))


def test_trace_is_non_increasing_and_ends_at_the_error(runs):
    for r in runs:
        errors = [error for _, error in r.trace]
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] == r.error
        assert r.trace[-1][0] == r.evaluations


def test_benchmark_is_reproducible():
    args = dict(problems=quick_problems()[:1], seeds=range(2), budget_per_dimension=20)
    assert run_benchmark(**args) == run_benchmark(**args)


def test_custom_optimizer_factory():
    def tiny(problem, budget, seed):
        return RandomSearch(problem.search_space, OptimizerConfig(population_size=5, generations=1, seed=seed))

    runs = run_benchmark(quick_problems()[:1], optimizers={"tiny": tiny}, seeds=range(2))
    assert {r.optimizer for r in runs} == {"tiny"}
    assert all(r.evaluations == 5 for r in runs)


# --- summaries ------------------------------------------------------------


def run(error, seed=0, problem="sphere-2d", optimizer="genetic"):
    return Run(problem, optimizer, seed, error, 10, [(10, error)])


def test_summary_statistics():
    summary = Summary.of([run(e, seed=i) for i, e in enumerate([0.0, 1e-6, 0.1, 1.0, 10.0])], tolerance=1e-4)
    assert summary.runs == 5
    assert summary.median_error == 0.1
    assert summary.lower_quartile == 1e-6
    assert summary.upper_quartile == 1.0
    assert summary.solved == 0.4


def test_summarize_groups_by_problem_and_optimizer(runs):
    summaries = summarize(runs)
    assert [(s.problem, s.optimizer, s.runs) for s in summaries] == [
        (problem, optimizer, 3) for problem in ("sphere-5d", "ellipsoid-5d") for optimizer in OPTIMIZERS
    ]


def test_markdown_table_marks_unsupported_problems():
    summaries = [Summary.of([run(0.5)], 1e-4), Summary.of([run(0.1, problem="material-2d")], 1e-4),
                 Summary.of([run(0.2, optimizer="cma-es")], 1e-4)]
    rows = {line.split(" | ")[0]: line for line in to_markdown(summaries).splitlines() if line.startswith("| ")}
    assert "n/a" in rows["| material-2d"]


def test_markdown_table(runs):
    table = to_markdown(summarize(runs))
    assert table.splitlines()[0] == (
        "| Problem | genetic median | genetic solved | cma-es median | cma-es solved | random median | random solved |"
    )
    assert "| sphere-5d |" in table and "| ellipsoid-5d |" in table


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


# --- command line ---------------------------------------------------------


def test_cli_checks_against_a_baseline(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    args = ["--quick", "--seeds", "1", "--budget-per-dimension", "20"]
    assert main([*args, "--save", str(baseline), "--markdown", str(tmp_path / "results.md")]) == 0
    results = (tmp_path / "results.md").read_text()
    assert results.startswith("# Benchmark results")
    assert "5 problems, 1 seeds, 20 evaluations per dimension" in results
    assert "| Problem |" in results
    assert main([*args, "--baseline", str(baseline)]) == 0
    assert "No regressions" in capsys.readouterr().out


def test_cli_fails_on_a_regression(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    good = [Summary(s.problem, s.optimizer, 1, 0.0, 0.0, 0.0, 1.0) for s in summarize(run_benchmark(quick_problems(), seeds=range(1), budget_per_dimension=20))]
    save_summaries(good, baseline)
    assert main(["--quick", "--seeds", "1", "--budget-per-dimension", "20", "--baseline", str(baseline)]) == 1
    assert "Regressions against the baseline" in capsys.readouterr().out
