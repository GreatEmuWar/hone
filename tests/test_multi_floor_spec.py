import pytest

from hone.benchmarks.multi import MultiRun
from hone.benchmarks.sharded import expected_runs, load_spec

spec = load_spec("benchmarks/specs/multi_floor.py")
PROBLEM = "noisy-osy"


def _pair(seed, base_gap, paired_gap, budget=10, kind="random", reported=10, feasible=5):
    common = {"problem": PROBLEM, "budget_per_dimension": budget, "seed": seed, "gap": 0.0, "front_size": reported, "evaluations": 0, "front": []}
    base = MultiRun(optimizer=kind, true_gap=base_gap, reported=reported, truly_feasible=feasible, any_truly_feasible=feasible > 0, **common)
    paired = MultiRun(optimizer=kind + "@paired", true_gap=paired_gap, reported=reported, truly_feasible=feasible, any_truly_feasible=True, **common)
    return [base, paired]


def test_order_statistics():
    assert spec.order_statistics(30) == (10, 19)
    assert spec.order_statistics(5) == (0, 4)
    assert spec.order_statistics(4) is None
    low, high = spec.order_statistics(100)
    assert low + high == 99 and low > 30


def test_interval_uses_sorted_differences():
    assert spec.interval(list(range(30, 0, -1))) == (11, 20)


def test_report_verdicts_and_baseline_figures():
    t = spec._threshold(PROBLEM)
    runs = []
    for seed in range(30):  # resolves: tiny, steady differences; one baseline run has an empty front
        runs += _pair(seed, 0.1, 0.1 + 0.001 * (seed % 3), budget=10, reported=10 if seed else 0, feasible=5 if seed else 0)
    for seed in range(30):  # doesn't: differences spread well past the threshold
        runs += _pair(seed, 0.1, 0.1 + (seed - 15) * t, budget=20)
    rows = {r["budget"]: r for r in spec._cell_rows(runs)}
    assert rows[10]["resolves"] and not rows[20]["resolves"]
    assert rows[10]["n"] == 30
    assert rows[10]["low"] == pytest.approx(0.001) and rows[10]["high"] == pytest.approx(0.001)
    assert rows[20]["low"] == pytest.approx((10 - 15) * t) and rows[20]["high"] == pytest.approx((19 - 15) * t)
    base = rows[10]["base"]
    assert base["empty"] == 1 and base["any"] == 29
    assert base["gap"] == pytest.approx(0.1)
    assert base["pooled"] == pytest.approx(0.5)
    assert base["mean_share"] == pytest.approx(0.5)
    assert base["front"] == 10 and base["feasible"] == 5
    text = spec.report(runs)
    assert text.startswith("## The paired noise floor")
    assert "1 of 2 cells resolve δ." in text
    assert f"- can't resolve δ: {PROBLEM}, random, 20/dim" in text


def test_optimizers_and_problems():
    expected = {f"{kind}{suffix}" for kind in ("nsga2", "bayesian-parego", "bayesian-models", "random") for suffix in ("", "@paired")}
    assert set(spec.optimizers()) == expected
    assert {p.name for p in spec.problems()} == {"noisy-osy", "noisy-tnk", "noisy-cantilever-outputs"}


def test_expected_runs_skip_large_bayesian_budgets():
    runs = expected_runs(spec)
    assert runs
    osy = [r for r in runs if r[0] == PROBLEM]
    assert any(r[1] == "random" and r[2] == 100 for r in osy)
    assert not any(r[1].startswith("bayesian") and r[2] in (50, 100) for r in osy)
    assert any(r[1].startswith("bayesian") and r[2] == 10 for r in osy)
