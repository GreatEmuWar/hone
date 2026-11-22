from hone.benchmarks import output_problems
from hone.benchmarks.runner import Run
from hone.benchmarks.sharded import expected_runs, load_spec

spec = load_spec("benchmarks/specs/choose_outputs_timing.py")
FLAG = "Over about 10 minutes at 8 limits: goes to the user."


def _runs(slow=None, machine="Test CPU"):
    """5 runs of each problem, 100 s at 500 evaluations and 40 s at 300, except `slow` takes 700 s at 500."""
    return [
        Run(problem=p.name, optimizer="bayesian@500", budget_per_dimension=1, seed=seed, error=0.0, evaluations=500,
            trace=[], seconds=700.0 if p.name == slow else 100.0, seconds_at={300: 40.0}, machine=machine)
        for p in output_problems()
        for seed in range(5)
    ]


def test_the_spec_makes_14_problems_of_5_seeds():
    assert len(expected_runs(spec)) == 14 * 5
    assert spec.TIME_AT == (300,) and spec.BUDGETS == (1,)


def test_the_report_has_a_table_the_machines_and_no_flag_when_quick():
    text = spec.report(_runs())
    assert "| g07 | 10 | 8 | 40.0 s | 40.0 s | 100.0 s (1.7 min) | 100.0 s (1.7 min) |" in text
    assert "- Test CPU: 70 runs" in text
    assert "Wall time including the objective, which is close to zero on these problems; one run at a time per runner." in text
    assert "nearest-rank" in text
    assert FLAG not in text


def test_the_flag_is_raised_only_by_a_problem_with_8_limits_or_more():
    assert FLAG in spec.report(_runs(slow="g07"))
    assert FLAG not in spec.report(_runs(slow="g10"))  # 6 limits
