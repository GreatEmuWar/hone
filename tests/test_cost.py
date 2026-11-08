"""The cost of a Bayesian candidate: the timing harness, the ceiling and the command line."""
import functools

import numpy as np
import pytest

from hone.benchmarks import cost
from hone.benchmarks.__main__ import main


def test_a_cell_times_its_candidates_after_the_cold_fit():
    cell = cost.measure_cell(20, 2, 2, dimensions=3, candidates=4)
    assert (cell.evaluations, cell.limits, cell.batch_size) == (20, 2, 2)
    assert cell.seconds_per_candidate > 0 and cell.cold_fit_seconds > 0


def test_a_cell_of_the_table_is_the_mean_over_seeds():
    (cell,) = cost.measure_cost((20,), (1,), (1,), dimensions=3, candidates=2, seeds=2)
    assert cell.seconds_per_candidate > 0 and cell.cold_fit_seconds > 0


def test_candidates_must_fill_whole_batches():
    with pytest.raises(ValueError, match="multiple of batch_size"):
        cost.measure_cell(20, 0, 8, candidates=12)


def test_every_default_batch_size_divides_the_candidates():
    assert all(cost.CANDIDATES % size == 0 for size in cost.BATCH_SIZES)


def test_the_problem_reports_one_output_per_limit_each_holding_on_most_of_the_box():
    space, objective = cost._problem(10, 4)
    assert len(space.output_constraints) == 4
    assert isinstance(cost._problem(10, 0)[1]({f"x{i}": 1.0 for i in range(10)}), float)
    rng = np.random.default_rng(1)
    points = [{f"x{i}": v for i, v in enumerate(rng.uniform(-5, 5, 10))} for _ in range(2000)]
    outputs = [objective(p) for p in points]
    for j in range(4):
        share = np.mean([o[f"g{j}"] <= 1.25 for o in outputs])
        assert 0.55 < share < 0.8, f"limit {j} holds on {share:.0%} of the box"


def _cell(n, k, b, seconds):
    return cost.CostCell(evaluations=n, limits=k, batch_size=b, seconds_per_candidate=seconds, cold_fit_seconds=1.0)


def test_the_ceiling_is_checked_at_500_evaluations_only():
    baseline = [_cell(200, 8, 1, 0.1), _cell(500, 8, 1, 0.2), _cell(500, 0, 8, 0.05)]
    within = [_cell(200, 8, 1, 1.0), _cell(500, 8, 1, 0.24), _cell(500, 0, 8, 0.05)]
    assert cost.ceiling_breaches(within, baseline) == []
    over = [_cell(500, 8, 1, 0.25), _cell(500, 0, 8, 0.05)]
    (found,) = cost.ceiling_breaches(over, baseline)
    assert found.startswith("500 evaluations, 8 limits, batches of 1") and "×1.25" in found


def test_a_cell_missing_from_the_baseline_is_not_a_breach():
    assert cost.ceiling_breaches([_cell(500, 2, 1, 9.0)], [_cell(500, 8, 1, 0.1)]) == []


def _table(profile=None):
    cells = [_cell(n, k, b, 0.01 * n / 50 * (k + 1)) for b in (1, 8) for n in (50, 500) for k in (0, 1, 8)]
    return cost.CostTable(hone="0.0", machine=cost.machine(), dimensions=10, candidates=24, cells=cells, profile=profile)


def test_a_table_saves_and_loads(tmp_path):
    table = _table(cost.CostProfile(problem="g07", evaluations=200, seconds=3.0, top=[("f", 2, 1.0, 1.5)]))
    cost.save_cost(table, tmp_path / "cost.json")
    assert cost.load_cost(tmp_path / "cost.json") == table


def test_the_markdown_has_a_table_per_batch_size_the_cold_fits_and_the_profile():
    text = cost.cost_markdown(_table(cost.CostProfile(problem="g07", evaluations=200, seconds=3.0, top=[("f", 2, 1.0, 1.5)])))
    assert "### Seconds per candidate, batches of 1" in text and "### Seconds per candidate, batches of 8" in text
    assert "| Evaluations | 0 limits | 1 limit | 8 limits |" in text
    assert "| 500 | 0.100 | 0.200 | 0.900 |" in text
    assert "### Cold fit, seconds, batches of 1" in text
    assert "### Profile: g07, 200 evaluations" in text and "| `f` | 2 | 1.00 | 1.50 |" in text


def test_the_profile_lists_the_costliest_functions_first():
    profile = cost.profile_run(evaluations=25, top=5)
    assert profile.problem == "g07" and len(profile.top) == 5
    own = [row[2] for row in profile.top]
    assert own == sorted(own, reverse=True)


def test_the_command_line_writes_the_table_and_checks_the_ceiling(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cost, "measure_cost", functools.partial(cost.measure_cost, (10, 500), (0, 1), (1,), dimensions=2, candidates=2, seeds=1))
    monkeypatch.setattr(cost, "profile_run", functools.partial(cost.profile_run, evaluations=15, top=3))
    results, saved = tmp_path / "cost.md", tmp_path / "cost.json"
    assert main(["--cost", "--markdown", str(results), "--save", str(saved)]) == 0
    assert "# The cost of a Bayesian candidate" in results.read_text()
    table = cost.load_cost(saved)
    assert {(c.evaluations, c.limits) for c in table.cells} == {(10, 0), (10, 1), (500, 0), (500, 1)}

    fast = table.model_copy(update={"cells": [c.model_copy(update={"seconds_per_candidate": c.seconds_per_candidate / 100}) for c in table.cells]})
    cost.save_cost(fast, tmp_path / "fast.json")
    assert main(["--cost", "--baseline", str(tmp_path / "fast.json")]) == 1
    assert "Over the ceiling of ×1.2 at 500 evaluations" in capsys.readouterr().out
