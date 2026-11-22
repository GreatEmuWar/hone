"""The cost of a multi-objective Bayesian candidate: the timing harness, the ceiling and the saved table."""
import json

import pytest

from hone.benchmarks import cost
from hone.optimizers import MultiObjectiveBayesian


@pytest.mark.parametrize("method", ["parego", "models"])
@pytest.mark.parametrize("limits", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_a_multi_cell_times_its_candidates_after_the_cold_fit(method, limits, batch_size):
    cell = cost.measure_multi_cell(method, 8, limits, batch_size, dimensions=3, candidates=2)
    assert (cell.method, cell.evaluations, cell.limits, cell.batch_size, cell.noise) == (method, 8, limits, batch_size, False)
    assert cell.seconds_per_candidate > 0 and cell.cold_fit_seconds > 0


@pytest.mark.skipif(not MultiObjectiveBayesian.supports_noisy_outputs, reason="noise with limits on outputs is not supported here")
def test_a_noisy_multi_cell_times_its_candidates():
    cell = cost.measure_multi_cell("parego", 8, 2, 1, noise=True, dimensions=3, candidates=2)
    assert cell.noise and cell.seconds_per_candidate > 0


def test_the_multi_problem_returns_both_objectives_and_the_limits_and_noise_makes_repeats_differ():
    _, objective = cost._multi_problem(3, 2)
    point = {"x0": 1.0, "x1": 2.0, "x2": 3.0}
    assert set(objective(point)) == {"f1", "f2", "g0", "g1"} and objective(point) == objective(point)
    _, noisy = cost._multi_problem(3, 2, noise=True)
    assert noisy(point) != noisy(point)


def test_multi_candidates_must_fill_whole_batches():
    with pytest.raises(ValueError, match="multiple of batch_size"):
        cost.measure_multi_cell("parego", 8, 0, 8, candidates=12)


def test_a_multi_cost_is_the_mean_over_seeds():
    cells = cost.measure_multi_cost(("parego",), (8,), (0,), (1,), dimensions=3, candidates=2, seeds=2)
    assert len(cells) == 1 and cells[0].seconds_per_candidate > 0


def _multi(method, n, k, b, seconds, noise=False):
    return cost.MultiCostCell(method=method, evaluations=n, limits=k, batch_size=b, noise=noise, seconds_per_candidate=seconds, cold_fit_seconds=1.0)


def test_the_multi_ceiling_is_checked_per_method_limits_batch_and_noise_at_500_evaluations():
    baseline = [_multi("parego", 500, 8, 1, 0.2), _multi("models", 500, 8, 1, 0.2), _multi("parego", 200, 8, 1, 0.1)]
    within = [_multi("parego", 500, 8, 1, 0.24), _multi("parego", 200, 8, 1, 9.0)]
    assert cost.multi_ceiling_breaches(within, baseline) == []
    over = [_multi("models", 500, 8, 1, 0.25), _multi("parego", 500, 8, 1, 0.5, True), _multi("parego", 500, 0, 1, 9.0)]
    found = cost.multi_ceiling_breaches(over, baseline)
    assert len(found) == 2
    assert found[0].startswith("models, 500 evaluations, 8 limits, batches of 1") and "×1.25" in found[0]
    assert found[1].startswith("parego, noisy, 500 evaluations")


def test_a_noisy_branch_cell_is_judged_against_mains_noise_free_cell():
    main = [_multi("parego", 500, 8, 1, 0.2), _multi("parego", 500, 8, 1, 0.01, True)]  # main's noisy cell is never the reference
    noisy_ok, noisy_over = _multi("parego", 500, 8, 1, 0.23, True), _multi("parego", 500, 8, 1, 0.25, True)
    assert cost.multi_ceiling_breaches([noisy_ok], main) == []
    (found,) = cost.multi_ceiling_breaches([noisy_over], main)
    assert found.startswith("parego, noisy, 500 evaluations") and "against 0.200s" in found
    (line,) = cost.compare_multi(main, [noisy_ok])
    assert "0.200s -> 0.230s, ×1.15" in line
    assert cost.compare_multi([_multi("parego", 500, 8, 1, 0.2, True)], [noisy_ok]) == []  # no noise-free cell on main: nothing to pair


def test_a_table_saved_before_multi_cells_still_loads():
    old = {
        "hone": "0.0", "machine": {}, "dimensions": 10, "candidates": 24, "seeds": 3, "profile": None,
        "cells": [{"evaluations": 50, "limits": 0, "batch_size": 1, "seconds_per_candidate": 0.1, "cold_fit_seconds": 1.0}],
    }
    table = cost.CostTable.model_validate(json.loads(json.dumps(old)))
    assert table.multi == [] and len(table.cells) == 1


def test_the_saved_table_loads_unchanged():
    from pathlib import Path

    saved = Path(__file__).parent.parent / "benchmarks" / "cost-summaries.json"
    if saved.exists():
        assert cost.load_cost(saved).multi == []


def test_the_markdown_gains_a_several_objectives_section_and_multi_only_tables_render():
    cells = [_multi("parego", n, k, 1, 0.01 * (k + 1)) for n in (50, 500) for k in (0, 8)]
    table = cost.CostTable(hone="0.0", machine=cost.machine(), dimensions=10, candidates=24, cells=[], multi=cells)
    text = cost.cost_markdown(table)
    assert "## Several objectives" in text and "### Seconds per candidate, parego, batches of 1" in text
    assert "| 500 | 0.010 | 0.090 |" in text


def test_compare_prints_ratios_and_exits_1_past_the_ceiling(tmp_path, capsys):
    def save(name, seconds):
        table = cost.CostTable(hone="0.0", machine={}, dimensions=10, candidates=24, cells=[], multi=[_multi("parego", 500, 8, 1, seconds), _multi("parego", 50, 8, 1, 1.0)])
        cost.save_cost(table, tmp_path / name)
        return str(tmp_path / name)

    main, ok, slow = save("main.json", 0.2), save("ok.json", 0.22), save("slow.json", 0.3)
    assert cost.cli(["compare", main, ok]) == 0
    assert "Noise-free branch cells against main's noise-free cells" in capsys.readouterr().out
    assert cost.cli(["compare", main, slow]) == 1
    assert "Over the ceiling of ×1.2" in capsys.readouterr().out
