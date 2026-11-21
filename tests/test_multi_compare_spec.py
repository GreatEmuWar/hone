import json
from pathlib import Path

import pytest

from hone.benchmarks.multi import MultiRun
from hone.benchmarks.sharded import load_spec

spec = load_spec("benchmarks/specs/multi_compare.py")
floor = spec.floor
OSY, CANT = "noisy-osy", "noisy-cantilever-outputs"
FLOOR_CELL = (OSY, "random", 50)  # in FLOOR_UNRESOLVABLE
RESOLVED_CELL = (CANT, "random", 50)  # not


def _runs(cell, variant, specs):
    """specs: one (reported, truly_feasible, true_gap) per seed."""
    problem, kind, budget = cell
    return [
        MultiRun(problem=problem, optimizer=f"{kind}@{variant}", budget_per_dimension=budget, seed=seed, gap=0.0, front_size=rep,
                 evaluations=0, front=[], reported=rep, truly_feasible=tf, any_truly_feasible=tf > 0, true_gap=gap)
        for seed, (rep, tf, gap) in enumerate(specs)
    ]


def _judge(cell, base, reserve):
    runs = _runs(cell, "baseline", base) + _runs(cell, "tenth", reserve) + _runs(cell, "fifth", reserve)
    (c,) = spec._cells(runs)
    return c["tenth"]


def _same(n=30, reported=10, tf=8, gap=0.1):
    return [(reported, tf, gap)] * n


def test_floor_unresolvable_matches_the_measured_floor():
    runs = [MultiRun.model_validate(r) for r in json.loads(Path("benchmarks/multi-floor-runs.json").read_text())]
    computed = {(r["problem"], r["kind"], r["budget"]) for r in floor._cell_rows(runs) if not r["resolves"]}
    assert computed == spec.FLOOR_UNRESOLVABLE and len(computed) == 21


def test_optimizers_names():
    expected = {f"{k}@{v}" for k in ("nsga2", "bayesian-parego", "bayesian-models", "random") for v in ("baseline", "tenth", "fifth", "alone")}
    assert set(spec.optimizers()) == expected and len(expected) == 16
    assert spec.SEEDS == 30 and spec.BUDGETS == (10, 20, 50, 100)


def test_noise_options():
    assert spec._noise_options("tenth", 100) == {"noise": True, "final_evaluations": None}
    assert spec._noise_options("fifth", 100) == {"noise": True, "final_evaluations": 20}
    assert spec._noise_options("fifth", 300)["final_evaluations"] == 60
    assert spec._noise_options("fifth", 30)["final_evaluations"] == 15  # half the budget caps the floor of 20
    assert spec._noise_options("alone", 100) == {"noise": True, "final_evaluations": 0, "feasibility_confidence": None}


def test_criterion_1_pooled_vs_per_run():
    base = _same()
    # pooled 8/2608 = 0.3%, but 4 of 30 runs have all their points infeasible: per-run mean 13%
    reserve = [(100, 100, 0.1)] * 26 + [(2, 0, 0.1)] * 4
    j = _judge(RESOLVED_CELL, base, reserve)
    assert spec._within(j["figures"]["pooled"]) and not spec._within(j["figures"]["mean_share"])
    assert not j["c1"]
    # pooled 50/129 = 39% but per-run mean 1.7%
    reserve = [(1, 1, 0.1)] * 29 + [(100, 50, 0.1)]
    j = _judge(RESOLVED_CELL, base, reserve)
    assert not j["c1"] and spec._within(j["figures"]["mean_share"]) and not spec._within(j["figures"]["pooled"])
    assert _judge(RESOLVED_CELL, base, _same(tf=10))["c1"]


def test_empty_fronts_counted_apart():
    base = _same()
    reserve = [(0, 0, 0.1)] * 2 + [(10, 10, 0.1)] * 28
    j = _judge(RESOLVED_CELL, base, reserve)
    assert j["figures"]["empty"] == 2 and j["figures"]["mean_share"] == 0 and j["c1"]
    assert j["c2"]  # two runs fewer with a feasible point is within one in ten
    assert not _judge(RESOLVED_CELL, base, [(0, 0, 0.1)] * 4 + [(10, 10, 0.1)] * 26)["c2"]


def test_criterion_2_rise_fall_and_unchanged():
    low_base = [(5, 0, 0.1)] * 20 + [(5, 3, 0.1)] * 10  # 10 of 30 with a feasible point
    same = _judge(RESOLVED_CELL, low_base, low_base)
    assert same["unchanged"] and same["c2"]  # exempt from rising
    no_rise = _judge(RESOLVED_CELL, low_base, [(5, 0, 0.1)] * 20 + [(5, 4, 0.1)] * 10)
    assert not no_rise["unchanged"] and not no_rise["c2"]
    assert _judge(RESOLVED_CELL, low_base, [(5, 0, 0.1)] * 18 + [(5, 3, 0.1)] * 12)["c2"]  # rises
    high_base = [(5, 3, 0.1)] * 28 + [(5, 0, 0.1)] * 2  # 28 of 30 is at least 90%: need not rise
    assert _judge(RESOLVED_CELL, high_base, [(5, 2, 0.1)] * 28 + [(5, 0, 0.1)] * 2)["c2"]
    everyone = [(5, 3, 0.1)] * 30
    assert _judge(RESOLVED_CELL, everyone, [(5, 3, 0.1)] * 27 + [(5, 0, 0.1)] * 3)["c2"]  # falls by 3: allowed
    assert not _judge(RESOLVED_CELL, everyone, [(5, 3, 0.1)] * 26 + [(5, 0, 0.1)] * 4)["c2"]  # by 4: not


def test_criterion_3_pass_fail_unresolved():
    t = floor._threshold(CANT)
    base = _same(gap=0.5)

    def with_d(ds):
        return [(10, 8, 0.5 + d) for d in ds]

    assert _judge(RESOLVED_CELL, base, with_d([0.0] * 30))["c3"] == "pass"
    assert _judge(RESOLVED_CELL, base, with_d([2 * t] * 30))["c3"] == "fail"
    assert _judge(RESOLVED_CELL, base, with_d([0.0] * 15 + [3 * t] * 15))["c3"] == "unresolved"
    assert _judge(RESOLVED_CELL, _same(4, gap=0.5), with_d([0.0] * 4))["c3"] == "unresolved"  # too few pairs


def test_half_of_baseline_flag():
    base = [(10, 8, 0.1)] * 30
    enough = [(10, 4, 0.1)] * 11 + [(10, 3, 0.1)] * 19  # 4 is at least half of 8 in 11 seeds
    assert _judge(FLOOR_CELL, base, enough)["half"] == 11
    assert "keeps half" not in spec.report(_runs(FLOOR_CELL, "baseline", base) + _runs(FLOOR_CELL, "tenth", enough) + _runs(FLOOR_CELL, "fifth", enough))
    short = [(10, 4, 0.1)] * 10 + [(10, 3, 0.1)] * 20
    assert _judge(FLOOR_CELL, base, short)["half"] == 10
    text = spec.report(_runs(FLOOR_CELL, "baseline", base) + _runs(FLOOR_CELL, "tenth", short) + _runs(FLOOR_CELL, "fifth", enough))
    assert "tenth keeps half the baseline's truly feasible points in 10 of 30 seeds" in text
    # only floor cells are checked
    assert "keeps half" not in spec.report(_runs(RESOLVED_CELL, "baseline", base) + _runs(RESOLVED_CELL, "tenth", short) + _runs(RESOLVED_CELL, "fifth", short))


def test_clearly_more_fewer_and_ties():
    def pair(ds):
        small = _runs(FLOOR_CELL, "tenth", [(10, 5, 0.1)] * len(ds))
        large = _runs(FLOOR_CELL, "fifth", [(10, 5 + d, 0.1) for d in ds])
        return spec._clearly(large, small)

    assert pair([1] * 20 + [0] * 10) == "more"  # 20 of 30 strictly more
    assert pair([1] * 19 + [0] * 11) == "neither"  # ties count against
    assert pair([-1] * 20 + [0] * 10) == "fewer"
    assert pair([-1] * 19 + [0] * 11) == "neither"
    assert pair([0] * 30) == "neither"


def _cell(floor_=True, key=FLOOR_CELL, tenth="pass", fifth="pass", more=None):
    return {
        "key": key, "floor": floor_, "more_fifth": more,
        "tenth": {"c1": True, "c2": True, "c3": tenth}, "fifth": {"c1": True, "c2": True, "c3": fifth},
    }


def _many(more):
    return [_cell(key=(OSY, "nsga2", i), more=m) for i, m in enumerate(more)]


def test_verdict_branch_2_both_pass():
    choice, outcomes, branch, _ = spec.choose_reserve(_many(["neither"] * 4))
    assert (choice, outcomes) == ("tenth", {"tenth": "pass", "fifth": "pass"}) and branch.startswith("2")
    assert spec.choose_reserve(_many(["more", "more", "neither", "neither"]))[0] == "fifth"
    assert spec.choose_reserve(_many(["more", "neither", "neither", "neither"]))[0] == "tenth"
    assert spec.choose_reserve(_many(["more", "more", "more", "fewer"]))[0] == "tenth"  # clearly fewer in one cell
    cells = _many(["more", "neither"]) + [_cell(floor_=False, key=RESOLVED_CELL, more="fewer")]
    assert spec.choose_reserve(cells)[0] == "fifth"  # only unresolvable cells are counted


def test_verdict_unresolved_inside_floor_list_does_not_count():
    choice, outcomes, _, _ = spec.choose_reserve([_cell(tenth="unresolved", fifth="unresolved", more="more")])
    assert outcomes == {"tenth": "pass", "fifth": "pass"} and choice == "fifth"
    # a fail inside the list still counts
    assert spec.choose_reserve([_cell(tenth="fail", fifth="pass")])[:2] == ("fifth", {"tenth": "fail", "fifth": "pass"})


def test_verdict_branch_3_exactly_one_passes():
    choice, _, branch, _ = spec.choose_reserve([_cell(floor_=False, key=RESOLVED_CELL, tenth="pass", fifth="fail")])
    assert choice == "tenth" and branch.startswith("3")
    assert spec.choose_reserve([_cell(floor_=False, key=RESOLVED_CELL, tenth="fail", fifth="pass")])[0] == "fifth"
    cell = _cell(floor_=False, key=RESOLVED_CELL)
    cell["tenth"]["c1"] = False  # criterion 1 failing is a fail too
    assert spec.choose_reserve([cell])[:2] == ("fifth", {"tenth": "fail", "fifth": "pass"})


def test_verdict_branch_4_both_fail():
    choice, outcomes, branch, reasons = spec.choose_reserve([_cell(tenth="fail", fifth="fail")])
    assert choice == "user" and outcomes == {"tenth": "fail", "fifth": "fail"} and branch.startswith("4") and reasons


def test_verdict_branch_1_undecided_goes_to_user():
    def out(ten, fif):
        return spec.choose_reserve([_cell(floor_=False, key=RESOLVED_CELL, tenth=ten, fifth=fif)])

    choice, outcomes, branch, _ = out("unresolved", "pass")  # the fifth must not win over a merely undecided tenth
    assert choice == "user" and outcomes == {"tenth": "undecided", "fifth": "pass"} and branch.startswith("1")
    assert out("pass", "unresolved")[0] == "user"
    assert out("unresolved", "unresolved")[0] == "user"
    assert out("unresolved", "fail")[0] == "user"  # undecided is checked before the fail
    # a reserve with an outright fail elsewhere is a fail, not undecided
    cells = [_cell(floor_=False, key=RESOLVED_CELL, tenth="unresolved", fifth="pass"), _cell(tenth="fail", fifth="pass")]
    assert spec.choose_reserve(cells)[:2] == ("fifth", {"tenth": "fail", "fifth": "pass"})


def test_report_end_to_end_on_synthetic_runs():
    runs = []
    for cell in (FLOOR_CELL, RESOLVED_CELL):
        runs += _runs(cell, "baseline", _same(tf=10, gap=0.5))
        runs += _runs(cell, "tenth", _same(tf=10, gap=0.5))
        runs += _runs(cell, "fifth", _same(tf=10, gap=0.5))
        runs += _runs(cell, "alone", _same(tf=10, gap=0.5))
    text = spec.report(runs)
    assert "| tenth | pass |" in text and "| fifth | pass |" in text and "Verdict: **tenth**" in text
    assert "Averaging alone, for the record" in text and "assumes they hold" in text


def test_no_reported_points_is_vacuous_for_criterion_1_but_not_shown_as_met():
    # (a) the baseline had feasible runs: criterion 2 catches the fall
    base = _same()
    j = _judge(RESOLVED_CELL, base, [(0, 0, 0.1)] * 30)
    assert j["no_points"] and j["c1"] and j["falls"] and not j["c2"]
    # (b) the baseline had none, under 90%, and the reserve has none: no rise, and not unchanged since the runs differ
    base = [(3, 0, 0.1)] * 30
    j = _judge(RESOLVED_CELL, base, [(0, 0, 0.2)] * 30)
    assert j["no_points"] and j["c1"] and not j["unchanged"] and not j["c2"]
    # both entirely empty and identical: the unchanged exemption applies as written
    j = _judge(RESOLVED_CELL, [(0, 0, 0.1)] * 30, [(0, 0, 0.1)] * 30)
    assert j["no_points"] and j["unchanged"] and j["c2"]
    runs = _runs(RESOLVED_CELL, "baseline", [(0, 0, 0.1)] * 30) + _runs(RESOLVED_CELL, "tenth", [(0, 0, 0.1)] * 30)
    runs += _runs(RESOLVED_CELL, "fifth", [(0, 0, 0.1)] * 30)
    text = spec.report(runs)
    assert "| no reported points |" in text and "| met | met" not in text
    assert any("no reported points" in reason for reason in spec.choose_reserve(spec._cells(runs))[3])
    assert "no information: no feasible point under either" in text
    assert text.count("no information: no feasible point under either") == 2  # tenth and fifth
    cells = spec._cells(runs)
    assert spec.choose_reserve(cells)[:2] == ("tenth", {"tenth": "pass", "fifth": "pass"})  # neither helped nor hurt
    # a cell that isn't empty and identical isn't flagged
    runs = _runs(RESOLVED_CELL, "baseline", _same()) + _runs(RESOLVED_CELL, "tenth", _same()) + _runs(RESOLVED_CELL, "fifth", _same())
    assert "no information" not in spec.report(runs)


def test_floor_cell_missing_a_reserve_raises():
    runs = _runs(FLOOR_CELL, "baseline", _same()) + _runs(FLOOR_CELL, "tenth", _same())
    with pytest.raises(ValueError, match="noisy-osy, random, 50/dim.*fifth"):
        spec.report(runs)
    # so must any cell with baseline runs, floor cell or not
    runs = _runs(RESOLVED_CELL, "baseline", _same()) + _runs(RESOLVED_CELL, "tenth", _same())
    with pytest.raises(ValueError, match="noisy-cantilever-outputs, random, 50/dim.*fifth"):
        spec._cells(runs)
    with pytest.raises(ValueError, match="tenth or fifth"):
        spec._cells(_runs(RESOLVED_CELL, "baseline", _same()))
    assert spec._cells([]) == []  # a cell with no runs at all is fine
