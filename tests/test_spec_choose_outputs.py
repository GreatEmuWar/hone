import math

from hone.benchmarks import noisy_output_problems, output_problems
from hone.benchmarks.runner import Run
from hone.benchmarks.sharded import expected_runs, load_spec

spec = load_spec("benchmarks/specs/choose_outputs.py")
NAMES = [p.name for p in output_problems()]
SHAPES = [n for n in NAMES if n not in spec.NON_SHAPES]


def _cell(problem, kind, total, median, feasible):
    """30 runs: `feasible` of them at the error median, the rest infeasible."""
    return [
        Run(problem=problem, optimizer=f"{kind}@{total}", budget_per_dimension=1, seed=seed,
            error=median if seed < feasible else math.inf, evaluations=total, trace=[])
        for seed in range(30)
    ]


def _runs(total, **cells):
    """Every problem at this total: bayesian 1 against 10 for the others, all feasible, unless cells[name] = (bayes, bfeas, pick, pfeas)."""
    runs = []
    for name in NAMES + [p.name for p in noisy_output_problems()]:
        bayes, bfeas, pick, pfeas = cells.get(name.replace("-", "_"), (1.0, 30, 10.0, 30))
        runs += _cell(name, "bayesian", total, bayes, bfeas)
        runs += _cell(name, "cma-es", total, pick, pfeas) + _cell(name, "genetic", total, pick, pfeas)
    return runs


def _cells(names, value):
    return {n.replace("-", "_"): value for n in names}


def _verdict(text):
    return text.split("Verdict: **")[1].split("**")[0]


def test_baseline_ships_500():
    text = spec.report(_runs(300) + _runs(500))
    assert _verdict(text) == "500"
    assert "Branch taken: 2:" in text
    assert "### 300 evaluations: passes" in text and "### 500 evaluations: passes" in text
    assert "About 1 of 14 problems comes out clear at 0.05 by chance." in text
    assert "noisy-welded-beam-outputs" in text  # the record table


def test_only_300_passes_ships_300():
    losing = _cells(NAMES[:4], (10.0, 30, 1.0, 30))  # 4 losses: 10 wins of 14
    text = spec.report(_runs(300) + _runs(500, **losing))
    assert "### 500 evaluations: fails" in text and "criterion 1" in text
    assert _verdict(text) == "300" and "Branch taken: 3:" in text


def test_loss_over_two_goes_to_the_user():
    cells = {"welded_beam_outputs": (30.0, 30, 10.0, 30)}  # x3, 13 wins
    text = spec.report(_runs(300, **cells) + _runs(500))
    assert "### 300 evaluations: goes to the user" in text
    assert "criterion 3, welded-beam-outputs" in text
    assert _verdict(text) == "user" and "Branch taken: 1:" in text


def test_feasibility_fail_when_pick_is_reliable():
    cells = {"g06": (1.0, 24, 10.0, 28)}  # 4 under, pick at 28 of 30
    text = spec.report(_runs(300, **cells) + _runs(500, **cells))
    assert "criterion 2, g06" in text and "(fail)" in text
    assert "### 300 evaluations: fails" in text
    assert _verdict(text) == "200" and "Branch taken: 5:" in text


def test_feasibility_trip_goes_to_the_user_when_pick_is_unreliable():
    cells = {"g06": (1.0, 16, 10.0, 20)}
    text = spec.report(_runs(300, **cells) + _runs(500))
    assert "criterion 2, g06" in text and "(to the user)" in text
    assert "### 300 evaluations: goes to the user" in text
    assert _verdict(text) == "user"


def test_a_gap_of_three_is_not_a_trip():
    cells = {"g06": (1.0, 25, 10.0, 28)}
    assert _verdict(spec.report(_runs(300, **cells) + _runs(500, **cells))) == "500"


def test_shapes_cannot_carry_criterion_1():
    # 8 shapes win, 3 of the 6 others win: 11 of 14 but only 3 non-shapes
    cells = _cells(sorted(spec.NON_SHAPES)[:3], (10.0, 30, 1.0, 30))
    text = spec.report(_runs(300, **cells) + _runs(500, **cells))
    assert "wins 11 of 14" in text and "3 of the 6 that aren't shapes" in text
    assert "### 300 evaluations: fails" in text
    assert _verdict(text) == "200"


def test_only_500_passes_goes_to_the_user():
    losing = _cells(NAMES[:4], (10.0, 30, 1.0, 30))
    text = spec.report(_runs(300, **losing) + _runs(500))
    assert _verdict(text) == "user" and "Branch taken: 4:" in text


def test_ratio_and_ties():
    inf = math.inf
    assert spec._ratio(inf, inf) == 1.0
    assert spec._ratio(3.0, inf) < 1e-6
    assert spec._ratio(inf, 3.0) == inf
    assert spec._ratio(0.0, 0.0) == 1.0
    tied = _cell("p", "bayesian", 300, 0.0, 30)
    assert spec._p_value([r.error for r in tied], [r.error for r in tied])[1] == "asymptotic"
    assert spec._p_value([float(i) for i in range(30)], [i + 0.5 for i in range(30)])[1] == "exact"
    assert spec._p_value([inf] * 30, [inf] * 30)[0] == 1.0


def test_pick_follows_the_rule():
    picks = {p.name: spec._pick(p) for p in output_problems()}
    assert picks["pressure-vessel-outputs"] == "genetic" and picks["g06"] == "cma-es"


def test_the_spec_runs_each_total_once_per_problem():
    expected = expected_runs(spec)
    assert len(expected) == 17 * 6 * 30
    assert {budget for _, _, budget, _ in expected} == {1}
    assert {name for _, name, _, _ in expected} == {f"{k}@{t}" for k in spec.KINDS for t in spec.TOTALS}
