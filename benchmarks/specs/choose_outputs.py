"""Item 4's measurement: how far Bayesian optimization stays the choice with output constraints, 300 or 500 evaluations.

Bayesian optimization, CMA-ES and the genetic optimizer on step 0's problems and the engineering designs with limits
on outputs, 30 seeds, fixed totals of 300 and 500 evaluations (not per dimension), and the three noisy designs for the
record. Bayesian optimization is not capped at 200 here; the default benchmark's cap stays. The report checks each total
against the pass criteria fixed in docs/design/output-constraints-quality.md, item 4.

The runner multiplies a budget per dimension by the dimensions, so this spec runs one budget per dimension and names
each optimizer after its total, such as bayesian@300; the factories ignore the budget they are given.
"""
import math
import statistics

from scipy.stats import mannwhitneyu

from hone.benchmarks import noisy_output_problems, output_problems
from hone.benchmarks.runner import _noisy
from hone.config import BayesianConfig, CMAESConfig, GeneticOptimizerConfig
from hone.optimizers import CMAES, BayesianOptimizer, GeneticOptimizer
from hone.parameters import ChoiceParameter, RangeParameter

KINDS = ("bayesian", "cma-es", "genetic")
TOTALS = (300, 500)
FLOOR = 1e-8
HUGE = 1e300  # stands for an infinite error, so that ranks work
NON_SHAPES = ("g06", "g07", "g10", "pressure-vessel-outputs", "tension-spring-outputs", "welded-beam-outputs")
WINS_NEEDED, NON_SHAPE_WINS_NEEDED = 11, 4
FEASIBILITY_GAP, FEASIBILITY_FAIL_FROM, LOSS_RATIO = 3, 27, 2.0


def _factory(kind, total):
    def make(problem, budget, seed):
        space, common = problem.search_space, {"max_evaluations": total, "seed": seed, "noise": _noisy(problem)}
        if kind == "bayesian":
            return BayesianOptimizer(space, BayesianConfig(**common))
        if kind == "cma-es":
            return CMAES(space, CMAESConfig(generations=total, **common))
        return GeneticOptimizer(space, GeneticOptimizerConfig(**common))

    return make


def problems():
    return output_problems() + noisy_output_problems()


def optimizers():
    return {f"{kind}@{total}": _factory(kind, total) for kind in KINDS for total in TOTALS}


SEEDS = 30
BUDGETS = (1,)  # one budget per dimension; the totals are in the optimizers' names


def _pick(problem):
    """The optimizer choose_optimizer picks past 200 evaluations: genetic with choice or integer parameters, else CMA-ES."""
    for p in problem.search_space.parameters:
        if isinstance(p, ChoiceParameter) or (isinstance(p, RangeParameter) and p.integer and p.maximum > p.minimum):
            return "genetic"
    return "cma-es"


def _ratio(bayes, pick):
    """Bayesian optimization's median error over the pick's, floored at 1e-8; inf over inf is 1, finite over inf 0."""
    if math.isinf(bayes) and math.isinf(pick):
        return 1.0
    return max(bayes, FLOOR) / max(pick, FLOOR)


def _p_value(a, b):
    """Two-sided Mann-Whitney p, infeasible runs as infinite; exact when no two of the values tie. Returns (p, method)."""
    x, y = [min(v, HUGE) for v in a], [min(v, HUGE) for v in b]
    method = "exact" if len(set(x + y)) == len(x) + len(y) else "asymptotic"
    p = float(mannwhitneyu(x, y, alternative="two-sided", method=method).pvalue)
    return (1.0 if math.isnan(p) else p), method


def _fmt(value):
    return "inf" if math.isinf(value) else f"{value:.3g}"


def _rows(runs, total, names, pick_of):
    """One row per problem in names, in order, for the runs at this total."""
    cells = {}
    for run in runs:
        kind, _, t = run.optimizer.partition("@")
        if int(t) == total:
            cells.setdefault((run.problem, kind), {})[run.seed] = run.error
    rows = []
    for name in names:
        by = {kind: [cells[(name, kind)][s] for s in sorted(cells[(name, kind)])] for kind in KINDS if (name, kind) in cells}
        if len(by) < len(KINDS):
            continue
        pick = pick_of[name]
        other = "cma-es" if pick == "genetic" else "genetic"
        b, p, o = (statistics.median(by[k]) for k in ("bayesian", pick, other))
        p_value, method = _p_value(by["bayesian"], by[pick])
        rows.append({
            "problem": name, "pick": pick, "bayes": b, "pick_median": p, "other": other, "other_median": o,
            "ratio": _ratio(b, p), "bayes_feasible": sum(map(math.isfinite, by["bayesian"])),
            "pick_feasible": sum(map(math.isfinite, by[pick])), "n": len(by["bayesian"]), "p": p_value, "method": method,
            "win": max(b, FLOOR) < max(p, FLOOR),
        })
    return rows


def _table(rows):
    lines = ["| Problem | Bayesian median | Pick | Pick median | Other median | Ratio | Feasible: Bayesian | Feasible: pick | Mann-Whitney p |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(
            f"| {r['problem']} | {_fmt(r['bayes'])} | {r['pick']} | {_fmt(r['pick_median'])} | {_fmt(r['other_median'])} | "
            f"{_fmt(r['ratio'])} | {r['bayes_feasible']} of {r['n']} | {r['pick_feasible']} of {r['n']} | {r['p']:.3g} ({r['method']}) |"
        )
    return lines


def _judge(rows):
    """(outcome, notes): outcome is 'fails', 'goes to the user' or 'passes'; notes list every item behind it."""
    wins = sum(r["win"] for r in rows)
    non_shape_wins = sum(r["win"] for r in rows if r["problem"] in NON_SHAPES)
    fails, user = [], []
    criterion_1 = wins >= WINS_NEEDED and non_shape_wins >= NON_SHAPE_WINS_NEEDED
    if not criterion_1:
        fails.append(
            f"criterion 1: Bayesian optimization wins {wins} of {len(rows)} (needs {WINS_NEEDED}) and "
            f"{non_shape_wins} of the 6 that aren't shapes (needs {NON_SHAPE_WINS_NEEDED})"
        )
    for r in rows:
        if r["pick_feasible"] - r["bayes_feasible"] > FEASIBILITY_GAP:
            note = f"criterion 2, {r['problem']}: feasible {r['bayes_feasible']} of {r['n']} for Bayesian optimization, {r['pick_feasible']} for the pick"
            (fails if r["pick_feasible"] >= FEASIBILITY_FAIL_FROM else user).append(note + (" (fail)" if r["pick_feasible"] >= FEASIBILITY_FAIL_FROM else " (to the user)"))
        if r["ratio"] > LOSS_RATIO:
            user.append(f"criterion 3, {r['problem']}: Bayesian median {_fmt(r['bayes'])} against the pick's {_fmt(r['pick_median'])}, ratio {_fmt(r['ratio'])} (to the user)")
    outcome = "fails" if fails else "goes to the user" if user else "passes"
    return outcome, wins, non_shape_wins, fails, user


def _decide(outcomes):
    """(row, text, verdict) of the decision table, from each total's outcome."""
    good = {t: outcomes[t] != "fails" for t in TOTALS}
    asks = {t: outcomes[t] == "goes to the user" for t in TOTALS}
    if good[300] and good[500]:
        row, text, verdict, used = 2, "both budgets pass, so 500 ships", "500", (300, 500)
    elif good[300]:
        row, text, verdict, used = 3, "only 300 passes, so 300 ships", "300", (300,)
    elif good[500]:
        row, text, verdict, used = 4, "500 passes and 300 doesn't, so the user decides", "user", (500,)
    else:
        return "5", "neither budget passes, so 200 stays and the rule is unchanged", "200"
    deciding = [t for t in used if asks[t]]
    if deciding:
        return "1", f"the budget{'s' if len(deciding) > 1 else ''} {' and '.join(map(str, deciding))} that would decide B go{'' if len(deciding) > 1 else 'es'} to the user (otherwise row {row})", "user"
    return str(row), text, verdict


def report(runs):
    names = [p.name for p in output_problems()]
    noisy = [p.name for p in noisy_output_problems()]
    pick_of = {p.name: _pick(p) for p in problems()}
    lines = ["## Item 4's pass criteria", ""]
    outcomes = {}
    for total in TOTALS:
        rows = _rows(runs, total, names, pick_of)
        outcome, wins, non_shape_wins, fails, user = _judge(rows)
        outcomes[total] = outcome
        lines += [f"### {total} evaluations: {outcome}", "", *_table(rows), "",
                  f"Bayesian optimization wins {wins} of {len(rows)} problems, {non_shape_wins} of the 6 that aren't shapes.", ""]
        lines += [f"- Fails: {note}" for note in fails] + [f"- To the user: {note}" for note in user]
        lines += [""] if fails or user else []
    lines += ["About 1 of 14 problems comes out clear at 0.05 by chance. The p-values are reported and don't bind.", ""]
    lines += ["### Which B ships", "", *(f"- {total} evaluations: {outcomes[total]}" for total in TOTALS), ""]
    row, text, verdict = _decide(outcomes)
    lines += [f"Branch taken: {row}: {text}. Verdict: **{verdict}**.", ""]
    for total in TOTALS:
        rows = _rows(runs, total, noisy, pick_of)
        lines += [f"### For the record: the noisy designs at {total} evaluations, not judged", "", *_table(rows), ""]
    return "\n".join(lines) + "\n"
