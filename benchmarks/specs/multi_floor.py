"""Item 8's paired noise floor: how small a difference the noisy multi-objective benchmark can resolve.

The three noisy engineering designs, 30 seeds, 10 to 100 evaluations per dimension, four optimizers each run
twice: as the baseline (every draw believed, noise=False) and paired, which searches with a different seed but
starts from the baseline's own first generation. The report gives each cell's interval for the paired-minus-baseline
true gap, and says whether the benchmark can resolve a difference of DELTA. The comparison spec, the default
against this baseline, comes later.
"""
import math
import statistics

from hone.benchmarks.multi import DEFAULT_MULTI_OPTIMIZERS, paired_first_generation
from hone.benchmarks.multi_outputs import noisy_multi_output_problems

KINDS = ("nsga2", "bayesian-parego", "bayesian-models", "random")
PAIRED = "@paired"
DELTA = 0.02  # a share of the true front's scaled hypervolume
UNRESOLVED = "can't resolve δ"


def problems():
    return noisy_multi_output_problems()


def optimizers():
    found = {}
    for kind in KINDS:
        found[kind] = DEFAULT_MULTI_OPTIMIZERS[kind]
        found[kind + PAIRED] = paired_first_generation(DEFAULT_MULTI_OPTIMIZERS[kind])
    return found


SEEDS = 30
BUDGETS = (10, 20, 50, 100)


def order_statistics(n):
    """Indices (low, high) into n sorted differences that bound the median with at least 95% confidence, or None if n is under 5.

    With k the largest count for which a binomial(n, 1/2) is at most k - 1 with probability 0.05 or less, the k-th
    smallest and the (n + 1 - k)-th smallest bracket the median. For n = 30 that is the 11th and the 20th.
    """
    k = 0
    for candidate in range(1, n + 1):
        if sum(math.comb(n, i) for i in range(candidate)) / 2**n <= 0.05:
            k = candidate
    return None if k == 0 else (k - 1, n - k)


def interval(differences):
    """(L, U) of the differences, or None with too few."""
    ordered = sorted(differences)
    bounds = order_statistics(len(ordered))
    return None if bounds is None else (ordered[bounds[0]], ordered[bounds[1]])


def _threshold(problem_name):
    # true_gap is the true front's scaled hypervolume less the run's, so it is in units of the true front's scaled
    # hypervolume (problem.true_hypervolume, which is not 1). DELTA is a share of that, hence the multiplication.
    problem = next(p for p in problems() if p.name == problem_name)
    return DELTA * problem.true_hypervolume


def _baseline_figures(runs):
    reported = sum(r.reported for r in runs)
    false = sum(r.reported - r.truly_feasible for r in runs)
    with_points = [(r.reported - r.truly_feasible) / r.reported for r in runs if r.reported > 0]
    return {
        "gap": statistics.median(r.true_gap for r in runs),
        "pooled": false / reported if reported else math.nan,
        "mean_share": statistics.mean(with_points) if with_points else math.nan,
        "empty": sum(r.reported == 0 for r in runs),
        "front": statistics.median(r.reported for r in runs),
        "feasible": statistics.median(r.truly_feasible for r in runs),
        "any": sum(r.any_truly_feasible for r in runs),
    }


def _cell_rows(runs):
    cells = {}
    for run in runs:
        kind = run.optimizer.removesuffix(PAIRED)
        side = "paired" if run.optimizer.endswith(PAIRED) else "baseline"
        cells.setdefault((run.problem, kind, run.budget_per_dimension), {"baseline": {}, "paired": {}})[side][run.seed] = run
    rows = []
    for (problem, kind, budget), sides in sorted(cells.items()):
        seeds = sorted(sides["baseline"].keys() & sides["paired"].keys())
        if not seeds:
            continue
        bounds = interval([sides["paired"][s].true_gap - sides["baseline"][s].true_gap for s in seeds])
        threshold = _threshold(problem)
        low, high = bounds if bounds is not None else (math.nan, math.nan)
        resolves = bounds is not None and high <= threshold and low >= -threshold and high - low <= threshold
        base = _baseline_figures([sides["baseline"][s] for s in seeds])
        rows.append(
            {"problem": problem, "kind": kind, "budget": budget, "n": len(seeds), "low": low, "high": high,
             "resolves": resolves, "threshold": threshold, "base": base}
        )
    return rows


def report(runs):
    rows = _cell_rows(runs)
    lines = ["## The paired noise floor", ""]
    lines += [
        "For each seed the difference d is the paired run's true gap less the baseline's, so a positive d means the paired run is worse. "
        "L and U are the order statistics that bracket the median of the d's with 95% confidence (the 11th and 20th smallest of 30). "
        f"A cell resolves δ, {DELTA:.0%} of the true front's scaled hypervolume, when U is at most δ, L is at least -δ and U - L is at most δ; "
        f"otherwise the benchmark can't resolve δ there. Gaps are in scaled hypervolume. The baseline's own figures follow.",
        "",
        "| Problem | Optimizer | Per dim | Pairs | L | U | U - L | δ | Verdict | Median true gap | Pooled infeasible | Mean infeasible share | Empty fronts | Median front | Median truly feasible | Runs with a feasible point |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        b = r["base"]
        verdict = "resolves" if r["resolves"] else UNRESOLVED
        lines.append(
            f"| {r['problem']} | {r['kind']} | {r['budget']} | {r['n']} | {r['low']:.4f} | {r['high']:.4f} | {r['high'] - r['low']:.4f} | {r['threshold']:.4f} "
            f"| {verdict} | {b['gap']:.4f} | {b['pooled']:.1%} | {b['mean_share']:.1%} | {b['empty']} "
            f"| {b['front']:g} | {b['feasible']:g} | {b['any']} |"
        )
    resolved = sum(r["resolves"] for r in rows)
    lines += ["", f"{resolved} of {len(rows)} cells resolve δ.", ""]
    for r in rows:
        if not r["resolves"]:
            lines.append(f"- {UNRESOLVED}: {r['problem']}, {r['kind']}, {r['budget']}/dim")
    return "\n".join(lines) + "\n"
