"""Item 8's comparison: noise with several objectives, each reserve against believing every draw.

Noisy OSY, TNK and the cantilever with its stress as an output, 30 seeds, 10 to 100 evaluations per dimension, four
optimizers, each run four ways on the same optimizer seed and noise seed (so runs pair by seed):
  baseline  every draw believed (noise=False), the only way to run these problems before the feature;
  tenth     noise=True with the default reserve (a tenth of the budget, at least 10, at most half) and the default margin;
  fifth     the same with a reserve of a fifth (at least 20, at most half of the budget);
  alone     averaging alone, final_evaluations=0 and the margin off, run for the record and never judged.

This spec runs only once the feature (PR (c)) exists: until then multi-objective optimizers refuse noise with output
constraints and MultiObjectiveConfig rejects final_evaluations above 0, so the factories below can't be built. The
rules the report applies are in section 8 of docs/design/output-constraints-quality.md ("Pass criteria, fixed before
measuring" and "Cells the floor can't resolve"). The report judges criteria 1 to 3 from the runs; criteria 4 (nothing
else moves) and 5 (cost) are checked by tests and the cost harness, and the verdict assumes they hold.
"""
import math
import statistics
from pathlib import Path
from typing import Any

from hone.benchmarks.multi import DEFAULT_MULTI_OPTIMIZERS, MultiProblem
from hone.benchmarks.multi_outputs import noisy_multi_output_problems
from hone.benchmarks.runner import BAYESIAN_MAX_EVALUATIONS, POPULATION_SIZE, SkipBudget
from hone.benchmarks.sharded import load_spec
from hone.config import MultiObjectiveBayesianConfig, MultiObjectiveConfig, NSGA2Config
from hone.optimizers import MultiObjectiveBayesian, MultiObjectiveRandomSearch
from hone.optimizers.genetic import NSGA2

floor = load_spec(Path(__file__).with_name("multi_floor.py"))

KINDS = ("nsga2", "bayesian-parego", "bayesian-models", "random")
VARIANTS = ("baseline", "tenth", "fifth", "alone")
RESERVES = ("tenth", "fifth")
SEEDS = 30
BUDGETS = (10, 20, 50, 100)
DELTA = floor.DELTA  # 0.02: a share of the true front's scaled hypervolume
SHARE = 0.10  # criterion 1's limit on the share of reported points that truly break a limit
ANY_FLOOR = 0.9  # criterion 2: a cell whose baseline share of runs with a feasible point is under this must rise

# The cells the paired noise floor can't resolve δ in: benchmarks/multi-floor-results.md, copied from its list. A test
# recomputes them from benchmarks/multi-floor-runs.json.
FLOOR_UNRESOLVABLE = frozenset(
    [
        ("noisy-osy", "bayesian-models", 10),
        ("noisy-osy", "bayesian-models", 20),
        ("noisy-osy", "bayesian-parego", 10),
        ("noisy-osy", "bayesian-parego", 20),
        ("noisy-osy", "nsga2", 10),
        ("noisy-osy", "nsga2", 20),
        ("noisy-osy", "nsga2", 50),
        ("noisy-osy", "nsga2", 100),
        ("noisy-osy", "random", 20),
        ("noisy-osy", "random", 50),
        ("noisy-osy", "random", 100),
        ("noisy-tnk", "bayesian-models", 10),
        ("noisy-tnk", "bayesian-models", 20),
        ("noisy-tnk", "bayesian-parego", 10),
        ("noisy-tnk", "bayesian-parego", 20),
        ("noisy-tnk", "nsga2", 10),
        ("noisy-tnk", "nsga2", 20),
        ("noisy-tnk", "nsga2", 50),
        ("noisy-tnk", "nsga2", 100),
        ("noisy-tnk", "random", 50),
        ("noisy-tnk", "random", 100),
    ]
)


def problems():
    return noisy_multi_output_problems()


# --- the optimizers ------------------------------------------------------------------


def _noise_options(variant: str, budget: int) -> dict[str, Any]:
    """The keywords a noisy variant adds to the baseline's config."""
    if variant == "tenth":
        return {"noise": True, "final_evaluations": None}  # None: the default reserve, a tenth; margin at its default, 0.95
    if variant == "fifth":
        return {"noise": True, "final_evaluations": min(max(20, budget // 5), budget // 2)}
    if variant == "alone":
        return {"noise": True, "final_evaluations": 0, "feasibility_confidence": None}
    raise ValueError(variant)


def _factory(kind: str, variant: str):
    if variant == "baseline":
        return DEFAULT_MULTI_OPTIMIZERS[kind]

    def build(problem: MultiProblem, budget: int, seed: int, *, initial_points=None):
        extra = _noise_options(variant, budget)
        if kind == "nsga2":
            config = NSGA2Config(objectives=problem.objectives, max_evaluations=budget, seed=seed, **extra)
            return NSGA2(problem.search_space, config, initial_points=initial_points)
        if kind == "random":
            config = MultiObjectiveConfig(
                objectives=problem.objectives, population_size=POPULATION_SIZE, generations=budget, max_evaluations=budget,
                seed=seed, **extra,
            )
            return MultiObjectiveRandomSearch(problem.search_space, config, initial_points=initial_points)
        if budget > BAYESIAN_MAX_EVALUATIONS:
            raise SkipBudget(f"{budget} evaluations is more than Bayesian optimization is run with")
        config = MultiObjectiveBayesianConfig(
            objectives=problem.objectives, max_evaluations=budget, seed=seed, method=kind.removeprefix("bayesian-"), **extra
        )
        return MultiObjectiveBayesian(problem.search_space, config, initial_points=initial_points)

    return build


def optimizers():
    return {f"{kind}@{variant}": _factory(kind, variant) for kind in KINDS for variant in VARIANTS}


# --- the report ----------------------------------------------------------------------


def _share_figures(runs):
    reported = sum(r.reported for r in runs)
    false = sum(r.reported - r.truly_feasible for r in runs)
    with_points = [(r.reported - r.truly_feasible) / r.reported for r in runs if r.reported > 0]
    return {
        "pooled": false / reported if reported else math.nan,
        "mean_share": statistics.mean(with_points) if with_points else math.nan,
        "empty": sum(r.reported == 0 for r in runs),
        "front": statistics.median(r.reported for r in runs),
        "feasible": statistics.median(r.truly_feasible for r in runs),
        "any": sum(r.any_truly_feasible for r in runs),
        "gap": statistics.median(r.true_gap for r in runs),
    }


def _within(value):
    """Whether a share is within SHARE. No points at all (nan) is vacuously within: criterion 2 is what catches an empty cell."""
    return math.isnan(value) or value <= SHARE


def _unchanged(reserve, base):
    return all((a.reported, a.truly_feasible, a.true_gap) == (b.reported, b.truly_feasible, b.true_gap) for a, b in zip(reserve, base))


def _criterion_3(differences, threshold):
    """("pass" | "fail" | "unresolved", (L, U) or None)."""
    bounds = floor.interval(differences)
    if bounds is None:
        return "unresolved", None
    low, high = bounds
    return ("pass" if high <= threshold else "fail" if low > threshold else "unresolved"), bounds


def _judge_reserve(reserve_runs, base_runs, threshold):
    """Criteria 1 to 3 for one reserve against the baseline, in one cell. Both lists are in the same seed order."""
    n = len(base_runs)
    figures = _share_figures(reserve_runs)
    base = _share_figures(base_runs)
    c1 = _within(figures["pooled"]) and _within(figures["mean_share"])
    unchanged = _unchanged(reserve_runs, base_runs)
    rises = figures["any"] > base["any"] or base["any"] >= ANY_FLOOR * n or unchanged
    falls = (base["any"] - figures["any"]) > n / 10
    c3, bounds = _criterion_3([r.true_gap - b.true_gap for r, b in zip(reserve_runs, base_runs)], threshold)
    low_count = (floor.order_statistics(n) or (10, 0))[0] + 1  # 11 of 30
    return {
        "figures": figures, "c1": c1, "no_points": figures["pooled"] != figures["pooled"], "c2": rises and not falls, "falls": falls, "unchanged": unchanged,
        "c3": c3, "bounds": bounds, "n": n,
        "half": sum(r.truly_feasible >= b.truly_feasible / 2 for r, b in zip(reserve_runs, base_runs)),
        "flag_at": low_count,
    }


def _clearly(larger_runs, smaller_runs):
    """"more", "fewer" or "neither": the larger reserve's truly feasible counts less the smaller's, per seed."""
    d = sorted(a.truly_feasible - b.truly_feasible for a, b in zip(larger_runs, smaller_runs))
    bounds = floor.order_statistics(len(d))
    if bounds is None:
        return "neither"
    low, high = d[bounds[0]], d[bounds[1]]
    return "more" if low >= 1 else "fewer" if high <= -1 else "neither"


def _cells(runs):
    grouped = {}
    for run in runs:
        kind, _, variant = run.optimizer.partition("@")
        grouped.setdefault((run.problem, kind, run.budget_per_dimension), {}).setdefault(variant, {})[run.seed] = run
    cells = []
    for key, variants in sorted(grouped.items()):
        base = variants.get("baseline")
        if not base:
            continue
        missing = [r for r in RESERVES if r not in variants]
        if missing:
            raise ValueError(f"{_label({'key': key})} has baseline runs and no runs of the {' or '.join(missing)}")
        threshold = DELTA * next(p for p in problems() if p.name == key[0]).true_hypervolume
        cell = {"key": key, "floor": key in FLOOR_UNRESOLVABLE, "baseline": None, "alone": None, "threshold": threshold}
        for reserve in RESERVES:
            if reserve in variants:
                seeds = sorted(base.keys() & variants[reserve].keys())
                if seeds:
                    cell[reserve] = _judge_reserve([variants[reserve][s] for s in seeds], [base[s] for s in seeds], threshold)
                    cell["baseline"] = _share_figures([base[s] for s in seeds])
        cell["more_fifth"] = None
        if "tenth" in variants and "fifth" in variants:
            seeds = sorted(variants["tenth"].keys() & variants["fifth"].keys())
            if seeds:
                cell["more_fifth"] = _clearly([variants["fifth"][s] for s in seeds], [variants["tenth"][s] for s in seeds])
        if "alone" in variants:
            cell["alone"] = _share_figures(list(variants["alone"].values()))
        if cell["baseline"] is not None:
            cells.append(cell)
    return cells


def _label(cell):
    return "{}, {}, {}/dim".format(*cell["key"])


def _status(reserve, cells):
    """("pass" | "fail" | "undecided", reasons) for a reserve over every cell that has it.

    fail: some criterion failed outright in some cell (criterion 1 or 2 not met, or criterion 3 a fail, wherever).
    undecided: nothing failed outright, but criterion 3 is unresolved in a cell outside FLOOR_UNRESOLVABLE.
    An unresolved criterion 3 inside FLOOR_UNRESOLVABLE counts for nothing.
    """
    failed, unresolved = [], []
    for cell in cells:
        r = cell.get(reserve)
        if r is None:
            continue
        if not r["c1"]:
            failed.append(f"{reserve}: {_label(cell)}: criterion 1 not met")
        if not r["c2"]:
            failed.append(f"{reserve}: {_label(cell)}: criterion 2 not met")
        if r["c3"] == "fail":
            failed.append(f"{reserve}: {_label(cell)}: criterion 3 fails")
        elif r["c3"] == "unresolved" and not cell["floor"]:
            unresolved.append(f"{reserve}: {_label(cell)}: criterion 3 unresolved outside the floor's unresolvable cells")
    if failed:
        return "fail", failed
    return ("undecided", unresolved) if unresolved else ("pass", [])


def choose_reserve(cells):
    """As _choose, with a note added to the reasons for every cell where a reserve reported no points at all."""
    choice, outcomes, branch, reasons = _choose(cells)
    notes = [
        f"{reserve}: {_label(cell)}: criterion 1 met only vacuously, no reported points (criterion 2 judges the cell)"
        for cell in cells for reserve in RESERVES if cell.get(reserve) and cell[reserve].get("no_points")
    ]
    return choice, outcomes, branch, reasons + notes


def _choose(cells):
    """(choice, outcomes, branch, reasons): choice is "tenth", "fifth" or "user"; outcomes maps each reserve to its outcome.

    cells are dicts as _cells builds them: "key", "floor", per reserve {"c1", "c2", "c3"} and "more_fifth" ("more", "fewer",
    "neither" or None). In order:
      1. either reserve undecided: user;
      2. both pass: the fifth ships only if clearly more in at least half of the floor-unresolvable cells with both reserves,
         clearly fewer in none, and it meets criteria 1 and 2 wherever the tenth does; otherwise the tenth;
      3. exactly one passes and the other fails: the one that passes;
      4. both fail: user.
    Criteria 4 and 5 are assumed to hold.
    """
    tenth, tenth_reasons = _status("tenth", cells)
    fifth, fifth_reasons = _status("fifth", cells)
    outcomes = {"tenth": tenth, "fifth": fifth}
    reasons = tenth_reasons + fifth_reasons
    if "undecided" in outcomes.values():
        return "user", outcomes, "1: a reserve is undecided", reasons
    if tenth == "pass" and fifth == "pass":
        both = [c for c in cells if c.get("tenth") and c.get("fifth")]
        floor_cells = [c for c in both if c["floor"]]
        more = [c for c in floor_cells if c["more_fifth"] == "more"]
        fewer = [c for c in floor_cells if c["more_fifth"] == "fewer"]
        keeps = all(c["fifth"]["c1"] and c["fifth"]["c2"] for c in both if c["tenth"]["c1"] and c["tenth"]["c2"])
        reasons = [
            f"the fifth gives clearly more in {len(more)} of {len(floor_cells)} unresolvable cells and clearly fewer in {len(fewer)}"
        ]
        reasons += [f"clearly more: {_label(c)}" for c in more] + [f"clearly fewer: {_label(c)}" for c in fewer]
        if floor_cells and 2 * len(more) >= len(floor_cells) and not fewer and keeps:
            return "fifth", outcomes, "2: both pass, the fifth is clearly more", reasons
        return "tenth", outcomes, "2: both pass, the tenth by default", reasons
    if tenth == "pass":
        return "tenth", outcomes, "3: only the tenth passes", reasons
    if fifth == "pass":
        return "fifth", outcomes, "3: only the fifth passes", reasons
    return "user", outcomes, "4: both fail", reasons


def _pct(x):
    return "n/a" if math.isnan(x) else f"{x:.1%}"


def report(runs):
    cells = _cells(runs)
    lines = ["## Noise with several objectives: each reserve against believing every draw", ""]
    lines += [
        "Each reserve is compared with the baseline (every draw believed), paired by seed. Criterion 1: pooled and per-run mean share of "
        f"reported points that truly break a limit, each at most {SHARE:.0%}; empty fronts are counted apart and are not passes. Criterion 2: "
        f"the share of runs holding a truly feasible point rises where the baseline's is under {ANY_FLOOR:.0%} (unless the cell is seed-for-seed "
        "unchanged) and falls by no more than one run in ten. Criterion 3: d = reserve's true gap less baseline's; L and U are the 11th and 20th "
        f"smallest of 30; pass if U is at most δ ({DELTA:.0%} of the true front's scaled hypervolume), fail if L is above δ, else unresolved. "
        "In a cell the floor can't resolve, an unresolved criterion 3 does not count against a reserve; a fail does. Criteria 4 and 5 are "
        "checked by tests and the cost harness, not from these runs, and the verdict assumes they hold. The `alone` runs are for the record.",
        "",
        "| Problem | Optimizer | Per dim | Floor | Reserve | Pairs | Pooled infeasible (base) | Mean share (base) | Empty (base) | Median front (base) "
        "| Median truly feasible (base) | Any feasible (base) | C1 | C2 | L | U | δ | C3 | Seeds at least half of base |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    flagged = []
    for cell in cells:
        problem, kind, budget = cell["key"]
        b = cell["baseline"]
        for reserve in RESERVES:
            r = cell.get(reserve)
            if r is None:
                continue
            f = r["figures"]
            low, high = r["bounds"] if r["bounds"] else (math.nan, math.nan)
            c2 = "met" if r["c2"] else ("falls" if r["falls"] else "no rise")
            if r["unchanged"]:
                c2 += " (unchanged)"
            if r["unchanged"] and r["no_points"] and b["pooled"] != b["pooled"] and r["figures"]["any"] == 0:
                flagged.append(f"{_label(cell)}: {reserve}: no information: no feasible point under either")
            half = ""
            if cell["floor"]:
                half = f"{r['half']} of {r['n']}"
                if r["half"] < r["flag_at"]:
                    flagged.append(f"{_label(cell)}: {reserve} keeps half the baseline's truly feasible points in {r['half']} of {r['n']} seeds")
            lines.append(
                f"| {problem} | {kind} | {budget} | {'unresolvable' if cell['floor'] else 'resolves'} | {reserve} | {r['n']} "
                f"| {_pct(f['pooled'])} ({_pct(b['pooled'])}) | {_pct(f['mean_share'])} ({_pct(b['mean_share'])}) | {f['empty']} ({b['empty']}) "
                f"| {f['front']:g} ({b['front']:g}) | {f['feasible']:g} ({b['feasible']:g}) | {f['any']} ({b['any']}) "
                f"| {'no reported points' if r['no_points'] else 'met' if r['c1'] else 'not met'} | {c2} | {low:.4f} | {high:.4f} | {cell['threshold']:.4f} | {r['c3']} | {half} |"
            )
    floor_cells = [c for c in cells if c["floor"] and c.get("more_fifth")]
    if floor_cells:
        names = {"more": "clearly more", "fewer": "clearly fewer", "neither": "neither"}
        lines += ["", "### The fifth against the tenth, in the cells the floor can't resolve", "",
                  "| Problem | Optimizer | Per dim | Fifth against tenth |", "| --- | --- | ---: | --- |"]
        lines += ["| {} | {} | {} | {} |".format(*c["key"], names[c["more_fifth"]]) for c in floor_cells]
    alone = [c for c in cells if c["alone"]]
    if alone:
        lines += ["", "### Averaging alone, for the record (never judged)", "",
                  "| Problem | Optimizer | Per dim | Pooled infeasible | Mean share | Empty | Median truly feasible | Any feasible | Median true gap |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for c in alone:
            a = c["alone"]
            lines.append("| {} | {} | {} | {} | {} | {} | {:g} | {} | {:.4f} |".format(
                *c["key"], _pct(a["pooled"]), _pct(a["mean_share"]), a["empty"], a["feasible"], a["any"], a["gap"]))
    choice, outcomes, branch, reasons = choose_reserve(cells)
    lines += ["", "### Which reserve ships", "",
              "| Reserve | Outcome |", "| --- | --- |",
              f"| tenth | {outcomes['tenth']} |", f"| fifth | {outcomes['fifth']} |", "",
              f"Branch taken: {branch}. Verdict: **{choice}**.", ""]
    lines += [f"- {reason}" for reason in reasons]
    if flagged:
        lines += ["", "Flagged for the user (not failures):"] + [f"- {item}" for item in flagged]
    return "\n".join(lines) + "\n"
