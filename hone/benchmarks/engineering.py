"""The engineering designs and CEC 2006 problems against their published optima, from saved benchmark summaries.

The table is built from the summaries `--save` writes, so its numbers are never copied by hand:
`python -m hone.benchmarks --engineering` writes it from the constrained and outputs suites' files.
"""
import math
from collections.abc import Iterable

from .constrained import (
    pressure_vessel,
    pressure_vessel_outputs,
    tension_spring,
    tension_spring_outputs,
    welded_beam,
    welded_beam_outputs,
)
from .outputs import g06, g07, g10
from .problems import Problem
from .runner import BAYESIAN_MAX_EVALUATIONS, Summary

SOURCES = ("benchmarks/constrained-summaries.json", "benchmarks/outputs-summaries.json")  # later files take precedence


def engineering_problems() -> list[Problem]:
    """The three engineering designs, each with its limits on the parameters and on outputs, then g06, g07 and g10."""
    return [
        pressure_vessel(),
        pressure_vessel_outputs(),
        tension_spring(),
        tension_spring_outputs(),
        welded_beam(),
        welded_beam_outputs(),
        g06(),
        g07(),
        g10(),
    ]


def best_budget(summaries: Iterable[Summary]) -> Summary | None:
    """The summary with the lowest median error, the smallest budget among ties; None if none found a feasible design."""
    finite = [s for s in summaries if math.isfinite(s.median_error)]
    return min(finite, key=lambda s: (s.median_error, s.budget_per_dimension), default=None)


def gap(error: float, optimum: float) -> str:
    """An error as a share of the optimum, or as it is where the optimum is 0."""
    if optimum == 0:
        return f"+{error:.3g}"
    percent = 100 * error / abs(optimum)
    if percent == 0:
        return "+0%"
    if percent < 0.001:
        return "<0.001%"
    return f"+{percent:.0f}%" if percent >= 10 else f"+{percent:#.2g}%"


def engineering_table(summaries: Iterable[Summary], problems: list[Problem] | None = None) -> str:
    """A row per problem and a column per optimizer: the best median value over the budgets, its gap and budget.

    Where the summaries hold the same problem, optimizer and budget more than once, the last wins,
    so a later file of summaries replaces an earlier one's results.
    """
    problems = engineering_problems() if problems is None else problems
    by_key = {(s.problem, s.optimizer, s.budget_per_dimension): s for s in summaries}
    names = {p.name for p in problems}
    optimizers = list(dict.fromkeys(o for p, o, _ in by_key if p in names))
    lines = [
        "| Problem | Published optimum | " + " | ".join(optimizers) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in optimizers) + " |",
    ]
    for problem in problems:
        cells = []
        for optimizer in optimizers:
            runs = [s for (p, o, _), s in by_key.items() if p == problem.name and o == optimizer]
            best = best_budget(runs)
            if not runs:
                cells.append("n/a")
            elif best is None:
                cells.append("none feasible")
            else:
                value = problem.optimum + best.median_error
                cells.append(f"{value:.6g} ({gap(best.median_error, problem.optimum)}) at {best.budget_per_dimension:,}")
        lines.append(f"| {problem.name} | {problem.optimum:.6g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Each cell is the median best value over the runs at the budget, in evaluations per dimension, where that median "
        "comes closest to the published optimum; in brackets, how far above it, as a share of the optimum. Ties go to "
        "the smaller budget. n/a: the optimizer wasn't run on the problem. None feasible: at every budget, most runs "
        "found no design satisfying every limit. Bayesian optimization runs only up to "
        f"{BAYESIAN_MAX_EVALUATIONS} evaluations, so at fewer budgets than the others, which run up to 1,000 per dimension.",
    ]
    return "\n".join(lines) + "\n"
