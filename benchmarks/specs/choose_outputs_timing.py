"""Bayesian optimization's real run time at 300 and 500 evaluations, on GitHub's runners, one run at a time per runner.

The 14 noise-free output-constrained problems, 5 seeds, Bayesian optimization uncapped at a total of 500 evaluations
(not per dimension), timing the moment each run reaches 300. Run it with the Measure workflow's jobs input set to 1.

The runner multiplies a budget per dimension by the dimensions, so this spec runs one budget per dimension and names
the optimizer after its total; the factory ignores the budget it is given.
"""
import math

from hone.benchmarks import output_problems
from hone.config import BayesianConfig
from hone.optimizers import BayesianOptimizer

TOTAL = 500
LIMITS_FOR_FLAG = 8  # output limits from which a slow run goes to the user
SLOW_SECONDS = 600.0


def problems():
    return output_problems()


def _bayesian(problem, budget, seed):
    return BayesianOptimizer(problem.search_space, BayesianConfig(max_evaluations=TOTAL, seed=seed, noise=False))


def optimizers():
    return {f"bayesian@{TOTAL}": _bayesian}


SEEDS = 5
BUDGETS = (1,)  # one budget per dimension; the total is in the optimizer's name
TIME_AT = (300,)


def _p90(values):
    """Nearest-rank 90th percentile: sorted(values)[ceil(0.9 * n) - 1], which is the largest of 5 values."""
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def _fmt(seconds):
    return f"{seconds:.1f} s ({seconds / 60:.1f} min)" if seconds > 60 else f"{seconds:.1f} s"


def report(runs):
    by_problem = {}
    for run in runs:
        by_problem.setdefault(run.problem, []).append(run)
    problem_of = {p.name: p for p in problems()}
    lines = [
        "## Bayesian optimization's run time",
        "",
        "Median and 90th percentile over the seeds. The 90th percentile is the nearest-rank one, "
        "sorted(x)[ceil(0.9 n) - 1], which is the largest value at n = 5.",
        "",
        "| Problem | Dimensions | Output limits | Median at 300 | p90 at 300 | Median at 500 | p90 at 500 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    slow = []
    for name, group in by_problem.items():
        problem = problem_of.get(name)
        dimensions = problem.dimensions if problem else "?"
        limits = len(problem.search_space.output_constraints) if problem else 0
        at_300 = [r.seconds_at[300] for r in group if r.seconds_at and 300 in r.seconds_at]
        at_500 = [r.seconds for r in group if r.seconds is not None]
        cells = [_fmt(f(v)) if v else "n/a" for v in (at_300, at_500) for f in (_median, _p90)]
        lines.append(f"| {name} | {dimensions} | {limits} | {' | '.join(cells)} |")
        if limits >= LIMITS_FOR_FLAG and at_500 and _p90(at_500) > SLOW_SECONDS:
            slow.append(name)
    machines = {}
    for run in runs:
        machines[run.machine or "unknown"] = machines.get(run.machine or "unknown", 0) + 1
    lines += ["", "Machines:", "", *(f"- {machine}: {count} runs" for machine, count in sorted(machines.items())), ""]
    lines += ["Wall time including the objective, which is close to zero on these problems; one run at a time per runner.", ""]
    if slow:
        lines += [f"Over about 10 minutes at {LIMITS_FOR_FLAG} limits: goes to the user. ({', '.join(slow)})", ""]
    return "\n".join(lines) + "\n"
