"""Item 3's measurement: whether limits without a size need an explicit scale=.

Each problem of the outputs suite as written, and rewritten with every output limit that has a size
moved to one side, `Output("x") <= 250` as `Output("x") - 250 <= 0`, so that every limit has size 0
and its violations and margins are in raw units. Bayesian optimization, CMA-ES and the genetic
optimizer, 30 seeds, 10 to 50 evaluations per dimension. The report checks the pass criteria fixed
in docs/design/output-constraints-quality.md, item 3.
"""
import math
import statistics

import numpy as np

from hone.benchmarks import output_problems
from hone.benchmarks.problems import Problem
from hone.benchmarks.runner import bayesian, cma_es, genetic
from hone.parameters import SearchSpace
from hone.parameters.constraints import BinaryOp, ExpressionConstraint, Number

SUFFIX = "@unscaled"
FLOOR = 1e-8
CHECK_POINTS = 300


def _unscaled(constraint):
    """constraint with nothing on its limit's side, left - right against 0: the same limit, of size 0."""
    if not isinstance(constraint, ExpressionConstraint) or not constraint.output_names() or constraint.size({}, {}) == 0:
        return constraint
    return ExpressionConstraint(left=BinaryOp(op="-", left=constraint.left, right=constraint.right), op=constraint.op, right=Number(value=0.0))


def _same_feasible_set(problem: Problem, rewritten: Problem) -> None:
    rng = np.random.default_rng(0)
    old, new = problem.search_space, rewritten.search_space
    for _ in range(CHECK_POINTS):
        params = old.sample(rng)
        outputs = {k: v for k, v in problem.function(**old.active(params)).items() if k != "objective"}
        assert [c.violation(params, outputs) == 0 for c in old.output_constraints] == [
            c.violation(params, outputs) == 0 for c in new.output_constraints
        ], f"{rewritten.name} doesn't have the same feasible set as {problem.name} at {params}"


def unscaled(problem: Problem) -> Problem | None:
    """problem with its sized output limits rewritten to size 0, or None when that changes nothing.

    A limit of size 1, such as the spring's `Output("shear") <= 1`, is already in raw units, so a
    problem whose sized limits are all of size 1 would run exactly as written.
    """
    space = problem.search_space
    constraints = [_unscaled(c) for c in space.constraints]
    if all(new is old or old.size({}, {}) == 1 for old, new in zip(space.constraints, constraints)):
        return None
    new_space = SearchSpace(space.parameters, constraints, space.conditions)
    assert all(c.size({}, {}) == 0 for c in new_space.output_constraints)
    rewritten = problem.model_copy(update={"name": problem.name + SUFFIX, "search_space": new_space})
    _same_feasible_set(problem, rewritten)
    return rewritten


def problems():
    pairs = []
    for problem in output_problems():
        rewritten = unscaled(problem)
        if rewritten is not None:
            pairs += [problem, rewritten]
    return pairs


def optimizers():
    return {"bayesian": bayesian, "cma-es": cma_es, "genetic": genetic}


SEEDS = 30
BUDGETS = (10, 20, 50)


def _ratio(new, old):
    return max(new, FLOOR) / max(old, FLOOR)


def report(runs):
    cells = {}
    for run in runs:
        version = "unscaled" if run.problem.endswith(SUFFIX) else "written"
        problem = run.problem.removesuffix(SUFFIX)
        cells.setdefault((run.optimizer, run.budget_per_dimension), {}).setdefault(problem, {}).setdefault(version, []).append(run)
    lines = ["## Item 3's pass criteria", ""]
    lines += ["| Optimizer | Per dim | Problems | Median error ×, geometric mean | Feasible, written | Feasible, unscaled |"]
    lines += ["| --- | ---: | ---: | ---: | ---: | ---: |"]
    detail = ["", "| Optimizer | Per dim | Problem | Median, written | Median, unscaled | × | Feasible, written / unscaled |"]
    detail += ["| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    failures, notes = [], []
    for (optimizer, budget), by in sorted(cells.items()):
        logs, shares = [], {"written": [], "unscaled": []}
        for problem, versions in sorted(by.items()):
            median = {v: statistics.median(r.error for r in rs) for v, rs in versions.items()}
            share = {v: sum(math.isfinite(r.error) for r in rs) / len(rs) for v, rs in versions.items()}
            for v in shares:
                shares[v].append(share[v])
            ratio = _ratio(median["unscaled"], median["written"]) if all(math.isfinite(m) for m in median.values()) else None
            if ratio is not None:
                logs.append(math.log(ratio))
            where = f"{problem}, {optimizer}, {budget}/dim"
            if ratio is not None and ratio >= 2:
                notes.append(f"{where}: median error ×{ratio:.2f}")
            if share["written"] - share["unscaled"] > 0.1 + 1e-9:
                notes.append(f"{where}: feasible share {share['written']:.0%} -> {share['unscaled']:.0%}")
            shown = "–" if ratio is None else f"×{ratio:.2f}"
            detail.append(
                f"| {optimizer} | {budget} | {problem} | {median['written']:.3g} | {median['unscaled']:.3g} | {shown} | "
                f"{share['written']:.0%} / {share['unscaled']:.0%} |"
            )
        mean = math.exp(statistics.fmean(logs)) if logs else 1.0
        written, unscaled_ = statistics.fmean(shares["written"]), statistics.fmean(shares["unscaled"])
        lines.append(f"| {optimizer} | {budget} | {len(logs)} of {len(by)} | ×{mean:.2f} | {written:.1%} | {unscaled_:.1%} |")
        if mean >= 2:
            failures.append(f"{optimizer}, {budget}/dim: median error ×{mean:.2f}")
        if written - unscaled_ > 0.1 + 1e-9:
            failures.append(f"{optimizer}, {budget}/dim: average feasible share {written:.1%} -> {unscaled_:.1%}")
    lines += detail + [""]
    verdict = "scale= is needed" if failures else "scale= is not needed"
    lines += [f"**{verdict}{f', {len(failures)} criteria' if failures else ''}**", ""]
    lines += [f"- {f}" for f in failures] + ([""] if failures else [])
    if notes:
        lines += ["Single problems past ×2 or 1 run in 10, which don't decide it:", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines) + "\n"
