"""Item 7's measurement: the margin of safety for limits under noise, against judging each limit on the average alone.

The three noisy engineering designs, 30 seeds, 10 to 100 evaluations per dimension, four optimizers at
confidence None (item 2's behaviour), 0.9, 0.95 and 0.99. The report checks each confidence against the
pass criteria fixed in docs/design/output-constraints-quality.md, item 7.
"""
import math
import statistics

from hone.benchmarks import noisy_output_problems
from hone.benchmarks.runner import BAYESIAN_MAX_EVALUATIONS, POPULATION_SIZE, SkipBudget
from hone.config import BayesianConfig, CMAESConfig, GeneticOptimizerConfig, OptimizerConfig
from hone.optimizers import CMAES, BayesianOptimizer, GeneticOptimizer, RandomSearch

CONFIDENCES = (None, 0.9, 0.95, 0.99)
OPTIMIZERS = ("bayesian", "cma-es", "genetic", "random")
FLOOR = 1e-8


def _factory(kind, confidence):
    def make(problem, budget, seed):
        space, common = problem.search_space, {"max_evaluations": budget, "seed": seed, "noise": True, "confidence": confidence}
        if kind == "bayesian":
            if budget > BAYESIAN_MAX_EVALUATIONS:
                raise SkipBudget(f"{budget} evaluations is more than Bayesian optimization is run with")
            return BayesianOptimizer(space, BayesianConfig(**common))
        if kind == "cma-es":
            return CMAES(space, CMAESConfig(generations=budget, **common))
        if kind == "genetic":
            return GeneticOptimizer(space, GeneticOptimizerConfig(**common))
        return RandomSearch(space, OptimizerConfig(population_size=POPULATION_SIZE, generations=budget, **common))

    return make


def _name(kind, confidence):
    return f"{kind}@{'none' if confidence is None else confidence}"


def problems():
    return noisy_output_problems()


def optimizers():
    return {_name(kind, c): _factory(kind, c) for kind in OPTIMIZERS for c in CONFIDENCES}


SEEDS = 30
BUDGETS = (10, 20, 50, 100)


def _ratio(new, old):
    if math.isinf(new) and math.isinf(old):
        return 1.0
    return max(new, FLOOR) / max(old, FLOOR)


def report(runs):
    cells = {}
    for run in runs:
        kind, confidence = run.optimizer.split("@")
        cells.setdefault((run.problem, kind, run.budget_per_dimension), {}).setdefault(confidence, {})[run.seed] = run
    lines = ["## Item 7's pass criteria", ""]
    lines += ["| Problem | Optimizer | Per dim | Feasible: none / 0.9 / 0.95 / 0.99 | False assurance: none / 0.9 / 0.95 / 0.99 | Median error ×, 0.95 | Feasible-both error ×, 0.95 |"]
    lines += ["| --- | --- | ---: | --- | --- | ---: | ---: |"]
    failures = {c: [] for c in ("0.9", "0.95", "0.99")}
    for (problem, kind, budget), by in sorted(cells.items()):
        base = by["none"]
        share = {c: sum(math.isfinite(r.error) for r in by[c].values()) / len(by[c]) for c in by}
        false = {c: sum(bool(r.reported_feasible) and math.isinf(r.error) for r in by[c].values()) / len(by[c]) for c in by}
        median = {c: statistics.median(r.error for r in by[c].values()) for c in by}
        both = {}
        for c in failures:
            paired = [(by[c][s].error, base[s].error) for s in base if math.isfinite(base[s].error) and math.isfinite(by[c][s].error)]
            both[c] = _ratio(statistics.median(p[0] for p in paired), statistics.median(p[1] for p in paired)) if paired else 1.0
            where = f"{problem}, {kind}, {budget}/dim"
            if share["none"] < 0.9 and not share[c] > share["none"]:
                failures[c].append(f"{where}: feasible share {share['none']:.0%} -> {share[c]:.0%}, not up")
            if share["none"] - share[c] > 0.1 + 1e-9:
                failures[c].append(f"{where}: feasible share fell {share['none']:.0%} -> {share[c]:.0%}")
            if false[c] > 0.1 + 1e-9:
                failures[c].append(f"{where}: false assurance {false[c]:.0%}")
            if _ratio(median[c], median["none"]) > 1.1:
                failures[c].append(f"{where}: median error ×{_ratio(median[c], median['none']):.2f}")
            if both[c] > 1.5:
                failures[c].append(f"{where}: error among runs feasible both ways ×{both[c]:.2f}")
        order = ("none", "0.9", "0.95", "0.99")
        lines += [
            f"| {problem} | {kind} | {budget} | {' / '.join(f'{share[c]:.0%}' for c in order)} | "
            f"{' / '.join(f'{false[c]:.0%}' for c in order)} | ×{_ratio(median['0.95'], median['none']):.2f} | ×{both['0.95']:.2f} |"
        ]
    lines += [""]
    for c, found in failures.items():
        lines += [f"**confidence {c}: {'passes' if not found else f'fails, {len(found)} criteria'}**", ""]
        lines += [f"- {f}" for f in found] + ([""] if found else [])
    return "\n".join(lines) + "\n"
