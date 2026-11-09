"""Item 4's measurement: how far choose_optimizer should pick Bayesian optimization when a search space has limits on outputs.

The outputs suite, its noisy designs included, at 10 to 100 evaluations per dimension: Bayesian optimization
with its benchmark cap lifted to 1,000 evaluations, CMA-ES, the genetic optimizer and random search, 30 seeds,
except Bayesian optimization past 200 evaluations, which runs 10 seeds by the user's decision of 8 Nov 2026.
The report scores rules that pick Bayesian optimization up to 200 (today's), 500 or 1,000 evaluations against
the pass criteria fixed in docs/design/output-constraints-quality.md, item 4.
"""
import math
import statistics

from hone import choose_optimizer
from hone.benchmarks import noisy_output_problems, output_problems
from hone.benchmarks.runner import _noisy, SkipBudget, cma_es, genetic, random_search
from hone.config import BayesianConfig
from hone.optimizers import CMAES, BayesianOptimizer, GeneticOptimizer

BAYESIAN_UP_TO = 1000  # the cap lifted from the benchmark's 200
FEW_SEEDS_PAST = 200  # Bayesian runs past this many evaluations use only the first FEW_SEEDS seeds
FEW_SEEDS = 10
RULES = {"R0": 200, "R1": 500, "R2": 1000}  # Bayesian optimization up to this many evaluations
FLOOR = 1e-8
NAMES = {BayesianOptimizer: "bayesian", CMAES: "cma-es", GeneticOptimizer: "genetic"}


def bayesian(problem, budget, seed):
    if budget > BAYESIAN_UP_TO or (budget > FEW_SEEDS_PAST and seed >= FEW_SEEDS):
        raise SkipBudget("past the cap, or a seed beyond the few run past 200 evaluations")
    return BayesianOptimizer(problem.search_space, BayesianConfig(max_evaluations=budget, seed=seed, noise=_noisy(problem)))


def problems():
    return output_problems() + noisy_output_problems()


def optimizers():
    return {"bayesian": bayesian, "cma-es": cma_es, "genetic": genetic, "random": random_search}


SEEDS = 30
BUDGETS = (10, 20, 50, 100)


def pick(rule_limit, space, evaluations):
    """What a rule chooses: Bayesian optimization up to rule_limit evaluations, else what choose_optimizer picks past 200."""
    if evaluations <= rule_limit:
        return "bayesian"
    return NAMES[type(choose_optimizer(space, max(evaluations, 201)))]


def report(runs):
    by = {}
    for run in runs:
        by.setdefault((run.problem, run.budget_per_dimension), {}).setdefault(run.optimizer, []).append(run)
    spaces = {p.name: p for p in problems()}
    lines = ["## Item 4's pass criteria", ""]
    lines += ["| Problem | Per dim | Evaluations | Median error: bayesian / cma-es / genetic / random | Feasible: same order | R0 | R1 | R2 |"]
    lines += ["| --- | ---: | ---: | --- | --- | --- | --- | --- |"]
    scores = {rule: {"logs": [], "best": 0, "cells": 0, "lost": []} for rule in RULES}
    feasibility = {rule: [] for rule in RULES}
    for (name, budget), cell in sorted(by.items()):
        problem = spaces[name]
        evaluations = budget * problem.dimensions
        median = {o: statistics.median(r.error for r in rs) for o, rs in cell.items()}
        share = {o: sum(math.isfinite(r.error) for r in rs) / len(rs) for o, rs in cell.items()}
        finite = [m for m in median.values() if math.isfinite(m)]
        best = min(finite) if finite else math.inf
        picks = {rule: pick(limit, problem.search_space, evaluations) for rule, limit in RULES.items()}
        for rule, chosen in picks.items():
            if chosen not in median or not math.isfinite(best):
                continue
            scores[rule]["cells"] += 1
            if math.isinf(median[chosen]):
                scores[rule]["lost"].append(f"{name} {budget}/dim")
                continue
            scores[rule]["logs"].append(math.log(max(median[chosen], FLOOR) / max(best, FLOOR)))
            scores[rule]["best"] += max(median[chosen], FLOOR) <= max(best, FLOOR)
            r0 = picks["R0"]
            if chosen != r0 and r0 in share and share[chosen] < share[r0] - 0.1 - 1e-9:
                feasibility[rule].append(f"{name} {budget}/dim: {chosen} {share[chosen]:.0%} against {r0} {share[r0]:.0%}")
        order = ("bayesian", "cma-es", "genetic", "random")
        lines += [
            f"| {name} | {budget} | {evaluations} | {' / '.join(f'{median[o]:.3g}' if o in median else '–' for o in order)} | "
            f"{' / '.join(f'{share[o]:.0%}' if o in share else '–' for o in order)} | {picks['R0']} | {picks['R1']} | {picks['R2']} |"
        ]
    lines += ["", "| Rule | Bayesian up to | Geometric mean of pick / best | Pick is best | Cells where the pick found nothing feasible |", "| --- | ---: | ---: | ---: | --- |"]
    gm = {}
    for rule, sc in scores.items():
        gm[rule] = math.exp(sum(sc["logs"]) / len(sc["logs"])) if sc["logs"] else math.inf
        lines += [f"| {rule} | {RULES[rule]} | ×{gm[rule]:.3f} | {sc['best']} of {sc['cells']} | {', '.join(sc['lost']) or 'none'} |"]
    lines += [""]
    shipped = "R0"
    for rule in ("R2", "R1"):
        better = gm[rule] <= 0.9 * gm["R0"]
        no_more_lost = len(scores[rule]["lost"]) <= len(scores["R0"]["lost"])
        kept = not feasibility[rule]
        verdict = "passes" if better and no_more_lost and kept else "fails"
        lines += [f"**{rule}: {verdict}.** ×{gm[rule] / gm['R0']:.2f} of R0's geometric mean (needs ≤ ×0.90); cells lost {len(scores[rule]['lost'])} against R0's {len(scores['R0']['lost'])}; feasibility: {'; '.join(feasibility[rule]) or 'kept everywhere'}.", ""]
        if verdict == "passes" and shipped == "R0":
            shipped = rule
    lines += [f"**Ships: {shipped}**, Bayesian optimization up to {RULES[shipped]} evaluations with limits on outputs.", ""]
    return "\n".join(lines) + "\n"
