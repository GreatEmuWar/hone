"""Run optimizers on benchmark problems and summarize how close they get to each optimum."""
import functools
import json
import math
import multiprocessing
import statistics
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict

from hone.config import BayesianConfig, CMAESConfig, GeneticOptimizerConfig, OptimizerConfig
from hone.optimizers import CMAES, BayesianOptimizer, GeneticOptimizer, Optimizer, RandomSearch, UnsupportedSearchSpace

from .problems import Problem

# Builds an optimizer for a problem, an evaluation budget and a seed. It may raise
# UnsupportedSearchSpace for a problem it cannot search, which is then skipped, or
# SkipBudget for a budget it isn't meant for.
OptimizerFactory = Callable[[Problem, int, int], Optimizer]


class SkipBudget(Exception):
    """Raised by an optimizer factory to leave out a budget it isn't meant for."""


# Bayesian optimization takes a fraction of a second per evaluation, which suits expensive
# objectives but makes long benchmark runs slow, so it runs only up to this many evaluations.
BAYESIAN_MAX_EVALUATIONS = 200

POPULATION_SIZE = 50  # random search's batch size, which doesn't change its results

# Evaluations per dimension that the benchmark gives each run. The small budgets show how
# an optimizer does when each evaluation is expensive, as with a slow simulation.
BUDGETS = (10, 20, 50, 100, 1000)

FLOOR = 1e-8  # errors below this count as equal when comparing results


def genetic(problem: Problem, budget: int, seed: int) -> Optimizer:
    config = GeneticOptimizerConfig(max_evaluations=budget, seed=seed, noise=problem.noise > 0)
    return GeneticOptimizer(problem.search_space, config)


def cma_es(problem: Problem, budget: int, seed: int) -> Optimizer:
    config = CMAESConfig(generations=budget, max_evaluations=budget, seed=seed, noise=problem.noise > 0)
    return CMAES(problem.search_space, config)


def bayesian(problem: Problem, budget: int, seed: int) -> Optimizer:
    if budget > BAYESIAN_MAX_EVALUATIONS:
        raise SkipBudget(f"{budget} evaluations is more than Bayesian optimization is run with")
    config = BayesianConfig(max_evaluations=budget, seed=seed, noise=problem.noise > 0)
    return BayesianOptimizer(problem.search_space, config)


def random_search(problem: Problem, budget: int, seed: int) -> Optimizer:
    config = OptimizerConfig(
        population_size=POPULATION_SIZE, generations=budget, max_evaluations=budget, seed=seed, noise=problem.noise > 0
    )
    return RandomSearch(problem.search_space, config)


def bayesian_batch(size: int) -> OptimizerFactory:
    """A factory for Bayesian optimization choosing size candidates per batch, as it would with that many workers."""

    def factory(problem: Problem, budget: int, seed: int) -> Optimizer:
        if budget > BAYESIAN_MAX_EVALUATIONS:
            raise SkipBudget(f"{budget} evaluations is more than Bayesian optimization is run with")
        config = BayesianConfig(max_evaluations=budget, population_size=size, seed=seed, noise=problem.noise > 0)
        return BayesianOptimizer(problem.search_space, config)

    return factory


def batch_optimizers(sizes: Iterable[int]) -> dict[str, OptimizerFactory]:
    """Bayesian optimization at each batch size, named bayesian-4 and so on, to add to the optimizers of a run."""
    return {f"bayesian-{size}": bayesian_batch(size) for size in sizes}


DEFAULT_OPTIMIZERS: dict[str, OptimizerFactory] = {
    "genetic": genetic,
    "cma-es": cma_es,
    "bayesian": bayesian,
    "random": random_search,
}


class Run(BaseModel):
    """One optimizer on one problem with one seed and one budget."""
    model_config = ConfigDict(frozen=True)

    problem: str
    optimizer: str
    budget_per_dimension: int
    seed: int
    error: float  # best value found minus the problem's optimum; 0 means the optimum was found, infinity that nothing feasible was
    evaluations: int
    trace: list[tuple[int, float]]  # (evaluations so far, error of the best so far), once per generation
    output_constrained: bool = False  # whether the problem has constraints on outputs, so a feasible design has to be found
    first_feasible: int | None = None  # evaluations up to the end of the generation that first found a feasible design; None if none did


class Summary(BaseModel):
    """All runs of one optimizer on one problem with one budget."""
    model_config = ConfigDict(frozen=True)

    problem: str
    optimizer: str
    budget_per_dimension: int
    runs: int
    median_error: float
    lower_quartile: float
    upper_quartile: float
    solved: float  # share of runs whose error is at most the tolerance
    # With constraints on outputs: the share of runs that found a feasible design, and the median
    # evaluations to the first one, with a run that found none counting as infinity. None without.
    feasible: float | None = None
    median_first_feasible: float | None = None

    @classmethod
    def of(cls, runs: list[Run], tolerance: float) -> "Summary":
        errors = sorted(run.error for run in runs)
        quartiles = [_quantile(errors, q) for q in (0.25, 0.5, 0.75)]
        feasible = median_first_feasible = None
        if runs[0].output_constrained:
            feasible = sum(math.isfinite(run.error) for run in runs) / len(runs)
            median_first_feasible = statistics.median(math.inf if run.first_feasible is None else run.first_feasible for run in runs)
        return cls(
            problem=runs[0].problem,
            optimizer=runs[0].optimizer,
            budget_per_dimension=runs[0].budget_per_dimension,
            runs=len(runs),
            median_error=statistics.median(errors),
            lower_quartile=quartiles[0],
            upper_quartile=quartiles[2],
            solved=sum(error <= tolerance for error in errors) / len(errors),
            feasible=feasible,
            median_first_feasible=median_first_feasible,
        )


def _quantile(ordered: list[float], q: float) -> float:
    """The q quantile of sorted values, interpolated as statistics.quantiles' inclusive method does, allowing infinity."""
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if ordered[low] == ordered[high] or position == low:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run_benchmark(
    problems: Iterable[Problem],
    optimizers: dict[str, OptimizerFactory] | None = None,
    seeds: Iterable[int] = range(10),
    budgets_per_dimension: Iterable[int] = BUDGETS,
    jobs: int = 1,
) -> list[Run]:
    """Run every optimizer on every problem with every seed and every budget.

    A budget of b evaluations per dimension gives a run b × dimensions evaluations. Each
    budget is a separate run, so an optimizer can choose its settings from the budget.
    An optimizer that cannot search a problem's parameters is skipped for that problem, and
    one whose factory raises SkipBudget is skipped at that budget. With jobs > 1, runs are
    shared among that many processes where the platform can fork them; the results are the same.
    """
    problems = list(problems)
    optimizers = optimizers if optimizers is not None else DEFAULT_OPTIMIZERS
    seeds, budgets = list(seeds), list(budgets_per_dimension)
    tasks = []
    for index, problem in enumerate(problems):
        for name, factory in optimizers.items():
            for per_dimension in budgets:
                try:
                    if seeds:
                        factory(problem, per_dimension * problem.dimensions, seeds[0])
                except UnsupportedSearchSpace:
                    break
                except SkipBudget:
                    continue
                tasks += [(index, name, per_dimension, seed) for seed in seeds]

    global _shared
    _shared = (problems, optimizers)
    if jobs == 1 or "fork" not in multiprocessing.get_all_start_methods():
        return [_task(task) for task in tasks]
    # Forked processes inherit the problems and factories, which needn't be picklable.
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(_task, tasks))


_shared: tuple[list[Problem], dict[str, OptimizerFactory]] = ([], {})  # what run_benchmark's tasks refer to


def _task(task: tuple[int, str, int, int]) -> Run:
    index, name, per_dimension, seed = task
    problems, optimizers = _shared
    problem = problems[index]
    optimizer = optimizers[name](problem, per_dimension * problem.dimensions, seed)
    return _run(problem, name, per_dimension, seed, optimizer)


def _run(problem: Problem, name: str, per_dimension: int, seed: int, optimizer: Optimizer) -> Run:
    noisy = problem.noise > 0 or bool(problem.output_noise)
    result = optimizer.optimize(_with_noise(problem, seed) if noisy else problem.function)
    constrained = bool(problem.search_space.output_constraints)

    def truth(fitness: float, params: dict, feasible: bool) -> tuple[float, bool]:
        # A noisy run is scored by the noise-free value of the point it chose, not by the noisy
        # value it saw there, which is likely to be a lucky one, and by whether the point truly
        # satisfies the constraints on outputs, not by whether it seemed to.
        return _true(problem, params) if noisy else (fitness, feasible)

    # With constraints on outputs, generations' bests are of feasible points only, so the trace is too.
    trace, best, first_feasible = [], float("inf"), None
    for generation in result.generations:
        if generation.best_fitness is not None and generation.best_params is not None:
            value, feasible = truth(generation.best_fitness, generation.best_params, True)
            if feasible:
                best = min(best, value - problem.optimum)
                if first_feasible is None:
                    first_feasible = generation.evaluations
        trace.append((generation.evaluations, max(best, 0.0)))
    value, feasible = truth(result.best_fitness, result.best_params, result.feasible)
    return Run(
        problem=problem.name,
        optimizer=name,
        budget_per_dimension=per_dimension,
        seed=seed,
        error=max(value - problem.optimum, 0.0) if feasible else math.inf,
        evaluations=result.evaluations,
        trace=trace,
        output_constrained=constrained,
        first_feasible=first_feasible if constrained else None,
    )


def _true(problem: Problem, params: dict) -> tuple[float, bool]:
    """The noise-free value at params, and whether params satisfies the problem's constraints on outputs."""
    value = problem.function(**params)
    if not isinstance(value, dict):
        return float(value), True
    outputs = {name: float(value[name]) for name in problem.search_space.outputs}
    return float(value["objective"]), problem.search_space.output_violation(params, outputs) == 0


def _with_noise(problem: Problem, seed: int) -> Callable[..., Any]:
    """problem's function plus Gaussian noise, repeatable for the seed.

    The noise has standard deviation problem.noise on the value and problem.output_noise[name]
    on each output the objective reports, each drawn independently.
    """
    rng = np.random.default_rng([seed, zlib.crc32(problem.name.encode())])

    @functools.wraps(problem.function)
    def noisy(**params):
        value = problem.function(**params)
        if not isinstance(value, dict):
            return value + problem.noise * rng.standard_normal()
        noise = {"objective": problem.noise, **problem.output_noise}
        return {name: v + noise.get(name, 0.0) * rng.standard_normal() for name, v in value.items()}

    return noisy


def summarize(runs: list[Run], tolerance: float = 1e-4) -> list[Summary]:
    """One Summary per problem, optimizer and budget, in the order the runs were made."""
    groups: dict[tuple[str, str, int], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.problem, run.optimizer, run.budget_per_dimension), []).append(run)
    return [Summary.of(group, tolerance) for group in groups.values()]


def wins(summaries: list[Summary], floor: float = FLOOR) -> dict[int, dict[str, int]]:
    """For each budget, how many problems each optimizer has the lowest median error on.

    Ties count for every optimizer in them, and errors below floor count as equal. Only
    problems that at least two optimizers ran on are counted.
    """
    optimizers = list(dict.fromkeys(s.optimizer for s in summaries))
    groups: dict[tuple[int, str], list[Summary]] = {}
    for s in summaries:
        groups.setdefault((s.budget_per_dimension, s.problem), []).append(s)
    counts: dict[int, dict[str, int]] = {}
    for (budget, _), group in groups.items():
        tally = counts.setdefault(budget, dict.fromkeys(optimizers, 0))
        if len(group) < 2:
            continue
        best = min(max(s.median_error, floor) for s in group)
        for s in group:
            tally[s.optimizer] += max(s.median_error, floor) == best
    return dict(sorted(counts.items()))


def to_markdown(summaries: list[Summary], tolerance: float = 1e-4) -> str:
    """An overview of wins at each budget, then a table per budget with a row per problem."""
    optimizers = list(dict.fromkeys(s.optimizer for s in summaries))
    budgets = sorted({s.budget_per_dimension for s in summaries})
    by_key = {(s.problem, s.optimizer, s.budget_per_dimension): s for s in summaries}
    problems = list(dict.fromkeys(s.problem for s in summaries))

    lines = [
        "## Overview",
        "",
        "Problems on which each optimizer has the lowest median error, at each budget. Ties count for each "
        f"optimizer in them, and errors below {FLOOR:g} count as equal.",
        "",
        "| Evaluations per dimension | " + " | ".join(optimizers) + " |",
        "| ---: | " + " | ".join("---:" for _ in optimizers) + " |",
        *(f"| {b} | " + " | ".join(str(t[o]) for o in optimizers) + " |" for b, t in wins(summaries).items()),
    ]
    for budget in budgets:
        lines += [
            "",
            f"## {budget} evaluations per dimension",
            "",
            "| Problem | " + " | ".join(f"{o} median | {o} solved" for o in optimizers) + " |",
            "| --- | " + " | ".join("---: | ---:" for _ in optimizers) + " |",
        ]
        for problem in problems:
            cells = []
            for o in optimizers:
                s = by_key.get((problem, o, budget))
                cells += [f"{s.median_error:.3g}", f"{s.solved:.0%}"] if s else ["n/a", "n/a"]
            lines.append(f"| {problem} | " + " | ".join(cells) + " |")
        constrained = [p for p in problems if any(_feasibility(by_key.get((p, o, budget))) for o in optimizers)]
        if constrained:
            lines += [
                "",
                "Finding a feasible design:",
                "",
                "| Problem | " + " | ".join(f"{o} feasible | {o} first" for o in optimizers) + " |",
                "| --- | " + " | ".join("---: | ---:" for _ in optimizers) + " |",
            ]
            for problem in constrained:
                cells = []
                for o in optimizers:
                    cells += _feasibility(by_key.get((problem, o, budget))) or ["n/a", "n/a"]
                lines.append(f"| {problem} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Median error over the runs (best value found minus the optimum); solved means error ≤ {tolerance:g}. "
        "n/a: the optimizer does not support the problem's parameters, or isn't run at this budget.",
    ]
    if any(s.feasible is not None for s in summaries):
        lines += [
            "",
            "Feasible: the share of runs that found a design satisfying every constraint on the outputs. First: the median "
            "evaluations to the first such design, counting a run that found none as infinity, so none means most runs found none.",
        ]
    return "\n".join(lines) + "\n"


def _feasibility(s: Summary | None) -> list[str] | None:
    """The feasible and first cells for a summary of a problem with constraints on outputs, else None."""
    if s is None or s.feasible is None or s.median_first_feasible is None:
        return None
    first = "none" if math.isinf(s.median_first_feasible) else f"{s.median_first_feasible:.0f}"
    return [f"{s.feasible:.0%}", first]


def save_summaries(summaries: list[Summary], path: str | Path) -> None:
    Path(path).write_text(json.dumps([s.model_dump() for s in summaries], indent=2) + "\n")


def load_summaries(path: str | Path) -> list[Summary]:
    return [Summary.model_validate(s) for s in json.loads(Path(path).read_text())]


def regressions(current: list[Summary], baseline: list[Summary], factor: float = 10.0, floor: float = FLOOR) -> list[str]:
    """Problems and budgets where an optimizer's median error grew by more than factor against the baseline.

    Errors below floor count as floor, so changes among values that are already essentially
    zero don't count as regressions.
    """
    previous = {(s.problem, s.optimizer, s.budget_per_dimension): s for s in baseline}
    found = []
    for s in current:
        before = previous.get((s.problem, s.optimizer, s.budget_per_dimension))
        if before is None:
            continue
        if max(s.median_error, floor) > factor * max(before.median_error, floor):
            found.append(
                f"{s.optimizer} on {s.problem} at {s.budget_per_dimension} evaluations per dimension: "
                f"median error {s.median_error:.3g}, baseline {before.median_error:.3g}"
            )
    return found
