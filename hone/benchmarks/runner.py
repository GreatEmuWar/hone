"""Run optimizers on benchmark problems and summarize how close they get to each optimum."""
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from hone.config import GeneticOptimizerConfig, OptimizerConfig
from hone.optimizers import GeneticOptimizer, Optimizer, RandomSearch

from .problems import Problem

# Builds an optimizer for a problem, an evaluation budget and a seed.
OptimizerFactory = Callable[[Problem, int, int], Optimizer]

POPULATION_SIZE = 50


def genetic(problem: Problem, budget: int, seed: int) -> Optimizer:
    config = GeneticOptimizerConfig(
        population_size=POPULATION_SIZE, generations=max(budget // POPULATION_SIZE, 1), seed=seed
    )
    return GeneticOptimizer(problem.search_space, config)


def random_search(problem: Problem, budget: int, seed: int) -> Optimizer:
    config = OptimizerConfig(population_size=POPULATION_SIZE, generations=max(budget // POPULATION_SIZE, 1), seed=seed)
    return RandomSearch(problem.search_space, config)


DEFAULT_OPTIMIZERS: dict[str, OptimizerFactory] = {"genetic": genetic, "random": random_search}


@dataclass(frozen=True)
class Run:
    """One optimizer on one problem with one seed."""
    problem: str
    optimizer: str
    seed: int
    error: float  # best value found minus the problem's optimum; 0 means the optimum was found
    evaluations: int
    trace: list[tuple[int, float]]  # (evaluations so far, error of the best so far), once per generation


@dataclass(frozen=True)
class Summary:
    """All runs of one optimizer on one problem."""
    problem: str
    optimizer: str
    runs: int
    median_error: float
    lower_quartile: float
    upper_quartile: float
    solved: float  # share of runs whose error is at most the tolerance

    @classmethod
    def of(cls, runs: list[Run], tolerance: float) -> "Summary":
        errors = sorted(run.error for run in runs)
        quartiles = statistics.quantiles(errors, n=4, method="inclusive") if len(errors) > 1 else errors * 3
        return cls(
            problem=runs[0].problem,
            optimizer=runs[0].optimizer,
            runs=len(runs),
            median_error=statistics.median(errors),
            lower_quartile=quartiles[0],
            upper_quartile=quartiles[2],
            solved=sum(error <= tolerance for error in errors) / len(errors),
        )


def run_benchmark(
    problems: Iterable[Problem],
    optimizers: dict[str, OptimizerFactory] | None = None,
    seeds: Iterable[int] = range(10),
    budget_per_dimension: int = 1000,
) -> list[Run]:
    """Run every optimizer on every problem with every seed, at budget_per_dimension × dimensions evaluations."""
    optimizers = optimizers if optimizers is not None else DEFAULT_OPTIMIZERS
    seeds = list(seeds)
    runs = []
    for problem in problems:
        budget = budget_per_dimension * problem.dimensions
        for name, factory in optimizers.items():
            for seed in seeds:
                result = factory(problem, budget, seed).optimize(problem.function)
                trace, best = [], float("inf")
                for generation in result.generations:
                    if generation.best_fitness is not None:
                        best = min(best, generation.best_fitness - problem.optimum)
                    trace.append((generation.evaluations, max(best, 0.0)))
                runs.append(
                    Run(
                        problem=problem.name,
                        optimizer=name,
                        seed=seed,
                        error=max(result.best_fitness - problem.optimum, 0.0),
                        evaluations=result.evaluations,
                        trace=trace,
                    )
                )
    return runs


def summarize(runs: list[Run], tolerance: float = 1e-4) -> list[Summary]:
    """One Summary per problem and optimizer, in the order the runs were made."""
    groups: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.problem, run.optimizer), []).append(run)
    return [Summary.of(group, tolerance) for group in groups.values()]


def to_markdown(summaries: list[Summary], tolerance: float = 1e-4) -> str:
    """A table of results, one row per problem, with a column group per optimizer."""
    optimizers = list(dict.fromkeys(s.optimizer for s in summaries))
    by_key = {(s.problem, s.optimizer): s for s in summaries}
    header = "| Problem | " + " | ".join(f"{o} median | {o} solved" for o in optimizers) + " |"
    rule = "| --- | " + " | ".join("---: | ---:" for _ in optimizers) + " |"
    rows = []
    for problem in dict.fromkeys(s.problem for s in summaries):
        cells = []
        for o in optimizers:
            s = by_key.get((problem, o))
            cells += [f"{s.median_error:.3g}", f"{s.solved:.0%}"] if s else ["", ""]
        rows.append(f"| {problem} | " + " | ".join(cells) + " |")
    note = f"Median error over the runs (best value found minus the optimum); solved means error ≤ {tolerance:g}."
    return "\n".join([header, rule, *rows, "", note]) + "\n"


def save_summaries(summaries: list[Summary], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(s) for s in summaries], indent=2) + "\n")


def load_summaries(path: str | Path) -> list[Summary]:
    return [Summary(**s) for s in json.loads(Path(path).read_text())]


def regressions(current: list[Summary], baseline: list[Summary], factor: float = 10.0, floor: float = 1e-8) -> list[str]:
    """Problems where an optimizer's median error grew by more than factor against the baseline.

    Errors below floor count as floor, so changes among values that are already essentially
    zero don't count as regressions.
    """
    previous = {(s.problem, s.optimizer): s for s in baseline}
    found = []
    for s in current:
        before = previous.get((s.problem, s.optimizer))
        if before is None:
            continue
        if max(s.median_error, floor) > factor * max(before.median_error, floor):
            found.append(
                f"{s.optimizer} on {s.problem}: median error {s.median_error:.3g}, baseline {before.median_error:.3g}"
            )
    return found
