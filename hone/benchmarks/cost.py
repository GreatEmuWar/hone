"""The cost of a Bayesian candidate: seconds to propose one, against the evaluations so far and the limits on outputs.

Each candidate costs a fit of k + 1 Gaussian processes, one for the objective and one per limit,
each O(n³) in the n evaluations so far, and a search over them. This measures that cost so later
changes can be held to it: a change may not raise seconds per candidate past ×1.2 of the saved
table at 500 evaluations without sign-off.

Each cell of the table starts an optimizer whose first generation is n points spread over the
space, so its history is n evaluations long without paying for n proposals first. The first
proposal after that fits every model's hyperparameters from scratch, which a real run pays only
once, and is reported on its own as the cold fit. The cell's seconds per candidate are from the
proposals after it, with the hyperparameters kept between fits and refitted every few
evaluations, as they are in a run that has reached n. The objective is a sphere and each limit a
random linear one, which costs no time itself, so the table is the optimizer's own cost.

Each cell is the mean over SEEDS seeds, each with its own limits and its own run: one seed alone
varies by up to about 15% between seeds at 500 evaluations, too close to the ceiling, while the
same seed repeats within about 5%.

Times depend on the machine, so the table records it, and a comparison against the table is
only fair on the same one.
"""
import cProfile
import io
import json
import os
import platform
import pstats
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from hone.config import BayesianConfig
from hone.optimizers.bayesian import BayesianOptimizer
from hone.parameters import Output, RangeParameter, SearchSpace

EVALUATIONS = (50, 100, 200, 500)
LIMITS = (0, 1, 2, 4, 8)
BATCH_SIZES = (1, 8)
DIMENSIONS = 10  # as g07, the problem with the most limits on the outputs suite
CANDIDATES = 24  # timed per cell and seed, after the cold fit: a multiple of every batch size
SEEDS = 3  # each cell is the mean over this many problems and runs; one seed alone varies by up to about 15%
CEILING = 1.2  # the most a later change may raise seconds per candidate, at CEILING_AT evaluations
CEILING_AT = 500


class CostCell(BaseModel):
    """Seconds per Bayesian candidate with this many evaluations so far, limits on outputs and candidates per batch."""
    evaluations: int
    limits: int
    batch_size: int
    seconds_per_candidate: float
    cold_fit_seconds: float  # the first proposal, fitting every model's hyperparameters from scratch


class CostProfile(BaseModel):
    """Where the time goes in one real run: the functions that took the most time themselves."""
    problem: str
    evaluations: int
    seconds: float
    top: list[tuple[str, int, float, float]]  # function, calls, its own seconds, cumulative seconds


class CostTable(BaseModel):
    """The cost table, the machine it was measured on, and one profiled run."""
    hone: str
    machine: dict[str, str]
    dimensions: int
    candidates: int
    seeds: int = 1
    cells: list[CostCell]
    profile: CostProfile | None = None


def machine() -> dict[str, str]:
    """What the times depend on: the processor, the cores, and the versions doing the arithmetic."""
    return {
        "platform": platform.platform(),
        "processor": _processor(),
        "cores": str(os.cpu_count()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
    }


def _processor() -> str:
    """The processor's model name where the system says it, else its architecture."""
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def _problem(dimensions: int, limits: int, seed: int = 0) -> tuple[SearchSpace, Any]:
    """A sphere in [-5, 5]^dimensions under `limits` random linear limits on outputs, each holding on about two thirds of the box."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((limits, dimensions))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    names = [f"x{i}" for i in range(dimensions)]
    space = SearchSpace(
        [RangeParameter(n, -5, 5) for n in names],
        [Output(f"g{j}") <= 1.25 for j in range(limits)],  # a unit direction times a uniform point has an sd of about 2.9
    )

    def objective(params: dict[str, Any]) -> float | dict[str, float]:
        x = np.array([params[n] for n in names])
        value = float(x @ x)
        if not limits:
            return value
        return {"objective": value, **{f"g{j}": float(directions[j] @ x) for j in range(limits)}}

    return space, objective


def measure_cell(
    evaluations: int, limits: int, batch_size: int, *, dimensions: int = DIMENSIONS, candidates: int = CANDIDATES, seed: int = 0
) -> CostCell:
    """Time the proposals of one cell: the cold fit, then `candidates` more in batches of batch_size."""
    if candidates % batch_size:
        raise ValueError(f"candidates {candidates} must be a multiple of batch_size {batch_size}")
    space, objective = _problem(dimensions, limits, seed)
    batches = candidates // batch_size + 1  # the first is the cold fit
    config = BayesianConfig(
        initial_samples=evaluations, population_size=batch_size, max_evaluations=evaluations + batches * batch_size, seed=seed
    )
    optimizer = BayesianOptimizer(space, config)
    for trial in optimizer.ask():
        optimizer.tell(trial, objective(trial.params))
    seconds = []
    for _ in range(batches):
        start = time.perf_counter()
        trials = optimizer.ask()
        seconds.append(time.perf_counter() - start)
        assert len(trials) == batch_size, f"expected a batch of {batch_size}, got {len(trials)}"
        for trial in trials:
            optimizer.tell(trial, objective(trial.params))
    return CostCell(
        evaluations=evaluations,
        limits=limits,
        batch_size=batch_size,
        seconds_per_candidate=sum(seconds[1:]) / candidates,
        cold_fit_seconds=seconds[0],
    )


def measure_cost(
    evaluations: tuple[int, ...] = EVALUATIONS,
    limits: tuple[int, ...] = LIMITS,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    *,
    dimensions: int = DIMENSIONS,
    candidates: int = CANDIDATES,
    seeds: int = SEEDS,
    progress: bool = False,
) -> list[CostCell]:
    """Every cell of the table, each the mean over seeds, one after another in this process so no two compete for a core."""
    cells = []
    for batch_size in batch_sizes:
        for n in evaluations:
            for k in limits:
                runs = [measure_cell(n, k, batch_size, dimensions=dimensions, candidates=candidates, seed=seed) for seed in range(seeds)]
                cell = CostCell(
                    evaluations=n,
                    limits=k,
                    batch_size=batch_size,
                    seconds_per_candidate=sum(r.seconds_per_candidate for r in runs) / seeds,
                    cold_fit_seconds=sum(r.cold_fit_seconds for r in runs) / seeds,
                )
                if progress:
                    print(f"  {n} evaluations, {k} limits, batches of {batch_size}: {cell.seconds_per_candidate:.3f}s per candidate", flush=True)
                cells.append(cell)
    return cells


def profile_run(evaluations: int = 200, top: int = 15, seed: int = 0) -> CostProfile:
    """Profile a real run of Bayesian optimization on g07 with its eight limits as outputs, one candidate at a time."""
    from .outputs import g07

    problem = g07()
    config = BayesianConfig(max_evaluations=evaluations, population_size=1, seed=seed)
    optimizer = BayesianOptimizer(problem.search_space, config)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    optimizer.optimize(problem.function)
    profiler.disable()
    seconds = time.perf_counter() - start
    stats = pstats.Stats(profiler, stream=io.StringIO())
    rows = []
    for (filename, line, name), (_, calls, own, cumulative, _) in stats.stats.items():  # type: ignore[attr-defined]
        rows.append((f"{_short(filename)}:{line}({name})", calls, own, cumulative))
    rows.sort(key=lambda row: -row[2])
    return CostProfile(problem=problem.name, evaluations=evaluations, seconds=seconds, top=rows[:top])


def _short(filename: str) -> str:
    """A profiled function's file, from the package it belongs to rather than the full path."""
    parts = Path(filename).parts
    for package in ("hone", "scipy", "numpy"):
        if package in parts:
            return "/".join(parts[len(parts) - parts[::-1].index(package) - 1:])
    return Path(filename).name


def ceiling_breaches(cells: list[CostCell], baseline: list[CostCell], ratio: float = CEILING, at: int = CEILING_AT) -> list[str]:
    """Cells at `at` evaluations whose seconds per candidate exceed ratio times the baseline's, described."""
    before = {(c.evaluations, c.limits, c.batch_size): c.seconds_per_candidate for c in baseline}
    found = []
    for cell in cells:
        key = (cell.evaluations, cell.limits, cell.batch_size)
        if cell.evaluations != at or key not in before:
            continue
        if cell.seconds_per_candidate > ratio * before[key]:
            found.append(
                f"{cell.evaluations} evaluations, {cell.limits} limits, batches of {cell.batch_size}: "
                f"{cell.seconds_per_candidate:.3f}s per candidate against {before[key]:.3f}s, ×{cell.seconds_per_candidate / before[key]:.2f}"
            )
    return found


def save_cost(table: CostTable, path: str | Path) -> None:
    Path(path).write_text(table.model_dump_json(indent=2) + "\n")


def load_cost(path: str | Path) -> CostTable:
    return CostTable.model_validate(json.loads(Path(path).read_text()))


def cost_markdown(table: CostTable) -> str:
    """The table as markdown: seconds per candidate by evaluations and limits, one table per batch size, then the cold fits and the profile."""
    m = table.machine
    lines = [
        f"Measured with hone {table.hone} on {m['processor']}, {m['cores']} cores, {m['platform']}, "
        f"Python {m['python']}, NumPy {m['numpy']}, SciPy {m['scipy']}: {table.dimensions} dimensions, "
        f"{table.candidates} candidates timed per cell after the cold fit, each cell the mean over {table.seeds} seeds.",
        "",
    ]
    limits = sorted({c.limits for c in table.cells})
    by_key = {(c.evaluations, c.limits, c.batch_size): c for c in table.cells}
    for batch_size in sorted({c.batch_size for c in table.cells}):
        lines += [f"### Seconds per candidate, batches of {batch_size}", ""]
        lines += ["| Evaluations | " + " | ".join(f"{k} limits" if k != 1 else "1 limit" for k in limits) + " |"]
        lines += ["| ---: | " + " | ".join("---:" for _ in limits) + " |"]
        for n in sorted({c.evaluations for c in table.cells if c.batch_size == batch_size}):
            row = [f"{by_key[(n, k, batch_size)].seconds_per_candidate:.3f}" if (n, k, batch_size) in by_key else "" for k in limits]
            lines += [f"| {n} | " + " | ".join(row) + " |"]
        lines += [""]
    lines += ["### Cold fit, seconds, batches of 1", ""]
    lines += ["| Evaluations | " + " | ".join(f"{k} limits" if k != 1 else "1 limit" for k in limits) + " |"]
    lines += ["| ---: | " + " | ".join("---:" for _ in limits) + " |"]
    smallest = min(c.batch_size for c in table.cells)
    for n in sorted({c.evaluations for c in table.cells}):
        row = [f"{by_key[(n, k, smallest)].cold_fit_seconds:.2f}" if (n, k, smallest) in by_key else "" for k in limits]
        lines += [f"| {n} | " + " | ".join(row) + " |"]
    lines += [""]
    if table.profile:
        p = table.profile
        lines += [
            f"### Profile: {p.problem}, {p.evaluations} evaluations, one candidate at a time, {p.seconds:.0f}s",
            "",
            "| Function | Calls | Own seconds | Cumulative seconds |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [f"| `{name}` | {calls:,} | {own:.2f} | {cumulative:.2f} |" for name, calls, own, cumulative in p.top]
        lines += [""]
    return "\n".join(lines)
