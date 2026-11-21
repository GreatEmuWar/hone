"""Multi-objective test problems, and a runner that scores fronts by hypervolume.

- zdt1, zdt2, zdt3 and zdt4: two objectives, with fronts that are convex, bend inward,
  come in five disconnected pieces, and (zdt4) sit behind many false fronts. They are shifted
  so the optimal designs lie inside the bounds rather than on them, which clipping and blend
  crossover would otherwise favour, with the distance from the front measured by a smooth
  quadratic. rotated-zdt1 turns the distance variables by a fixed rotation, so they interact.
- dtlz2: three objectives, whose front is an eighth of a sphere, shifted the same way.
- welded-beam-2: the welded beam's cost against its deflection, under its stress, buckling
  and geometry constraints, with a front found by long reference runs.

Each problem's objectives are scaled from 0 at the true front's ideal to 1 at its nadir, and
hypervolume is measured in that space up to a reference point of 1.1 in every objective, so
scores compare across runs and versions. A run's gap is the true front's hypervolume less
the run's: 0 is the true front itself.
"""
import functools
import math
import multiprocessing
import statistics
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict

from hone.config import BayesianConfig, GeneticOptimizerConfig, MultiObjectiveBayesianConfig, MultiObjectiveConfig, NSGA2Config
from hone.optimizers import (
    BayesianOptimizer,
    GeneticOptimizer,
    MultiObjectiveBayesian,
    MultiObjectiveOptimizer,
    MultiObjectiveRandomSearch,
)
from hone.optimizers.genetic import NSGA2
from hone.parameters import Constraint, RangeParameter, SearchSpace
from hone.pareto import Direction, hypervolume, non_dominated

from .constrained import E, L, P, _buckling, _shear
from .runner import BAYESIAN_MAX_EVALUATIONS, BUDGETS, POPULATION_SIZE, SkipBudget
from .problems import _rotation

REFERENCE = 1.1  # the reference point in every scaled objective


class MultiProblem(BaseModel):
    """A function of several objectives, its search space, and its true front's scale and hypervolume."""
    model_config = ConfigDict(frozen=True)

    name: str
    function: Callable[..., dict[str, float]]
    search_space: SearchSpace
    objectives: dict[str, Direction]
    ideal: dict[str, float]  # the true front's best value of each objective
    nadir: dict[str, float]  # and its worst
    true_hypervolume: float  # of the true front, scaled, up to REFERENCE
    tags: frozenset[str] = frozenset()
    # Gaussian noise the benchmark adds: a standard deviation for each objective and for each
    # output that the search space's constraints on outputs use. Empty means none.
    noise: dict[str, float] = {}
    output_noise: dict[str, float] = {}

    @property
    def dimensions(self) -> int:
        return len(self.search_space.parameters)

    @property
    def outputs(self) -> list[str]:
        """The quantities the function reports besides the objectives, for its constraints on outputs."""
        return [name for name in self.search_space.outputs if name not in self.objectives]

    @property
    def noisy(self) -> bool:
        """Whether the benchmark adds noise to any objective or output."""
        return any(v > 0 for v in self.noise.values()) or any(v > 0 for v in self.output_noise.values())

    def scaled(self, values: dict[str, float]) -> np.ndarray:
        """values scaled to minimize, from 0 at the ideal to 1 at the nadir."""
        return np.array([(values[n] - self.ideal[n]) / (self.nadir[n] - self.ideal[n]) for n in self.objectives])

    def hypervolume(self, points: Iterable[dict[str, float]]) -> float:
        vectors = np.array([self.scaled(v) for v in points]).reshape(-1, len(self.objectives))
        return hypervolume(vectors, np.full(len(self.objectives), REFERENCE))


def _centre(name: str, size: int, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(zlib.crc32(f"{name}/centre".encode())).uniform(low, high, size)


def _front_hypervolume(f1: np.ndarray, f2: np.ndarray, ideal: tuple[float, float], nadir: tuple[float, float]) -> float:
    """The scaled hypervolume of a two-objective front sampled densely."""
    points = np.column_stack([(f1 - ideal[0]) / (nadir[0] - ideal[0]), (f2 - ideal[1]) / (nadir[1] - ideal[1])])
    return hypervolume(points, np.full(2, REFERENCE))


ZDT_SHAPES: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "zdt1": lambda f1, g: 1 - np.sqrt(f1 / g),
    "zdt2": lambda f1, g: 1 - (f1 / g) ** 2,
    "zdt3": lambda f1, g: 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1),
    "zdt4": lambda f1, g: 1 - np.sqrt(f1 / g),
}


@functools.cache
def _zdt_front(shape: str) -> tuple[tuple[float, float], tuple[float, float], float]:
    """The ideal, nadir and scaled hypervolume of a ZDT front, from 200,001 points along it."""
    f1 = np.linspace(0, 1, 200_001)
    f2 = ZDT_SHAPES[shape](f1, np.ones_like(f1))
    vectors = np.column_stack([f1, f2])
    if shape == "zdt3":  # the disconnected front: keep only the points nothing dominates
        order = np.argsort(f1)
        best = np.minimum.accumulate(f2[order])
        keep = order[np.concatenate([[True], f2[order][1:] < best[:-1]])]
        vectors = vectors[keep]
    ideal, nadir = (float(vectors[:, 0].min()), float(vectors[:, 1].min())), (float(vectors[:, 0].max()), float(vectors[:, 1].max()))
    return ideal, nadir, _front_hypervolume(vectors[:, 0], vectors[:, 1], ideal, nadir)


def zdt(shape: str, dimensions: int, rotate: bool = False) -> MultiProblem:
    name = f"{'rotated-' if rotate else ''}{shape}-{dimensions}d"
    wide = shape == "zdt4"
    first = RangeParameter("x0", 0, 1)
    rest = [RangeParameter(f"x{i}", -5 if wide else 0, 5 if wide else 1) for i in range(1, dimensions)]
    centre = _centre(name, dimensions - 1, -2 if wide else 0.2, 2 if wide else 0.8)
    width = np.maximum(centre, 1 - centre)
    rotation = _rotation(name, dimensions - 1) if rotate else np.eye(dimensions - 1)
    shape_of = ZDT_SHAPES[shape]

    def function(**params: float) -> dict[str, float]:
        f1 = float(params["x0"])
        x = np.array([params[p.name] for p in rest], dtype=float)
        if wide:
            z = x - centre
            g = 1 + 10 * len(z) + float(np.sum(z**2 - 10 * np.cos(4 * np.pi * z)))
        else:
            d = rotation @ ((x - centre) / width)
            g = 1 + 9 * float(np.mean(d**2))
        return {"f1": f1, "f2": float(g * shape_of(np.array(f1), np.array(g)))}

    function.__name__ = name.replace("-", "_")
    ideal, nadir, volume = _zdt_front(shape)
    return MultiProblem(
        name=name,
        function=function,
        search_space=SearchSpace([first, *rest]),
        objectives={"f1": "minimize", "f2": "minimize"},
        ideal={"f1": ideal[0], "f2": ideal[1]},
        nadir={"f1": nadir[0], "f2": nadir[1]},
        true_hypervolume=volume,
        tags=frozenset({"zdt", "rotated"} if rotate else {"zdt"}),
    )


def dtlz2(dimensions: int = 7) -> MultiProblem:
    """Three objectives on an eighth of the unit sphere; the first two variables place a point on it."""
    name = f"dtlz2-{dimensions}d"
    parameters = [RangeParameter(f"x{i}", 0, 1) for i in range(dimensions)]
    centre = _centre(name, dimensions - 2, 0.25, 0.75)

    def function(**params: float) -> dict[str, float]:
        x = np.array([params[p.name] for p in parameters], dtype=float)
        g = float(np.sum((x[2:] - centre) ** 2))
        a, b = x[0] * np.pi / 2, x[1] * np.pi / 2
        return {"f1": (1 + g) * math.cos(a) * math.cos(b), "f2": (1 + g) * math.cos(a) * math.sin(b), "f3": (1 + g) * math.sin(a)}

    function.__name__ = name.replace("-", "_")
    return MultiProblem(
        name=name,
        function=function,
        search_space=SearchSpace(parameters),
        objectives={"f1": "minimize", "f2": "minimize", "f3": "minimize"},
        ideal={"f1": 0.0, "f2": 0.0, "f3": 0.0},
        nadir={"f1": 1.0, "f2": 1.0, "f3": 1.0},
        true_hypervolume=REFERENCE**3 - math.pi / 6,  # the reference cube less an eighth of the unit ball
        tags=frozenset({"dtlz"}),
    )


# The welded beam's front isn't known in closed form: its scale and hypervolume are those of
# the front of WELDED_BEAM_REFERENCE_SEARCH.
WELDED_BEAM_REFERENCE_SEARCH = "NSGA2 with 50,000 evaluations, seeds 0 to 2, combined"
WELDED_BEAM_SCALE = (
    {"cost": 1.7279633786270103, "deflection": 0.0004390400449635821},
    {"cost": 35.308391288678266, "deflection": 0.014327892424181738},
)
WELDED_BEAM_HYPERVOLUME = 1.1323322942394598  # a front of 2,867 points


def welded_beam() -> MultiProblem:
    """The welded beam's cost against its deflection under the end load (Deb's two-objective version)."""
    weld, length = RangeParameter("weld", 0.125, 5.0), RangeParameter("length", 0.1, 10.0)
    thickness, width = RangeParameter("thickness", 0.1, 10.0), RangeParameter("width", 0.125, 5.0)

    def welded_beam_2(weld: float, length: float, thickness: float, width: float) -> dict[str, float]:
        cost = 1.10471 * weld**2 * length + 0.04811 * thickness * width * (14 + length)
        return {"cost": cost, "deflection": 4 * P * L**3 / (E * thickness**3 * width)}

    def shear(weld: float, length: float, thickness: float) -> float:
        return _shear(weld, length, thickness) - 13600

    def buckling(thickness: float, width: float) -> float:
        return P - _buckling(thickness, width)

    space = SearchSpace(
        [weld, length, thickness, width],
        [Constraint.function(shear), 6 * P * L / (width * thickness**2) <= 30000, weld <= width, Constraint.function(buckling)],
    )
    ideal, nadir = WELDED_BEAM_SCALE
    return MultiProblem(
        name="welded-beam-2",
        function=welded_beam_2,
        search_space=space,
        objectives={"cost": "minimize", "deflection": "minimize"},
        ideal=ideal,
        nadir=nadir,
        true_hypervolume=WELDED_BEAM_HYPERVOLUME,
        tags=frozenset({"constrained", "engineering"}),
    )


def multi_problems() -> list[MultiProblem]:
    """The multi-objective suite."""
    return [
        zdt("zdt1", 8),
        zdt("zdt1", 8, rotate=True),
        zdt("zdt2", 8),
        zdt("zdt3", 8),
        zdt("zdt4", 6),
        dtlz2(7),
        welded_beam(),
    ]


# --- optimizers --------------------------------------------------------------------

# A factory takes the problem, the budget and the run's seed, and optionally the points to evaluate first.
MultiFactory = Callable[..., Any]
Points = list[dict[str, Any]]


def nsga2(problem: MultiProblem, budget: int, seed: int, *, initial_points: Points | None = None) -> MultiObjectiveOptimizer:
    config = NSGA2Config(objectives=problem.objectives, max_evaluations=budget, seed=seed)
    return NSGA2(problem.search_space, config, initial_points=initial_points)


def _bayesian(method: str) -> MultiFactory:
    def factory(problem: MultiProblem, budget: int, seed: int, *, initial_points: Points | None = None) -> MultiObjectiveOptimizer:
        if budget > BAYESIAN_MAX_EVALUATIONS:
            raise SkipBudget(f"{budget} evaluations is more than Bayesian optimization is run with")
        config = MultiObjectiveBayesianConfig(objectives=problem.objectives, max_evaluations=budget, seed=seed, method=method)
        return MultiObjectiveBayesian(problem.search_space, config, initial_points=initial_points)

    return factory


def random_search(problem: MultiProblem, budget: int, seed: int, *, initial_points: Points | None = None) -> MultiObjectiveOptimizer:
    config = MultiObjectiveConfig(
        objectives=problem.objectives, population_size=POPULATION_SIZE, generations=budget, max_evaluations=budget, seed=seed
    )
    return MultiObjectiveRandomSearch(problem.search_space, config, initial_points=initial_points)


def first_generation(optimizer: MultiObjectiveOptimizer) -> Points:
    """The points a fresh optimizer would evaluate first, in order, without evaluating anything.

    The optimizer is used up by this: build another to run.
    """
    return [dict(trial.params) for trial in optimizer.ask()]


def paired_first_generation(factory: MultiFactory, offset: int = 1000) -> MultiFactory:
    """factory's optimizer, searching with seed + offset but starting from the first generation that factory's own seed makes.

    A run of the pair and a run of factory itself evaluate the same first generation in the same
    order, so with the same noise seed they see the same noisy values there and differ only after it.
    """

    def paired(problem: MultiProblem, budget: int, seed: int, **kwargs: Any) -> Any:
        first = first_generation(factory(problem, budget, seed))
        return factory(problem, budget, seed + offset, initial_points=first, **kwargs)

    return paired


def weighted_sums(problem: MultiProblem, budget: int, seed: int) -> "WeightedSums":
    return WeightedSums(problem, budget, seed)


class WeightedSums:
    """The way to search for trade-offs without multi-objective optimization: several single-objective runs, each of a weighted sum.

    The budget is shared among runs with weights spread evenly over the scaled objectives (5
    runs for two objectives, 6 for three), each by the optimizer choose_optimizer would pick
    for its share, and the front is that of every point any of them evaluated. The objectives
    are scaled by the true front's ideal and nadir, which a real user wouldn't know: a
    generous baseline.
    """

    def __init__(self, problem: MultiProblem, budget: int, seed: int) -> None:
        self.problem, self.budget, self.seed = problem, budget, seed
        m = len(problem.objectives)
        if m == 2:
            self.weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, 5)]
        else:
            self.weights = [np.array(w, dtype=float) / 2 for w in [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]]

    def run(self) -> tuple[list[dict[str, float]], int]:
        """Every point's values, and the evaluations made; the points' parameters are left in self.params, in the same order."""
        seen: list[dict[str, float]] = []
        self.params: list[dict[str, Any]] = []
        evaluations = 0
        for i, weights in enumerate(self.weights):
            share = self.budget // len(self.weights) + (i < self.budget % len(self.weights))

            def weighted(**params: Any) -> float:
                values = self.problem.function(**params)
                seen.append(values)
                self.params.append(dict(params))
                return float(weights @ self.problem.scaled(values))

            if share <= BAYESIAN_MAX_EVALUATIONS:
                optimizer: Any = BayesianOptimizer(self.problem.search_space, BayesianConfig(max_evaluations=share, seed=self.seed + i))
            else:
                optimizer = GeneticOptimizer(self.problem.search_space, GeneticOptimizerConfig(max_evaluations=share, seed=self.seed + i))
            evaluations += optimizer.optimize(weighted).evaluations
        return seen, evaluations


DEFAULT_MULTI_OPTIMIZERS: dict[str, MultiFactory] = {
    "nsga2": nsga2,
    "bayesian-parego": _bayesian("parego"),
    "bayesian-models": _bayesian("models"),
    "random": random_search,
    "weighted-sums": weighted_sums,
}


# --- runs and summaries ------------------------------------------------------------


class MultiRun(BaseModel):
    """One optimizer on one problem with one seed and one budget."""
    model_config = ConfigDict(frozen=True)

    problem: str
    optimizer: str
    budget_per_dimension: int
    seed: int
    gap: float  # the true front's scaled hypervolume less the run's front's; 0 is the true front
    front_size: int
    evaluations: int
    front: list[list[float]]  # the run's front, scaled, for attainment surfaces
    # The fields below judge the reported points by their noise-free values, which differ from
    # the values the run saw only when the problem is noisy.
    optimizer_seed: int | None = None  # the seed the optimizer searched with, which a factory may offset from the run's
    reported: int = 0  # the points the optimizer reported as its front
    truly_feasible: int = 0  # of those, how many satisfy every constraint on outputs when evaluated without noise
    any_truly_feasible: bool = False
    true_gap: float = 0.0  # the true front's scaled hypervolume less that of the truly feasible reported points, noise-free


class MultiSummary(BaseModel):
    """All runs of one optimizer on one problem with one budget."""
    model_config = ConfigDict(frozen=True)

    problem: str
    optimizer: str
    budget_per_dimension: int
    runs: int
    median_gap: float
    lower_quartile: float
    upper_quartile: float
    attainment: list[list[float]] | None = None  # for two objectives, the median attainment surface's corners, scaled


def multi_tasks(
    problems: list[MultiProblem], optimizers: dict[str, MultiFactory], seeds: Iterable[int], budgets_per_dimension: Iterable[int]
) -> list[tuple[int, str, int, int]]:
    """Every run run_multi_benchmark makes, in its order: (problem index, optimizer, budget per dimension, seed)."""
    seeds, budgets = list(seeds), list(budgets_per_dimension)
    tasks = []
    for index, problem in enumerate(problems):
        for name, factory in optimizers.items():
            for per_dimension in budgets:
                try:
                    if seeds:
                        factory(problem, per_dimension * problem.dimensions, seeds[0])
                except SkipBudget:
                    continue
                tasks += [(index, name, per_dimension, seed) for seed in seeds]
    return tasks


def run_multi_benchmark(
    problems: Iterable[MultiProblem],
    optimizers: dict[str, MultiFactory] | None = None,
    seeds: Iterable[int] = range(10),
    budgets_per_dimension: Iterable[int] = BUDGETS,
    jobs: int = 1,
    shard: tuple[int, int] | None = None,
) -> list[MultiRun]:
    """Run every optimizer on every problem with every seed and budget, as run_benchmark does for one objective.

    With shard=(index, count), only a count-th of the runs are made, dealt as run_benchmark deals
    them: round-robin by problem, budget and seed, so every optimizer's run of one seed is made on
    the same machine and each problem's seeds spread over every shard.
    """
    if shard is not None and not 0 <= shard[0] < shard[1]:
        raise ValueError(f"shard {shard[0]} of {shard[1]} doesn't exist")
    problems = list(problems)
    optimizers = optimizers if optimizers is not None else DEFAULT_MULTI_OPTIMIZERS
    tasks = multi_tasks(problems, optimizers, seeds, budgets_per_dimension)
    if shard is not None:
        groups = list(dict.fromkeys((index, per_dimension, seed) for index, _, per_dimension, seed in tasks))
        mine = set(groups[shard[0] :: shard[1]])
        tasks = [task for task in tasks if (task[0], task[2], task[3]) in mine]
    global _shared
    _shared = (problems, optimizers)
    if jobs == 1 or "fork" not in multiprocessing.get_all_start_methods():
        return [_task(task) for task in tasks]
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(_task, tasks))


_shared: tuple[list[MultiProblem], dict[str, MultiFactory]] = ([], {})


def _with_noise(problem: MultiProblem, seed: int) -> Callable[..., dict[str, float]]:
    """problem's function plus Gaussian noise, repeatable for the run's seed.

    Every quantity the function returns, in the order it returns them, gets an independent draw
    from one stream, with the standard deviation problem.noise or problem.output_noise gives it
    (none for a quantity in neither).
    """
    rng = np.random.default_rng([seed, zlib.crc32(problem.name.encode())])
    sds = {**problem.output_noise, **problem.noise}

    @functools.wraps(problem.function)
    def noisy(**params: Any) -> dict[str, float]:
        return {name: v + sds.get(name, 0.0) * rng.standard_normal() for name, v in problem.function(**params).items()}

    return noisy


def _task(task: tuple[int, str, int, int]) -> MultiRun:
    index, name, per_dimension, seed = task
    problems, optimizers = _shared
    problem = problems[index]
    optimizer = optimizers[name](problem, per_dimension * problem.dimensions, seed)
    reported: list[tuple[dict[str, Any], dict[str, float]]]  # the reported points' parameters and the values the run saw
    if isinstance(optimizer, WeightedSums):
        if problem.noisy:
            raise ValueError("weighted-sums is run on noise-free problems only")
        seen, evaluations = optimizer.run()
        optimizer_seed: int | None = seed
        every = np.array([problem.scaled(v) for v in seen]).reshape(-1, len(problem.objectives))
        reported = [(optimizer.params[i], seen[i]) for i in (non_dominated(every) if len(every) else [])]
    else:
        result = optimizer.optimize(_with_noise(problem, seed) if problem.noisy else problem.function)
        evaluations, optimizer_seed = result.evaluations, optimizer.config.seed
        reported = [(point.params, point.values) for point in result.front]
        seen = [values for _, values in reported]
    vectors = np.array([problem.scaled(v) for v in seen]).reshape(-1, len(problem.objectives))
    front = vectors[non_dominated(vectors)] if len(vectors) else vectors
    volume = hypervolume(front, np.full(len(problem.objectives), REFERENCE))
    truth = [(params, problem.function(**params)) for params, _ in reported]
    feasible = [values for params, values in truth if _truly_feasible(problem, params, values)]
    true_volume = problem.hypervolume(feasible)  # a point beyond the reference counts for nothing
    return MultiRun(
        problem=problem.name,
        optimizer=name,
        budget_per_dimension=per_dimension,
        seed=seed,
        gap=max(problem.true_hypervolume - volume, 0.0),
        front_size=len(front),
        evaluations=evaluations,
        front=front.tolist(),
        optimizer_seed=optimizer_seed,
        reported=len(reported),
        truly_feasible=len(feasible),
        any_truly_feasible=bool(feasible),
        true_gap=max(problem.true_hypervolume - true_volume, 0.0),
    )


def _truly_feasible(problem: MultiProblem, params: dict[str, Any], values: dict[str, float]) -> bool:
    """Whether the noise-free values at params satisfy every constraint on outputs."""
    outputs = {name: float(values[name]) for name in problem.search_space.outputs}
    return problem.search_space.output_violation(params, outputs) == 0


def attainment_surface(fronts: list[np.ndarray], share: float = 0.5) -> list[list[float]]:
    """The corners of the surface that the given share of two-objective fronts attain, scaled values to minimize.

    A point is attained by a front when some point of the front is at least as good on both
    objectives. The median surface (share 0.5) is the usual picture of where a method's
    fronts typically lie.
    """
    xs = sorted({float(x) for front in fronts for x in front[:, 0]} if fronts else set())
    needed = max(1, math.ceil(share * len(fronts)))
    corners: list[list[float]] = []
    for x in xs:
        best = sorted(float(front[front[:, 0] <= x, 1].min()) if np.any(front[:, 0] <= x) else math.inf for front in fronts)
        y = best[needed - 1]
        if math.isfinite(y) and (not corners or y < corners[-1][1]):
            corners.append([x, y])
    return corners


def summarize_multi(runs: list[MultiRun]) -> list[MultiSummary]:
    groups: dict[tuple[str, str, int], list[MultiRun]] = {}
    for run in runs:
        groups.setdefault((run.problem, run.optimizer, run.budget_per_dimension), []).append(run)
    summaries = []
    for (problem, optimizer, budget), group in groups.items():
        gaps = sorted(run.gap for run in group)
        quartiles = statistics.quantiles(gaps, n=4, method="inclusive") if len(gaps) > 1 else gaps * 3
        fronts = [np.array(run.front).reshape(-1, len(run.front[0]) if run.front else 2) for run in group]
        two = all(f.shape[1] == 2 for f in fronts)
        summaries.append(
            MultiSummary(
                problem=problem,
                optimizer=optimizer,
                budget_per_dimension=budget,
                runs=len(group),
                median_gap=statistics.median(gaps),
                lower_quartile=quartiles[0],
                upper_quartile=quartiles[2],
                attainment=attainment_surface(fronts) if two else None,
            )
        )
    return summaries


def multi_markdown(summaries: list[MultiSummary]) -> str:
    """Median hypervolume gaps, by budget, with the lowest of each row in bold."""
    optimizers = list(dict.fromkeys(s.optimizer for s in summaries))
    problems = list(dict.fromkeys(s.problem for s in summaries))
    lines = []
    for budget in sorted({s.budget_per_dimension for s in summaries}):
        by = {(s.problem, s.optimizer): s.median_gap for s in summaries if s.budget_per_dimension == budget}
        lines += [f"## {budget} evaluations per dimension", "", "| Problem | " + " | ".join(optimizers) + " |", "| --- |" + " ---: |" * len(optimizers)]
        for problem in problems:
            row = [by.get((problem, o)) for o in optimizers]
            known = [g for g in row if g is not None]
            best = min(known) if known else None
            cells = ["n/a" if g is None else (f"**{g:.3g}**" if g <= (best or 0) * (1 + 1e-9) + 1e-12 else f"{g:.3g}") for g in row]
            lines.append(f"| {problem} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(
        f"Median hypervolume gap over the runs: the true front's hypervolume less the run's, with objectives scaled from 0 at "
        f"the true front's ideal to 1 at its nadir and a reference point of {REFERENCE}. 0 is the true front; lower is better."
    )
    return "\n".join(lines) + "\n"
