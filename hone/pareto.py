"""Pareto fronts: the best trade-offs between several objectives, and the result of a multi-objective run.

One point dominates another when it is at least as good on every objective and better on at
least one. The front is the set of points that no other point dominates. Values within a small
relative tolerance of each other count as equal, so rounding doesn't make two points out of one.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from hone.result import RunInfo

if TYPE_CHECKING:
    from hone.optimizers.multi import MultiObjectiveOptimizer

TOLERANCE = 1e-9  # values this close, relative to their size, count as equal
AUGMENTATION = 0.05  # the weight of the sum in the augmented Chebyshev distance, which breaks its ties

type Direction = Literal["minimize", "maximize"]


class ParetoPoint(BaseModel):
    """A point on the front: its parameters and the value of each objective there."""
    model_config = ConfigDict(frozen=True)

    params: dict[str, Any]
    values: dict[str, float]  # with noise, the average of every evaluation of the point
    evaluations: int = 1  # how many times the point was evaluated
    standard_error: dict[str, float] | None = None  # with noise, of each average, once evaluated twice


class Front(BaseModel):
    """The best trade-offs found: no point on the front is beaten on every objective by any point evaluated.

    Points are in order of the first objective, best first. Choose among them with best(),
    for the best value of one objective within limits on the others, or closest(), for the
    most balanced point.
    """
    model_config = ConfigDict(frozen=True)

    objectives: dict[str, Direction]
    points: list[ParetoPoint]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[ParetoPoint]:  # type: ignore[override]
        return iter(self.points)

    def __getitem__(self, index: int) -> ParetoPoint:
        return self.points[index]

    @property
    def ideal(self) -> dict[str, float]:
        """The best value of each objective on the front, each from its own point."""
        return {name: _best_of(self.objectives[name], [p.values[name] for p in self.points]) for name in self.objectives}

    @property
    def nadir(self) -> dict[str, float]:
        """The worst value of each objective on the front."""
        worst = {"minimize": "maximize", "maximize": "minimize"}
        return {name: _best_of(worst[self.objectives[name]], [p.values[name] for p in self.points]) for name in self.objectives}

    def best(self, objective: str, where: dict[str, float] | None = None) -> ParetoPoint:
        """The point with the best value of objective, among those within the limits in where.

        A limit is a bound on another objective in its own direction: at most the limit for an
        objective to minimize, at least it for one to maximize. front.best("accuracy",
        where={"cost": 20}) is the most accurate point costing at most 20.
        """
        self._check_names([objective, *(where or {})])
        allowed = [p for p in self.points if all(self._within(name, p.values[name], limit) for name, limit in (where or {}).items())]
        if not allowed:
            limits = ", ".join(f"{name} {'at most' if self.objectives[name] == 'minimize' else 'at least'} {limit:g}" for name, limit in (where or {}).items())
            raise ValueError(f"no point on the front has {limits}; the front's best values are {self.ideal}")
        direction = self.objectives[objective]
        return min(allowed, key=lambda p: p.values[objective] if direction == "minimize" else -p.values[objective])

    def closest(self, weights: dict[str, float] | None = None) -> ParetoPoint:
        """The point closest to the ideal by augmented Chebyshev distance, on the front's own scale.

        Each objective is scaled so 0 is its best value on the front and 1 its worst, then a
        point's distance is its largest weighted scaled gap from the ideal, plus a twentieth of
        their sum to break ties. Unlike a weighted sum, this can pick any point on the front,
        including those in a dent of it. Weights default to 1 each; give an objective more
        weight to count its gap for more.
        """
        weights = {name: 1.0 for name in self.objectives} | (weights or {})
        self._check_names(list(weights))
        ideal, nadir = self.ideal, self.nadir
        names = list(self.objectives)
        gaps = np.array([[_scaled(p.values[n], ideal[n], nadir[n]) for n in names] for p in self.points]) * [weights[n] for n in names]
        distance = gaps.max(axis=1) + AUGMENTATION * gaps.sum(axis=1)
        return self.points[int(np.argmin(distance))]

    def hypervolume(self, reference: dict[str, float]) -> float:
        """The size of the region of objective space the front dominates, up to the reference point.

        The reference gives a value of each objective worse than any point that should count.
        A better front always has a larger hypervolume.
        """
        self._check_names(list(reference))
        names = list(self.objectives)
        points = np.array([_minimizing(p.values, self.objectives) for p in self.points]).reshape(-1, len(names))
        return hypervolume(points, _minimizing(reference, self.objectives))

    def _within(self, name: str, value: float, limit: float) -> bool:
        return value <= limit if self.objectives[name] == "minimize" else value >= limit

    def _check_names(self, names: list[str]) -> None:
        unknown = sorted(set(names) - set(self.objectives))
        if unknown:
            raise ValueError(f"no objective is named {', '.join(map(repr, unknown))}; the objectives are {', '.join(self.objectives)}")


class ParetoGenerationStats(BaseModel):
    """Summary of one generation of a multi-objective run."""
    model_config = ConfigDict(frozen=True)

    generation: int
    evaluations: int  # evaluations up to the end of this generation, failed ones included
    failures: int = 0  # failed evaluations in this generation
    front_size: int = 0  # points on the front of everything evaluated so far
    hypervolume: float | None = None  # of that front, up to the run's reference point


class ParetoProgress(BaseModel):
    """What a multi-objective optimizer's optimize() passes its callback after each generation."""
    model_config = ConfigDict(frozen=True)

    generation: ParetoGenerationStats  # the generation just finished
    evaluations: int  # evaluations made so far, failed ones included
    front: Front  # the front of everything evaluated so far
    elapsed: float  # seconds since the run started


class ParetoResult(BaseModel):
    """Outcome of a multi-objective run: the front of everything evaluated. Save it with save() and read it back with load().

    Two results are equal when they found the same things; their run records are not compared.
    """
    model_config = ConfigDict(frozen=True)

    front: Front
    generations: list[ParetoGenerationStats]
    evaluations: int  # evaluations made, failed ones included
    failures: int = 0
    stopped: bool = False  # whether the run was stopped before it finished, by its callback or Ctrl+C
    run: RunInfo | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParetoResult):
            return NotImplemented
        return self.model_dump(exclude={"run"}) == other.model_dump(exclude={"run"})

    def rebuild_optimizer(
        self, optimizer_class: "type[MultiObjectiveOptimizer] | None" = None, *, functions: dict[str, Callable[..., float]] | None = None
    ) -> "MultiObjectiveOptimizer":
        """A new optimizer set up exactly as this result's was, to repeat or extend the run.

        As with Result.rebuild_optimizer: pass the class of an optimizer that isn't one of
        hone's own, and the functions of any function constraints, by name.
        """
        from hone.optimizers import multi

        if self.run is None:
            raise ValueError("this result has no record of its run")
        if optimizer_class is None:
            builtin = {cls.__name__: cls for cls in multi.builtin()}
            if self.run.optimizer not in builtin:
                raise ValueError(f"{self.run.optimizer} isn't one of hone's optimizers; pass its class to rebuild it")
            optimizer_class = builtin[self.run.optimizer]
        config = optimizer_class.config_class.model_validate(self.run.config)
        return optimizer_class(self.run.search_space.with_functions(functions or {}), config, initial_points=self.run.initial_points or None)

    def save(self, filepath: str | Path) -> None:
        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        return cls.model_validate_json(Path(filepath).read_text())


# --- the mathematics, on arrays of values to minimize ---------------------------


def _minimizing(values: dict[str, float], objectives: dict[str, Direction]) -> np.ndarray:
    """values as a vector to minimize: objectives to maximize are negated."""
    return np.array([values[name] if direction == "minimize" else -values[name] for name, direction in objectives.items()], dtype=float)


def _best_of(direction: str, values: list[float]) -> float:
    return min(values) if direction == "minimize" else max(values)


def _scaled(value: float, ideal: float, nadir: float) -> float:
    """How far value is from the ideal toward the nadir, from 0 to 1."""
    return 0.0 if nadir == ideal else (value - ideal) / (nadir - ideal)


def _tolerance(values: np.ndarray) -> np.ndarray:
    return TOLERANCE * np.maximum(1.0, np.abs(values))


def dominates_or_ties(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether a is at least as good as b on every objective, within the tolerance. Values to minimize."""
    return bool(np.all(a <= b + _tolerance(b)))


def non_dominated(values: np.ndarray) -> list[int]:
    """The rows of values, to minimize, that no other row dominates, in order of the first column.

    Rows equal within the tolerance count once: the one that comes first in the order, by
    value and then by row, is kept. So with rows in the order points were evaluated, the
    earliest of tied points stays.
    """
    n = len(values)
    if n == 0:
        return []
    order = np.lexsort([np.arange(n), *values.T[::-1]])  # by the first column, then the next, then by row
    front: list[int] = []
    for i in order:
        point = values[i]
        if front:
            kept = values[front]
            if np.any(np.all(kept <= point + _tolerance(point), axis=1)):
                continue  # dominated by, or tied with, a point already on the front
            # Keep only the points this one doesn't dominate.
            beaten = np.all(point <= kept + _tolerance(kept), axis=1)
            front = [j for j, gone in zip(front, beaten) if not gone]
        front.append(int(i))
    return sorted(front, key=lambda j: (tuple(values[j]), j))


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    """The volume dominated by points, to minimize, and bounded by reference.

    Exact: a sweep in two dimensions, and slices along the last objective above that, which
    is fast for the two or three objectives hone measures.
    """
    points = np.asarray(points, dtype=float).reshape(-1, len(reference))
    points = points[np.all(points < reference, axis=1)]
    if len(points) == 0:
        return 0.0
    if len(reference) == 1:
        return float(reference[0] - points[:, 0].min())
    if len(reference) == 2:
        volume, height = 0.0, reference[1]
        for x, y in points[np.lexsort((points[:, 1], points[:, 0]))]:
            if y < height:
                volume += (reference[0] - x) * (height - y)
                height = y
        return float(volume)
    order = np.argsort(points[:, -1], kind="stable")
    points = points[order]
    volume = 0.0
    for i in range(len(points)):
        top = points[i + 1, -1] if i + 1 < len(points) else reference[-1]
        if top > points[i, -1]:
            volume += (top - points[i, -1]) * hypervolume(points[: i + 1, :-1], reference[:-1])
    return float(volume)
