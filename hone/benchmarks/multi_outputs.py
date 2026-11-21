"""Multi-objective problems with constraints on outputs, for measuring noise with several objectives.

- osy: Osyczka and Kundu's problem (1995), six variables, two objectives and six limits. The
  whole of its front lies on limits, in pieces with an analytic form.
- tnk: Tanaka et al.'s problem (1995), two variables, two objectives and two limits, with a
  disconnected front along a wavy limit.
- cantilever_outputs: the cantilever beam of examples/cantilever_beam.py, with the bending
  stress reported as an output instead of a function constraint.

Every limit is reported by the function as an output, so a design only learns that it breaks
one by being evaluated. Each problem's true front gives its ideal, nadir and scaled
hypervolume, as in multi.py. The noisy versions add Gaussian noise to every objective and
every output, for the noisy multi-objective benchmark.
"""
import functools
import math
from typing import Any

import numpy as np

from hone.parameters import ChoiceParameter, Output, RangeParameter, SearchSpace
from hone.pareto import hypervolume, non_dominated

from .multi import REFERENCE, MultiProblem, _front_hypervolume

TAGS = frozenset({"constrained", "outputs"})


def _front_2d(values: np.ndarray) -> np.ndarray:
    """The rows of values, two columns to minimize, that no other row dominates: a sort and a sweep."""
    order = np.lexsort((values[:, 1], values[:, 0]))
    best = np.minimum.accumulate(values[order, 1])
    return order[np.concatenate([[True], values[order, 1][1:] < best[:-1]])]


def _sum_front(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pairs (i, j) of rows of a and b, two columns to minimize, whose sum no other pair's sum dominates.

    Done in blocks of rows of a, so the table of every sum never exists at once.
    """
    step = max(1, 1_000_000 // len(b))
    firsts, seconds, sums = [], [], []
    for start in range(0, len(a), step):
        table = (a[start : start + step, None, :] + b[None, :, :]).reshape(-1, 2)
        keep = _front_2d(table)
        firsts.append(start + keep // len(b))
        seconds.append(keep % len(b))
        sums.append(table[keep])
    first, second, total = np.concatenate(firsts), np.concatenate(seconds), np.vstack(sums)
    keep = _front_2d(total)
    return first[keep], second[keep]


# --- OSY -------------------------------------------------------------------------


def _osy(x1: Any, x2: Any, x3: Any, x4: Any, x5: Any, x6: Any) -> tuple[Any, Any, list[Any]]:
    """OSY's two objectives and six limits, each satisfied when at least 0, on numbers or on arrays.

    Osyczka and Kundu (1995), as in Deb et al.'s NSGA-II paper (2002) and pymoo's OSY:
    minimize f1 = -[25(x1-2)^2 + (x2-2)^2 + (x3-1)^2 + (x4-4)^2 + (x5-1)^2] and f2 = sum of x_i^2,
    with x1 + x2 - 2 >= 0, 6 - x1 - x2 >= 0, 2 - x2 + x1 >= 0, 2 - x1 + 3 x2 >= 0,
    4 - (x3-3)^2 - x4 >= 0 and (x5-3)^2 + x6 - 4 >= 0.
    """
    f1 = -(25 * (x1 - 2) ** 2 + (x2 - 2) ** 2 + (x3 - 1) ** 2 + (x4 - 4) ** 2 + (x5 - 1) ** 2)
    f2 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    limits = [x1 + x2 - 2, 6 - x1 - x2, 2 - x2 + x1, 2 - x1 + 3 * x2, 4 - (x3 - 3) ** 2 - x4, (x5 - 3) ** 2 + x6 - 4]
    return f1, f2, limits


@functools.cache
def _osy_front(resolution: int = 2001) -> tuple[np.ndarray, np.ndarray]:
    """The designs (x1 to x6) and objectives (f1, f2) of OSY's true front, built from its analytic form.

    The front has x4 = x6 = 0 and x5 = 1 or 5, and (x1, x2) on the edges of the polygon that the
    first four limits cut out of the plane (its vertices are (0, 2), (2, 0), (5, 1) and (2, 4)),
    with x3 anywhere from 1 to 5 (Deb, 2001, the table of the front's five pieces). Only the
    edges g1 and g4 and the vertices (5, 1) and (0, 2) turn out to be on it. The front is the
    non-dominated part of every combination of resolution values of x3, 2 values of x5 and
    4 * resolution points along the polygon's edges. Each group of variables is filtered on its
    own first, since a design that is dominated within its group is dominated in the whole.
    """
    corners = [(0, 2), (2, 0), (5, 1), (2, 4)]
    t = np.linspace(0, 1, resolution)
    edges = np.vstack(
        [np.column_stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]) for a, b in zip(corners, corners[1:] + corners[:1])]
    )
    groups = [([0, 1], edges), ([2], np.linspace(1, 5, resolution)[:, None]), ([4], np.array([[1.0], [5.0]]))]

    def evaluate(designs: np.ndarray) -> np.ndarray:
        f1, f2, _ = _osy(*designs.T)
        return np.column_stack([f1, f2])

    designs = np.zeros((1, 6))
    for columns, values in groups:
        rows = np.zeros((len(values), 6))
        rows[:, columns] = values
        rows = rows[_front_2d(evaluate(rows))]
        first, second = _sum_front(evaluate(designs), evaluate(rows) - evaluate(np.zeros((1, 6))))
        designs = designs[first] + rows[second]
    return designs, evaluate(designs)


@functools.cache
def _scale(front: str) -> tuple[tuple[float, float], tuple[float, float], float]:
    """The ideal, nadir and scaled hypervolume of the named two-objective true front."""
    values = _osy_front()[1] if front == "osy" else _tnk_front()
    ideal = (float(values[:, 0].min()), float(values[:, 1].min()))
    nadir = (float(values[:, 0].max()), float(values[:, 1].max()))
    return ideal, nadir, _front_hypervolume(values[:, 0], values[:, 1], ideal, nadir)


def _two_objective(name: str, function: Any, space: SearchSpace, front: str, tags: frozenset[str]) -> MultiProblem:
    ideal, nadir, volume = _scale(front)
    names = ["f1", "f2"]
    return MultiProblem(
        name=name,
        function=function,
        search_space=space,
        objectives={n: "minimize" for n in names},
        ideal=dict(zip(names, ideal)),
        nadir=dict(zip(names, nadir)),
        true_hypervolume=volume,
        tags=tags,
    )


def osy() -> MultiProblem:
    """Osyczka and Kundu's problem: two objectives under six limits, all of them reported as outputs."""
    bounds = [(0, 10), (0, 10), (1, 5), (0, 6), (1, 5), (0, 10)]
    parameters = [RangeParameter(f"x{i}", low, high) for i, (low, high) in enumerate(bounds, 1)]

    def osy(x1: float, x2: float, x3: float, x4: float, x5: float, x6: float) -> dict[str, float]:
        f1, f2, limits = _osy(x1, x2, x3, x4, x5, x6)
        return {"f1": float(f1), "f2": float(f2), **{f"g{i}": float(g) for i, g in enumerate(limits, 1)}}

    space = SearchSpace(parameters, [Output(f"g{i}") >= 0 for i in range(1, 7)])
    return _two_objective("osy", osy, space, "osy", TAGS)


# --- TNK -------------------------------------------------------------------------


def _tnk_limits(x1: Any, x2: Any) -> tuple[Any, Any]:
    """TNK's two limits, c1 >= 0 and c2 <= 0.5, on numbers or arrays.

    The angle is atan2(x1, x2): the same as arctan(x1 / x2) for x2 > 0, and defined at x2 = 0.
    """
    c1 = x1**2 + x2**2 - 1 - 0.1 * np.cos(16 * np.arctan2(x1, x2))
    return c1, (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2


@functools.cache
def _tnk_front(resolution: int = 200_001) -> np.ndarray:
    """The (f1, f2) = (x1, x2) points of TNK's true front: the limit c1 = 0 swept by angle, inside the limit c2.

    Feasible designs are outside the wavy curve r(a)^2 = 1 + 0.1 cos(16 a), at radius r from the
    origin and angle a from the x2 axis, and inside the circle of c2; the front is the part of
    that curve inside the circle that no other point of it dominates.
    """
    angle = np.linspace(0, np.pi / 2, resolution)
    radius = np.sqrt(1 + 0.1 * np.cos(16 * angle))
    points = np.column_stack([radius * np.sin(angle), radius * np.cos(angle)])
    points = points[(points[:, 0] - 0.5) ** 2 + (points[:, 1] - 0.5) ** 2 <= 0.5]
    return points[_front_2d(points)]


def tnk() -> MultiProblem:
    """Tanaka et al.'s problem: minimize x1 and x2 under a wavy limit and a circle, both reported as outputs."""
    parameters = [RangeParameter("x1", 0, math.pi), RangeParameter("x2", 0, math.pi)]

    def tnk(x1: float, x2: float) -> dict[str, float]:
        c1, c2 = _tnk_limits(x1, x2)
        return {"f1": x1, "f2": x2, "c1": float(c1), "c2": float(c2)}

    space = SearchSpace(parameters, [Output("c1") >= 0, Output("c2") <= 0.5])
    return _two_objective("tnk", tnk, space, "tnk", TAGS)


# --- the cantilever --------------------------------------------------------------

LENGTH = 1.0  # m
LOAD = 5_000.0  # N, at the free end
SAFETY = 1.5

# Young's modulus (Pa), density (kg/m³), strength (Pa) and a rough price (per kg) of each material.
MATERIALS = {
    "steel": (200e9, 7850, 250e6, 1.5),
    "aluminium": (69e9, 2700, 276e6, 4.0),
    "carbon fibre": (135e9, 1600, 1000e6, 40.0),
}
WIDTH, HEIGHT = (10.0, 100.0), (10.0, 150.0)  # mm


def _beam(material: str, width: Any, height: Any) -> tuple[Any, Any, Any, Any]:
    """The mass in kg, cost, tip deflection in mm and stress ratio, on numbers or arrays.

    The stress ratio is the bending stress at the fixed end over the material's allowed stress,
    its strength over the safety factor, so a limit that varies with the material is one number.
    """
    modulus, density, strength, price = MATERIALS[material]
    b, h = width / 1000, height / 1000
    mass = density * b * h * LENGTH
    deflection = LOAD * LENGTH**3 / (3 * modulus * b * h**3 / 12)
    stress = 6 * LOAD * LENGTH / (b * h**2)
    return mass, price * mass, 1000 * deflection, stress / (strength / SAFETY)


def _cantilever_front(n: int) -> tuple[list[tuple[str, float, float]], np.ndarray]:
    """The designs and (mass, cost, deflection) of the true front found on an n by n grid of widths and heights per material.

    Feasible points only (stress ratio at most 1). Within a material cost is proportional to
    mass, so each material's front is found on mass and deflection by a sweep; the three are
    then pooled and the three-objective front taken.
    """
    widths, heights = (a.ravel() for a in np.meshgrid(np.linspace(*WIDTH, n), np.linspace(*HEIGHT, n)))
    designs: list[tuple[str, float, float]] = []
    values = []
    for material in MATERIALS:
        mass, cost, deflection, ratio = _beam(material, widths, heights)
        feasible = np.flatnonzero(ratio <= 1 + 1e-9)
        keep = feasible[_front_2d(np.column_stack([mass[feasible], deflection[feasible]]))]
        designs += [(material, float(widths[i]), float(heights[i])) for i in keep]
        values.append(np.column_stack([mass[keep], cost[keep], deflection[keep]]))
    pooled = np.vstack(values)
    front = non_dominated(pooled)
    return [designs[i] for i in front], pooled[front]


def _cantilever_reference_front(n: int) -> tuple[dict[str, float], dict[str, float], float, int]:
    """The ideal, nadir, scaled hypervolume and number of points of the cantilever's true front on an n by n grid."""
    _, values = _cantilever_front(n)
    ideal, nadir = values.min(axis=0), values.max(axis=0)
    volume = hypervolume((values - ideal) / (nadir - ideal), np.full(3, REFERENCE))
    names = ["mass", "cost", "deflection"]
    return dict(zip(names, map(float, ideal))), dict(zip(names, map(float, nadir))), float(volume), len(values)


# The cantilever's scale and hypervolume are those of the front on a noise-free grid of
# CANTILEVER_GRID widths by CANTILEVER_GRID heights for each material, made once with
# _cantilever_reference_front(CANTILEVER_GRID) (about a minute). The hypervolume moves by under
# 1e-4 from a grid of 100 on a side (1.31451) to 2000 (1.31524): 800 gives 1.31523.
CANTILEVER_GRID = 2000
CANTILEVER_SCALE = (
    {"mass": 1.0743771885942972, "cost": 13.79471735867934, "deflection": 0.29629629629629634},
    {"mass": 117.75, "cost": 960.0, "deflection": 48.93119688271012},
)
CANTILEVER_HYPERVOLUME = 1.3152384829585464  # a front of 9,656 points


def cantilever_outputs() -> MultiProblem:
    """The cantilever beam's mass, cost and deflection, with the bending stress as an output.

    The output is `stress_ratio`, the stress over the material's allowed stress (its strength
    over the safety factor), limited to at most 1: a limit that varies with the material is
    written as a ratio so it is one number. examples/cantilever_beam.py keeps its function constraint.
    """
    space = SearchSpace(
        [ChoiceParameter("material", list(MATERIALS)), RangeParameter("width", *WIDTH), RangeParameter("height", *HEIGHT)],
        [Output("stress_ratio") <= 1],
    )

    def cantilever(material: str, width: float, height: float) -> dict[str, float]:
        mass, cost, deflection, ratio = _beam(material, width, height)
        return {"mass": mass, "cost": cost, "deflection": deflection, "stress_ratio": ratio}

    ideal, nadir = CANTILEVER_SCALE
    return MultiProblem(
        name="cantilever-outputs",
        function=cantilever,
        search_space=space,
        objectives={"mass": "minimize", "cost": "minimize", "deflection": "minimize"},
        ideal=ideal,
        nadir=nadir,
        true_hypervolume=CANTILEVER_HYPERVOLUME,
        tags=TAGS | {"engineering"},
    )


def multi_output_problems() -> list[MultiProblem]:
    """The three problems, noise-free."""
    return [osy(), tnk(), cantilever_outputs()]


# Noise for the noisy versions: about 1% of each quantity's typical value, as a standard deviation
# (objectives, then outputs), as ENGINEERING_NOISE does for the engineering designs, to two
# significant figures. The typical value of an objective is its median over the true front. An
# output with a limit of some size, such as the stress ratio's 1, is taken at the limit, where
# a design is judged (ENGINEERING_NOISE does the same). An output whose limit is 0 has no such
# size, so its typical value is its median magnitude over designs spread uniformly across the
# box that the true front's designs span.
MULTI_OUTPUT_NOISE: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "osy": (
        {"f1": 2.2, "f2": 0.26},
        {"g1": 0.018, "g2": 0.022, "g3": 0.032, "g4": 0.035, "g5": 0.03, "g6": 0.03},
    ),
    "tnk": ({"f1": 0.0074, "f2": 0.0074}, {"c1": 0.004, "c2": 0.005}),
    "cantilever-outputs": ({"mass": 0.14, "cost": 0.91, "deflection": 0.016}, {"stress_ratio": 0.01}),
}


def noisy_multi_output_problems() -> list[MultiProblem]:
    """The three problems with noise on every objective and every output."""
    problems = []
    for problem in multi_output_problems():
        noise, output_noise = MULTI_OUTPUT_NOISE[problem.name]
        problems.append(
            problem.model_copy(
                update={"name": f"noisy-{problem.name}", "noise": noise, "output_noise": output_noise, "tags": problem.tags | {"noisy"}}
            )
        )
    return problems
