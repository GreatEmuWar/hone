"""Problems with constraints on outputs, chosen for what they say about finding a feasible design.

Each limit is reported by the objective, as a simulation would report it, so an optimizer only
learns that a design breaks it after evaluating it. Besides the engineering designs from the
constrained suite, there are problems whose feasible region is known exactly or is tiny:

- ball: the feasible region is a ball of known volume, so the feasible fraction of the box is
  exact, from one in ten to one in a thousand. The objective is a sphere centred outside the
  ball, so the optimum is on its surface, at a value of 1.
- corner: two linear limits whose boundaries cross at the optimum, so the active limit switches
  along the way there, and the total violation has a kink where it does.
- pocket: a deceptive one. The violation falls toward the objective's own minimum but bottoms
  out above zero there, in a wide basin; the only feasible designs are in a small pocket
  elsewhere, at worse values. Following the violation downhill never finds it.
- g06, g07 and g10 from the CEC 2006 constrained suite (Liang et al., 2006), whose feasible
  regions are a fraction of a percent of their boxes, with their limits as outputs.

Each noisy version adds Gaussian noise to the objective and to every output, so a design can
pass a limit by luck; the benchmark scores each run by the true value and the true feasibility
of the design it returns.
"""
import math
import zlib
from typing import Any

import numpy as np

from hone.parameters import Output, RangeParameter, SearchSpace

from .constrained import pressure_vessel_outputs, tension_spring_outputs, welded_beam_outputs
from .problems import BOUND, Problem, _shift

TAGS = frozenset({"constrained", "outputs"})


def _unit_ball_volume(dimensions: int) -> float:
    return math.pi ** (dimensions / 2) / math.gamma(dimensions / 2 + 1)


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(name.encode()))


def _direction(rng: np.random.Generator, dimensions: int) -> np.ndarray:
    v = rng.standard_normal(dimensions)
    return v / np.linalg.norm(v)


def _names(dimensions: int) -> list[str]:
    return [f"x{i}" for i in range(dimensions)]


def _vector(params: dict[str, Any], names: list[str]) -> np.ndarray:
    return np.array([params[n] for n in names], dtype=float)


def ball(dimensions: int, one_in: int) -> Problem:
    """Minimize a sphere centred outside a ball whose volume is 1/one_in of the box; feasible inside the ball.

    The ball's radius follows from its volume, and it lies inside the box. The sphere's centre is
    one unit beyond the ball's surface, so the optimum is on the surface, at a value of exactly 1.
    """
    name = f"ball-{dimensions}d-1in{one_in}"
    names = _names(dimensions)
    radius = ((2 * BOUND) ** dimensions / one_in / _unit_ball_volume(dimensions)) ** (1 / dimensions)
    assert radius < BOUND, f"{name}: the ball must fit in the box"
    rng = _rng(name)
    centre = rng.uniform(-0.8 * (BOUND - radius), 0.8 * (BOUND - radius), dimensions)
    target = centre + (radius + 1.0) * _direction(rng, dimensions)

    def function(**params: Any) -> dict[str, float]:
        x = _vector(params, names)
        return {"objective": float(np.sum((x - target) ** 2)), "distance": float(np.linalg.norm(x - centre))}

    function.__name__ = name.replace("-", "_")
    space = SearchSpace([RangeParameter(n, -BOUND, BOUND) for n in names], [Output("distance") <= radius])
    return Problem(name=name, function=function, search_space=space, optimum=1.0, tags=TAGS | {"ball"})


def corner(dimensions: int) -> Problem:
    """Minimize a sphere under two linear limits whose boundaries cross at the optimum.

    The optimum x* is a fixed point in the box. Each limit is a half-space through x* with a
    unit normal n, reported as 1 + n·(x − x*) ≤ 1. The sphere's centre is x* + n₁ + n₂, so both
    limits are active at the optimum, whose value is |n₁ + n₂|², and which limit is the active
    one switches along the way there.
    """
    name = f"corner-{dimensions}d"
    names = _names(dimensions)
    rng = _rng(name)
    best = _shift(name, dimensions)
    first = _direction(rng, dimensions)
    second = _direction(rng, dimensions)
    while abs(first @ second) > 0.7:  # neither nearly parallel nor nearly opposite
        second = _direction(rng, dimensions)
    target = best + first + second

    def function(**params: Any) -> dict[str, float]:
        x = _vector(params, names)
        return {
            "objective": float(np.sum((x - target) ** 2)),
            "first": 1.0 + float(first @ (x - best)),
            "second": 1.0 + float(second @ (x - best)),
        }

    function.__name__ = name.replace("-", "_")
    space = SearchSpace([RangeParameter(n, -BOUND, BOUND) for n in names], [Output("first") <= 1, Output("second") <= 1])
    optimum = float(np.sum((first + second) ** 2))
    return Problem(name=name, function=function, search_space=space, optimum=optimum, tags=TAGS | {"corner"})


def pocket(dimensions: int, one_in: int = 100) -> Problem:
    """A deceptive problem: the violation falls toward the objective's minimum but never reaches zero there.

    The objective is a sphere centred at p. The reported quantity is the smaller of a wide basin,
    0.5 + 0.05 |x − p|², which is lowest at p but never below 0.5, and |x − c|² / r² − 1 for a
    ball of radius r around c, which is where the only feasible designs are: 1/one_in of the box.
    The limit is that the quantity is at most 0. The optimum is the point of the ball nearest p,
    at a value of (|p − c| − r)².
    """
    name = f"pocket-{dimensions}d-1in{one_in}"
    names = _names(dimensions)
    radius = ((2 * BOUND) ** dimensions / one_in / _unit_ball_volume(dimensions)) ** (1 / dimensions)
    rng = _rng(name)
    target = _shift(name, dimensions)
    centre = rng.uniform(-(BOUND - radius), BOUND - radius, dimensions)
    while np.linalg.norm(centre - target) < radius + 3.0:
        centre = rng.uniform(-(BOUND - radius), BOUND - radius, dimensions)

    def function(**params: Any) -> dict[str, float]:
        x = _vector(params, names)
        basin = 0.5 + 0.05 * float(np.sum((x - target) ** 2))
        hole = float(np.sum((x - centre) ** 2)) / radius**2 - 1.0
        return {"objective": float(np.sum((x - target) ** 2)), "quantity": min(basin, hole)}

    function.__name__ = name.replace("-", "_")
    space = SearchSpace([RangeParameter(n, -BOUND, BOUND) for n in names], [Output("quantity") <= 0])
    optimum = (float(np.linalg.norm(target - centre)) - radius) ** 2
    return Problem(name=name, function=function, search_space=space, optimum=optimum, tags=TAGS | {"pocket"})


# --- CEC 2006 ------------------------------------------------------------------


# The optimum of g06 is where its two circles cross: x1 = 14.095 from subtracting their equations,
# and x2 = 5 − √(100 − 9.095²), which gives the published value -6961.81387558.
G06_BEST = (14.095, 5 - math.sqrt(100 - 9.095**2))


def g06() -> Problem:
    """CEC 2006 g06: a cubic objective in two variables, feasible in a sliver between two circles.

    The best known value is -6961.81387558 at (14.095, 0.84296).
    """
    x1, x2 = RangeParameter("x1", 13, 100), RangeParameter("x2", 0, 100)

    def g06(x1: float, x2: float) -> dict[str, float]:
        return {
            "objective": (x1 - 10) ** 3 + (x2 - 20) ** 3,
            "outer": (x1 - 5) ** 2 + (x2 - 5) ** 2,
            "inner": (x1 - 6) ** 2 + (x2 - 5) ** 2,
        }

    space = SearchSpace([x1, x2], [Output("outer") >= 100, Output("inner") <= 82.81])
    return Problem(name="g06", function=g06, search_space=space, optimum=-6961.81387558, tags=TAGS | {"cec"})


G07_BEST = (
    2.17199634142692, 2.3636830416034, 8.77392573913157, 5.09598443745173, 0.990654756560493,
    1.43057392853463, 1.32164415364306, 9.82872580358362, 8.2800915887356, 8.3759266477347,
)


def g07() -> Problem:
    """CEC 2006 g07: a quadratic objective in ten variables under eight limits, three of them linear.

    The best known value is 24.3062090681.
    """
    names = _names(10)

    def g07(**params: float) -> dict[str, float]:
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = (params[n] for n in names)
        return {
            "objective": (
                x1**2 + x2**2 + x1 * x2 - 14 * x1 - 16 * x2 + (x3 - 10) ** 2 + 4 * (x4 - 5) ** 2 + (x5 - 3) ** 2
                + 2 * (x6 - 1) ** 2 + 5 * x7**2 + 7 * (x8 - 11) ** 2 + 2 * (x9 - 10) ** 2 + (x10 - 7) ** 2 + 45
            ),
            "g1": 4 * x1 + 5 * x2 - 3 * x7 + 9 * x8,
            "g2": 10 * x1 - 8 * x2 - 17 * x7 + 2 * x8,
            "g3": -8 * x1 + 2 * x2 + 5 * x9 - 2 * x10,
            "g4": 3 * (x1 - 2) ** 2 + 4 * (x2 - 3) ** 2 + 2 * x3**2 - 7 * x4,
            "g5": 5 * x1**2 + 8 * x2 + (x3 - 6) ** 2 - 2 * x4,
            "g6": x1**2 + 2 * (x2 - 2) ** 2 - 2 * x1 * x2 + 14 * x5 - 6 * x6,
            "g7": 0.5 * (x1 - 8) ** 2 + 2 * (x2 - 4) ** 2 + 3 * x5**2 - x6,
            "g8": -3 * x1 + 6 * x2 + 12 * (x9 - 8) ** 2 - 7 * x10,
        }

    limits = {"g1": 105, "g2": 0, "g3": 12, "g4": 120, "g5": 40, "g6": 0, "g7": 30, "g8": 0}
    space = SearchSpace([RangeParameter(n, -10, 10) for n in names], [Output(g) <= limit for g, limit in limits.items()])
    return Problem(name="g07", function=g07, search_space=space, optimum=24.3062090681, tags=TAGS | {"cec"})


# The published design; x7 is derived from the others by the two limits active at it, g2 and g5.
G10_BEST = (
    579.306685017979589, 1359.97067807935605, 5109.97065743133317, 182.01769963061534,
    295.601173702746792, 217.982300369384632, 286.4165259278685, 395.601173702746735,
)


def g10() -> Problem:
    """CEC 2006 g10: a heat exchanger network, a linear objective in eight variables under six limits.

    The best known value is 7049.24802052867.
    """
    names = _names(8)
    bounds = [(100, 10000), (1000, 10000), (1000, 10000)] + [(10, 1000)] * 5

    def g10(**params: float) -> dict[str, float]:
        x1, x2, x3, x4, x5, x6, x7, x8 = (params[n] for n in names)
        return {
            "objective": x1 + x2 + x3,
            "g1": 0.0025 * (x4 + x6),
            "g2": 0.0025 * (x5 + x7 - x4),
            "g3": 0.01 * (x8 - x5),
            "g4": -x1 * x6 + 833.33252 * x4 + 100 * x1,
            "g5": -x2 * x7 + 1250 * x5 + x2 * x4 - 1250 * x4,
            "g6": -x3 * x8 + x3 * x5 - 2500 * x5,
        }

    limits = {"g1": 1, "g2": 1, "g3": 1, "g4": 83333.333, "g5": 0, "g6": -1250000}
    parameters = [RangeParameter(n, low, high) for n, (low, high) in zip(names, bounds)]
    space = SearchSpace(parameters, [Output(g) <= limit for g, limit in limits.items()])
    return Problem(name="g10", function=g10, search_space=space, optimum=7049.24802052867, tags=TAGS | {"cec"})


# --- suites --------------------------------------------------------------------


def output_problems() -> list[Problem]:
    """The suite: the engineering designs with their limits on outputs, the balls, corners, pockets and the CEC problems."""
    return [
        pressure_vessel_outputs(),
        tension_spring_outputs(),
        welded_beam_outputs(),
        ball(2, 10),
        ball(2, 100),
        ball(5, 100),
        ball(5, 1000),
        corner(2),
        corner(5),
        pocket(2),
        pocket(5),
        g06(),
        g07(),
        g10(),
    ]


# Noise for the noisy versions of the engineering designs: about 1% of each quantity's scale
# at the best design, as a standard deviation.
ENGINEERING_NOISE: dict[str, tuple[float, dict[str, float]]] = {
    "pressure-vessel-outputs": (60.0, {"shell_needed": 0.008, "head_needed": 0.004, "volume": 13_000.0}),
    "tension-spring-outputs": (0.00013, {"deflection": 0.01, "shear": 0.01, "surge": 0.01}),
    "welded-beam-outputs": (0.017, {"shear": 136.0, "bending": 300.0, "buckling": 60.0, "deflection": 0.0025}),
}


def noisy_output_problems() -> list[Problem]:
    """The engineering designs with noise on the objective and on every reported quantity."""
    problems = []
    for problem in (pressure_vessel_outputs(), tension_spring_outputs(), welded_beam_outputs()):
        noise, output_noise = ENGINEERING_NOISE[problem.name]
        problems.append(problem.model_copy(update={
            "name": f"noisy-{problem.name}", "noise": noise, "output_noise": output_noise, "tags": problem.tags | {"noisy"},
        }))
    return problems
