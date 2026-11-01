"""Constrained test problems, whose optima lie on a constraint's boundary, as they usually do in engineering.

- boundary-sphere and boundary-ellipsoid: a shifted sphere or ellipsoid with a linear constraint
  that cuts off its unconstrained optimum, so the answer is on the cutting plane. Both optima are
  known exactly.
- rosenbrock-disk: Rosenbrock's function inside the disk x² + y² ≤ 2, whose optimum (1, 1) sits
  on the disk's edge.
- pressure-vessel, tension-spring and welded-beam: three standard engineering design problems,
  scored against the best designs published for them. The welded beam's stress formulas need
  square roots, so its constraints are functions; the others are expressions.

Each engineering problem also comes with its limits on outputs, as the -outputs version: the
objective reports the volume, stresses and so on that the limits apply to, as a simulation
would, and the optimizer only learns that a design breaks a limit after evaluating it. The
pairs show what is lost by not knowing a limit beforehand.
"""
import math
import zlib
from typing import Any

import numpy as np

from hone.parameters import Constraint, Output, RangeParameter, SearchSpace

from .problems import BOUND, Problem, _shift


def _normal(name: str, dimensions: int) -> np.ndarray:
    """A fixed unit vector for the named problem, with no component near zero."""
    rng = np.random.default_rng(zlib.crc32(f"{name}/normal".encode()))
    w = rng.uniform(0.3, 1.0, dimensions) * rng.choice([-1.0, 1.0], dimensions)
    return w / np.linalg.norm(w)


def _boundary(name: str, dimensions: int, weights: np.ndarray) -> Problem:
    """Minimize sum(weights * (x - c)²) subject to w·x ≤ w·c - 1, which the unconstrained optimum c breaks.

    The optimum is at c + z with z = -D⁻¹w / (wᵀD⁻¹w), where D is diag(weights), and its value is 1 / (wᵀD⁻¹w).
    """
    names = [f"x{i}" for i in range(dimensions)]
    parameters = [RangeParameter(n, -BOUND, BOUND) for n in names]
    centre, w = _shift(name, dimensions), _normal(name, dimensions)
    limit = float(w @ centre) - 1.0
    terms = [float(wi) * p for wi, p in zip(w, parameters)]
    plane = terms[0]
    for term in terms[1:]:
        plane = plane + term
    inverse = 1.0 / weights
    best = centre - inverse * w / float(w @ (inverse * w))
    assert np.all(np.abs(best) <= BOUND), f"{name}: the constrained optimum must lie inside the bounds"

    def function(**params: Any) -> float:
        x = np.array([params[n] for n in names], dtype=float)
        return float(np.sum(weights * (x - centre) ** 2))

    function.__name__ = name.replace("-", "_")
    space = SearchSpace(parameters, [plane <= limit])
    return Problem(name=name, function=function, search_space=space, optimum=1.0 / float(w @ (inverse * w)), tags=frozenset({"constrained"}))


def boundary_sphere(dimensions: int) -> Problem:
    return _boundary(f"boundary-sphere-{dimensions}d", dimensions, np.ones(dimensions))


def boundary_ellipsoid(dimensions: int) -> Problem:
    # Axis weights from 1 to 1,000: wide enough to need scaling, narrow enough to keep the optimum in bounds.
    return _boundary(f"boundary-ellipsoid-{dimensions}d", dimensions, 10.0 ** (3 * np.arange(dimensions) / max(dimensions - 1, 1)))


def rosenbrock_disk() -> Problem:
    x, y = RangeParameter("x", -1.5, 1.5), RangeParameter("y", -1.5, 1.5)

    def rosenbrock_disk(x: float, y: float) -> float:
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    space = SearchSpace([x, y], [x**2 + y**2 <= 2])
    return Problem(name="rosenbrock-disk-2d", function=rosenbrock_disk, search_space=space, optimum=0.0, tags=frozenset({"constrained"}))


def pressure_vessel() -> Problem:
    """A cylindrical vessel with hemispherical heads, at least 1,296,000 cubic inches, of least cost.

    Shell and head thicknesses come in sixteenths of an inch, so they are integers here. The best
    published design costs 6059.714 (Ts = 13/16, Th = 7/16, R = 42.098, L = 176.637).
    """
    shell, head = RangeParameter("shell", 1, 99, integer=True), RangeParameter("head", 1, 99, integer=True)
    radius, length = RangeParameter("radius", 10, 200), RangeParameter("length", 10, 200)

    def pressure_vessel(shell: int, head: int, radius: float, length: float) -> float:
        ts, th = shell / 16, head / 16
        return 0.6224 * ts * radius * length + 1.7781 * th * radius**2 + 3.1661 * ts**2 * length + 19.84 * ts**2 * radius

    space = SearchSpace(
        [shell, head, radius, length],
        [
            shell / 16 >= 0.0193 * radius,
            head / 16 >= 0.00954 * radius,
            math.pi * radius**2 * length + 4 / 3 * math.pi * radius**3 >= 1_296_000,
        ],
    )
    return Problem(name="pressure-vessel", function=pressure_vessel, search_space=space, optimum=6059.714335, tags=frozenset({"constrained", "mixed", "engineering"}))


def tension_spring() -> Problem:
    """A coil spring of least weight under limits on deflection, shear stress, surge frequency and diameter.

    The best published design weighs 0.0126652 (d = 0.051690, D = 0.356750, N = 11.287126).
    """
    d, coil, n = RangeParameter("wire", 0.05, 2.0), RangeParameter("coil", 0.25, 1.3), RangeParameter("turns", 2.0, 15.0)

    def tension_spring(wire: float, coil: float, turns: float) -> float:
        return (turns + 2) * coil * wire**2

    space = SearchSpace(
        [d, coil, n],
        [
            coil**3 * n / (71785 * d**4) >= 1,
            (4 * coil**2 - d * coil) / (12566 * (coil * d**3 - d**4)) + 1 / (5108 * d**2) <= 1,
            140.45 * d / (coil**2 * n) >= 1,
            d + coil <= 1.5,
        ],
    )
    return Problem(name="tension-spring", function=tension_spring, search_space=space, optimum=0.012665233, tags=frozenset({"constrained", "engineering"}))


# The welded beam's load and material: load P (lb), length L (in), Young's and shear moduli (psi).
P, L, E, G = 6000.0, 14.0, 30e6, 12e6


def _shear(weld: float, length: float, thickness: float) -> float:
    primary = P / (math.sqrt(2) * weld * length)
    moment = P * (L + length / 2)
    r = math.sqrt(length**2 / 4 + ((weld + thickness) / 2) ** 2)
    j = 2 * math.sqrt(2) * weld * length * (length**2 / 12 + ((weld + thickness) / 2) ** 2)
    secondary = moment * r / j
    return math.sqrt(primary**2 + primary * secondary * length / r + secondary**2)


def _buckling(thickness: float, width: float) -> float:
    return 4.013 * E * math.sqrt(thickness**2 * width**6 / 36) / L**2 * (1 - thickness / (2 * L) * math.sqrt(E / (4 * G)))


def welded_beam() -> Problem:
    """A beam welded to a support, of least cost under limits on shear, bending, buckling and deflection.

    The best published design costs 1.724852 (h = 0.205730, l = 3.470489, t = 9.036624, b = 0.205730).
    """
    weld, length = RangeParameter("weld", 0.1, 2.0), RangeParameter("length", 0.1, 10.0)
    thickness, width = RangeParameter("thickness", 0.1, 10.0), RangeParameter("width", 0.1, 2.0)

    def welded_beam(weld: float, length: float, thickness: float, width: float) -> float:
        return 1.10471 * weld**2 * length + 0.04811 * thickness * width * (14 + length)

    def shear(weld: float, length: float, thickness: float) -> float:
        return _shear(weld, length, thickness) - 13600

    def buckling(thickness: float, width: float) -> float:
        return P - _buckling(thickness, width)

    space = SearchSpace(
        [weld, length, thickness, width],
        [
            Constraint.function(shear),
            6 * P * L / (width * thickness**2) <= 30000,
            weld <= width,
            0.10471 * weld**2 + 0.04811 * thickness * width * (14 + length) <= 5,
            4 * P * L**3 / (E * thickness**3 * width) <= 0.25,
            Constraint.function(buckling),
        ],
    )
    return Problem(name="welded-beam", function=welded_beam, search_space=space, optimum=1.724852, tags=frozenset({"constrained", "engineering"}))


def pressure_vessel_outputs() -> Problem:
    """The pressure vessel, with its limits on what it reports: the plate thicknesses its radius needs, and its volume."""
    problem = pressure_vessel()
    shell, head = RangeParameter("shell", 1, 99, integer=True), RangeParameter("head", 1, 99, integer=True)
    radius, length = RangeParameter("radius", 10, 200), RangeParameter("length", 10, 200)
    cost = problem.function

    def pressure_vessel_outputs(shell: int, head: int, radius: float, length: float) -> dict[str, float]:
        return {
            "objective": cost(shell=shell, head=head, radius=radius, length=length),
            "shell_needed": 0.0193 * radius,
            "head_needed": 0.00954 * radius,
            "volume": math.pi * radius**2 * length + 4 / 3 * math.pi * radius**3,
        }

    space = SearchSpace(
        [shell, head, radius, length],
        [shell / 16 >= Output("shell_needed"), head / 16 >= Output("head_needed"), Output("volume") >= 1_296_000],
    )
    return problem.model_copy(update={"name": "pressure-vessel-outputs", "function": pressure_vessel_outputs, "search_space": space, "tags": problem.tags | {"outputs"}})


def tension_spring_outputs() -> Problem:
    """The tension spring, with its deflection, shear stress and surge frequency limits on what it reports."""
    problem = tension_spring()
    d, coil, n = RangeParameter("wire", 0.05, 2.0), RangeParameter("coil", 0.25, 1.3), RangeParameter("turns", 2.0, 15.0)
    weight = problem.function

    def tension_spring_outputs(wire: float, coil: float, turns: float) -> dict[str, float]:
        if coil == wire:
            return {"objective": math.nan, "deflection": math.nan, "shear": math.nan, "surge": math.nan}  # the shear divides by zero: failed
        return {
            "objective": weight(wire=wire, coil=coil, turns=turns),
            "deflection": coil**3 * turns / (71785 * wire**4),
            "shear": (4 * coil**2 - wire * coil) / (12566 * (coil * wire**3 - wire**4)) + 1 / (5108 * wire**2),
            "surge": 140.45 * wire / (coil**2 * turns),
        }

    space = SearchSpace(
        [d, coil, n],
        [Output("deflection") >= 1, Output("shear") <= 1, Output("surge") >= 1, d + coil <= 1.5],
    )
    return problem.model_copy(update={"name": "tension-spring-outputs", "function": tension_spring_outputs, "search_space": space, "tags": problem.tags | {"outputs"}})


def welded_beam_outputs() -> Problem:
    """The welded beam, with its shear, bending, buckling and deflection limits on what it reports.

    The geometric limits, weld <= width and the cost cap, stay on the parameters, as they would
    in practice: they need no simulation.
    """
    problem = welded_beam()
    weld, length = RangeParameter("weld", 0.1, 2.0), RangeParameter("length", 0.1, 10.0)
    thickness, width = RangeParameter("thickness", 0.1, 10.0), RangeParameter("width", 0.1, 2.0)
    cost = problem.function

    def welded_beam_outputs(weld: float, length: float, thickness: float, width: float) -> dict[str, float]:
        return {
            "objective": cost(weld=weld, length=length, thickness=thickness, width=width),
            "shear": _shear(weld, length, thickness),
            "bending": 6 * P * L / (width * thickness**2),
            "buckling": _buckling(thickness, width),
            "deflection": 4 * P * L**3 / (E * thickness**3 * width),
        }

    space = SearchSpace(
        [weld, length, thickness, width],
        [
            Output("shear") <= 13600,
            Output("bending") <= 30000,
            weld <= width,
            0.10471 * weld**2 + 0.04811 * thickness * width * (14 + length) <= 5,
            Output("deflection") <= 0.25,
            Output("buckling") >= P,
        ],
    )
    return problem.model_copy(update={"name": "welded-beam-outputs", "function": welded_beam_outputs, "search_space": space, "tags": problem.tags | {"outputs"}})


def constrained_problems(dimensions: tuple[int, ...] = (2, 5, 10)) -> list[Problem]:
    """The constrained suite: boundary sphere and ellipsoid in each dimension, Rosenbrock in a disk, and three engineering designs.

    Each engineering design comes twice: with its limits on the parameters, and on its outputs.
    """
    problems = [factory(d) for d in dimensions for factory in (boundary_sphere, boundary_ellipsoid)]
    return problems + [
        rosenbrock_disk(),
        pressure_vessel(),
        tension_spring(),
        welded_beam(),
        pressure_vessel_outputs(),
        tension_spring_outputs(),
        welded_beam_outputs(),
    ]


def output_problems() -> list[Problem]:
    """The engineering designs with their limits on outputs only, for a quick comparison with the parameter versions."""
    return [pressure_vessel_outputs(), tension_spring_outputs(), welded_beam_outputs()]
