"""Test problems with known optima.

The continuous functions follow the conventions of the BBOB benchmark: every variable lies
in [-5, 5], and each function is shifted so its optimum sits at a fixed, problem-specific
point away from the centre of the box. Optimizers that drift toward the centre would
otherwise look better than they are.

Each continuous function except sphere also comes in a rotated version, which turns the
coordinate axes by a fixed random rotation. The plain versions are separable: every
variable affects the value independently, so an optimizer can improve one variable at a
time, and crossover that swaps whole variables between parents does unusually well. In
the rotated versions the variables interact, as they usually do in engineering problems.
"""
import math
import zlib
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

from hone.parameters import ChoiceParameter, Parameter, RangeParameter, SearchSpace

BOUND = 5.0


class Problem(BaseModel):
    """An objective function, its search space, and the lowest value it can take."""
    model_config = ConfigDict(frozen=True)

    name: str
    function: Callable[..., float]
    search_space: SearchSpace
    optimum: float
    tags: frozenset[str]
    noise: float = 0.0  # standard deviation of Gaussian noise the benchmark adds to each evaluation

    @property
    def dimensions(self) -> int:
        return len(self.search_space.parameters)


def _shift(name: str, dimensions: int, integer: bool = False) -> np.ndarray:
    """A fixed optimum location for the named problem, within 60% of the bounds."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    shift = rng.uniform(-0.6 * BOUND, 0.6 * BOUND, size=dimensions)
    return np.round(shift) if integer else shift


def _rotation(name: str, dimensions: int) -> np.ndarray:
    """A fixed random rotation for the named problem: an orthogonal matrix with determinant 1."""
    rng = np.random.default_rng(zlib.crc32(f"{name}/rotation".encode()))
    q, r = np.linalg.qr(rng.standard_normal((dimensions, dimensions)))
    q = q * np.sign(np.diag(r))  # makes the rotation uniformly random
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _vector_problem(
    name: str,
    base: Callable[[np.ndarray], float],
    dimensions: int,
    parameters: list[Parameter] | None = None,
    integer: bool = False,
    rotate: bool = False,
    tags: frozenset[str] = frozenset({"continuous"}),
) -> Problem:
    """Wrap a function of a vector z, minimal at z = 0, as a shifted problem over x0..x{n-1}.

    With rotate, z is the shifted point turned by a fixed rotation, so the variables interact.
    """
    names = [f"x{i}" for i in range(dimensions)]
    shift = _shift(name, dimensions, integer)
    rotation = _rotation(name, dimensions) if rotate else np.eye(dimensions)

    def function(**params: Any) -> float:
        x = np.array([params[n] for n in names], dtype=float)
        return float(base(rotation @ (x - shift)))

    function.__name__ = name.replace("-", "_")
    if parameters is None:
        parameters = [RangeParameter(n, -BOUND, BOUND, integer=integer) for n in names]
    return Problem(name=name, function=function, search_space=SearchSpace(parameters), optimum=0.0, tags=tags)


# --- base functions, each with minimum 0 at z = 0 ---------------------------


def sphere(z: np.ndarray) -> float:
    return np.sum(z**2)


def ellipsoid(z: np.ndarray) -> float:
    """Sphere with axis weights from 1 to 1e6: tests how an optimizer handles badly scaled variables."""
    weights = 10 ** (6 * np.arange(len(z)) / max(len(z) - 1, 1))
    return np.sum(weights * z**2)


def rosenbrock(z: np.ndarray) -> float:
    """A long, curved valley. The usual form has its minimum at 1, so z is offset by 1."""
    x = z + 1
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(z: np.ndarray) -> float:
    """A regular grid of local minima around one global minimum."""
    return 10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))


def ackley(z: np.ndarray) -> float:
    """A nearly flat outer region with a deep, narrow hole at the minimum."""
    n = len(z)
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * z)) / n)
        + 20
        + math.e
    )


CONTINUOUS = {
    "sphere": sphere,
    "ellipsoid": ellipsoid,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}

# Sphere is the same under any rotation, so it has no rotated version.
ROTATED = {name: base for name, base in CONTINUOUS.items() if name != "sphere"}


def rotated(name: str, dimensions: int) -> Problem:
    return _vector_problem(
        f"rotated-{name}-{dimensions}d", ROTATED[name], dimensions, rotate=True, tags=frozenset({"continuous", "rotated"})
    )


# --- mixed problems --------------------------------------------------------


def integer_sphere(dimensions: int) -> Problem:
    """Sphere over integer variables only; the optimum is at an integer point."""
    return _vector_problem(
        f"integer-sphere-{dimensions}d", sphere, dimensions, integer=True, tags=frozenset({"integer"})
    )


def mixed_rastrigin(dimensions: int) -> Problem:
    """Rastrigin with the first half of the variables continuous and the rest integer."""
    name = f"mixed-rastrigin-{dimensions}d"
    n_int = dimensions // 2
    parameters = [
        RangeParameter(f"x{i}", -BOUND, BOUND, integer=i >= dimensions - n_int) for i in range(dimensions)
    ]
    shift = _shift(name, dimensions)
    shift[dimensions - n_int :] = np.round(shift[dimensions - n_int :])
    names = [p.name for p in parameters]

    def function(**params: Any) -> float:
        x = np.array([params[n] for n in names], dtype=float)
        return float(rastrigin(x - shift))

    function.__name__ = name.replace("-", "_")
    return Problem(name=name, function=function, search_space=SearchSpace(parameters), optimum=0.0, tags=frozenset({"mixed"}))


MATERIALS = {
    # A synthetic material choice: each material has its own best design point and a base cost.
    "steel": 3.0,
    "aluminium": 1.0,
    "titanium": 0.0,
    "composite": 2.0,
}


def material_selection(dimensions: int) -> Problem:
    """A categorical choice that changes where the best continuous design lies.

    Each material moves the optimum of the continuous variables and adds its own base cost,
    so the optimizer has to search both together. Titanium, at cost 0, gives the global optimum.
    """
    name = f"material-selection-{dimensions}d"
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    targets = {m: rng.uniform(-0.6 * BOUND, 0.6 * BOUND, size=dimensions) for m in MATERIALS}
    names = [f"x{i}" for i in range(dimensions)]
    parameters: list[Parameter] = [ChoiceParameter("material", list(MATERIALS))]
    parameters += [RangeParameter(n, -BOUND, BOUND) for n in names]

    def function(material: str, **params: Any) -> float:
        x = np.array([params[n] for n in names], dtype=float)
        return float(MATERIALS[material] + sphere(x - targets[material]))

    function.__name__ = name.replace("-", "_")
    return Problem(name=name, function=function, search_space=SearchSpace(parameters), optimum=0.0, tags=frozenset({"mixed", "categorical"}))


def standard_problems(dimensions: tuple[int, ...] = (2, 5, 10)) -> list[Problem]:
    """The full suite: every continuous function, plain and rotated, and every mixed problem, in each dimension."""
    problems = [
        _vector_problem(f"{name}-{d}d", base, d) for d in dimensions for name, base in CONTINUOUS.items()
    ]
    problems += [rotated(name, d) for d in dimensions for name in ROTATED]
    for d in dimensions:
        problems += [integer_sphere(d), mixed_rastrigin(d), material_selection(d)]
    return problems


def quick_problems() -> list[Problem]:
    """A small subset for CI: one dimension, and a problem of each kind."""
    return [
        _vector_problem("sphere-5d", sphere, 5),
        _vector_problem("ellipsoid-5d", ellipsoid, 5),
        _vector_problem("rastrigin-5d", rastrigin, 5),
        rotated("rastrigin", 5),
        integer_sphere(5),
        material_selection(5),
    ]


# Noise the noisy suite adds to each evaluation, as a standard deviation. The problems' values
# near their optima are a few units, so this hides the differences an optimizer has to find.
NOISE = 1.0


def noisy_problems() -> list[Problem]:
    """Problems with Gaussian noise added to every evaluation, scored by the true value of the result."""
    plain = [
        _vector_problem("sphere-5d", sphere, 5),
        _vector_problem("rosenbrock-5d", rosenbrock, 5),
        _vector_problem("rastrigin-5d", rastrigin, 5),
        rotated("ackley", 5),
        mixed_rastrigin(5),
        material_selection(5),
    ]
    return [p.model_copy(update={"name": f"noisy-{p.name}", "noise": NOISE, "tags": p.tags | {"noisy"}}) for p in plain]
