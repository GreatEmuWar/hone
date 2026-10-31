"""Conditional test problems, each in two versions: with conditions, and with every parameter switched on.

- branches-3x2d and branches-3x5d: a choice of three branches, each a function of its own
  numbers. The best branch is a steep, narrow bowl reaching 0; the decoys are a wide, shallow
  bowl whose floor is 1 and a shallow Rastrigin grid whose floor is 0.5. Random points score
  better in the decoys, so an optimizer has to look past them.
- nested-9: a choice of model; one model has a choice of kernel, and one kernel has numbers of
  its own, so the best point is two conditions deep. The other model is a wide, shallow decoy
  whose floor is 0.25.
- layers-7: a layer count from 1 to 5 whose larger values switch on more numbers, as a
  network's units per layer would be. Three layers is best. Its conditions test a range, so
  CMA-ES can search it.

The "-flat" version of each has the same parameters, function and optimum, with no conditions:
every parameter is always on and the function ignores those it doesn't use, which was the way
to search such a space before conditions. Comparing the two shows what conditions are worth.
"""
from typing import Any, Callable

import numpy as np

from hone.parameters import ChoiceParameter, Condition, Parameter, RangeParameter, SearchSpace

from .problems import BOUND, Problem, _shift, rastrigin, sphere


def _problem(name: str, function: Callable[..., float], parameters: list[Parameter], conditions: list[Condition]) -> list[Problem]:
    function.__name__ = name.replace("-", "_")
    return [
        Problem(
            name=name,
            function=function,
            search_space=SearchSpace(parameters, conditions=conditions),
            optimum=0.0,
            tags=frozenset({"conditional"}),
        ),
        Problem(
            name=f"{name}-flat",
            function=function,
            search_space=SearchSpace(parameters),
            optimum=0.0,
            tags=frozenset({"conditional", "flat"}),
        ),
    ]


def _numbers(prefix: str, count: int) -> list[RangeParameter]:
    return [RangeParameter(f"{prefix}{i}", -BOUND, BOUND) for i in range(count)]


def _vector(params: dict[str, Any], numbers: list[RangeParameter]) -> np.ndarray:
    return np.array([params[p.name] for p in numbers], dtype=float)


def branches(dimensions: int) -> list[Problem]:
    name = f"branches-3x{dimensions}d"
    branch = ChoiceParameter("branch", ["narrow", "wide", "grid"])
    numbers = {b: _numbers(f"{b}_", dimensions) for b in branch.choices}
    shifts = {b: _shift(f"{name}/{b}", dimensions) for b in branch.choices}
    # (function, scale, floor) for each branch
    bases = {"narrow": (sphere, 4.0, 0.0), "wide": (sphere, 0.05, 1.0), "grid": (rastrigin, 0.1, 0.5)}

    def function(branch: str, **params: Any) -> float:
        base, scale, floor = bases[branch]
        return floor + scale * float(base(_vector(params, numbers[branch]) - shifts[branch]))

    parameters: list[Parameter] = [branch, *(p for b in branch.choices for p in numbers[b])]
    return _problem(name, function, parameters, [branch[b] >> numbers[b] for b in branch.choices])


def nested() -> list[Problem]:
    name = "nested-9"
    model = ChoiceParameter("model", ["kernel_machine", "forest"])
    kernel = ChoiceParameter("kernel", ["linear", "rbf"])
    shared, width, forest = _numbers("s", 2), _numbers("g", 2), _numbers("f", 3)
    centres = {key: _shift(f"{name}/{key}", len(group)) for key, group in [("s", shared), ("g", width), ("f", forest)]}

    def function(model: str, kernel: str | None = None, **params: Any) -> float:
        if model == "forest":
            return 0.25 + 0.05 * float(sphere(_vector(params, forest) - centres["f"]))
        value = float(sphere(_vector(params, shared) - centres["s"]))
        if kernel == "linear":
            return 0.5 + value
        return value + 4 * float(sphere(_vector(params, width) - centres["g"]))

    parameters: list[Parameter] = [model, kernel, *shared, *width, *forest]
    conditions = [model["kernel_machine"] >> [kernel, *shared], kernel["rbf"] >> width, model["forest"] >> forest]
    return _problem(name, function, parameters, conditions)


def layers() -> list[Problem]:
    name = "layers-7"
    count = RangeParameter("layers", 1, 5, integer=True)
    rate = RangeParameter("rate", -BOUND, BOUND)
    units = _numbers("units_", 5)
    centre = _shift(f"{name}/units", 5)
    best_rate = float(_shift(f"{name}/rate", 1)[0])

    def function(layers: int, rate: float, **params: Any) -> float:
        used = units[:layers]
        return 0.5 * abs(layers - 3) + float(sphere(_vector(params, used) - centre[:layers])) + (rate - best_rate) ** 2

    parameters: list[Parameter] = [count, rate, *units]
    return _problem(name, function, parameters, [(count >= i + 1) >> units[i] for i in range(1, 5)])


def conditional_problems() -> list[Problem]:
    """Each conditional problem, followed by its flat version."""
    return [*branches(2), *branches(5), *nested(), *layers()]
