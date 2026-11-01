import inspect
import math
import numbers
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .conditions import AnyConstraint, ChoiceTest, Condition, ConditionalConstraint, describe, gate_error, holds
from .constraints import Constraint, ExpressionConstraint, FunctionConstraint, InfeasibleSearchSpace, Variable, _Arithmetic

type Value = float | int | str


class _Parameter(BaseModel):
    """A named dimension of the search space, and the values it may take."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str

    def __init__(self, **data: Any) -> None:
        # Each kind takes its own fields positionally and passes them all here as keywords.
        super().__init__(**data)

    def __rshift__(self, other: Any) -> Any:
        raise TypeError(f"{self.name} can't switch parameters on by itself: the left of >> must be a test of it")

    def __rrshift__(self, other: Any) -> Any:
        raise gate_error(self, other)


class RangeParameter(_Arithmetic, _Parameter):
    """A number between minimum and maximum, optionally restricted to integers.

    With log=True the range is searched on a logarithmic scale: values are drawn evenly in
    their logarithm, so 0.001 to 0.01 gets as much attention as 100 to 1000. Use it for
    settings that span orders of magnitude, such as learning rates and regularization
    strengths. A log range needs a positive minimum.
    """
    kind: Literal["range"] = "range"
    minimum: float
    maximum: float
    integer: bool = False
    log: bool = False

    def __init__(self, name: str, minimum: float, maximum: float, **data: Any) -> None:
        super().__init__(name=name, minimum=minimum, maximum=maximum, **data)

    def _as_term(self) -> Variable:
        return Variable(name=self.name)

    def __rshift__(self, other: Any) -> Any:
        raise TypeError(
            f"{self.name} can't switch parameters on by itself: compare it in brackets, such as ({self.name} >= ...) >> ..."
        )

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.minimum > self.maximum:
            raise ValueError(f"{self.name}: minimum {self.minimum} is greater than maximum {self.maximum}")
        if self.integer and math.ceil(self.minimum) > math.floor(self.maximum):
            raise ValueError(f"{self.name}: no integer lies between {self.minimum} and {self.maximum}")
        if self.log and self.minimum <= 0:
            raise ValueError(f"{self.name}: a log range needs a positive minimum, got {self.minimum}")
        return self

    def _scale(self, value):
        return np.log(value) if self.log else value

    def to_unit(self, value: Any) -> Any:
        """Where value lies in the range, from 0 at the minimum to 1 at the maximum, on the range's scale.

        Accepts a number or an array of them.
        """
        low, high = self._scale(self.minimum), self._scale(self.maximum)
        if high == low:
            return np.zeros_like(np.asarray(value, dtype=float)) if np.ndim(value) else 0.0
        return (self._scale(np.asarray(value, dtype=float)) - low) / (high - low)

    def from_unit(self, unit: Any) -> Any:
        """The value at position unit in the range, clipped to it, and rounded for an integer range.

        The inverse of to_unit. Accepts a number, returning an int or float, or an array.
        """
        low, high = self._scale(self.minimum), self._scale(self.maximum)
        value = low + np.clip(unit, 0, 1) * (high - low)
        if self.log:
            value = np.exp(value)
        if self.integer:
            value = np.clip(np.round(value), math.ceil(self.minimum), math.floor(self.maximum))
        else:
            value = np.clip(value, self.minimum, self.maximum)
        if np.ndim(value):
            return value
        return int(value) if self.integer else float(value)

    def sample(self, rng: np.random.Generator) -> float | int:
        if self.log:
            return self.from_unit(rng.uniform())
        if self.integer:
            return int(rng.integers(math.ceil(self.minimum), math.floor(self.maximum), endpoint=True))
        return float(rng.uniform(self.minimum, self.maximum))

    def check(self, value: Any) -> float | int:
        """value as this parameter holds it, or ValueError if it isn't one of its values."""
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{self.name}: {value!r} is not a number")
        number = float(value)
        if not self.minimum <= number <= self.maximum:
            raise ValueError(f"{self.name}: {value} is outside [{self.minimum}, {self.maximum}]")
        if self.integer:
            if not number.is_integer():
                raise ValueError(f"{self.name}: {value} is not an integer")
            return int(number)
        return number

    def mutate(self, value: float | int, rng: np.random.Generator, scale: float) -> float | int:
        """Nudge value by Gaussian noise with a standard deviation of scale times the range width.

        The result is clipped to the bounds. Integer ranges use a step of at least 1, so a
        mutation can always move the value. A log range is nudged in its logarithm, so the
        step is a proportion of the value rather than a fixed amount.
        """
        if self.log:
            mutated = self.from_unit(self.to_unit(value) + rng.normal(0, scale))
            if self.integer and mutated == value:
                neighbour = min(max(value + rng.choice([-1, 1]), math.ceil(self.minimum)), math.floor(self.maximum))
                mutated = int(neighbour)
            return mutated
        sigma = scale * (self.maximum - self.minimum)
        if self.integer:
            step = round(rng.normal(0, max(sigma, 1)))
            return int(np.clip(value + step, math.ceil(self.minimum), math.floor(self.maximum)))
        return float(np.clip(value + rng.normal(0, sigma), self.minimum, self.maximum))


class FixedParameter(_Parameter):
    """A parameter held at a single value."""
    kind: Literal["fixed"] = "fixed"
    value: Value

    def __init__(self, name: str, value: Value, **data: Any) -> None:
        super().__init__(name=name, value=value, **data)

    def sample(self, rng: np.random.Generator) -> Value:
        return self.value

    def check(self, value: Any) -> Value:
        """value as this parameter holds it, or ValueError if it isn't one of its values."""
        if value != self.value:
            raise ValueError(f"{self.name}: {value!r} is not its fixed value {self.value!r}")
        return self.value

    def mutate(self, value: Value, rng: np.random.Generator, scale: float) -> Value:
        return self.value


class ChoiceParameter(_Parameter):
    """One value picked from a list of choices, or a list of choose_n distinct choices."""
    kind: Literal["choice"] = "choice"
    choices: list[Value] = Field(min_length=1)
    choose_n: int | None = Field(default=None, ge=1)

    def __init__(self, name: str, choices: list[Value], **data: Any) -> None:
        super().__init__(name=name, choices=choices, **data)

    @model_validator(mode="after")
    def _check_choose_n(self):
        if self.choose_n is not None and self.choose_n > len(self.choices):
            raise ValueError(f"{self.name}: choose_n {self.choose_n} exceeds the {len(self.choices)} choices")
        return self

    def __getitem__(self, values: Value | tuple[Value, ...]) -> ChoiceTest:
        """A test that this parameter takes one of values, for the left of >>: model["svm"] >> C."""
        tested = list(values) if isinstance(values, tuple) else [values]
        if self.choose_n is not None:
            raise TypeError(f"{self.name} picks {self.choose_n} choices at once, so conditions can't test it")
        unknown = [value for value in tested if value not in self.choices]
        if unknown:
            raise ValueError(f"{', '.join(map(repr, unknown))} isn't one of {self.name}'s choices, {self.choices}")
        return ChoiceTest(name=self.name, values=tested)

    def __rshift__(self, other: Any) -> Any:
        raise TypeError(
            f"{self.name} can't switch parameters on by itself: pick the values that do, such as {self.name}[{self.choices[0]!r}] >> ..."
        )

    def sample(self, rng: np.random.Generator) -> Value | list[Value]:
        if self.choose_n is None:
            return self.choices[rng.integers(len(self.choices))]
        indices = rng.choice(len(self.choices), size=self.choose_n, replace=False)
        return [self.choices[i] for i in indices]

    def check(self, value: Any) -> Value | list[Value]:
        """value as this parameter holds it, or ValueError if it isn't one of its values."""
        if self.choose_n is None:
            if value not in self.choices:
                raise ValueError(f"{self.name}: {value!r} is not one of {self.choices}")
            return value
        if not isinstance(value, (list, tuple)) or len(value) != self.choose_n:
            raise ValueError(f"{self.name}: expected a list of {self.choose_n} choices, got {value!r}")
        if any(v not in self.choices for v in value) or len(set(map(repr, value))) != len(value):
            raise ValueError(f"{self.name}: {value!r} must be {self.choose_n} different values from {self.choices}")
        return list(value)

    def mutate(self, value: Value | list[Value], rng: np.random.Generator, scale: float) -> Value | list[Value]:
        """Draw a new value. A single choice switches to a different one when there is one to switch to.

        Choices have no order, so scale does not apply.
        """
        if self.choose_n is not None or len(self.choices) == 1:
            return self.sample(rng)
        others = [choice for choice in self.choices if choice != value]
        return others[rng.integers(len(others))] if others else self.sample(rng)


type Parameter = Annotated[RangeParameter | FixedParameter | ChoiceParameter, Field(discriminator="kind")]


MAX_DRAWS = 100_000  # draws in a row without a feasible point before a search space counts as infeasible


class SearchSpace(BaseModel):
    """The set of parameters an objective function is optimized over, and any constraints and conditions on them.

    Every point an optimizer asks for satisfies every constraint on the parameters, so the
    objective is never evaluated where one rules it out. A constraint that uses an output, such
    as `Output("stress") <= 250`, can only be judged after evaluating: the objective returns
    the outputs in a dict, and points that break such a limit rank below those that don't.
    A condition, written `test >> parameters`, switches parameters on only while the test
    holds, such as `model["svm"] >> [kernel, C]`; a parameter that is switched off isn't part
    of the point.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: list[Parameter] = Field(min_length=1)
    constraints: list[AnyConstraint] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)

    # Worked out from the conditions: each conditional parameter's test, and every parameter in
    # an order that puts each parameter after those its test uses.
    _tests: dict[str, ChoiceTest | ExpressionConstraint] = PrivateAttr(default_factory=dict)
    _order: list[Parameter] = PrivateAttr(default_factory=list)
    # The constraints sorted by when they can be judged: before evaluating, or only after, from outputs.
    _before: list[AnyConstraint] = PrivateAttr(default_factory=list)
    _after: list[AnyConstraint] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        parameters: Sequence[Parameter],
        constraints: Sequence[AnyConstraint] = (),
        conditions: Sequence[Condition] = (),
        **data: Any,
    ) -> None:
        for constraint in constraints:
            if isinstance(constraint, Condition):
                raise TypeError(f"`{constraint}` switches parameters on, so it goes in conditions=, not constraints=")
        for condition in conditions:
            if isinstance(condition, Constraint):
                raise TypeError(f"`{condition}` is a constraint, so it goes in constraints=, not conditions=")
        super().__init__(parameters=list(parameters), constraints=list(constraints), conditions=list(conditions), **data)

    @model_validator(mode="after")
    def _check_unique_names(self):
        names = [parameter.name for parameter in self.parameters]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"each parameter needs a unique name; repeated: {', '.join(duplicates)}")
        return self

    @model_validator(mode="after")
    def _check_constraints(self):
        by_name = {parameter.name: parameter for parameter in self.parameters}
        for constraint in self.constraints:
            _check_constraint(constraint, by_name)
            clashing = sorted(constraint.output_names() & set(by_name))
            if clashing:
                raise ValueError(
                    f"the constraint `{constraint}` uses an output named {clashing[0]!r}, which is also a parameter: "
                    "give the output another name"
                )
        self._before = [c for c in self.constraints if not c.output_names()]
        self._after = [c for c in self.constraints if c.output_names()]
        return self

    @model_validator(mode="after")
    def _check_conditions(self):
        by_name = {parameter.name: parameter for parameter in self.parameters}
        tests: dict[str, ChoiceTest | ExpressionConstraint] = {}
        for condition in self.conditions:
            _check_test(condition.when, by_name, f"the condition `{condition}`")
            for name in condition.parameters:
                if name not in by_name:
                    raise ValueError(f"the condition `{condition}` switches on {name!r}, which isn't a parameter of this search space")
                if name in tests:
                    earlier = next(c for c in self.conditions if name in c.parameters)
                    hint = ""
                    if isinstance(earlier.when, ChoiceTest) and isinstance(condition.when, ChoiceTest) and earlier.when.name == condition.when.name:
                        merged = ChoiceTest(name=earlier.when.name, values=[*earlier.when.values, *condition.when.values])
                        hint = f": to switch it on for any of those values, write `{merged} >> {name}`"
                    raise ValueError(f"{name} is switched on by two conditions, `{earlier}` and `{condition}`, but can have only one{hint}")
                if name in condition.when.names():
                    raise ValueError(f"the condition `{condition}` makes {name} depend on itself")
                tests[name] = condition.when
        self._tests = tests
        self._order = _dependency_order(self.parameters, tests)
        return self

    @property
    def names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    @property
    def parameter_constraints(self) -> list[AnyConstraint]:
        """The constraints judged from the parameters alone, before evaluating: every point asked for satisfies them."""
        return list(self._before)

    @property
    def output_constraints(self) -> list[AnyConstraint]:
        """The constraints that use an output, so can only be judged once an evaluation reports it."""
        return list(self._after)

    @property
    def outputs(self) -> list[str]:
        """The outputs the constraints use, which the objective must return, in alphabetical order."""
        return sorted({name for constraint in self._after for name in constraint.output_names()})

    def output_violation(self, params: dict[str, Any], outputs: dict[str, float]) -> float:
        """The total violation of the output constraints at params: the sum of each one's relative violation.

        Zero when every one that applies is satisfied. Each violation is a share of its limit's
        size, so 0.1 is 10% over, and limits in different units add up fairly.
        """
        point = self.active(params)
        return float(sum(c.relative_violation(point, outputs) for c in self._after if c.applies(point)))

    def output_broken(self, params: dict[str, Any], outputs: dict[str, float]) -> list[AnyConstraint]:
        """The output constraints params and their outputs don't satisfy."""
        point = self.active(params)
        return [c for c in self._after if c.applies(point) and not c.satisfied(point, outputs)]

    def output_margins(self, params: dict[str, Any], outputs: dict[str, float]) -> list[float | None]:
        """Each output constraint's relative margin at params, in order: None where it doesn't apply.

        A relative margin is the margin divided by the size of its limit, as for relative_violation,
        so it is zero or less where the constraint is satisfied: stress <= 250 at a stress of 200
        has -0.2. A limit of size zero keeps its margin as it is.
        """
        point = self.active(params)
        margins: list[float | None] = []
        for c in self._after:
            if not c.applies(point):
                margins.append(None)
                continue
            margin, size = c.margin(point, outputs), c.size(point, outputs)
            margins.append(margin / size if math.isfinite(margin) and math.isfinite(size) and size > 0 else margin)
        return margins

    def is_conditional(self, name: str) -> bool:
        """Whether a condition switches the parameter on and off."""
        return name in self._tests

    def active(self, params: dict[str, Any]) -> dict[str, Any]:
        """params without the parameters the conditions switch off, in parameter order.

        This is the point an objective is called with. params may hold a value for every
        parameter, or only for those switched on.
        """
        if not self._tests:
            return dict(params)
        kept: dict[str, Any] = {}
        for parameter in self._order:
            test = self._tests.get(parameter.name)
            if parameter.name in params and (test is None or holds(test, kept)):
                kept[parameter.name] = params[parameter.name]
        return {name: kept[name] for name in self.names if name in kept}

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw one value for every parameter, at a point that satisfies every constraint.

        Parameters the conditions switch off get a value too, which active() leaves out. Points
        are drawn until one is feasible. Raises InfeasibleSearchSpace after MAX_DRAWS draws in
        a row without one.
        """
        for _ in range(MAX_DRAWS if self._before else 1):
            point = {parameter.name: parameter.sample(rng) for parameter in self.parameters}
            if self.feasible(point):
                return point
        raise InfeasibleSearchSpace(
            f"no point satisfying every constraint was found in {MAX_DRAWS:,} random draws: the constraints may "
            "contradict each other, or leave too small a region to sample"
        )

    def with_functions(self, functions: dict[str, Any]) -> "SearchSpace":
        """This search space with function constraints given their functions by name, as after loading one.

        Raises ValueError if a function constraint is left without one.
        """
        constraints = [_with_function(constraint, functions) for constraint in self.constraints]
        named = {c.name for constraint in self.constraints for c in _function_constraints(constraint)}
        unknown = sorted(set(functions) - named)
        if unknown:
            raise ValueError(f"no function constraint is named {', '.join(map(repr, unknown))}")
        return SearchSpace(self.parameters, constraints, self.conditions)

    def feasible(self, params: dict[str, Any]) -> bool:
        """Whether params satisfy every constraint on the parameters that applies to those switched on.

        Constraints on outputs aren't checked: they can't be until the point is evaluated.
        """
        point = self.active(params)
        return all(not constraint.applies(point) or constraint.satisfied(point) for constraint in self._before)

    def broken(self, params: dict[str, Any]) -> list[AnyConstraint]:
        """The constraints on the parameters that params don't satisfy."""
        point = self.active(params)
        return [c for c in self._before if c.applies(point) and not c.satisfied(point)]

    def check(self, params: dict[str, Any]) -> dict[str, Any]:
        """params as a complete point of this space, in parameter order, or ValueError if they aren't one.

        Fixed parameters may be left out; every other parameter needs a value, unless the
        conditions switch it off at this point, when it must be left out.
        """
        unknown = sorted(set(params) - set(self.names))
        if unknown:
            raise ValueError(f"unknown parameters: {', '.join(unknown)}")
        checked: dict[str, Any] = {}
        missing = []
        for p in self._order:
            test = self._tests.get(p.name)
            if test is not None and not holds(test, checked):
                if p.name in params:
                    raise ValueError(f"{p.name} isn't used at this point, since it's only used when {describe(test)}: leave it out")
                continue
            if p.name in params:
                checked[p.name] = p.check(params[p.name])
            elif isinstance(p, FixedParameter):
                checked[p.name] = p.value
            else:
                missing.append(p.name)
        if missing:
            raise ValueError(f"missing values for: {', '.join(name for name in self.names if name in missing)}")
        return {name: checked[name] for name in self.names if name in checked}

    def __getitem__(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)


def _check_test(test: ChoiceTest | ExpressionConstraint, by_name: dict[str, Parameter], where: str) -> None:
    if test.output_names():
        raise ValueError(
            f"{where} tests the output {sorted(test.output_names())[0]!r}, but a test, on the left of >>, can only use "
            "parameters: which parameters the objective is passed, or which limits apply, can't depend on what it returns. "
            "Use Constraint.function(..., outputs=[...]) for a limit that depends on an output"
        )
    if isinstance(test, ChoiceTest):
        parameter = by_name.get(test.name)
        if parameter is None:
            raise ValueError(f"{where} tests {test.name!r}, which isn't a parameter of this search space")
        if not isinstance(parameter, ChoiceParameter):
            raise ValueError(f"{where} tests {test.name!r} by its values, which needs a choice parameter")
        if parameter.choose_n is not None:
            raise ValueError(f"{where} tests {test.name!r}, which picks {parameter.choose_n} choices at once")
        unknown = [value for value in test.values if value not in parameter.choices]
        if unknown:
            raise ValueError(f"{where} tests {', '.join(map(repr, unknown))}, which isn't one of {test.name}'s choices")
    else:
        _check_expression(test, by_name, where)


def _check_expression(constraint: ExpressionConstraint, by_name: dict[str, Parameter], where: str) -> None:
    for name in sorted(constraint.names()):
        if name not in by_name:
            raise ValueError(f"{where} uses {name!r}, which isn't a parameter of this search space")
        if not isinstance(by_name[name], RangeParameter):
            raise ValueError(f"{where} uses {name!r}, which isn't a range parameter; use Constraint.function for constraints on other kinds")


def _check_constraint(constraint: AnyConstraint, by_name: dict[str, Parameter]) -> None:
    if isinstance(constraint, ExpressionConstraint):
        _check_expression(constraint, by_name, f"the constraint `{constraint}`")
    elif isinstance(constraint, FunctionConstraint):
        missing = sorted(constraint.required() - set(by_name))
        if missing:
            raise ValueError(
                f"the constraint {constraint.name!r} needs parameters missing from the search space: {', '.join(missing)}; "
                f"if {'it is an output' if len(missing) == 1 else 'they are outputs'}, list {'it' if len(missing) == 1 else 'them'} "
                f"in Constraint.function(..., outputs={missing!r})"
            )
        if constraint.func is not None and constraint.outputs:
            signature = inspect.signature(constraint.func)
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                unaccepted = [name for name in constraint.outputs if name not in signature.parameters]
                if unaccepted:
                    raise ValueError(f"the constraint {constraint.name!r} is passed the outputs {', '.join(unaccepted)}, but its function doesn't take them")
    elif isinstance(constraint, ConditionalConstraint):
        _check_test(constraint.when, by_name, f"the constraint `{constraint}`")
        _check_constraint(constraint.constraint, by_name)


def _function_constraints(constraint: AnyConstraint) -> list[FunctionConstraint]:
    """The function constraints in constraint, including inside conditional constraints."""
    if isinstance(constraint, FunctionConstraint):
        return [constraint]
    if isinstance(constraint, ConditionalConstraint):
        return _function_constraints(constraint.constraint)
    return []


def _with_function(constraint: AnyConstraint, functions: dict[str, Any]) -> AnyConstraint:
    if isinstance(constraint, ConditionalConstraint):
        return constraint.model_copy(update={"constraint": _with_function(constraint.constraint, functions)})
    if isinstance(constraint, FunctionConstraint) and constraint.name in functions:
        constraint = constraint.model_copy(update={"func": functions[constraint.name]})
    if isinstance(constraint, FunctionConstraint) and constraint.func is None:
        raise ValueError(f"the constraint {constraint.name!r} needs its function, which isn't saved; pass it in functions")
    return constraint


def _dependency_order(parameters: list[Parameter], tests: dict[str, ChoiceTest | ExpressionConstraint]) -> list[Parameter]:
    """parameters ordered so each comes after every parameter its test uses, or ValueError if they form a loop."""
    by_name = {parameter.name: parameter for parameter in parameters}
    order: list[Parameter] = []
    state: dict[str, str] = {}  # "visiting" while its dependencies are placed, then "placed"

    def place(name: str, path: list[str]) -> None:
        if state.get(name) == "placed":
            return
        if state.get(name) == "visiting":
            loop = path[path.index(name):] + [name]
            raise ValueError(f"the conditions form a loop: {' needs '.join(loop)}")
        state[name] = "visiting"
        test = tests.get(name)
        for needed in sorted(test.names()) if test is not None else []:
            place(needed, path + [name])
        state[name] = "placed"
        order.append(by_name[name])

    for parameter in parameters:
        place(parameter.name, [])
    return order
