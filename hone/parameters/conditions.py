"""Conditional parameters: parameters that exist only while others take certain values.

`test >> parameter` reads "if test, then parameter":

    model["svm"] >> [kernel, C]       # kernel and C exist only when model is "svm"
    kernel["rbf"] >> gamma            # gamma only when kernel is "rbf", so only with an svm too
    (layers >= 3) >> units_3          # a comparison of range parameters, in brackets

A test is a choice parameter indexed by the values that switch the parameters on, or a
comparison of range parameters like a constraint's. A parameter that is switched off isn't part
of the point: the objective isn't passed it, and results leave it out.

With a constraint on the right, `test >> constraint` is a constraint that only applies while
the test holds:

    optimizer["sgd"] >> (momentum * lr <= 0.1)

Conditions and tests are pydantic models, so a search space with conditions saves and loads
like any other.
"""
import math
import numbers
from typing import Annotated, Any, Literal, Union, overload

from pydantic import BaseModel, ConfigDict, Field

from .constraints import Constraint, ExpressionConstraint, FunctionConstraint


class ChoiceTest(BaseModel):
    """Whether a choice parameter takes one of some values: written choice["a", "b"]."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["choice"] = "choice"
    name: str
    values: list[float | int | str] = Field(min_length=1)

    def holds(self, params: dict[str, Any]) -> bool:
        """Whether params give the parameter one of the values. False when it has none, being switched off."""
        return self.name in params and params[self.name] in self.values

    def names(self) -> set[str]:
        return {self.name}

    def output_names(self) -> set[str]:
        return set()

    def describe(self) -> str:
        return f"{self.name} is {' or '.join(map(repr, self.values))}"

    def __str__(self) -> str:
        return f"{self.name}[{', '.join(map(repr, self.values))}]"

    @overload
    def __rshift__(self, other: Constraint) -> "ConditionalConstraint": ...
    @overload
    def __rshift__(self, other: Any) -> "Condition": ...
    def __rshift__(self, other: Any) -> "Condition | ConditionalConstraint":
        return _gate(self, other)

    def __bool__(self) -> bool:
        raise TypeError(f"`{self}` has no truth value: it is a test, for the left of >>, such as `{self} >> parameter`")


type Test = Annotated[ChoiceTest | ExpressionConstraint, Field(discriminator="kind")]


def holds(test: ChoiceTest | ExpressionConstraint, params: dict[str, Any]) -> bool:
    """Whether test holds at params. A test of a parameter that is switched off doesn't hold."""
    if isinstance(test, ChoiceTest):
        return test.holds(params)
    return test.names() <= params.keys() and test.satisfied(params)


def describe(test: ChoiceTest | ExpressionConstraint) -> str:
    return test.describe() if isinstance(test, ChoiceTest) else str(test)


def notation(test: ChoiceTest | ExpressionConstraint) -> str:
    """The test as written on the left of >>."""
    return str(test) if isinstance(test, ChoiceTest) else f"({test})"


class Condition(BaseModel):
    """`when >> parameters`: the parameters exist only while the test when holds."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    when: Test
    parameters: list[str] = Field(min_length=1)

    def __str__(self) -> str:
        targets = self.parameters[0] if len(self.parameters) == 1 else f"[{', '.join(self.parameters)}]"
        return f"{notation(self.when)} >> {targets}"


class ConditionalConstraint(Constraint):
    """`when >> constraint`: a constraint that only applies while the test when holds."""
    kind: Literal["conditional"] = "conditional"
    when: Test
    constraint: "AnyConstraint"

    def margin(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        return self.constraint.margin(params, outputs) if holds(self.when, params) else -math.inf

    def size(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        return self.constraint.size(params, outputs) if holds(self.when, params) else 0.0

    def violation(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        return self.constraint.violation(params, outputs) if holds(self.when, params) else 0.0

    def relative_violation(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        return self.constraint.relative_violation(params, outputs) if holds(self.when, params) else 0.0

    def names(self) -> set[str]:
        return self.when.names() | self.constraint.names()

    def output_names(self) -> set[str]:
        return self.when.output_names() | self.constraint.output_names()

    def applies(self, params: dict[str, Any]) -> bool:
        return holds(self.when, params) and self.constraint.applies(params)

    def __str__(self) -> str:
        inner = f"({self.constraint})" if isinstance(self.constraint, ExpressionConstraint) else str(self.constraint)
        return f"{notation(self.when)} >> {inner}"


type AnyConstraint = Annotated[
    Union[ExpressionConstraint, FunctionConstraint, ConditionalConstraint], Field(discriminator="kind")
]


def _gate(test: ChoiceTest | ExpressionConstraint, other: Any) -> Condition | ConditionalConstraint:
    """test >> other: a condition on parameters, or a constraint that applies only while test holds."""
    from .parameters import _Parameter

    if isinstance(other, Constraint):
        return ConditionalConstraint(when=test, constraint=other)
    targets = list(other) if isinstance(other, (list, tuple)) else [other]
    if not targets:
        raise ValueError(f"`{notation(test)} >> []` switches nothing on: give it a parameter or a list of them")
    if all(isinstance(target, _Parameter) for target in targets):
        return Condition(when=test, parameters=[target.name for target in targets])
    raise TypeError(
        f"the right of >> is a parameter, a list of parameters or a constraint, not {other!r}: "
        f"`{notation(test)} >> momentum` switches momentum on only while the test holds"
    )


def gate_error(parameter: Any, other: Any) -> TypeError:
    """The error for a parameter on the right of >> when the left isn't a test."""
    if isinstance(other, numbers.Real):
        return TypeError(
            f"`... >= {other!r} >> {parameter.name}` reads as `... >= ({other!r} >> {parameter.name})`, since >> comes "
            f"before comparisons in Python: put the comparison in brackets, such as `(layers >= {other!r}) >> {parameter.name}`"
        )
    return TypeError(
        f"the left of `>> {parameter.name}` must be a test: a choice parameter's values, such as model['svm'], "
        "or a comparison of range parameters in brackets, such as (layers >= 3)"
    )


ConditionalConstraint.model_rebuild()
