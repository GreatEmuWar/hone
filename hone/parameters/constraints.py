"""Constraints on a search space: limits on the parameters, and on what an evaluation reports.

An expression is built from range parameters, outputs and numbers with ordinary arithmetic, and
comparing it with <= or >= makes a constraint:

    width * height <= 40             # on the parameters: every point an optimizer asks for satisfies it
    Output("stress") <= 250          # on an output: judged once the evaluation reports the stress

A constraint on the parameters is checked before evaluating, so the objective is never called
where it is broken. One that uses an output can only be judged afterwards: the objective returns
the output, and the optimizer ranks the points that break the limit below those that don't.

Expressions are pydantic models, so a search space with constraints saves and loads like any other.
Constraint.function covers what an expression can't say, at the cost of not being saved.
A constraint that only applies while a condition holds is written with >>; see conditions.py.
"""
import inspect
import math
import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, overload

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .conditions import Condition, ConditionalConstraint

TOLERANCE = 1e-9  # a point this close to a limit, relative to the limit's size, satisfies it

_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "**": 4}


class _Arithmetic:
    """Arithmetic and comparison for anything that can stand in an expression: range parameters and expressions."""

    def _as_term(self) -> "Term":
        raise NotImplementedError

    def __add__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="+", left=self._as_term(), right=_term(other))

    def __radd__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="+", left=_term(other), right=self._as_term())

    def __sub__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="-", left=self._as_term(), right=_term(other))

    def __rsub__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="-", left=_term(other), right=self._as_term())

    def __mul__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="*", left=self._as_term(), right=_term(other))

    def __rmul__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="*", left=_term(other), right=self._as_term())

    def __truediv__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="/", left=self._as_term(), right=_term(other))

    def __rtruediv__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="/", left=_term(other), right=self._as_term())

    def __pow__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="**", left=self._as_term(), right=_term(other))

    def __rpow__(self, other: Any) -> "BinaryOp":
        return BinaryOp(op="**", left=_term(other), right=self._as_term())

    def __neg__(self) -> "UnaryOp":
        return UnaryOp(op="neg", operand=self._as_term())

    def __abs__(self) -> "UnaryOp":
        return UnaryOp(op="abs", operand=self._as_term())

    # < and > mean the same as <= and >=: for numbers searched continuously the difference has no effect.
    def __le__(self, other: Any) -> "ExpressionConstraint":
        return ExpressionConstraint(left=self._as_term(), op="<=", right=_term(other))

    def __ge__(self, other: Any) -> "ExpressionConstraint":
        return ExpressionConstraint(left=self._as_term(), op=">=", right=_term(other))

    __lt__ = __le__
    __gt__ = __ge__


def _term(value: Any) -> "Term":
    if isinstance(value, _Arithmetic):
        return value._as_term()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return Number(value=float(value))
    raise TypeError(
        f"constraints are built from range parameters, outputs and numbers, not {value!r}; "
        "for anything else, such as choice parameters, use Constraint.function"
    )


class _Term(_Arithmetic, BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def __init__(self, **data: Any) -> None:
        # Output takes its name positionally and passes it here as a keyword.
        super().__init__(**data)

    def _as_term(self) -> "Term":
        return self  # type: ignore[return-value]

    def evaluate(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        """The expression's value at params, with outputs giving the value of each output it uses."""
        raise NotImplementedError

    def names(self) -> set[str]:
        """The parameters the expression uses."""
        raise NotImplementedError

    def output_names(self) -> set[str]:
        """The outputs the expression uses."""
        raise NotImplementedError

    def _precedence(self) -> int:
        return 5

    def __str__(self) -> str:
        raise NotImplementedError


class Variable(_Term):
    """A range parameter's value, by name."""
    kind: Literal["variable"] = "variable"
    name: str

    def evaluate(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        return float(params[self.name])

    def names(self) -> set[str]:
        return {self.name}

    def output_names(self) -> set[str]:
        return set()

    def __str__(self) -> str:
        return self.name


class Output(_Term):
    """A value the objective reports, by name, for a constraint judged after evaluating.

    `Output("stress") <= 250` limits the stress a simulation reports. With constraints on
    outputs, the objective returns a dict: its value under "objective" (or each objective, with
    several) and a number for each output the constraints use.
    """
    kind: Literal["output"] = "output"
    name: str

    def __init__(self, name: str, **data: Any) -> None:
        super().__init__(name=name, **data)

    def evaluate(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        if outputs is None or self.name not in outputs:
            raise KeyError(f"the output {self.name!r} is needed but wasn't given")
        return float(outputs[self.name])

    def names(self) -> set[str]:
        return set()

    def output_names(self) -> set[str]:
        return {self.name}

    def __str__(self) -> str:
        return self.name


class Number(_Term):
    kind: Literal["number"] = "number"
    value: float

    def evaluate(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        return self.value

    def names(self) -> set[str]:
        return set()

    def output_names(self) -> set[str]:
        return set()

    def __str__(self) -> str:
        return repr(int(self.value)) if self.value.is_integer() and abs(self.value) < 1e15 else repr(self.value)


class BinaryOp(_Term):
    kind: Literal["binary"] = "binary"
    op: Literal["+", "-", "*", "/", "**"]
    left: "Term"
    right: "Term"

    def evaluate(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        a, b = self.left.evaluate(params, outputs), self.right.evaluate(params, outputs)
        match self.op:
            case "+":
                return a + b
            case "-":
                return a - b
            case "*":
                return a * b
            case "/":
                return a / b
            case "**":
                return a**b

    def names(self) -> set[str]:
        return self.left.names() | self.right.names()

    def output_names(self) -> set[str]:
        return self.left.output_names() | self.right.output_names()

    def _precedence(self) -> int:
        return _PRECEDENCE[self.op]

    def __str__(self) -> str:
        mine = self._precedence()
        # ** groups to the right, the others to the left, so the other side needs brackets at equal precedence.
        left_brackets = self.left._precedence() < mine or (self.op == "**" and self.left._precedence() == mine)
        right_brackets = self.right._precedence() < mine or (self.op in ("-", "/") and self.right._precedence() == mine)
        left = f"({self.left})" if left_brackets else str(self.left)
        right = f"({self.right})" if right_brackets else str(self.right)
        return f"{left} {self.op} {right}"


class UnaryOp(_Term):
    kind: Literal["unary"] = "unary"
    op: Literal["neg", "abs"]
    operand: "Term"

    def evaluate(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        value = self.operand.evaluate(params, outputs)
        return -value if self.op == "neg" else abs(value)

    def names(self) -> set[str]:
        return self.operand.names()

    def output_names(self) -> set[str]:
        return self.operand.output_names()

    def _precedence(self) -> int:
        return _PRECEDENCE["neg"] if self.op == "neg" else 5

    def __str__(self) -> str:
        if self.op == "abs":
            return f"abs({self.operand})"
        return f"-({self.operand})" if self.operand._precedence() < self._precedence() else f"-{self.operand}"


type Term = Annotated[Variable | Output | Number | BinaryOp | UnaryOp, Field(discriminator="kind")]


class Constraint(BaseModel):
    """A limit on the parameters, which every point an optimizer asks for satisfies, or on an output.

    Build one by comparing an expression of range parameters and outputs with <= or >=, such as
    `width * height <= 40` or `Output("stress") <= 250`, or with Constraint.function for anything
    an expression can't say.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    def margin(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        """By how much params break the constraint, in its own units: zero or less when they satisfy it.

        For left <= right it is left - right, for left >= right it is right - left, and for a
        function it is the function's value. Infinity when it can't be computed.
        """
        raise NotImplementedError

    def size(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        """The size of the limit, |right|, which relative_violation divides by; 0 for a function."""
        return 0.0

    def violation(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        """How far params break the constraint: 0 when they satisfy it, infinity when it can't be computed."""
        return _limit(self.margin(params, outputs), self.size(params, outputs))

    def relative_violation(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        """The violation as a share of the limit's size, so limits in different units compare: 0.1 is 10% over.

        A limit of size zero, such as a function's, has nothing to divide by, so its violation counts as it is.
        """
        violation = self.violation(params, outputs)
        size = self.size(params, outputs)
        return violation / size if violation and math.isfinite(size) and size > 0 else violation

    def satisfied(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> bool:
        return self.violation(params, outputs) == 0

    def names(self) -> set[str]:
        """The parameters the constraint uses."""
        raise NotImplementedError

    def output_names(self) -> set[str]:
        """The outputs the constraint uses: none for a constraint checked before evaluating."""
        return set()

    def applies(self, params: dict[str, Any]) -> bool:
        """Whether the constraint applies at params: not when a parameter it needs is switched off by a condition."""
        return self.names() <= params.keys()

    def __bool__(self) -> bool:
        raise TypeError(
            f"the constraint `{self}` has no truth value. A chained comparison such as `0 <= x + y <= 10` "
            "keeps only one half; write two constraints, `x + y >= 0` and `x + y <= 10`"
        )

    @staticmethod
    def function(
        function: Callable[..., float], name: str | None = None, *, outputs: Sequence[str] = ()
    ) -> "FunctionConstraint":
        """A constraint computed by function, which is satisfied when it returns zero or less.

        The function takes parameters by name, like an objective function, and may take only
        some of them. It can use anything, such as square roots or choice parameters, but it
        isn't saved with the search space: a run record stores only its name.

        With outputs, the function is also passed those outputs by name, and the constraint
        is judged after evaluating. Return a relative margin, such as stress / LIMIT - 1,
        when other output constraints share the problem, so their violations compare.
        """
        return FunctionConstraint(name=name or getattr(function, "__name__", "constraint"), func=function, outputs=list(outputs))


def _limit(value: float, scale: float) -> float:
    """value as a violation: 0 within the tolerance, infinity when it couldn't be computed."""
    if isinstance(value, complex) or math.isnan(value):
        return math.inf
    return 0.0 if value <= TOLERANCE * max(1.0, abs(scale)) else float(value)


def _usable(value: Any) -> float | None:
    """value as a real number, or None for one that can't be judged: complex, from a negative number to a fractional power."""
    return None if isinstance(value, complex) else float(value)


class ExpressionConstraint(Constraint):
    """An expression compared with another: left <= right, or left >= right."""
    kind: Literal["expression"] = "expression"
    left: Term
    op: Literal["<=", ">="]
    right: Term

    def _sides(self, params: dict[str, Any], outputs: dict[str, float] | None) -> tuple[float, float] | None:
        """Both sides' values, or None when they can't be computed."""
        try:
            left, right = _usable(self.left.evaluate(params, outputs)), _usable(self.right.evaluate(params, outputs))
        except (ArithmeticError, ValueError):
            return None  # division by zero, or an overflow: the point can't be judged, so it isn't allowed
        if left is None or right is None or math.isnan(left) or math.isnan(right):
            return None
        return left, right

    def margin(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        sides = self._sides(params, outputs)
        if sides is None:
            return math.inf
        left, right = sides
        value = left - right if self.op == "<=" else right - left
        return math.inf if math.isnan(value) else value

    def size(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        sides = self._sides(params, outputs)
        return 0.0 if sides is None else abs(sides[1])

    def violation(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        # Both sides once, since every candidate an optimizer considers is checked.
        sides = self._sides(params, outputs)
        if sides is None:
            return math.inf
        left, right = sides
        return _limit(left - right if self.op == "<=" else right - left, right)

    def names(self) -> set[str]:
        return self.left.names() | self.right.names()

    def output_names(self) -> set[str]:
        return self.left.output_names() | self.right.output_names()

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"

    @overload
    def __rshift__(self, other: "Constraint") -> "ConditionalConstraint": ...
    @overload
    def __rshift__(self, other: Any) -> "Condition": ...
    def __rshift__(self, other: Any) -> "Condition | ConditionalConstraint":
        # As the test of a condition: (layers >= 3) >> units_3.
        from .conditions import _gate

        return _gate(self, other)


class FunctionConstraint(Constraint):
    """A constraint computed by a function of the parameters, and of outputs if it names any: satisfied when it returns zero or less."""
    kind: Literal["function"] = "function"
    name: str
    func: Callable[..., float] | None = Field(default=None, exclude=True)  # not saved; None once loaded
    outputs: list[str] = Field(default_factory=list)  # the outputs it is passed, by name

    def margin(self, params: dict[str, Any], outputs: dict[str, float] | None = None) -> float:
        if self.func is None:
            raise ValueError(
                f"the constraint {self.name!r} was loaded without its function, which can't be saved; "
                "pass it again, such as with rebuild_optimizer(functions=...)"
            )
        signature = inspect.signature(self.func)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            arguments = dict(params)
        else:
            arguments = {name: value for name, value in params.items() if name in signature.parameters}
        if self.outputs:
            missing = [name for name in self.outputs if outputs is None or name not in outputs]
            if missing:
                raise KeyError(f"the constraint {self.name!r} needs the outputs {', '.join(missing)}, which weren't given")
            assert outputs is not None
            arguments.update({name: outputs[name] for name in self.outputs})
        value = self.func(**arguments)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"the constraint {self.name!r} must return a number, but returned {value!r}")
        return math.inf if math.isnan(value) else float(value)

    def names(self) -> set[str]:
        return self._arguments() - set(self.outputs)

    def output_names(self) -> set[str]:
        return set(self.outputs)

    def applies(self, params: dict[str, Any]) -> bool:
        return self.required() <= params.keys()

    def _arguments(self) -> set[str]:
        if self.func is None:
            return set()
        signature = inspect.signature(self.func)
        return {
            name
            for name, p in signature.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    def required(self) -> set[str]:
        """The parameters the function can't be called without."""
        if self.func is None:
            return set()
        return {
            name
            for name, p in inspect.signature(self.func).parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in self.outputs
        }

    def __str__(self) -> str:
        return self.name


class InfeasibleSearchSpace(ValueError):
    """No point satisfying every constraint could be found."""


BinaryOp.model_rebuild()
UnaryOp.model_rebuild()
