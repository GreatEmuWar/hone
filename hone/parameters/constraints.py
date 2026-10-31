"""Constraints on a search space: limits that every point an optimizer asks for must satisfy.

An expression is built from range parameters and numbers with ordinary arithmetic, and comparing
it with <= or >= makes a constraint:

    width * height <= 40

Expressions are pydantic models, so a search space with constraints saves and loads like any other.
Constraint.function covers what an expression can't say, at the cost of not being saved.
A constraint that only applies while a condition holds is written with >>; see conditions.py.
"""
import inspect
import math
import numbers
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
        f"constraints are built from range parameters and numbers, not {value!r}; "
        "for anything else, such as choice parameters, use Constraint.function"
    )


class _Term(_Arithmetic, BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def _as_term(self) -> "Term":
        return self  # type: ignore[return-value]

    def evaluate(self, params: dict[str, Any]) -> float:
        """The expression's value at params."""
        raise NotImplementedError

    def names(self) -> set[str]:
        """The parameters the expression uses."""
        raise NotImplementedError

    def _precedence(self) -> int:
        return 5

    def __str__(self) -> str:
        raise NotImplementedError


class Variable(_Term):
    """A range parameter's value, by name."""
    kind: Literal["variable"] = "variable"
    name: str

    def evaluate(self, params: dict[str, Any]) -> float:
        return float(params[self.name])

    def names(self) -> set[str]:
        return {self.name}

    def __str__(self) -> str:
        return self.name


class Number(_Term):
    kind: Literal["number"] = "number"
    value: float

    def evaluate(self, params: dict[str, Any]) -> float:
        return self.value

    def names(self) -> set[str]:
        return set()

    def __str__(self) -> str:
        return repr(int(self.value)) if self.value.is_integer() and abs(self.value) < 1e15 else repr(self.value)


class BinaryOp(_Term):
    kind: Literal["binary"] = "binary"
    op: Literal["+", "-", "*", "/", "**"]
    left: "Term"
    right: "Term"

    def evaluate(self, params: dict[str, Any]) -> float:
        a, b = self.left.evaluate(params), self.right.evaluate(params)
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

    def evaluate(self, params: dict[str, Any]) -> float:
        value = self.operand.evaluate(params)
        return -value if self.op == "neg" else abs(value)

    def names(self) -> set[str]:
        return self.operand.names()

    def _precedence(self) -> int:
        return _PRECEDENCE["neg"] if self.op == "neg" else 5

    def __str__(self) -> str:
        if self.op == "abs":
            return f"abs({self.operand})"
        return f"-({self.operand})" if self.operand._precedence() < self._precedence() else f"-{self.operand}"


type Term = Annotated[Variable | Number | BinaryOp | UnaryOp, Field(discriminator="kind")]


class Constraint(BaseModel):
    """A limit every point an optimizer asks for must satisfy.

    Build one by comparing an expression of range parameters with <= or >=, such as
    `width * height <= 40`, or with Constraint.function for anything an expression can't say.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    def violation(self, params: dict[str, Any]) -> float:
        """How far params break the constraint: 0 when they satisfy it, infinity when it can't be computed."""
        raise NotImplementedError

    def satisfied(self, params: dict[str, Any]) -> bool:
        return self.violation(params) == 0

    def names(self) -> set[str]:
        """The parameters the constraint uses."""
        raise NotImplementedError

    def applies(self, params: dict[str, Any]) -> bool:
        """Whether the constraint applies at params: not when a parameter it needs is switched off by a condition."""
        return self.names() <= params.keys()

    def __bool__(self) -> bool:
        raise TypeError(
            f"the constraint `{self}` has no truth value. A chained comparison such as `0 <= x + y <= 10` "
            "keeps only one half; write two constraints, `x + y >= 0` and `x + y <= 10`"
        )

    @staticmethod
    def function(function: Callable[..., float], name: str | None = None) -> "FunctionConstraint":
        """A constraint computed by function, which is satisfied when it returns zero or less.

        The function takes parameters by name, like an objective function, and may take only
        some of them. It can use anything, such as square roots or choice parameters, but it
        isn't saved with the search space: a run record stores only its name.
        """
        return FunctionConstraint(name=name or getattr(function, "__name__", "constraint"), func=function)


def _limit(value: float, scale: float) -> float:
    """value as a violation: 0 within the tolerance, infinity when it couldn't be computed."""
    if isinstance(value, complex) or math.isnan(value):
        return math.inf
    return 0.0 if value <= TOLERANCE * max(1.0, abs(scale)) else float(value)


class ExpressionConstraint(Constraint):
    """An expression compared with another: left <= right, or left >= right."""
    kind: Literal["expression"] = "expression"
    left: Term
    op: Literal["<=", ">="]
    right: Term

    def violation(self, params: dict[str, Any]) -> float:
        try:
            left, right = self.left.evaluate(params), self.right.evaluate(params)
        except (ArithmeticError, ValueError):
            return math.inf  # division by zero, or an overflow: the point can't be judged, so it isn't allowed
        if isinstance(left, complex) or isinstance(right, complex):
            return math.inf  # a negative number to a fractional power
        return _limit(left - right if self.op == "<=" else right - left, right)

    def names(self) -> set[str]:
        return self.left.names() | self.right.names()

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
    """A constraint computed by a function of the parameters: satisfied when it returns zero or less."""
    kind: Literal["function"] = "function"
    name: str
    func: Callable[..., float] | None = Field(default=None, exclude=True)  # not saved; None once loaded

    def violation(self, params: dict[str, Any]) -> float:
        if self.func is None:
            raise ValueError(
                f"the constraint {self.name!r} was loaded without its function, which can't be saved; "
                "pass it again, such as with rebuild_optimizer(functions=...)"
            )
        signature = inspect.signature(self.func)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            arguments = params
        else:
            arguments = {name: value for name, value in params.items() if name in signature.parameters}
        value = self.func(**arguments)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"the constraint {self.name!r} must return a number, but returned {value!r}")
        return _limit(float(value), 0.0)

    def names(self) -> set[str]:
        if self.func is None:
            return set()
        signature = inspect.signature(self.func)
        return {
            name
            for name, p in signature.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    def applies(self, params: dict[str, Any]) -> bool:
        return self.required() <= params.keys()

    def required(self) -> set[str]:
        """The parameters the function can't be called without."""
        if self.func is None:
            return set()
        return {
            name
            for name, p in inspect.signature(self.func).parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    def __str__(self) -> str:
        return self.name


class InfeasibleSearchSpace(ValueError):
    """No point satisfying every constraint could be found."""


BinaryOp.model_rebuild()
UnaryOp.model_rebuild()
