import math
import numbers
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

type Value = float | int | str


class _Parameter(BaseModel):
    """A named dimension of the search space, and the values it may take."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str

    def __init__(self, **data: Any) -> None:
        # Each kind takes its own fields positionally and passes them all here as keywords.
        super().__init__(**data)


class RangeParameter(_Parameter):
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


class SearchSpace(BaseModel):
    """The set of parameters an objective function is optimized over."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: list[Parameter] = Field(min_length=1)

    def __init__(self, parameters: Sequence[Parameter], **data: Any) -> None:
        super().__init__(parameters=list(parameters), **data)

    @model_validator(mode="after")
    def _check_unique_names(self):
        names = [parameter.name for parameter in self.parameters]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"each parameter needs a unique name; repeated: {', '.join(duplicates)}")
        return self

    @property
    def names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw one value for every parameter."""
        return {parameter.name: parameter.sample(rng) for parameter in self.parameters}

    def check(self, params: dict[str, Any]) -> dict[str, Any]:
        """params as a complete point of this space, in parameter order, or ValueError if they aren't one.

        Fixed parameters may be left out; every other parameter needs a value.
        """
        unknown = sorted(set(params) - set(self.names))
        if unknown:
            raise ValueError(f"unknown parameters: {', '.join(unknown)}")
        missing = [p.name for p in self.parameters if p.name not in params and not isinstance(p, FixedParameter)]
        if missing:
            raise ValueError(f"missing values for: {', '.join(missing)}")
        checked = {}
        for p in self.parameters:
            if p.name in params:
                checked[p.name] = p.check(params[p.name])
            else:
                assert isinstance(p, FixedParameter)  # every other kind was found missing above
                checked[p.name] = p.value
        return checked

    def __getitem__(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)
