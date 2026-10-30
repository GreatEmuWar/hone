import math
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

type Value = float | int | str


class _Parameter(BaseModel):
    """A named dimension of the search space, and the values it may take."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str


class RangeParameter(_Parameter):
    """A number drawn uniformly from [minimum, maximum], optionally restricted to integers."""
    kind: Literal["range"] = "range"
    minimum: float
    maximum: float
    integer: bool = False

    def __init__(self, name: str, minimum: float, maximum: float, **data: Any) -> None:
        super().__init__(name=name, minimum=minimum, maximum=maximum, **data)

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.minimum > self.maximum:
            raise ValueError(f"{self.name}: minimum {self.minimum} is greater than maximum {self.maximum}")
        if self.integer and math.ceil(self.minimum) > math.floor(self.maximum):
            raise ValueError(f"{self.name}: no integer lies between {self.minimum} and {self.maximum}")
        return self

    def sample(self, rng: np.random.Generator) -> float | int:
        if self.integer:
            return int(rng.integers(math.ceil(self.minimum), math.floor(self.maximum), endpoint=True))
        return float(rng.uniform(self.minimum, self.maximum))

    def mutate(self, value: float | int, rng: np.random.Generator, scale: float) -> float | int:
        """Nudge value by Gaussian noise with a standard deviation of scale times the range width.

        The result is clipped to the bounds. Integer ranges use a step of at least 1, so a
        mutation can always move the value.
        """
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

    def __init__(self, parameters: list[Parameter], **data: Any) -> None:
        super().__init__(parameters=parameters, **data)

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

    def __getitem__(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)
