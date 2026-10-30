import inspect
import numbers
from dataclasses import dataclass
from typing import Any, Callable

from hone.parameters import SearchSpace


@dataclass(frozen=True)
class ObjectiveFunction:
    """Objective function to be optimized, paired with the search space it is called over."""
    name: str
    function: Callable
    search_space: SearchSpace

    def __post_init__(self):
        self._check_signature()

    def __call__(self, params: dict[str, Any]) -> float:
        """Evaluate the function at params, which must return a real number."""
        value = self.function(**params)
        if not isinstance(value, numbers.Real):
            raise TypeError(f"{self.name}() must return a number, but returned {value!r}")
        return float(value)

    def _check_signature(self):
        """Fail before optimizing if the function cannot be called with the search space's parameters."""
        signature = inspect.signature(self.function)
        accepted = signature.parameters
        takes_any_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())

        names = set(self.search_space.names)
        unknown = sorted(name for name in names if name not in accepted and not takes_any_keyword)
        missing = sorted(
            name
            for name, p in accepted.items()
            if name not in names
            and p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        if unknown:
            raise ValueError(f"{self.name}() does not accept parameters: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"{self.name}() requires parameters missing from the search space: {', '.join(missing)}")
