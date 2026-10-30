import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict

from hone.parameters import SearchSpace


@dataclass
class ObjectiveFunction:
    """Objective function to be optimized."""
    name: str
    function: Callable
    search_space: SearchSpace
    optimum: float | None = None
    optimum_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        self._check_signature()

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
