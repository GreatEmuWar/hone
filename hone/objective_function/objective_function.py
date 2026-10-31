import inspect
import numbers
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from hone.parameters import SearchSpace


class ObjectiveFunction(BaseModel):
    """Objective function to be optimized, paired with the search space it is called over."""
    model_config = ConfigDict(frozen=True)

    name: str
    function: Callable
    search_space: SearchSpace

    def model_post_init(self, context: Any) -> None:
        # Checked after validation rather than in a validator, so a mismatch raises a plain ValueError.
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
        # A parameter a condition switches off isn't passed, so the function needs a default for it.
        no_default = sorted(
            name
            for name in names
            if self.search_space.is_conditional(name)
            and name in accepted
            and accepted[name].default is inspect.Parameter.empty
            and accepted[name].kind is not inspect.Parameter.VAR_KEYWORD
        )
        if no_default:
            raise ValueError(
                f"{self.name}() isn't passed {', '.join(no_default)} at points where a condition switches "
                f"{'it' if len(no_default) == 1 else 'them'} off, so give {'it' if len(no_default) == 1 else 'each'} a default, "
                f"such as {no_default[0]}=None"
            )
        if missing:
            raise ValueError(f"{self.name}() requires parameters missing from the search space: {', '.join(missing)}")
