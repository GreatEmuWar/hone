from dataclasses import dataclass, field
from typing import List

@dataclass
class Constraint:
    """Constraint to be applied to the objective function."""
    param_name: str # name of the corresponding parameter in the objective function
    minimum: float = None # minimum value for the parameter, if it exists
    maximum: float = None # maximum value for the parameter, if it exists
    literal: float | str = None # literal value for the parameter, provided as a float or string
    choices: List[float | str] = field(default_factory=list) # list of possible values for the parameter, provided as floats or strings
    choose_n: int = None # number of values to choose from the list of choices