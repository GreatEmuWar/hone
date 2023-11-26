from typing import Callable, Dict, List
from dataclasses import dataclass, field
from hone.constraint import Constraint

@dataclass
class ObjectiveFunction:
    """Objective function to be optimized."""
    name: str
    function: Callable
    params: Dict = field(default_factory=dict)
    optimum: float = 0.0
    optimum_params: Dict = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    dimensions: int = 0
    maximize: bool = False

    def __post_init__(self):
        self.dimensions = len(self.constraints)

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def add_constraints(self, constraints: List[Constraint]):
        self.constraints.extend(constraints)