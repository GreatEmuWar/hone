from pydantic import Field

from .optimizer_config import OptimizerConfig


class CMAESConfig(OptimizerConfig):
    """Configuration for CMA-ES. The defaults suit most problems; usually only the budget needs setting."""
    population_size: int | None = Field(default=None, ge=2)  # None picks 4 + 3 ln(n) for n variables
    generations: int = Field(default=1000, ge=1)  # an upper bound; max_evaluations is the usual way to set a budget
    initial_step: float = Field(default=0.3, gt=0, le=1)  # starting step size, as a fraction of each range
    restarts: bool = True  # restart with double the population when a run converges or stalls (IPOP-CMA-ES)
