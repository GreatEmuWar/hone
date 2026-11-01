from typing import Literal

from pydantic import Field, model_validator

from .optimizer_config import OptimizerConfig

type Direction = Literal["minimize", "maximize"]


class MultiObjectiveConfig(OptimizerConfig):
    """Settings for optimizing several objectives together.

    objectives names each objective and whether to minimize or maximize it, such as
    {"accuracy": "maximize", "cost": "minimize"}; the objective function returns a dict with
    exactly those keys. It takes the place of maximize, and threshold and final_evaluations
    don't apply.
    """
    objectives: dict[str, Direction] = Field(min_length=2)
    # A point every objective must be better than to count toward hypervolume, by objective. None:
    # the worst value of each objective in the first generation, a tenth of its range further out.
    reference: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_objectives(self):
        if self.maximize:
            raise ValueError("objectives says whether to minimize or maximize each objective, so maximize doesn't apply")
        if self.threshold is not None:
            raise ValueError("threshold doesn't apply to several objectives; stop a run with a callback instead")
        if self.final_evaluations:
            raise ValueError("final_evaluations doesn't apply to several objectives yet")
        if self.reference is not None and set(self.reference) != set(self.objectives):
            raise ValueError(f"reference needs a value for each objective: {', '.join(self.objectives)}")
        return self
