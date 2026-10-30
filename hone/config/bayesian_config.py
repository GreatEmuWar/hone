from typing import Any

from pydantic import Field, model_validator

from .optimizer_config import OptimizerConfig


class BayesianConfig(OptimizerConfig):
    """Configuration for Bayesian optimization. Usually only the budget, max_evaluations, needs setting.

    Each generation is a batch of candidates chosen together. When max_evaluations is set and
    generations isn't, generations defaults to the budget, so the run ends when the budget is spent.
    """
    population_size: int | None = Field(default=None, ge=1)  # candidates per batch; None: 1, or the number of workers
    generations: int = Field(default=100, ge=1)
    initial_samples: int | None = Field(default=None, ge=1)  # spread-out candidates before the model is used; None: 2 × (variables + 1)

    @model_validator(mode="before")
    @classmethod
    def _generations_from_budget(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("max_evaluations") is not None and data.get("generations") is None:
            data = {**data, "generations": data["max_evaluations"]}
        return data
