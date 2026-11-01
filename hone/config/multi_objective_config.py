from typing import Any, Literal

from pydantic import Field, model_validator

from .genetic_optimizer_config import GENERATIONS_PER_BUDGET, MAX_POPULATION, MIN_POPULATION

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
    """Each objective's name, and whether to minimize or maximize it."""
    reference: dict[str, float] | None = None
    """A point every objective must be better than to count toward hypervolume, by objective. None:
    the worst value of each objective in the first generation, a tenth of its range further out."""

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


class NSGA2Config(MultiObjectiveConfig):
    """Configuration for NSGA2, the multi-objective genetic optimizer.

    Breeding works as in the genetic optimizer, with the same settings; selection and elitism
    are NSGA-II's own. With max_evaluations set, population_size and generations default as
    for the genetic optimizer.
    """
    mutation_rate: float | None = Field(default=None, ge=0, le=1)
    """As in GeneticOptimizerConfig: chance each gene of a child is mutated."""
    mutation_scale: float = Field(default=0.1, gt=0)
    """As in GeneticOptimizerConfig: size of a range mutation, as a fraction of the range width."""
    adaptive_mutation: bool | None = None
    """As in GeneticOptimizerConfig: each individual carries its own mutation size."""
    crossover_rate: float = Field(default=0.8, ge=0, le=1)
    """As in GeneticOptimizerConfig: chance each gene of a child comes from its first parent."""
    blend_crossover: bool = True
    """As in GeneticOptimizerConfig: draw each range value of a child from between its parents' values."""
    blend_alpha: float = Field(default=0.1, ge=0)
    """As in GeneticOptimizerConfig: how far beyond its parents' values a blended value may fall."""
    niche_share: float | None = Field(default=None, ge=0, lt=1)
    """As in GeneticOptimizerConfig: share of children bred within niches of parents with the same choices."""

    @model_validator(mode="before")
    @classmethod
    def _defaults_from_budget(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("max_evaluations") is None:
            return data
        budget = data["max_evaluations"]
        data = dict(data)
        if data.get("population_size") is None:
            data["population_size"] = min(max(budget // GENERATIONS_PER_BUDGET, MIN_POPULATION), MAX_POPULATION)
        if data.get("generations") is None:
            data["generations"] = budget
        return data


class MultiObjectiveBayesianConfig(MultiObjectiveConfig):
    """Configuration for multi-objective Bayesian optimization. Usually only the budget, max_evaluations, needs setting.

    As with BayesianConfig, each generation is a batch of candidates chosen together, and with
    max_evaluations set, generations defaults to the budget.
    """
    population_size: int | None = Field(default=None, ge=1)  # type: ignore[assignment]
    """Candidates per batch. None: 1, or the number of workers."""
    generations: int = Field(default=100, ge=1)
    """Batches to run, unless the run stops sooner."""
    initial_samples: int | None = Field(default=None, ge=1)
    """Spread-out candidates evaluated before the models are used. None: 2 × (variables + 1)."""
    method: Literal["parego", "models"] = "parego"
    """How candidates are chosen. "parego": a random weighting of the scaled objectives per
    candidate, scored by one Gaussian process. "models": one Gaussian process per objective,
    with the weighting applied to their predictions."""

    @model_validator(mode="before")
    @classmethod
    def _generations_from_budget(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("max_evaluations") is not None and data.get("generations") is None:
            data = {**data, "generations": data["max_evaluations"]}
        return data
