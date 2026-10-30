from enum import StrEnum

from typing import Any

from pydantic import Field, model_validator

from .optimizer_config import OptimizerConfig


class SelectionMethod(StrEnum):
    """Selection methods for the genetic algorithm."""
    TOURNAMENT = 'tournament'
    ROULETTE = 'roulette'


# With a budget and no population size, the population is sized for about this many
# generations, within these bounds. Chosen from the benchmark's small budgets.
GENERATIONS_PER_BUDGET = 20
MIN_POPULATION = 16
MAX_POPULATION = 50


class GeneticOptimizerConfig(OptimizerConfig):
    """Configuration for the genetic optimizer.

    When max_evaluations is set, population_size and generations default to suit it: the
    population to about a twentieth of the budget, between 16 and 50, and generations to
    as many as the budget allows. Without a budget they default to 100 and 25.
    """
    mutation_rate: float = Field(default=0.1, ge=0, le=1)  # chance each gene of a child is mutated
    mutation_scale: float = Field(default=0.1, gt=0)  # size of a range mutation, as a fraction of the range width
    crossover_rate: float = Field(default=0.8, ge=0, le=1)  # chance each gene of a child comes from its first parent
    elitism: bool = True
    elitism_rate: float = Field(default=0.1, ge=0, lt=1)  # share of each generation carried over unchanged
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = Field(default=4, ge=2)
    tournament_p: float = Field(default=1.0, ge=0, le=1)  # chance a tournament picks its best rather than its worst

    @model_validator(mode="before")
    @classmethod
    def _defaults_from_budget(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("max_evaluations") is None:
            return data
        budget = data["max_evaluations"]
        data = dict(data)
        if data.get("population_size") is None:
            population = min(max(budget // GENERATIONS_PER_BUDGET, MIN_POPULATION), MAX_POPULATION)
            data["population_size"] = max(population, data.get("tournament_size") or 0)
        if data.get("generations") is None:
            data["generations"] = budget
        return data

    @model_validator(mode="after")
    def _check_tournament_size(self):
        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size {self.tournament_size} cannot exceed population_size {self.population_size}"
            )
        return self
