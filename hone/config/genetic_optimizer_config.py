from enum import StrEnum

from pydantic import Field, model_validator

from .optimizer_config import OptimizerConfig


class SelectionMethod(StrEnum):
    """Selection methods for the genetic algorithm."""
    TOURNAMENT = 'tournament'
    ROULETTE = 'roulette'


class GeneticOptimizerConfig(OptimizerConfig):
    """Configuration for the genetic optimizer."""
    mutation_rate: float = Field(default=0.1, ge=0, le=1)  # chance each gene of a child is mutated
    mutation_scale: float = Field(default=0.1, gt=0)  # size of a range mutation, as a fraction of the range width
    crossover_rate: float = Field(default=0.8, ge=0, le=1)  # chance each gene of a child comes from its first parent
    elitism: bool = True
    elitism_rate: float = Field(default=0.1, ge=0, lt=1)  # share of each generation carried over unchanged
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = Field(default=4, ge=2)
    tournament_p: float = Field(default=0.75, ge=0, le=1)  # chance a tournament picks its best rather than its worst

    @model_validator(mode="after")
    def _check_tournament_size(self):
        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size {self.tournament_size} cannot exceed population_size {self.population_size}"
            )
        return self
