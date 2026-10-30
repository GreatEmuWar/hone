from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SelectionMethod(StrEnum):
    """Selection methods for the genetic algorithm."""
    TOURNAMENT = 'tournament'
    ROULETTE = 'roulette'


class GeneticOptimizerConfig(BaseModel):
    """Configuration for the genetic optimizer."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    population_size: int = Field(default=100, ge=2)
    generations: int = Field(default=25, ge=1)
    mutation_rate: float = Field(default=0.05, ge=0, le=1)  # chance a child is mutated at all
    mutation_threshold: float = Field(default=0.1, ge=0, le=1)  # chance each gene of a mutated child is resampled
    crossover_rate: float = Field(default=0.8, ge=0, le=1)
    elitism: bool = True
    elitism_rate: float = Field(default=0.1, ge=0, lt=1)
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = Field(default=4, ge=2)
    tournament_p: float = Field(default=0.75, ge=0, le=1)
    maximize: bool = False
    threshold: float | None = None
    seed: int | None = None  # fixes the random number generator, so runs can be repeated

    @model_validator(mode="after")
    def _check_tournament_size(self):
        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size {self.tournament_size} cannot exceed population_size {self.population_size}"
            )
        return self

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        """Create a GeneticOptimizerConfig from a JSON or YAML file."""
        filepath = Path(filepath)
        text = filepath.read_text()
        if filepath.suffix.lower() in {".yaml", ".yml"}:
            return cls.model_validate(yaml.safe_load(text) or {})
        return cls.model_validate_json(text)
