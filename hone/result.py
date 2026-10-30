from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


class GenerationStats(BaseModel):
    """Summary of one generation's fitness."""
    model_config = ConfigDict(frozen=True)

    generation: int
    best_fitness: float
    best_params: dict[str, Any]
    worst_fitness: float
    mean: float
    median: float
    std: float


class Result(BaseModel):
    """Outcome of an optimization run. Save it with save() and read it back with load()."""
    model_config = ConfigDict(frozen=True)

    best_fitness: float
    best_params: dict[str, Any]
    generations: list[GenerationStats] = Field(min_length=1)
    evaluations: int  # calls made to the objective function
    reached_threshold: bool  # whether the run stopped early because it reached config.threshold

    def save(self, filepath: str | Path) -> None:
        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        return cls.model_validate_json(Path(filepath).read_text())
