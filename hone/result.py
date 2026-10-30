from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class GenerationStats(BaseModel):
    """Summary of one generation's fitness, over its successful evaluations.

    The fitness fields are None when every evaluation in the generation failed.
    """
    model_config = ConfigDict(frozen=True)

    generation: int
    evaluations: int  # evaluations up to the end of this generation, failed ones included
    failures: int = 0  # failed evaluations in this generation
    best_fitness: float | None = None
    best_params: dict[str, Any] | None = None
    worst_fitness: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class Result(BaseModel):
    """Outcome of an optimization run. Save it with save() and read it back with load()."""
    model_config = ConfigDict(frozen=True)

    best_fitness: float
    best_params: dict[str, Any]
    generations: list[GenerationStats]  # completed generations; empty partway through the first
    evaluations: int  # evaluations made, failed ones included
    failures: int = 0  # evaluations that failed
    reached_threshold: bool  # whether the run stopped early because it reached config.threshold

    def save(self, filepath: str | Path) -> None:
        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        return cls.model_validate_json(Path(filepath).read_text())
