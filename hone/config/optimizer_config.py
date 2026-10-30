from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field


class OptimizerConfig(BaseModel):
    """Settings shared by every optimizer."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    population_size: int = Field(default=100, ge=2)  # individuals evaluated per generation
    generations: int = Field(default=25, ge=1)
    max_evaluations: int | None = Field(default=None, ge=1)  # stop after this many evaluations, even mid-generation
    maximize: bool = False
    threshold: float | None = None  # stop once the best fitness reaches this
    seed: int | None = None  # fixes the random number generator, so runs can be repeated

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        """Create a config from a JSON or YAML file."""
        filepath = Path(filepath)
        text = filepath.read_text()
        if filepath.suffix.lower() in {".yaml", ".yml"}:
            return cls.model_validate(yaml.safe_load(text) or {})
        return cls.model_validate_json(text)
