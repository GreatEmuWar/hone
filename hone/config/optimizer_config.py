from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OptimizerConfig(BaseModel):
    """Settings shared by every optimizer."""
    # use_attribute_docstrings makes each field's docstring its description, in the JSON schema
    # and in the documentation.
    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    population_size: int = Field(default=100, ge=2)
    """Individuals evaluated per generation."""
    generations: int = Field(default=25, ge=1)
    """Generations to run, unless the run stops sooner."""
    max_evaluations: int | None = Field(default=None, ge=1)
    """Stop after this many evaluations, even partway through a generation."""
    maximize: bool = False
    """Maximize the objective instead of minimizing it."""
    threshold: float | None = None
    """Stop once the best fitness reaches this."""
    seed: int | None = None
    """Fixes the random number generator, so runs can be repeated."""
    noise: bool = False
    """The objective is noisy: evaluating the same point twice can give different values."""
    final_evaluations: int = Field(default=0, ge=0)
    """With noise, evaluations kept back to re-evaluate the most promising points at the end, so
    the result is chosen, and its value reported, from averages rather than single lucky values."""
    min_finished: float = Field(default=1.0, gt=0, le=1)
    """The share of each generation's evaluations that must finish before the next generation is
    bred. Below 1, the slowest evaluations don't hold up the run: they finish in the background,
    and their results join the generation under way when they arrive. Results then depend on
    timing, so runs are no longer exactly repeatable."""

    @model_validator(mode="after")
    def _check_final_evaluations(self):
        if self.final_evaluations and self.max_evaluations is not None and self.final_evaluations >= self.max_evaluations:
            raise ValueError(
                f"final_evaluations {self.final_evaluations} leaves nothing of max_evaluations {self.max_evaluations} for the search"
            )
        return self

    @property
    def reserved_evaluations(self) -> int:
        """Evaluations kept back for the final re-evaluation: none without noise."""
        return self.final_evaluations if self.noise else 0

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        """Create a config from a JSON or YAML file."""
        filepath = Path(filepath)
        text = filepath.read_text()
        if filepath.suffix.lower() in {".yaml", ".yml"}:
            return cls.model_validate(yaml.safe_load(text) or {})
        return cls.model_validate_json(text)
