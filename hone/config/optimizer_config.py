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
    final_evaluations: int | None = Field(default=None, ge=0)
    """With noise, evaluations kept back to re-evaluate the most promising points at the end, so
    the result is chosen, and its value reported, from averages rather than single lucky values.
    None: none, except with constraints on outputs, where a tenth of max_evaluations is kept
    back, at least 10 and at most half (10 without max_evaluations), so whether the result
    satisfies each limit rests on more than one noisy draw. 0 keeps none back either way."""
    confidence: float | None = Field(default=0.95, gt=0, lt=1)
    """With noise and constraints on outputs, how sure a design must be to count as satisfying a
    limit: its average margin must clear the limit by Φ⁻¹(confidence) standard errors, so 0.95
    asks about 1.64 per limit. A design measured more often has a smaller standard error and may
    sit closer to the limit. The noise in each limit is estimated by pooling the spread of every
    design evaluated more than once, assuming it is about the same everywhere; until there are 3
    such degrees of freedom, a limit is judged on the average alone. None judges every limit on
    the average alone, as before 0.15.0. Without noise or constraints on outputs it does nothing."""
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
        """Evaluations final_evaluations keeps back for the final re-evaluation: none without noise.

        The optimizer adds the default for noise with constraints on outputs when final_evaluations is None.
        """
        return (self.final_evaluations or 0) if self.noise else 0

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        """Create a config from a JSON or YAML file."""
        filepath = Path(filepath)
        text = filepath.read_text()
        if filepath.suffix.lower() in {".yaml", ".yml"}:
            return cls.model_validate(yaml.safe_load(text) or {})
        return cls.model_validate_json(text)
