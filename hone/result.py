from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Self

from pydantic import BaseModel, ConfigDict

from hone.parameters import SearchSpace

if TYPE_CHECKING:
    from hone.optimizers import Optimizer


class _Unpickles(BaseModel):
    """A model that a checkpoint from an earlier version of hone can unpickle: fields added since take their defaults."""

    def __setstate__(self, state: Any) -> None:
        fields = state.get("__dict__", {})
        for name, field in type(self).model_fields.items():
            if name not in fields and not field.is_required():
                fields[name] = field.get_default(call_default_factory=True)
        super().__setstate__(state)


class GenerationStats(_Unpickles):
    """Summary of one generation's fitness, over its successful evaluations that satisfy any constraints on outputs.

    The fitness fields are None when every evaluation in the generation failed or broke an
    output constraint.
    """
    model_config = ConfigDict(frozen=True)

    generation: int
    evaluations: int  # evaluations up to the end of this generation, failed ones included
    failures: int = 0  # failed evaluations in this generation
    infeasible: int = 0  # evaluations in this generation that broke a constraint on the outputs
    best_fitness: float | None = None
    best_params: dict[str, Any] | None = None
    worst_fitness: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class RunInfo(BaseModel):
    """How a result was produced: enough to repeat the run, and to tell runs apart later."""
    model_config = ConfigDict(frozen=True)

    optimizer: str  # the optimizer's class name, such as "CMAES"
    config: dict[str, Any]  # the optimizer's full config, seed included
    search_space: SearchSpace
    initial_points: list[dict[str, Any]] = []
    objective: str | None = None  # the objective function's name; None when run with ask() and tell()
    workers: int | None = None  # optimize()'s workers, which CMA-ES and Bayesian optimization size their batches by
    timeout: float | None = None  # optimize()'s time limit per evaluation, in seconds
    hone_version: str
    python_version: str
    numpy_version: str
    started_at: datetime | None = None  # when the first candidate was handed out, in UTC
    elapsed: float | None = None  # seconds from then until the run finished, or until the result was taken


class Progress(BaseModel):
    """What optimize() passes its callback after each generation."""
    model_config = ConfigDict(frozen=True)

    generation: GenerationStats  # the generation just finished
    evaluations: int  # evaluations made so far, failed ones included
    best_fitness: float | None  # the best value so far; None while every evaluation has failed
    best_params: dict[str, Any] | None
    elapsed: float  # seconds since the run started
    feasible: bool = True  # whether best_params satisfies every constraint on the outputs


def print_progress(progress: Any) -> None:
    """A ready-made optimize() callback: print one line per generation.

    For several objectives, the line gives the size of the front and its hypervolume.
    """
    if isinstance(progress, Progress):
        found = "best none yet" if progress.best_fitness is None else f"best {progress.best_fitness:.6g}"
        if progress.best_fitness is not None and not progress.feasible:
            found += " (breaks an output constraint)"
    else:
        stats = progress.generation
        volume = "" if stats.hypervolume is None else f", hypervolume {stats.hypervolume:.6g}"
        found = f"front of {stats.front_size}{volume}"
    print(
        f"generation {progress.generation.generation}: {progress.evaluations:,} evaluations, {found}, {progress.elapsed:.1f} s",
        flush=True,
    )


class Result(BaseModel):
    """Outcome of an optimization run. Save it with save() and read it back with load().

    With constraints on outputs, best_params is the best point that satisfies them all. If no
    evaluation did, it is the point that came closest, by total violation, and feasible is False.

    Two results are equal when they found the same things; their run records are not compared,
    since a repeated run differs in its timing.
    """
    model_config = ConfigDict(frozen=True)

    best_fitness: float
    best_params: dict[str, Any]
    best_outputs: dict[str, float] | None = None  # the outputs at best_params, with constraints on outputs
    feasible: bool = True  # whether best_params satisfies every constraint on the outputs; False only if no evaluation did
    generations: list[GenerationStats]  # completed generations; empty partway through the first
    evaluations: int  # evaluations made, failed ones included
    failures: int = 0  # evaluations that failed
    infeasible: int = 0  # evaluations that broke a constraint on the outputs
    reached_threshold: bool  # whether the run stopped early because it reached config.threshold
    stopped: bool = False  # whether the run was stopped before it finished, by its callback or Ctrl+C
    run: RunInfo | None = None  # how the result was produced; None in results saved before hone 0.7.1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Result):
            return NotImplemented
        return self.model_dump(exclude={"run"}) == other.model_dump(exclude={"run"})

    def rebuild_optimizer(
        self, optimizer_class: "type[Optimizer] | None" = None, *, functions: dict[str, Callable[..., float]] | None = None
    ) -> "Optimizer":
        """A new optimizer set up exactly as this result's was, to repeat or extend the run.

        Running it gives the same result, given the same objective and, for CMA-ES and
        Bayesian optimization, the same number of workers (run.workers). An optimizer that
        isn't one of hone's own is rebuilt by passing its class.

        A constraint made with Constraint.function isn't saved with a result: after loading
        one, pass each such constraint's function by name, as functions={"stress": stress}.
        """
        from hone import optimizers

        if self.run is None:
            raise ValueError("this result has no record of its run: it was saved by a version of hone before 0.7.1")
        if optimizer_class is None:
            builtin = {cls.__name__: cls for cls in (
                optimizers.GeneticOptimizer, optimizers.CMAES, optimizers.BayesianOptimizer, optimizers.RandomSearch
            )}
            if self.run.optimizer not in builtin:
                raise ValueError(f"{self.run.optimizer} isn't one of hone's optimizers; pass its class to rebuild it")
            optimizer_class = builtin[self.run.optimizer]
        config = optimizer_class.config_class.model_validate(self.run.config)
        return optimizer_class(self.run.search_space.with_functions(functions or {}), config, initial_points=self.run.initial_points or None)

    def save(self, filepath: str | Path) -> None:
        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        return cls.model_validate_json(Path(filepath).read_text())
