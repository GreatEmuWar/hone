"""The machinery every optimizer shares, whatever it optimizes: the generation loop, ask and tell, workers, checkpoints and run records.

OptimizerCore knows nothing of how evaluations are compared. A subclass decides what a value
is, how it is recorded, when a run has reached its goal, and what the result is: Optimizer
does it for a single objective.
"""
import math
import os
import pickle
import platform
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from functools import cache, partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, ClassVar, List

import numpy as np
from pydantic import BaseModel, ConfigDict

from hone.config import OptimizerConfig
from hone.objective_function import ObjectiveFunction
from hone.parameters import Parameter, SearchSpace
from hone.result import RunInfo, _Unpickles

from ._pool import EvaluationPool


OBJECTIVE = "objective"  # the key of the objective's value in the dict it returns when there are output constraints


class UnsupportedSearchSpace(ValueError):
    """The optimizer cannot search this kind of parameter."""


class EvaluationFailed(Exception):
    """Raise this from an objective function to record the evaluation as failed rather than stop the run."""


class Trial(BaseModel):
    """A candidate handed out by ask(): evaluate params, then report the outcome with tell()."""
    model_config = ConfigDict(frozen=True)

    id: int
    params: dict[str, Any]
    generation: int


class Individual(_Unpickles):
    """One candidate solution: a value for every parameter, and its fitness once evaluated.

    A single-objective optimizer sets fitness; a multi-objective one sets values, by objective.
    With constraints on outputs, outputs holds what the evaluation reported and violation the
    total relative violation of those constraints: 0 when it satisfies them all.
    """
    params: dict[str, Any]
    generation: int
    fitness: float | None = None
    values: dict[str, float] | None = None
    outputs: dict[str, float] | None = None
    violation: float = 0.0
    failed: bool = False
    id: int = -1

    @property
    def evaluated(self) -> bool:
        """Whether the individual has been evaluated, successfully or not."""
        return self.fitness is not None or self.values is not None or self.failed

    # Each individual is a candidate of its own, even when another holds the same values:
    # optimizers keep track of them by identity.
    def __eq__(self, other: object) -> bool:
        return self is other

    __hash__ = object.__hash__


class OptimizerCore[ResultT](ABC):
    """The generation loop, ask and tell, workers, checkpoints and run records, for any kind of objective.

    A subclass decides what an objective's value is and what the result of a run is, through
    the methods marked as hooks below, and how each generation is produced, through
    _next_population.

    Args:
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: The optimizer's configuration. Defaults are used if omitted.
        initial_points: Points to evaluate first, such as a known good design or the best
            parameters of an earlier run: one dict of parameter values, or a list of them.
            Fixed parameters may be left out, and so must parameters a condition switches off.

    Every individual an optimizer breeds holds a value for every parameter, including those
    the conditions switch off at its point, so an optimizer can keep them to use again. The
    core leaves them out of what it hands out and reports: trials, results and progress.
    """
    config_class: ClassVar[type[OptimizerConfig]] = OptimizerConfig
    supports_constraints: ClassVar[bool] = False  # whether it keeps to a search space's constraints
    supports_conditions: ClassVar[bool] = False  # whether it handles parameters that conditions switch on and off

    def __init__(
        self,
        search_space: SearchSpace | List[Parameter],
        config: OptimizerConfig | None = None,
        *,
        initial_points: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(search_space, SearchSpace):
            search_space = SearchSpace(search_space)
        self.search_space = search_space
        self.config = config if config is not None else self.config_class()
        if not isinstance(self.config, self.config_class):
            raise TypeError(f"{type(self).__name__} needs a {self.config_class.__name__}, got {type(self.config).__name__}")
        if search_space.constraints and not self.supports_constraints:
            raise UnsupportedSearchSpace(f"{type(self).__name__} doesn't support constraints yet; RandomSearch does")
        if search_space.conditions and not self.supports_conditions:
            raise UnsupportedSearchSpace(f"{type(self).__name__} doesn't support conditional parameters yet; RandomSearch does")
        if search_space.output_constraints and self.config.noise:
            raise UnsupportedSearchSpace(
                "constraints on outputs aren't supported with noise yet: whether a noisy point satisfies a limit would need "
                "the same averaging as its value. Turn noise off, or penalize the limit in the objective"
            )
        if isinstance(initial_points, dict):
            initial_points = [initial_points]
        self.initial_points = [search_space.check(point) for point in initial_points or []]
        for point in self.initial_points:
            if broken := search_space.broken(point):
                raise ValueError(f"the initial point {point} breaks the constraint `{broken[0]}`")
        if search_space.parameter_constraints and not self.initial_points:
            # Fail now, before any evaluation, if the constraints leave nothing to search.
            search_space.sample(np.random.default_rng(0))
        self._workers = 1  # how many evaluations run at once; optimizers may size their population from it
        self.reset()

    @abstractmethod
    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        """The individuals of the given generation, or of the first one when previous is None.

        Individuals already evaluated, such as elites carried over, are not evaluated again.
        Failed individuals in previous have failed set and the worst possible fitness.
        """

    # --- hooks: what a value is, and what a run produces ---------------------

    @abstractmethod
    def _check_outcome(self, value: Any, failed: bool) -> bool:
        """Whether a told outcome counts as failed, raising for a value that isn't valid."""

    @abstractmethod
    def _record_failure(self, individual: Individual) -> None:
        """Record that the individual's evaluation failed. Its failed flag is already set."""

    @abstractmethod
    def _record(self, individual: Individual, value: Any) -> None:
        """Record the individual's successful value."""

    @abstractmethod
    def _summarize(self, generation: int, evaluations: int, population: list[Individual]) -> Any:
        """The summary of a finished generation, kept in the history."""

    @abstractmethod
    def result(self) -> ResultT:
        """The result so far."""

    @abstractmethod
    def _report(self, callback: Callable[[Any], bool | None] | None, reported: int) -> int:
        """Pass callback each generation finished since the first reported ones, and stop the run if it asks.

        Returns the number of generations reported.
        """

    @abstractmethod
    def _has_result(self) -> bool:
        """Whether there is a result to return yet: at least one successful evaluation."""

    def _goal_reached(self) -> bool:
        """Whether the run can stop early, such as at a threshold. Never, unless a subclass says."""
        return False

    def _begin_final_round(self) -> bool:
        """Start a final round of evaluations after the search, if the subclass has one. Returns whether it did."""
        return False

    def _kept_back(self) -> int:
        """Evaluations of the budget kept back from the search, such as for a final round."""
        return 0

    def _make_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        """The next generation's individuals: the search's, unless a subclass has a round of its own."""
        return self._next_population(previous, generation)

    def _objective(self, objective_function: Callable) -> Callable[[dict[str, Any]], Any]:
        """The objective as optimize() calls it, checked against the search space."""
        return ObjectiveFunction(
            name=getattr(objective_function, "__name__", "objective"), function=objective_function, search_space=self.search_space
        )

    # --- ask and tell ------------------------------------------------------

    def reset(self) -> None:
        """Start over: forget every evaluation and reseed the random number generator."""
        self.rng = np.random.default_rng(self.config.seed)
        self._generation = -1
        self._population: list[Individual] | None = None  # the current generation, or the last one finished
        self._needs_generation = True
        self._waiting: dict[int, Individual] = {}  # created this generation, not yet told
        self._late: dict[int, Individual] = {}  # handed out in an earlier generation, still being evaluated
        self._told_this_generation = 0
        self._quorum = 0  # evaluations of the current generation needed before the next is bred
        self._handed_out: set[int] = set()
        self._handed_out_total = 0  # trials handed out over the whole run
        self._next_id = 0
        self._evaluations = 0
        self._failures = 0
        self._history: list[Any] = []
        self._reached_threshold = False
        self._done = False
        # For the result's run record: set by optimize(), and timed from the first ask().
        self._objective_name: str | None = None
        self._run_workers: int | None = None
        self._run_timeout: float | None = None
        self._started_at: datetime | None = None
        self._start_clock: float | None = None
        self._end_clock: float | None = None
        self._stopped = False
        self._reissue: list[int] = []  # trials handed out before a checkpoint was loaded, to hand out again

    @property
    def done(self) -> bool:
        """Whether the run has finished: every generation is complete, or the threshold was reached.

        With min_finished below 1, the run isn't done until every late result has been told.
        """
        return self._done and not self._late

    def ask(self, n: int | None = None) -> list[Trial]:
        """Hand out up to n candidates to evaluate, or all that are ready when n is None.

        Returns an empty list when the run is done, or when every candidate of the current
        generation has been handed out but not all have been told: the next generation
        depends on their results.
        """
        if n is not None and n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if self._start_clock is None:
            self._started_at, self._start_clock = datetime.now(UTC), time.monotonic()
        while self._needs_generation and not self._done:
            self._start_generation()
        # Trials handed out before a checkpoint was loaded come first, stragglers included;
        # the budget already counts them.
        outstanding = {**self._late, **self._waiting}
        reissued = [outstanding[i] for i in self._reissue if i in outstanding][:n]
        self._reissue = [i for i in self._reissue if i in outstanding and i not in {ind.id for ind in reissued}]
        if self._done:
            return [self._trial(ind) for ind in reissued]
        ready = [ind for ind_id, ind in self._waiting.items() if ind_id not in self._handed_out]
        limits = [None if n is None else n - len(reissued), self._budget_left()]
        ready = ready[: min((limit for limit in limits if limit is not None), default=len(ready))]
        self._handed_out.update(ind.id for ind in ready)
        self._handed_out_total += len(ready)
        return [self._trial(ind) for ind in reissued + ready]

    def _trial(self, individual: Individual) -> Trial:
        return Trial(id=individual.id, params=self.search_space.active(individual.params), generation=individual.generation)

    def _key(self, params: dict[str, Any]) -> tuple:
        """A hashable form of the point params are at: two that differ only where conditions switch parameters off share it."""
        return _key(self.search_space.active(params))

    def tell(self, trial: Trial | int, value: Any = None, *, failed: bool = False) -> None:
        """Report the outcome of a trial from ask(), by the trial or its id.

        Pass the objective's value, or failed=True when the evaluation could not produce one
        (a crashed simulation, say). A value of NaN or infinity also counts as failed. Failed
        trials count as evaluations, rank below every successful one, and are never used as
        parents.
        """
        trial_id = trial.id if isinstance(trial, Trial) else trial
        late = trial_id in self._late
        individual = self._late.get(trial_id) if late else self._waiting.get(trial_id)
        if individual is None or not (late or trial_id in self._handed_out):
            raise ValueError(f"trial {trial_id} is not waiting for a result; it was never asked for or was already told")
        failed = self._check_outcome(value, failed)

        del (self._late if late else self._waiting)[trial_id]
        self._evaluations += 1
        if failed:
            self._failures += 1
            individual.failed = True
            self._record_failure(individual)
        else:
            self._record(individual, value)

        if late:
            # A straggler from an earlier generation joins the one under way, to breed from.
            if self._population is not None and not self._done:
                self._population.append(individual)
            return
        self._told_this_generation += 1
        if not self._waiting:
            self._finish_generation()
        elif self._handed_out.issuperset(self._waiting) and self._told_this_generation >= self._quorum:
            # Enough of the generation has finished: the rest go on in the background.
            self._late.update(self._waiting)
            self._population = [ind for ind in self._population or [] if ind.evaluated]
            self._waiting = {}
            self._finish_generation()
        elif self._budget_left() == 0 and self._handed_out.isdisjoint(self._waiting):
            # Out of budget, with every trial handed out so far reported: end on a partial generation.
            self._finish_generation(final=True)

    def _budget_left(self) -> int | None:
        if self.config.max_evaluations is None:
            return None
        return max(self.config.max_evaluations - self._kept_back() - self._handed_out_total, 0)

    def _run_info(self) -> RunInfo:
        return RunInfo(
            optimizer=type(self).__name__,
            config=self.config.model_dump(mode="json"),
            search_space=self.search_space,
            initial_points=[dict(point) for point in self.initial_points],
            objective=self._objective_name,
            workers=self._run_workers,
            timeout=self._run_timeout,
            hone_version=_hone_version(),
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            started_at=self._started_at,
            elapsed=self._elapsed() if self._start_clock is not None else None,
        )

    def _elapsed(self) -> float:
        assert self._start_clock is not None
        return (self._end_clock if self._end_clock is not None else time.monotonic()) - self._start_clock

    def _finish(self, stopped: bool = False) -> None:
        self._done = True
        self._stopped = stopped
        self._end_clock = time.monotonic()

    # --- checkpoints ------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the run's whole state to path, so load_checkpoint() can carry it on later, in another process.

        The file is replaced in one step, so a crash while saving leaves the previous checkpoint intact.
        """
        # The search space is left out: the loading optimizer has its own, which must match, and a
        # constraint function may not be picklable.
        state = {name: value for name, value in self.__dict__.items() if name != "search_space"}
        elapsed = self._elapsed() if self._start_clock is not None else None
        state.update(_start_clock=None, _end_clock=None, _workers=1)
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "hone_version": _hone_version(),
            "setup": self._setup(),
            "elapsed": elapsed,
            "state": state,
        }
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temporary, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Carry on the run saved in path by save_checkpoint(), from where it was saved.

        The checkpoint must come from an optimizer set up like this one: the same class, config,
        search space and initial points. Trials handed out but not yet told when it was saved
        are handed out again by ask(), and can still be told by their ids.

        A checkpoint is a pickle, which can run code when it is loaded: only load checkpoints
        you saved yourself.
        """
        checkpoint = pickle.loads(Path(path).read_bytes())
        if not isinstance(checkpoint, dict) or checkpoint.get("format") not in READABLE_FORMATS:
            raise ValueError(f"{path} is not a checkpoint this version of hone can read")
        if checkpoint["setup"] != self._setup():
            raise ValueError(
                f"{path} was saved by an optimizer set up differently: a checkpoint needs the same "
                "optimizer, config, search space and initial points"
            )
        workers = self._workers
        self.__dict__.update(checkpoint["state"])
        self._workers = workers
        elapsed = checkpoint["elapsed"]
        if elapsed is not None:
            self._start_clock = time.monotonic() - elapsed
            self._end_clock = self._start_clock + elapsed if self._done else None
        self._reissue = list(self._late) + [i for i in self._waiting if i in self._handed_out]

    def _setup(self) -> dict[str, Any]:
        """What a checkpoint must share with the optimizer that loads it."""
        return {
            "optimizer": type(self).__name__,
            "config": self.config.model_dump(mode="json"),
            "search_space": self.search_space.model_dump(mode="json"),
            "initial_points": [dict(point) for point in self.initial_points],
        }

    # --- running the objective for you ----------------------------------------

    def optimize(
        self,
        objective_function: Callable,
        *,
        workers: int = 1,
        timeout: float | None = None,
        callback: Callable[[Any], bool | None] | None = None,
        checkpoint: str | Path | None = None,
    ) -> ResultT:
        """Run the optimizer from the start, calling objective_function for every candidate.

        The function is called with one keyword argument per parameter. To record an
        evaluation as failed, raise EvaluationFailed or return NaN or infinity; any other
        exception stops the run. With a seed, every call with the same number of workers
        returns the same result. The genetic optimizer and random search return it however
        many workers are used; CMA-ES and Bayesian optimization do not, because their default
        population grows with the number of workers.

        Args:
            objective_function: The function to optimize.
            workers: Evaluate candidates in this many background processes at once.
            timeout: The longest, in seconds, one evaluation may take. An evaluation that runs
                longer is stopped and counts as failed, as does one whose process crashes.
            callback: Called with the progress after each generation. Return True from it to
                stop the run; optimize() then returns the result so far.
            checkpoint: A file to save the run's state to after each generation. If the file
                exists, the run carries on from it instead of starting over, so a run killed
                partway through picks up where it left off; one that finished returns its
                result at once. The result is the same as an uninterrupted run's.

        Pressing Ctrl+C also stops the run and returns the result so far, marked as stopped,
        once at least one evaluation has succeeded.

        With workers > 1 or a timeout, each evaluation runs in a separate process, so the
        function must be defined at the top level of a module.
        """
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        objective = self._objective(objective_function)
        name = getattr(objective, "name", getattr(objective_function, "__name__", "objective"))
        evaluate = partial(_evaluate, objective)
        self._workers = workers
        try:
            self.reset()
            self._objective_name, self._run_workers, self._run_timeout = name, workers, timeout
            if checkpoint is not None and Path(checkpoint).exists():
                self.load_checkpoint(checkpoint)
            reported = len(self._history)

            def after_generation() -> int:
                done = self._report(callback, reported)
                if checkpoint is not None:
                    self.save_checkpoint(checkpoint)
                return done

            try:
                if workers == 1 and timeout is None:
                    while not self._done:
                        for trial in self.ask():
                            self._tell_outcome(trial, evaluate(trial.params))
                        reported = after_generation()
                elif self.config.min_finished == 1:
                    with EvaluationPool(evaluate, workers, timeout) as pool:
                        while not self._done:
                            trials = self.ask()
                            for trial, outcome in zip(trials, pool.map([trial.params for trial in trials])):
                                self._tell_outcome(trial, outcome)
                            reported = after_generation()
                else:
                    with EvaluationPool(evaluate, workers, timeout) as pool:
                        running: dict[int, Trial] = {}
                        while not self.done:
                            # Keep every worker busy: start what the optimizer has ready.
                            if pool.idle:
                                for trial in self.ask(pool.idle):
                                    running[trial.id] = trial
                                    pool.submit(trial.id, trial.params)
                            if not running:
                                break  # nothing out, and nothing more to hand out
                            for trial_id, outcome in pool.collect():
                                self._tell_outcome(running.pop(trial_id), outcome)
                            if len(self._history) > reported:
                                reported = after_generation()
                            if self._done and (self._stopped or self._reached_threshold):
                                self._late = {}  # stopped early: the stragglers' results aren't needed
                                break
            except KeyboardInterrupt:
                if not self._has_result():
                    raise  # nothing to return
                self._finish(stopped=True)
        finally:
            self._workers = 1
        return self.result()

    def _tell_outcome(self, trial: Trial, outcome: Any) -> None:
        if outcome is None:
            self.tell(trial, failed=True)
        else:
            self.tell(trial, outcome)

    # --- the generation loop --------------------------------------------------

    def _start_generation(self) -> None:
        self._needs_generation = False
        self._generation += 1
        self._population = self._make_population(self._population, self._generation)
        self._waiting = {}
        self._handed_out = set()
        self._told_this_generation = 0
        for individual in self._population:
            if not individual.evaluated:
                individual.id = self._next_id
                self._next_id += 1
                self._waiting[individual.id] = individual
        self._quorum = math.ceil(self.config.min_finished * len(self._waiting) - 1e-9)
        if not self._waiting:
            self._finish_generation()

    def _finish_generation(self, final: bool = False) -> None:
        assert self._population is not None
        self._history.append(self._summarize(self._generation, self._evaluations, self._population))
        if self._goal_reached():
            self._reached_threshold = True
            self._finish()
        elif final or self._generation + 1 >= self.config.generations or self._budget_left() == 0:
            if not self._begin_final_round():
                self._finish()
        else:
            # The next generation is bred from this one when ask() is next called.
            self._needs_generation = True

    def _outstanding(self) -> list[Individual]:
        """Individuals of earlier generations still being evaluated."""
        return list(self._late.values())

    def _initial_individuals(self, generation: int) -> list[Individual]:
        """The initial points as individuals, with a random value for each parameter a condition switches off."""
        return [
            Individual(
                params={p.name: point[p.name] if p.name in point else p.sample(self.rng) for p in self.search_space.parameters},
                generation=generation,
            )
            for point in self.initial_points
        ]

    def _random_individual(self, generation: int) -> Individual:
        return Individual(params=self.search_space.sample(self.rng), generation=generation)


CHECKPOINT_FORMAT = 2  # raised when the checkpoint layout changes, so old files are refused clearly
READABLE_FORMATS = (1, 2)  # 1, from hone 0.13.0 and before, lacks each evaluation's outputs and violation, which default


@cache
def _hone_version() -> str:
    try:
        return version("hone-python")
    except PackageNotFoundError:  # run from a source checkout that isn't installed
        return "unknown"


def _key(params: dict[str, Any]) -> tuple:
    """A hashable form of params, so evaluations of the same point can be grouped."""
    return tuple((name, tuple(value) if isinstance(value, list) else value) for name, value in params.items())


def _evaluate(objective: Callable[[dict[str, Any]], Any], params: dict[str, Any]) -> Any:
    """Call the objective, returning None for a failed evaluation. Runs in worker processes too."""
    try:
        return objective(params)
    except EvaluationFailed:
        return None
