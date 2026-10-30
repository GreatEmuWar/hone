import math
import numbers
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, List

import numpy as np

from hone.config import OptimizerConfig
from hone.objective_function import ObjectiveFunction
from hone.parameters import Parameter, SearchSpace
from hone.result import GenerationStats, Result


class EvaluationFailed(Exception):
    """Raise this from an objective function to record the evaluation as failed rather than stop the run."""


@dataclass(frozen=True, slots=True)
class Trial:
    """A candidate handed out by ask(): evaluate params, then report the outcome with tell()."""
    id: int
    params: dict[str, Any]
    generation: int


@dataclass(slots=True, eq=False)
class Individual:
    """One candidate solution: a value for every parameter, and its fitness once evaluated."""
    params: dict[str, Any]
    generation: int
    fitness: float | None = None
    failed: bool = False
    id: int = -1


class Optimizer(ABC):
    """Base class for population-based optimizers.

    Use it either way round:

    - optimize(objective) calls the objective for every candidate and returns the Result.
    - ask() and tell() let you evaluate candidates yourself, anywhere and in any order;
      result() returns the Result so far.

    The base class runs the generation loop: it hands out each generation's candidates,
    records their outcomes, tracks the best so far, stops at the threshold, and builds the
    Result. A subclass only decides how each generation's population is produced, by
    implementing _next_population.

    Args:
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: The optimizer's configuration. Defaults are used if omitted.
    """
    config_class: ClassVar[type[OptimizerConfig]] = OptimizerConfig

    def __init__(self, search_space: SearchSpace | List[Parameter], config: OptimizerConfig | None = None) -> None:
        if not isinstance(search_space, SearchSpace):
            search_space = SearchSpace(search_space)
        self.search_space = search_space
        self.config = config if config is not None else self.config_class()
        if not isinstance(self.config, self.config_class):
            raise TypeError(f"{type(self).__name__} needs a {self.config_class.__name__}, got {type(self.config).__name__}")
        self.reset()

    @abstractmethod
    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        """The individuals of the given generation, or of the first one when previous is None.

        Individuals that already have a fitness, such as elites carried over, are not evaluated again.
        Failed individuals in previous have failed set and the worst possible fitness.
        """

    # --- ask and tell ------------------------------------------------------

    def reset(self) -> None:
        """Start over: forget every evaluation and reseed the random number generator."""
        self.rng = np.random.default_rng(self.config.seed)
        self._generation = -1
        self._population: list[Individual] | None = None  # the current generation, or the last one finished
        self._needs_generation = True
        self._waiting: dict[int, Individual] = {}  # created this generation, not yet told
        self._handed_out: set[int] = set()
        self._next_id = 0
        self._evaluations = 0
        self._failures = 0
        self._best: Individual | None = None
        self._history: list[GenerationStats] = []
        self._reached_threshold = False
        self._done = False

    @property
    def done(self) -> bool:
        """Whether the run has finished: every generation is complete, or the threshold was reached."""
        return self._done

    def ask(self, n: int | None = None) -> list[Trial]:
        """Hand out up to n candidates to evaluate, or all that are ready when n is None.

        Returns an empty list when the run is done, or when every candidate of the current
        generation has been handed out but not all have been told: the next generation
        depends on their results.
        """
        if n is not None and n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        while self._needs_generation and not self._done:
            self._start_generation()
        if self._done:
            return []
        ready = [ind for ind_id, ind in self._waiting.items() if ind_id not in self._handed_out]
        ready = ready if n is None else ready[:n]
        self._handed_out.update(ind.id for ind in ready)
        return [Trial(id=ind.id, params=dict(ind.params), generation=ind.generation) for ind in ready]

    def tell(self, trial: Trial | int, value: float | None = None, *, failed: bool = False) -> None:
        """Report the outcome of a trial from ask(), by the trial or its id.

        Pass the objective's value, or failed=True when the evaluation could not produce one
        (a crashed simulation, say). A value of NaN also counts as failed. Failed trials count
        as evaluations, rank below every successful one, and are never used as parents.
        """
        trial_id = trial.id if isinstance(trial, Trial) else trial
        individual = self._waiting.get(trial_id)
        if individual is None or trial_id not in self._handed_out:
            raise ValueError(f"trial {trial_id} is not waiting for a result; it was never asked for or was already told")
        if not failed:
            if value is None:
                raise ValueError("pass the trial's value, or failed=True")
            if not isinstance(value, numbers.Real):
                raise TypeError(f"the value must be a number, got {value!r}")
            failed = math.isnan(value)

        del self._waiting[trial_id]
        self._evaluations += 1
        if failed:
            self._failures += 1
            individual.failed = True
            individual.fitness = -math.inf if self.config.maximize else math.inf
        else:
            individual.fitness = float(value)
            if self._best is None or self._is_better(individual.fitness, self._best.fitness):
                self._best = individual

        if not self._waiting:
            self._finish_generation()

    def result(self) -> Result:
        """The best result found so far, with a summary of every completed generation."""
        if self._best is None:
            raise RuntimeError("no successful evaluations yet")
        return Result(
            best_fitness=self._best.fitness,
            best_params=dict(self._best.params),
            generations=self._history,
            evaluations=self._evaluations,
            failures=self._failures,
            reached_threshold=self._reached_threshold,
        )

    # --- running the objective for you ----------------------------------------

    def optimize(self, objective_function: Callable, *, workers: int = 1) -> Result:
        """Run the optimizer from the start, calling objective_function for every candidate.

        The function is called with one keyword argument per parameter and must return a
        number. To record an evaluation as failed instead, raise EvaluationFailed or return NaN;
        any other exception stops the run. With workers > 1, candidates are evaluated in that
        many background processes, so the function must be defined at the top level of a
        module. With a seed, every call returns the same result, however many workers are used.
        """
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        objective = ObjectiveFunction(
            name=getattr(objective_function, "__name__", "objective"), function=objective_function, search_space=self.search_space
        )
        self.reset()
        if workers == 1:
            while not self._done:
                for trial in self.ask():
                    self._tell_outcome(trial, _evaluate(objective, trial.params))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                while not self._done:
                    trials = self.ask()
                    futures = [pool.submit(_evaluate, objective, trial.params) for trial in trials]
                    for trial, future in zip(trials, futures):
                        self._tell_outcome(trial, future.result())
        return self.result()

    def _tell_outcome(self, trial: Trial, outcome: float | None) -> None:
        if outcome is None:
            self.tell(trial, failed=True)
        else:
            self.tell(trial, outcome)

    # --- the generation loop --------------------------------------------------

    def _start_generation(self) -> None:
        self._needs_generation = False
        self._generation += 1
        self._population = self._next_population(self._population, self._generation)
        self._waiting = {}
        self._handed_out = set()
        for individual in self._population:
            if individual.fitness is None:
                individual.id = self._next_id
                self._next_id += 1
                self._waiting[individual.id] = individual
        if not self._waiting:
            self._finish_generation()

    def _finish_generation(self) -> None:
        self._history.append(self._summarize(self._generation, self._evaluations, self._population))
        if self._best is not None and self._reached(self._best.fitness):
            self._reached_threshold = True
            self._done = True
        elif self._generation + 1 >= self.config.generations:
            self._done = True
        else:
            # The next generation is bred from this one when ask() is next called.
            self._needs_generation = True

    def _random_individual(self, generation: int) -> Individual:
        return Individual(params=self.search_space.sample(self.rng), generation=generation)

    def _is_better(self, fitness: float, other: float) -> bool:
        """Whether fitness beats other in the configured direction."""
        return fitness > other if self.config.maximize else fitness < other

    def _ranked(self, individuals: list[Individual]) -> list[Individual]:
        """The individuals, best first. Failed individuals sort last."""
        return sorted(individuals, key=lambda x: x.fitness, reverse=self.config.maximize)

    def _reached(self, fitness: float) -> bool:
        """Whether fitness is good enough to stop early. Never true without a threshold."""
        if self.config.threshold is None:
            return False
        return fitness >= self.config.threshold if self.config.maximize else fitness <= self.config.threshold

    def _summarize(self, generation: int, evaluations: int, population: list[Individual]) -> GenerationStats:
        succeeded = self._ranked([ind for ind in population if not ind.failed])
        failures = len(population) - len(succeeded)
        if not succeeded:
            return GenerationStats(generation=generation, evaluations=evaluations, failures=failures)
        fitnesses = [ind.fitness for ind in succeeded]
        return GenerationStats(
            generation=generation,
            evaluations=evaluations,
            failures=failures,
            best_fitness=succeeded[0].fitness,
            best_params=dict(succeeded[0].params),
            worst_fitness=succeeded[-1].fitness,
            mean=float(np.mean(fitnesses)),
            median=float(np.median(fitnesses)),
            std=float(np.std(fitnesses)),
        )


def _evaluate(objective: ObjectiveFunction, params: dict[str, Any]) -> float | None:
    """Call the objective, returning None for a failed evaluation. Runs in worker processes too."""
    try:
        return objective(params)
    except EvaluationFailed:
        return None
