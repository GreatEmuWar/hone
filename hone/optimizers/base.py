from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, List

import numpy as np

from hone.config import OptimizerConfig
from hone.objective_function import ObjectiveFunction
from hone.parameters import Parameter, SearchSpace
from hone.result import GenerationStats, Result


@dataclass(slots=True, eq=False)
class Individual:
    """One candidate solution: a value for every parameter, and its fitness once evaluated."""
    params: dict[str, Any]
    generation: int
    fitness: float | None = None


class Optimizer(ABC):
    """Base class for population-based optimizers.

    It runs the generation loop: it evaluates each population, tracks the best individual
    so far, stops at the threshold, and builds the Result. A subclass only decides how
    each generation's population is produced, by implementing _next_population.

    Args:
        objective_function: The objective function to be optimized. It is called with one
            keyword argument per parameter in the search space and must return a number.
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: The optimizer's configuration. Defaults are used if omitted.
    """
    config_class: ClassVar[type[OptimizerConfig]] = OptimizerConfig

    def __init__(
        self,
        objective_function: Callable,
        search_space: SearchSpace | List[Parameter],
        config: OptimizerConfig | None = None,
    ) -> None:
        if not isinstance(search_space, SearchSpace):
            search_space = SearchSpace(search_space)
        self.objective_function = ObjectiveFunction(
            name=objective_function.__name__,
            function=objective_function,
            search_space=search_space,
        )
        self.config = config if config is not None else self.config_class()
        if not isinstance(self.config, self.config_class):
            raise TypeError(f"{type(self).__name__} needs a {self.config_class.__name__}, got {type(self.config).__name__}")
        self.rng = np.random.default_rng(self.config.seed)

    @property
    def search_space(self) -> SearchSpace:
        return self.objective_function.search_space

    @abstractmethod
    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        """The individuals of the given generation, or of the first one when previous is None.

        Individuals that already have a fitness, such as elites carried over, are not evaluated again.
        """

    def optimize(self) -> Result:
        """Run the optimizer for config.generations generations, or until the best fitness
        reaches config.threshold if one is set. With a seed, every call returns the same result.
        """
        self.rng = np.random.default_rng(self.config.seed)
        evaluations = 0
        history: list[GenerationStats] = []
        population: list[Individual] | None = None
        best: Individual | None = None
        reached_threshold = False

        for generation in range(self.config.generations):
            population = self._next_population(population, generation)
            for individual in population:
                if individual.fitness is None:
                    individual.fitness = self.objective_function(individual.params)
                    evaluations += 1

            ranked = self._ranked(population)
            history.append(self._summarize(generation, evaluations, ranked))
            if best is None or self._is_better(ranked[0].fitness, best.fitness):
                best = ranked[0]

            if self._reached_threshold(best.fitness):
                reached_threshold = True
                break

        return Result(
            best_fitness=best.fitness,
            best_params=dict(best.params),
            generations=history,
            evaluations=evaluations,
            reached_threshold=reached_threshold,
        )

    def _random_individual(self, generation: int) -> Individual:
        return Individual(params=self.search_space.sample(self.rng), generation=generation)

    def _is_better(self, fitness: float, other: float) -> bool:
        """Whether fitness beats other in the configured direction."""
        return fitness > other if self.config.maximize else fitness < other

    def _ranked(self, individuals: list[Individual]) -> list[Individual]:
        """The individuals, best first."""
        return sorted(individuals, key=lambda x: x.fitness, reverse=self.config.maximize)

    def _reached_threshold(self, fitness: float) -> bool:
        """Whether fitness is good enough to stop early. Never true without a threshold."""
        if self.config.threshold is None:
            return False
        return fitness >= self.config.threshold if self.config.maximize else fitness <= self.config.threshold

    @staticmethod
    def _summarize(generation: int, evaluations: int, ranked: list[Individual]) -> GenerationStats:
        fitnesses = [individual.fitness for individual in ranked]
        return GenerationStats(
            generation=generation,
            evaluations=evaluations,
            best_fitness=ranked[0].fitness,
            best_params=dict(ranked[0].params),
            worst_fitness=ranked[-1].fitness,
            mean=float(np.mean(fitnesses)),
            median=float(np.median(fitnesses)),
            std=float(np.std(fitnesses)),
        )
