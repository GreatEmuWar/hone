from functools import partial

from hone.config import GeneticOptimizerConfig, SelectionMethod
from hone.optimizers.base import Individual, Optimizer

from . import selection
from .variation import mutate, uniform_crossover


class GeneticOptimizer(Optimizer):
    """Function optimizer that employs an implementation of the genetic algorithm.
    The genetic algorithm is a metaheuristic inspired by the process of natural selection.
    It is a population-based algorithm that uses the concepts of selection, crossover, and mutation
    to evolve a population of individuals towards a solution to the objective function.

    Each generation keeps the previous one's elites unchanged, then fills the rest of the
    population with children: two parents are selected, crossed over, and mutated.

    Args:
        objective_function: The objective function to be optimized. It is called with one
            keyword argument per parameter in the search space and must return a number.
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: A GeneticOptimizerConfig. Defaults are used if omitted.

    Example:

        >>> from hone import GeneticOptimizer, GeneticOptimizerConfig, RangeParameter
        >>>
        >>> def objective_function(x, y):
        >>>     return x**2 + y**2
        >>>
        >>> x = RangeParameter("x", -5, 5)
        >>> y = RangeParameter("y", -5, 5)
        >>>
        >>> optimizer = GeneticOptimizer(
        >>>     objective_function=objective_function,
        >>>     search_space=[x, y],
        >>>     config=GeneticOptimizerConfig(threshold=0.01),
        >>> )
        >>>
        >>> result = optimizer.optimize()
    """
    config_class = GeneticOptimizerConfig
    config: GeneticOptimizerConfig

    def _select(self, individuals: list[Individual]) -> Individual:
        match self.config.selection_method:
            case SelectionMethod.TOURNAMENT:
                select = partial(selection.tournament, size=self.config.tournament_size, p=self.config.tournament_p)
            case SelectionMethod.ROULETTE:
                select = selection.roulette
        return select(individuals, self.rng, maximize=self.config.maximize)

    def _child(self, parents: list[Individual], generation: int) -> Individual:
        first, second = self._select(parents), self._select(parents)
        params = uniform_crossover(first.params, second.params, self.rng, rate=self.config.crossover_rate)
        params = mutate(
            params, self.search_space, self.rng, rate=self.config.mutation_rate, scale=self.config.mutation_scale
        )
        return Individual(params=params, generation=generation)

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        size = self.config.population_size
        if previous is None:
            return [self._random_individual(generation) for _ in range(size)]

        elites = self._ranked(previous)[: round(size * self.config.elitism_rate)] if self.config.elitism else []
        return elites + [self._child(previous, generation) for _ in range(size - len(elites))]
