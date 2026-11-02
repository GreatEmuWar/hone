from functools import partial
from typing import Callable

from hone.config import GeneticOptimizerConfig, SelectionMethod
from hone.optimizers.base import Individual, Optimizer

from . import selection
from .breeding import CONDITIONAL_NICHE_SHARE, MAX_STEP, MIN_STEP, REPAIR_STEPS, REPEAT_TRIES, STEP_SPREAD, Breeding

__all__ = ["CONDITIONAL_NICHE_SHARE", "MAX_STEP", "MIN_STEP", "REPAIR_STEPS", "REPEAT_TRIES", "STEP_SPREAD", "GeneticOptimizer"]


class GeneticOptimizer(Optimizer, Breeding):
    """Function optimizer that employs an implementation of the genetic algorithm.
    The genetic algorithm is a metaheuristic inspired by the process of natural selection.
    It is a population-based algorithm that uses the concepts of selection, crossover, and mutation
    to evolve a population of individuals towards a solution to the objective function.

    The first generation is any initial points, then random candidates. Each later generation
    keeps the previous one's elites unchanged, then fills the rest of the population with
    children: two parents are selected, crossed over, and mutated. Failed evaluations are
    never elites or parents.

    With blend crossover, each range value of a child is drawn from between its parents' values,
    and a little beyond, rather than copied from one of them; choices and fixed values are
    still copied.

    With adaptive mutation, each individual carries its own mutation size, as in evolution
    strategies: a child takes the geometric mean of its parents' sizes, varied at random, so
    individuals with sizes that suit the problem's current scale breed more.

    Without noise, a child that repeats a point already asked for, such as a copy of a parent
    that no mutation touched, has one gene at a time mutated again until it is new, since
    evaluating it again would only spend the budget. With noise, repeats stay: they refine
    the point's average.

    With conditional parameters, every individual keeps a value for each parameter, even one
    switched off at its point, so a child that switches a branch back on inherits settings
    that once did well there. A gene switched off in one parent but on in the other comes
    from the parent it is on in, and only genes switched on are mutated.

    Args:
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
        >>> optimizer = GeneticOptimizer(search_space=[x, y], config=GeneticOptimizerConfig(threshold=0.01))
        >>> result = optimizer.optimize(objective_function)
    """
    config_class = GeneticOptimizerConfig
    supports_constraints = True
    supports_conditions = True
    config: GeneticOptimizerConfig

    def reset(self) -> None:
        super().reset()
        self._setup_breeding()

    def _select(self, individuals: list[Individual]) -> Individual:
        select: Callable[..., Individual]
        match self.config.selection_method:
            case SelectionMethod.TOURNAMENT:
                select = partial(selection.tournament, size=self.config.tournament_size, p=self.config.tournament_p)
            case SelectionMethod.ROULETTE:
                select = selection.roulette
        return select(individuals, self.rng, maximize=self.config.maximize)

    def _fitter(self, first: Individual, second: Individual) -> Individual:
        first_rank, second_rank = self._rank(first), self._rank(second)
        if first_rank != second_rank:
            return first if first_rank < second_rank else second
        return second if self.config.maximize else first  # a tie, broken as it always has been

    def _ranked_niches(self, niches: list[list[Individual]]) -> list[list[Individual]]:
        return sorted(niches, key=lambda members: self._rank(self._ranked(members)[0]))

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        size = self.config.population_size
        if previous is None:
            initial = self._initial_individuals(generation)
            population = initial + [self._random_individual(generation) for _ in range(size - len(initial))]
            self._asked.update(self._key(individual.params) for individual in population)
            return population
        parents = [individual for individual in previous if not individual.failed]
        if not parents:
            # Every evaluation of the previous generation failed: start afresh.
            population = [self._random_individual(generation) for _ in range(size)]
            self._asked.update(self._key(individual.params) for individual in population)
            return population

        if self.config.noise:
            # A point bred again since it was evaluated has more evaluations now; rank by their average.
            self._refresh_averages(parents)
        # At least one child per generation, however the share rounds, so the search keeps moving.
        elites = self._ranked(parents)[: min(round(size * self.config.elitism_rate), size - 1)] if self.config.elitism else []
        niched = self._niche_children(parents, int(self._niche_share * (size - len(elites))), generation)
        population = elites + niched + [self._child(parents, generation) for _ in range(size - len(elites) - len(niched))]
        # Keep the sizes of individuals that can still breed: this generation, and any still being evaluated.
        self._keep_steps(set(population) | set(self._outstanding()))
        return population
