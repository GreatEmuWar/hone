from dataclasses import dataclass, field
from typing import Callable, Dict, List
from uuid import uuid4

import numpy as np

from hone.config import GeneticOptimizerConfig, SelectionMethod
from hone.objective_function import ObjectiveFunction
from hone.parameters import Parameter, SearchSpace

class Individual(object):
    def __init__(self, **params):
        if params:
            self.params = params
        else:
            self.params = {}
        self.fitness = None
        self.generation = None
        self.individual_id = str(uuid4())

@dataclass
class Population:
    size: int = 0
    individuals: List[Individual] = field(default_factory=list)
    population_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        self.size = len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]

    def __setitem__(self, index, value):
        self.individuals[index] = value

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.individuals)

    def __repr__(self):
        return f'Population(size={self.size})'

    def add_individual(self, individual: Individual):
        self.individuals.append(individual)
        self.size += 1

    def add_individuals(self, individuals: List[Individual]):
        self.individuals.extend(individuals)
        self.size += len(individuals)

@dataclass
class Generation:
    """Generation of the genetic algorithm."""
    generation_id: int
    population: Population = field(default=None)
    best: Individual = field(default=None)
    worst: Individual = field(default=None)
    mean: float = field(default=None)
    median: float = field(default=None)
    std: float = field(default=None)

@dataclass
class Result:
    """Result of the genetic algorithm."""
    generations: List[Generation]
    best: Individual
    best_fitness: float
    optimum: float
    optimum_params: Dict[str, float | str]


class GeneticOptimizer(object):
    """Function optimizer that employs an implementation of the genetic algorithm.
    The genetic algorithm is a metaheuristic inspired by the process of natural selection.
    It is a population-based algorithm that uses the concepts of selection, crossover, and mutation
    to evolve a population of individuals towards a solution to the objective function. Over enough
    generations, the population will converge to the global optimum of the objective function.

    Args:
        objective_function: The objective function to be optimized. It is called with one
            keyword argument per parameter in the search space.
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: A configuration object for the genetic optimizer. Defaults are used if omitted.

    Returns:
        A Result object containing the results of the genetic algorithm optimization.

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
    def __init__(
        self,
        objective_function: Callable,
        search_space: SearchSpace | List[Parameter],
        config: GeneticOptimizerConfig | None = None,
    ) -> None:
        """Initialize the genetic optimizer."""
        if not isinstance(search_space, SearchSpace):
            search_space = SearchSpace(search_space)
        self.objective_function = ObjectiveFunction(
            name=objective_function.__name__,
            function=objective_function,
            search_space=search_space,
        )
        self.config = config if config is not None else GeneticOptimizerConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.genealogy: List[Generation] = []

    def _random_parameters(self) -> Dict[str, float | int | str]:
        """Draw a random value for every parameter in the search space."""
        return self.objective_function.search_space.sample(self.rng)

    def _new_generation(self, iteration: int) -> Generation:
        """Spawns a new generation with the id of the iteration provided."""
        return Generation(iteration)

    def _is_better(self, fitness: float, other: float) -> bool:
        """Whether fitness beats other in the configured direction."""
        return fitness > other if self.config.maximize else fitness < other

    def _ranked(self, population: Population) -> List[Individual]:
        """Individuals of the population, best first."""
        return sorted(population, key=lambda x: x.fitness, reverse=self.config.maximize)

    def _roulette_selection(self, generation: Generation) -> Individual:
        """Select an individual with probability proportional to its fitness.

        Fitnesses are shifted so the worst individual has weight zero, which makes the
        selection work for negative fitnesses and for minimization alike.
        """
        fitnesses = np.array([individual.fitness for individual in generation.population], dtype=float)
        weights = fitnesses - fitnesses.min() if self.config.maximize else fitnesses.max() - fitnesses
        total = weights.sum()
        if total == 0:
            # Every individual is equally fit.
            return generation.population[self.rng.integers(len(fitnesses))]
        return generation.population[self.rng.choice(len(fitnesses), p=weights / total)]

    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Build a child gene by gene, taking each from parent1 with probability crossover_rate."""
        child = Individual()
        for param in parent1.params.keys():
            if self.rng.uniform(0, 1) < self.config.crossover_rate:
                child.params[param] = parent1.params[param]
            else:
                child.params[param] = parent2.params[param]

        return child

    def _mutate_individual(self, individual: Individual) -> Individual:
        """Function which simulates mutation of an individual."""
        random_params = self._random_parameters()
        for param in individual.params.keys():
            if self.rng.uniform(0, 1) < self.config.mutation_threshold:
                individual.params[param] = random_params[param]

        return individual

    def _tournament_selection(self, generation: Generation) -> Individual:
        """Select an individual using tournament selection."""
        indices = self.rng.choice(len(generation.population), size=self.config.tournament_size)
        tournament = sorted((generation.population[i] for i in indices), key=lambda x: x.fitness, reverse=self.config.maximize)
        return tournament[0] if self.rng.uniform(0, 1) < self.config.tournament_p else tournament[-1]

    def _select_individual(self, generation: Generation) -> Individual:
        """Select an individual using the selection method specified in the config.
        Args:
            generation: The generation from which to select an individual.

        Returns:
            An individual selected from the generation.
        """
        match self.config.selection_method:
            case SelectionMethod.TOURNAMENT:
                return self._tournament_selection(generation)
            case SelectionMethod.ROULETTE:
                return self._roulette_selection(generation)

    def _new_population(self) -> Population:
        """Spawns a new empty population."""
        return Population()

    def _initial_population(self, generation_id: int) -> Population:
        """A population of individuals with randomly drawn parameters."""
        population = self._new_population()
        for _ in range(self.config.population_size):
            individual = Individual(**self._random_parameters())
            individual.generation = generation_id
            population.add_individual(individual)
        return population

    def _next_population(self, previous: Generation, generation_id: int) -> Population:
        """Breed a population from the previous generation, carrying over its elites if enabled."""
        population = self._new_population()
        if self.config.elitism:
            n_elites = round(self.config.population_size * self.config.elitism_rate)
            population.add_individuals(self._ranked(previous.population)[:n_elites])

        while population.size < self.config.population_size:
            parent1 = self._select_individual(previous)
            parent2 = self._select_individual(previous)
            child = self._uniform_crossover(parent1, parent2)
            if self.rng.uniform(0, 1) < self.config.mutation_rate:
                child = self._mutate_individual(child)
            child.generation = generation_id
            population.add_individual(child)

        return population

    def _evaluate_population(self, generation: Generation) -> None:
        """Evaluate the fitness of each individual in the population.
        Args:
            generation: The generation to evaluate.

        Returns:
            None

        Behavior:
            evaluation results are stored in the generation object.
            Elites carried over from an earlier generation keep their fitness
            rather than being evaluated again.
        """
        for individual in generation.population:
            if individual.fitness is None:
                individual.fitness = self.objective_function.function(**individual.params)

        ranked = self._ranked(generation.population)
        generation.best = ranked[0]
        generation.worst = ranked[-1]

        generation.mean = np.mean([individual.fitness for individual in generation.population])
        generation.median = np.median([individual.fitness for individual in generation.population])
        generation.std = np.std([individual.fitness for individual in generation.population])
        self.genealogy.append(generation)

    def _reached_threshold(self, fitness: float) -> bool:
        """Whether fitness is good enough to stop early. Never true without a threshold."""
        if self.config.threshold is None:
            return False
        return fitness >= self.config.threshold if self.config.maximize else fitness <= self.config.threshold

    def optimize(self) -> Result:
        """Optimize the objective function using the genetic algorithm.

        Runs for config.generations generations, or until the best fitness reaches
        config.threshold if one is set.
        """
        self.genealogy = []
        best: Individual | None = None

        for generation_id in range(self.config.generations):
            generation = self._new_generation(generation_id)
            if generation_id == 0:
                generation.population = self._initial_population(generation_id)
            else:
                generation.population = self._next_population(self.genealogy[-1], generation_id)

            self._evaluate_population(generation)
            if best is None or self._is_better(generation.best.fitness, best.fitness):
                best = generation.best

            if self._reached_threshold(best.fitness):
                break

        self.objective_function.optimum = best.fitness
        self.objective_function.optimum_params = dict(best.params)

        return Result(
            generations=self.genealogy,
            best=best,
            best_fitness=best.fitness,
            optimum=best.fitness,
            optimum_params=dict(best.params),
        )
