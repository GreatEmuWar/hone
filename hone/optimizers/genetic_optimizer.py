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
    population_id: str = str(uuid4())

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
        self.geaneology: List[Generation] = []

    def _random_parameters(self) -> Dict[str, float | int | str]:
        """Draw a random value for every parameter in the search space."""
        return self.objective_function.search_space.sample(self.rng)

    def _new_generation(self, iteration: int) -> Generation:
        """Spawns a new generation with the id of the iteration provided."""
        return Generation(iteration)
    
    def _roulette_selection(self, generation: Generation) -> Individual:
        """Select an individual using roulette selection."""
        fitnesses = [individual.fitness for individual in generation.population]
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        return generation.population[self.rng.choice(len(fitnesses), p=probabilities)]

    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform single point crossover."""
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

    def _evaluate_population(self, generation: Generation) -> None:
        """Evaluate the fitness of each individual in the population.
        Args:
            generation: The generation to evaluate.

        Returns:
            None

        Behavior:
            evaluation results are stored in the generation object.
            this function simply triggers the evaluation event in the algorithm.
        """
        for individual in generation.population:
            individual.fitness = self.objective_function.function(**individual.params)

        if self.config.maximize:
            generation.best = max(generation.population, key=lambda x: x.fitness)
            generation.worst = min(generation.population, key=lambda x: x.fitness)
        else:
            generation.best = min(generation.population, key=lambda x: x.fitness)
            generation.worst = max(generation.population, key=lambda x: x.fitness)
        
        generation.mean = np.mean([individual.fitness for individual in generation.population])
        generation.median = np.median([individual.fitness for individual in generation.population])
        generation.std = np.std([individual.fitness for individual in generation.population])
        self.geaneology.append(generation)

    def optimize(self) -> Result:
        """Optimize the objective function using the genetic algorithm."""
        for i in range(self.config.generations):
            if i == 0:
                generation = self._new_generation(i)
                population = self._new_population()
                for i in range(self.config.population_size):
                    population.add_individual(
                        Individual(
                            params=None,
                            fitness=None,
                            generation=i,
                            population=population.population_id
                        )
                    )
                    population[i].params = self._random_parameters()
                    
                generation.population = population
            
            else:
                generation = self._new_generation(i)
                population = self._new_population()
                if self.config.elitism:
                    n_elites = round(self.config.population_size * self.config.elitism_rate)
                    elites = sorted(self.geaneology[i - 1].population, key=lambda x: x.fitness, reverse=self.config.maximize)[:n_elites]
                    population.add_individuals(elites)
                    assert population.size == len(elites)

                else:
                    elites = []
                    previous = self.geaneology[i - 1].population
                    survivors = self.rng.choice(len(previous), round(self.config.population_size * self.config.elitism_rate), replace=False)
                    population.add_individuals([previous[j] for j in survivors])

                while population.size <= self.config.population_size:
                    parent1 = self._select_individual(self.geaneology[i - 1])
                    parent2 = self._select_individual(self.geaneology[i - 1])
                    child = self._single_point_crossover(parent1, parent2)
                    if self.rng.uniform(0, 1) < self.config.mutation_rate:
                        child = self._mutate_individual(child)
                        
                    population.add_individual(child)
                
                generation.population = population
            
            self._evaluate_population(generation)
            if self.config.maximize:
                # Maximize the objective function.
                # If the best fitness is greater than or equal to the threshold, stop.
                if generation.best.fitness >= self.config.threshold:
                    break
                
                # If the best fitness is greater than the current optimum, update the optimum.
                # Update iterator.
                elif generation.best.fitness > self.objective_function.optimum or self.objective_function.optimum is None:
                    self.objective_function.optimum = generation.best.fitness
                    self.objective_function.optimum_params = generation.best.params
                    i += 1
                
                # Otherwise, update iterator.
                else:
                    i += 1

            else:
                # Minimize the objective function.
                # If the best fitness is less than or equal to the threshold, stop.
                if generation.best.fitness <= self.config.threshold:
                    break

                # If the best fitness is less than the current optimum, update the optimum.
                # Update iterator.
                elif generation.best.fitness < self.objective_function.optimum or self.objective_function.optimum is None:
                    self.objective_function.optimum = generation.best.fitness
                    self.objective_function.optimum_params = generation.best.params
                    i += 1

                # Otherwise, update iterator.
                else:
                    i += 1

        # Find the best individual from all generations.
        best = max(self.geaneology, key=lambda x: x.best.fitness)

        # Return the optimization result.
        return Result(
            generations=self.geaneology,
            best=best.best.individual_id,
            best_fitness=best.best.fitness,
            optimum=self.objective_function.optimum,
            optimum_params=self.objective_function.optimum_params
        )