import math
from functools import partial
from typing import Callable

from hone.config import GeneticOptimizerConfig, SelectionMethod
from hone.optimizers.base import Individual, Optimizer, _fitness, _key
from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter

from . import selection
from .variation import blend_crossover, mutate, uniform_crossover


# Adaptive mutation: how much a child's mutation size varies from its parents', divided by the
# square root of the number of variables, and the bounds on the size. Chosen on the benchmark.
STEP_SPREAD = 0.5
MIN_STEP = 1e-9
MAX_STEP = 0.5
# A child that repeats a point already asked for is mutated again, one gene at a time, up to this many times.
REPEAT_TRIES = 10
# A child that breaks a constraint is moved halfway back toward its fitter parent this many times.
REPAIR_STEPS = 10


class GeneticOptimizer(Optimizer):
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
    config: GeneticOptimizerConfig

    def reset(self) -> None:
        super().reset()
        # The parameters a mutation can change.
        self._movable = [
            p.name for p in self.search_space.parameters
            if not isinstance(p, FixedParameter)
            and not (isinstance(p, RangeParameter) and p.maximum == p.minimum)
            and not (isinstance(p, ChoiceParameter) and len(p.choices) == 1)
        ]
        variables = sum(
            not isinstance(p, FixedParameter) and not (isinstance(p, RangeParameter) and p.maximum == p.minimum)
            for p in self.search_space.parameters
        )
        if self.config.mutation_rate is not None:
            self._mutation_rate = self.config.mutation_rate
        else:
            self._mutation_rate = 0.1 if self.config.noise else max(0.1, 1 / max(variables, 1))
        self._adaptive = self.config.adaptive_mutation if self.config.adaptive_mutation is not None else not self.config.noise
        self._step_spread = STEP_SPREAD / math.sqrt(max(variables, 1))
        self._steps: dict[Individual, float] = {}  # each individual's mutation size, with adaptive mutation
        self._asked: set[tuple] = set()  # every point put in a population, so children can avoid repeating one

    def _select(self, individuals: list[Individual]) -> Individual:
        select: Callable[..., Individual]
        match self.config.selection_method:
            case SelectionMethod.TOURNAMENT:
                select = partial(selection.tournament, size=self.config.tournament_size, p=self.config.tournament_p)
            case SelectionMethod.ROULETTE:
                select = selection.roulette
        return select(individuals, self.rng, maximize=self.config.maximize)

    def _child(self, parents: list[Individual], generation: int, niche: dict | None = None) -> Individual:
        """A child of two parents selected from parents. A niche fixes the child's choice parameters.

        A child that breaks a constraint is repaired: moved toward the fitter of its parents,
        which satisfies every constraint, by halving the gap REPAIR_STEPS times and keeping the
        last point inside. Its numbers end up near the constraint's boundary, where optima
        under constraints usually are.
        """
        child, first, second = self._bred(parents, generation, niche)
        if self.search_space.feasible(child.params):
            return child
        fitter = first if (_fitness(first) > _fitness(second)) == self.config.maximize else second
        return self._repaired(child, fitter, generation)

    def _repaired(self, child: Individual, anchor: Individual, generation: int) -> Individual:
        inside, outside = dict(anchor.params), dict(child.params)
        if not self.search_space.feasible(inside):
            inside = self.search_space.sample(self.rng)  # can't happen: every evaluated point is feasible
        for _ in range(REPAIR_STEPS):
            middle = self._between(inside, outside)
            if self.search_space.feasible(middle):
                inside = middle
            else:
                outside = middle
        repaired = Individual(params=inside, generation=generation)
        if child in self._steps:
            self._steps[repaired] = self._steps.pop(child)
        self._asked.add(_key(inside))
        return repaired

    def _between(self, inside: dict, outside: dict) -> dict:
        """The midpoint of two points, in each range's unit scale; other values are taken from inside."""
        middle = dict(inside)
        for parameter in self.search_space.parameters:
            if isinstance(parameter, RangeParameter) and parameter.maximum > parameter.minimum:
                unit = (float(parameter.to_unit(inside[parameter.name])) + float(parameter.to_unit(outside[parameter.name]))) / 2
                value = parameter.from_unit(unit)
                middle[parameter.name] = int(value) if parameter.integer else float(value)
        return middle

    def _bred(self, parents: list[Individual], generation: int, niche: dict | None) -> tuple[Individual, Individual, Individual]:
        """A child bred from two parents selected from parents, and the parents."""
        first, second = self._select(parents), self._select(parents)
        if self.config.blend_crossover:
            params = blend_crossover(
                first.params, second.params, self.search_space, self.rng, rate=self.config.crossover_rate, alpha=self.config.blend_alpha
            )
        else:
            params = uniform_crossover(first.params, second.params, self.rng, rate=self.config.crossover_rate)
        step = self.config.mutation_scale
        if self._adaptive:
            inherited = math.sqrt(self._step(first) * self._step(second))
            step = min(max(inherited * math.exp(self.rng.normal(0, self._step_spread)), MIN_STEP), MAX_STEP)
        params = mutate(params, self.search_space, self.rng, rate=self._mutation_rate, scale=step)
        params = {**params, **(niche or {})}
        if not self.config.noise:
            params = self._new_point(params, step, niche)
        child = Individual(params=params, generation=generation)
        if self._adaptive:
            self._steps[child] = step
        return child, first, second

    def _new_point(self, params: dict, step: float, niche: dict | None) -> dict:
        """params, mutated again a gene at a time while they repeat a point already asked for.

        Gives up after REPEAT_TRIES, which only happens when the space has few points left.
        A niche's choices stay fixed.
        """
        movable = [name for name in self._movable if not niche or name not in niche]
        for _ in range(REPEAT_TRIES):
            if not movable or _key(params) not in self._asked:
                break
            name = movable[self.rng.integers(len(movable))]
            params = {**params, name: self.search_space[name].mutate(params[name], self.rng, step)}
        self._asked.add(_key(params))
        return params

    def _step(self, individual: Individual) -> float:
        return self._steps.get(individual, self.config.mutation_scale)

    def _niche_children(self, parents: list[Individual], count: int, generation: int) -> list[Individual]:
        """Up to count children bred within niches: parents sharing every choice parameter's value.

        Each niche gets an equal share, so every combination of choices still in the
        population keeps tuning its own numbers instead of being outbred by the leader.
        """
        choices = [p.name for p in self.search_space.parameters if isinstance(p, ChoiceParameter) and len(p.choices) > 1]
        if not choices or count == 0:
            return []
        niches: dict[tuple, list[Individual]] = {}
        for parent in parents:
            niches.setdefault(tuple(repr(parent.params[name]) for name in choices), []).append(parent)
        # With more niches than children, the niches with the fittest members breed.
        ranked = sorted(niches.values(), key=lambda members: _fitness(self._ranked(members)[0]), reverse=self.config.maximize)
        per_niche = max(1, count // len(ranked))
        children: list[Individual] = []
        for members in ranked:
            niche = {name: members[0].params[name] for name in choices}
            for _ in range(min(per_niche, count - len(children))):
                children.append(self._child(members, generation, niche))
        return children

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        size = self.config.population_size
        if previous is None:
            initial = self._initial_individuals(generation)
            population = initial + [self._random_individual(generation) for _ in range(size - len(initial))]
            self._asked.update(_key(individual.params) for individual in population)
            return population
        parents = [individual for individual in previous if not individual.failed]
        if not parents:
            # Every evaluation of the previous generation failed: start afresh.
            population = [self._random_individual(generation) for _ in range(size)]
            self._asked.update(_key(individual.params) for individual in population)
            return population

        if self.config.noise:
            # A point bred again since it was evaluated has more evaluations now; rank by their average.
            self._refresh_averages(parents)
        # At least one child per generation, however the share rounds, so the search keeps moving.
        elites = self._ranked(parents)[: min(round(size * self.config.elitism_rate), size - 1)] if self.config.elitism else []
        niched = self._niche_children(parents, int(self.config.niche_share * (size - len(elites))), generation)
        population = elites + niched + [self._child(parents, generation) for _ in range(size - len(elites) - len(niched))]
        if self._adaptive:
            # Keep the sizes of individuals that can still breed: this generation, and any still being evaluated.
            keep = set(population) | set(self._outstanding())
            self._steps = {individual: step for individual, step in self._steps.items() if individual in keep}
        return population
