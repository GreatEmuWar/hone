"""Breeding children from parents, shared by the genetic optimizer and NSGA-II.

A child is made by crossover of two selected parents, then mutation, then, without noise,
mutated again while it repeats a point already asked for. A child that breaks a constraint
is repaired toward its fitter parent. With conditions, a gene switched off in one parent comes
from the other, and only genes switched on are mutated. Children can be bred within niches,
parents that share their choices, so each option keeps tuning its own numbers.

An optimizer using Breeding decides how parents are selected, which of two parents is
fitter, and how niches rank, and calls _setup_breeding() from reset().
"""
import math
from abc import abstractmethod
from typing import Any

from hone.optimizers.core import Individual
from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter, SearchSpace

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
# The share of children bred within niches when niche_share isn't set and the search space has
# conditions: on the conditional benchmark it keeps every branch tuning its own numbers, so a
# narrow best branch isn't outbred by a wide decoy.
CONDITIONAL_NICHE_SHARE = 0.5


class Breeding:
    """Crossover, mutation, repeat avoidance, repair and niches, for an optimizer with a config holding their settings."""
    search_space: SearchSpace
    config: Any
    rng: Any

    def _key(self, params: dict[str, Any]) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def _select(self, individuals: list[Individual]) -> Individual:
        """One parent, selected from individuals."""

    @abstractmethod
    def _fitter(self, first: Individual, second: Individual) -> Individual:
        """The fitter of two parents, which a child that breaks a constraint is repaired toward."""

    @abstractmethod
    def _ranked_niches(self, niches: list[list[Individual]]) -> list[list[Individual]]:
        """The niches, those with the fittest members first."""

    def _setup_breeding(self) -> None:
        """Set up breeding for a new run. Called from reset()."""
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
        if self.config.niche_share is not None:
            self._niche_share = self.config.niche_share
        else:
            self._niche_share = CONDITIONAL_NICHE_SHARE if self.search_space.conditions else 0.0

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
        return self._repaired(child, self._fitter(first, second), generation)

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
        mother, father = self._aligned(first.params, second.params)
        if self.config.blend_crossover:
            params = blend_crossover(
                mother, father, self.search_space, self.rng, rate=self.config.crossover_rate, alpha=self.config.blend_alpha
            )
        else:
            params = uniform_crossover(mother, father, self.rng, rate=self.config.crossover_rate)
        step = self.config.mutation_scale
        if self._adaptive:
            inherited = math.sqrt(self._step(first) * self._step(second))
            step = min(max(inherited * math.exp(self.rng.normal(0, self._step_spread)), MIN_STEP), MAX_STEP)
        only = set(self.search_space.active(params)) if self.search_space.conditions else None
        params = mutate(params, self.search_space, self.rng, rate=self._mutation_rate, scale=step, only=only)
        params = {**params, **(niche or {})}
        if not self.config.noise:
            params = self._new_point(params, step, niche)
        child = Individual(params=params, generation=generation)
        if self._adaptive:
            self._steps[child] = step
        return child, first, second

    def _aligned(self, first: dict, second: dict) -> tuple[dict, dict]:
        """The parents' values, each gene switched off in one parent but on in the other taken from the other.

        So crossover passes on a gene's value from a parent it was in use in, rather than one
        that no evaluation has judged.
        """
        if not self.search_space.conditions:
            return first, second
        on_first, on_second = self.search_space.active(first), self.search_space.active(second)
        mother = {name: on_second[name] if name not in on_first and name in on_second else value for name, value in first.items()}
        father = {name: on_first[name] if name not in on_second and name in on_first else value for name, value in second.items()}
        return mother, father

    def _new_point(self, params: dict, step: float, niche: dict | None) -> dict:
        """params, mutated again a gene at a time while they repeat a point already asked for.

        Gives up after REPEAT_TRIES, which only happens when the space has few points left.
        A niche's choices stay fixed.
        """
        for _ in range(REPEAT_TRIES):
            # Only a gene switched on changes the point.
            on = self.search_space.active(params)
            movable = [name for name in self._movable if name in on and (not niche or name not in niche)]
            if not movable or self._key(params) not in self._asked:
                break
            name = movable[self.rng.integers(len(movable))]
            params = {**params, name: self.search_space[name].mutate(params[name], self.rng, step)}
        self._asked.add(self._key(params))
        return params

    def _step(self, individual: Individual) -> float:
        return self._steps.get(individual, self.config.mutation_scale)

    def _niche_children(self, parents: list[Individual], count: int, generation: int) -> list[Individual]:
        """Up to count children bred within niches: parents sharing every choice parameter's value.

        Each niche gets an equal share, so every combination of choices still in the
        population keeps tuning its own numbers instead of being outbred by the leader. Choices
        switched off by a condition don't count, so each branch is a niche.
        """
        choices = [p.name for p in self.search_space.parameters if isinstance(p, ChoiceParameter) and len(p.choices) > 1]
        if not choices or count == 0:
            return []
        niches: dict[tuple, list[Individual]] = {}
        for parent in parents:
            on = self.search_space.active(parent.params)
            niches.setdefault(tuple(repr(on[name]) if name in on else "off" for name in choices), []).append(parent)
        # With more niches than children, the niches with the fittest members breed.
        ranked = self._ranked_niches(list(niches.values()))
        per_niche = max(1, count // len(ranked))
        children: list[Individual] = []
        for members in ranked:
            on = self.search_space.active(members[0].params)
            niche = {name: on[name] for name in choices if name in on}
            for _ in range(min(per_niche, count - len(children))):
                children.append(self._child(members, generation, niche))
        return children

    def _keep_steps(self, keep: set[Individual]) -> None:
        """Forget the mutation sizes of individuals that can no longer breed."""
        if self._adaptive:
            self._steps = {individual: step for individual, step in self._steps.items() if individual in keep}
