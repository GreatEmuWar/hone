from hone.optimizers.base import Individual, Optimizer


class RandomSearch(Optimizer):
    """Evaluates population_size random points per generation and keeps the best.

    Useful as a baseline: a smarter optimizer should beat it for the same number of evaluations.
    Any initial points take the place of random points in the first generation.
    """

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        initial = self._initial_individuals(generation) if previous is None else []
        return initial + [self._random_individual(generation) for _ in range(self.config.population_size - len(initial))]
