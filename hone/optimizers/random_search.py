from hone.optimizers.base import Individual, Optimizer


class RandomSearch(Optimizer):
    """Evaluates population_size random points per generation and keeps the best.

    Useful as a baseline: a smarter optimizer should beat it for the same number of evaluations.
    """

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        return [self._random_individual(generation) for _ in range(self.config.population_size)]
