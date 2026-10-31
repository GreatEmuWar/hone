"""CMA-ES: the covariance matrix adaptation evolution strategy.

The update follows N. Hansen, "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772),
with positive recombination weights. Three additions adapt it to hone's search spaces:

- Range parameters are scaled to [0, 1], on a log scale where they have one, and a sample
  that falls outside is mirrored back in. The mirrored point is the one evaluated and used
  in the update.
- Integer parameters keep a minimum spread of a quarter of one integer step, so the search
  can always move to a neighbouring value. Samples are widened to that spread, but the
  distribution learns from the steps as drawn before widening, as in Hamano et al.,
  "CMA-ES with Margin" (GECCO 2022), so the widening doesn't inflate the step size.
- With restarts on, a run that converges or stalls starts again from a random point with
  twice the population (Auger and Hansen, "A Restart CMA Evolution Strategy With Increasing
  Population Size", CEC 2005).
"""
import math
from typing import Any, List

import numpy as np

from hone.config import CMAESConfig
from hone.optimizers.base import Individual, Optimizer, UnsupportedSearchSpace, _fitness
from hone.parameters import ChoiceParameter, FixedParameter, Parameter, RangeParameter, SearchSpace

INTEGER_SPREAD = 0.25  # minimum standard deviation of an integer variable, in integer steps
MAX_CONDITION = 1e14  # restart when the covariance matrix is this badly conditioned
MIN_STEP = 1e-12  # restart when every continuous step, in [0, 1] units, falls below this
FLAT = 1e-12  # restart when recent generations' best values differ by less than this
RESAMPLES = 100  # draws from the distribution for a feasible candidate before falling back to a random feasible point


class CMAES(Optimizer):
    """Covariance matrix adaptation evolution strategy, for range parameters.

    Each generation samples candidates from a multivariate normal distribution, then moves
    the distribution's centre toward the best of them and reshapes it to stretch along the
    directions that improved. It learns the scale of each variable and how variables
    interact, which makes it strong on smooth continuous problems.

    Range parameters, continuous or integer, are searched; fixed parameters are passed
    through. Choice parameters are not supported: use GeneticOptimizer for those.

    The search starts at the centre of every range. With initial points, it evaluates them
    first and starts at the best one instead; a smaller initial_step then keeps the search
    close to it.

    Args:
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: A CMAESConfig. Defaults are used if omitted.
        initial_points: Points to evaluate first: one dict of parameter values, or a list of them.
    """
    config_class = CMAESConfig
    supports_constraints = True
    _constraint_handling = "resample"  # or "rank": chosen on the constrained benchmark
    config: CMAESConfig

    def __init__(
        self,
        search_space: SearchSpace | List[Parameter],
        config: CMAESConfig | None = None,
        *,
        initial_points: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(search_space, SearchSpace):
            search_space = SearchSpace(search_space)
        choices = [p.name for p in search_space.parameters if isinstance(p, ChoiceParameter)]
        if choices:
            raise UnsupportedSearchSpace(
                f"CMA-ES searches range parameters only; use GeneticOptimizer for choice parameters ({', '.join(choices)})"
            )
        searched = [p for p in search_space.parameters if isinstance(p, RangeParameter) and p.maximum > p.minimum]
        if not searched:
            raise UnsupportedSearchSpace("CMA-ES needs at least one range parameter with minimum below maximum")
        self._searched = searched
        self._constant = {
            p.name: (p.from_unit(0.0) if isinstance(p, RangeParameter) else p.value)
            for p in search_space.parameters
            if isinstance(p, (RangeParameter, FixedParameter)) and p not in searched
        }
        self._integer = np.array([p.integer for p in searched])
        super().__init__(search_space, config, initial_points=initial_points)

    # --- state ----------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        n = len(self._searched)
        # Each generation waits for its slowest evaluation, so with many workers a population of
        # at least twice their number keeps them busy.
        self.population_size = self.config.population_size or max(4 + int(3 * math.log(n)), 2 * self._workers)
        self._start(mean=np.full(n, 0.5))
        self._evaluating_initial_points = False

    def _start(self, mean: np.ndarray) -> None:
        """Set up a fresh search distribution around mean for the current population size."""
        n = len(mean)
        lam = self.population_size
        mu = lam // 2
        weights = math.log((lam + 1) / 2) - np.log(np.arange(1, mu + 1))
        self._weights = weights / weights.sum()
        self._mu_eff = 1 / np.sum(self._weights**2)
        mu_eff = self._mu_eff

        self._c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        self._d_sigma = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + self._c_sigma
        self._c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        self._c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        self._c_mu = min(1 - self._c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        self._chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        self._mean = mean
        self._sigma = self.config.initial_step
        self._C = np.eye(n)
        self._B = np.eye(n)
        self._D = np.ones(n)
        self._p_sigma = np.zeros(n)
        self._p_c = np.zeros(n)
        self._iteration = 0
        self._recent_bests: list[float] = []
        self._vectors: dict[Individual, np.ndarray] = {}  # each individual of the current generation -> its point in [0, 1]^n
        self._infeasible: list[tuple[float, np.ndarray]] = []

    # --- the generation -------------------------------------------------------

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        if previous is None and self.initial_points:
            self._evaluating_initial_points = True
            return self._initial_individuals(generation)
        if self._evaluating_initial_points:
            assert previous is not None
            self._evaluating_initial_points = False
            succeeded = self._ranked([ind for ind in previous if not ind.failed])
            if succeeded:
                self._mean = self._encode(succeeded[0].params)
        elif previous is not None:
            self._update(previous)
        spread = self._sigma * np.sqrt(np.diag(self._C))
        minimum = self._minimum_spread()
        # Integer variables are sampled wider when their spread falls below the minimum.
        self._widen = np.where(spread < minimum, minimum / np.maximum(spread, 1e-300), 1.0)
        self._infeasible: list[tuple[float, np.ndarray]] = []  # with "rank": violation and point of each infeasible sample
        population = []
        for _ in range(self.population_size):
            individual = self._sample(generation)
            if individual is not None:
                population.append(individual)
        return population

    def _draw(self) -> np.ndarray:
        z = self.rng.standard_normal(len(self._mean))
        return _mirror(self._mean + self._sigma * self._widen * (self._B @ (self._D * z)))

    def _sample(self, generation: int) -> Individual | None:
        """A candidate from the search distribution that satisfies every constraint, or None if it's set aside unevaluated."""
        x = self._draw()
        params = self._decode(x)
        constraints = self.search_space.constraints
        if constraints and not self.search_space.feasible(params):
            if self._constraint_handling == "rank":
                self._infeasible.append((sum(c.violation(params) for c in constraints), x))
                return None
            for _ in range(RESAMPLES):
                x = self._draw()
                params = self._decode(x)
                if self.search_space.feasible(params):
                    break
            else:
                params = self.search_space.sample(self.rng)
                x = self._encode(params)
        individual = Individual(params=params, generation=generation)
        self._vectors[individual] = x
        return individual

    def _update(self, previous: list[Individual]) -> None:
        ranked = [ind for ind in self._ranked(previous) if not ind.failed and ind in self._vectors]
        # With "rank", samples that broke a constraint follow every feasible one, least violation first.
        vectors = [self._vectors[ind] for ind in ranked] + [x for _, x in sorted(self._infeasible, key=lambda pair: pair[0])]
        if vectors:
            self._adapt(np.array(vectors[: len(self._weights)]))
        if ranked:
            self._recent_bests.append(_fitness(ranked[0]))
        self._vectors = {}
        if self.config.restarts and self._should_restart():
            self.population_size *= 2
            self._start(mean=self.rng.uniform(0, 1, len(self._mean)))

    def _adapt(self, best: np.ndarray) -> None:
        """Move and reshape the distribution toward best, the top points ordered best first."""
        n = len(self._mean)
        weights = self._weights[: len(best)] / self._weights[: len(best)].sum()
        # The paths, step size and covariance learn from the steps as sampled before integer
        # widening, as in CMA-ES with margin; otherwise the widening would read as the search
        # moving faster than it is, and inflate the step size.
        steps = (best - self._mean) / (self._sigma * self._widen)
        step = weights @ steps
        self._mean = self._mean + self._sigma * self._widen * step

        inv_sqrt_C = self._B @ np.diag(1 / self._D) @ self._B.T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * (inv_sqrt_C @ step)
        self._iteration += 1
        norm = np.linalg.norm(self._p_sigma)
        h_sigma = norm / math.sqrt(1 - (1 - self._c_sigma) ** (2 * self._iteration)) < (1.4 + 2 / (n + 1)) * self._chi_n
        self._p_c = (1 - self._c_c) * self._p_c + h_sigma * math.sqrt(self._c_c * (2 - self._c_c) * self._mu_eff) * step

        rank_mu = steps.T @ (weights[:, None] * steps)
        correction = (1 - h_sigma) * self._c_c * (2 - self._c_c)
        self._C = (
            (1 - self._c_1 - self._c_mu + self._c_1 * correction) * self._C
            + self._c_1 * np.outer(self._p_c, self._p_c)
            + self._c_mu * rank_mu
        )
        self._sigma *= math.exp((self._c_sigma / self._d_sigma) * (norm / self._chi_n - 1))

        self._C = (self._C + self._C.T) / 2
        eigenvalues, self._B = np.linalg.eigh(self._C)
        self._D = np.sqrt(np.maximum(eigenvalues, 1e-300))

    def _should_restart(self) -> bool:
        continuous = ~self._integer
        if continuous.any() and np.all(self._sigma * np.sqrt(np.diag(self._C))[continuous] < MIN_STEP):
            return True
        if (self._D.max() / self._D.min()) ** 2 > MAX_CONDITION:
            return True
        window = 10 + math.ceil(30 * len(self._mean) / self.population_size)
        recent = self._recent_bests[-window:]
        return len(recent) == window and max(recent) - min(recent) <= FLAT * max(1.0, abs(min(recent)))

    def _minimum_spread(self) -> np.ndarray:
        """The smallest standard deviation each variable keeps, in [0, 1] units: INTEGER_SPREAD integer steps near the mean.

        On a log scale an integer step is wider at the bottom of the range than at the top,
        so it is measured where the search currently is.
        """
        spread = np.zeros(len(self._searched))
        for i, p in enumerate(self._searched):
            if not p.integer:
                continue
            if p.log:
                value = float(np.exp(np.log(p.minimum) + self._mean[i] * (np.log(p.maximum) - np.log(p.minimum))))
                spread[i] = INTEGER_SPREAD * (p.to_unit(value + 1) - p.to_unit(value))
            else:
                spread[i] = INTEGER_SPREAD / (p.maximum - p.minimum)
        return spread

    # --- between [0, 1]^n and parameter values --------------------------------

    def _encode(self, params: dict) -> np.ndarray:
        return np.array([p.to_unit(params[p.name]) for p in self._searched], dtype=float)

    def _decode(self, x: np.ndarray) -> dict:
        values = dict(self._constant)
        for parameter, u in zip(self._searched, x):
            values[parameter.name] = parameter.from_unit(float(u))
        return {name: values[name] for name in self.search_space.names}


def _mirror(x: np.ndarray) -> np.ndarray:
    """Reflect each coordinate back into [0, 1] at the edges."""
    x = np.mod(x, 2.0)
    return np.where(x > 1, 2 - x, x)
