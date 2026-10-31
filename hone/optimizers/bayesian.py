"""Bayesian optimization: model the objective with a Gaussian process, and evaluate where the model expects the most improvement.

Each candidate costs a model fit and a search over the model, a fraction of a second up to a few
hundred evaluations, so Bayesian optimization suits objectives that take far longer than that:
simulations, experiments, training runs. It makes the most of a small budget.

Range parameters are scaled to [0, 1], on a log scale where they have one; integers are searched
as numbers and rounded. Each choice
parameter becomes one column per choice, holding 1 for the chosen one and 0 for the rest.
"""
import math
from typing import Any, List

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfcx, log_ndtr
from threadpoolctl import threadpool_limits

from hone.config import BayesianConfig
from hone.optimizers._gp import GaussianProcess
from hone.optimizers.base import Individual, Optimizer, UnsupportedSearchSpace
from hone.parameters import ChoiceParameter, Parameter, RangeParameter, SearchSpace

GLOBAL_CANDIDATES = 1000  # random candidates per search of the model
LOCAL_CANDIDATES = 500  # candidates near the best points found so far
REFINED = 5  # the best candidates are refined by gradient search
REFIT_EVERY = 5  # past FIT_EVERY_UNTIL observations, refit the model's hyperparameters only this often
FIT_EVERY_UNTIL = 50


class BayesianOptimizer(Optimizer):
    """Bayesian optimization with a Gaussian process and expected improvement.

    The first generation is spread over the search space (with any initial points). After that,
    each candidate is the point where a Gaussian process model of the objective predicts the
    largest expected improvement on the best value so far. Failed evaluations are modelled as
    the worst value seen, which steers the search away from them.

    Range parameters, integer or not, and choice parameters are supported, but not choice
    parameters with choose_n. Fixed parameters are passed through.

    Args:
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: A BayesianConfig. Defaults are used if omitted.
        initial_points: Points to evaluate first: one dict of parameter values, or a list of them.
    """
    config_class = BayesianConfig
    supports_constraints = True
    config: BayesianConfig

    def __init__(
        self,
        search_space: SearchSpace | List[Parameter],
        config: BayesianConfig | None = None,
        *,
        initial_points: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(search_space, SearchSpace):
            search_space = SearchSpace(search_space)
        self._encoding = _Encoding(search_space)
        super().__init__(search_space, config, initial_points=initial_points)

    def reset(self) -> None:
        super().reset()
        self.batch_size = self.config.population_size or self._workers
        self._X: list[np.ndarray] = []  # every evaluated point, encoded
        self._y: list[float | None] = []  # its value to minimize, or None if it failed
        self._theta: np.ndarray | None = None  # the model's hyperparameters, kept between fits
        self._fitted_at = 0  # observations at the last hyperparameter fit

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        if previous is None:
            initial = self._initial_individuals(generation)
            count = self.config.initial_samples or 2 * (self._encoding.variables + 1)
            taken = {tuple(self._encoding.encode(ind.params)) for ind in initial}
            spread = []
            for x in self._feasible_spread(max(count - len(initial), 0)):
                # Rounding integers can land two points on the same values; try others instead.
                for _ in range(100):
                    if tuple(x) not in taken:
                        break
                    x = self._feasible_spread(1)[0]
                taken.add(tuple(x))
                spread.append(x)
            return initial + [Individual(params=self._encoding.decode(x), generation=generation) for x in spread]
        for individual in previous:
            if individual.fitness is not None:
                self._X.append(self._encoding.encode(individual.params))
                value = individual.fitness if not self.config.maximize else -individual.fitness
                self._y.append(None if individual.failed else value)
        # The model's matrices are small, and splitting their arithmetic across threads makes it many
        # times slower. One thread also makes results the same on every machine.
        with threadpool_limits(limits=1, user_api="blas"):
            proposals = self._propose(self.batch_size)
        return [Individual(params=self._encoding.decode(x), generation=generation) for x in proposals]

    # --- choosing candidates --------------------------------------------------

    def _propose(self, count: int) -> list[np.ndarray]:
        succeeded = [y for y in self._y if y is not None]
        if len(succeeded) < 2:
            return list(self._feasible_spread(count))
        worst = max(succeeded)
        X = np.array(self._X)
        model = GaussianProcess(X, np.array([worst if y is None else y for y in self._y]), self._theta)
        if self._theta is None or len(X) <= FIT_EVERY_UNTIL or len(X) - self._fitted_at >= REFIT_EVERY:
            model.fit(restarts=2 if self._theta is None else 0, rng=self.rng)
            self._theta, self._fitted_at = model.theta, len(X)

        # The best value so far. With noise, the best observation is likely a lucky one, so use the
        # lowest the model predicts at any evaluated point instead.
        best = float(model.predict(X)[0].min()) if self.config.noise else float(model.z.min())
        # Candidates still being evaluated are assumed to come out as predicted, so none is proposed again.
        for individual in self._outstanding():
            x = self._encoding.encode(individual.params)
            model = model.with_observation(x, float(model.predict(x)[0][0]))
        chosen = []
        for _ in range(count):
            x = self._maximize_improvement(model, best)
            chosen.append(x)
            # Pretend the model's prediction came true, so the next candidate of the batch goes elsewhere.
            mean, _ = model.predict(x)
            model = model.with_observation(x, float(mean[0]))
        return chosen

    def _maximize_improvement(self, model: GaussianProcess, best: float) -> np.ndarray:
        encoding = self._encoding
        leaders = model.X[np.argsort(model.z)[:5]]
        local = leaders[self.rng.integers(len(leaders), size=LOCAL_CANDIDATES)]
        scales = self.rng.choice([0.01, 0.05, 0.2], size=(LOCAL_CANDIDATES, 1))
        local = encoding.perturb(self.rng, local, scales)
        candidates = encoding.snap(np.vstack([encoding.spread(self.rng, GLOBAL_CANDIDATES), local]))
        scores = self._only_feasible(candidates, _log_expected_improvement(model, candidates, best))

        columns = encoding.numeric
        if columns.size:
            starts = candidates[np.argsort(-scores)[:REFINED]]
            refined = encoding.snap(_refine(model, starts, columns, best))
            candidates = np.vstack([candidates, refined])
            scores = np.append(scores, self._only_feasible(refined, _log_expected_improvement(model, refined, best)))

        seen = model.X
        for index in np.argsort(-scores):
            if scores[index] == -np.inf and self.search_space.constraints:
                break  # the rest break a constraint
            if not np.any(np.all(np.abs(seen - candidates[index]) < 1e-9, axis=1)):
                return candidates[index]
        return self._feasible_spread(1)[0]  # every feasible candidate was already evaluated

    # --- constraints --------------------------------------------------------------

    def _feasible_spread(self, count: int) -> np.ndarray:
        """count points spread over the space, any that break a constraint replaced by random feasible points."""
        X = self._encoding.spread(self.rng, count)
        if self.search_space.constraints:
            for i, x in enumerate(X):
                if not self.search_space.feasible(self._encoding.decode(x)):
                    X[i] = self._encoding.encode(self.search_space.sample(self.rng))
        return X

    def _only_feasible(self, X: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """scores, with those of points that break a constraint set to minus infinity."""
        if not self.search_space.constraints:
            return scores
        feasible = np.array([self.search_space.feasible(self._encoding.decode(x)) for x in X], dtype=bool)
        return np.where(feasible, scores, -np.inf)


def _refine(model: GaussianProcess, starts: np.ndarray, columns: np.ndarray, best: float) -> np.ndarray:
    """Each start moved uphill in expected improvement by gradient search over the numeric columns.

    The gradient comes from finite differences, all taken in one vectorized call to the model.
    """
    step = 1e-6
    refined = starts.copy()
    for row, start in enumerate(starts):

        def negative_and_gradient(u: np.ndarray, start=start) -> tuple[float, np.ndarray]:
            X = np.repeat(start[None, :], len(columns) + 1, axis=0)
            X[:, columns] = u
            X[np.arange(1, len(columns) + 1), columns] += step
            values = _log_expected_improvement(model, X, best)
            return -values[0], -(values[1:] - values[0]) / step

        found = minimize(negative_and_gradient, start[columns], jac=True, method="L-BFGS-B", bounds=[(0, 1 - step)] * len(columns))
        if found.fun <= -_log_expected_improvement(model, start[None, :], best)[0]:
            refined[row, columns] = found.x
    return refined


def _log_expected_improvement(model: GaussianProcess, X: np.ndarray, best: float) -> np.ndarray:
    """log E[max(best - f(x), 0)] for each row of X, computed stably even where it is tiny."""
    mean, sd = model.predict(X)
    u = (best - mean) / sd
    log_h = np.empty_like(u)
    # h(u) = phi(u) + u Phi(u), the expected improvement of a standard normal past -u.
    upper = u > -1
    log_h[upper] = np.log(np.exp(-0.5 * u[upper] ** 2) / math.sqrt(2 * math.pi) + u[upper] * np.exp(log_ndtr(u[upper])))
    lower = np.maximum(u[~upper], -40.0)
    # For u <= -1: h(u) = phi(u) (1 + u Phi(u) / phi(u)), where Phi(u) / phi(u) = sqrt(pi/2) erfcx(-u / sqrt 2).
    log_h[~upper] = -0.5 * lower**2 - 0.5 * math.log(2 * math.pi) + np.log1p(
        lower * math.sqrt(math.pi / 2) * erfcx(-lower / math.sqrt(2))
    )
    return log_h + np.log(sd)


class _Encoding:
    """Between parameter values and points in [0, 1]^columns, where the model works."""

    def __init__(self, search_space: SearchSpace) -> None:
        self.names = search_space.names  # not the search space itself, whose function constraints may not pickle
        self._blocks: list[tuple[RangeParameter | ChoiceParameter, slice]] = []
        self._constant: dict[str, Any] = {}
        column = 0
        numeric = []
        for p in search_space.parameters:
            if isinstance(p, ChoiceParameter) and p.choose_n is not None:
                raise UnsupportedSearchSpace(
                    f"Bayesian optimization doesn't support choose_n choices ({p.name}); use GeneticOptimizer"
                )
            if isinstance(p, RangeParameter) and p.maximum > p.minimum:
                self._blocks.append((p, slice(column, column + 1)))
                numeric.append(column)
                column += 1
            elif isinstance(p, ChoiceParameter) and len(p.choices) > 1:
                self._blocks.append((p, slice(column, column + len(p.choices))))
                column += len(p.choices)
            else:
                self._constant[p.name] = p.sample(np.random.default_rng(0))  # fixed, or only one possible value
        if not self._blocks:
            raise UnsupportedSearchSpace("Bayesian optimization needs at least one parameter with more than one possible value")
        self.columns = column
        self.variables = len(self._blocks)
        self.numeric = np.array(numeric, dtype=int)

    def encode(self, params: dict) -> np.ndarray:
        x = np.zeros(self.columns)
        for p, block in self._blocks:
            if isinstance(p, RangeParameter):
                x[block] = p.to_unit(params[p.name])
            else:
                x[block.start + p.choices.index(params[p.name])] = 1.0
        return x

    def decode(self, x: np.ndarray) -> dict:
        values = dict(self._constant)
        for p, block in self._blocks:
            if isinstance(p, RangeParameter):
                values[p.name] = p.from_unit(float(x[block][0]))
            else:
                values[p.name] = p.choices[int(np.argmax(x[block]))]
        return {name: values[name] for name in self.names}

    def snap(self, X: np.ndarray) -> np.ndarray:
        """Each row moved to the nearest point that parameter values can take: whole integers and one choice per block."""
        X = np.clip(X, 0, 1)
        for p, block in self._blocks:
            if isinstance(p, RangeParameter) and p.integer:
                X[:, block] = p.to_unit(p.from_unit(X[:, block]))
            elif isinstance(p, ChoiceParameter):
                chosen = np.argmax(X[:, block], axis=1)
                X[:, block] = 0.0
                X[np.arange(len(X)), block.start + chosen] = 1.0
        return X

    def spread(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """count points spread over the space: a Latin hypercube for numbers, random choices otherwise."""
        X = np.zeros((count, self.columns))
        for p, block in self._blocks:
            if isinstance(p, RangeParameter):
                X[:, block.start] = (rng.permutation(count) + rng.uniform(size=count)) / max(count, 1)
            else:
                X[np.arange(count), block.start + rng.integers(len(p.choices), size=count)] = 1.0
        return self.snap(X)

    def perturb(self, rng: np.random.Generator, X: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """X nudged: numbers by Gaussian steps of the given scale, and each choice switched with probability 0.2."""
        X = X.copy()
        if self.numeric.size:
            X[:, self.numeric] += rng.normal(size=(len(X), self.numeric.size)) * scales
        for p, block in self._blocks:
            if isinstance(p, ChoiceParameter):
                switch = rng.uniform(size=len(X)) < 0.2
                X[switch, block] = 0.0
                X[np.flatnonzero(switch), block.start + rng.integers(len(p.choices), size=switch.sum())] = 1.0
        return X
