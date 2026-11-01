"""Bayesian optimization of several objectives: Gaussian process models, and a random weighting of the objectives for each candidate."""
from typing import Any, Callable, List

import numpy as np
from threadpoolctl import threadpool_limits

from hone.config.multi_objective_config import MultiObjectiveBayesianConfig
from hone.parameters import Parameter, SearchSpace
from hone.pareto import AUGMENTATION, _minimizing

from ._gp import GaussianProcess
from .bayesian import FIT_EVERY_UNTIL, REFIT_EVERY, _Encoding, _log_expected_improvement, _ModelSearch
from .core import Individual
from .multi import MultiObjectiveOptimizer

SAMPLES = 64  # draws from the models' predictions per candidate, for the expected improvement of the weighted objectives


class MultiObjectiveBayesian(MultiObjectiveOptimizer, _ModelSearch):
    """Bayesian optimization for several objectives, returning the front of best trade-offs.

    The first generation is spread over the search space, as in BayesianOptimizer. Each later
    candidate aims at a different part of the front: it draws random weights for the objectives,
    each scaled by the range seen so far, and goes where the model expects the most improvement
    in their augmented Chebyshev combination, the largest weighted gap from the best values plus
    a twentieth of their sum. Unlike a weighted sum, this reaches every part of the front,
    including any dent in it. With method "parego" one Gaussian process models the combination
    (ParEGO, Knowles 2006); with "models" one models each objective, and the combination is
    applied to their predictions.

    Range and choice parameters are supported, as in BayesianOptimizer, with constraints and
    conditions. Each candidate takes a fraction of a second or more to choose, so it suits
    expensive objectives and budgets of a few hundred evaluations.
    """
    config_class = MultiObjectiveBayesianConfig
    supports_constraints = True
    supports_conditions = True
    config: MultiObjectiveBayesianConfig

    def __init__(
        self,
        search_space: SearchSpace | List[Parameter],
        config: MultiObjectiveBayesianConfig | None = None,
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
        self._Y: list[np.ndarray | None] = []  # its values to minimize, or None if it failed
        self._thetas: dict[int, np.ndarray] = {}  # each model's hyperparameters, kept between fits
        self._fitted_at: dict[int, int] = {}  # observations at each model's last hyperparameter fit

    def _next_population(self, previous: list[Individual] | None, generation: int) -> list[Individual]:
        if previous is None:
            initial = self._initial_individuals(generation)
            count = self.config.initial_samples or 2 * (self._encoding.variables + 1)
            taken = {tuple(self._encoding.encode(ind.params)) for ind in initial}
            spread = []
            for x in self._feasible_spread(max(count - len(initial), 0)):
                for _ in range(100):  # rounding integers can land two points on the same values; try others instead
                    if tuple(x) not in taken:
                        break
                    x = self._feasible_spread(1)[0]
                taken.add(tuple(x))
                spread.append(x)
            return initial + [Individual(params=self._encoding.decode(x), generation=generation) for x in spread]
        for individual in previous:
            if individual.evaluated:
                self._X.append(self._encoding.encode(individual.params))
                self._Y.append(None if individual.failed else _minimizing(individual.values or {}, self._objectives))
        with threadpool_limits(limits=1, user_api="blas"):
            proposals = self._propose(self.batch_size)
        return [Individual(params=self._encoding.decode(x), generation=generation) for x in proposals]

    # --- choosing candidates --------------------------------------------------

    def _propose(self, count: int) -> list[np.ndarray]:
        succeeded = [y for y in self._Y if y is not None]
        if len(succeeded) < 2:
            return list(self._feasible_spread(count))
        X = np.array(self._X)
        worst = np.max(succeeded, axis=0)
        Y = np.array([worst if y is None else y for y in self._Y])  # a failure counts as the worst of every objective
        low, high = Y.min(axis=0), Y.max(axis=0)
        scaled = (Y - low) / np.where(high > low, high - low, 1.0)
        pending = [self._encoding.encode(ind.params) for ind in self._outstanding()]
        chosen: list[np.ndarray] = []
        for _ in range(count):
            weights = self.rng.dirichlet(np.ones(len(self._objectives)))
            combined = _chebyshev(scaled, weights)
            leaders = X[np.argsort(combined, kind="stable")[:5]]
            seen = np.vstack([X, *pending, *chosen]) if pending or chosen else X
            if self.config.method == "parego":
                model = self._model(0, X, combined, [*pending, *chosen])
                x = self._search(_improvement(model), leaders, seen)
            else:
                models = [self._model(j, X, scaled[:, j], [*pending, *chosen]) for j in range(scaled.shape[1])]
                draws = self.rng.standard_normal((SAMPLES, len(models)))
                x = self._search(_sampled_improvement(models, weights, float(combined.min()), draws), leaders, seen)
            chosen.append(x)
        return chosen

    def _model(self, index: int, X: np.ndarray, y: np.ndarray, pending: list[np.ndarray]) -> GaussianProcess:
        """A Gaussian process of y at X, with pending candidates assumed to come out as it predicts.

        Its hyperparameters are fitted as in BayesianOptimizer: every time at first, then every
        few observations, starting from the last fit.
        """
        theta = self._thetas.get(index)
        model = GaussianProcess(X, y, theta)
        if theta is None or len(X) <= FIT_EVERY_UNTIL or len(X) - self._fitted_at.get(index, 0) >= REFIT_EVERY:
            model.fit(restarts=2 if theta is None else 0, rng=self.rng)
            self._thetas[index], self._fitted_at[index] = model.theta, len(X)
        for x in pending:
            model = model.with_observation(x, float(model.predict(x)[0][0]))
        return model


def _improvement(model: GaussianProcess) -> Callable[[np.ndarray], np.ndarray]:
    """The log expected improvement of model on the best value it has seen."""
    best = float(model.z.min())
    return lambda C: _log_expected_improvement(model, C, best)


def _chebyshev(scaled: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The augmented Chebyshev combination of each row of scaled objectives: the largest weighted value, plus a twentieth of their sum."""
    weighted = scaled * weights
    return weighted.max(axis=-1) + AUGMENTATION * weighted.sum(axis=-1)


def _sampled_improvement(models: list[GaussianProcess], weights: np.ndarray, best: float, draws: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """The log expected improvement on best of the weighted combination, estimated from draws of each model's prediction.

    The same draws serve every candidate, so the estimate changes smoothly from point to point
    and gradient search can follow it.
    """

    def acquisition(C: np.ndarray) -> np.ndarray:
        predictions = [model.predict_original(C) for model in models]
        means = np.stack([mean for mean, _ in predictions], axis=-1)  # candidates × objectives
        sds = np.stack([sd for _, sd in predictions], axis=-1)
        samples = means[None] + sds[None] * draws[:, None, :]  # draws × candidates × objectives
        improvement = np.maximum(best - _chebyshev(samples, weights), 0.0).mean(axis=0)
        return np.log(improvement + 1e-300)

    return acquisition
