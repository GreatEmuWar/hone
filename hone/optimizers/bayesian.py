"""Bayesian optimization: model the objective with a Gaussian process, and evaluate where the model expects the most improvement.

Each candidate costs a model fit and a search over the model, a fraction of a second up to a few
hundred evaluations, so Bayesian optimization suits objectives that take far longer than that:
simulations, experiments, training runs. It makes the most of a small budget.

Range parameters are scaled to [0, 1], on a log scale where they have one; integers are searched
as numbers and rounded. Each choice
parameter becomes one column per choice, holding 1 for the chosen one and 0 for the rest.
A parameter a condition switches off is encoded the same way whatever value it holds: a range at
INACTIVE, and a choice with every column 0. So points that differ only where parameters are
switched off are the same point to the model, and the values they happen to hold add no noise.

With constraints on outputs, a Gaussian process models each one's margin, which is zero or less
where it is satisfied. The acquisition is multiplied by the probability that every one holds,
so the search learns where the infeasible region is from the evaluations that broke a limit
(Gardner et al., 2014; Gelbart, Snoek and Adams, 2014).
"""
import math
from typing import Any, Callable, List

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfcx, log_ndtr
from threadpoolctl import threadpool_limits

from hone.config import BayesianConfig
from hone.optimizers._gp import GaussianProcess
from hone.optimizers.base import Individual, Optimizer, UnsupportedSearchSpace
from hone.parameters import ChoiceParameter, Parameter, RangeParameter, SearchSpace
from hone.parameters.conditions import ChoiceTest, ConditionalConstraint, holds

GLOBAL_CANDIDATES = 1000  # random candidates per search of the model
LOCAL_CANDIDATES = 500  # candidates near the best points found so far
REFINED = 5  # the best candidates are refined by gradient search
REFIT_EVERY = 5  # past FIT_EVERY_UNTIL observations, refit the model's hyperparameters only this often
FIT_EVERY_UNTIL = 50
INACTIVE = 0.5  # where a range switched off by a condition is encoded
UNCOMPUTED_MARGIN = 1.0  # the least relative margin, 100% over, modelled where a constraint couldn't be computed


class _ModelSearch:
    """Searching an acquisition function for the next candidate, over the encoded space, keeping to constraints.

    Shared by single- and multi-objective Bayesian optimization, which also share the models of
    constraints on outputs: _absorb_outputs records each evaluation's margins, and
    _constraint_models and _log_feasibility turn them into the probability that a point
    satisfies every one.
    """
    search_space: SearchSpace
    rng: np.random.Generator
    _encoding: "_Encoding"

    def _reset_outputs(self) -> None:
        self._margins: list[list[float | None] | None] = []  # each evaluated point's margins, or None if it failed
        self._violations: list[float] = []  # each evaluated point's total violation; infinity if it failed
        self._constraint_thetas: dict[int, np.ndarray] = {}  # each constraint model's hyperparameters, kept between fits
        self._constraint_fitted_at: dict[int, int] = {}

    def _absorb_outputs(self, individual: Individual) -> None:
        """Record an evaluated individual's margins and violation, alongside its row of X."""
        if individual.failed or individual.outputs is None:
            self._margins.append(None)
        else:
            self._margins.append(self.search_space.output_margins(individual.params, individual.outputs))
        self._violations.append(math.inf if individual.failed else individual.violation)

    def _constraint_models(self, X: np.ndarray, pending: list[np.ndarray]) -> list[tuple[int, GaussianProcess]]:
        """A Gaussian process of each output constraint's margin, by the constraint's position, with pending points believed.

        Each is fitted to the points where its constraint applies, to its relative margin
        through the bilog transform, sign(m) log(1 + |m|) (Eriksson and Poloczek, 2021), which
        keeps the sign, so the limit stays at zero, and tames the far tails that stresses and
        the like have near a design's edges. A margin that couldn't be computed counts as the
        largest seen, and at least UNCOMPUTED_MARGIN, so it is never taken for feasible. A constraint with fewer than two such points has no model yet.
        """
        models = []
        for j in range(len(self.search_space.output_constraints)):
            rows = [i for i, margins in enumerate(self._margins) if margins is not None and margins[j] is not None]
            if len(rows) < 2:
                continue
            values = np.array([_margin(self._margins[i], j) for i in rows], dtype=np.float64)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            # A margin that couldn't be computed counts as infeasible: at least 100% over, or the worst seen.
            values[~finite] = max(values[finite].max(), UNCOMPUTED_MARGIN)
            values = _bilog(values)
            theta = self._constraint_thetas.get(j)
            model = GaussianProcess(X[rows], values, theta)
            if theta is None or len(rows) <= FIT_EVERY_UNTIL or len(rows) - self._constraint_fitted_at.get(j, 0) >= REFIT_EVERY:
                model.fit(restarts=2 if theta is None else 0, rng=self.rng)
                self._constraint_thetas[j], self._constraint_fitted_at[j] = model.theta, len(rows)
            for x in pending:
                model = _believed(model, x)
            models.append((j, model))
        return models

    def _log_feasibility(self, models: list[tuple[int, GaussianProcess]]) -> Callable[[np.ndarray], np.ndarray]:
        """log P(every output constraint holds) at each row of X, by the constraints' models, treated as independent.

        A conditional constraint counts only at points where it applies.
        """
        constraints = self.search_space.output_constraints
        conditional = self.search_space.conditions or any(isinstance(c, ConditionalConstraint) for c in constraints)

        def log_probability(X: np.ndarray) -> np.ndarray:
            total = np.zeros(len(X))
            params = [self._encoding.decode(x) for x in X] if conditional and models else []
            for j, model in models:
                mean, sd = model.predict(X)
                log_p = log_ndtr((float(model.standardize(0.0)) - mean) / sd)
                if conditional:
                    applies = np.array([constraints[j].applies(self.search_space.active(p)) for p in params], dtype=bool)
                    log_p = np.where(applies, log_p, 0.0)
                total += log_p
            return total

        return log_probability

    def _feasible_leaders(self, X: np.ndarray, scores: np.ndarray, count: int = 5) -> np.ndarray:
        """The count best rows of X: feasible points by score, lowest first, then the rest by violation."""
        violations = np.array(self._violations) if self._violations else np.zeros(len(X))
        order = np.lexsort((scores, violations))
        return X[order[:count]]

    def _search(self, acquisition: Callable[[np.ndarray], np.ndarray], leaders: np.ndarray, seen: np.ndarray) -> np.ndarray:
        """The candidate with the highest acquisition not yet evaluated: from random points, points near the leaders, and the best of those refined."""
        encoding = self._encoding
        local = leaders[self.rng.integers(len(leaders), size=LOCAL_CANDIDATES)]
        scales = self.rng.choice([0.01, 0.05, 0.2], size=(LOCAL_CANDIDATES, 1))
        local = encoding.perturb(self.rng, local, scales)
        candidates = encoding.snap(np.vstack([encoding.spread(self.rng, GLOBAL_CANDIDATES), local]))
        scores = self._only_feasible(candidates, acquisition(candidates))

        columns = encoding.numeric
        if columns.size:
            starts = candidates[np.argsort(-scores)[:REFINED]]
            refined = encoding.snap(_refine(acquisition, starts, columns))
            candidates = np.vstack([candidates, refined])
            scores = np.append(scores, self._only_feasible(refined, acquisition(refined)))

        for index in np.argsort(-scores):
            if scores[index] == -np.inf and self.search_space.parameter_constraints:
                break  # the rest break a constraint
            if not np.any(np.all(np.abs(seen - candidates[index]) < 1e-9, axis=1)):
                return candidates[index]
        return self._feasible_spread(1)[0]  # every feasible candidate was already evaluated

    # --- constraints --------------------------------------------------------------

    def _feasible_spread(self, count: int) -> np.ndarray:
        """count points spread over the space, any that break a constraint replaced by random feasible points."""
        X = self._encoding.spread(self.rng, count)
        if self.search_space.parameter_constraints:
            for i, x in enumerate(X):
                if not self.search_space.feasible(self._encoding.decode(x)):
                    X[i] = self._encoding.encode(self.search_space.sample(self.rng))
        return X

    def _only_feasible(self, X: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """scores, with those of points that break a constraint on the parameters set to minus infinity."""
        if not self.search_space.parameter_constraints:
            return scores
        feasible = np.array([self.search_space.feasible(self._encoding.decode(x)) for x in X], dtype=bool)
        return np.where(feasible, scores, -np.inf)


class BayesianOptimizer(Optimizer, _ModelSearch):
    """Bayesian optimization with a Gaussian process and expected improvement.

    The first generation is spread over the search space (with any initial points). After that,
    each candidate is the point where a Gaussian process model of the objective predicts the
    largest expected improvement on the best value so far. Failed evaluations are modelled as
    the worst value seen, which steers the search away from them.

    With constraints on outputs, each one's margin gets a Gaussian process of its own, and the
    expected improvement on the best feasible value is multiplied by the probability that every
    constraint holds. Until a feasible point is found, candidates maximize that probability.

    Range parameters, integer or not, and choice parameters are supported, but not choice
    parameters with choose_n. Fixed parameters are passed through.

    Args:
        search_space: The parameters to optimize over, as a SearchSpace or a list of parameters.
        config: A BayesianConfig. Defaults are used if omitted.
        initial_points: Points to evaluate first: one dict of parameter values, or a list of them.
    """
    config_class = BayesianConfig
    supports_constraints = True
    supports_conditions = True
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
        self._reset_outputs()

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
                self._absorb_outputs(individual)
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

        # Candidates still being evaluated are assumed to come out as predicted, so none is proposed again.
        pending = [self._encoding.encode(individual.params) for individual in self._outstanding()]
        constraints = self._constraint_models(X, pending) if self.search_space.output_constraints else []
        feasible = np.array(self._violations) == 0 if self.search_space.output_constraints else np.ones(len(X), dtype=bool)
        best: float | None
        if self.config.noise:
            # The best value so far. With noise, the best observation is likely a lucky one, so use the
            # lowest the model predicts at any evaluated point instead.
            best = float(model.predict(X)[0].min())
        elif feasible.any():
            best = float(model.z[feasible].min())  # the best value of a point that satisfies every output constraint
        else:
            best = None if constraints else float(model.z.min())
        # With output constraints, search near the best observed points that satisfy them.
        leaders = self._feasible_leaders(X, model.z) if self.search_space.output_constraints else None
        for x in pending:
            model = _believed(model, x)
        chosen = []
        for _ in range(count):
            x = self._maximize_improvement(model, best, constraints, leaders)
            chosen.append(x)
            # Pretend the models' predictions came true, so the next candidate of the batch goes elsewhere.
            model = _believed(model, x)
            constraints = [(j, _believed(m, x)) for j, m in constraints]
        return chosen

    def _maximize_improvement(
        self, model: GaussianProcess, best: float | None, constraints: list[tuple[int, GaussianProcess]], leaders: np.ndarray | None
    ) -> np.ndarray:
        """The candidate with the largest expected improvement on best, times the probability it satisfies the output constraints.

        With no feasible point yet, best is None, and the candidate is the one likeliest to be feasible.
        Without leaders, the search starts near the lowest values the model holds, believed ones included.
        """
        if leaders is None:
            leaders = model.X[np.argsort(model.z)[:5]]
        if not constraints:
            assert best is not None
            return self._search(lambda X: _log_expected_improvement(model, X, best), leaders, model.X)
        log_feasibility = self._log_feasibility(constraints)
        if best is None:
            return self._search(log_feasibility, leaders, model.X)
        return self._search(lambda X: _log_expected_improvement(model, X, best) + log_feasibility(X), leaders, model.X)


def _bilog(values: np.ndarray) -> np.ndarray:
    """sign(v) log(1 + |v|): the same sign, the same near zero, and logarithmic far from it."""
    return np.sign(values) * np.log1p(np.abs(values))


def _margin(margins: list[float | None] | None, j: int) -> float:
    assert margins is not None and margins[j] is not None
    return float(margins[j])  # type: ignore[arg-type]


def _believed(model: GaussianProcess, x: np.ndarray) -> GaussianProcess:
    """model with an observation at x of what it predicts there: the kriging believer, for points not yet evaluated."""
    return model.with_observation(x, float(model.predict(x)[0][0]))


def _refine(acquisition: Callable[[np.ndarray], np.ndarray], starts: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Each start moved uphill in the acquisition by gradient search over the numeric columns.

    The gradient comes from finite differences, all taken in one vectorized call.
    """
    step = 1e-6
    refined = starts.copy()
    for row, start in enumerate(starts):

        def negative_and_gradient(u: np.ndarray, start=start) -> tuple[float, np.ndarray]:
            X = np.repeat(start[None, :], len(columns) + 1, axis=0)
            X[:, columns] = u
            X[np.arange(1, len(columns) + 1), columns] += step
            values = acquisition(X)
            return -values[0], -(values[1:] - values[0]) / step

        found = minimize(negative_and_gradient, start[columns], jac=True, method="L-BFGS-B", bounds=[(0, 1 - step)] * len(columns))
        if found.fun <= -acquisition(start[None, :])[0]:
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
        # The conditions alone, to tell which parameters a point switches off.
        self._conditions = SearchSpace(search_space.parameters, conditions=search_space.conditions) if search_space.conditions else None
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
        on = self._conditions.active(params) if self._conditions is not None else params
        x = np.zeros(self.columns)
        for p, block in self._blocks:
            if p.name not in on:
                x[block] = INACTIVE if isinstance(p, RangeParameter) else 0.0
            elif isinstance(p, RangeParameter):
                x[block] = p.to_unit(params[p.name])
            else:
                x[block.start + p.choices.index(params[p.name])] = 1.0
        return x

    def _switch_off(self, X: np.ndarray) -> np.ndarray:
        """X with the columns of parameters each row switches off set as encode() sets them."""
        if self._conditions is None:
            return X
        blocks = {p.name: (p, block) for p, block in self._blocks}
        on: dict[str, np.ndarray] = {}
        for parameter in self._conditions._order:
            test = self._conditions._tests.get(parameter.name)
            if test is None:
                on[parameter.name] = np.ones(len(X), dtype=bool)
                continue
            mask = np.logical_and.reduce([on[name] for name in test.names()])
            if isinstance(test, ChoiceTest):
                if test.name in blocks:
                    p, block = blocks[test.name]
                    assert isinstance(p, ChoiceParameter)
                    mask &= X[:, [block.start + p.choices.index(value) for value in test.values]].max(axis=1) > 0.5
            else:
                for row in np.flatnonzero(mask):
                    mask[row] = holds(test, self.decode(X[row]))
            on[parameter.name] = mask
        for p, block in self._blocks:
            X[~on[p.name], block] = INACTIVE if isinstance(p, RangeParameter) else 0.0
        return X

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
        return self._switch_off(X)

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
