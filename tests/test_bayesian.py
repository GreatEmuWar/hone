"""Bayesian optimization: the model, the search for improvement, and the optimizer around them."""
import math

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import norm

from hone import BayesianConfig, BayesianOptimizer, RandomSearch, UnsupportedSearchSpace
from hone.config import OptimizerConfig
from hone.optimizers import bayesian
from hone.optimizers._gp import GaussianProcess
from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter, SearchSpace

import objectives

CONTINUOUS = [RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)]


def make(space=CONTINUOUS, initial_points=None, **config):
    config.setdefault("seed", 0)
    return BayesianOptimizer(space, BayesianConfig(**config), initial_points=initial_points)


def shifted(x, y):
    return (x - 1.5) ** 2 + (y + 2) ** 2


# --- the model ------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(25, 3))
    return X, np.sin(5 * X[:, 0]) + X[:, 1] ** 2


def test_likelihood_gradient_matches_finite_differences(data):
    model = GaussianProcess(*data)
    theta = np.array([-1.0, 0.2, -0.5, 0.3, -6.0])
    numeric = approx_fprime(theta, lambda t: model._objective(t)[0], 1e-6)
    assert np.allclose(model._objective(theta)[1], numeric, rtol=1e-4, atol=1e-4)


def test_fitted_model_predicts_unseen_points_and_learns_which_inputs_matter(data):
    model = GaussianProcess(*data).fit(restarts=2, rng=np.random.default_rng(0))
    X = np.random.default_rng(2).uniform(size=(100, 3))
    mean, _ = model.predict(X)
    error = mean * model._scale + model._offset - (np.sin(5 * X[:, 0]) + X[:, 1] ** 2)
    assert np.sqrt(np.mean(error**2)) < 0.1
    length = np.exp(model.theta[:3])
    assert length[2] > 3 * max(length[0], length[1]), "the unused third input gets a long length scale"


def test_model_is_certain_at_its_data_and_unsure_far_from_it(data):
    X, y = data
    model = GaussianProcess(X, y).fit(rng=np.random.default_rng(0))
    mean, sd = model.predict(X)
    assert np.allclose(mean, model.standardize(y), atol=0.05)
    assert sd.max() < 0.1 < model.predict(np.full((1, 3), 3.0))[1][0]


def test_added_observation_is_believed(data):
    model = GaussianProcess(*data)
    x = np.array([0.5, 0.5, 0.5])
    assert model.with_observation(x, 2.0).predict(x)[0][0] == pytest.approx(2.0, abs=0.05)


def test_constant_outputs_dont_break_the_model():
    model = GaussianProcess(np.random.default_rng(0).uniform(size=(5, 2)), np.ones(5))
    mean, sd = model.predict(np.full((1, 2), 0.5))
    assert np.isfinite(mean).all() and np.isfinite(sd).all()


# --- expected improvement -------------------------------------------------


class Fixed:
    """A stand-in model with given predictions."""

    def __init__(self, mean, sd):
        self.mean, self.sd = np.asarray(mean, float), np.asarray(sd, float)

    def predict(self, X):
        return self.mean, self.sd


def test_log_expected_improvement_matches_the_formula():
    mean, sd, best = np.array([0.0, 1.0, -1.0, 2.0]), np.array([1.0, 0.5, 2.0, 0.3]), 0.5
    u = (best - mean) / sd
    expected = sd * (u * norm.cdf(u) + norm.pdf(u))
    assert np.allclose(bayesian._log_expected_improvement(Fixed(mean, sd), None, best), np.log(expected))


def test_log_expected_improvement_stays_finite_and_ordered_far_from_the_best():
    mean = np.array([5.0, 10.0, 50.0, 1e4])
    values = bayesian._log_expected_improvement(Fixed(mean, np.ones(4)), None, 0.0)
    assert np.isfinite(values).all() and list(values) == sorted(values, reverse=True)


# --- encoding -------------------------------------------------------------

MIXED = SearchSpace(
    [
        RangeParameter("x", -5, 5),
        RangeParameter("n", 0, 10, integer=True),
        ChoiceParameter("material", ["steel", "aluminium", "titanium"]),
        FixedParameter("mode", "fast"),
        RangeParameter("same", 2, 2),
    ]
)


def test_encoding_round_trips():
    encoding = bayesian._Encoding(MIXED)
    params = {"x": 1.25, "n": 7, "material": "titanium", "mode": "fast", "same": 2.0}
    x = encoding.encode(params)
    assert encoding.columns == 5 and list(x[2:]) == [0, 0, 1]
    assert encoding.decode(x) == params


def test_snap_moves_points_onto_values_parameters_can_take():
    encoding = bayesian._Encoding(MIXED)
    snapped = encoding.snap(np.array([[1.2, 0.33, 0.2, 0.7, 0.1]]))[0]
    assert snapped[0] == 1.0 and snapped[1] == pytest.approx(0.3) and list(snapped[2:]) == [0, 1, 0]


def test_spread_puts_one_point_in_each_slice_of_every_range():
    X = bayesian._Encoding(MIXED).spread(np.random.default_rng(0), 10)
    assert sorted(np.floor(X[:, 0] * 10).astype(int)) == list(range(10))
    assert (X[:, 2:].sum(1) == 1).all()


def test_unsupported_spaces_are_rejected():
    with pytest.raises(UnsupportedSearchSpace, match="choose_n"):
        make([RangeParameter("x", 0, 1), ChoiceParameter("tags", ["a", "b", "c"], choose_n=2)])
    with pytest.raises(UnsupportedSearchSpace, match="more than one possible value"):
        make([FixedParameter("x", 1), RangeParameter("y", 2, 2)])


# --- the optimizer --------------------------------------------------------


def test_finds_the_minimum_of_a_smooth_function_in_few_evaluations():
    result = make(max_evaluations=30).optimize(shifted)
    assert result.best_fitness < 0.05


def test_beats_random_search_on_a_small_budget():
    errors = {"bayesian": [], "random": []}
    for seed in range(3):
        errors["bayesian"].append(make(max_evaluations=40, seed=seed).optimize(shifted).best_fitness)
        random = RandomSearch(CONTINUOUS, OptimizerConfig(population_size=10, max_evaluations=40, seed=seed))
        errors["random"].append(random.optimize(shifted).best_fitness)
    assert max(errors["bayesian"]) < min(errors["random"])


def test_maximizes():
    result = make(max_evaluations=30, maximize=True).optimize(lambda x, y: -shifted(x, y))
    assert result.best_fitness > -0.05


def test_handles_mixed_parameters():
    def cost(x, n, material, mode, same):
        return (x - 1) ** 2 + (n - 4) ** 2 + {"steel": 3, "aluminium": 1, "titanium": 0}[material]

    result = make(MIXED, max_evaluations=40).optimize(cost)
    assert result.best_params["material"] == "titanium" and result.best_params["n"] == 4
    assert type(result.best_params["n"]) is int and result.best_params["mode"] == "fast"


def test_never_repeats_a_point_while_there_are_new_ones():
    space = [RangeParameter("a", 0, 2, integer=True), RangeParameter("b", 0, 2, integer=True)]
    optimizer = make(space, max_evaluations=9)
    seen = []
    while not optimizer.done:
        for trial in optimizer.ask():
            seen.append((trial.params["a"], trial.params["b"]))
            optimizer.tell(trial, trial.params["a"] + trial.params["b"])
    assert len(set(seen)) == 9


def test_steers_away_from_failures():
    result = make(max_evaluations=40).optimize(objectives.fails_right_of_zero)
    assert result.failures > 0 and result.best_params["x"] <= 0 and result.best_fitness < 0.1


def test_starts_from_initial_points():
    optimizer = make(max_evaluations=20, initial_points={"x": 1.5, "y": -2.0})
    assert optimizer.ask()[0].params == {"x": 1.5, "y": -2.0}


def test_first_generation_is_spread_and_later_ones_are_single_candidates():
    optimizer = make(max_evaluations=20)
    first = optimizer.ask()
    assert len(first) == 6, "2 × (2 variables + 1)"
    for trial in first:
        optimizer.tell(trial, shifted(**trial.params))
    assert len(optimizer.ask()) == 1


def test_batch_size_follows_the_config_or_the_workers():
    assert make(population_size=3).batch_size == 3
    optimizer = make(max_evaluations=12, initial_samples=4)
    result = optimizer.optimize(objectives.sphere, workers=4)
    assert [g.evaluations for g in result.generations] == [4, 8, 12], "batches of 4 after 4 spread-out candidates"


def test_is_reproducible():
    assert make(max_evaluations=15).optimize(shifted) == make(max_evaluations=15).optimize(shifted)


def test_budget_sets_the_generations():
    assert BayesianConfig(max_evaluations=37).generations == 37
    assert BayesianConfig(max_evaluations=37, generations=5).generations == 5
    assert BayesianConfig().generations == 100


def test_all_failures_at_first_still_moves_on():
    optimizer = make(max_evaluations=10, initial_samples=3)
    for trial in optimizer.ask():
        optimizer.tell(trial, failed=True)
    assert len(optimizer.ask()) == 1


def test_refinement_improves_on_its_starting_points():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 2))
    model = GaussianProcess(X, ((X - 0.3) ** 2).sum(1)).fit(rng=rng)
    starts = np.array([[0.9, 0.9], [0.1, 0.8]])
    best = float(model.z.min())
    refined = bayesian._refine(lambda C: bayesian._log_expected_improvement(model, C, best), starts, np.array([0, 1]))
    before = bayesian._log_expected_improvement(model, starts, best)
    after = bayesian._log_expected_improvement(model, refined, best)
    assert (after >= before).all() and (after > before).any() and math.isfinite(after.max())
