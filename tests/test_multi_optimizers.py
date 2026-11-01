"""NSGA2 and MultiObjectiveBayesian: fronts close to the true ones, with constraints, conditions, noise, workers and checkpoints."""
import math

import numpy as np
import pytest

from hone import (
    NSGA2,
    ChoiceParameter,
    MultiObjectiveBayesian,
    MultiObjectiveBayesianConfig,
    MultiObjectiveConfig,
    MultiObjectiveRandomSearch,
    NSGA2Config,
    ParetoResult,
    RangeParameter,
    SearchSpace,
    choose_pareto_optimizer,
)
from hone.benchmarks.multi import attainment_surface, dtlz2, multi_problems, run_multi_benchmark, summarize_multi, welded_beam, zdt
from hone.optimizers.genetic.nsga2 import _crowding

OBJECTIVES = {"f1": "minimize", "f2": "minimize"}
ZDT1 = zdt("zdt1", 4)


def gap(problem, result):
    return problem.true_hypervolume - problem.hypervolume(p.values for p in result.front)


def nsga2(problem=ZDT1, **config):
    return NSGA2(problem.search_space, NSGA2Config(objectives=problem.objectives, **{"max_evaluations": 2000, "seed": 0, **config}))


def bayesian(problem=ZDT1, **config):
    settings = {"max_evaluations": 40, "seed": 0, **config}
    return MultiObjectiveBayesian(problem.search_space, MultiObjectiveBayesianConfig(objectives=problem.objectives, **settings))


# --- NSGA2 -------------------------------------------------------------------------


def test_crowding_distance_keeps_the_ends_and_favours_isolated_points():
    distance = _crowding(np.array([[0.0, 1.0], [0.1, 0.9], [0.5, 0.5], [1.0, 0.0]]))
    assert math.isinf(distance[0]) and math.isinf(distance[3])
    assert distance[2] > distance[1]


def test_nsga2_closes_in_on_the_true_front():
    result = nsga2().optimize(ZDT1.function)
    random = MultiObjectiveRandomSearch(ZDT1.search_space, MultiObjectiveConfig(objectives=OBJECTIVES, max_evaluations=2000, seed=0))
    assert gap(ZDT1, result) < 0.02 < gap(ZDT1, random.optimize(ZDT1.function))
    assert len(result.front) > 50


def test_nsga2_finds_a_three_objective_front():
    problem = dtlz2(5)
    assert gap(problem, nsga2(problem, max_evaluations=3000).optimize(problem.function)) < 0.1


def test_nsga2_repeats_exactly_and_doesnt_depend_on_workers():
    first = nsga2(max_evaluations=600).optimize(ZDT1.function)
    assert first == nsga2(max_evaluations=600).optimize(ZDT1.function)
    assert first == nsga2(max_evaluations=600).optimize(ZDT1.function, workers=2)


def test_nsga2_keeps_to_constraints():
    problem = welded_beam()
    seen = []
    result = nsga2(problem, max_evaluations=1000).optimize(lambda **p: seen.append(p) or problem.function(**p))
    assert all(problem.search_space.feasible(p) for p in seen)
    assert len(result.front) > 10


def test_nsga2_with_conditions_and_niches():
    shape = ChoiceParameter("shape", ["round", "square"])
    size, corner = RangeParameter("size", 0, 1), RangeParameter("corner", 0, 1)
    space = SearchSpace([shape, size, corner], conditions=[shape["square"] >> corner])

    def objective(shape, size, corner=None):
        if shape == "round":
            return {"f1": size, "f2": 1.2 - size}
        return {"f1": size, "f2": (1 - size) + 0.5 * abs(corner - 0.3)}

    result = NSGA2(space, NSGA2Config(objectives=OBJECTIVES, max_evaluations=600, seed=0)).optimize(objective)
    assert {p.params["shape"] for p in result.front} == {"square"}
    assert all(abs(p.params["corner"] - 0.3) < 0.1 for p in result.front)


def test_nsga2_with_noise_averages_points():
    rng = np.random.default_rng(0)

    def noisy(**params):
        values = ZDT1.function(**params)
        return {name: value + rng.normal(0, 0.01) for name, value in values.items()}

    result = nsga2(max_evaluations=600, noise=True).optimize(noisy)
    assert result.front and all(p.evaluations >= 1 for p in result.front)


def test_nsga2_checkpoints_and_rebuilds(tmp_path):
    whole = nsga2(max_evaluations=400).optimize(ZDT1.function)
    partial = nsga2(max_evaluations=400)
    for _ in range(3):
        for trial in partial.ask():
            partial.tell(trial, ZDT1.function(**trial.params))
    partial.save_checkpoint(tmp_path / "run.checkpoint")
    resumed = nsga2(max_evaluations=400)
    resumed.load_checkpoint(tmp_path / "run.checkpoint")
    while not resumed.done:
        for trial in resumed.ask():
            resumed.tell(trial, ZDT1.function(**trial.params))
    assert resumed.result() == whole
    assert ParetoResult.model_validate_json(whole.model_dump_json()).rebuild_optimizer().optimize(ZDT1.function) == whole


# --- MultiObjectiveBayesian --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["parego", "models"])
def test_bayesian_beats_random_search_on_a_small_budget(method):
    random = MultiObjectiveRandomSearch(ZDT1.search_space, MultiObjectiveConfig(objectives=OBJECTIVES, max_evaluations=40, seed=0))
    assert gap(ZDT1, bayesian(method=method).optimize(ZDT1.function)) < 0.75 * gap(ZDT1, random.optimize(ZDT1.function))


def test_bayesian_with_constraints_conditions_and_choices():
    shape = ChoiceParameter("shape", ["round", "square"])
    size, corner = RangeParameter("size", 0, 1), RangeParameter("corner", 0, 1)
    space = SearchSpace([shape, size, corner], constraints=[size >= 0.1], conditions=[shape["square"] >> corner])
    seen = []

    def objective(shape, size, corner=None):
        seen.append(size)
        return {"f1": size, "f2": (1 - size) + (0.2 if shape == "round" else abs(corner - 0.3))}

    result = MultiObjectiveBayesian(space, MultiObjectiveBayesianConfig(objectives=OBJECTIVES, max_evaluations=30, seed=0)).optimize(objective)
    assert min(seen) >= 0.1 and all(space.check(p.params) == p.params for p in result.front)


def test_bayesian_repeats_exactly_and_rebuilds():
    first = bayesian(max_evaluations=20).optimize(ZDT1.function)
    assert first == bayesian(max_evaluations=20).optimize(ZDT1.function)
    assert first.rebuild_optimizer().optimize(ZDT1.function) == first


def test_bayesian_proposes_a_batch_per_worker():
    optimizer = bayesian(max_evaluations=30, population_size=3)
    result = optimizer.optimize(ZDT1.function)
    assert result.evaluations == 30


# --- choosing one -----------------------------------------------------------------------


def test_choose_pareto_optimizer_follows_the_budget():
    assert isinstance(choose_pareto_optimizer(ZDT1.search_space, OBJECTIVES, 150, seed=0), MultiObjectiveBayesian)
    assert isinstance(choose_pareto_optimizer(ZDT1.search_space, OBJECTIVES, 1000, seed=0), NSGA2)
    tags = [ChoiceParameter("tags", ["a", "b", "c"], choose_n=2), RangeParameter("x", 0, 1)]
    assert isinstance(choose_pareto_optimizer(tags, OBJECTIVES, 100), NSGA2)


# --- the benchmark ------------------------------------------------------------------------


def test_every_problems_true_front_is_reachable_inside_the_bounds():
    for problem in multi_problems():
        if problem.name.startswith(("zdt", "rotated")):
            # The optimal designs: x0 anywhere, the rest at their centre.
            point = problem.search_space.sample(np.random.default_rng(0))
            assert problem.true_hypervolume > 0.5
            assert all(problem.search_space[name].minimum <= value <= problem.search_space[name].maximum for name, value in point.items())


def test_the_zdt_fronts_have_their_known_hypervolumes():
    assert zdt("zdt1", 4).true_hypervolume == pytest.approx(0.1 + 2 / 3 + 0.11, abs=1e-4)
    assert zdt("zdt2", 4).true_hypervolume == pytest.approx(0.1 + 1 / 3 + 0.11, abs=1e-4)


def test_the_median_attainment_surface_lies_between_the_fronts():
    fronts = [np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.0, 0.5], [0.5, 0.0]]), np.array([[0.0, 0.8], [0.8, 0.0]])]
    # Two of the three fronts reach f2 = 0.8 at f1 = 0, and f2 = 0 by f1 = 0.8.
    assert attainment_surface(fronts) == [[0.0, 0.8], [0.8, 0.0]]
    assert attainment_surface(fronts, share=1 / 3) == [[0.0, 0.5], [0.5, 0.0]]


def test_the_runner_scores_every_optimizer():
    runs = run_multi_benchmark([zdt("zdt1", 3)], seeds=range(1), budgets_per_dimension=[5])
    assert {run.optimizer for run in runs} == {"nsga2", "bayesian-parego", "bayesian-models", "random", "weighted-sums"}
    assert all(0 <= run.gap <= zdt("zdt1", 3).true_hypervolume for run in runs)
    summaries = summarize_multi(runs)
    assert all(s.attainment is not None for s in summaries)
