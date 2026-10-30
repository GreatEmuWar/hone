"""The record of how a result was produced, and rebuilding an optimizer from it."""
import json

import pytest

from hone import (
    CMAES,
    BayesianConfig,
    BayesianOptimizer,
    CMAESConfig,
    GeneticOptimizer,
    GeneticOptimizerConfig,
    OptimizerConfig,
    RandomSearch,
    Result,
)
from hone.parameters import ChoiceParameter, RangeParameter

from conftest import bounded_parameters, make_optimizer, sphere
import objectives

START = {"x": 1.0, "y": -1.0}


def test_the_result_records_how_it_was_produced():
    result = make_optimizer(generations=3, population_size=8, seed=7).optimize(sphere)
    run = result.run
    assert run is not None
    assert run.optimizer == "GeneticOptimizer" and run.config["seed"] == 7 and run.config["population_size"] == 8
    assert run.search_space.names == ["x", "y"] and run.objective == "sphere"
    assert run.workers == 1 and run.timeout is None and run.initial_points == []
    assert run.hone_version and run.python_version.count(".") == 2 and run.numpy_version
    assert run.started_at is not None and run.started_at.tzinfo is not None and run.elapsed >= 0


def test_workers_and_timeout_are_recorded():
    result = make_optimizer(generations=2, population_size=8).optimize(objectives.sphere, workers=2, timeout=30)
    assert result.run.workers == 2 and result.run.timeout == 30


def test_ask_and_tell_runs_are_recorded_without_optimize_settings():
    optimizer = make_optimizer(generations=1, population_size=4)
    for trial in optimizer.ask():
        optimizer.tell(trial, sphere(**trial.params))
    run = optimizer.result().run
    assert run.objective is None and run.workers is None and run.started_at is not None


def test_a_finished_run_keeps_its_elapsed_time():
    optimizer = make_optimizer(generations=2, population_size=4)
    optimizer.optimize(sphere)
    assert optimizer.result().run.elapsed == optimizer.result().run.elapsed


def test_the_record_survives_saving(tmp_path):
    result = make_optimizer(generations=2, population_size=4).optimize(sphere)
    result.save(tmp_path / "run.json")
    assert Result.load(tmp_path / "run.json").run == result.run


def test_results_are_equal_when_they_found_the_same_things():
    first = make_optimizer(generations=2, population_size=4).optimize(sphere)
    second = make_optimizer(generations=2, population_size=4).optimize(sphere)
    assert first.run != second.run, "started at different times"
    assert first == second


@pytest.mark.parametrize(
    "optimizer",
    [
        GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(max_evaluations=60, seed=3), initial_points=START),
        CMAES(bounded_parameters(), CMAESConfig(max_evaluations=60, seed=3), initial_points=START),
        BayesianOptimizer(bounded_parameters(), BayesianConfig(max_evaluations=12, seed=3), initial_points=START),
        RandomSearch(bounded_parameters(), OptimizerConfig(population_size=10, max_evaluations=30, seed=3), initial_points=START),
    ],
    ids=["genetic", "cma-es", "bayesian", "random"],
)
def test_a_rebuilt_optimizer_repeats_the_run(optimizer, tmp_path):
    result = optimizer.optimize(sphere)
    result.save(tmp_path / "run.json")
    rebuilt = Result.load(tmp_path / "run.json").rebuild_optimizer()
    assert type(rebuilt) is type(optimizer) and rebuilt.config == optimizer.config
    assert rebuilt.initial_points == optimizer.initial_points
    assert rebuilt.optimize(sphere) == result


def test_choice_parameters_survive_the_record(tmp_path):
    space = [ChoiceParameter("material", ["steel", "titanium"]), RangeParameter("n", 0, 10, integer=True)]
    optimizer = GeneticOptimizer(space, GeneticOptimizerConfig(max_evaluations=40, seed=0))

    def cost(material, n):
        return (n - 3) ** 2 + (0 if material == "titanium" else 1)

    result = optimizer.optimize(cost)
    result.save(tmp_path / "run.json")
    assert Result.load(tmp_path / "run.json").rebuild_optimizer().optimize(cost) == result


class TunedSearch(RandomSearch):
    """An optimizer of the user's own."""


def test_an_optimizer_of_your_own_is_rebuilt_from_its_class():
    result = TunedSearch(bounded_parameters(), OptimizerConfig(population_size=5, generations=2, seed=0)).optimize(sphere)
    with pytest.raises(ValueError, match="TunedSearch isn't one of hone's optimizers"):
        result.rebuild_optimizer()
    assert type(result.rebuild_optimizer(TunedSearch)) is TunedSearch


def test_results_saved_before_the_record_still_load(tmp_path):
    result = make_optimizer(generations=2, population_size=4).optimize(sphere)
    old = result.model_dump(mode="json", exclude={"run", "stopped"})
    (tmp_path / "old.json").write_text(json.dumps(old))
    loaded = Result.load(tmp_path / "old.json")
    assert loaded == result and loaded.run is None
    with pytest.raises(ValueError, match="no record of its run"):
        loaded.rebuild_optimizer()
