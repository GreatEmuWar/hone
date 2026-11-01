"""min_finished: breeding the next generation before the slowest evaluations of the last one finish."""
import multiprocessing
import time

import pytest
from pydantic import ValidationError

from hone import CMAES, BayesianConfig, BayesianOptimizer, CMAESConfig, GeneticOptimizer, GeneticOptimizerConfig

from conftest import bounded_parameters, sphere
import objectives


def genetic(**config):
    config.setdefault("seed", 0)
    return GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(**config))


def test_min_finished_must_be_a_share():
    for bad in (0, 1.5):
        with pytest.raises(ValidationError):
            GeneticOptimizerConfig(min_finished=bad)


def test_the_next_generation_starts_once_enough_have_finished():
    optimizer = genetic(population_size=4, generations=3, min_finished=0.5)
    first = optimizer.ask()
    assert len(first) == 4
    optimizer.tell(first[0], 1.0)
    assert optimizer.ask() == [], "one of four isn't enough"
    optimizer.tell(first[1], 2.0)
    second = optimizer.ask()
    assert second and all(trial.generation == 1 for trial in second), "two of four is"
    assert len(optimizer.result().generations) == 1


def test_late_results_count_and_join_the_generation_under_way():
    optimizer = genetic(population_size=4, generations=3, min_finished=0.5, elitism=False)
    first = optimizer.ask()
    optimizer.tell(first[0], 5.0)
    optimizer.tell(first[1], 6.0)
    optimizer.ask()
    optimizer.tell(first[2], 0.5)  # a straggler, and the best so far
    result = optimizer.result()
    assert result.evaluations == 3 and result.best_fitness == 0.5
    assert first[2].params in [ind.params for ind in optimizer._population], "it can be bred from"


def test_a_straggler_can_fail_late():
    optimizer = genetic(population_size=4, generations=3, min_finished=0.5)
    first = optimizer.ask()
    optimizer.tell(first[0], 1.0)
    optimizer.tell(first[1], 2.0)
    optimizer.ask()
    optimizer.tell(first[2], failed=True)
    assert optimizer.result().failures == 1


def test_the_run_is_done_only_once_every_straggler_is_told():
    optimizer = genetic(population_size=4, generations=1, min_finished=0.5)
    trials = optimizer.ask()
    optimizer.tell(trials[0], 1.0)
    optimizer.tell(trials[1], 2.0)
    assert not optimizer.done and optimizer.ask() == []
    optimizer.tell(trials[2], 3.0)
    optimizer.tell(trials[3], 4.0)
    assert optimizer.done and optimizer.result().evaluations == 4


def test_cma_es_counts_a_straggler_as_a_result_but_does_not_adapt_to_it():
    def run(straggler_arrives):
        optimizer = CMAES(bounded_parameters(), CMAESConfig(population_size=4, generations=4, min_finished=0.5, seed=0))
        first = optimizer.ask()
        optimizer.tell(first[0], 9.0)
        optimizer.tell(first[1], 8.0)
        second = optimizer.ask()
        if straggler_arrives:
            optimizer.tell(first[2], 0.1)
        for value, trial in enumerate(second):
            optimizer.tell(trial, 5.0 + value)
        return optimizer, optimizer.ask()

    late, third = run(straggler_arrives=True)
    _, third_without = run(straggler_arrives=False)
    assert late.result().best_fitness == 0.1
    assert [t.params for t in third] == [t.params for t in third_without], "its sample belongs to an earlier distribution"


def test_bayesian_optimization_does_not_propose_points_still_being_evaluated():
    optimizer = BayesianOptimizer(bounded_parameters(), BayesianConfig(population_size=4, initial_samples=6, max_evaluations=20, min_finished=0.5, seed=0))
    for trial in optimizer.ask():
        optimizer.tell(trial, sphere(**trial.params))
    batch = optimizer.ask()
    for trial in batch[:2]:
        optimizer.tell(trial, sphere(**trial.params))
    pending = [trial.params for trial in batch[2:]]
    proposed = [trial.params for trial in optimizer.ask()]
    assert proposed and not any(params in pending for params in proposed)


def test_min_finished_of_one_changes_nothing():
    assert genetic(max_evaluations=200).optimize(sphere) == genetic(max_evaluations=200, min_finished=1.0).optimize(sphere)


def timed(optimizer, **options):
    start = time.monotonic()
    result = optimizer.optimize(objectives.sometimes_slow, workers=4, **options)
    return result, time.monotonic() - start


def test_slow_evaluations_no_longer_hold_up_every_generation():
    waiting, waited = timed(genetic(population_size=8, max_evaluations=64, mutation_scale=0.5))
    moving_on, moved_on = timed(genetic(population_size=8, max_evaluations=64, mutation_scale=0.5, min_finished=0.5))
    assert waiting.evaluations == moving_on.evaluations == 64, "the whole budget is used either way"
    # Compared in seconds saved rather than as a ratio: starting the workers takes the same time in
    # both runs, seconds on a slow machine, which would push a ratio toward 1. Here the saving is
    # about 1.2 seconds, on a fast machine and a slow one alike.
    assert waited - moved_on > 0.6


def test_stopping_early_does_not_wait_for_stragglers():
    result, elapsed = timed(
        genetic(population_size=8, max_evaluations=400, mutation_scale=0.5, min_finished=0.5),
        callback=lambda progress: progress.generation.generation == 2,
    )
    assert result.stopped and elapsed < 5
    assert multiprocessing.active_children() == []
