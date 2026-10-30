"""Evaluating candidates yourself with ask() and tell(), failed evaluations, and parallel workers."""
import math

import pytest

from hone import EvaluationFailed, RandomSearch, Trial
from hone.config import OptimizerConfig

from conftest import bounded_parameters, make_optimizer, run, sphere
import objectives


def drive(optimizer, objective=sphere, batch=None):
    """Run the optimizer to completion with ask and tell, batch candidates at a time."""
    while not optimizer.done:
        for trial in optimizer.ask(batch):
            optimizer.tell(trial, objective(**trial.params))
    return optimizer.result()


# --- ask and tell ---------------------------------------------------------


def test_ask_tell_matches_optimize():
    assert drive(make_optimizer(generations=4, population_size=10)) == run(generations=4, population_size=10)


@pytest.mark.parametrize("batch", [1, 3, 7])
def test_batch_size_does_not_change_the_result(batch):
    assert drive(make_optimizer(generations=4, population_size=10), batch=batch) == run(generations=4, population_size=10)


def test_results_can_arrive_in_any_order():
    optimizer = make_optimizer(generations=4, population_size=10)
    while not optimizer.done:
        trials = optimizer.ask()
        for trial in reversed(trials):
            optimizer.tell(trial.id, sphere(**trial.params))
    in_order = run(generations=4, population_size=10)
    assert optimizer.result().best_fitness == in_order.best_fitness
    assert optimizer.result().evaluations == in_order.evaluations


def test_ask_hands_out_each_candidate_once():
    optimizer = make_optimizer(generations=2, population_size=10)
    first, second = optimizer.ask(4), optimizer.ask()
    assert len(first) == 4 and len(second) == 6
    assert not {t.id for t in first} & {t.id for t in second}
    assert all(isinstance(t, Trial) and t.generation == 0 for t in first + second)


def test_ask_waits_for_results_before_the_next_generation():
    optimizer = make_optimizer(generations=2, population_size=10)
    trials = optimizer.ask()
    assert optimizer.ask() == []
    for trial in trials[:-1]:
        optimizer.tell(trial, sphere(**trial.params))
    assert optimizer.ask() == []
    optimizer.tell(trials[-1], sphere(**trials[-1].params))
    assert {t.generation for t in optimizer.ask()} == {1}


def test_genetic_elites_are_not_asked_for_again():
    optimizer = make_optimizer(generations=2, population_size=10, elitism_rate=0.3)
    for trial in optimizer.ask():
        optimizer.tell(trial, sphere(**trial.params))
    assert len(optimizer.ask()) == 7


def test_ask_returns_nothing_once_done():
    optimizer = make_optimizer(generations=1, population_size=4)
    drive(optimizer)
    assert optimizer.done
    assert optimizer.ask() == []


def test_changing_a_trials_params_does_not_affect_the_optimizer():
    optimizer = make_optimizer(generations=1, population_size=4)
    trial = optimizer.ask(1)[0]
    original = dict(trial.params)
    trial.params["x"] = 99
    optimizer.tell(trial, sphere(**original))
    for t in optimizer.ask():
        optimizer.tell(t, sphere(**t.params))
    assert optimizer.result().generations[0].best_params["x"] != 99


def test_result_is_available_partway_through():
    optimizer = make_optimizer(generations=5, population_size=10)
    trial = optimizer.ask(1)[0]
    optimizer.tell(trial, 3.0)
    partial = optimizer.result()
    assert partial.best_fitness == 3.0 and partial.evaluations == 1 and partial.generations == []


def test_result_needs_a_successful_evaluation():
    optimizer = make_optimizer()
    with pytest.raises(RuntimeError, match="no successful evaluations"):
        optimizer.result()


def test_reset_starts_over():
    optimizer = make_optimizer(generations=3, population_size=10)
    first = drive(optimizer)
    optimizer.reset()
    assert not optimizer.done
    assert drive(optimizer) == first


@pytest.mark.parametrize(
    "tell, error",
    [
        (lambda o, t: o.tell(12345, 1.0), "not waiting for a result"),
        (lambda o, t: (o.tell(t, 1.0), o.tell(t, 1.0)), "not waiting for a result"),
        (lambda o, t: o.tell(t), "pass the trial's value"),
    ],
)
def test_tell_rejects_mistakes(tell, error):
    optimizer = make_optimizer(generations=1, population_size=4)
    trial = optimizer.ask(1)[0]
    with pytest.raises(ValueError, match=error):
        tell(optimizer, trial)


def test_tell_rejects_a_trial_that_was_not_handed_out():
    optimizer = make_optimizer(generations=1, population_size=4)
    trial = optimizer.ask(1)[0]
    with pytest.raises(ValueError, match="not waiting"):
        optimizer.tell(trial.id + 1, 1.0)


def test_tell_rejects_non_numbers():
    optimizer = make_optimizer(generations=1, population_size=4)
    with pytest.raises(TypeError, match="must be a number"):
        optimizer.tell(optimizer.ask(1)[0], "fast")


def test_ask_rejects_a_batch_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        make_optimizer().ask(0)


# --- failed evaluations ---------------------------------------------------


def test_failed_evaluations_are_counted_and_never_best():
    result = run(objective_function=objectives.fails_right_of_zero, generations=5, population_size=20)
    assert result.failures > 0
    assert result.failures == sum(g.failures for g in result.generations)
    assert result.best_params["x"] <= 0
    assert result.evaluations == result.generations[-1].evaluations


@pytest.mark.parametrize("maximize", [False, True])
def test_failed_evaluations_are_never_parents_or_elites(maximize):
    optimizer = make_optimizer(generations=2, population_size=10, maximize=maximize, tournament_p=0.0)
    trials = optimizer.ask()
    for i, trial in enumerate(trials):
        if i % 2:
            optimizer.tell(trial, failed=True)
        else:
            optimizer.tell(trial, float(i))
    succeeded = [ind for ind in optimizer._population if not ind.failed]
    population = optimizer._next_population(optimizer._population, 1)
    elites = population[: round(10 * optimizer.config.elitism_rate)]
    assert all(not e.failed for e in elites) and set(map(id, elites)) <= set(map(id, succeeded))


def test_nan_counts_as_failed():
    optimizer = make_optimizer(generations=1, population_size=2, tournament_size=2)
    first, second = optimizer.ask()
    optimizer.tell(first, math.nan)
    optimizer.tell(second, 4.0)
    result = optimizer.result()
    assert result.failures == 1 and result.best_fitness == 4.0


def test_a_generation_where_everything_fails_has_no_fitness_stats():
    optimizer = make_optimizer(generations=2, population_size=4)
    for trial in optimizer.ask():
        optimizer.tell(trial, failed=True)
    stats = optimizer._history[0]
    assert stats.failures == 4 and stats.best_fitness is None and stats.mean is None
    assert len(optimizer.ask()) == 4, "the next generation starts afresh from random candidates"


def test_a_run_where_everything_fails_has_no_result():
    with pytest.raises(RuntimeError, match="no successful evaluations"):
        run(objective_function=objectives.always_fails, generations=2, population_size=4)


def test_other_exceptions_stop_the_run():
    with pytest.raises(ZeroDivisionError):
        run(objective_function=objectives.raises_a_bug, generations=1, population_size=4)


def test_evaluation_failed_is_an_exception_users_raise():
    assert issubclass(EvaluationFailed, Exception)


def test_random_search_handles_failures():
    result = RandomSearch(bounded_parameters(), OptimizerConfig(generations=3, population_size=20, seed=0)).optimize(
        objectives.fails_right_of_zero
    )
    assert result.failures > 0 and result.best_params["x"] <= 0


# --- parallel workers -----------------------------------------------------


@pytest.mark.parametrize("workers", [2, 4])
def test_workers_give_the_same_result(workers):
    serial = make_optimizer(generations=3, population_size=12).optimize(objectives.sphere)
    parallel = make_optimizer(generations=3, population_size=12).optimize(objectives.sphere, workers=workers)
    assert parallel == serial


def test_workers_record_failures():
    serial = make_optimizer(generations=3, population_size=12).optimize(objectives.fails_right_of_zero)
    parallel = make_optimizer(generations=3, population_size=12).optimize(objectives.fails_right_of_zero, workers=2)
    assert parallel == serial and parallel.failures > 0


def test_workers_pass_other_exceptions_on():
    with pytest.raises(ZeroDivisionError):
        make_optimizer(generations=1, population_size=4).optimize(objectives.raises_a_bug, workers=2)


def test_a_generation_with_nothing_new_to_evaluate_is_skipped():
    """With elitism_rate 0.9 and two individuals, both are elites, so later generations need no evaluations."""
    optimizer = make_optimizer(generations=4, population_size=2, tournament_size=2, elitism_rate=0.9)
    assert len(optimizer.ask()) == 2
    for trial in optimizer._waiting.copy().values():
        optimizer.tell(trial.id, sphere(**trial.params))
    assert optimizer.ask() == [] and optimizer.done
    result = optimizer.result()
    assert len(result.generations) == 4 and result.evaluations == 2


# --- evaluation budget ----------------------------------------------------


def test_budget_stops_partway_through_a_generation():
    result = run(generations=10, population_size=10, max_evaluations=25)
    assert result.evaluations == 25
    assert [g.evaluations for g in result.generations] == [10, 19, 25]


def test_ask_never_hands_out_more_than_the_budget():
    optimizer = make_optimizer(generations=10, population_size=10, max_evaluations=4)
    assert len(optimizer.ask()) == 4
    assert optimizer.ask() == []


def test_budget_waits_for_outstanding_trials():
    optimizer = make_optimizer(generations=10, population_size=10, max_evaluations=4)
    trials = optimizer.ask()
    for trial in trials[:3]:
        optimizer.tell(trial, sphere(**trial.params))
    assert not optimizer.done
    optimizer.tell(trials[3], sphere(**trials[3].params))
    assert optimizer.done
    result = optimizer.result()
    assert result.evaluations == 4 and result.generations[-1].evaluations == 4


def test_budget_counts_failures():
    result = run(objective_function=objectives.fails_right_of_zero, generations=10, population_size=10, max_evaluations=15)
    assert result.evaluations == 15


def test_generations_still_limit_a_run_with_a_large_budget():
    assert run(generations=2, population_size=10, max_evaluations=10_000).evaluations == 19
