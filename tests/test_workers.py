"""Evaluating in worker processes: time limits, crashed workers, and stopping cleanly on errors."""
import multiprocessing
import time
from functools import partial

import pytest

from hone.optimizers._pool import EvaluationPool

from conftest import make_optimizer
import objectives

SETTINGS = dict(generations=2, population_size=8)



@pytest.mark.parametrize("workers", [1, 3])
def test_evaluations_past_the_time_limit_count_as_failed(workers):
    start = time.monotonic()
    result = make_optimizer(**SETTINGS).optimize(objectives.hangs_right_of_zero, workers=workers, timeout=0.3)
    elapsed = time.monotonic() - start
    assert result.failures > 0 and result.best_params["x"] <= 0
    assert result.evaluations == result.generations[-1].evaluations
    assert elapsed < 30, "hung evaluations were stopped rather than waited for"


def test_timeouts_give_the_same_result_however_many_workers():
    def run(workers):
        return make_optimizer(**SETTINGS).optimize(objectives.hangs_right_of_zero, workers=workers, timeout=0.3)

    assert run(1) == run(4)


def test_a_timeout_matches_failing_those_evaluations_directly():
    with_timeout = make_optimizer(**SETTINGS).optimize(objectives.hangs_right_of_zero, timeout=0.3)
    failing = make_optimizer(**SETTINGS).optimize(objectives.fails_right_of_zero)
    assert with_timeout == failing


def test_a_crashed_worker_counts_as_a_failed_evaluation():
    crashed = make_optimizer(**SETTINGS).optimize(objectives.crashes_right_of_zero, workers=2)
    failing = make_optimizer(**SETTINGS).optimize(objectives.fails_right_of_zero)
    assert crashed == failing and crashed.failures > 0


def test_an_error_stops_every_worker_at_once():
    start = time.monotonic()
    with pytest.raises(ZeroDivisionError):
        make_optimizer(**SETTINGS).optimize(objectives.raises_while_others_hang, workers=4)
    assert time.monotonic() - start < 30, "the run didn't wait for the hanging evaluations"
    assert multiprocessing.active_children() == []


def test_an_error_that_cannot_be_pickled_still_stops_the_run():
    with pytest.raises(RuntimeError, match="UnpicklableError: carries a lock"):
        make_optimizer(**SETTINGS).optimize(objectives.raises_unpicklable, workers=2)


def test_workers_exit_after_a_run():
    make_optimizer(**SETTINGS).optimize(objectives.sphere, workers=2, timeout=10)
    assert multiprocessing.active_children() == []


def test_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout must be positive"):
        make_optimizer().optimize(objectives.sphere, timeout=0)


def square(params):
    return params["x"] ** 2


def test_pool_returns_outcomes_in_order():
    with EvaluationPool(square, workers=3) as pool:
        assert pool.map([{"x": x} for x in range(10)]) == [x**2 for x in range(10)]
        assert pool.map([]) == []


def test_pool_keeps_working_after_replacing_a_worker():
    evaluate = partial(objectives_evaluate, objectives.hangs_right_of_zero)
    with EvaluationPool(evaluate, workers=1, timeout=0.3) as pool:
        assert pool.map([{"x": 1, "y": 0}, {"x": -1, "y": 2}]) == [None, 5]


def objectives_evaluate(function, params):
    return function(**params)
