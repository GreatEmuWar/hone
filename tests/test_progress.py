"""Following a run as it goes, stopping it from a callback, and stopping it with Ctrl+C."""
import multiprocessing
import os
import signal
import threading

import pytest

from hone import Progress, print_progress

from conftest import make_optimizer, sphere
import objectives


def test_the_callback_hears_every_generation():
    heard: list[Progress] = []
    result = make_optimizer(generations=5, population_size=8).optimize(sphere, callback=heard.append)
    assert [p.generation for p in heard] == result.generations
    assert [p.evaluations for p in heard] == [g.evaluations for g in result.generations]
    assert heard[-1].best_fitness == result.best_fitness and heard[-1].best_params == result.best_params
    assert all(later.elapsed >= earlier.elapsed for earlier, later in zip(heard, heard[1:]))
    assert not result.stopped


def test_best_so_far_never_gets_worse():
    heard: list[Progress] = []
    make_optimizer(generations=10, population_size=8).optimize(sphere, callback=heard.append)
    bests = [p.best_fitness for p in heard]
    assert bests == sorted(bests, reverse=True)


@pytest.mark.parametrize("workers", [1, 2])
def test_returning_true_stops_the_run(workers):
    def stop_after_three(progress):
        return progress.generation.generation == 2

    result = make_optimizer(generations=10, population_size=8).optimize(objectives.sphere, workers=workers, callback=stop_after_three)
    assert result.stopped and len(result.generations) == 3
    assert result.evaluations == result.generations[-1].evaluations, "no evaluations past the stop"


def test_a_callback_returning_true_at_the_end_does_not_mark_the_run_stopped():
    result = make_optimizer(generations=3, population_size=8).optimize(sphere, callback=lambda progress: True if progress.generation.generation == 2 else None)
    assert not result.stopped and len(result.generations) == 3


def test_print_progress_prints_a_line_per_generation(capsys):
    make_optimizer(generations=3, population_size=8).optimize(sphere, callback=print_progress)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and lines[0].startswith("generation 0: 8 evaluations, best ")


def interrupted_after(calls):
    """A sphere that is interrupted, as by Ctrl+C, on its given call."""
    count = 0

    def objective(x, y):
        nonlocal count
        count += 1
        if count == calls:
            raise KeyboardInterrupt
        return x**2 + y**2

    return objective


def test_ctrl_c_returns_the_best_result_so_far():
    result = make_optimizer(generations=10, population_size=8).optimize(interrupted_after(20))
    assert result.stopped and len(result.generations) == 2 and result.evaluations == 19, "the 20th never returned"


def test_ctrl_c_before_any_result_is_raised():
    with pytest.raises(KeyboardInterrupt):
        make_optimizer(generations=10, population_size=8).optimize(interrupted_after(1))


def test_ctrl_c_with_workers_stops_them_and_returns_the_result():
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))

    def interrupt_soon(progress):
        # Started once the workers are running, so no thread is alive when they are forked.
        if progress.generation.generation == 0:
            timer.start()

    try:
        result = make_optimizer(generations=100, population_size=8).optimize(objectives.slow_sphere, workers=2, callback=interrupt_soon)
    finally:
        timer.cancel()
    assert result.stopped and 0 < result.evaluations < 800
    assert multiprocessing.active_children() == []
