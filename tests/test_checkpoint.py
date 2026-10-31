"""Checkpoints: saving a run's state, and carrying the run on from it later."""
import pickle
import subprocess
import sys
import textwrap

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
)
from hone.parameters import ChoiceParameter, RangeParameter

from conftest import bounded_parameters, sphere

START = {"x": 1.0, "y": -1.0}

OPTIMIZERS = {
    "genetic": lambda: GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(max_evaluations=200, seed=3), initial_points=START),
    "cma-es": lambda: CMAES(bounded_parameters(), CMAESConfig(max_evaluations=200, seed=3), initial_points=START),
    "bayesian": lambda: BayesianOptimizer(bounded_parameters(), BayesianConfig(max_evaluations=15, seed=3)),
    "random": lambda: RandomSearch(bounded_parameters(), OptimizerConfig(population_size=10, max_evaluations=60, seed=3)),
    "noisy genetic": lambda: GeneticOptimizer(
        bounded_parameters(), GeneticOptimizerConfig(max_evaluations=200, noise=True, final_evaluations=20, seed=3)
    ),
}


class Counted:
    """sphere, counting its calls."""

    def __init__(self):
        self.calls = 0
        self.__name__ = "sphere"

    def __call__(self, x, y):
        self.calls += 1
        return sphere(x, y)


def interrupt_at(generation):
    def callback(progress):
        if progress.generation.generation == generation:
            raise KeyboardInterrupt  # as if the run were killed here

    return callback


@pytest.mark.parametrize("make", OPTIMIZERS.values(), ids=OPTIMIZERS.keys())
def test_a_resumed_run_gives_the_same_result_as_an_uninterrupted_one(make, tmp_path):
    uninterrupted = make().optimize(sphere)
    path = tmp_path / "run.checkpoint"
    stopped = make().optimize(sphere, checkpoint=path, callback=interrupt_at(3))
    assert stopped.stopped and path.exists()
    counted = Counted()
    resumed = make().optimize(counted, checkpoint=path)
    assert resumed == uninterrupted and not resumed.stopped
    saved = stopped.generations[2].evaluations  # the checkpoint was saved after generation 2
    assert counted.calls == uninterrupted.evaluations - saved, "only the evaluations after the checkpoint were made"


def test_a_run_killed_in_another_process_carries_on(tmp_path):
    path = tmp_path / "run.checkpoint"
    script = textwrap.dedent(f"""
        import os
        from hone import GeneticOptimizer, GeneticOptimizerConfig, RangeParameter

        calls = 0

        def dies_on_call_150(x, y):
            global calls
            calls += 1
            if calls == 150:
                os._exit(1)  # killed, with no chance to clean up
            return x**2 + y**2

        space = [RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)]
        GeneticOptimizer(space, GeneticOptimizerConfig(max_evaluations=400, seed=5)).optimize(dies_on_call_150, checkpoint={str(path)!r})
    """)
    assert subprocess.run([sys.executable, "-c", script]).returncode == 1
    config = GeneticOptimizerConfig(max_evaluations=400, seed=5)
    counted = Counted()
    resumed = GeneticOptimizer(bounded_parameters(), config).optimize(counted, checkpoint=path)
    assert resumed == GeneticOptimizer(bounded_parameters(), config).optimize(sphere)
    assert resumed.evaluations == 400 and 250 < counted.calls < 400, "carried on from a checkpoint before call 150"


def test_a_finished_checkpoint_returns_its_result_without_evaluating(tmp_path):
    path = tmp_path / "run.checkpoint"
    finished = OPTIMIZERS["genetic"]().optimize(sphere, checkpoint=path)

    def must_not_run(x, y):
        raise AssertionError("evaluated again")

    assert OPTIMIZERS["genetic"]().optimize(must_not_run, checkpoint=path) == finished


def test_the_callback_hears_only_generations_after_the_checkpoint(tmp_path):
    path = tmp_path / "run.checkpoint"
    OPTIMIZERS["genetic"]().optimize(sphere, checkpoint=path, callback=interrupt_at(3))
    heard = []
    result = OPTIMIZERS["genetic"]().optimize(sphere, checkpoint=path, callback=heard.append)
    assert [p.generation.generation for p in heard] == [g.generation for g in result.generations[3:]]


def test_elapsed_time_carries_over(tmp_path):
    path = tmp_path / "run.checkpoint"
    first = OPTIMIZERS["genetic"]().optimize(sphere, checkpoint=path, callback=interrupt_at(3))
    resumed = OPTIMIZERS["genetic"]().optimize(sphere, checkpoint=path)
    assert resumed.run.started_at == first.run.started_at
    assert resumed.run.elapsed >= 0


def test_a_checkpoint_from_a_different_setup_is_refused(tmp_path):
    path = tmp_path / "run.checkpoint"
    OPTIMIZERS["genetic"]().optimize(sphere, checkpoint=path)
    other_seed = GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(max_evaluations=200, seed=4), initial_points=START)
    with pytest.raises(ValueError, match="set up differently"):
        other_seed.optimize(sphere, checkpoint=path)
    with pytest.raises(ValueError, match="set up differently"):
        OPTIMIZERS["cma-es"]().load_checkpoint(path)


def test_other_files_are_refused(tmp_path):
    path = tmp_path / "not-a-checkpoint"
    path.write_bytes(pickle.dumps({"something": "else"}))
    with pytest.raises(ValueError, match="not a checkpoint"):
        OPTIMIZERS["genetic"]().load_checkpoint(path)


def test_saving_leaves_no_temporary_file(tmp_path):
    OPTIMIZERS["genetic"]().optimize(sphere, checkpoint=tmp_path / "run.checkpoint")
    assert [p.name for p in tmp_path.iterdir()] == ["run.checkpoint"]


def ask_tell_run(optimizer, stop_after=None, path=None):
    """Run with ask and tell, evaluating one trial at a time; optionally save and stop partway."""
    told = 0
    while not optimizer.done:
        for trial in optimizer.ask(2):
            if stop_after is not None and told == stop_after:
                optimizer.save_checkpoint(path)
                return None
            optimizer.tell(trial, sphere(**trial.params))
            told += 1
    return optimizer.result()


def test_ask_and_tell_runs_carry_on_with_trials_handed_out_again(tmp_path):
    def make():
        return GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(max_evaluations=100, population_size=16, seed=1))

    uninterrupted = ask_tell_run(make())
    path = tmp_path / "run.checkpoint"
    first = make()
    assert ask_tell_run(first, stop_after=21, path=path) is None, "saved with a trial handed out but not told"
    resumed = make()
    resumed.load_checkpoint(path)
    assert resumed.ask(1)[0].id == 21, "the trial handed out before the checkpoint comes back first"
    resumed.load_checkpoint(path)
    assert ask_tell_run(resumed) == uninterrupted


def test_choice_parameters_survive_a_checkpoint(tmp_path):
    space = [ChoiceParameter("material", ["steel", "titanium"]), RangeParameter("n", 0, 10, integer=True)]

    def make():
        return GeneticOptimizer(space, GeneticOptimizerConfig(max_evaluations=120, niche_share=0.5, seed=0))

    def cost(material, n):
        return (n - 3) ** 2 + (0 if material == "titanium" else 1)

    path = tmp_path / "run.checkpoint"
    make().optimize(cost, checkpoint=path, callback=interrupt_at(2))
    assert make().optimize(cost, checkpoint=path) == make().optimize(cost)


def test_checkpoints_work_with_workers(tmp_path):
    import objectives

    def make():
        return GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(max_evaluations=120, seed=2))

    path = tmp_path / "run.checkpoint"
    make().optimize(objectives.sphere, workers=2, checkpoint=path, callback=interrupt_at(2))
    assert make().optimize(objectives.sphere, workers=2, checkpoint=path) == make().optimize(objectives.sphere)


def test_stragglers_out_at_a_checkpoint_are_handed_out_again(tmp_path):
    def make():
        return GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(population_size=4, generations=2, min_finished=0.5, seed=0))

    optimizer = make()
    first = optimizer.ask()
    optimizer.tell(first[0], 1.0)
    optimizer.tell(first[1], 2.0)
    second = optimizer.ask()  # first[2] and first[3] are now stragglers
    path = tmp_path / "run.checkpoint"
    optimizer.save_checkpoint(path)

    resumed = make()
    resumed.load_checkpoint(path)
    again = resumed.ask()
    assert {t.id for t in again} == {t.id for t in first[2:]} | {t.id for t in second}
    for trial in again:
        resumed.tell(trial, sphere(**trial.params))
    while not resumed.done:
        for trial in resumed.ask():
            resumed.tell(trial, sphere(**trial.params))
    assert resumed.result().evaluations == 8


def test_an_asynchronous_run_carries_on_from_its_checkpoint(tmp_path):
    import objectives

    def make():
        return GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(population_size=8, max_evaluations=64, min_finished=0.5, seed=0))

    path = tmp_path / "run.checkpoint"
    stopped = make().optimize(objectives.sometimes_slow, workers=4, checkpoint=path, callback=interrupt_at(2))
    assert stopped.stopped
    resumed = make().optimize(objectives.sometimes_slow, workers=4, checkpoint=path)
    assert resumed.evaluations == 64 and not resumed.stopped
