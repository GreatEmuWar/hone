"""Property tests for hone.result, from its docstrings and the README's Results section."""
import contextlib
import io
import os

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from hone import GenerationStats, Progress, Result, print_progress
from pbt_runs import Recorder, build, problems

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(
    max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
)
json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-10**6, 10**6) | st.floats(-1e6, 1e6) | st.text(max_size=5),
    lambda children: st.lists(children, max_size=3),
    max_leaves=5,
)
fitnesses = st.floats(min_value=-1e9, max_value=1e9)
param_dicts = st.dictionaries(st.text(min_size=1, max_size=5), json_values, max_size=4)


@st.composite
def results(draw):
    generations = []
    evaluations = 0
    for g in range(draw(st.integers(min_value=0, max_value=4))):
        evaluations += draw(st.integers(min_value=1, max_value=50))
        values = sorted(draw(st.lists(fitnesses, min_size=1, max_size=5)))
        generations.append(GenerationStats(
            generation=g, evaluations=evaluations, failures=draw(st.integers(0, 3)), best_fitness=values[0],
            best_params=draw(param_dicts), worst_fitness=values[-1], mean=float(np.mean(values)),
            median=float(np.median(values)), std=float(np.std(values)),
        ))
    return Result(
        best_fitness=draw(fitnesses), best_params=draw(param_dicts), generations=generations,
        evaluations=evaluations, failures=draw(st.integers(0, 3)), reached_threshold=draw(st.booleans()),
        stopped=draw(st.booleans()),
    )


# README: "A result can be saved to JSON and loaded back".
@run_settings
@given(results())
def test_save_load_round_trip(tmp_path, result):
    path = tmp_path / "run.json"
    result.save(path)
    assert Result.load(path) == result
    assert Result.load(path).model_dump() == result.model_dump()


# The same for results of real runs, run record included.
@run_settings
@given(problems())
def test_run_results_round_trip(tmp_path, problem):
    kind, space, seed, budget = problem
    result = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    path = tmp_path / "run.json"
    result.save(path)
    loaded = Result.load(path)
    assert loaded == result
    assert loaded.run == result.run
    assert loaded.run.optimizer == type(build(kind, space)).__name__
    assert loaded.run.objective == "recorded"


# Docstring: "Two results are equal when they found the same things; their run records are not compared".
@run_settings
@given(problems())
def test_equality_ignores_the_run_record(problem):
    kind, space, seed, budget = problem
    a = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    b = a.model_copy(update={"run": a.run.model_copy(update={"elapsed": 123.0, "workers": 7})})
    assert a == b
    assert a != a.model_copy(update={"evaluations": a.evaluations + 1})
    assert a != a.model_copy(update={"best_fitness": a.best_fitness + 1})


# rebuild_optimizer: "A new optimizer set up exactly as this result's was ... Running it gives the same result, given
# the same objective"; README: "Run it on the same objective and it returns the same result".
@run_settings
@given(problems(), st.integers(min_value=0, max_value=2))
def test_rebuild_repeats_the_run(tmp_path, problem, initial_count):
    kind, space, seed, budget = problem
    rng = np.random.default_rng(seed)
    points = [space.sample(rng) for _ in range(initial_count)] or None
    original = build(kind, space, initial_points=points, seed=seed, max_evaluations=budget)
    result = original.optimize(Recorder(space))
    path = tmp_path / "run.json"
    result.save(path)
    rebuilt = Result.load(path).rebuild_optimizer()
    assert type(rebuilt) is type(original)
    assert rebuilt.config == original.config
    assert rebuilt.search_space == original.search_space
    assert rebuilt.initial_points == original.initial_points
    assert rebuilt.optimize(Recorder(space)) == result


# "this result has no record of its run" for results without one.
@given(results())
def test_rebuild_needs_a_run_record(result):
    with pytest.raises(ValueError):
        result.rebuild_optimizer()


# Results are frozen.
@given(results())
def test_results_are_frozen(result):
    with pytest.raises(Exception):
        result.best_fitness = 0.0


# print_progress: "print one line per generation", as in the README: "generation 0: 50 evaluations, best 12.4, 61.3 s".
@given(results(), st.data())
def test_print_progress_prints_one_line(result, data):
    for stats in result.generations:
        best = data.draw(st.none() | fitnesses)
        progress = Progress(
            generation=stats, evaluations=stats.evaluations, best_fitness=best,
            best_params=None if best is None else {}, elapsed=data.draw(st.floats(0, 1e6)),
        )
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print_progress(progress)
        line = out.getvalue()
        assert line.count("\n") == 1
        assert line.startswith(f"generation {stats.generation}: {stats.evaluations:,} evaluations, best ")
        assert ("none yet" in line) == (best is None)
