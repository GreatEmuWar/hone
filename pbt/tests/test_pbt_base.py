"""Property tests for hone.optimizers.base: the Optimizer base class every optimizer shares.

Each property is run on every optimizer that supports the generated search space.
"""
import math
import os
import random

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings, strategies as st

from hone import RangeParameter, SearchSpace, Trial
from pbt_runs import Recorder, build, drive, problems, value_of
from pbt_runs import seeds as seeds_

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))  # raise for a deeper search
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
tmp_settings = settings(
    max_examples=max(EXAMPLES // 2, 10), deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
)


def key(params):
    return repr(sorted(params.items()))


def fails_sometimes(seed):
    """A failure rule that fails about a quarter of points, by each of the three documented ways."""
    def rule(params):
        h = hash((seed, key(params))) % 12
        return {0: "raise", 1: "nan", 2: "inf"}.get(h)
    return rule


# --- every candidate is a point of the search space -----------------------------------------

# README: "every value is checked against the search space"; ask() hands out candidates "to evaluate".
@run_settings
@given(problems())
def test_every_trial_is_a_valid_point(problem):
    kind, space, seed, budget = problem
    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    objective = Recorder(space)
    optimizer.optimize(objective)
    for params, _ in objective.calls:
        assert list(params) == space.names
        assert space.check(params) == params


# --- the budget and the counts ---------------------------------------------------------------

# README: "max_evaluations sets a budget: the run stops after that many evaluations, even partway through a
# generation"; "evaluations, failures: How many evaluations were made, and how many of them failed".
@run_settings
@given(problems(), st.booleans())
def test_budget_and_counts(problem, with_failures):
    kind, space, seed, budget = problem
    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    objective = Recorder(space, fail_when=fails_sometimes(seed) if with_failures else None)
    assume(not with_failures or True)
    try:
        result = optimizer.optimize(objective)
    except RuntimeError:
        assume(False)  # every evaluation failed: there is no result to check
    assert result.evaluations == len(objective.calls) <= budget
    assert result.failures == objective.failures


# README: given a budget, runs are "directly comparable" because every optimizer spends the same number of
# evaluations; "At the end ... the whole budget is used". With a small budget the generations default never binds.
@run_settings
@given(problems())
def test_whole_budget_is_spent(problem):
    kind, space, seed, budget = problem
    result = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    assert result.evaluations == budget


# --- the best result -------------------------------------------------------------------------

# README: "best_fitness, best_params: The best individual found in the whole run". Failed evaluations "rank below
# every successful one".
@run_settings
@given(problems(), st.booleans(), st.booleans())
def test_best_is_the_best_value_seen(problem, maximize, with_failures):
    kind, space, seed, budget = problem
    sign = -1.0 if maximize else 1.0
    objective = Recorder(space, sign=sign, fail_when=fails_sometimes(seed) if with_failures else None)
    try:
        result = build(kind, space, seed=seed, max_evaluations=budget, maximize=maximize).optimize(objective)
    except RuntimeError:
        assume(False)
    best = max(objective.values) if maximize else min(objective.values)
    assert result.best_fitness == best
    assert sign * value_of(space, result.best_params) == result.best_fitness
    assert space.check(result.best_params) == result.best_params


# --- generation statistics -------------------------------------------------------------------

# GenerationStats: "the evaluations used so far, its best fitness and parameters, worst fitness, mean, median and
# standard deviation", over the generation's successful evaluations.
@run_settings
@given(problems(), st.booleans(), st.booleans())
def test_generation_stats_are_consistent(problem, maximize, with_failures):
    kind, space, seed, budget = problem
    sign = -1.0 if maximize else 1.0
    objective = Recorder(space, sign=sign, fail_when=fails_sometimes(seed) if with_failures else None)
    try:
        result = build(kind, space, seed=seed, max_evaluations=budget, maximize=maximize).optimize(objective)
    except RuntimeError:
        assume(False)
    previous = 0
    for index, stats in enumerate(result.generations):
        assert stats.generation == index
        assert previous <= stats.evaluations <= result.evaluations
        previous = stats.evaluations
        assert stats.failures >= 0
        if stats.best_fitness is None:
            continue
        low, high = sorted([stats.best_fitness, stats.worst_fitness])
        if maximize:
            assert stats.best_fitness >= stats.worst_fitness
        else:
            assert stats.best_fitness <= stats.worst_fitness
        assert low - 1e-12 <= stats.mean <= high + 1e-12
        assert low <= stats.median <= high
        assert stats.std >= 0
        assert sign * value_of(space, stats.best_params) == stats.best_fitness
        assert (stats.best_fitness >= result.best_fitness) if not maximize else (stats.best_fitness <= result.best_fitness)
    if result.generations:
        assert result.generations[-1].evaluations == result.evaluations


# --- threshold -------------------------------------------------------------------------------

# Config: "threshold: stop once the best fitness reaches this"; README: "reached_threshold: Whether the run stopped
# early because it reached threshold".
@run_settings
@given(problems(), st.booleans(), st.floats(min_value=0.0, max_value=0.5))
def test_threshold(problem, maximize, threshold):
    kind, space, seed, budget = problem
    sign = -1.0 if maximize else 1.0
    objective = Recorder(space, sign=sign)
    result = build(
        kind, space, seed=seed, max_evaluations=budget, maximize=maximize, threshold=sign * threshold
    ).optimize(objective)
    met = [v for v in objective.values if (v >= sign * threshold if maximize else v <= sign * threshold)]
    assert result.reached_threshold == bool(met)
    if result.reached_threshold:
        assert (result.best_fitness >= -threshold) if maximize else (result.best_fitness <= threshold)
        # It stops at the end of the generation in which the threshold was reached.
        first = next(i for i, (_, v) in enumerate(objective.calls) if v in met)
        assert result.generations[-1].evaluations >= first + 1
        assert len([s for s in result.generations if s.evaluations > first]) == 1


# --- maximizing is minimizing the negation ---------------------------------------------------

# README: "The optimizer minimizes by default; set maximize=True in the config to maximize instead."
@run_settings
@given(problems())
def test_maximize_mirrors_minimize(problem):
    kind, space, seed, budget = problem
    low = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    high = build(kind, space, seed=seed, max_evaluations=budget, maximize=True).optimize(Recorder(space, sign=-1.0))
    assert high.best_fitness == -low.best_fitness
    assert high.best_params == low.best_params
    assert high.evaluations == low.evaluations


# --- repeatability ---------------------------------------------------------------------------

# README: "Set seed to make a run repeatable"; optimize(): "Run the optimizer from the start ... With a seed, every
# call with the same number of workers returns the same result."
@run_settings
@given(problems())
def test_same_seed_same_result(problem):
    kind, space, seed, budget = problem
    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    first, second = Recorder(space), Recorder(space)
    a = optimizer.optimize(first)
    b = optimizer.optimize(second)  # the same optimizer again starts from the start
    c = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    assert a == b == c
    assert first.calls == second.calls


# The base class docstring: "Use it either way round": optimize() or ask() and tell() give the same run.
@run_settings
@given(problems())
def test_ask_tell_matches_optimize(problem):
    kind, space, seed, budget = problem
    by_optimize = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    drive(optimizer, Recorder(space))
    assert optimizer.result() == by_optimize


# README: "ask the optimizer for candidates and tell it the results yourself, in any order". With min_finished=1
# the next generation is bred only once all are told, so the order of tells within a generation can't matter.
@run_settings
@given(problems(), st.randoms(use_true_random=False))
def test_tell_order_does_not_matter(problem, shuffler):
    kind, space, seed, budget = problem
    in_order = build(kind, space, seed=seed, max_evaluations=budget)
    drive(in_order, Recorder(space))
    shuffled = build(kind, space, seed=seed, max_evaluations=budget)

    def shuffle(outcomes):
        outcomes = list(outcomes)
        shuffler.shuffle(outcomes)
        return outcomes

    recorder = Recorder(space)
    drive(shuffled, recorder, order=shuffle)
    a, b = in_order.result(), shuffled.result()
    assert (a.best_fitness, a.evaluations) == (b.best_fitness, b.evaluations)
    # Different points tied for the best value: the first told is kept, so which one depends on the order.
    tied = {key(params) for params, value in recorder.calls if value == b.best_fitness}
    if len(tied) == 1:
        assert a.best_params == b.best_params


# --- ask and tell ----------------------------------------------------------------------------

# ask(): "Hand out up to n candidates"; "Returns an empty list when the run is done, or when every candidate of the
# current generation has been handed out but not all have been told". tell(): rejects trials never asked for or
# already told, booleans, and a missing value.
@run_settings
@given(problems(kinds=("genetic", "random", "cmaes")), st.integers(min_value=1, max_value=7))
def test_ask_and_tell_contract(problem, n):
    kind, space, seed, budget = problem
    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    objective = Recorder(space)
    ids = set()
    while not optimizer.done:
        batch = []
        while chunk := optimizer.ask(n):
            assert len(chunk) <= n
            batch += chunk
        assert batch, "nothing to ask for, yet nothing outstanding and not done"
        assert optimizer.ask() == []  # the generation waits for its results
        assert not ids & {t.id for t in batch}
        ids |= {t.id for t in batch}
        with pytest.raises(TypeError):
            optimizer.tell(batch[0], True)
        with pytest.raises(ValueError):
            optimizer.tell(batch[0])
        for trial in batch:
            optimizer.tell(trial.id, objective(**trial.params))
        with pytest.raises(ValueError):
            optimizer.tell(batch[0], 1.0)
    assert optimizer.ask() == []
    with pytest.raises(ValueError):
        optimizer.tell(max(ids) + 1, 1.0)
    assert optimizer.result().evaluations == len(ids) == budget


# tell(): "A value of NaN or infinity also counts as failed"; failed trials "count as evaluations, rank below every
# successful one".
@run_settings
@given(problems(kinds=("genetic", "random", "cmaes")), st.sampled_from([math.nan, math.inf, -math.inf, None]))
def test_non_finite_values_fail(problem, bad):
    kind, space, seed, budget = problem
    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    trials = optimizer.ask()
    assume(len(trials) >= 2)
    optimizer.tell(trials[0], failed=True) if bad is None else optimizer.tell(trials[0], bad)
    optimizer.tell(trials[1], 5.0)
    result = optimizer.result()
    assert (result.failures, result.evaluations, result.best_fitness) == (1, 2, 5.0)
    assert result.best_params == trials[1].params


# --- initial points --------------------------------------------------------------------------

# README: "The points are evaluated first and count toward the budget."
@run_settings
@given(problems(), st.integers(min_value=1, max_value=3), st.data())
def test_initial_points_are_evaluated_first(problem, count, data):
    kind, space, seed, budget = problem
    rng = np.random.default_rng(seed)
    points = [space.sample(rng) for _ in range(count)]
    assume(budget >= count)
    objective = Recorder(space)
    result = build(kind, space, initial_points=points, seed=seed, max_evaluations=budget).optimize(objective)
    assert [params for params, _ in objective.calls[:count]] == points
    assert result.evaluations == budget
    assert result.run.initial_points == points


# --- callbacks -------------------------------------------------------------------------------

# README: "callback is called after each generation with a Progress: the generation's statistics, the evaluations
# so far, the best value and parameters so far, and the seconds elapsed." "Return True from a callback to stop."
@run_settings
@given(problems(kinds=("genetic", "random", "cmaes")), st.integers(min_value=0, max_value=6))
def test_callback_sees_every_generation_and_can_stop(problem, stop_after):
    kind, space, seed, budget = problem
    full = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    seen = []

    def callback(progress):
        seen.append(progress)
        return len(seen) > stop_after

    result = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space), callback=callback)
    assert [p.generation for p in seen] == result.generations
    assert all(p.evaluations == p.generation.evaluations for p in seen)
    bests = [p.best_fitness for p in seen]
    assert bests == sorted(bests, reverse=True)
    assert all(a.elapsed <= b.elapsed for a, b in zip(seen, seen[1:]))
    assert seen[-1].best_fitness == result.best_fitness
    if len(full.generations) > stop_after + 1:
        assert result.stopped and len(result.generations) == stop_after + 1
        assert result.generations == full.generations[: stop_after + 1]
    else:
        assert not result.stopped and result == full


# --- checkpoints -----------------------------------------------------------------------------

class Crash(Exception):
    pass


# README: "Run the same script again and the run carries on from the last generation saved, instead of starting
# over; the result is the same as if it had never stopped. Once the run has finished, running it again returns its
# result at once."
@tmp_settings
@given(problems(), st.integers(min_value=0, max_value=5))
def test_checkpoint_resume_gives_the_same_result(tmp_path_factory, problem, crash_after):
    kind, space, seed, budget = problem
    path = tmp_path_factory.mktemp("checkpoint") / "run.checkpoint"
    uninterrupted = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space))
    calls = []

    def crashing(progress):
        calls.append(progress)
        if len(calls) > crash_after:
            raise Crash()

    try:
        build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space), callback=crashing, checkpoint=path)
    except Crash:
        pass
    resumed = build(kind, space, seed=seed, max_evaluations=budget).optimize(Recorder(space), checkpoint=path)
    assert resumed == uninterrupted
    again = Recorder(space)
    assert build(kind, space, seed=seed, max_evaluations=budget).optimize(again, checkpoint=path) == uninterrupted
    assert again.calls == []


# README: "With ask and tell, call optimizer.save_checkpoint(path) whenever you like, and load_checkpoint(path) on a
# new optimizer set up the same way. Trials that were handed out but not yet told are handed out again by the next
# ask(), and can still be told by their ids."
@tmp_settings
@given(problems(kinds=("genetic", "random", "cmaes")), st.integers(min_value=0, max_value=60), st.data())
def test_ask_tell_checkpoint_mid_generation(tmp_path_factory, problem, save_at, data):
    kind, space, seed, budget = problem
    path = tmp_path_factory.mktemp("checkpoint") / "run.checkpoint"
    reference = build(kind, space, seed=seed, max_evaluations=budget)
    drive(reference, Recorder(space))

    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    objective = Recorder(space)
    told = 0
    outstanding: list[Trial] = []
    while not optimizer.done and told < save_at:
        outstanding += optimizer.ask()
        trial = outstanding.pop(0)
        optimizer.tell(trial, objective(**trial.params))
        told += 1
    assume(not optimizer.done)
    outstanding += optimizer.ask(data.draw(st.integers(min_value=1, max_value=5)))
    optimizer.save_checkpoint(path)

    restored = build(kind, space, seed=seed, max_evaluations=budget)
    restored.load_checkpoint(path)
    reissued = restored.ask(len(outstanding)) if outstanding else []
    assert [(t.id, t.params) for t in reissued] == [(t.id, t.params) for t in outstanding]
    for trial in reissued:
        restored.tell(trial.id, objective(**trial.params))
    drive(restored, objective)
    assert restored.result() == reference.result()


# load_checkpoint: "The checkpoint must come from an optimizer set up like this one".
@tmp_settings
@given(problems(kinds=("genetic", "random", "cmaes")))
def test_checkpoint_refuses_other_setups(tmp_path_factory, problem):
    kind, space, seed, budget = problem
    path = tmp_path_factory.mktemp("checkpoint") / "run.checkpoint"
    build(kind, space, seed=seed, max_evaluations=budget).save_checkpoint(path)
    with pytest.raises(ValueError):
        build(kind, space, seed=seed, max_evaluations=budget + 1).load_checkpoint(path)


# --- min_finished ----------------------------------------------------------------------------

# README: with min_finished below 1, "each late result joins the generation under way when it arrives ... every
# optimizer counts it toward the best result. At the end, optimize() waits for the last of them, so the whole budget
# is used." done: "the run isn't done until every late result has been told".
@run_settings
@given(problems(kinds=("genetic", "random", "cmaes")), st.floats(min_value=0.3, max_value=0.95), st.randoms(use_true_random=False))
def test_late_results_count(problem, min_finished, rnd):
    kind, space, seed, budget = problem
    optimizer = build(kind, space, seed=seed, max_evaluations=budget, min_finished=min_finished)
    objective = Recorder(space)
    held: list[Trial] = []
    while not optimizer.done:
        trials = optimizer.ask()
        if not trials and not held:
            pytest.fail("not done, yet nothing to ask for and nothing outstanding")
        for trial in trials:
            if rnd.random() < 0.3:
                held.append(trial)
            else:
                optimizer.tell(trial, objective(**trial.params))
        if held and (not trials or rnd.random() < 0.3):
            trial = held.pop(rnd.randrange(len(held)))
            optimizer.tell(trial, objective(**trial.params))
    assert held == []
    result = optimizer.result()
    assert result.evaluations == budget == len(objective.calls)
    assert result.best_fitness == min(objective.values)


# --- noise -----------------------------------------------------------------------------------

# README: "Every evaluation of the same point is averaged, and a point's fitness is that average"; with
# final_evaluations "Its best_fitness is then an average too".
@run_settings
@given(problems(), st.booleans(), st.integers(min_value=0, max_value=6))
def test_noisy_best_is_an_average(problem, maximize, final):
    kind, space, seed, budget = problem
    assume(final < budget)
    sign = -1.0 if maximize else 1.0
    objective = Recorder(space, sign=sign, noise=random.Random(seed))
    result = build(
        kind, space, seed=seed, max_evaluations=budget, maximize=maximize, noise=True, final_evaluations=final
    ).optimize(objective)
    at_best = [v for params, v in objective.calls if params == result.best_params]
    assert result.best_fitness == pytest.approx(float(np.mean(at_best)))
    assert result.evaluations == budget


# README: "final_evaluations keeps that many evaluations back for the end, to evaluate the most promising points
# again (up to 5 of them, each at least twice) and return the one with the best average."
@run_settings
@given(problems(), st.integers(min_value=2, max_value=12))
def test_final_evaluations(problem, final):
    kind, space, seed, budget = problem
    assume(budget - final >= 2)
    objective = Recorder(space, noise=random.Random(seed))
    result = build(kind, space, seed=seed, max_evaluations=budget, noise=True, final_evaluations=final).optimize(objective)
    searched, finals = objective.calls[: budget - final], objective.calls[budget - final:]
    assert len(finals) == final
    finalists = []
    for params, _ in finals:
        if params not in finalists:
            finalists.append(params)
    assert 1 <= len(finalists) <= 5
    assert all(any(params == p for p, _ in searched) for params in finalists)
    assert all(sum(params == p for p, _ in finals) >= 2 for params in finalists)
    assert result.best_params in finalists
    averages = [np.mean([v for p, v in objective.calls if p == params]) for params in finalists]
    assert result.best_fitness == pytest.approx(min(averages))


# With noise, reached_threshold still means the reported best reached the threshold. The example is the case from
# pbt/bug_reports/bug_report_hone_optimizers_base_*.md, where the check once used a stale average.
@run_settings
@example(
    problem=("genetic", SearchSpace([RangeParameter("n", 0, 10, integer=True)]), 10, 200),
    threshold=0.0,
)
@given(problems(), st.floats(min_value=0.0, max_value=0.3))
def test_noisy_threshold_consistent(problem, threshold):
    kind, space, seed, budget = problem
    objective = Recorder(space, noise=random.Random(seed))
    result = build(kind, space, seed=seed, max_evaluations=budget, noise=True, threshold=threshold).optimize(objective)
    if result.reached_threshold:
        assert result.best_fitness <= threshold


# --- Ctrl+C ----------------------------------------------------------------------------------

# README: "Pressing Ctrl+C stops the run too ... optimize() stops the workers and returns the best result so far,
# with stopped set." CHANGELOG 0.7.1: "If no evaluation has succeeded yet, Ctrl+C still raises KeyboardInterrupt."
@run_settings
@given(problems(), st.integers(min_value=0, max_value=30))
def test_ctrl_c_returns_the_best_so_far(problem, interrupt_at):
    kind, space, seed, budget = problem
    assume(interrupt_at < budget)
    recorder = Recorder(space)

    def objective(**params):
        if len(recorder.calls) == interrupt_at:
            raise KeyboardInterrupt
        return recorder(**params)

    optimizer = build(kind, space, seed=seed, max_evaluations=budget)
    if interrupt_at == 0:
        with pytest.raises(KeyboardInterrupt):
            optimizer.optimize(objective)
        return
    result = optimizer.optimize(objective)
    assert result.stopped
    assert result.evaluations == interrupt_at
    assert result.best_fitness == min(recorder.values)


# --- checkpoints with noise and late results -------------------------------------------------

@tmp_settings
@given(problems(), st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=4))
def test_noisy_checkpoint_resume_gives_the_same_result(tmp_path_factory, problem, crash_after, final):
    kind, space, seed, budget = problem
    assume(final < budget)
    path = tmp_path_factory.mktemp("checkpoint") / "run.checkpoint"
    settings_ = dict(seed=seed, max_evaluations=budget, noise=True, final_evaluations=final)
    # A noise sequence keyed by call count, so a resumed run sees the values the uninterrupted one did.
    def noisy(counter):
        rnd = random.Random(seed)
        noise = [rnd.gauss(0, 0.05) for _ in range(budget)]
        def objective(**params):
            counter.append(1)
            return value_of(space, params) + noise[len(counter) - 1]
        return objective

    uninterrupted = build(kind, space, **settings_).optimize(noisy([]))
    calls = []

    def crashing(progress):
        calls.append(progress)
        if len(calls) > crash_after:
            raise Crash()

    try:
        build(kind, space, **settings_).optimize(noisy([]), callback=crashing, checkpoint=path)
    except Crash:
        pass
    evaluations = 0  # the crash came before the first checkpoint, or before anything succeeded
    if path.exists():
        saved = build(kind, space, **settings_)
        saved.load_checkpoint(path)
        try:
            evaluations = saved.result().evaluations
        except RuntimeError:
            pass
    resumed = build(kind, space, **settings_).optimize(noisy([1] * evaluations), checkpoint=path)
    assert resumed == uninterrupted


# The same counts and best with min_finished below 1 inside optimize(), where late results come back from worker
# processes: README "At the end, optimize() waits for the last of them, so the whole budget is used ... every
# optimizer counts it toward the best result."
@settings(max_examples=3, deadline=None)
@given(st.sampled_from(["genetic", "random", "cmaes"]), seeds_, st.floats(min_value=0.4, max_value=0.9))
def test_min_finished_with_workers(kind, seed, min_finished):
    space = SearchSpace([RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)])
    result = build(kind, space, seed=seed, max_evaluations=40, min_finished=min_finished).optimize(sphere, workers=3)
    assert result.evaluations == 40
    assert result.best_fitness == sphere(**result.best_params)
    assert all(g.best_fitness >= result.best_fitness for g in result.generations if g.best_fitness is not None)


def sphere(x, y):
    return x**2 + y**2


# README: "Every parameter the objective function requires must be in the search space. The optimizer checks this
# before it starts." optimize(): the function "must return a number".
@run_settings
@given(problems(kinds=("genetic", "random", "cmaes")))
def test_objective_is_checked_before_the_run(problem):
    kind, space, seed, budget = problem
    calls = []

    def needs_more(extra_parameter_, **params):
        calls.append(1)
        return 0.0

    def accepts_less(extra_parameter_=0):
        calls.append(1)
        return 0.0

    def returns_text(**params):
        calls.append(1)
        return "1.0"

    for objective in (needs_more, accepts_less):
        with pytest.raises(ValueError):
            build(kind, space, seed=seed, max_evaluations=budget).optimize(objective)
    assert calls == []
    with pytest.raises(TypeError):
        build(kind, space, seed=seed, max_evaluations=budget).optimize(returns_text)
