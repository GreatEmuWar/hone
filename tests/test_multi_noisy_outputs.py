"""Noise with output constraints for several objectives: averaging, the margin, the final round for a front, and nothing else moving."""
import math
import pickle

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
    Output,
    RangeParameter,
    SearchSpace,
)
from hone.benchmarks.multi import _with_noise
from hone.benchmarks.runner import SkipBudget
from hone.benchmarks.sharded import load_spec
from hone.optimizers import multi_bayesian
from hone.optimizers.core import Individual

OBJECTIVES = {"a": "minimize", "b": "minimize"}
X = RangeParameter("x", 0, 10)
Y = RangeParameter("y", 0, 10)
MODE = ChoiceParameter("mode", ["a", "b"])
LIMIT = Output("t") >= 10  # a relative margin of (10 - t) / 10
SPACE = SearchSpace([X, Y], [LIMIT])
REFERENCE = {"a": 10.0, "b": 10.0}


def random_search(space=SPACE, **config):
    options = {"objectives": OBJECTIVES, "max_evaluations": 100, "seed": 0, "noise": True, "reference": REFERENCE, **config}
    return MultiObjectiveRandomSearch(space, MultiObjectiveConfig(**options))


def record(optimizer, params, a, b, **outputs):
    """Tell the optimizer one evaluation of params directly, as tell() does, and return its individual."""
    individual = Individual(params=params, generation=0)
    optimizer._evaluations += 1
    optimizer._record(individual, {"a": a, "b": b, **outputs})
    return individual


def key(optimizer, **params):
    return optimizer._key(params)


def noisy(seed=0, sd=0.4):
    """A noisy problem on x and y with one limit, whose noise is drawn from its own seeded generator."""
    rng = np.random.default_rng(seed)

    def function(x, y):
        return {"a": x**2 + y**2 + sd * rng.normal(), "b": (x - 2) ** 2 + y**2 + sd * rng.normal(), "t": 8 + x + y + sd * rng.normal()}

    return function


FRONT_SPACE = SearchSpace([RangeParameter("x", -3, 3), RangeParameter("y", -3, 3)], [Output("t") >= 10])


def nsga2(**config):
    options = {"objectives": OBJECTIVES, "max_evaluations": 100, "seed": 1, "noise": True, **config}
    return NSGA2(FRONT_SPACE, NSGA2Config(**options))


def bayesian(**config):
    options = {"objectives": OBJECTIVES, "max_evaluations": 30, "seed": 1, "noise": True, **config}
    return MultiObjectiveBayesian(FRONT_SPACE, MultiObjectiveBayesianConfig(**options))


# --- what is accepted ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["nsga2", "parego", "models", "random"])
def test_the_multi_objective_optimizers_accept_noise_with_output_constraints(kind):
    if kind == "nsga2":
        optimizer = nsga2()
    elif kind == "random":
        optimizer = random_search()
    else:
        optimizer = bayesian(method=kind)
    assert optimizer.supports_noisy_outputs


def test_a_multi_objective_config_accepts_final_evaluations():
    assert MultiObjectiveConfig(objectives=OBJECTIVES, max_evaluations=50, final_evaluations=10).final_evaluations == 10
    with pytest.raises(ValueError, match="leaves nothing"):
        MultiObjectiveConfig(objectives=OBJECTIVES, max_evaluations=50, final_evaluations=50)
    with pytest.raises(ValueError, match="threshold"):
        MultiObjectiveConfig(objectives=OBJECTIVES, threshold=1.0)


def test_the_shared_names_still_import_from_base():
    from hone.optimizers import base, core

    assert base.DEFAULT_FINAL_EVALUATIONS is core.DEFAULT_FINAL_EVALUATIONS == 10
    assert base.MARGIN_HALF_AT_LEAST is core.MARGIN_HALF_AT_LEAST == 2


# --- averaging ----------------------------------------------------------------------------


def test_a_point_whose_draws_straddle_a_limit_is_judged_on_its_average():
    optimizer = random_search()
    record(optimizer, {"x": 1.0, "y": 1.0}, 1.0, 9.0, t=12.0)
    record(optimizer, {"x": 1.0, "y": 1.0}, 3.0, 7.0, t=9.0)  # this draw breaks the limit; the average, 10.5, doesn't
    result = optimizer.result()
    assert result.feasible and len(result.front.points) == 1
    point = result.front.points[0]
    assert point.evaluations == 2
    assert point.outputs == {"t": 10.5}, "the outputs are the averages"
    assert point.values == {"a": 2.0, "b": 8.0}, "the draw that broke the limit still counts in the average"
    assert result.infeasible == 1, "infeasible counts the draws that broke a limit"
    # And the other way round: a draw that passes doesn't save a point whose average fails.
    other = random_search()
    record(other, {"x": 2.0, "y": 2.0}, 1.0, 9.0, t=11.0)
    record(other, {"x": 2.0, "y": 2.0}, 1.0, 9.0, t=6.0)
    assert not other.result().feasible and other.result().front.points == []
    assert other.result().infeasible == 1


def test_the_front_holds_only_points_that_pass_and_feasible_is_false_when_none_do():
    optimizer = random_search()
    record(optimizer, {"x": 1.0, "y": 0.0}, 1.0, 9.0, t=12.0)
    record(optimizer, {"x": 2.0, "y": 0.0}, 0.5, 0.5, t=5.0)  # dominates the first, but breaks the limit
    result = optimizer.result()
    assert [p.params["x"] for p in result.front.points] == [1.0]
    assert result.feasible
    none = random_search()
    record(none, {"x": 3.0, "y": 0.0}, 1.0, 1.0, t=5.0)
    assert none.result().front.points == [] and not none.result().feasible


def test_progress_says_whether_any_point_passes():
    seen = []
    nsga2(max_evaluations=40).optimize(lambda x, y: {"a": x, "b": y, "t": 0.0}, callback=seen.append)
    assert seen and not any(progress.feasible for progress in seen)


def test_the_generation_summaries_count_infeasible_points_on_their_averages():
    optimizer = random_search()
    individual = record(optimizer, {"x": 1.0, "y": 1.0}, 1.0, 9.0, t=12.0)
    assert individual.violation == 0
    record(optimizer, {"x": 1.0, "y": 1.0}, 1.0, 9.0, t=6.0)
    assert optimizer._violation(key(optimizer, x=1.0, y=1.0)) > 0


def test_without_noise_a_failing_draw_still_leaves_the_front_as_before():
    optimizer = random_search(noise=False)
    record(optimizer, {"x": 1.0, "y": 0.0}, 1.0, 9.0, t=5.0)
    assert not optimizer._observed and optimizer.result().front.points == []
    record(optimizer, {"x": 2.0, "y": 0.0}, 2.0, 8.0, t=12.0)
    assert len(optimizer.result().front.points) == 1 and optimizer.result().infeasible == 1


def test_noise_without_output_constraints_reserves_nothing_even_with_final_evaluations():
    plain = SearchSpace([X, Y])
    optimizer = random_search(plain, final_evaluations=10)
    assert optimizer._reserve == 0


# --- the reserve and the final round ------------------------------------------------------


@pytest.mark.parametrize(
    "options, reserve",
    [
        ({"max_evaluations": 40}, 10),
        ({"max_evaluations": 100}, 10),
        ({"max_evaluations": 300}, 30),
        ({"max_evaluations": 1000}, 100),
        ({"max_evaluations": 10}, 5),
        ({"max_evaluations": None, "generations": 5}, 10),
        ({"max_evaluations": 100, "final_evaluations": 20}, 20),
        ({"max_evaluations": 100, "final_evaluations": 0}, 0),
        ({"max_evaluations": 100, "noise": False}, 0),
    ],
)
def test_the_default_reserve_for_a_front_is_a_tenth_at_least_ten_and_at_most_half(options, reserve):
    assert random_search(**options)._reserve == reserve
    assert nsga2(**{"generations": 5, **options})._reserve == reserve


def test_without_output_constraints_nothing_is_reserved_by_default():
    assert random_search(SearchSpace([X, Y]))._reserve == 0


def round_sizes(optimizer, function):
    result = optimizer.optimize(function)
    evaluations = [g.evaluations for g in result.generations]
    return result, [b - a for a, b in zip([0, *evaluations], evaluations)]


def test_with_the_margin_the_final_round_comes_in_two_halves():
    result, sizes = round_sizes(nsga2(max_evaluations=100), noisy())
    assert result.evaluations == 100
    assert sizes[-2:] == [5, 5], "a reserve of 10 splits into 5 and 5"
    bigger, sizes = round_sizes(nsga2(max_evaluations=300), noisy(1))
    assert bigger.evaluations == 300 and sizes[-2:] == [15, 15]
    odd, sizes = round_sizes(nsga2(max_evaluations=100, final_evaluations=11), noisy(2))
    assert odd.evaluations == 100 and sizes[-2:] == [6, 5], "the first half takes the odd one"


def test_without_the_margin_the_final_round_is_one_round_of_the_whole_reserve():
    optimizer = nsga2(max_evaluations=100, feasibility_confidence=None)
    result, sizes = round_sizes(optimizer, noisy())
    assert result.evaluations == 100 and sizes[-1] == 10
    assert optimizer._final_half == 0 and optimizer._second_half == 0
    assert len(optimizer._finalists) <= 5


def test_a_reserve_too_small_to_split_is_one_round():
    optimizer = nsga2(max_evaluations=100, final_evaluations=3)
    result, sizes = round_sizes(optimizer, noisy())
    assert result.evaluations == 100 and sizes[-1] == 3 and optimizer._final_half == 0


def front_of(optimizer, rows):
    """Record one draw of each (a, b) row, on the front, at params x = a."""
    for a, b, extra in rows:
        record(optimizer, {"x": a, "y": 0.0}, a, b, t=12.0, **extra)
    return [key(optimizer, x=a, y=0.0) for a, _, _ in rows]


def test_the_first_half_spreads_over_the_front_by_hypervolume_contribution():
    optimizer = random_search(feasibility_confidence=None, final_evaluations=4)
    # Three points on a front, the first two crowded together; room for two finalists.
    keys = front_of(optimizer, [(1.0, 9.0, {}), (2.0, 8.0, {}), (9.0, 1.0, {})])
    assert optimizer._spread_finalists(4) == [keys[1], keys[2]], "the greedy pick takes the point of most volume, then the one that adds most"


@pytest.mark.parametrize("reserve, count", [(2, 1), (3, 1), (4, 2), (6, 3), (20, 3)])
def test_the_first_half_gives_each_finalist_at_least_two_evaluations(reserve, count):
    optimizer = random_search(feasibility_confidence=None)
    front_of(optimizer, [(1.0, 9.0, {}), (2.0, 8.0, {}), (9.0, 1.0, {})])
    assert len(optimizer._spread_finalists(reserve)) == count


def test_the_first_half_falls_back_to_the_least_violating_when_nothing_passes():
    optimizer = random_search(feasibility_confidence=None)
    record(optimizer, {"x": 1.0, "y": 0.0}, 1.0, 9.0, t=2.0)
    record(optimizer, {"x": 2.0, "y": 0.0}, 2.0, 8.0, t=8.0)
    record(optimizer, {"x": 3.0, "y": 0.0}, 3.0, 7.0, t=5.0)
    assert optimizer._front_indices()[1] == []
    assert optimizer._spread_finalists(4) == [key(optimizer, x=2.0, y=0.0), key(optimizer, x=3.0, y=0.0)]


def conditional_space():
    return SearchSpace([X, Y, MODE], [LIMIT, MODE["b"] >> (Output("u") >= 10)])


def test_the_first_half_adds_a_point_for_a_conditional_limit_the_finalists_dont_exercise():
    optimizer = random_search(conditional_space(), final_evaluations=16)
    on_front = [(1.0, 9.0), (2.0, 8.0), (9.0, 1.0)]
    for a, b in on_front:
        record(optimizer, {"x": a, "y": 0.0, "mode": "a"}, a, b, t=12.0, u=0.0)
    dominated = {"x": 9.5, "y": 0.0, "mode": "b"}
    record(optimizer, dominated, 9.5, 9.5, t=12.0, u=12.0)
    chosen = optimizer._spread_finalists(8)
    assert len(chosen) == 4 and chosen[-1] == optimizer._key(dominated), "the point where the conditional limit applies joins the three of the front"
    # Without the conditional limit there is nothing to add: three finalists.
    plain = random_search(SearchSpace([X, Y, MODE], [LIMIT]), final_evaluations=16)
    for a, b in on_front:
        record(plain, {"x": a, "y": 0.0, "mode": "a"}, a, b, t=12.0)
    record(plain, dominated, 9.5, 9.5, t=12.0)
    assert len(plain._spread_finalists(8)) == 3


def pooled_front(optimizer, means, spread=1.0):
    """Points on the front a = i, b = 10 - i, each with two draws of t around its mean: they give the limit its noise."""
    keys = []
    for i, mean in enumerate(means):
        params = {"x": float(i), "y": 0.0}
        record(optimizer, params, float(i), 10.0 - i, t=mean + spread)
        record(optimizer, params, float(i), 10.0 - i, t=mean - spread)
        keys.append(optimizer._key(params))
    return keys


MEANS = [30.0, 12.0, 20.0, 11.5, 25.0, 13.0]


def test_the_second_half_goes_to_the_four_least_certain_front_points():
    optimizer = random_search(final_evaluations=20)
    keys = pooled_front(optimizer, MEANS)
    assert optimizer._limit_noise()[0][1] >= 3
    # 12 evaluations: four points, three more each; the passing points are judged as they would be then.
    assert optimizer._least_certain(12) == [keys[3], keys[1], keys[5], keys[2]]


@pytest.mark.parametrize("reserve, count", [(12, 4), (9, 3), (6, 2), (3, 1), (2, 1), (30, 4)])
def test_the_second_half_takes_at_most_four_points_and_a_third_of_its_evaluations(reserve, count):
    optimizer = random_search(final_evaluations=40)
    pooled_front(optimizer, MEANS)
    assert len(optimizer._least_certain(reserve)) == min(count, 6)


def test_the_second_half_drops_points_that_would_fail_with_the_margin_it_gives_them():
    optimizer = random_search(final_evaluations=20)
    keys = pooled_front(optimizer, [30.0, 10.4, 20.0, 25.0])  # 10.4 clears the limit by too little to pass
    assert keys[1] not in optimizer._least_certain(12)


def test_a_point_with_no_applicable_limit_of_known_noise_comes_last_in_the_second_half():
    space = SearchSpace([X, Y, MODE], [MODE["b"] >> (Output("t") >= 10)])
    optimizer = random_search(space, final_evaluations=20)
    unlimited = []  # mode a: the limit doesn't apply, so no limit gives them a certainty (infinite)
    for i, a in enumerate([0.5, 1.5, 2.5]):
        params = {"x": a, "y": 0.0, "mode": "a"}
        record(optimizer, params, a, 10.0 - a, t=0.0)
        unlimited.append(optimizer._key(params))
    limited = []
    for a, mean in [(1.0, 30.0), (2.0, 25.0), (3.0, 20.0)]:
        params = {"x": a, "y": 0.0, "mode": "b"}
        record(optimizer, params, a, 10.0 - a, t=mean + 1)
        record(optimizer, params, a, 10.0 - a, t=mean - 1)
        limited.append(optimizer._key(params))
    assert optimizer._limit_noise()[0][1] == 3
    # Four points are wanted: the three with a finite certainty, least certain first, then the first of the others.
    assert optimizer._least_certain(12) == [limited[2], limited[1], limited[0], unlimited[0]]
    # Two are wanted: the others aren't chosen while two points have a finite certainty.
    assert optimizer._least_certain(6) == [limited[2], limited[1]]
    assert all(optimizer._certainty(k, 3) == math.inf for k in unlimited)


def test_a_front_that_all_fails_with_the_margin_falls_back_to_the_least_violating():
    optimizer = random_search(final_evaluations=20)
    keys = pooled_front(optimizer, [10.1, 10.2, 10.3])
    assert optimizer._least_certain(12)[0] == keys[2], "the one nearest to passing"


# --- the end-to-end run -------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["nsga2", "parego", "models", "random"])
def test_an_explicit_reserve_equal_to_the_default_gives_exactly_the_same_run(kind):
    def make(final):
        if kind == "nsga2":
            return nsga2(max_evaluations=40, final_evaluations=final)
        if kind == "random":
            return random_search(FRONT_SPACE, max_evaluations=40, final_evaluations=final, population_size=10, generations=40, reference=None)
        return bayesian(max_evaluations=30 if kind == "parego" else 24, method=kind, final_evaluations=final)

    default = make(None)
    explicit = make(default._reserve)
    assert default._reserve == 10 or kind == "models"
    first, second = default.optimize(noisy(3)), explicit.optimize(noisy(3))
    assert first == second and first.evaluations == default.config.max_evaluations


def test_noisy_runs_with_every_optimizer_return_a_front_of_points_that_pass():
    for optimizer in [nsga2(max_evaluations=60), bayesian(method="parego", max_evaluations=24), bayesian(method="models", max_evaluations=24)]:
        result = optimizer.optimize(noisy(5, sd=0.3))
        assert result.evaluations == optimizer.config.max_evaluations
        assert result.feasible and result.front.points
        for point in result.front.points:
            assert point.outputs is not None and point.outputs["t"] >= 10, "the average output satisfies the limit"


def test_the_front_of_a_run_is_judged_with_the_margin_at_the_points_actual_draws():
    optimizer = nsga2(max_evaluations=60)
    result = optimizer.optimize(noisy(7, sd=0.3))
    keys = list(optimizer._observed)
    passing = [k for k in keys if optimizer._violation(k) == 0]
    assert len(result.front.points) <= len(passing)
    assert all(p.evaluations >= 1 for p in result.front.points)


# --- NSGA-II ------------------------------------------------------------------------------


def test_nsga2_ranks_survivors_on_the_averages_of_their_points():
    optimizer = nsga2(population_size=2, max_evaluations=100)
    optimizer._next_population(None, 0)
    first = record(optimizer, {"x": 1.0, "y": 0.0}, 1.0, 1.0, t=12.0)  # the best on both objectives, on one draw
    other = record(optimizer, {"x": 2.0, "y": 0.0}, 5.0, 5.0, t=12.0)
    again = record(optimizer, {"x": 1.0, "y": 0.0}, 1.0, 1.0, t=6.0)  # its average, 9, now breaks the limit
    assert first.violation == 0, "judged on one draw when it was recorded"
    optimizer._next_population([first, other, again], 1)
    assert first.violation > 0 and first.outputs == {"t": 9.0}
    assert optimizer._standing[other][0] == 0 and optimizer._standing[first][0] > 0


# --- Bayesian optimization ----------------------------------------------------------------


def test_bayesian_optimization_rejudges_every_row_by_its_points_current_averages():
    optimizer = bayesian(method="parego", max_evaluations=100)
    optimizer._next_population(None, 0)
    first = record(optimizer, {"x": 1.0, "y": 0.0}, 1.0, 5.0, t=12.0)
    second = record(optimizer, {"x": 2.0, "y": 0.0}, 2.0, 4.0, t=12.0)
    third = record(optimizer, {"x": 0.5, "y": 1.0}, 3.0, 3.0, t=12.0)
    optimizer._next_population([first, second, third], 1)
    assert optimizer._violations == [0.0, 0.0, 0.0] and len(optimizer._keys) == 3
    worse = record(optimizer, {"x": 1.0, "y": 0.0}, 1.0, 5.0, t=4.0)  # the first point's average, 8, breaks the limit
    optimizer._next_population([worse], 2)
    assert optimizer._violations[0] > 0 and optimizer._violations[3] > 0 and optimizer._violations[1:3] == [0.0, 0.0]


@pytest.mark.parametrize("method", ["parego", "models"])
def test_bayesian_optimization_takes_its_incumbent_from_the_model_with_noise(method, monkeypatch):
    checked = []
    original = multi_bayesian._improvement if method == "parego" else multi_bayesian._sampled_improvement

    def parego(model, best=None):
        feasible = np.array(optimizer._violations) == 0
        if feasible.any():
            rows = len(optimizer._X)
            checked.append((best, float(model.predict(model.X[:rows])[0][feasible].min())))
        return original(model, best)

    def models(models_, weights, best, draws):
        feasible = np.array(optimizer._violations) == 0
        if feasible.any():
            rows = len(optimizer._X)
            from hone.optimizers.multi_bayesian import _chebyshev

            predicted = np.stack([m.predict_original(m.X[:rows][feasible])[0] for m in models_], axis=-1)
            checked.append((best, float(_chebyshev(predicted, weights).min())))
        return original(models_, weights, best, draws)

    optimizer = bayesian(method=method, max_evaluations=24)
    monkeypatch.setattr(multi_bayesian, "_improvement" if method == "parego" else "_sampled_improvement", parego if method == "parego" else models)
    optimizer.optimize(noisy(4, sd=0.5))
    assert checked and all(math.isclose(best, expected, rel_tol=1e-9, abs_tol=1e-12) for best, expected in checked)


# --- checkpoints --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Pickle checkpoints are deprecated:FutureWarning")  # loads a checkpoint
def test_a_checkpoint_saved_before_feasibility_confidence_existed_is_refused(tmp_path):
    path = tmp_path / "old.checkpoint"
    nsga2(max_evaluations=40).save_checkpoint(path)
    checkpoint = pickle.loads(path.read_bytes())
    del checkpoint["setup"]["config"]["feasibility_confidence"]
    path.write_bytes(pickle.dumps(checkpoint))
    with pytest.raises(ValueError, match="set up differently"):
        nsga2(max_evaluations=40).load_checkpoint(path)


def interrupt_at(generation):
    def callback(progress):
        if progress.generation.generation == generation:
            raise KeyboardInterrupt

    return callback


class Indexed:
    """A noisy objective whose noise depends only on how many evaluations came before, so a resumed run can carry on its sequence."""

    def __init__(self, start=0):
        self.calls = start

    def __call__(self, x, y):
        rng = np.random.default_rng([11, self.calls])
        self.calls += 1
        return {"a": x**2 + y**2 + 0.3 * rng.normal(), "b": (x - 2) ** 2 + y**2 + 0.3 * rng.normal(), "t": 8 + x + y + 0.3 * rng.normal()}


@pytest.mark.filterwarnings("ignore:Pickle checkpoints are deprecated:FutureWarning")  # loads a checkpoint
def test_a_noisy_constrained_run_interrupted_resumes_to_the_same_result(tmp_path):
    def make():
        return nsga2(max_evaluations=60, seed=2)

    whole = make().optimize(Indexed())
    last = len(whole.generations) - 1
    assert last >= 5
    # Interrupted in the search, in the final round's first half, and in its second half.
    for stop in (2, last - 1, last):
        path = tmp_path / f"run{stop}.checkpoint"
        assert make().optimize(Indexed(), checkpoint=path, callback=interrupt_at(stop)).stopped
        resumed = make().optimize(Indexed(whole.generations[stop - 1].evaluations), checkpoint=path)
        assert resumed == whole, f"interrupted at generation {stop} of {last}"


# --- the comparison spec's factories --------------------------------------------------------


def test_every_factory_of_the_comparison_spec_builds_and_runs():
    spec = load_spec("benchmarks/specs/multi_compare.py")
    problem = next(p for p in spec.problems() if p.name == "noisy-tnk")
    built = 0
    for name, factory in spec.optimizers().items():
        try:
            optimizer = factory(problem, 10, 0)
        except SkipBudget:
            continue
        result = optimizer.optimize(_with_noise(problem, 0))
        assert result.evaluations == 10, name
        built += 1
    assert built == 16
