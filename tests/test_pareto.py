"""Several objectives: domination, fronts, hypervolume, choosing from a front, and runs that return one."""
import math

import numpy as np
import pytest
from pydantic import ValidationError

from hone import (
    ChoiceParameter,
    EvaluationFailed,
    Front,
    MultiObjectiveConfig,
    MultiObjectiveRandomSearch,
    ParetoPoint,
    ParetoResult,
    RangeParameter,
    SearchSpace,
    print_progress,
)
from hone.pareto import hypervolume, non_dominated

X = RangeParameter("x", 0, 1)
Y = RangeParameter("y", 0, 1)
BOTH = {"f1": "minimize", "f2": "minimize"}


def zdt1(x, y):
    """Two objectives in conflict, whose front is f2 = 1 - sqrt(f1) where y = 0."""
    g = 1 + 9 * y
    return {"f1": x, "f2": g * (1 - math.sqrt(x / g))}


def search(space=(X, Y), objectives=BOTH, **config):
    settings = {"max_evaluations": 300, "seed": 0, "population_size": 30, **config}
    return MultiObjectiveRandomSearch(list(space), MultiObjectiveConfig(objectives=objectives, **settings))


def front_of(*points, objectives=BOTH):
    names = list(objectives)
    return Front(objectives=objectives, points=[ParetoPoint(params={"i": i}, values=dict(zip(names, p))) for i, p in enumerate(points)])


# --- domination and the front ----------------------------------------------------


def test_the_front_keeps_what_nothing_else_beats_on_every_objective():
    values = np.array([[0.0, 1.0], [0.6, 0.6], [1.0, 0.0], [1.0, 1.0], [0.5, 0.7]])
    assert non_dominated(values) == [0, 4, 1, 2]  # (1, 1) is beaten by (0.6, 0.6), in order of the first column


def test_points_equal_within_the_tolerance_count_once_and_the_first_stays():
    values = np.array([[0.6, 0.6], [0.0, 1.0], [0.6, 0.6 + 1e-13], [0.6, 0.6]])
    assert non_dominated(values) == [1, 0]


def test_a_point_a_hair_worse_on_one_objective_and_better_on_another_replaces_a_tied_point():
    values = np.array([[1.0, 1.0], [1.0 + 1e-12, 0.5]])
    assert non_dominated(values) == [1]


def test_hypervolume_is_exact_in_two_and_three_dimensions():
    assert hypervolume(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([2.0, 2.0])) == pytest.approx(3.0)
    assert hypervolume(np.array([[1.0, 1.0, 1.0]]), np.array([2.0, 3.0, 4.0])) == pytest.approx(6.0)
    # Two boxes overlapping in a 1 x 1 x 1 corner: 4 + 4 - 1.
    assert hypervolume(np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]), np.array([2.0, 2.0, 2.0])) == pytest.approx(3.0)
    assert hypervolume(np.array([[3.0, 0.0]]), np.array([2.0, 2.0])) == 0.0  # outside the reference counts for nothing


def test_hypervolume_agrees_with_counting_random_points():
    rng = np.random.default_rng(0)
    for dimensions in (2, 3):
        points, reference = rng.uniform(size=(20, dimensions)), np.full(dimensions, 1.0)
        samples = rng.uniform(size=(200_000, dimensions))
        dominated = np.zeros(len(samples), dtype=bool)
        for point in points:
            dominated |= np.all(samples >= point, axis=1)
        assert hypervolume(points, reference) == pytest.approx(dominated.mean(), abs=0.01)


# --- choosing from a front -------------------------------------------------------


def test_best_gives_the_best_of_one_objective_within_limits_on_the_others():
    front = front_of((0.0, 1.0), (0.3, 0.5), (0.6, 0.2), (1.0, 0.0))
    assert front.best("f2").values == {"f1": 1.0, "f2": 0.0}
    assert front.best("f2", where={"f1": 0.5}).values == {"f1": 0.3, "f2": 0.5}
    with pytest.raises(ValueError, match="no point on the front has f1 at most -1"):
        front.best("f2", where={"f1": -1})
    with pytest.raises(ValueError, match="no objective is named 'cost'"):
        front.best("cost")


def test_limits_on_an_objective_to_maximize_are_lower_bounds():
    front = front_of((0.9, 10.0), (0.95, 20.0), (0.99, 40.0), objectives={"accuracy": "maximize", "cost": "minimize"})
    assert front.best("cost", where={"accuracy": 0.95}).values == {"accuracy": 0.95, "cost": 20.0}
    assert front.ideal == {"accuracy": 0.99, "cost": 10.0} and front.nadir == {"accuracy": 0.9, "cost": 40.0}


def test_closest_reaches_a_point_in_a_dent_that_no_weighted_sum_picks():
    # B at (0.6, 0.6) is on the front, but for any weights w, A or C has a smaller weighted sum.
    front = front_of((0.0, 1.0), (0.6, 0.6), (1.0, 0.0))
    for w in np.linspace(0, 1, 21):
        sums = [w * p.values["f1"] + (1 - w) * p.values["f2"] for p in front]
        assert int(np.argmin(sums)) != 1
    assert front.closest().values == {"f1": 0.6, "f2": 0.6}
    assert front.closest({"f1": 10}).values == {"f1": 0.0, "f2": 1.0}


def test_closest_doesnt_depend_on_an_objectives_units():
    dollars = front_of((0.9, 8000.0), (0.95, 9000.0), (0.99, 15000.0), objectives={"accuracy": "maximize", "cost": "minimize"})
    thousands = front_of((0.9, 8.0), (0.95, 9.0), (0.99, 15.0), objectives={"accuracy": "maximize", "cost": "minimize"})
    assert dollars.closest().params == thousands.closest().params == {"i": 1}


# --- the config --------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"objectives": {"f1": "minimize"}}, "at least 2"),
        ({"objectives": {"f1": "lower", "f2": "minimize"}}, "minimize"),
        ({"objectives": BOTH, "maximize": True}, "maximize doesn't apply"),
        ({"objectives": BOTH, "threshold": 0.1}, "threshold doesn't apply"),
        ({"objectives": BOTH, "noise": True, "final_evaluations": 5}, "final_evaluations doesn't apply"),
        ({"objectives": BOTH, "reference": {"f1": 1.0}}, "reference needs a value for each objective"),
    ],
)
def test_the_config_refuses_what_doesnt_apply_to_several_objectives(settings, message):
    with pytest.raises(ValidationError, match=message):
        MultiObjectiveConfig(**settings)


# --- runs ----------------------------------------------------------------------------


def test_a_run_returns_the_front_of_everything_it_evaluated():
    seen = []

    def recording(x, y):
        seen.append(zdt1(x, y))
        return seen[-1]

    result = search().optimize(recording)
    assert isinstance(result, ParetoResult) and result.evaluations == 300
    vectors = np.array([[v["f1"], v["f2"]] for v in seen])
    on_front = {(p.values["f1"], p.values["f2"]) for p in result.front}
    assert on_front == {tuple(vectors[i]) for i in non_dominated(vectors)}
    for point in result.front:  # every point evaluated is on the front or beaten by a point on it
        assert zdt1(**point.params) == point.values


def test_the_fronts_hypervolume_never_falls_during_a_run():
    result = search(reference={"f1": 1.1, "f2": 11.0}).optimize(zdt1)
    volumes = [stats.hypervolume for stats in result.generations]
    assert all(later >= earlier for earlier, later in zip(volumes, volumes[1:]))
    assert volumes[-1] == pytest.approx(result.front.hypervolume({"f1": 1.1, "f2": 11.0}))


def test_flipping_an_objectives_direction_and_sign_gives_the_same_front():
    minimized = search().optimize(zdt1)
    maximized = search(objectives={"f1": "minimize", "f2": "maximize"}).optimize(
        lambda x, y: {"f1": zdt1(x, y)["f1"], "f2": -zdt1(x, y)["f2"]}
    )
    assert [p.params for p in minimized.front] == [p.params for p in maximized.front]


def test_the_front_doesnt_change_when_an_objective_is_rescaled():
    plain = search().optimize(zdt1)
    rescaled = search().optimize(lambda x, y: {"f1": zdt1(x, y)["f1"], "f2": math.exp(3 * zdt1(x, y)["f2"])})
    assert [p.params for p in plain.front] == [p.params for p in rescaled.front]


@pytest.mark.parametrize(
    "value, message",
    [
        (1.0, "must be a dict"),
        ({"f1": 1.0}, "missing f2"),
        ({"f1": 1.0, "f2": 2.0, "f3": 3.0}, "unknown f3"),
        ({"f1": 1.0, "f2": "low"}, "the value of f2 must be a number"),
        ({"f1": True, "f2": 2.0}, "the value of f1 must be a number"),
    ],
)
def test_values_that_dont_match_the_objectives_are_a_type_error(value, message):
    optimizer = search()
    trial = optimizer.ask(1)[0]
    with pytest.raises(TypeError, match=message):
        optimizer.tell(trial, value)
    with pytest.raises(TypeError, match=message):
        search().optimize(lambda x, y: value)


def test_failed_evaluations_are_never_on_the_front():
    def sometimes_failing(x, y):
        if x < 0.2:
            raise EvaluationFailed
        return {"f1": x, "f2": math.nan} if x > 0.9 else zdt1(x, y)

    result = search().optimize(sometimes_failing)
    assert result.failures > 0 and all(0.2 <= p.values["f1"] <= 0.9 for p in result.front)


def test_with_noise_points_are_averaged_and_come_with_standard_errors():
    space = [ChoiceParameter("design", ["a", "b"])]
    rng = np.random.default_rng(0)

    def noisy(design):
        return {"f1": (0.0 if design == "a" else 1.0) + rng.normal(0, 0.1), "f2": (1.0 if design == "a" else 0.0) + rng.normal(0, 0.1)}

    result = search(space, noise=True, max_evaluations=40, population_size=10).optimize(noisy)
    assert sorted(p.params["design"] for p in result.front) == ["a", "b"]
    assert sum(point.evaluations for point in result.front) == 40
    for point in result.front:
        assert point.standard_error is not None
        assert all(0.0 < error < 0.05 for error in point.standard_error.values())


def test_without_noise_a_repeated_point_is_one_point():
    result = search([ChoiceParameter("design", ["a", "b"])], max_evaluations=20, population_size=10).optimize(
        lambda design: {"f1": float(design == "b"), "f2": float(design == "a")}
    )
    assert len(result.front) == 2 and all(p.standard_error is None and p.evaluations > 1 for p in result.front)


def test_a_result_saves_loads_rebuilds_and_repeats(tmp_path):
    result = search().optimize(zdt1)
    result.save(tmp_path / "front.json")
    loaded = ParetoResult.load(tmp_path / "front.json")
    assert loaded == result and loaded.front.closest() == result.front.closest()
    assert loaded.rebuild_optimizer().optimize(zdt1) == result


def test_a_checkpointed_run_resumes(tmp_path):
    whole = search().optimize(zdt1)
    partial = search()
    for _ in range(4):
        for trial in partial.ask():
            partial.tell(trial, zdt1(**trial.params))
    partial.save_checkpoint(tmp_path / "run.checkpoint")
    resumed = search()
    resumed.load_checkpoint(tmp_path / "run.checkpoint")
    while not resumed.done:
        for trial in resumed.ask():
            resumed.tell(trial, zdt1(**trial.params))
    assert resumed.result() == whole


def test_workers_give_the_same_front():
    assert search().optimize(zdt1, workers=2) == search().optimize(zdt1)


def test_constraints_and_conditions_apply_as_with_one_objective():
    shape = ChoiceParameter("shape", ["square", "round"])
    space = SearchSpace([shape, X, Y], constraints=[X + Y <= 1], conditions=[shape["square"] >> Y])
    seen = []

    def objective(shape, x, y=0.0):
        seen.append({"shape": shape, "x": x, "y": y})
        return {"f1": x, "f2": 1 - x + y}

    result = MultiObjectiveRandomSearch(space, MultiObjectiveConfig(objectives=BOTH, max_evaluations=100, seed=0)).optimize(objective)
    assert all(point["x"] + point["y"] <= 1 for point in seen)
    assert all(space.check(p.params) == p.params for p in result.front)


def test_progress_reports_the_front(capsys):
    reports = []
    search(max_evaluations=90).optimize(zdt1, callback=lambda progress: reports.append(progress) or print_progress(progress))
    assert len(reports) == 3 and reports[-1].front.points and reports[-1].generation.front_size == len(reports[-1].front)
    assert "front of" in capsys.readouterr().out


def test_a_callback_can_stop_the_run():
    result = search().optimize(zdt1, callback=lambda progress: progress.evaluations >= 60)
    assert result.stopped and result.evaluations == 60
