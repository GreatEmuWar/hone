"""The multi-objective problems with limits on outputs: their functions, true fronts, scales and noise."""
import math

import numpy as np
import pytest

from hone.benchmarks import cantilever_outputs, noisy_multi_output_problems, osy, tnk
from hone.benchmarks import multi_outputs as mo
from hone.config import NSGA2Config
from hone.optimizers.genetic import NSGA2
from hone.pareto import non_dominated

PROBLEMS = {"osy": osy(), "tnk": tnk(), "cantilever-outputs": cantilever_outputs()}


def feasible(problem, params):
    """The function's values at params, and whether its outputs satisfy every limit."""
    values = problem.function(**params)
    outputs = {name: values[name] for name in problem.search_space.outputs}
    return values, problem.search_space.output_violation(params, outputs) == 0


def test_each_function_returns_every_objective_and_output():
    points = {
        "osy": dict(x1=3, x2=2, x3=2, x4=1, x5=2, x6=3),
        "tnk": dict(x1=0.5, x2=0.9),
        "cantilever-outputs": dict(material="aluminium", width=40, height=80),
    }
    for name, problem in PROBLEMS.items():
        values = problem.function(**points[name])
        assert set(values) == set(problem.objectives) | set(problem.outputs)
        assert all(math.isfinite(v) for v in values.values())
        assert problem.search_space.outputs == sorted(problem.outputs)
        assert problem.search_space.output_constraints and not problem.search_space.parameter_constraints
        assert {"outputs", "constrained"} <= problem.tags


def test_osy_matches_the_published_formulas_at_points_of_its_front():
    # The corners of the front, where the limits g2, g4, g5 and g6 are active: Deb (2001), OSY.
    cases = [((5, 1, 1, 0, 1, 0), -242, 28), ((5, 1, 5, 0, 5, 0), -274, 76), ((1, 1, 1, 0, 1, 0), -42, 4)]
    problem = PROBLEMS["osy"]
    for x, f1, f2 in cases:
        values, ok = feasible(problem, dict(zip(problem.search_space.names, x)))
        assert ok
        assert (values["f1"], values["f2"]) == (f1, f2)
    values, _ = feasible(problem, dict(x1=5, x2=1, x3=1, x4=0, x5=1, x6=0))
    assert [values[f"g{i}"] for i in range(1, 7)] == [4, 0, 6, 0, 0, 0]
    assert not feasible(problem, dict(x1=1, x2=0.5, x3=1, x4=0, x5=1, x6=0))[1]  # x1 + x2 < 2
    assert not feasible(problem, dict(x1=3, x2=2, x3=1, x4=0, x5=3, x6=0))[1]  # (x5 - 3)^2 + x6 < 4


def test_tnk_is_feasible_on_its_wavy_limit_and_infeasible_inside_it():
    problem = PROBLEMS["tnk"]
    edge = math.sqrt(1.1) / math.sqrt(2)  # at 45 degrees, cos(16 * pi / 4) = 1, so the limit is at radius sqrt(1.1)
    values, ok = feasible(problem, dict(x1=edge, x2=edge))
    assert ok and abs(values["c1"]) < 1e-12 and values["c2"] < 0.5
    assert (values["f1"], values["f2"]) == (edge, edge)
    assert not feasible(problem, dict(x1=0.1, x2=0.1))[1]
    assert not feasible(problem, dict(x1=1.2, x2=1.2))[1]
    assert problem.function(x1=0.3, x2=0.0)["c1"] == pytest.approx(0.09 - 1 - 0.1 * math.cos(8 * math.pi))  # x2 = 0 is defined


def test_cantilever_matches_the_example_and_reports_the_stress_ratio():
    problem = PROBLEMS["cantilever-outputs"]
    length, load = 1.0, 5000.0
    materials = {"steel": (200e9, 7850, 250e6, 1.5), "aluminium": (69e9, 2700, 276e6, 4.0), "carbon fibre": (135e9, 1600, 1000e6, 40.0)}
    for material, width, height in [("steel", 50, 100), ("aluminium", 25, 60), ("carbon fibre", 10, 70), ("steel", 100, 150)]:
        modulus, density, strength, price = materials[material]
        b, h = width / 1000, height / 1000
        values = problem.function(material=material, width=width, height=height)
        mass = density * b * h * length
        assert values["mass"] == pytest.approx(mass)
        assert values["cost"] == pytest.approx(price * mass)
        assert values["deflection"] == pytest.approx(1000 * load * length**3 / (3 * modulus * b * h**3 / 12))
        margin_mpa = (6 * load * length / (b * h**2) - strength / 1.5) / 1e6  # the example's stress_margin
        assert (values["stress_ratio"] <= 1) == (margin_mpa <= 0)
        assert feasible(problem, dict(material=material, width=width, height=height))[1] == (margin_mpa <= 0)


def true_front(name):
    """Designs as parameter dicts and the objective values of each problem's true front, as the module builds it."""
    problem = PROBLEMS[name]
    if name == "osy":
        designs, values = mo._osy_front()
        return [dict(zip(problem.search_space.names, d)) for d in designs], values
    if name == "tnk":
        points = mo._tnk_front()
        return [dict(x1=x, x2=y) for x, y in points[::50]], points
    designs, values = mo._cantilever_front(200)
    return [dict(material=m, width=w, height=h) for m, w, h in designs], values


@pytest.mark.parametrize("name", ["osy", "tnk", "cantilever-outputs"])
def test_every_point_of_a_true_front_is_feasible_and_nothing_in_it_dominates_another(name):
    problem = PROBLEMS[name]
    designs, values = true_front(name)
    for design in designs[::7]:
        assert feasible(problem, design)[1], design
    names = list(problem.objectives)
    ordered = values[np.lexsort(values.T[::-1])]
    if name == "cantilever-outputs":
        assert len(non_dominated(values)) == len(values)
    else:
        assert np.all(np.diff(ordered[:, 1]) < 0)  # sorted by the first objective, the second strictly falls
        assert np.all(np.diff(ordered[:, 0]) > 0)
    for design, row in zip(designs[::7], values[::7] if name != "tnk" else values[::350]):
        assert [problem.function(**design)[n] for n in names] == pytest.approx(row)


@pytest.mark.parametrize("name", ["osy", "tnk"])
def test_the_scale_and_hypervolume_are_those_of_the_true_front(name):
    problem = PROBLEMS[name]
    values = mo._osy_front()[1] if name == "osy" else mo._tnk_front()
    assert [problem.ideal["f1"], problem.ideal["f2"]] == list(values.min(axis=0))
    assert [problem.nadir["f1"], problem.nadir["f2"]] == list(values.max(axis=0))
    scaled = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))
    assert problem.hypervolume([{"f1": a, "f2": b} for a, b in values]) == pytest.approx(problem.true_hypervolume)
    assert 0.5 < problem.true_hypervolume < 1.1**2 and scaled.min() == 0 and scaled.max() == 1


def test_osy_scale_is_the_analytic_one():
    problem = PROBLEMS["osy"]
    # f1 runs from -(225 + 1 + 16 + 16 + 16) at x = (5, 1, 5, 0, 5, 0) to -(25 + 1 + 16) at (1, 1, 1, 0, 1, 0), and f2 from 4 to 76.
    assert (problem.ideal, problem.nadir) == ({"f1": -274.0, "f2": 4.0}, {"f1": -42.0, "f2": 76.0})


def test_osy_hypervolume_changes_by_under_1e_4_when_the_resolution_doubles():
    def volume(resolution):
        values = mo._osy_front(resolution)[1]
        ideal, nadir = values.min(axis=0), values.max(axis=0)
        return mo._front_hypervolume(values[:, 0], values[:, 1], tuple(ideal), tuple(nadir))

    # 1,001 gives 0.968717, 2,001 gives 0.968790 (the default), 4,001 gives 0.968827.
    assert abs(volume(2001) - volume(1001)) < 2e-4
    assert abs(volume(4001) - volume(2001)) < 1e-4


def test_tnk_hypervolume_changes_by_under_1e_4_when_the_resolution_doubles():
    def volume(resolution):
        points = mo._tnk_front(resolution)
        ideal, nadir = tuple(points.min(axis=0)), tuple(points.max(axis=0))
        return mo._front_hypervolume(points[:, 0], points[:, 1], ideal, nadir)

    # 50,001 angles give 0.518876, 100,001 give 0.518880, 200,001 (the default) 0.518882 and 400,001 give 0.518883.
    assert abs(volume(100_001) - volume(50_001)) < 1e-4
    assert abs(volume(400_001) - volume(200_001)) < 1e-4
    assert abs(volume(200_001) - volume(100_001)) < 1e-5


def test_tnk_front_is_not_missing_feasible_territory():
    grid = np.linspace(0, math.pi, 1001)
    x1, x2 = (a.ravel() for a in np.meshgrid(grid, grid))
    c1, c2 = mo._tnk_limits(x1, x2)
    ok = (c1 >= 0) & (c2 <= 0.5)
    candidates = np.column_stack([x1[ok], x2[ok]])
    front = mo._tnk_front()
    problem = PROBLEMS["tnk"]
    both = np.vstack([front, candidates[mo._front_2d(candidates)]])
    ideal, nadir = tuple(front.min(axis=0)), tuple(front.max(axis=0))
    assert mo._front_hypervolume(both[:, 0], both[:, 1], ideal, nadir) == pytest.approx(problem.true_hypervolume, abs=1e-3)


def test_cantilever_front_is_feasible_and_its_constants_come_from_it():
    designs, values = mo._cantilever_front(100)
    problem = PROBLEMS["cantilever-outputs"]
    for material, width, height in designs:
        assert problem.function(material=material, width=width, height=height)["stress_ratio"] <= 1 + 1e-9
    assert {m for m, _, _ in designs} == set(mo.MATERIALS)
    assert len(non_dominated(values)) == len(values)
    ideal, nadir, volume, size = mo._cantilever_reference_front(100)
    assert size == len(values)
    assert ideal == dict(zip(["mass", "cost", "deflection"], values.min(axis=0)))
    assert nadir == dict(zip(["mass", "cost", "deflection"], values.max(axis=0)))


def test_cantilever_constants_match_a_coarser_grid_of_the_front():
    ideal, nadir, volume, _ = mo._cantilever_reference_front(200)
    problem = PROBLEMS["cantilever-outputs"]
    assert mo.CANTILEVER_GRID == 2000
    assert volume == pytest.approx(problem.true_hypervolume, abs=1e-3)  # 1.31481 against 1.31524
    for name in problem.objectives:
        assert ideal[name] == pytest.approx(problem.ideal[name], rel=0.02)
        assert nadir[name] == pytest.approx(problem.nadir[name], rel=0.03)
        assert ideal[name] >= problem.ideal[name] - 1e-12 and nadir[name] <= problem.nadir[name] + 1e-12  # a finer grid reaches further


def test_noisy_versions_have_noise_on_exactly_the_objectives_and_outputs():
    noisy = noisy_multi_output_problems()
    assert [p.name for p in noisy] == ["noisy-osy", "noisy-tnk", "noisy-cantilever-outputs"]
    for problem, base in zip(noisy, PROBLEMS.values()):
        assert problem.noisy and not base.noisy
        assert set(problem.noise) == set(problem.objectives) and set(problem.output_noise) == set(problem.outputs)
        assert all(v > 0 for v in [*problem.noise.values(), *problem.output_noise.values()])
        assert "noisy" in problem.tags and base.tags < problem.tags
        assert problem.true_hypervolume == base.true_hypervolume and problem.search_space == base.search_space


@pytest.mark.parametrize("name", ["osy", "tnk", "cantilever-outputs"])
def test_nsga2_finds_a_feasible_front_no_better_than_the_true_one(name):
    problem = PROBLEMS[name]
    optimizer = NSGA2(problem.search_space, NSGA2Config(objectives=problem.objectives, max_evaluations=600, seed=0))
    result = optimizer.optimize(problem.function)
    assert result.feasible and result.front.points
    assert all(feasible(problem, p.params)[1] for p in result.front.points)
    assert 0 < problem.hypervolume([p.values for p in result.front.points]) <= problem.true_hypervolume + 1e-3
