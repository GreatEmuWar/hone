"""The benchmark problems with constraints on outputs: their optima are where they say, and feasible."""
import math

import numpy as np
import pytest

from hone.benchmarks import noisy_output_problems, output_problems
from hone.benchmarks import outputs as o

ALL = output_problems()


def evaluate(problem, params):
    """The value at params, its outputs, and whether it satisfies the problem's constraints on outputs."""
    value = problem.function(**params)
    outputs = {name: value[name] for name in problem.search_space.outputs}
    return value["objective"], outputs, problem.search_space.output_violation(params, outputs) == 0


def test_the_suite_has_the_engineering_designs_and_the_new_problems():
    names = [problem.name for problem in ALL]
    assert len(names) == len(set(names)) == 14
    assert names[:3] == ["pressure-vessel-outputs", "tension-spring-outputs", "welded-beam-outputs"]
    assert {"ball-2d-1in10", "ball-5d-1in1000", "corner-5d", "pocket-2d-1in100", "g06", "g07", "g10"} <= set(names)


@pytest.mark.parametrize("problem", ALL, ids=lambda problem: problem.name)
def test_every_problem_has_its_limits_on_outputs(problem):
    assert problem.search_space.output_constraints
    assert "outputs" in problem.tags and "constrained" in problem.tags
    assert problem.search_space.outputs == sorted(problem.search_space.outputs)


@pytest.mark.parametrize(
    "problem, best",
    [(o.g06(), o.G06_BEST), (o.g07(), o.G07_BEST), (o.g10(), o.G10_BEST)],
    ids=["g06", "g07", "g10"],
)
def test_cec_best_known_designs_are_feasible_at_their_published_values(problem, best):
    params = dict(zip(problem.search_space.names, best))
    assert problem.search_space.feasible(params)
    value, outputs, _ = evaluate(problem, params)
    # The published designs are rounded, so they break their active limits by up to 1e-6; the
    # CEC 2006 rules count a design as feasible within 1e-4.
    assert problem.search_space.output_violation(params, outputs) <= 1e-4
    assert value == pytest.approx(problem.optimum, rel=1e-6)


@pytest.mark.parametrize("dimensions, one_in", [(2, 10), (2, 100), (5, 100), (5, 1000)])
def test_a_ball_has_the_volume_it_says_and_its_optimum_on_its_surface(dimensions, one_in):
    problem = o.ball(dimensions, one_in)
    (limit,) = problem.search_space.output_constraints
    radius = limit.right.value  # Output("distance") <= radius
    volume = math.pi ** (dimensions / 2) / math.gamma(dimensions / 2 + 1) * radius**dimensions
    assert volume / 10**dimensions == pytest.approx(1 / one_in)
    assert problem.optimum == 1.0
    # The centre of the sphere objective is one unit outside the ball, so the optimum is a unit away, on the surface.
    rng = np.random.default_rng(0)
    best, feasible_count = math.inf, 0
    for _ in range(20_000):
        params = problem.search_space.sample(rng)
        value, outputs, feasible = evaluate(problem, params)
        feasible_count += feasible
        if feasible:
            best = min(best, value)
    assert best > problem.optimum
    assert feasible_count / 20_000 == pytest.approx(1 / one_in, rel=0.3)


@pytest.mark.parametrize("dimensions", [2, 5])
def test_a_corner_has_both_limits_active_at_its_optimum(dimensions):
    problem = o.corner(dimensions)
    # The optimum reports exactly 1 for both limits: the point where their boundaries cross.
    rng = np.random.default_rng(0)
    best, at = math.inf, None
    for _ in range(50_000):
        params = problem.search_space.sample(rng)
        value, outputs, feasible = evaluate(problem, params)
        if feasible and value < best:
            best, at = value, outputs
    assert best > problem.optimum
    assert at is not None and at["first"] <= 1 and at["second"] <= 1
    # Both limits are nearly active at the best sampled design: it is near where the boundaries cross.
    assert at["first"] > 0.5 and at["second"] > 0.5


@pytest.mark.parametrize("dimensions", [2, 5])
def test_a_pocket_is_infeasible_at_the_objectives_minimum_and_feasible_in_the_pocket(dimensions):
    problem = o.pocket(dimensions)
    rng = np.random.default_rng(0)
    feasible_points, best, worst_violation = [], math.inf, 0.0
    for _ in range(50_000):
        params = problem.search_space.sample(rng)
        value, outputs, feasible = evaluate(problem, params)
        if feasible:
            feasible_points.append(value)
            best = min(best, value)
        else:
            worst_violation = max(worst_violation, outputs["quantity"])
    assert 0.002 < len(feasible_points) / 50_000 < 0.05
    assert best > problem.optimum
    # The basin bottoms out at 0.5 above the limit, so the violation never reaches zero outside the pocket.
    assert all(v > problem.optimum for v in feasible_points)
    assert worst_violation > 0.5


@pytest.mark.parametrize("problem", ALL[3:], ids=lambda problem: problem.name)
def test_synthetic_optima_are_not_beaten_by_random_search(problem):
    rng = np.random.default_rng(1)
    for _ in range(5_000):
        params = problem.search_space.sample(rng)
        value, _, feasible = evaluate(problem, params)
        assert not feasible or value >= problem.optimum - 1e-9


def test_noisy_versions_add_noise_to_the_value_and_every_output():
    noisy = noisy_output_problems()
    assert [problem.name for problem in noisy] == [f"noisy-{problem.name}" for problem in ALL[:3]]
    for problem, plain in zip(noisy, ALL[:3]):
        assert problem.noise > 0 and set(problem.output_noise) == set(plain.search_space.outputs)
        assert all(sd > 0 for sd in problem.output_noise.values())
        assert problem.optimum == plain.optimum and "noisy" in problem.tags
